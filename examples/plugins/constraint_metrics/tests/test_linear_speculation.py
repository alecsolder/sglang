import inspect
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.constrained.llguidance_backend import GuidanceGrammar
from sglang.srt.sampling.sampling_observer import LinearSpeculationObservation
from sglang_constraint_metrics import layout
from sglang_constraint_metrics.capture import CaptureScratch
from sglang_constraint_metrics.config import ConstraintMetricsConfig
from sglang_constraint_metrics.linear_speculation import (
    LinearSpeculationCapture,
    _build_linear_record_metadata,
    begin_linear_speculation_capture,
    warmup_linear_record_metadata,
)
from sglang_constraint_metrics.records import ConstraintMetricsTensorBatch


def test_linear_capture_finalize_matches_sglang_contract():
    expected = inspect.signature(LinearSpeculationObservation.finalize).parameters
    actual = inspect.signature(LinearSpeculationCapture.finalize).parameters

    assert tuple(actual) == tuple(expected)


class _Grammar(GuidanceGrammar):
    def __init__(
        self,
        mode: str,
        *,
        top_k: int = 8,
        stream_all_token_metrics: bool = False,
    ):
        self._finished = False
        self.constraint_metrics_config = ConstraintMetricsConfig(
            mode,
            top_k,
            stream_all_token_metrics=stream_all_token_metrics,
        )

    def is_terminated(self):
        return False


def _packed_allowed_mask(rows: int, vocab_size: int) -> torch.Tensor:
    words = (vocab_size + 31) // 32
    mask = torch.zeros(rows, words, dtype=torch.int32)
    for token_id in range(vocab_size):
        if token_id != 1:
            mask[:, token_id // 32] |= 1 << (token_id % 32)
    return mask


def _recording_compute(calls):
    def compute(pre, grammar_mask, emitted_ids, **kwargs):
        calls.append(
            {
                "pre": pre.clone(),
                "mask": grammar_mask,
                "emitted_ids": emitted_ids.clone(),
                "spec_sources": kwargs["spec_sources"].clone(),
                "row_enabled": kwargs["row_enabled"].clone(),
                "top_ks": list(kwargs["top_ks"]),
            }
        )
        rows = pre.shape[0]
        max_k = max(kwargs["top_ks"])
        return ConstraintMetricsTensorBatch(
            floats=torch.zeros(
                rows,
                layout.KERNEL_ROW_FLOAT_COUNT + layout.TOP_K_FLOAT_GROUP_COUNT * max_k,
                device=pre.device,
            ),
            ints=torch.zeros(
                rows,
                layout.KERNEL_ROW_INT_COUNT + layout.TOP_K_INT_GROUP_COUNT * max_k,
                dtype=torch.int64,
                device=pre.device,
            ),
            top_ks=list(kwargs["top_ks"]),
            grammar_active=list(kwargs["grammar_active"]),
            output_indices=list(kwargs["output_indices"]),
            vocab_size=pre.shape[1],
        )

    return compute


def _begin_capture(logits, batch, vocab_mask):
    return begin_linear_speculation_capture(
        logits,
        batch,
        vocab_mask,
        eos_token_ids=[2],
        scratch=CaptureScratch.allocate(
            logits.shape[0] * logits.shape[1],
            logits.shape[2],
            logits.device,
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_linear_metadata_matches_committed_prefix():
    device = torch.device("cuda")
    committed_token_ids = torch.tensor(
        [[12, 90, 91], [20, 22, 92], [30, 31, 32]], device=device
    )
    commit_lens = torch.tensor([1, 2, 3], dtype=torch.int32, device=device)
    emitted_ids = torch.empty(9, dtype=torch.long, device=device)
    spec_sources = torch.empty_like(emitted_ids)
    row_enabled = torch.empty(9, dtype=torch.bool, device=device)

    _build_linear_record_metadata(
        committed_token_ids,
        commit_lens,
        emitted_ids,
        spec_sources,
        row_enabled,
        width=3,
    )
    torch.cuda.synchronize()

    assert emitted_ids.view(3, 3).tolist() == [
        [12, 0, 0],
        [20, 22, 0],
        [30, 31, 32],
    ]
    assert spec_sources.view(3, 3).tolist() == [
        [layout.SPEC_SOURCE_BONUS, 0, 0],
        [layout.SPEC_SOURCE_ACCEPTED_DRAFT, layout.SPEC_SOURCE_BONUS, 0],
        [
            layout.SPEC_SOURCE_ACCEPTED_DRAFT,
            layout.SPEC_SOURCE_ACCEPTED_DRAFT,
            layout.SPEC_SOURCE_BONUS,
        ],
    ]
    assert row_enabled.view(3, 3).tolist() == [
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_linear_metadata_honors_sliced_row_stride():
    device = torch.device("cuda")
    width = 5
    storage = torch.tensor(
        [
            [100, 10, 11, 12, 13, 14],
            [200, 20, 21, 22, 23, 24],
            [300, 30, 31, 32, 33, 34],
        ],
        device=device,
    )
    committed_token_ids = storage[:, 1:]
    commit_lens = torch.full((3,), width, dtype=torch.int32, device=device)
    emitted_ids = torch.empty(3 * width, dtype=torch.long, device=device)
    spec_sources = torch.empty_like(emitted_ids)
    row_enabled = torch.empty(3 * width, dtype=torch.bool, device=device)

    _build_linear_record_metadata(
        committed_token_ids,
        commit_lens,
        emitted_ids,
        spec_sources,
        row_enabled,
        width=width,
    )
    torch.cuda.synchronize()

    assert committed_token_ids.stride() == (width + 1, 1)
    assert emitted_ids.view(3, width).tolist() == [
        [10, 11, 12, 13, 14],
        [20, 21, 22, 23, 24],
        [30, 31, 32, 33, 34],
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(("rows", "width"), [(2, 1), (129, 3)])
def test_linear_metadata_handles_edge_grid_shapes(rows, width):
    device = torch.device("cuda")
    committed_token_ids = torch.arange(
        rows * width, dtype=torch.int64, device=device
    ).view(rows, width)
    commit_lens = torch.full((rows,), width, dtype=torch.int32, device=device)
    emitted_ids = torch.empty(rows * width, dtype=torch.int64, device=device)
    spec_sources = torch.empty_like(emitted_ids)
    row_enabled = torch.empty(rows * width, dtype=torch.bool, device=device)

    _build_linear_record_metadata(
        committed_token_ids,
        commit_lens,
        emitted_ids,
        spec_sources,
        row_enabled,
        width=width,
    )
    torch.cuda.synchronize()

    assert torch.equal(emitted_ids, committed_token_ids.flatten())
    assert row_enabled.all()
    assert torch.equal(
        spec_sources.view(rows, width)[:, -1],
        torch.full((rows,), layout.SPEC_SOURCE_BONUS, device=device),
    )
    if width > 1:
        assert torch.all(
            spec_sources.view(rows, width)[:, :-1] == layout.SPEC_SOURCE_ACCEPTED_DRAFT
        )


@pytest.mark.parametrize("width", [1, 5])
def test_linear_metadata_warmup_uses_dflash_slice(monkeypatch, width):
    import sglang_constraint_metrics.linear_speculation as module

    captured = {}

    def capture(committed_token_ids, commit_lens, *args, **kwargs):
        captured["committed_token_ids"] = committed_token_ids
        captured["commit_lens"] = commit_lens

    monkeypatch.setattr(module, "_build_linear_record_metadata", capture)
    scratch = SimpleNamespace(
        emitted_ids=torch.empty(width, dtype=torch.int64),
        spec_sources=torch.empty(width, dtype=torch.int64),
        row_enabled=torch.empty(width, dtype=torch.bool),
    )

    warmup_linear_record_metadata(width=width, device="cpu", scratch=scratch)

    committed_token_ids = captured["committed_token_ids"]
    assert committed_token_ids.shape == (1, width)
    assert committed_token_ids.stride() == (width, 1)
    assert captured["commit_lens"].dtype == torch.int32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_linear_capture_aligns_committed_sources_for_mixed_modes(monkeypatch):
    import sglang_constraint_metrics.linear_speculation as module

    device = torch.device("cuda")
    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(
                grammar=_Grammar(
                    "forensic",
                    top_k=8,
                    stream_all_token_metrics=True,
                )
            ),
            SimpleNamespace(grammar=_Grammar("summary", top_k=0)),
        ],
        sampling_info=SimpleNamespace(
            temperatures=torch.tensor([[2.0], [0.5]], device=device),
        ),
    )
    original = torch.arange(24, dtype=torch.float32, device=device).reshape(2, 3, 4)
    capture = _begin_capture(
        original.clone(),
        batch,
        _packed_allowed_mask(6, vocab_size=4).to(device),
    )

    assert torch.equal(capture.inputs.pre, original[[1, 0]])
    assert torch.equal(
        capture.inputs.temperatures,
        torch.tensor([[0.5], [2.0]], device=device),
    )

    calls = []
    monkeypatch.setattr(
        module, "_compute_record_tensor_batch_chunked", _recording_compute(calls)
    )
    device_output = capture.finalize(
        accept_tokens=torch.tensor([[10, 12, 90], [20, 21, 22]], device=device),
        num_accept_tokens=torch.tensor([2, 3], device=device),
    )

    assert [call["emitted_ids"].tolist() for call in calls] == [
        [20, 21, 22],
        [10, 12, 0],
    ]
    assert [call["spec_sources"].tolist() for call in calls] == [
        [1, 1, 2],
        [1, 2, 0],
    ]
    assert [payload.stream_to_host for payload in device_output.record_payloads] == [
        False,
        True,
    ]
    assert all(payload.speculative for payload in device_output.record_payloads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_linear_full_batch_capture_avoids_index_gathers(monkeypatch):
    import sglang_constraint_metrics.linear_speculation as module

    device = torch.device("cuda")
    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(grammar=_Grammar("forensic")),
            SimpleNamespace(grammar=_Grammar("forensic")),
        ],
        sampling_info=SimpleNamespace(
            temperatures=torch.tensor([[2.0], [0.5]], device=device)
        ),
    )
    logits = torch.arange(24, dtype=torch.float32, device=device).reshape(2, 3, 4)
    grammar_mask = _packed_allowed_mask(6, vocab_size=4).to(device)
    capture = _begin_capture(logits, batch, grammar_mask)

    assert capture.inputs.diagnostic_indices_device is None
    assert capture.inputs.grammar_mask is grammar_mask
    assert capture.inputs.temperatures is batch.sampling_info.temperatures

    calls = []
    monkeypatch.setattr(
        module, "_compute_record_tensor_batch_chunked", _recording_compute(calls)
    )
    output = capture.finalize(
        accept_tokens=torch.tensor([[10, 12, 90], [20, 21, 22]], device=device),
        num_accept_tokens=torch.tensor([2, 3], device=device),
    )

    assert len(calls) == 1
    assert calls[0]["pre"].shape == (6, 4)
    assert calls[0]["mask"].data_ptr() == grammar_mask.data_ptr()
    assert calls[0]["emitted_ids"].tolist() == [10, 12, 0, 20, 21, 22]
    assert output.record_payloads[0].row_spans == [(0, 0, 3), (1, 3, 3)]
