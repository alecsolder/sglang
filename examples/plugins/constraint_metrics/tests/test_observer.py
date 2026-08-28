import gc
import pickle
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import sglang_constraint_metrics.records as records
from sglang.srt.constrained.llguidance_backend import GuidanceGrammar
from sglang.srt.managers.io_struct import unwrap_from_pickle, wrap_as_pickle
from sglang.srt.sampling.sampling_observer import CommittedTokens
from sglang_constraint_metrics import layout
from sglang_constraint_metrics.config import (
    CUSTOM_PARAMS_KEY,
    ConstraintMetricsConfig,
    config_from_sampling_custom_params,
)
from sglang_constraint_metrics.observer import (
    CONSTRAINT_METRICS_STREAM_ENABLED_KEY,
    CONSTRAINT_METRICS_STREAM_KEY,
    ConstraintMetricsDeviceOutput,
    ConstraintMetricsHostOutput,
    ConstraintMetricsObserver,
    ConstraintMetricsState,
    ConstraintMetricsStreamRecord,
    bind_constraint_metrics_state,
    prepare_streamed_constraint_metrics,
    release_streamed_constraint_metrics,
    snapshot_constraint_metrics,
)
from sglang_constraint_metrics.records import (
    CommittedRecordSlice,
    ConstraintMetricsRecordPayload,
    ConstraintMetricsTensorBatch,
    archive_record_payloads,
    compact_record_payloads,
    retain_record_payload_on_device,
)
from sglang_constraint_metrics.schema import (
    SEVERE_PRESSURE_THRESHOLD_NATS,
    ConstraintMetricsSummaryAccumulator,
    extend_kernel_row_segments,
)


def _float_width(top_k: int = 0) -> int:
    return layout.KERNEL_ROW_FLOAT_COUNT + layout.TOP_K_FLOAT_GROUP_COUNT * top_k


def _int_width(top_k: int = 0) -> int:
    return layout.KERNEL_ROW_INT_COUNT + layout.TOP_K_INT_GROUP_COUNT * top_k


def _token_records(response):
    tokens = response["tokens"]
    if isinstance(tokens, list):
        return tokens

    def table_row(table, index, value_key):
        width = table["width"]
        start = index * width
        count = table["count"][index]
        return [
            {"token_id": token_id, value_key: value}
            for token_id, value in zip(
                table["token_id"][start : start + count],
                table[value_key][start : start + count],
                strict=True,
            )
        ]

    result = []
    for index in range(tokens["count"]):
        record = {
            "output_index": tokens["output_index"][index],
            "token_id": tokens["token_id"][index],
            "grammar_active": tokens["grammar_active"][index],
            "effective_legal_token_count": tokens["effective_legal_token_count"][index],
            "constraint_pressure_nats": tokens["constraint_pressure_nats"][index],
            "argmax_masked": tokens["argmax_masked"][index],
            "unconstrained_top1": {
                key: values[index]
                for key, values in tokens["unconstrained_top1"].items()
            },
            "entropy_nats": tokens["entropy_nats"][index],
            "emitted": {
                key: values[index] for key, values in tokens["emitted"].items()
            },
            "top_masked": table_row(tokens["top_masked"], index, "prob"),
            "top_legal_after_conditioning": table_row(
                tokens["top_legal_after_conditioning"],
                index,
                "constrained_prob",
            ),
            "flags": [
                name for name, values in tokens["flags"].items() if values[index]
            ],
        }
        spec_source = tokens["spec_source"][index]
        if spec_source is not None:
            record["spec_source"] = spec_source
        result.append(record)
    return result


def _assert_token_columns(tokens):
    assert set(tokens) == {
        "count",
        "output_index",
        "token_id",
        "grammar_active",
        "effective_legal_token_count",
        "constraint_pressure_nats",
        "argmax_masked",
        "unconstrained_top1",
        "entropy_nats",
        "emitted",
        "top_masked",
        "top_legal_after_conditioning",
        "flags",
        "spec_source",
    }
    count = tokens["count"]
    for key, values in tokens.items():
        if key == "count":
            continue
        if isinstance(values, dict):
            if key in ("top_masked", "top_legal_after_conditioning"):
                width = values["width"]
                value_key = "prob" if key == "top_masked" else "constrained_prob"
                assert isinstance(width, int)
                assert width >= 0
                assert len(values["count"]) == count
                assert all(0 <= row_count <= width for row_count in values["count"])
                assert len(values["token_id"]) == count * width
                assert len(values[value_key]) == count * width
            else:
                assert all(len(column) == count for column in values.values())
        else:
            assert len(values) == count


class _Grammar(GuidanceGrammar):
    def __init__(
        self,
        mode: str | None = None,
        *,
        top_k: int = 8,
        stream_all_token_metrics: bool = False,
    ):
        self._finished = False
        self.constraint_metrics_config = (
            ConstraintMetricsConfig(
                mode,
                top_k,
                stream_all_token_metrics=stream_all_token_metrics,
            )
            if mode is not None
            else None
        )

    def is_terminated(self):
        return False


def _dense_blocked_mask(rows: int, vocab_size: int) -> SimpleNamespace:
    mask = torch.zeros(rows, vocab_size, dtype=torch.bool)
    if vocab_size > 1:
        mask[:, 1] = True
    return SimpleNamespace(vocab_mask=mask)


def _custom(mode: str, *, top_k: int = 8):
    payload = {"mode": mode}
    if mode == "forensic":
        payload["top_k"] = top_k
    return {CUSTOM_PARAMS_KEY: payload}


def _sampling_params(**overrides):
    values = {
        "custom_params": {},
        "ebnf": None,
        "regex": None,
        "json_schema": None,
        "structural_tag": None,
        "temperature": 1.0,
        "top_k": -1,
        "top_p": 1.0,
        "min_p": 0.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "logit_bias": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _record_slice(
    payload: ConstraintMetricsRecordPayload,
    batch_index: int,
    *,
    count: int | None = None,
    output_index: int = 0,
) -> CommittedRecordSlice:
    start, available = payload.span_for(batch_index)
    return CommittedRecordSlice(
        payload=payload,
        start=start,
        count=available if count is None else count,
        output_index_start=output_index,
    )


def _metric_payload(
    row_specs: list[dict],
    *,
    top_k: int = 2,
    speculative: bool = False,
) -> ConstraintMetricsRecordPayload:
    row_count = len(row_specs)
    floats = torch.zeros(row_count, _float_width(top_k))
    ints = torch.zeros(row_count, _int_width(top_k), dtype=torch.int64)
    grammar_active = []
    for row_index, spec in enumerate(row_specs):
        pressure = float(spec.get("pressure", 0.0))
        empty_support = bool(spec.get("empty_support", False))
        token_id = int(spec.get("token_id", 10 + row_index))
        floats[row_index, layout.FLOAT_PRESSURE] = pressure
        floats[row_index, layout.FLOAT_ENTROPY] = 0.5
        floats[row_index, layout.FLOAT_UNCONSTRAINED_TOP1_PROB] = 0.25
        floats[row_index, layout.FLOAT_TOP1_LOGIT_GAP] = 0.75
        floats[row_index, layout.FLOAT_EMITTED_MODEL_LOG_PROB] = -1.25

        ints[row_index, layout.KERNEL_INT_UNCONSTRAINED_TOP1_TOKEN_ID] = 100 + row_index
        ints[row_index, layout.KERNEL_INT_EMITTED_RANK] = row_index
        ints[row_index, layout.KERNEL_INT_EFFECTIVE_LEGAL_TOKEN_COUNT] = (
            0 if empty_support else int(spec.get("legal_count", 3))
        )
        ints[row_index, layout.KERNEL_INT_BLOCKED_TOKEN_COUNT] = int(
            spec.get("blocked_count", 5)
        )
        ints[row_index, layout.KERNEL_INT_TOKEN_ID] = token_id
        ints[row_index, layout.KERNEL_INT_SPEC_SOURCE] = int(
            spec.get("spec_source", layout.SPEC_SOURCE_NORMAL)
        )
        ints[row_index, layout.KERNEL_INT_ARGMAX_MASKED] = int(
            spec.get("argmax_masked", False)
        )
        ints[row_index, layout.KERNEL_INT_EMPTY_SUPPORT] = int(empty_support)
        ints[row_index, layout.KERNEL_INT_EOS_ONLY_SUPPORT] = int(
            spec.get("eos_only_support", False)
        )
        ints[row_index, layout.KERNEL_INT_HAS_NON_FINITE_LOGITS] = int(
            spec.get("has_non_finite_logits", False)
        )
        ints[row_index, layout.KERNEL_INT_EMITTED_ALLOWED] = int(
            spec.get("emitted_allowed", True)
        )
        grammar_active.append(bool(spec.get("grammar_active", True)))

        for candidate_index in range(top_k):
            ints[
                row_index,
                layout.KERNEL_ROW_INT_COUNT + candidate_index,
            ] = (
                300 + row_index * 10 + candidate_index
            )
            ints[
                row_index,
                layout.KERNEL_ROW_INT_COUNT + top_k + candidate_index,
            ] = (
                400 + row_index * 10 + candidate_index
            )
            floats[
                row_index,
                layout.KERNEL_ROW_FLOAT_COUNT + candidate_index,
            ] = 0.1 * (candidate_index + 1)
            floats[
                row_index,
                layout.KERNEL_ROW_FLOAT_COUNT + top_k + candidate_index,
            ] = 0.2 * (candidate_index + 1)

    return ConstraintMetricsRecordPayload(
        tensor_batch=ConstraintMetricsTensorBatch(
            floats=floats,
            ints=ints,
            top_ks=[top_k] * row_count,
            grammar_active=grammar_active,
            output_indices=[0] * row_count,
            vocab_size=512,
        ),
        row_spans=[(0, 0, row_count)],
        speculative=speculative,
    )


def _archive_payload(
    token_ids: list[int],
    *,
    physical_top_k: int = 0,
    requested_top_ks: list[int] | None = None,
    output_indices: list[int] | None = None,
    row_spans: list[tuple[int, int, int]],
    speculative: bool = False,
    spec_sources: list[int] | None = None,
) -> ConstraintMetricsRecordPayload:
    row_count = len(token_ids)
    floats = torch.arange(
        row_count * _float_width(physical_top_k), dtype=torch.float32
    ).view(row_count, _float_width(physical_top_k))
    ints = torch.zeros(row_count, _int_width(physical_top_k), dtype=torch.int64)
    ints[:, layout.KERNEL_INT_TOKEN_ID] = torch.tensor(token_ids)
    if spec_sources is not None:
        ints[:, layout.KERNEL_INT_SPEC_SOURCE] = torch.tensor(spec_sources)
    return ConstraintMetricsRecordPayload(
        tensor_batch=ConstraintMetricsTensorBatch(
            floats=floats,
            ints=ints,
            top_ks=(
                requested_top_ks
                if requested_top_ks is not None
                else [physical_top_k] * row_count
            ),
            grammar_active=[True] * row_count,
            output_indices=(
                output_indices if output_indices is not None else [0] * row_count
            ),
            vocab_size=32,
        ),
        row_spans=row_spans,
        speculative=speculative,
    )


def _request_with_state(
    state: ConstraintMetricsState,
    output_ids: list[int],
    *,
    finished: bool = False,
) -> SimpleNamespace:
    req = SimpleNamespace(
        origin_input_ids=[1],
        output_ids_through_stop=list(output_ids),
        grammar=_Grammar("forensic", top_k=0),
        is_retracted=False,
        sampling_params=_sampling_params(
            custom_params={},
            ebnf="start: /x/",
            regex=None,
            json_schema=None,
            structural_tag=None,
            temperature=1.0,
            top_k=-1,
            top_p=1.0,
            min_p=0.0,
        ),
        _constraint_metrics_state=state,
        _finished=finished,
        stream=False,
        customized_info=None,
    )
    req.finished = lambda: req._finished
    return req


def _enable_metrics_stream(req: SimpleNamespace) -> None:
    req._constraint_metrics_state.streaming = True
    req.stream = True
    req.customized_info = {
        CONSTRAINT_METRICS_STREAM_KEY: [],
        CONSTRAINT_METRICS_STREAM_ENABLED_KEY: [],
    }


def _materialize_stream_records(req: SimpleNamespace) -> list:
    return [
        (
            record.materialize()
            if isinstance(record, ConstraintMetricsStreamRecord)
            else record
        )
        for record in req.customized_info[CONSTRAINT_METRICS_STREAM_KEY]
    ]


def _observer():
    return ConstraintMetricsObserver(
        model_config=SimpleNamespace(hf_eos_token_id={2}),
    )


def test_linear_speculation_activation_reads_requests_without_mutating_sampling_info():
    sampling_info = SimpleNamespace(grammars=None)
    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(grammar=_Grammar()),
            SimpleNamespace(grammar=_Grammar("summary")),
        ],
        sampling_info=sampling_info,
    )

    assert _observer().is_linear_speculation_active(batch)
    assert sampling_info.grammars is None


def test_linear_speculation_activation_is_false_without_configured_grammar():
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(grammar=None), SimpleNamespace(grammar=_Grammar())],
        sampling_info=SimpleNamespace(grammars=None),
    )

    assert not _observer().is_linear_speculation_active(batch)


def test_mixed_forensic_k_values_use_isolated_payload_widths(monkeypatch):
    import sglang_constraint_metrics.observer as observer_module

    info = SimpleNamespace(
        custom_params=[
            _custom("summary"),
            _custom("forensic", top_k=8),
            _custom("forensic", top_k=32),
        ],
        grammars=[
            _Grammar("summary", top_k=32),
            _Grammar(
                "forensic",
                top_k=8,
                stream_all_token_metrics=True,
            ),
            _Grammar("forensic", top_k=32),
        ],
        logit_bias=None,
        temperatures=torch.ones(3, 1),
        grammar_mask=_dense_blocked_mask(3, 65),
    )
    state = _observer().before_grammar(
        torch.arange(3 * 65, dtype=torch.float32).reshape(3, 65),
        info,
    )
    calls = []

    def fake_compute(pre, grammar_mask, emitted, **kwargs):
        top_ks = list(kwargs["top_ks"])
        max_k = max(top_ks)
        calls.append((top_ks, pre.shape, grammar_mask.shape, emitted.tolist()))
        return ConstraintMetricsTensorBatch(
            floats=torch.zeros(pre.shape[0], _float_width(max_k)),
            ints=torch.zeros(pre.shape[0], _int_width(max_k), dtype=torch.int64),
            top_ks=top_ks,
            grammar_active=list(kwargs["grammar_active"]),
            output_indices=list(kwargs["output_indices"]),
            vocab_size=pre.shape[1],
        )

    monkeypatch.setattr(observer_module, "_compute_record_tensor_batch", fake_compute)
    output = _observer().after_sample(state, torch.tensor([10, 11, 12]))

    assert [call[0] for call in calls] == [[0], [8], [32]]
    assert [
        payload.tensor_batch.floats.shape[1] for payload in output.record_payloads
    ] == [
        _float_width(),
        _float_width(8),
        _float_width(32),
    ]
    assert [payload.row_spans for payload in output.record_payloads] == [
        [(0, 0, 1)],
        [(1, 0, 1)],
        [(2, 0, 1)],
    ]
    assert [payload.stream_to_host for payload in output.record_payloads] == [
        False,
        True,
        False,
    ]


def test_observer_captures_effective_logits_and_real_grammar_mask():
    original = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dtype=torch.float32
    )
    logits = original.clone()
    info = SimpleNamespace(
        custom_params=[_custom("summary"), _custom("summary")],
        grammars=[_Grammar("summary"), _Grammar("summary")],
        logit_bias=torch.tensor([[0.0, 0.5, 0.0, 0.0], [0.0, 0.0, -0.5, 0.0]]),
        temperatures=torch.tensor([[2.0], [0.5]]),
        grammar_mask=_dense_blocked_mask(2, 4),
    )
    observer = _observer()

    state = observer.before_grammar(logits, info)
    expected_pre = original + info.logit_bias
    inputs = state.inputs

    assert inputs.diagnostic_indices_device is None
    assert torch.equal(inputs.pre, expected_pre)
    assert torch.equal(inputs.temperatures, info.temperatures)
    assert inputs.grammar_mask is info.grammar_mask.vocab_mask
    assert inputs.grammar_mask.tolist() == [
        [False, True, False, False],
        [False, True, False, False],
    ]
    assert torch.equal(logits, original)


def test_forensic_payload_stays_on_device_until_terminal_archive():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for device-retained records")
    baseline = records._DEVICE_RETENTION_BYTES
    state = ConstraintMetricsState(
        config=ConstraintMetricsConfig("forensic", 0),
    )
    ints = torch.zeros(2, _int_width(), dtype=torch.int64, device="cuda")
    ints[:, layout.KERNEL_INT_TOKEN_ID] = torch.tensor([12, 13], device="cuda")
    ints[:, layout.KERNEL_INT_SPEC_SOURCE] = layout.SPEC_SOURCE_ACCEPTED_DRAFT
    payload = retain_record_payload_on_device(
        ConstraintMetricsRecordPayload(
            tensor_batch=ConstraintMetricsTensorBatch(
                floats=torch.zeros(2, _float_width(), device="cuda"),
                ints=ints,
                top_ks=[0, 0],
                grammar_active=[True, True],
                output_indices=[0, 0],
                vocab_size=32,
            ),
            row_spans=[(0, 0, 2)],
            speculative=True,
        ),
        enabled=True,
    )
    assert payload.device_retained
    copy_tensor = Mock(side_effect=lambda tensor: tensor.to("cpu"))
    output = ConstraintMetricsDeviceOutput([payload]).copy_to_host(copy_tensor)
    copy_tensor.assert_not_called()
    req = SimpleNamespace(
        origin_input_ids=[1],
        output_ids_through_stop=[12],
        grammar=_Grammar("forensic", top_k=0),
        is_retracted=False,
        sampling_params=_sampling_params(
            custom_params={},
            ebnf="start: /x/",
            regex=None,
            json_schema=None,
            structural_tag=None,
            temperature=1.0,
            top_k=-1,
            top_p=1.0,
            min_p=0.0,
        ),
        _constraint_metrics_state=state,
    )
    req.finished = lambda: True

    with patch.object(
        ConstraintMetricsRecordPayload,
        "move_to_pageable_storage",
        autospec=True,
        side_effect=ConstraintMetricsRecordPayload.move_to_pageable_storage,
    ) as move_to_pageable:
        output.consume(
            SimpleNamespace(reqs=[req]),
            [CommittedTokens(0, (12,))],
        )

    assert move_to_pageable.call_count == 0
    assert output.record_payloads == []
    assert state.record_slices[0].payload is payload
    assert state.summary_accumulator.output_token_count == 0
    assert payload.tensor_batch.floats.is_cuda
    snapshot = snapshot_constraint_metrics([req])[0]
    assert state.record_slices == []
    assert not payload.device_retained
    assert not payload.tensor_batch.floats.is_cuda
    assert payload.tensor_batch.floats.shape[0] == 0
    assert records._DEVICE_RETENTION_BYTES == baseline
    assert not snapshot.record_slices[0].payload.tensor_batch.floats.is_cuda
    assert not snapshot.record_slices[0].payload.tensor_batch.ints.is_cuda
    owned = unwrap_from_pickle(wrap_as_pickle([snapshot]))[0]
    response = owned.materialize_constraint_metrics()
    _assert_token_columns(response["tokens"])
    assert _token_records(response)[0]["token_id"] == 12
    assert response["summary"]["output_token_count"] == 1


def test_retained_forensic_stream_copies_only_scalar_columns(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for device-retained records")
    baseline = records._DEVICE_RETENTION_BYTES
    floats = torch.zeros(2, _float_width(8), device="cuda")
    ints = torch.zeros(2, _int_width(8), dtype=torch.int64, device="cuda")
    monkeypatch.setattr(
        records,
        "_DEVICE_RETENTION_MAX_BYTES",
        baseline + floats.nbytes + ints.nbytes,
    )
    payload = retain_record_payload_on_device(
        ConstraintMetricsRecordPayload(
            tensor_batch=ConstraintMetricsTensorBatch(
                floats=floats,
                ints=ints,
                top_ks=[8, 8],
                grammar_active=[True, True],
                output_indices=[0, 0],
                vocab_size=32,
            ),
            row_spans=[(0, 0, 2)],
            stream_to_host=True,
        ),
        enabled=True,
    )
    assert payload.device_retained
    copy_tensor = Mock(side_effect=lambda tensor: tensor.to("cpu"))

    output = ConstraintMetricsDeviceOutput([payload]).copy_to_host(copy_tensor)

    assert output.record_payloads == [payload]
    assert payload.tensor_batch.floats.is_cuda
    assert len(output.stream_record_payloads) == 1
    stream_batch = output.stream_record_payloads[0].tensor_batch
    assert stream_batch.floats.shape == (2, layout.KERNEL_ROW_FLOAT_COUNT)
    assert stream_batch.ints.shape == (2, layout.KERNEL_ROW_INT_COUNT)
    assert stream_batch.top_ks == [0, 0]
    assert copy_tensor.call_count == 2

    payload.release_device_storage_if_unreferenced()
    assert records._DEVICE_RETENTION_BYTES == baseline


def test_shared_retained_payload_survives_staggered_completion(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for device-retained records")
    baseline = records._DEVICE_RETENTION_BYTES
    floats = torch.zeros(2, _float_width(), device="cuda")
    ints = torch.zeros(2, _int_width(), dtype=torch.int64, device="cuda")
    ints[:, layout.KERNEL_INT_TOKEN_ID] = torch.tensor([12, 13], device="cuda")
    retained_bytes = floats.nbytes + ints.nbytes
    monkeypatch.setattr(
        records, "_DEVICE_RETENTION_MAX_BYTES", baseline + retained_bytes
    )
    payload = retain_record_payload_on_device(
        ConstraintMetricsRecordPayload(
            tensor_batch=ConstraintMetricsTensorBatch(
                floats=floats,
                ints=ints,
                top_ks=[0, 0],
                grammar_active=[True, True],
                output_indices=[0, 0],
                vocab_size=32,
            ),
            row_spans=[(0, 0, 1), (1, 1, 1)],
        ),
        enabled=True,
    )
    assert records._DEVICE_RETENTION_BYTES == baseline + retained_bytes
    output = ConstraintMetricsDeviceOutput([payload]).copy_to_host(
        lambda tensor: tensor.to("cpu")
    )
    states = [
        ConstraintMetricsState(
            config=ConstraintMetricsConfig("forensic", 0),
        )
        for _ in range(2)
    ]
    reqs = [
        _request_with_state(states[0], [12], finished=True),
        _request_with_state(states[1], [13]),
    ]
    output.consume(
        SimpleNamespace(reqs=reqs),
        [CommittedTokens(0, (12,)), CommittedTokens(0, (13,))],
    )

    first = snapshot_constraint_metrics(reqs)
    assert first[0] is not None and first[1] is None
    assert (
        _token_records(first[0].materialize_constraint_metrics())[0]["token_id"] == 12
    )
    assert states[0].record_slices == []
    assert states[1].record_slices[0].payload is payload
    assert states[1].record_slices[0].payload.tensor_batch.floats.is_cuda
    assert records._DEVICE_RETENTION_BYTES == baseline + retained_bytes

    reqs[1]._finished = True
    second = snapshot_constraint_metrics([reqs[1]])[0]
    assert second is not None
    assert _token_records(second.materialize_constraint_metrics())[0]["token_id"] == 13
    assert states[1].record_slices == []
    assert records._DEVICE_RETENTION_BYTES == baseline

    del output, payload
    gc.collect()
    assert records._DEVICE_RETENTION_BYTES == baseline


def test_retained_speculative_payload_archives_only_committed_prefixes(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for device-retained records")
    baseline = records._DEVICE_RETENTION_BYTES
    floats = torch.zeros(9, _float_width(), device="cuda")
    ints = torch.zeros(9, _int_width(), dtype=torch.int64, device="cuda")
    ints[:, layout.KERNEL_INT_TOKEN_ID] = torch.tensor(
        [10, 90, 91, 20, 21, 92, 30, 31, 32], device="cuda"
    )
    ints[:, layout.KERNEL_INT_SPEC_SOURCE] = torch.tensor(
        [
            layout.SPEC_SOURCE_BONUS,
            layout.SPEC_SOURCE_NORMAL,
            layout.SPEC_SOURCE_NORMAL,
            layout.SPEC_SOURCE_ACCEPTED_DRAFT,
            layout.SPEC_SOURCE_BONUS,
            layout.SPEC_SOURCE_NORMAL,
            layout.SPEC_SOURCE_ACCEPTED_DRAFT,
            layout.SPEC_SOURCE_ACCEPTED_DRAFT,
            layout.SPEC_SOURCE_BONUS,
        ],
        device="cuda",
    )
    retained_bytes = floats.nbytes + ints.nbytes
    monkeypatch.setattr(
        records, "_DEVICE_RETENTION_MAX_BYTES", baseline + retained_bytes
    )
    payload = retain_record_payload_on_device(
        ConstraintMetricsRecordPayload(
            tensor_batch=ConstraintMetricsTensorBatch(
                floats=floats,
                ints=ints,
                top_ks=[0] * 9,
                grammar_active=[True] * 9,
                output_indices=[0] * 9,
                vocab_size=128,
            ),
            row_spans=[(0, 0, 3), (1, 3, 3), (2, 6, 3)],
            speculative=True,
        ),
        enabled=True,
    )
    output = ConstraintMetricsDeviceOutput([payload]).copy_to_host(
        lambda tensor: tensor.to("cpu")
    )
    expected_tokens = [[10], [20, 21], [30, 31, 32]]
    states = [
        ConstraintMetricsState(
            config=ConstraintMetricsConfig("forensic", 0),
        )
        for _ in expected_tokens
    ]
    reqs = [
        _request_with_state(state, token_ids, finished=True)
        for state, token_ids in zip(states, expected_tokens, strict=True)
    ]
    output.consume(
        SimpleNamespace(reqs=reqs),
        [CommittedTokens(0, tuple(token_ids)) for token_ids in expected_tokens],
    )

    snapshots = snapshot_constraint_metrics(reqs)
    for snapshot, token_ids in zip(snapshots, expected_tokens, strict=True):
        response = snapshot.materialize_constraint_metrics()
        token_records = _token_records(response)
        assert [record["token_id"] for record in token_records] == token_ids
        assert [record["output_index"] for record in token_records] == list(
            range(len(token_ids))
        )
        assert all(
            record["spec_source"] in {"accepted_draft", "bonus"}
            for record in token_records
        )

    del output, payload
    gc.collect()
    assert records._DEVICE_RETENTION_BYTES == baseline


def test_terminal_archive_handles_mixed_retained_and_fallback_steps(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for device-retained records")
    baseline = records._DEVICE_RETENTION_BYTES

    def payload_for(token_id: int, output_index: int):
        floats = torch.zeros(1, _float_width(), device="cuda")
        ints = torch.zeros(1, _int_width(), dtype=torch.int64, device="cuda")
        ints[0, layout.KERNEL_INT_TOKEN_ID] = token_id
        return ConstraintMetricsRecordPayload(
            tensor_batch=ConstraintMetricsTensorBatch(
                floats=floats,
                ints=ints,
                top_ks=[0],
                grammar_active=[True],
                output_indices=[output_index],
                vocab_size=32,
            ),
            row_spans=[(0, 0, 1)],
        )

    retained = payload_for(12, 0)
    retained_bytes = (
        retained.tensor_batch.floats.nbytes + retained.tensor_batch.ints.nbytes
    )
    monkeypatch.setattr(
        records, "_DEVICE_RETENTION_MAX_BYTES", baseline + retained_bytes
    )
    retained = retain_record_payload_on_device(retained, enabled=True)
    fallback = retain_record_payload_on_device(payload_for(13, 1), enabled=True)
    assert retained.device_retained
    assert retained.defer_commit_validation
    assert not fallback.device_retained
    assert not fallback.defer_commit_validation

    state = ConstraintMetricsState(
        config=ConstraintMetricsConfig("forensic", 0),
    )
    req = _request_with_state(state, [12, 13])
    first_output = ConstraintMetricsDeviceOutput([retained]).copy_to_host(
        lambda tensor: tensor.to("cpu")
    )
    first_output.consume(
        SimpleNamespace(reqs=[req]),
        [CommittedTokens(0, (12,))],
    )
    second_output = ConstraintMetricsDeviceOutput([fallback]).copy_to_host(
        lambda tensor: tensor.to("cpu")
    )
    second_output.consume(
        SimpleNamespace(reqs=[req]),
        [CommittedTokens(1, (13,))],
    )
    assert state.record_slices[0].payload.tensor_batch.floats.is_cuda
    assert not state.record_slices[1].payload.tensor_batch.floats.is_cuda

    req._finished = True
    snapshot = snapshot_constraint_metrics([req])[0]
    owned = unwrap_from_pickle(wrap_as_pickle(snapshot))
    response = owned.materialize_constraint_metrics()
    token_records = _token_records(response)
    assert [record["token_id"] for record in token_records] == [12, 13]
    assert [record["output_index"] for record in token_records] == [0, 1]
    assert not any(
        record_slice.payload.tensor_batch.floats.is_cuda
        or record_slice.payload.tensor_batch.ints.is_cuda
        for record_slice in owned.record_slices
    )

    del first_output, second_output, retained, fallback
    gc.collect()
    assert records._DEVICE_RETENTION_BYTES == baseline


def test_device_retention_budget_is_bounded_and_released(monkeypatch):
    baseline = records._DEVICE_RETENTION_BYTES
    monkeypatch.setattr(records, "_DEVICE_RETENTION_MAX_BYTES", baseline + 10)

    first = records._reserve_device_retention(6)
    assert first is not None
    assert records._reserve_device_retention(5) is None
    boundary = records._reserve_device_retention(4)
    assert boundary is not None
    assert records._DEVICE_RETENTION_BYTES == baseline + 10

    first.release()
    first.release()
    boundary.release()
    assert records._DEVICE_RETENTION_BYTES == baseline

    natural = records._reserve_device_retention(10)
    assert natural is not None
    del natural
    gc.collect()
    assert records._DEVICE_RETENTION_BYTES == baseline


def test_summary_commits_only_the_scheduler_selected_prefix():
    state = ConstraintMetricsState(
        config=ConstraintMetricsConfig("summary", 8),
    )
    payload = _archive_payload(
        [10, 11, 12],
        row_spans=[(0, 0, 3)],
        speculative=True,
        spec_sources=[
            layout.SPEC_SOURCE_ACCEPTED_DRAFT,
            layout.SPEC_SOURCE_ACCEPTED_DRAFT,
            layout.SPEC_SOURCE_NORMAL,
        ],
    )
    payload.tensor_batch.floats.zero_()

    record_slice = state.commit(payload, 0, CommittedTokens(0, (10, 11)))
    packed_rows = payload.packed_rows_for_slices([record_slice])
    extend_kernel_row_segments(
        [state.summary_accumulator],
        packed_rows.floats,
        packed_rows.ints,
        packed_rows.grammar_active,
        packed_rows.output_indices,
        [len(packed_rows.floats)],
        [False],
    )

    assert state.summary_accumulator.output_token_count == 2
    assert state.summary_accumulator.top_pressure_events[0]["output_index"] >= 0
    assert state.record_slices == []


def test_forensic_defers_token_validation_until_terminal_materialization():
    state = ConstraintMetricsState(
        config=ConstraintMetricsConfig("forensic", 0),
    )
    payload = _archive_payload([999], row_spans=[(0, 0, 1)])
    payload.tensor_batch.floats.zero_()
    payload.defer_commit_validation = True
    req = _request_with_state(state, [2], finished=True)

    host_output = ConstraintMetricsDeviceOutput([payload]).copy_to_host(
        lambda tensor: tensor.clone()
    )
    assert host_output.record_payloads[0].defer_commit_validation
    host_output.consume(
        SimpleNamespace(reqs=[req]),
        [CommittedTokens(0, (2,))],
    )
    response = snapshot_constraint_metrics([req])[0].materialize_constraint_metrics()

    assert response["status"] == "partially_dropped"
    assert response["tokens"]["count"] == 0
    assert response["warnings"] == [
        "constraint metrics are missing 1 committed token record(s)"
    ]


def test_forensic_terminal_validation_rejects_uncommitted_speculative_row():
    state = ConstraintMetricsState(
        config=ConstraintMetricsConfig("forensic", 0),
    )
    payload = _archive_payload(
        [2],
        row_spans=[(0, 0, 1)],
        speculative=True,
        spec_sources=[layout.SPEC_SOURCE_NORMAL],
    )
    payload.tensor_batch.floats.zero_()
    payload.defer_commit_validation = True
    req = _request_with_state(state, [2], finished=True)
    ConstraintMetricsHostOutput(record_payloads=[payload]).consume(
        SimpleNamespace(reqs=[req]),
        [CommittedTokens(0, (2,))],
    )
    snapshot = snapshot_constraint_metrics([req])[0]

    with pytest.raises(ValueError, match="speculative rows are not a prefix"):
        snapshot.materialize_constraint_metrics()


def test_forensic_terminal_validation_stops_before_mismatched_normal_row():
    state = ConstraintMetricsState(
        config=ConstraintMetricsConfig("forensic", 0),
    )
    payload = _archive_payload(
        [10, 999],
        row_spans=[(0, 0, 2)],
        speculative=True,
        spec_sources=[layout.SPEC_SOURCE_ACCEPTED_DRAFT, layout.SPEC_SOURCE_NORMAL],
    )
    payload.tensor_batch.floats.zero_()
    payload.defer_commit_validation = True
    req = _request_with_state(state, [10, 11], finished=True)
    ConstraintMetricsHostOutput(record_payloads=[payload]).consume(
        SimpleNamespace(reqs=[req]),
        [CommittedTokens(0, (10, 11))],
    )

    response = snapshot_constraint_metrics([req])[0].materialize_constraint_metrics()

    assert response["status"] == "partially_dropped"
    assert response["tokens"]["token_id"] == [10]
    assert response["summary"]["output_token_count"] == 1
    assert response["warnings"] == [
        "constraint metrics are missing 1 committed token record(s)"
    ]


def test_forensic_terminal_validation_allows_normal_prefill_before_speculative():
    state = ConstraintMetricsState(
        config=ConstraintMetricsConfig("forensic", 0),
    )
    req = _request_with_state(state, [1, 2], finished=True)
    for output_index, (token_id, speculative, spec_source) in enumerate(
        (
            (1, False, layout.SPEC_SOURCE_NORMAL),
            (2, True, layout.SPEC_SOURCE_ACCEPTED_DRAFT),
        )
    ):
        payload = _archive_payload(
            [token_id],
            output_indices=[output_index],
            row_spans=[(0, 0, 1)],
            speculative=speculative,
            spec_sources=[spec_source],
        )
        payload.tensor_batch.floats.zero_()
        payload.defer_commit_validation = True
        ConstraintMetricsHostOutput(record_payloads=[payload]).consume(
            SimpleNamespace(reqs=[req]),
            [CommittedTokens(output_index, (token_id,))],
        )

    response = snapshot_constraint_metrics([req])[0].materialize_constraint_metrics()

    assert response["status"] == "complete"
    assert response["tokens"]["token_id"] == [1, 2]


def test_token_mismatch_keeps_only_the_valid_prefix():
    state = ConstraintMetricsState(
        config=ConstraintMetricsConfig("forensic", 0),
    )
    payload = _archive_payload(
        [10, 999, 12],
        output_indices=[0, 1, 2],
        row_spans=[(0, 0, 3)],
    )
    payload.tensor_batch.floats.zero_()
    payload.tensor_batch.ints[:, layout.KERNEL_INT_EFFECTIVE_LEGAL_TOKEN_COUNT] = 1

    state.commit(payload, 0, CommittedTokens(0, (10, 11, 12)))

    assert state.captured_prefix_length == 1
    assert state.capture_failed_at == 1
    assert len(state.record_slices) == 1
    assert state.record_slices[0].count == 1

    req = SimpleNamespace(
        origin_input_ids=[1],
        output_ids_through_stop=[10, 11, 12],
        grammar=_Grammar(),
        sampling_params=_sampling_params(
            custom_params={},
            ebnf="start: /x/",
            regex=None,
            json_schema=None,
            structural_tag=None,
            temperature=1.0,
            top_k=-1,
            top_p=1.0,
            min_p=0.0,
        ),
    )
    response = state.snapshot(req, state.record_slices).materialize_constraint_metrics()

    assert response["status"] == "partially_dropped"
    assert response["tokens"]["count"] == 1
    assert response["warnings"] == [
        "constraint metrics are missing 2 committed token record(s)"
    ]


def test_summary_batch_converts_each_payload_once_without_retaining_it():
    config = ConstraintMetricsConfig("summary", 0)
    states = [
        ConstraintMetricsState(
            config=config,
        )
        for _ in range(2)
    ]
    reqs = [
        SimpleNamespace(
            _constraint_metrics_state=state,
            grammar=_Grammar("summary"),
            is_retracted=False,
        )
        for state in states
    ]
    payload = _archive_payload(
        [12, 22],
        row_spans=[(0, 0, 1), (1, 1, 1)],
    )
    payload.tensor_batch.floats.zero_()
    payload.tensor_batch.ints[:, layout.KERNEL_INT_EFFECTIVE_LEGAL_TOKEN_COUNT] = 1
    output = ConstraintMetricsHostOutput(
        record_payloads=[payload],
    )

    with (
        patch.object(
            ConstraintMetricsTensorBatch,
            "to_packed_rows",
            autospec=True,
            side_effect=ConstraintMetricsTensorBatch.to_packed_rows,
        ) as to_packed_rows,
        patch.object(
            ConstraintMetricsRecordPayload,
            "move_to_pageable_storage",
            autospec=True,
            side_effect=ConstraintMetricsRecordPayload.move_to_pageable_storage,
        ) as move_to_pageable,
    ):
        output.consume(
            SimpleNamespace(reqs=reqs),
            [CommittedTokens(0, (12,)), CommittedTokens(0, (22,))],
        )

    assert to_packed_rows.call_count == 1
    assert move_to_pageable.call_count == 0
    assert [state.summary_accumulator.output_token_count for state in states] == [1, 1]


def test_segmented_packed_summary_updates_requests_independently():
    batches = [
        (
            [
                {"token_id": 10, "pressure": 1.5, "legal_count": 1},
                {"token_id": 11, "pressure": 1.5, "argmax_masked": True},
                {"token_id": 12, "pressure": 0.25, "blocked_count": 0},
                {"token_id": 20, "pressure": 0.0, "grammar_active": False},
                {"token_id": 21, "pressure": 2.0, "empty_support": True},
            ],
            [3, 2],
            [0, 1, 2, 0, 1],
        ),
        (
            [
                {"token_id": 13, "pressure": 2.0, "legal_count": 1024},
                {"token_id": 14, "pressure": 0.0},
                {"token_id": 22, "pressure": 1.25},
                {"token_id": 23, "pressure": 1.25},
                {"token_id": 24, "eos_only_support": True},
            ],
            [2, 3],
            [3, 4, 2, 3, 4],
        ),
    ]
    actual = [ConstraintMetricsSummaryAccumulator(top_k_events=2) for _ in range(2)]
    first_request_triggers = []

    for row_specs, row_counts, output_indices in batches:
        tensor_batch = _metric_payload(row_specs).tensor_batch
        packed_rows = tensor_batch.to_packed_rows(
            list(range(len(row_specs))), output_indices=output_indices
        )
        actual_triggers = extend_kernel_row_segments(
            actual,
            packed_rows.floats,
            packed_rows.ints,
            packed_rows.grammar_active,
            packed_rows.output_indices,
            row_counts,
            [True, False],
        )

        first_request_triggers.append(actual_triggers[0])
        assert actual_triggers[1] == [False] * row_counts[1]

    assert first_request_triggers == [[True, True, False], [True, False]]
    first, second = (accumulator.to_summary() for accumulator in actual)
    assert first["output_token_count"] == 5
    assert first["grammar_active_token_count"] == 5
    assert first["finite_active_token_count"] == 5
    assert first["cumulative_constraint_pressure_nats"] == pytest.approx(5.25)
    assert first["max_constraint_pressure_nats"] == 2.0
    assert first["max_pressure_output_index"] == 3
    assert first["argmax_masked_count"] == 1
    assert first["longest_high_pressure_run"] == 2

    assert second["output_token_count"] == 5
    assert second["grammar_active_token_count"] == 4
    assert second["finite_active_token_count"] == 3
    assert second["cumulative_constraint_pressure_nats"] == pytest.approx(2.5)
    assert second["max_constraint_pressure_nats"] == 1.25
    assert second["max_pressure_output_index"] == 2
    assert second["empty_support_count"] == 1
    assert second["eos_only_support_count"] == 1
    assert second["longest_high_pressure_run"] == 2


def test_summary_consumes_records_without_retaining_payload_history():
    config = ConstraintMetricsConfig("summary", 8)
    req = SimpleNamespace(
        origin_input_ids=[10],
        output_ids_through_stop=[12, 13],
        inflight_middle_chunks=0,
        grammar=_Grammar(),
        finished=lambda: False,
        is_retracted=False,
        stream=False,
        sampling_params=_sampling_params(
            custom_params={},
            json_schema=None,
            regex=None,
            ebnf="start: /x/",
            structural_tag=None,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
        ),
    )
    state = bind_constraint_metrics_state(
        req,
        config=config,
    )
    for output_index, token_id in enumerate((12, 13)):
        payload = _archive_payload(
            [token_id],
            row_spans=[(0, 0, 1)],
        )
        payload.tensor_batch.floats.zero_()
        payload.tensor_batch.ints[0, layout.KERNEL_INT_EFFECTIVE_LEGAL_TOKEN_COUNT] = 1
        ConstraintMetricsHostOutput(
            record_payloads=[payload],
        ).consume(
            SimpleNamespace(reqs=[req]),
            [CommittedTokens(output_index, (token_id,))],
        )

    assert state.record_slices == []
    assert state.summary_accumulator.output_token_count == 2
    deferred = state.snapshot(req, [])
    assert deferred.validation_token_ids == []
    assert deferred.input_token_count == 1
    assert deferred.output_token_count == 2
    response = deferred.materialize_constraint_metrics()
    assert response["summary"]["output_token_count"] == 2
    assert response["input_token_count"] == 1
    assert response["output_token_count"] == 2
    assert response["tokens"] == []


def test_triggered_no_event_returns_empty_columns_and_complete_summary():
    boundary = torch.tensor(SEVERE_PRESSURE_THRESHOLD_NATS, dtype=torch.float32)
    below_boundary = torch.nextafter(boundary, torch.tensor(float("-inf"))).item()
    row_specs = [
        {"token_id": 12},
        {"token_id": 13, "pressure": below_boundary},
    ]
    payload = _metric_payload(row_specs)
    state = ConstraintMetricsState(
        config=ConstraintMetricsConfig("triggered", 2),
    )
    req = _request_with_state(state, [12, 13], finished=True)
    req.grammar = _Grammar("triggered", top_k=2)
    _enable_metrics_stream(req)

    ConstraintMetricsHostOutput(record_payloads=[payload]).consume(
        SimpleNamespace(reqs=[req]),
        [CommittedTokens(0, (12, 13))],
    )

    stream_records = _materialize_stream_records(req)
    assert stream_records == [None, None]

    snapshot = snapshot_constraint_metrics([req])[0]
    assert snapshot.validation_token_ids == []
    response = snapshot.materialize_constraint_metrics()
    assert response["status"] == "complete"
    assert "tokens" not in response
    assert response["summary"]["output_token_count"] == 2
    assert response["summary"]["argmax_masked_count"] == 0
    assert response["summary"]["max_constraint_pressure_nats"] == pytest.approx(
        below_boundary
    )
    assert response["output_token_count"] == 2
    assert response["triggered_token_count"] == 0


def test_triggered_rules_retain_sparse_rows_in_output_order_with_exact_summary():
    boundary = torch.tensor(SEVERE_PRESSURE_THRESHOLD_NATS, dtype=torch.float32)
    below_boundary = torch.nextafter(boundary, torch.tensor(float("-inf"))).item()
    at_boundary = boundary.item()
    assert below_boundary < SEVERE_PRESSURE_THRESHOLD_NATS <= at_boundary
    row_specs = [
        {"pressure": below_boundary},
        {"pressure": at_boundary},
        {"argmax_masked": True},
        {"empty_support": True},
        {"eos_only_support": True},
        {"has_non_finite_logits": True},
        {"emitted_allowed": False},
        {
            "pressure": at_boundary + 1.0,
            "argmax_masked": True,
            "empty_support": True,
            "eos_only_support": True,
            "has_non_finite_logits": True,
            "emitted_allowed": False,
            "grammar_active": False,
        },
        {},
    ]
    token_ids = list(range(10, 10 + len(row_specs)))
    triggered_state = ConstraintMetricsState(
        config=ConstraintMetricsConfig("triggered", 2),
    )
    triggered_req = _request_with_state(triggered_state, token_ids, finished=True)
    triggered_req.grammar = _Grammar("triggered", top_k=2)
    ConstraintMetricsHostOutput(record_payloads=[_metric_payload(row_specs)]).consume(
        SimpleNamespace(reqs=[triggered_req]),
        [CommittedTokens(0, tuple(token_ids))],
    )

    expected_indices = [1, 2, 3, 4, 5, 6]
    retained_payloads = {
        id(record.payload): record.payload for record in triggered_state.record_slices
    }
    assert len(retained_payloads) == 1
    assert next(iter(retained_payloads.values())).tensor_batch.floats.shape[0] == len(
        expected_indices
    )
    summary_state = ConstraintMetricsState(
        config=ConstraintMetricsConfig("summary", 2),
    )
    summary_req = _request_with_state(summary_state, token_ids, finished=True)
    summary_req.grammar = _Grammar("summary", top_k=2)
    ConstraintMetricsHostOutput(
        record_payloads=[_metric_payload(row_specs, top_k=0)]
    ).consume(
        SimpleNamespace(reqs=[summary_req]),
        [CommittedTokens(0, tuple(token_ids))],
    )

    triggered_response = snapshot_constraint_metrics([triggered_req])[
        0
    ].materialize_constraint_metrics()
    summary_response = snapshot_constraint_metrics([summary_req])[
        0
    ].materialize_constraint_metrics()

    tokens = triggered_response["tokens"]
    assert set(tokens) == {
        "count",
        "output_index",
        "token_id",
        "grammar_active",
        "effective_legal_token_count",
        "constraint_pressure_nats",
        "argmax_masked",
        "unconstrained_top1",
        "entropy_nats",
        "emitted",
        "flags",
        "spec_source",
    }
    assert tokens["count"] == len(expected_indices)
    assert tokens["output_index"] == expected_indices
    assert tokens["token_id"] == [token_ids[index] for index in expected_indices]
    assert triggered_response["summary"] == summary_response["summary"]
    assert triggered_response["summary"]["output_token_count"] == len(row_specs)
    assert triggered_response["status"] == "complete"
    assert triggered_response["triggered_token_count"] == len(expected_indices)


def test_triggered_speculative_retains_only_the_committed_speculative_prefix():
    row_specs = [
        {
            "token_id": 30,
            "argmax_masked": True,
            "spec_source": layout.SPEC_SOURCE_ACCEPTED_DRAFT,
        },
        {
            "token_id": 31,
            "empty_support": True,
            "spec_source": layout.SPEC_SOURCE_BONUS,
        },
        {
            "token_id": 32,
            "has_non_finite_logits": True,
            "spec_source": layout.SPEC_SOURCE_NORMAL,
        },
    ]
    state = ConstraintMetricsState(
        config=ConstraintMetricsConfig("triggered", 2),
    )
    req = _request_with_state(state, [30, 31], finished=True)
    req.grammar = _Grammar("triggered", top_k=2)
    _enable_metrics_stream(req)

    ConstraintMetricsHostOutput(
        record_payloads=[_metric_payload(row_specs, speculative=True)]
    ).consume(
        SimpleNamespace(reqs=[req]),
        [CommittedTokens(0, (30, 31))],
    )

    stream_records = _materialize_stream_records(req)
    assert [record["token_id"] for record in stream_records] == [30, 31]
    assert [record["spec_source"] for record in stream_records] == [
        "accepted_draft",
        "bonus",
    ]
    assert all(record["triggered"] is True for record in stream_records)

    response = snapshot_constraint_metrics([req])[0].materialize_constraint_metrics()
    assert response["status"] == "complete"
    assert response["output_token_count"] == 2
    assert response["summary"]["output_token_count"] == 2
    assert "tokens" not in response
    assert response["triggered_token_count"] == 2


def test_triggered_payload_can_mix_streaming_and_buffered_requests():
    states = [
        ConstraintMetricsState(
            config=ConstraintMetricsConfig(
                mode,
                2,
                stream_all_token_metrics=index == 1,
            ),
        )
        for index, mode in enumerate(("triggered", "triggered", "summary"))
    ]
    reqs = [
        _request_with_state(states[0], [30, 31]),
        _request_with_state(states[1], [40, 41, 42]),
        _request_with_state(states[2], [50]),
    ]
    reqs[0].grammar = _Grammar("triggered", top_k=2)
    reqs[1].grammar = _Grammar("triggered", top_k=2)
    reqs[2].grammar = _Grammar("summary", top_k=2)
    _enable_metrics_stream(reqs[1])

    payload = _metric_payload(
        [
            {"token_id": 30, "pressure": 1.5},
            {"token_id": 31, "pressure": 0.1},
            {"token_id": 40, "pressure": 0.1},
            {"token_id": 41, "pressure": 1.5},
            {"token_id": 42, "argmax_masked": True},
            {"token_id": 50, "pressure": 1.5},
        ]
    )
    payload.row_spans = [(0, 0, 2), (1, 2, 3), (2, 5, 1)]
    ConstraintMetricsHostOutput(record_payloads=[payload]).consume(
        SimpleNamespace(reqs=reqs),
        [
            CommittedTokens(0, (30, 31)),
            CommittedTokens(0, (40, 41, 42)),
            CommittedTokens(0, (50,)),
        ],
    )

    streamed = _materialize_stream_records(reqs[1])
    assert [record["output_index"] for record in streamed] == [0, 1, 2]
    assert [record["token_id"] for record in streamed] == [40, 41, 42]
    assert [record["triggered"] for record in streamed] == [False, True, True]
    assert all("top_masked" not in record for record in streamed)
    assert reqs[0].customized_info is None
    assert reqs[2].customized_info is None
    assert [state.summary_accumulator.output_token_count for state in states] == [
        2,
        3,
        1,
    ]
    assert [state.triggered_event_count for state in states] == [1, 2, 0]


def test_forensic_dense_stream_preserves_full_terminal_payload():
    row_specs = [
        {"token_id": 30, "pressure": 0.1},
        {"token_id": 31, "pressure": 1.5},
    ]
    state = ConstraintMetricsState(
        config=ConstraintMetricsConfig(
            "forensic",
            2,
            stream_all_token_metrics=True,
        ),
        streaming=True,
    )
    req = _request_with_state(state, [30, 31], finished=True)
    req.grammar = _Grammar(
        "forensic",
        top_k=2,
        stream_all_token_metrics=True,
    )
    _enable_metrics_stream(req)
    full_payload = _metric_payload(row_specs, top_k=2)
    full_payload.defer_commit_validation = True
    stream_payload = _metric_payload(row_specs, top_k=0)

    ConstraintMetricsHostOutput(
        record_payloads=[full_payload],
        stream_record_payloads=[stream_payload],
    ).consume(
        SimpleNamespace(reqs=[req]),
        [CommittedTokens(0, (30, 31))],
    )

    streamed = _materialize_stream_records(req)
    assert [record["token_id"] for record in streamed] == [30, 31]
    assert [record["triggered"] for record in streamed] == [False, True]
    assert all("top_masked" not in record for record in streamed)
    assert state.record_slices[0].payload is full_payload

    response = snapshot_constraint_metrics([req])[0].materialize_constraint_metrics()
    assert response["mode"] == "forensic"
    assert response["tokens"]["count"] == 2
    assert response["tokens"]["top_masked"]["width"] == 2
    assert response["tokens"]["top_legal_after_conditioning"]["width"] == 2


def test_forensic_dense_stream_drops_mismatched_retained_row_before_publish():
    state = ConstraintMetricsState(
        config=ConstraintMetricsConfig(
            "forensic",
            2,
            stream_all_token_metrics=True,
        ),
        streaming=True,
    )
    req = _request_with_state(state, [11], finished=True)
    req.grammar = _Grammar(
        "forensic",
        top_k=2,
        stream_all_token_metrics=True,
    )
    _enable_metrics_stream(req)
    full_payload = _metric_payload([{"token_id": 999}], top_k=2)
    full_payload.defer_commit_validation = True
    stream_payload = _metric_payload([{"token_id": 999}], top_k=0)

    ConstraintMetricsHostOutput(
        record_payloads=[full_payload],
        stream_record_payloads=[stream_payload],
    ).consume(
        SimpleNamespace(reqs=[req]),
        [CommittedTokens(0, (11,))],
    )

    assert _materialize_stream_records(req) == []
    assert state.capture_failed_at == 0
    response = snapshot_constraint_metrics([req])[0].materialize_constraint_metrics()
    assert response["status"] == "partially_dropped"
    assert response["tokens"]["count"] == 0


def test_triggered_stream_storage_grows_lazily_and_releases_sent_records():
    req = SimpleNamespace(
        sampling_params=_sampling_params(
            custom_params=_custom("triggered"),
            ebnf="start: /x/",
            max_new_tokens=4,
        ),
        stream=True,
        customized_info=None,
        send_token_offset=0,
        origin_input_ids=[],
        output_ids_through_stop=[],
    )
    state = bind_constraint_metrics_state(
        req,
        config=ConstraintMetricsConfig("triggered", 2),
    )

    assert state.streaming is True
    records = req.customized_info[CONSTRAINT_METRICS_STREAM_KEY]
    assert records == []

    records.append({"output_index": 0})
    req.output_ids_through_stop = [10, 11, 12]
    prepare_streamed_constraint_metrics([req])
    assert records == [{"output_index": 0}, None, None]
    records[2] = {"output_index": 2}

    req.send_token_offset = 2
    release_streamed_constraint_metrics([req])
    assert records == [None, None, {"output_index": 2}]
    assert state.stream_release_offset == 2

    req.send_token_offset = 3
    release_streamed_constraint_metrics([req])
    assert records == [None, None, None]
    assert state.stream_release_offset == 3


def test_chunked_prefill_middle_results_are_not_committed_as_summary_records():
    config = ConstraintMetricsConfig("summary", 8)
    state = ConstraintMetricsState(
        config=config,
    )
    req = SimpleNamespace(
        _constraint_metrics_state=state,
        inflight_middle_chunks=1,
        output_ids=[],
        output_ids_through_stop=[],
        finished=lambda: False,
        is_retracted=False,
        grammar=_Grammar(),
    )
    payload = _archive_payload([12], row_spans=[(0, 0, 1)])
    payload.tensor_batch.floats.zero_()
    payload.tensor_batch.ints[0, layout.KERNEL_INT_EFFECTIVE_LEGAL_TOKEN_COUNT] = 1
    # Reproduce the production failure shape: a long prompt produced 17 sampled
    # rows from middle chunks before its final prefill result committed a token.
    for _ in range(17):
        output = ConstraintMetricsHostOutput(record_payloads=[payload])
        output.consume(SimpleNamespace(reqs=[req]), [None])
        assert output.record_payloads == []

    assert state.summary_accumulator.output_token_count == 0

    req.inflight_middle_chunks = 0
    output = ConstraintMetricsHostOutput(record_payloads=[payload])
    output.consume(SimpleNamespace(reqs=[req]), [CommittedTokens(0, (12,))])
    assert state.summary_accumulator.output_token_count == 1


def test_host_output_consumes_rows_once_and_releases_uncommitted_rows():
    states = [
        ConstraintMetricsState(
            config=ConstraintMetricsConfig("summary", 0),
        )
        for _ in range(2)
    ]
    reqs = [
        _request_with_state(states[0], [12]),
        _request_with_state(states[1], [13]),
    ]
    payload = _archive_payload(
        [12, 13],
        row_spans=[(0, 0, 1), (1, 1, 1)],
    )
    payload.tensor_batch.floats.zero_()
    output = ConstraintMetricsHostOutput(record_payloads=[payload])

    output.consume(
        SimpleNamespace(reqs=reqs),
        [CommittedTokens(0, (12,)), None],
    )
    assert states[0].summary_accumulator.output_token_count == 1
    assert states[1].summary_accumulator.output_token_count == 0
    assert output.record_payloads == []


def test_capture_scratch_survives_diagnostic_batch_growth():
    observer = _observer()
    grammars = [_Grammar("summary") for _ in range(4)]

    first = SimpleNamespace(
        custom_params=[
            _custom("summary"),
            _custom("summary"),
        ],
        grammars=grammars[:2],
        logit_bias=None,
        temperatures=torch.ones(2, 1),
        grammar_mask=_dense_blocked_mask(2, 4),
    )
    observer.before_grammar(torch.ones(2, 4), first)

    second = SimpleNamespace(
        custom_params=[
            _custom("summary"),
            _custom("summary"),
            _custom("summary"),
            _custom("summary"),
        ],
        grammars=grammars,
        logit_bias=None,
        temperatures=torch.ones(4, 1),
        grammar_mask=_dense_blocked_mask(4, 4),
    )
    state = observer.before_grammar(torch.ones(4, 4), second)

    assert state.inputs.pre.shape == (4, 4)


def test_reserved_capture_scratch_rejects_growth():
    observer = _observer()
    observer.reserve_capture_scratch(max_rows=2, vocab_size=4, device="cpu")
    info = SimpleNamespace(
        custom_params=[_custom("summary")] * 3,
        grammars=[_Grammar("summary") for _ in range(3)],
        logit_bias=None,
        temperatures=torch.ones(3, 1),
        grammar_mask=_dense_blocked_mask(3, 4),
    )

    with pytest.raises(RuntimeError, match="startup-reserved workspace"):
        observer.before_grammar(torch.ones(3, 4), info)


def test_host_consume_defers_record_materialization_until_terminal_output():
    config = ConstraintMetricsConfig("forensic", 0)
    payload = _archive_payload([12], row_spans=[(0, 0, 1)])
    payload.tensor_batch.floats.zero_()
    payload.tensor_batch.ints[0, layout.KERNEL_INT_EFFECTIVE_LEGAL_TOKEN_COUNT] = 1
    req = SimpleNamespace(
        origin_input_ids=[10],
        output_ids_through_stop=[12],
        inflight_middle_chunks=0,
        grammar=_Grammar(),
        finished=lambda: False,
        is_retracted=False,
        sampling_params=_sampling_params(
            custom_params={},
            json_schema=None,
            regex=None,
            ebnf="start: /x/",
            structural_tag=None,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
        ),
    )
    output = ConstraintMetricsHostOutput(
        record_payloads=[payload],
    )
    state = bind_constraint_metrics_state(
        req,
        config=config,
    )

    with patch.object(
        ConstraintMetricsTensorBatch,
        "to_columns",
        autospec=True,
        side_effect=ConstraintMetricsTensorBatch.to_columns,
    ) as materialize:
        output.consume(SimpleNamespace(reqs=[req]), [CommittedTokens(0, (12,))])
        assert materialize.call_count == 0
        packed = state.snapshot(req, state.record_slices)
        assert materialize.call_count == 0
        packed.materialize_constraint_metrics()
        assert materialize.call_count == 1


def test_terminal_packing_trims_batch_wide_top_k_columns():
    source_top_k = 20
    floats = torch.zeros(1, _float_width(source_top_k))
    ints = torch.zeros(1, _int_width(source_top_k), dtype=torch.int64)
    ints[
        0,
        layout.KERNEL_ROW_INT_COUNT : layout.KERNEL_ROW_INT_COUNT + source_top_k,
    ] = torch.arange(100, 100 + source_top_k)
    ints[0, layout.KERNEL_ROW_INT_COUNT + source_top_k :] = torch.arange(
        200, 200 + source_top_k
    )
    floats[
        0,
        layout.KERNEL_ROW_FLOAT_COUNT : layout.KERNEL_ROW_FLOAT_COUNT + source_top_k,
    ] = torch.arange(source_top_k)
    floats[0, layout.KERNEL_ROW_FLOAT_COUNT + source_top_k :] = torch.arange(
        40, 40 + source_top_k
    )
    payload = ConstraintMetricsRecordPayload(
        tensor_batch=ConstraintMetricsTensorBatch(
            floats=floats,
            ints=ints,
            top_ks=[8],
            grammar_active=[True],
            output_indices=[0],
            vocab_size=256,
        ),
        row_spans=[(0, 0, 1)],
    )

    packed = compact_record_payloads([_record_slice(payload, 0)], max_top_k=8)
    columns = packed.to_columns()

    assert packed.floats.shape == (1, _float_width(8))
    assert packed.ints.shape == (1, _int_width(8))
    assert columns["top_masked"]["token_id"] == list(range(100, 108))
    assert columns["top_masked"]["prob"] == [float(value) for value in range(8)]
    assert columns["top_legal_after_conditioning"]["token_id"] == list(range(200, 208))
    assert columns["top_legal_after_conditioning"]["constrained_prob"] == [
        float(value) for value in range(40, 48)
    ]


def test_batch_terminal_archive_stays_shared_during_transport():
    payload = _archive_payload(
        [12, 22],
        row_spans=[(0, 0, 1), (1, 1, 1)],
    )
    payload.tensor_batch.floats.zero_()
    payload.tensor_batch.ints[:, layout.KERNEL_INT_EFFECTIVE_LEGAL_TOKEN_COUNT] = 1
    states = [
        ConstraintMetricsState(config=ConstraintMetricsConfig("forensic", 0))
        for _ in range(2)
    ]
    for batch_index, state in enumerate(states):
        record_slice = state.commit(
            payload,
            batch_index,
            CommittedTokens(0, (12 + 10 * batch_index,)),
        )
        assert record_slice is not None
    reqs = [
        SimpleNamespace(
            origin_input_ids=[10 + index],
            output_ids_through_stop=[12 + 10 * index],
            grammar=_Grammar(),
            sampling_params=_sampling_params(
                custom_params={},
                ebnf="start: /x/",
                regex=None,
                json_schema=None,
                structural_tag=None,
                temperature=1.0,
                top_k=-1,
                top_p=1.0,
                min_p=0.0,
            ),
        )
        for index in range(2)
    ]
    for req, state in zip(reqs, states, strict=True):
        req._constraint_metrics_state = state
        req.finished = lambda: True

    with (
        patch.object(
            ConstraintMetricsTensorBatch,
            "to_columns",
            autospec=True,
            side_effect=ConstraintMetricsTensorBatch.to_columns,
        ) as materialize,
        patch(
            "sglang_constraint_metrics.observer.compact_record_payloads",
            wraps=compact_record_payloads,
        ) as compact,
    ):
        deferred = snapshot_constraint_metrics(reqs)
        assert materialize.call_count == 0
        assert all(state.record_slices == [] for state in states)
        assert (
            deferred[0].record_slices[0].payload is deferred[1].record_slices[0].payload
        )
        assert [
            metadata.record_slices[0].payload.tensor_batch.floats.shape[0]
            for metadata in deferred
        ] == [2, 2]
        shared_batch = deferred[0].record_slices[0].payload.tensor_batch
        for tensor in (shared_batch.floats, shared_batch.ints):
            assert tensor.storage_offset() == 0
            assert (
                tensor.untyped_storage().nbytes()
                == tensor.numel() * tensor.element_size()
            )

        pack_calls = compact.call_count
        owned = pickle.loads(pickle.dumps(deferred))
        pickle.dumps(owned)
        assert compact.call_count == pack_calls
        assert owned[0].record_slices[0].payload is owned[1].record_slices[0].payload
        responses = [metadata.materialize_constraint_metrics() for metadata in owned]

    assert materialize.call_count == 2
    assert compact.call_count == 0
    assert [_token_records(response)[0]["token_id"] for response in responses] == [
        12,
        22,
    ]
    assert owned[0].record_slices[0].payload is owned[1].record_slices[0].payload

    record_slice = owned[0].record_slices[0]
    owned[0].record_slices = [
        CommittedRecordSlice(
            payload=record_slice.payload,
            start=record_slice.start,
            count=record_slice.count,
            output_index_start=1,
        )
    ]
    with pytest.raises(RuntimeError, match="not a contiguous prefix"):
        owned[0].materialize_constraint_metrics()


def test_terminal_snapshot_failure_preserves_request_history():
    payload = _archive_payload([12], row_spans=[(0, 0, 1)])
    payload.tensor_batch.floats.zero_()
    state = ConstraintMetricsState(config=ConstraintMetricsConfig("forensic", 0))
    record_slice = state.commit(payload, 0, CommittedTokens(0, (12,)))
    assert record_slice is not None
    req = SimpleNamespace(
        _constraint_metrics_state=state,
        finished=lambda: True,
    )

    with (
        patch.object(
            ConstraintMetricsState,
            "snapshot",
            side_effect=RuntimeError("injected snapshot failure"),
        ),
        pytest.raises(RuntimeError, match="injected snapshot failure"),
    ):
        snapshot_constraint_metrics([req])

    assert len(state.record_slices) == 1
    assert state.record_slices[0].payload is payload


def test_terminal_d2h_synchronizes_queued_copies_after_failure():
    device = torch.device("cuda", 0)
    floats = Mock(is_cuda=True, shape=(1, 1), dtype=torch.float32, device=device)
    ints = Mock(is_cuda=True, shape=(1, 1), dtype=torch.int64, device=device)
    batch = SimpleNamespace(floats=floats, ints=ints)
    payload = SimpleNamespace(tensor_batch=batch)
    history = [[SimpleNamespace(payload=payload)]]
    host = Mock()
    stream = Mock()

    with (
        patch.object(
            torch,
            "empty",
            side_effect=[host, RuntimeError("injected allocation failure")],
        ),
        patch.object(torch.cuda, "current_stream", return_value=stream),
        pytest.raises(RuntimeError, match="injected allocation failure"),
    ):
        records.move_record_payloads_to_host(history)

    host.copy_.assert_called_once_with(floats, non_blocking=True)
    stream.synchronize.assert_called_once_with()


def test_terminal_archive_packs_aligned_histories_request_major():
    payloads = [
        _archive_payload(
            token_ids,
            physical_top_k=8,
            requested_top_ks=[1, 1],
            output_indices=[output_index] * 2,
            row_spans=[(0, 0, 1), (1, 1, 1)],
        )
        for output_index, token_ids in enumerate(([10, 20], [11, 21]))
    ]
    histories = [
        [
            _record_slice(payload, batch_index, output_index=output_index)
            for output_index, payload in enumerate(payloads)
        ]
        for batch_index in range(2)
    ]

    archived = archive_record_payloads(histories)

    assert len(archived[0]) == 1
    assert archived[0][0].payload is archived[1][0].payload
    assert archived[0][0].count == archived[1][0].count == 2
    assert archived[0][0].payload.tensor_batch.floats.shape[1] == _float_width(1)
    assert archived[0][0].payload.tensor_batch.ints.shape[1] == _int_width(1)
    request_token_ids = [
        compact_record_payloads(history, max_top_k=0).to_columns()["token_id"]
        for history in archived
    ]
    assert request_token_ids == [
        [10, 11],
        [20, 21],
    ]


def test_terminal_archive_splits_mixed_requested_top_k():
    payload = _archive_payload(
        [10, 20],
        physical_top_k=8,
        requested_top_ks=[1, 8],
        row_spans=[(0, 0, 1), (1, 1, 1)],
    )

    archived = archive_record_payloads(
        [[_record_slice(payload, 0)], [_record_slice(payload, 1)]]
    )

    assert archived[0][0].payload is not archived[1][0].payload
    assert archived[0][0].payload.tensor_batch.floats.shape[1] == _float_width(1)
    assert archived[0][0].payload.tensor_batch.ints.shape[1] == _int_width(1)
    assert archived[1][0].payload.tensor_batch.floats.shape[1] == _float_width(8)
    assert archived[1][0].payload.tensor_batch.ints.shape[1] == _int_width(8)


def test_sparse_terminal_archive_gathers_once_and_survives_ipc_round_trip():
    payloads = [
        _archive_payload(
            list(range(base, base + 6)),
            row_spans=[],
            speculative=True,
        )
        for base in (10, 20)
    ]
    histories = [
        [
            CommittedRecordSlice(payloads[0], 5, 1, 0),
            CommittedRecordSlice(payloads[1], 1, 1, 1),
        ],
        [
            CommittedRecordSlice(payloads[0], 2, 1, 0),
            CommittedRecordSlice(payloads[1], 4, 1, 1),
        ],
    ]

    archived = archive_record_payloads(histories)
    archived_payload = archived[0][0].payload
    assert archived_payload is archived[1][0].payload
    payloads[0].tensor_batch.ints.fill_(-1)
    payloads[1].tensor_batch.ints.fill_(-1)

    owned = unwrap_from_pickle(wrap_as_pickle(archived))
    assert owned[0][0].payload is owned[1][0].payload
    token_ids = (
        owned[0][0].payload.tensor_batch.ints[:, layout.KERNEL_INT_TOKEN_ID].tolist()
    )
    assert token_ids == [15, 21, 12, 24]
    for tensor in (
        owned[0][0].payload.tensor_batch.floats,
        owned[0][0].payload.tensor_batch.ints,
    ):
        assert tensor.storage_offset() == 0
        assert (
            tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
        )


def _assert_tensor_batches_equal(left, right):
    assert torch.equal(left.floats, right.floats)
    assert torch.equal(left.ints, right.ints)
    assert left.top_ks == right.top_ks
    assert left.grammar_active == right.grammar_active
    assert left.output_indices == right.output_indices
    assert left.vocab_size == right.vocab_size


def test_terminal_archive_preserves_speculative_multirow_filtering():
    spec_sources = [
        layout.SPEC_SOURCE_ACCEPTED_DRAFT,
        layout.SPEC_SOURCE_BONUS,
        layout.SPEC_SOURCE_NORMAL,
    ] * 2
    payloads = [
        _archive_payload(
            list(range(10 * output_index, 10 * output_index + 6)),
            physical_top_k=8,
            requested_top_ks=[1] * 6,
            output_indices=[output_index] * 6,
            row_spans=[(0, 0, 3), (1, 3, 3)],
            speculative=True,
            spec_sources=spec_sources,
        )
        for output_index in range(2)
    ]
    histories = [
        [
            _record_slice(
                payload,
                batch_index,
                count=2,
                output_index=output_index * 2,
            )
            for output_index, payload in enumerate(payloads)
        ]
        for batch_index in range(2)
    ]
    expected = [compact_record_payloads(history, max_top_k=0) for history in histories]

    archived = archive_record_payloads(histories)

    assert all(len(history) == 1 for history in archived)
    assert archived[0][0].payload is archived[1][0].payload
    assert archived[0][0].payload.speculative
    assert archived[0][0].payload.tensor_batch.floats.shape[1] == _float_width(1)
    assert archived[0][0].payload.tensor_batch.ints.shape[1] == _int_width(1)
    for expected_batch, history in zip(expected, archived, strict=True):
        _assert_tensor_batches_equal(
            expected_batch,
            compact_record_payloads(history, max_top_k=0),
        )


def test_terminal_archive_supports_mixed_completion_subset():
    width = 3
    normal_payload = _archive_payload(
        list(range(100, 103)),
        physical_top_k=8,
        requested_top_ks=[1] * 3,
        row_spans=[(row, row, 1) for row in range(3)],
    )
    payloads = [
        _archive_payload(
            list(range(10 * step, 10 * step + 3 * width)),
            physical_top_k=8,
            requested_top_ks=[1] * (3 * width),
            output_indices=[step] * (3 * width),
            row_spans=[(row, row * width, width) for row in range(3)],
            speculative=True,
        )
        for step in range(2)
    ]
    histories = []
    for source_row, counts in ((0, (2, 1)), (2, (1, 2))):
        output_index = 1
        history = [CommittedRecordSlice(normal_payload, source_row, 1, 0)]
        for payload, count in zip(payloads, counts, strict=True):
            history.append(
                CommittedRecordSlice(
                    payload,
                    source_row * width,
                    count,
                    output_index,
                )
            )
            output_index += count
        histories.append(history)
    expected = [compact_record_payloads(history, max_top_k=0) for history in histories]

    archived = archive_record_payloads(histories)

    for expected_batch, history in zip(expected, archived, strict=True):
        _assert_tensor_batches_equal(
            expected_batch,
            compact_record_payloads(history, max_top_k=0),
        )


def test_terminal_archive_preserves_staggered_mixed_width_histories():
    first = _archive_payload(
        [10, 20],
        row_spans=[(0, 0, 1), (1, 1, 1)],
    )
    second = _archive_payload(
        [11],
        physical_top_k=8,
        requested_top_ks=[0],
        output_indices=[1],
        row_spans=[(0, 0, 1)],
    )
    histories = [
        [
            _record_slice(first, 0, output_index=0),
            _record_slice(second, 0, output_index=1),
        ],
        [_record_slice(first, 1, output_index=0)],
    ]
    expected = [compact_record_payloads(history, max_top_k=0) for history in histories]

    archived = archive_record_payloads(histories)

    assert [len(history) for history in archived] == [2, 1]
    for expected_batch, history in zip(expected, archived, strict=True):
        _assert_tensor_batches_equal(
            expected_batch,
            compact_record_payloads(history, max_top_k=0),
        )


def test_response_state_exists_before_first_sample():
    req = SimpleNamespace(
        origin_input_ids=[10],
        output_ids_through_stop=[],
        grammar=_Grammar(),
        sampling_params=_sampling_params(
            custom_params=_custom("summary"),
            json_schema=None,
            regex=None,
            ebnf="start: /x/",
            structural_tag=None,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
        ),
    )

    state = bind_constraint_metrics_state(
        req,
        config=config_from_sampling_custom_params(req.sampling_params.custom_params),
    )

    assert state is not None
    assert state.record_slices == []
    response = state.snapshot(req, []).materialize_constraint_metrics()
    assert response["status"] == "complete"
    assert response["summary"]["output_token_count"] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
