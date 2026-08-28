import json
import math
import sys
from unittest import mock

import pytest
import torch

import sglang_constraint_metrics.kernel as kernel
from sglang_constraint_metrics import layout
from sglang_constraint_metrics.kernel import (
    compute_constraint_metrics,
)
from sglang_constraint_metrics.mask import (
    MASK_DENSE_ALLOWED,
    MASK_DENSE_BLOCKED,
    MASK_PACKED_ALLOWED,
)
from sglang_constraint_metrics.records import (
    _SANITIZE_FILL,
    _compute_record_tensor_batch,
    _compute_record_tensor_batch_chunked,
)
from sglang_constraint_metrics.schema import (
    INF_PRESSURE_SENTINEL,
)


def _compute(
    pre,
    allowed,
    emitted,
    *,
    top_k,
    grammar_active=None,
    spec_sources=None,
    row_enabled=None,
    grammar_mask_kind=MASK_DENSE_ALLOWED,
    temperatures=None,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the constraint-metrics kernel")
    device = torch.device("cuda")
    rows = pre.shape[0]
    if grammar_active is None:
        grammar_active = torch.ones(rows, dtype=torch.bool)
    if spec_sources is None:
        spec_sources = torch.zeros(rows, dtype=torch.long)
    if temperatures is None:
        temperatures = torch.ones(rows, dtype=torch.float32)
    return compute_constraint_metrics(
        pre.to(device),
        allowed.to(device),
        temperatures.to(device),
        emitted.to(device),
        grammar_active.to(device),
        spec_sources.to(device),
        row_enabled=None if row_enabled is None else row_enabled.to(device),
        max_top_k=min(top_k, pre.shape[1]),
        eos_token_ids=list(range(min(3, pre.shape[1]))),
        sanitize_fill=_SANITIZE_FILL,
        inf_pressure=INF_PRESSURE_SENTINEL,
        grammar_mask_kind=grammar_mask_kind,
    )


def _effective_logits(pre: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(
        pre.float(), nan=-_SANITIZE_FILL, posinf=_SANITIZE_FILL, neginf=-_SANITIZE_FILL
    ).div(temperatures.float().reshape(-1, 1))


def _pack_allowed(allowed: torch.Tensor) -> torch.Tensor:
    rows, vocab_size = allowed.shape
    packed = torch.zeros(rows, (vocab_size + 31) // 32, dtype=torch.int64)
    for token_id in range(vocab_size):
        packed[:, token_id // 32] |= allowed[:, token_id].to(torch.int64) << (
            token_id % 32
        )
    return packed.to(torch.int32)


@pytest.mark.parametrize(
    "rows,vocab_size,top_k",
    [
        (1, 1, 0),
        (7, 33, 8),
        (7, 65, 32),
        (7, 4097, 20),
        (2, 202048, 20),
    ],
)
def test_kernel_packed_contract(rows, vocab_size, top_k):
    generator = torch.Generator().manual_seed(1234 + vocab_size)
    pre = torch.randn(rows, vocab_size, generator=generator) * 3.0
    allowed = torch.rand(rows, vocab_size, generator=generator) > 0.4
    allowed[:, 0] = True
    emitted = torch.randint(0, vocab_size, (rows,), generator=generator)
    if vocab_size >= 3:
        pre[:, 0] = float("inf")
        pre[:, 1] = float("-inf")
        pre[:, 2] = float("nan")
        allowed[:, 1:3] = False
    grammar_active = torch.tensor([(row % 2) == 0 for row in range(rows)])
    spec_sources = torch.tensor([row % 3 for row in range(rows)])

    floats, ints = _compute(
        pre,
        allowed,
        emitted,
        top_k=top_k,
        grammar_active=grammar_active,
        spec_sources=spec_sources,
    )
    torch.cuda.synchronize()

    width = min(top_k, vocab_size)
    assert floats.shape == (
        rows,
        layout.KERNEL_ROW_FLOAT_COUNT + layout.TOP_K_FLOAT_GROUP_COUNT * width,
    )
    assert ints.shape == (
        rows,
        layout.KERNEL_ROW_INT_COUNT + layout.TOP_K_INT_GROUP_COUNT * width,
    )
    assert torch.isfinite(floats).all()
    json.dumps(floats.cpu().tolist(), allow_nan=False)
    torch.testing.assert_close(ints[:, 2].cpu(), allowed.sum(dim=1))
    torch.testing.assert_close(ints[:, 3].cpu(), (~allowed).sum(dim=1))
    torch.testing.assert_close(ints[:, 4].cpu(), emitted)
    torch.testing.assert_close(ints[:, 5].cpu(), spec_sources)


def test_kernel_row_filter_preserves_enabled_rows():
    generator = torch.Generator().manual_seed(20260731)
    pre = torch.randn(4, 4097, generator=generator)
    allowed = torch.rand(4, 4097, generator=generator) > 0.4
    allowed[:, 0] = True
    emitted = torch.tensor([0, 1, 2, 3])
    enabled = torch.tensor([True, False, True, False])

    expected = _compute(pre, allowed, emitted, top_k=8)
    actual = _compute(
        pre,
        allowed,
        emitted,
        top_k=8,
        row_enabled=enabled,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual[0][enabled], expected[0][enabled])
    assert torch.equal(actual[1][enabled], expected[1][enabled])


def test_kernel_known_distribution():
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
    floats, ints = _compute(
        torch.log(probs),
        torch.tensor([[True, False, True, False]]),
        torch.tensor([2]),
        top_k=2,
    )
    torch.cuda.synchronize()
    floats = floats.cpu()
    ints = ints.cpu()

    assert float(floats[0, layout.FLOAT_PRESSURE]) == pytest.approx(
        -math.log(0.4), abs=1e-5
    )
    assert float(floats[0, layout.FLOAT_UNCONSTRAINED_TOP1_PROB]) == pytest.approx(
        0.4, abs=1e-5
    )
    assert float(floats[0, layout.FLOAT_TOP1_LOGIT_GAP]) == pytest.approx(
        math.log(0.4 / 0.3), abs=1e-5
    )
    assert float(floats[0, layout.FLOAT_EMITTED_MODEL_LOG_PROB]) == pytest.approx(
        math.log(0.3), abs=1e-5
    )
    assert int(ints[0, layout.KERNEL_INT_UNCONSTRAINED_TOP1_TOKEN_ID]) == 3
    assert int(ints[0, layout.KERNEL_INT_EMITTED_RANK]) == 2
    assert int(ints[0, layout.KERNEL_INT_EFFECTIVE_LEGAL_TOKEN_COUNT]) == 2
    assert int(ints[0, layout.KERNEL_INT_BLOCKED_TOKEN_COUNT]) == 2
    assert int(ints[0, layout.KERNEL_INT_TOKEN_ID]) == 2
    candidate_start = layout.KERNEL_ROW_INT_COUNT
    assert ints[0, candidate_start : candidate_start + 2].tolist() == [3, 1]
    assert ints[0, candidate_start + 2 : candidate_start + 4].tolist() == [2, 0]


def test_backend_native_mask_formats_are_bitwise_equivalent():
    generator = torch.Generator().manual_seed(8675309)
    pre = torch.randn(3, 65, generator=generator)
    pre[0, 0] = float("inf")
    pre[1, 1] = float("-inf")
    pre[2, 2] = float("nan")
    allowed = torch.rand(3, 65, generator=generator) > 0.4
    allowed[:, 3] = True
    emitted = torch.tensor([3, 4, 5])

    dense = _compute(
        pre,
        allowed,
        emitted,
        top_k=8,
        grammar_mask_kind=MASK_DENSE_ALLOWED,
    )
    blocked = _compute(
        pre,
        ~allowed,
        emitted,
        top_k=8,
        grammar_mask_kind=MASK_DENSE_BLOCKED,
    )
    packed = _compute(
        pre,
        _pack_allowed(allowed),
        emitted,
        top_k=8,
        grammar_mask_kind=MASK_PACKED_ALLOWED,
    )
    torch.cuda.synchronize()

    for actual in (blocked, packed):
        assert torch.equal(dense[0], actual[0])
        assert torch.equal(dense[1], actual[1])


def test_short_packed_mask_allows_logits_beyond_its_words():
    pre = torch.linspace(-1.0, 1.0, 65, dtype=torch.float32).unsqueeze(0)
    pre[0, 64] = 4.0
    dense_allowed = torch.zeros_like(pre, dtype=torch.bool)
    dense_allowed[:, 32:] = True
    short_packed = torch.zeros(1, 1, dtype=torch.int32)
    emitted = torch.tensor([64])

    dense = _compute(pre, dense_allowed, emitted, top_k=8)
    packed = _compute(
        pre,
        short_packed,
        emitted,
        top_k=8,
        grammar_mask_kind=MASK_PACKED_ALLOWED,
    )
    torch.cuda.synchronize()

    assert torch.equal(dense[0], packed[0])
    assert torch.equal(dense[1], packed[1])
    assert int(packed[1][0, layout.KERNEL_INT_EFFECTIVE_LEGAL_TOKEN_COUNT]) == 33
    assert int(packed[1][0, layout.KERNEL_INT_UNCONSTRAINED_TOP1_TOKEN_ID]) == 64
    assert not bool(packed[1][0, layout.KERNEL_INT_ARGMAX_MASKED])


def test_nonfinite_logits_are_sanitized_before_temperature_and_remain_legal():
    temperatures = torch.tensor([2.0])
    pre = torch.tensor([[float("inf"), 0.0]], dtype=torch.float32)
    expected_gap = float(torch.tensor(_SANITIZE_FILL, dtype=torch.float32) / 2.0)

    floats, ints = _compute(
        pre,
        torch.tensor([[True, False]]),
        torch.tensor([0]),
        top_k=1,
        temperatures=temperatures,
    )
    torch.cuda.synchronize()
    floats = floats.cpu()
    ints = ints.cpu()

    assert float(floats[0, layout.FLOAT_PRESSURE]) == pytest.approx(0.0, abs=1e-6)
    assert float(floats[0, layout.FLOAT_TOP1_LOGIT_GAP]) == pytest.approx(
        expected_gap, rel=1e-6
    )
    assert float(floats[0, layout.FLOAT_EMITTED_MODEL_LOG_PROB]) == pytest.approx(
        0.0, abs=1e-6
    )
    assert float(floats[0, layout.KERNEL_ROW_FLOAT_COUNT + 1]) == pytest.approx(
        1.0, abs=1e-6
    )
    assert int(ints[0, layout.KERNEL_INT_EFFECTIVE_LEGAL_TOKEN_COUNT]) == 1
    assert int(ints[0, layout.KERNEL_ROW_INT_COUNT + 1]) == 0
    assert bool(ints[0, layout.KERNEL_INT_HAS_NON_FINITE_LOGITS])
    assert bool(ints[0, layout.KERNEL_INT_EMITTED_ALLOWED])
    assert not bool(ints[0, layout.KERNEL_INT_EMPTY_SUPPORT])


def test_blocked_positive_infinity_contributes_to_pressure_and_candidates():
    temperatures = torch.tensor([2.0])
    pre = torch.tensor([[float("inf"), 0.0]], dtype=torch.float32)
    expected_pressure = float(torch.tensor(_SANITIZE_FILL, dtype=torch.float32) / 2.0)

    floats, ints = _compute(
        pre,
        torch.tensor([[False, True]]),
        torch.tensor([1]),
        top_k=1,
        temperatures=temperatures,
    )
    torch.cuda.synchronize()
    floats = floats.cpu()
    ints = ints.cpu()

    assert float(floats[0, layout.FLOAT_PRESSURE]) == pytest.approx(
        expected_pressure, rel=1e-6
    )
    assert float(floats[0, layout.KERNEL_ROW_FLOAT_COUNT + 1]) == pytest.approx(
        1.0, abs=1e-6
    )
    assert int(ints[0, layout.KERNEL_ROW_INT_COUNT]) == 0
    assert int(ints[0, layout.KERNEL_ROW_INT_COUNT + 1]) == 1
    assert bool(ints[0, layout.KERNEL_INT_ARGMAX_MASKED])
    assert bool(ints[0, layout.KERNEL_INT_HAS_NON_FINITE_LOGITS])


def test_all_nonfinite_logits_keep_sanitized_support_and_finite_outputs():
    pre = torch.tensor(
        [[float("nan"), float("inf"), float("-inf")]], dtype=torch.float32
    )
    floats, ints = _compute(
        pre,
        torch.ones_like(pre, dtype=torch.bool),
        torch.tensor([1]),
        top_k=3,
        temperatures=torch.tensor([4.0]),
    )
    torch.cuda.synchronize()
    floats = floats.cpu()
    ints = ints.cpu()

    assert torch.isfinite(floats).all()
    json.dumps(floats.tolist(), allow_nan=False)
    assert float(floats[0, layout.FLOAT_PRESSURE]) == pytest.approx(0.0, abs=1e-6)
    assert float(floats[0, layout.FLOAT_ENTROPY]) == pytest.approx(0.0, abs=1e-6)
    assert float(floats[0, layout.FLOAT_EMITTED_MODEL_LOG_PROB]) == pytest.approx(
        0.0, abs=1e-6
    )
    assert int(ints[0, layout.KERNEL_INT_EFFECTIVE_LEGAL_TOKEN_COUNT]) == 3
    assert bool(ints[0, layout.KERNEL_INT_HAS_NON_FINITE_LOGITS])
    assert bool(ints[0, layout.KERNEL_INT_EMITTED_ALLOWED])
    assert not bool(ints[0, layout.KERNEL_INT_EMPTY_SUPPORT])


def test_kernel_is_invariant_to_large_common_logit_offset():
    pre = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    shifted = pre + 1.0e8
    allowed = torch.tensor([[True, False]])
    emitted = torch.tensor([0])

    base_floats, base_ints = _compute(pre, allowed, emitted, top_k=1)
    shifted_floats, shifted_ints = _compute(shifted, allowed, emitted, top_k=1)
    torch.cuda.synchronize()

    assert torch.equal(base_floats, shifted_floats)
    assert torch.equal(base_ints, shifted_ints)
    assert float(shifted_floats[0, layout.FLOAT_PRESSURE]) == pytest.approx(
        math.log(2.0), abs=1e-6
    )
    assert float(shifted_floats[0, layout.FLOAT_ENTROPY]) == pytest.approx(
        math.log(2.0), abs=1e-6
    )
    assert float(
        shifted_floats[0, layout.FLOAT_EMITTED_MODEL_LOG_PROB]
    ) == pytest.approx(math.log(0.5), abs=1e-6)


def test_kernel_three_tile_metrics_match_float64_reference():
    vocab_size = 8193
    top_k = 4
    pre = torch.linspace(-6.0, 6.0, vocab_size, dtype=torch.float32).unsqueeze(0)
    allowed = (torch.arange(vocab_size) % 7 == 0).unsqueeze(0)
    emitted = torch.tensor([8190])
    temperatures = torch.tensor([1.7], dtype=torch.float32)

    floats, ints = _compute(
        pre,
        allowed,
        emitted,
        top_k=top_k,
        temperatures=temperatures,
    )
    torch.cuda.synchronize()
    floats = floats.cpu()
    ints = ints.cpu()

    effective = _effective_logits(pre, temperatures).double()[0]
    model_log_probs = torch.log_softmax(effective, dim=0)
    probabilities = torch.exp(model_log_probs)
    legal_mass = probabilities[allowed[0]].sum()
    pressure = -torch.log(legal_mass)
    entropy = -(probabilities * model_log_probs).sum()
    emitted_model_log_prob = model_log_probs[emitted.item()]
    top_values, top_ids = torch.topk(effective, 2)
    blocked_ids = torch.arange(vocab_size)[~allowed[0]]
    legal_ids = torch.arange(vocab_size)[allowed[0]]
    _, top_masked_indices = torch.topk(effective[~allowed[0]], top_k)
    _, top_legal_indices = torch.topk(effective[allowed[0]], top_k)
    top_masked_ids = blocked_ids[top_masked_indices]
    top_legal_ids = legal_ids[top_legal_indices]

    expected_floats = {
        layout.FLOAT_PRESSURE: pressure,
        layout.FLOAT_ENTROPY: entropy,
        layout.FLOAT_UNCONSTRAINED_TOP1_PROB: probabilities[top_ids[0]],
        layout.FLOAT_TOP1_LOGIT_GAP: top_values[0] - top_values[1],
        layout.FLOAT_EMITTED_MODEL_LOG_PROB: emitted_model_log_prob,
    }
    for column, expected in expected_floats.items():
        assert float(floats[0, column]) == pytest.approx(float(expected), abs=1e-4)

    assert int(ints[0, layout.KERNEL_INT_UNCONSTRAINED_TOP1_TOKEN_ID]) == int(
        top_ids[0]
    )
    assert int(ints[0, layout.KERNEL_INT_EFFECTIVE_LEGAL_TOKEN_COUNT]) == int(
        allowed.sum()
    )
    assert int(ints[0, layout.KERNEL_INT_BLOCKED_TOKEN_COUNT]) == int((~allowed).sum())
    assert bool(ints[0, layout.KERNEL_INT_ARGMAX_MASKED])
    assert bool(ints[0, layout.KERNEL_INT_EMITTED_ALLOWED])
    int_candidate_start = layout.KERNEL_ROW_INT_COUNT
    assert (
        ints[0, int_candidate_start : int_candidate_start + top_k].tolist()
        == top_masked_ids.tolist()
    )
    assert (
        ints[0, int_candidate_start + top_k : int_candidate_start + 2 * top_k].tolist()
        == top_legal_ids.tolist()
    )
    float_candidate_start = layout.KERNEL_ROW_FLOAT_COUNT
    for slot, token_id in enumerate(top_masked_ids):
        assert float(floats[0, float_candidate_start + slot]) == pytest.approx(
            float(probabilities[token_id]), abs=1e-6
        )
    for slot, token_id in enumerate(top_legal_ids):
        assert float(floats[0, float_candidate_start + top_k + slot]) == pytest.approx(
            float(probabilities[token_id] / legal_mass), abs=1e-6
        )


def test_kernel_keeps_q_in_log_domain_when_legal_mass_underflows():
    floats, ints = _compute(
        torch.tensor([[0.0, -120.0]], dtype=torch.float32),
        torch.tensor([[False, True]]),
        torch.tensor([1]),
        top_k=1,
    )
    torch.cuda.synchronize()
    floats = floats.cpu()
    ints = ints.cpu()

    assert float(floats[0, layout.FLOAT_PRESSURE]) == pytest.approx(120.0, abs=1e-5)
    assert float(floats[0, layout.FLOAT_EMITTED_MODEL_LOG_PROB]) == pytest.approx(
        -120.0, abs=1e-5
    )
    assert float(floats[0, layout.KERNEL_ROW_FLOAT_COUNT + 1]) == pytest.approx(
        1.0, abs=1e-6
    )
    assert int(ints[0, layout.KERNEL_ROW_INT_COUNT + 1]) == 1


def test_kernel_empty_support_uses_finite_sentinel():
    floats, ints = _compute(
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        torch.tensor([[False, False]]),
        torch.tensor([0]),
        top_k=0,
    )
    torch.cuda.synchronize()
    floats = floats.cpu()
    ints = ints.cpu()

    assert float(floats[0, layout.FLOAT_PRESSURE]) == float(
        torch.tensor(INF_PRESSURE_SENTINEL)
    )
    assert bool(ints[0, layout.KERNEL_INT_EMPTY_SUPPORT])
    json.dumps(floats.tolist(), allow_nan=False)


def test_kernel_ties_use_lowest_token_id():
    pre = torch.zeros(1, 8)
    allowed = torch.tensor([[False, True, False, True, False, True, False, True]])
    floats, ints = _compute(pre, allowed, torch.tensor([1]), top_k=4)
    torch.cuda.synchronize()
    floats = floats.cpu()
    ints = ints.cpu()

    assert int(ints[0, layout.KERNEL_INT_UNCONSTRAINED_TOP1_TOKEN_ID]) == 0
    assert bool(ints[0, layout.KERNEL_INT_ARGMAX_MASKED])
    assert float(floats[0, layout.FLOAT_TOP1_LOGIT_GAP]) == 0.0
    candidate_start = layout.KERNEL_ROW_INT_COUNT
    assert ints[0, candidate_start : candidate_start + 4].tolist() == [0, 2, 4, 6]
    assert ints[0, candidate_start + 4 : candidate_start + 8].tolist() == [
        1,
        3,
        5,
        7,
    ]


def test_kernel_isolates_stream_scratch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the constraint-metrics kernel")
    rows, vocab_size, top_k = 4, 4097, 8
    inputs = []
    expected = []
    for seed in (17, 29):
        generator = torch.Generator().manual_seed(seed)
        pre = torch.randn(rows, vocab_size, generator=generator)
        allowed = torch.rand(rows, vocab_size, generator=generator) > 0.5
        allowed[:, 0] = True
        emitted = torch.randint(0, vocab_size, (rows,), generator=generator)
        inputs.append((pre, allowed, emitted))
        expected.append(_compute(pre, allowed, emitted, top_k=top_k))
    torch.cuda.synchronize()

    outputs = []
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream, (pre, allowed, emitted) in zip(streams, inputs, strict=True):
        with torch.cuda.stream(stream):
            outputs.append(_compute(pre, allowed, emitted, top_k=top_k))
    torch.cuda.synchronize()

    for (floats, ints), (expected_floats, expected_ints) in zip(
        outputs, expected, strict=True
    ):
        torch.testing.assert_close(ints, expected_ints, atol=0, rtol=0)
        torch.testing.assert_close(floats, expected_floats, atol=0, rtol=0)


def test_kernel_k_specific_scratch_does_not_change_results():
    rows, vocab_size = 64, 16385
    generator = torch.Generator().manual_seed(41)
    pre = torch.randn(rows, vocab_size, generator=generator)
    allowed = torch.rand(rows, vocab_size, generator=generator) > 0.5
    allowed[:, 0] = True
    emitted = torch.randint(0, vocab_size, (rows,), generator=generator)

    expected_floats, expected_ints = _compute(pre, allowed, emitted, top_k=0)
    _compute(
        pre[:2, :1025],
        allowed[:2, :1025],
        emitted[:2].remainder(1025),
        top_k=32,
    )
    actual_floats, actual_ints = _compute(pre, allowed, emitted, top_k=0)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_ints, expected_ints, atol=0, rtol=0)
    torch.testing.assert_close(actual_floats, expected_floats, atol=0, rtol=0)


def test_sm103_k8_register_cap_preserves_results():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the constraint-metrics kernel")
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("SM103 is required for the register-cap specialization")
    generator = torch.Generator().manual_seed(20260802)
    rows, vocab_size = 7, 8193
    pre = torch.randn(rows, vocab_size, generator=generator)
    allowed = _pack_allowed(torch.rand(rows, vocab_size, generator=generator) > 0.4)
    emitted = torch.randint(0, vocab_size, (rows,), generator=generator)
    row_enabled = torch.tensor([True, True, False, True, False, True, True])

    with mock.patch.object(kernel, "_is_sm103", return_value=False):
        uncapped = _compute(
            pre,
            allowed,
            emitted,
            top_k=8,
            row_enabled=row_enabled,
            grammar_mask_kind=MASK_PACKED_ALLOWED,
        )
    with mock.patch.object(kernel, "_is_sm103", return_value=True):
        capped = _compute(
            pre,
            allowed,
            emitted,
            top_k=8,
            row_enabled=row_enabled,
            grammar_mask_kind=MASK_PACKED_ALLOWED,
        )
    torch.cuda.synchronize()

    for capped_tensor, uncapped_tensor in zip(capped, uncapped, strict=True):
        torch.testing.assert_close(
            capped_tensor[row_enabled],
            uncapped_tensor[row_enabled],
            atol=0,
            rtol=0,
        )


def test_kernel_validates_devices_and_temperature_shape():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the constraint-metrics kernel")
    pre = torch.zeros(1, 2, dtype=torch.float32, device="cuda")
    allowed = torch.ones_like(pre, dtype=torch.bool)
    temperatures = torch.ones(1, dtype=torch.float32, device="cuda")
    emitted = torch.zeros(1, dtype=torch.long, device="cuda")
    metadata_bool = torch.ones(1, dtype=torch.bool, device="cuda")
    metadata_int = torch.zeros(1, dtype=torch.long, device="cuda")
    kwargs = {
        "max_top_k": 0,
        "eos_token_ids": None,
        "sanitize_fill": _SANITIZE_FILL,
        "inf_pressure": INF_PRESSURE_SENTINEL,
    }

    with pytest.raises(ValueError, match="grammar_mask must be on the same device"):
        compute_constraint_metrics(
            pre,
            allowed.cpu(),
            temperatures,
            emitted,
            metadata_bool,
            metadata_int,
            **kwargs,
        )
    with pytest.raises(ValueError, match="temperatures must be on the same device"):
        compute_constraint_metrics(
            pre,
            allowed,
            temperatures.cpu(),
            emitted,
            metadata_bool,
            metadata_int,
            **kwargs,
        )
    with pytest.raises(ValueError, match="one value per logits row"):
        compute_constraint_metrics(
            pre,
            allowed,
            torch.ones(2, device="cuda"),
            emitted,
            metadata_bool,
            metadata_int,
            **kwargs,
        )


def test_kernel_accepts_strided_allowed():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the constraint-metrics kernel")
    rows, vocab_size = 3, 65
    pre = torch.randn(rows, vocab_size, device="cuda")
    allowed_storage = torch.rand(rows, vocab_size * 2, device="cuda") > 0.5
    allowed = allowed_storage[:, ::2]
    emitted = torch.tensor([0, 1, 2], device="cuda")
    metadata_bool = torch.ones(rows, dtype=torch.bool, device="cuda")
    metadata_int = torch.zeros(rows, dtype=torch.long, device="cuda")
    temperatures = torch.ones(rows, dtype=torch.float32, device="cuda")
    kwargs = {
        "max_top_k": 8,
        "eos_token_ids": None,
        "sanitize_fill": _SANITIZE_FILL,
        "inf_pressure": INF_PRESSURE_SENTINEL,
    }

    actual = compute_constraint_metrics(
        pre, allowed, temperatures, emitted, metadata_bool, metadata_int, **kwargs
    )
    expected = compute_constraint_metrics(
        pre,
        allowed.contiguous(),
        temperatures,
        emitted,
        metadata_bool,
        metadata_int,
        **kwargs,
    )
    torch.cuda.synchronize()
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=0, rtol=0)


def test_dense_rows_are_chunked_without_changing_results():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the constraint-metrics kernel")
    rows, vocab_size = 7, 65
    pre = torch.randn(rows, vocab_size, device="cuda")
    allowed = torch.rand(rows, vocab_size, device="cuda") > 0.4
    allowed[:, 0] = True
    emitted = torch.arange(rows, device="cuda")
    spec_sources = torch.arange(rows, device="cuda") % 3
    row_enabled = torch.tensor(
        [True, False, True, True, False, True, False], device="cuda"
    )
    kwargs = {
        "top_ks": [0, 1, 8, 3, 2, 8, 1],
        "grammar_active": [True] * rows,
        "output_indices": list(range(rows)),
        "eos_token_ids": [0],
        "spec_sources": spec_sources,
        "row_enabled": row_enabled,
    }
    expected = _compute_record_tensor_batch_chunked(
        pre,
        allowed,
        emitted,
        max_vocab_elements=rows * vocab_size,
        **kwargs,
    )
    assert expected.floats.shape[1] == (
        layout.KERNEL_ROW_FLOAT_COUNT + layout.TOP_K_FLOAT_GROUP_COUNT * 8
    )
    with mock.patch(
        "sglang_constraint_metrics.records._compute_record_tensor_batch",
        wraps=_compute_record_tensor_batch,
    ) as compute:
        actual = _compute_record_tensor_batch_chunked(
            pre,
            allowed,
            emitted,
            max_vocab_elements=2 * vocab_size,
            **kwargs,
        )
    torch.cuda.synchronize()

    assert compute.call_count == 4
    torch.testing.assert_close(
        actual.floats[row_enabled], expected.floats[row_enabled], atol=0, rtol=0
    )
    torch.testing.assert_close(
        actual.ints[row_enabled], expected.ints[row_enabled], atol=0, rtol=0
    )
    assert actual.top_ks == expected.top_ks
    assert actual.output_indices == expected.output_indices


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
