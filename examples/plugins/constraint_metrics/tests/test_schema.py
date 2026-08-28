import json
import sys
from dataclasses import dataclass

import pytest
import torch

from sglang_constraint_metrics import layout
from sglang_constraint_metrics.config import ConstraintMetricsConfig
from sglang_constraint_metrics.observer import (
    ConstraintMetricsResponse,
)
from sglang_constraint_metrics.records import (
    ConstraintMetricsTensorBatch,
    _compute_record_tensor_batch_chunked,
    packed_kernel_row_to_record,
    top_k_compute_bucket,
    uniform_top_k_ranges,
)
from sglang_constraint_metrics.schema import (
    CONSTRAINT_METRIC_FLAG_ORDER,
    INF_PRESSURE_SENTINEL,
    ConstraintMetricsSummaryAccumulator,
    extend_kernel_row_segments,
)


@dataclass
class _MetricRow:
    output_index: int
    top_k: int
    grammar_active: bool
    floats: list
    kernel_ints: list
    masked_token_ids: list
    masked_probabilities: list
    legal_token_ids: list
    conditioned_legal_probabilities: list


def _metric_row(
    output_index: int,
    *,
    pressure: float = 0.0,
    entropy: float = 0.0,
    logit_gap: float = 0.0,
    effective_legal_token_count: int = 1,
    blocked_token_count: int = 1,
    grammar_active: bool = True,
    argmax_masked: bool = False,
    empty_support: bool = False,
    eos_only_support: bool = False,
    has_non_finite_logits: bool = False,
    spec_source: int = 0,
    top_k: int = 0,
) -> _MetricRow:
    floats = [0.0] * len(layout.FLOAT_FIELDS)
    floats[layout.FLOAT_PRESSURE] = pressure
    floats[layout.FLOAT_ENTROPY] = entropy
    floats[layout.FLOAT_UNCONSTRAINED_TOP1_PROB] = 0.25
    floats[layout.FLOAT_TOP1_LOGIT_GAP] = logit_gap
    floats[layout.FLOAT_EMITTED_MODEL_LOG_PROB] = -1.5
    kernel_ints = [0] * layout.KERNEL_ROW_INT_COUNT
    kernel_ints[layout.KERNEL_INT_TOKEN_ID] = 100 + output_index
    kernel_ints[layout.KERNEL_INT_EFFECTIVE_LEGAL_TOKEN_COUNT] = (
        effective_legal_token_count
    )
    kernel_ints[layout.KERNEL_INT_BLOCKED_TOKEN_COUNT] = blocked_token_count
    kernel_ints[layout.KERNEL_INT_UNCONSTRAINED_TOP1_TOKEN_ID] = 200 + output_index
    kernel_ints[layout.KERNEL_INT_SPEC_SOURCE] = spec_source
    kernel_ints[layout.KERNEL_INT_ARGMAX_MASKED] = argmax_masked
    kernel_ints[layout.KERNEL_INT_EMPTY_SUPPORT] = empty_support
    kernel_ints[layout.KERNEL_INT_EOS_ONLY_SUPPORT] = eos_only_support
    kernel_ints[layout.KERNEL_INT_HAS_NON_FINITE_LOGITS] = has_non_finite_logits
    kernel_ints[layout.KERNEL_INT_EMITTED_ALLOWED] = True
    candidate_ids = list(range(300, 300 + max(8, top_k)))
    candidate_probs = [0.125 * index for index in range(len(candidate_ids))]
    return _MetricRow(
        output_index=output_index,
        top_k=top_k,
        grammar_active=grammar_active,
        floats=floats,
        kernel_ints=kernel_ints,
        masked_token_ids=candidate_ids,
        masked_probabilities=candidate_probs,
        legal_token_ids=candidate_ids,
        conditioned_legal_probabilities=candidate_probs,
    )


def _tensor_batch(rows: list[_MetricRow]) -> ConstraintMetricsTensorBatch:
    max_top_k = max(row.top_k for row in rows)
    packed_floats = []
    packed_ints = []
    for row in rows:
        floats = row.floats
        packed_floats.append(
            floats
            + row.masked_probabilities[:max_top_k]
            + row.conditioned_legal_probabilities[:max_top_k]
        )
        packed_ints.append(
            [
                *row.kernel_ints,
                *row.masked_token_ids[:max_top_k],
                *row.legal_token_ids[:max_top_k],
            ]
        )
    return ConstraintMetricsTensorBatch(
        floats=torch.tensor(packed_floats),
        ints=torch.tensor(packed_ints, dtype=torch.int64),
        top_ks=[row.top_k for row in rows],
        grammar_active=[row.grammar_active for row in rows],
        output_indices=[row.output_index for row in rows],
        vocab_size=1024,
    )


def test_summary_matches_hand_calculated_oracle():
    rows = [
        _metric_row(
            0,
            pressure=0.2,
            entropy=0.6,
            logit_gap=0.4,
            effective_legal_token_count=1,
            blocked_token_count=0,
        ),
        _metric_row(
            1,
            pressure=1.5,
            entropy=0.4,
            logit_gap=0.8,
            effective_legal_token_count=4,
            blocked_token_count=2,
            argmax_masked=True,
            spec_source=1,
        ),
        _metric_row(
            2,
            pressure=9.0,
            entropy=0.0,
            logit_gap=0.2,
            effective_legal_token_count=0,
            argmax_masked=True,
            empty_support=True,
            eos_only_support=True,
            has_non_finite_logits=True,
        ),
        _metric_row(
            3,
            pressure=0.0,
            entropy=0.9,
            logit_gap=0.6,
            effective_legal_token_count=8,
            grammar_active=False,
        ),
    ]
    accumulator = ConstraintMetricsSummaryAccumulator(top_k_events=2)
    batch = _tensor_batch(rows)
    packed_rows = batch.to_packed_rows(
        list(range(len(rows))),
        output_indices=[row.output_index for row in rows],
    )
    extend_kernel_row_segments(
        [accumulator],
        packed_rows.floats,
        packed_rows.ints,
        packed_rows.grammar_active,
        packed_rows.output_indices,
        [len(rows)],
        [False],
    )
    summary = accumulator.to_summary()

    assert summary["output_token_count"] == 4
    assert summary["grammar_active_token_count"] == 3
    assert summary["finite_active_token_count"] == 2
    assert summary["cumulative_constraint_pressure_nats"] == pytest.approx(1.7)
    assert summary["mean_constraint_pressure_nats"] == pytest.approx(0.85)
    assert summary["max_constraint_pressure_nats"] == 1.5
    assert summary["max_pressure_output_index"] == 1
    assert summary["argmax_masked_count"] == 2
    assert summary["mean_top1_logit_gap_to_top2_nats"] == pytest.approx(0.5)
    assert summary["max_top1_logit_gap_to_top2_nats"] == pytest.approx(0.8)
    assert summary["high_pressure_low_entropy_count"] == 1
    assert summary["longest_high_pressure_run"] == 1
    assert summary["empty_support_count"] == 1
    assert summary["eos_only_support_count"] == 1
    assert summary["has_non_finite_count"] == 1
    assert summary["effective_legal_token_count_histogram"] == [
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert summary["top_pressure_events"] == [
        {
            "output_index": 1,
            "constraint_pressure_nats": 1.5,
            "effective_legal_token_count": 4,
            "emitted_token_id": 101,
            "unconstrained_top1_id": 201,
            "top1_logit_gap_to_top2_nats": pytest.approx(0.8),
            "argmax_masked": True,
            "flags": [],
            "spec_source": "accepted_draft",
        },
        {
            "output_index": 0,
            "constraint_pressure_nats": pytest.approx(0.2),
            "effective_legal_token_count": 1,
            "emitted_token_id": 100,
            "unconstrained_top1_id": 200,
            "top1_logit_gap_to_top2_nats": pytest.approx(0.4),
            "argmax_masked": False,
            "flags": ["all_allow"],
            "spec_source": None,
        },
    ]


def test_public_stream_record_schema():
    row = _metric_row(
        0,
        effective_legal_token_count=8,
        blocked_token_count=8,
        top_k=2,
    )
    packed = _tensor_batch([row]).to_packed_rows([0], output_indices=[0])
    record = packed_kernel_row_to_record(
        packed.floats[0],
        packed.ints[0],
        grammar_active=row.grammar_active,
        output_index=row.output_index,
    )

    assert record == {
        "output_index": 0,
        "token_id": 100,
        "grammar_active": True,
        "effective_legal_token_count": 8,
        "constraint_pressure_nats": 0.0,
        "argmax_masked": False,
        "unconstrained_top1": {
            "token_id": 200,
            "prob": 0.25,
            "logit_gap_to_top2_nats": 0.0,
        },
        "entropy_nats": 0.0,
        "emitted": {"allowed": True, "model_log_prob": -1.5},
        "flags": [],
    }


def test_summary_response_schema_and_counts():
    summary = ConstraintMetricsSummaryAccumulator().to_summary()
    summary["output_token_count"] = 1
    response = ConstraintMetricsResponse(
        config=ConstraintMetricsConfig("summary", 0),
        record_slices=[],
        validation_token_ids=[],
        input_token_count=2,
        output_token_count=1,
        summary=summary,
        captured_prefix_length=1,
    ).materialize_constraint_metrics()

    assert response == {
        "status": "complete",
        "mode": "summary",
        "warnings": [],
        "summary": summary,
        "input_token_count": 2,
        "output_token_count": 1,
        "tokens": [],
    }


def test_top_k_compute_widths_are_bounded_without_changing_requested_values():
    requested = [0, 1, 7, 8, 9, 16, 17, 32]
    assert [top_k_compute_bucket(value) for value in requested] == [
        0,
        8,
        8,
        8,
        16,
        16,
        32,
        32,
    ]
    assert uniform_top_k_ranges(requested) == [
        (0, 1, 0),
        (1, 4, 8),
        (4, 6, 16),
        (6, 8, 32),
    ]
    assert requested == [0, 1, 7, 8, 9, 16, 17, 32]

    with pytest.raises(ValueError, match="sorted by top_k"):
        uniform_top_k_ranges([8, 7])
    with pytest.raises(ValueError, match="exceeds maximum compute width 32"):
        top_k_compute_bucket(33)


def test_chunked_compute_rejects_bucket_narrower_than_requested_rows():
    with pytest.raises(
        ValueError, match="compute max_top_k 8 is smaller than requested row top_k 9"
    ):
        _compute_record_tensor_batch_chunked(
            torch.zeros(1, 4),
            torch.ones(1, 4, dtype=torch.bool),
            torch.zeros(1, dtype=torch.long),
            top_ks=[9],
            grammar_active=[True],
            output_indices=[0],
            max_top_k=8,
        )


def test_forensic_columns_preserve_schema_width_and_summary():
    rows = [
        _metric_row(
            0,
            pressure=0.2,
            effective_legal_token_count=8,
            blocked_token_count=8,
            top_k=1,
        ),
        _metric_row(
            1,
            pressure=1.5,
            entropy=0.4,
            effective_legal_token_count=8,
            blocked_token_count=8,
            argmax_masked=True,
            spec_source=layout.SPEC_SOURCE_ACCEPTED_DRAFT,
            top_k=8,
        ),
        _metric_row(
            2,
            pressure=9.0,
            effective_legal_token_count=0,
            blocked_token_count=8,
            empty_support=True,
            eos_only_support=True,
            has_non_finite_logits=True,
            top_k=0,
        ),
        _metric_row(
            3,
            grammar_active=False,
            effective_legal_token_count=8,
            blocked_token_count=0,
            top_k=0,
        ),
    ]
    batch = _tensor_batch(rows)
    column_summary = ConstraintMetricsSummaryAccumulator()
    columns = batch.to_columns(summary_accumulator=column_summary)

    assert set(columns) == {
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
        "top_masked",
        "top_legal_after_conditioning",
    }
    assert columns["count"] == 4
    assert columns["output_index"] == [0, 1, 2, 3]
    assert columns["token_id"] == [100, 101, 102, 103]
    assert columns["grammar_active"] == [True, True, True, False]
    assert columns["effective_legal_token_count"] == [8, 8, 0, 8]
    assert columns["constraint_pressure_nats"] == pytest.approx(
        [0.2, 1.5, INF_PRESSURE_SENTINEL, 0.0]
    )
    assert columns["argmax_masked"] == [False, True, False, False]
    assert columns["entropy_nats"] == pytest.approx([0.0, 0.4, 0.0, 0.0])
    assert set(columns["unconstrained_top1"]) == {
        "token_id",
        "prob",
        "logit_gap_to_top2_nats",
    }
    assert columns["unconstrained_top1"]["token_id"] == [200, 201, 202, 203]
    assert columns["unconstrained_top1"]["prob"] == pytest.approx([0.25] * 4)
    assert columns["unconstrained_top1"]["logit_gap_to_top2_nats"] == pytest.approx(
        [0.0] * 4
    )
    assert set(columns["emitted"]) == {"allowed", "model_log_prob", "rank"}
    assert columns["emitted"]["allowed"] == [True] * 4
    assert columns["emitted"]["model_log_prob"] == pytest.approx([-1.5] * 4)
    assert columns["emitted"]["rank"] == [0] * 4
    assert tuple(columns["flags"]) == CONSTRAINT_METRIC_FLAG_ORDER
    assert all(
        isinstance(value, bool)
        for values in columns["flags"].values()
        for value in values
    )
    assert columns["flags"] == {
        "grammar_inactive": [False, False, False, True],
        "empty_support": [False, False, True, False],
        "all_allow": [False, False, False, False],
        "eos_only_support": [False, False, True, False],
        "has_non_finite_logits": [False, False, True, False],
    }
    assert columns["top_masked"]["width"] == 8
    assert columns["top_masked"]["count"] == [1, 8, 0, 0]
    assert columns["top_legal_after_conditioning"]["width"] == 8
    assert columns["top_legal_after_conditioning"]["count"] == [1, 8, 0, 0]
    assert columns["spec_source"] == [None, "accepted_draft", None, None]

    assert column_summary.to_summary()["output_token_count"] == len(rows)
    assert column_summary.to_summary()["max_constraint_pressure_nats"] == 1.5
    assert json.loads(json.dumps(columns, allow_nan=False)) == columns

    sliced_columns = batch.to_columns(
        row_start=1,
        row_count=2,
        output_index_start=10,
    )
    assert sliced_columns["output_index"] == [10, 11]
    assert sliced_columns["token_id"] == [101, 102]


def test_forensic_columns_preserve_physical_width_for_empty_slices():
    k0_batch = _tensor_batch([_metric_row(0, top_k=0)])
    k0_columns = k0_batch.to_columns()
    assert k0_columns["count"] == 1
    assert k0_columns["top_masked"] == {
        "width": 0,
        "count": [0],
        "token_id": [],
        "prob": [],
    }

    k8_batch = _tensor_batch([_metric_row(0, top_k=8)])
    empty_k8 = k8_batch.to_columns(row_start=1, row_count=0)
    assert empty_k8 == ConstraintMetricsTensorBatch.empty_columns(width=8)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
