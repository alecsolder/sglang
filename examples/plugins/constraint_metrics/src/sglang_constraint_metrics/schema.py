"""Response shaping for constraint metrics.

The runtime transports packed tensors. This module is the only place that
expands them into the public JSON contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from sglang_constraint_metrics import layout

INF_PRESSURE_SENTINEL = 1.0e30
SEVERE_PRESSURE_THRESHOLD_NATS = math.log(2.0)
HIGH_PRESSURE_NATS = 1.0
LOW_ENTROPY_NATS = 0.5
DEFAULT_SUMMARY_TOP_K_EVENTS = 8
CONSTRAINT_METRIC_FLAG_ORDER = (
    "grammar_inactive",
    "empty_support",
    "all_allow",
    "eos_only_support",
    "has_non_finite_logits",
)
EFFECTIVE_LEGAL_TOKEN_COUNT_HISTOGRAM_LABELS = (
    "0",
    "1",
    "2-3",
    "4-7",
    "8-15",
    "16-31",
    "32-63",
    "64-127",
    "128-255",
    "256-511",
    "512-1023",
    ">=1024",
)

STATUS_COMPLETE = "complete"
STATUS_PARTIALLY_DROPPED = "partially_dropped"


@dataclass
class ConstraintMetricsSummaryAccumulator:
    """Fixed-size request summary updated from fused metric columns."""

    top_k_events: int = DEFAULT_SUMMARY_TOP_K_EVENTS
    output_token_count: int = 0
    grammar_active_token_count: int = 0
    finite_active_token_count: int = 0
    pressure_sum: float = 0.0
    max_pressure: float = 0.0
    max_pressure_output_index: int = -1
    argmax_masked_count: int = 0
    logit_gap_sum: float = 0.0
    max_logit_gap: float = 0.0
    high_pressure_low_entropy_count: int = 0
    current_high_pressure_run: int = 0
    longest_high_pressure_run: int = 0
    empty_support_count: int = 0
    eos_only_support_count: int = 0
    has_non_finite_count: int = 0
    effective_legal_token_count_histogram: List[int] = field(
        default_factory=lambda: [0] * len(EFFECTIVE_LEGAL_TOKEN_COUNT_HISTOGRAM_LABELS)
    )
    top_pressure_events: List[Dict[str, Any]] = field(default_factory=list)
    _top_pressure_worst_index: int = field(default=-1, init=False, repr=False)
    _top_pressure_worst_key: Optional[tuple[float, int]] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.top_k_events < 0:
            raise ValueError("summary top_k_events must be non-negative")

    def _consider_top_pressure_event(
        self,
        *,
        output_index: int,
        token_id: int,
        effective_legal_token_count: int,
        pressure: float,
        unconstrained_top1_token_id: int,
        top1_logit_gap: float,
        argmax_masked: bool,
        grammar_active: bool,
        empty_support: bool,
        all_allow: bool,
        eos_only_support: bool,
        has_non_finite_logits: bool,
        flags: Optional[Sequence[str]],
        spec_source: Optional[str],
    ) -> None:
        if not self.top_k_events:
            return
        candidate_key = (-pressure, output_index)
        replace_index = None
        if len(self.top_pressure_events) >= self.top_k_events:
            if (
                self._top_pressure_worst_key is None
                or self._top_pressure_worst_index < 0
            ):
                self._refresh_top_pressure_worst()
            assert self._top_pressure_worst_key is not None
            if candidate_key >= self._top_pressure_worst_key:
                return
            replace_index = self._top_pressure_worst_index

        if flags is None:
            flag_values = (
                not grammar_active,
                empty_support,
                all_allow,
                eos_only_support,
                has_non_finite_logits,
            )
            flags = [
                name
                for name, enabled in zip(
                    CONSTRAINT_METRIC_FLAG_ORDER, flag_values, strict=True
                )
                if enabled
            ]

        event = {
            "output_index": output_index,
            "constraint_pressure_nats": pressure,
            "effective_legal_token_count": effective_legal_token_count,
            "emitted_token_id": token_id,
            "unconstrained_top1_id": unconstrained_top1_token_id,
            "top1_logit_gap_to_top2_nats": top1_logit_gap,
            "argmax_masked": argmax_masked,
            "flags": list(flags),
            "spec_source": spec_source,
        }
        if replace_index is None:
            self.top_pressure_events.append(event)
            if len(self.top_pressure_events) < self.top_k_events:
                return
        else:
            self.top_pressure_events[replace_index] = event
        self._refresh_top_pressure_worst()

    def _refresh_top_pressure_worst(self) -> None:
        worst_index = -1
        worst_key: Optional[tuple[float, int]] = None
        for index, event in enumerate(self.top_pressure_events):
            event_key = (
                -float(event["constraint_pressure_nats"]),
                int(event["output_index"]),
            )
            if worst_key is None or event_key > worst_key:
                worst_index = index
                worst_key = event_key
        self._top_pressure_worst_index = worst_index
        self._top_pressure_worst_key = worst_key

    def to_summary(self) -> Dict[str, Any]:
        n = self.output_token_count
        n_finite = self.finite_active_token_count
        return {
            "output_token_count": n,
            "grammar_active_token_count": self.grammar_active_token_count,
            "finite_active_token_count": n_finite,
            "cumulative_constraint_pressure_nats": self.pressure_sum,
            "mean_constraint_pressure_nats": (
                self.pressure_sum / n_finite if n_finite else 0.0
            ),
            "max_constraint_pressure_nats": self.max_pressure,
            "max_pressure_output_index": self.max_pressure_output_index,
            "argmax_masked_count": self.argmax_masked_count,
            "mean_top1_logit_gap_to_top2_nats": (self.logit_gap_sum / n if n else 0.0),
            "max_top1_logit_gap_to_top2_nats": self.max_logit_gap,
            "high_pressure_low_entropy_count": self.high_pressure_low_entropy_count,
            "longest_high_pressure_run": self.longest_high_pressure_run,
            "empty_support_count": self.empty_support_count,
            "eos_only_support_count": self.eos_only_support_count,
            "has_non_finite_count": self.has_non_finite_count,
            "effective_legal_token_count_histogram": list(
                self.effective_legal_token_count_histogram
            ),
            "effective_legal_token_count_histogram_labels": list(
                EFFECTIVE_LEGAL_TOKEN_COUNT_HISTOGRAM_LABELS
            ),
            "top_pressure_events": sorted(
                self.top_pressure_events,
                key=lambda event: (
                    -float(event["constraint_pressure_nats"]),
                    int(event["output_index"]),
                ),
            ),
        }


def extend_kernel_row_segments(
    accumulators: Sequence[ConstraintMetricsSummaryAccumulator],
    floats: Sequence[Sequence[Any]],
    ints: Sequence[Sequence[Any]],
    grammar_active: Sequence[bool],
    output_indices: Sequence[int],
    row_counts: Sequence[int],
    collect_triggers: Sequence[bool],
) -> List[List[bool]]:
    """Update request summaries from contiguous fused-kernel row segments."""

    segment_count = len(accumulators)
    if len(row_counts) != segment_count or len(collect_triggers) != segment_count:
        raise ValueError("constraint metric summary segments are misaligned")

    counts = np.asarray(row_counts, dtype=np.int64)
    if np.any(counts < 0):
        raise ValueError("constraint metric summary segment size is negative")
    row_count = len(floats)
    if (
        len(ints) != row_count
        or len(grammar_active) != row_count
        or len(output_indices) != row_count
        or int(counts.sum()) != row_count
    ):
        raise ValueError("constraint metric packed rows are misaligned")
    if row_count == 0:
        return [[] for _ in accumulators]

    float_values = np.asarray(floats, dtype=np.float64)
    int_values = np.asarray(ints, dtype=np.int64)
    active = np.asarray(grammar_active, dtype=np.bool_)
    output_index_values = np.asarray(output_indices, dtype=np.int64)
    if (
        float_values.ndim != 2
        or float_values.shape[1] < layout.KERNEL_ROW_FLOAT_COUNT
        or int_values.ndim != 2
        or int_values.shape[1] < layout.KERNEL_ROW_INT_COUNT
    ):
        raise ValueError("constraint metric packed row columns are truncated")

    owners = np.repeat(np.arange(segment_count, dtype=np.int64), counts)
    starts = np.cumsum(counts) - counts
    nonempty = counts > 0
    nonempty_starts = starts[nonempty]
    entropy = float_values[:, layout.FLOAT_ENTROPY]
    top1_logit_gap = float_values[:, layout.FLOAT_TOP1_LOGIT_GAP]
    empty_support = int_values[:, layout.KERNEL_INT_EMPTY_SUPPORT].astype(bool)
    argmax_masked = int_values[:, layout.KERNEL_INT_ARGMAX_MASKED].astype(bool)
    eos_only_support = int_values[:, layout.KERNEL_INT_EOS_ONLY_SUPPORT].astype(bool)
    has_non_finite_logits = int_values[
        :, layout.KERNEL_INT_HAS_NON_FINITE_LOGITS
    ].astype(bool)
    pressure = float_values[:, layout.FLOAT_PRESSURE].copy()
    pressure[empty_support] = INF_PRESSURE_SENTINEL
    finite_active = active & ~empty_support
    pressure_triggered = finite_active & (pressure >= SEVERE_PRESSURE_THRESHOLD_NATS)
    high_pressure = ~empty_support & (pressure >= HIGH_PRESSURE_NATS)

    def segment_max(values: np.ndarray, default: float = 0.0) -> np.ndarray:
        result = np.full(segment_count, default, dtype=np.float64)
        result[nonempty] = np.maximum.reduceat(values, nonempty_starts)
        return result

    aggregate_values = np.column_stack(
        (
            np.where(finite_active, pressure, 0.0),
            top1_logit_gap,
            active,
            finite_active,
            argmax_masked,
            empty_support,
            eos_only_support,
            has_non_finite_logits,
            high_pressure & (entropy <= LOW_ENTROPY_NATS),
        )
    )
    aggregate_width = aggregate_values.shape[1]
    aggregate_keys = owners[:, None] * aggregate_width + np.arange(
        aggregate_width, dtype=np.int64
    )
    aggregates = np.bincount(
        aggregate_keys.ravel(),
        weights=aggregate_values.ravel(),
        minlength=segment_count * aggregate_width,
    ).reshape(segment_count, aggregate_width)
    (
        pressure_sums,
        logit_gap_sums,
        grammar_active_counts,
        finite_active_counts,
        argmax_masked_counts,
        empty_support_counts,
        eos_only_support_counts,
        non_finite_counts,
        high_pressure_low_entropy_counts,
    ) = aggregates.T
    max_logit_gap = segment_max(top1_logit_gap)

    finite_pressure = np.where(finite_active, pressure, -np.inf)
    max_pressure = segment_max(finite_pressure, -np.inf)
    first_max_row = np.full(segment_count, row_count, dtype=np.int64)
    max_candidates = finite_active & (pressure == max_pressure[owners])
    np.minimum.at(
        first_max_row,
        owners[max_candidates],
        np.flatnonzero(max_candidates),
    )

    legal_token_counts = int_values[:, layout.KERNEL_INT_EFFECTIVE_LEGAL_TOKEN_COUNT]
    histogram_bins = np.zeros(row_count, dtype=np.int64)
    positive_legal_counts = legal_token_counts > 0
    histogram_bins[positive_legal_counts] = np.minimum(
        np.floor(np.log2(legal_token_counts[positive_legal_counts])).astype(np.int64)
        + 1,
        len(EFFECTIVE_LEGAL_TOKEN_COUNT_HISTOGRAM_LABELS) - 1,
    )
    histogram = np.bincount(
        owners * len(EFFECTIVE_LEGAL_TOKEN_COUNT_HISTOGRAM_LABELS) + histogram_bins,
        minlength=segment_count * len(EFFECTIVE_LEGAL_TOKEN_COUNT_HISTOGRAM_LABELS),
    ).reshape(segment_count, -1)

    emitted_allowed = int_values[:, layout.KERNEL_INT_EMITTED_ALLOWED].astype(bool)
    triggered = active & (
        pressure_triggered
        | argmax_masked
        | empty_support
        | eos_only_support
        | has_non_finite_logits
        | ~emitted_allowed
    )
    triggered_by_segment = []
    for segment_index, should_collect in enumerate(collect_triggers):
        start = int(starts[segment_index])
        stop = start + int(counts[segment_index])
        triggered_by_segment.append(
            triggered[start:stop].tolist()
            if should_collect
            else [False] * (stop - start)
        )

    all_allow = (
        active
        & ~empty_support
        & (int_values[:, layout.KERNEL_INT_BLOCKED_TOKEN_COUNT] == 0)
    )
    token_ids = int_values[:, layout.KERNEL_INT_TOKEN_ID]
    top1_token_ids = int_values[:, layout.KERNEL_INT_UNCONSTRAINED_TOP1_TOKEN_ID]
    spec_sources = int_values[:, layout.KERNEL_INT_SPEC_SOURCE]
    for segment_index, accumulator in enumerate(accumulators):
        start = int(starts[segment_index])
        stop = start + int(counts[segment_index])
        accumulator.output_token_count += stop - start
        accumulator.grammar_active_token_count += int(
            grammar_active_counts[segment_index]
        )
        accumulator.finite_active_token_count += int(
            finite_active_counts[segment_index]
        )
        accumulator.pressure_sum += float(pressure_sums[segment_index])
        accumulator.argmax_masked_count += int(argmax_masked_counts[segment_index])
        accumulator.logit_gap_sum += float(logit_gap_sums[segment_index])
        accumulator.max_logit_gap = max(
            accumulator.max_logit_gap, float(max_logit_gap[segment_index])
        )
        accumulator.high_pressure_low_entropy_count += int(
            high_pressure_low_entropy_counts[segment_index]
        )
        accumulator.empty_support_count += int(empty_support_counts[segment_index])
        accumulator.eos_only_support_count += int(
            eos_only_support_counts[segment_index]
        )
        accumulator.has_non_finite_count += int(non_finite_counts[segment_index])
        for bin_index, value in enumerate(histogram[segment_index]):
            accumulator.effective_legal_token_count_histogram[bin_index] += int(value)

        segment_max_pressure = float(max_pressure[segment_index])
        if segment_max_pressure != -np.inf and (
            accumulator.max_pressure_output_index == -1
            or segment_max_pressure > accumulator.max_pressure
        ):
            max_row = int(first_max_row[segment_index])
            accumulator.max_pressure = segment_max_pressure
            accumulator.max_pressure_output_index = int(output_index_values[max_row])

    current_runs = [
        accumulator.current_high_pressure_run for accumulator in accumulators
    ]
    longest_runs = [
        accumulator.longest_high_pressure_run for accumulator in accumulators
    ]
    accepts_all_top_events = np.zeros(segment_count, dtype=np.bool_)
    top_events_enabled = np.zeros(segment_count, dtype=np.bool_)
    worst_top_pressure = np.zeros(segment_count, dtype=np.float64)
    worst_top_output_index = np.zeros(segment_count, dtype=np.int64)
    for segment_index, accumulator in enumerate(accumulators):
        if not accumulator.top_k_events:
            continue
        top_events_enabled[segment_index] = True
        if len(accumulator.top_pressure_events) < accumulator.top_k_events:
            accepts_all_top_events[segment_index] = True
            continue
        if accumulator._top_pressure_worst_key is None:
            accumulator._refresh_top_pressure_worst()
        assert accumulator._top_pressure_worst_key is not None
        worst_key = accumulator._top_pressure_worst_key
        worst_top_pressure[segment_index] = -worst_key[0]
        worst_top_output_index[segment_index] = worst_key[1]
    eligible_top_event = (
        finite_active
        & top_events_enabled[owners]
        & (
            accepts_all_top_events[owners]
            | (pressure > worst_top_pressure[owners])
            | (
                (pressure == worst_top_pressure[owners])
                & (output_index_values < worst_top_output_index[owners])
            )
        )
    )
    for row_index, segment_index_value in enumerate(owners):
        segment_index = int(segment_index_value)
        accumulator = accumulators[segment_index]
        if high_pressure[row_index]:
            current_runs[segment_index] += 1
            longest_runs[segment_index] = max(
                longest_runs[segment_index], current_runs[segment_index]
            )
        else:
            current_runs[segment_index] = 0
        if eligible_top_event[row_index]:
            accumulator._consider_top_pressure_event(
                output_index=int(output_index_values[row_index]),
                token_id=int(token_ids[row_index]),
                effective_legal_token_count=int(legal_token_counts[row_index]),
                pressure=float(pressure[row_index]),
                unconstrained_top1_token_id=int(top1_token_ids[row_index]),
                top1_logit_gap=float(top1_logit_gap[row_index]),
                argmax_masked=bool(argmax_masked[row_index]),
                grammar_active=bool(active[row_index]),
                empty_support=bool(empty_support[row_index]),
                all_allow=bool(all_allow[row_index]),
                eos_only_support=bool(eos_only_support[row_index]),
                has_non_finite_logits=bool(has_non_finite_logits[row_index]),
                flags=None,
                spec_source=layout.SPEC_SOURCE_NAMES.get(int(spec_sources[row_index])),
            )

    for segment_index, accumulator in enumerate(accumulators):
        accumulator.current_high_pressure_run = current_runs[segment_index]
        accumulator.longest_high_pressure_run = longest_runs[segment_index]

    return triggered_by_segment
