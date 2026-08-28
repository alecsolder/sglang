"""Strict, UI-independent parsing of constraint-metrics responses."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

HIGH_PRESSURE_NATS = 1.0
INF_PRESSURE_SENTINEL = 1.0e30
_MODES = {"summary", "triggered", "forensic"}
_STATUSES = {"complete", "partially_dropped"}
_FLAG_NAMES = (
    "grammar_inactive",
    "empty_support",
    "all_allow",
    "eos_only_support",
    "has_non_finite_logits",
)


class SchemaError(ValueError):
    """The response does not match the current public metrics schema."""


def terminal_safe_text(value: str, *, multiline: bool = False) -> str:
    """Make untrusted response text inert before writing it to a terminal."""

    result = []
    for character in value:
        codepoint = ord(character)
        if character == "\n" and multiline:
            result.append(character)
        elif character == "\n":
            result.append("\\n")
        elif character == "\r":
            result.append("\\r")
        elif character == "\t":
            result.append("\\t")
        elif codepoint < 0x20 or 0x7F <= codepoint < 0xA0:
            result.append(f"\\x{codepoint:02x}")
        elif 0xD800 <= codepoint <= 0xDFFF:
            result.append(f"\\u{codepoint:04x}")
        else:
            result.append(character)
    return "".join(result)


@dataclass(frozen=True)
class Candidate:
    token_id: int
    model_probability: float
    conditioned_probability: float | None = None


@dataclass(frozen=True)
class SummaryEvent:
    output_index: int
    constraint_pressure_nats: float
    effective_legal_token_count: int
    emitted_token_id: int
    unconstrained_top1_id: int
    top1_logit_gap_to_top2_nats: float
    argmax_masked: bool
    flags: tuple[str, ...]
    spec_source: str | None


@dataclass(frozen=True)
class TokenMetric:
    output_index: int
    token_id: int
    grammar_active: bool
    effective_legal_token_count: int
    constraint_pressure_nats: float
    argmax_masked: bool
    unconstrained_top1_token_id: int
    unconstrained_top1_probability: float
    top1_logit_gap_to_top2_nats: float
    entropy_nats: float
    emitted_allowed: bool
    emitted_model_log_probability: float
    emitted_rank: int | None
    flags: tuple[str, ...]
    spec_source: str | None
    top_masked: tuple[Candidate, ...]
    top_legal_after_conditioning: tuple[Candidate, ...]

    @property
    def legal_mass(self) -> float:
        if self.constraint_pressure_nats >= INF_PRESSURE_SENTINEL:
            return 0.0
        return math.exp(-self.constraint_pressure_nats)

    @property
    def masked_mass(self) -> float:
        if self.constraint_pressure_nats >= INF_PRESSURE_SENTINEL:
            return 1.0
        return -math.expm1(-self.constraint_pressure_nats)

    @property
    def emitted_model_probability(self) -> float:
        return math.exp(self.emitted_model_log_probability)

    @property
    def emitted_conditioned_probability(self) -> float | None:
        if (
            not self.emitted_allowed
            or self.constraint_pressure_nats >= INF_PRESSURE_SENTINEL
        ):
            return None
        return math.exp(
            min(
                0.0,
                self.emitted_model_log_probability + self.constraint_pressure_nats,
            )
        )

    @property
    def high_pressure(self) -> bool:
        return (
            self.constraint_pressure_nats >= HIGH_PRESSURE_NATS
            or "empty_support" in self.flags
        )


@dataclass(frozen=True)
class Summary:
    output_token_count: int
    grammar_active_token_count: int
    finite_active_token_count: int
    cumulative_constraint_pressure_nats: float
    mean_constraint_pressure_nats: float
    max_constraint_pressure_nats: float
    max_pressure_output_index: int
    argmax_masked_count: int
    mean_top1_logit_gap_to_top2_nats: float
    max_top1_logit_gap_to_top2_nats: float
    high_pressure_low_entropy_count: int
    longest_high_pressure_run: int
    empty_support_count: int
    eos_only_support_count: int
    has_non_finite_count: int
    effective_legal_token_count_histogram: tuple[tuple[str, int], ...]
    top_pressure_events: tuple[SummaryEvent, ...]


@dataclass(frozen=True)
class ConstraintMetrics:
    status: str
    mode: str
    warnings: tuple[str, ...]
    input_token_count: int
    output_token_count: int
    summary: Summary
    tokens: tuple[TokenMetric, ...]
    token_labels: Mapping[int, str]
    triggered_token_count: int | None = None
    output_text: str | None = None
    source: str | None = None
    endpoint: str | None = None

    @property
    def initial_token_position(self) -> int | None:
        if not self.tokens:
            return None
        for position, token in enumerate(self.tokens):
            if "empty_support" in token.flags:
                return position
        peak = self.summary.max_pressure_output_index
        for position, token in enumerate(self.tokens):
            if token.output_index == peak:
                return position
        return max(
            range(len(self.tokens)),
            key=lambda position: self.tokens[position].constraint_pressure_nats,
        )

    def token_label(self, token_id: int) -> str:
        return self.token_labels.get(token_id, f"<{token_id}>")

    def token_ids(self) -> set[int]:
        ids: set[int] = set()
        for token in self.tokens:
            ids.add(token.token_id)
            ids.add(token.unconstrained_top1_token_id)
            ids.update(candidate.token_id for candidate in token.top_masked)
            ids.update(
                candidate.token_id for candidate in token.top_legal_after_conditioning
            )
        for event in self.summary.top_pressure_events:
            ids.add(event.emitted_token_id)
            ids.add(event.unconstrained_top1_id)
        return {token_id for token_id in ids if token_id >= 0}

    def with_token_labels(self, labels: Mapping[int, str]) -> ConstraintMetrics:
        return ConstraintMetrics(
            status=self.status,
            mode=self.mode,
            warnings=self.warnings,
            input_token_count=self.input_token_count,
            output_token_count=self.output_token_count,
            summary=self.summary,
            tokens=self.tokens,
            token_labels=dict(labels),
            triggered_token_count=self.triggered_token_count,
            output_text=self.output_text,
            source=self.source,
            endpoint=self.endpoint,
        )


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise SchemaError(f"{path} must be an object")
    return value


def _list(value: Any, path: str, *, length: int | None = None) -> list[Any]:
    if not isinstance(value, list):
        raise SchemaError(f"{path} must be an array")
    if length is not None and len(value) != length:
        raise SchemaError(f"{path} has {len(value)} values; expected {length}")
    return value


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{path} must be an integer")
    return value


def _nonnegative_integer(value: Any, path: str) -> int:
    result = _integer(value, path)
    if result < 0:
        raise SchemaError(f"{path} must be non-negative")
    return result


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{path} must be a number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise SchemaError(f"{path} must be a finite number") from error
    if not math.isfinite(result):
        raise SchemaError(f"{path} must be finite")
    return result


def _nonnegative_number(value: Any, path: str) -> float:
    result = _number(value, path)
    if result < 0.0:
        raise SchemaError(f"{path} must be non-negative")
    return result


def _probability(value: Any, path: str) -> float:
    result = _number(value, path)
    if result < 0.0 or result > 1.0:
        raise SchemaError(f"{path} must be between 0 and 1")
    return result


def _log_probability(value: Any, path: str) -> float:
    result = _number(value, path)
    if result > 0.0:
        raise SchemaError(f"{path} must be non-positive")
    return result


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaError(f"{path} must be a boolean")
    return value


def _column(obj: Mapping[str, Any], key: str, path: str, count: int) -> list[Any]:
    if key not in obj:
        raise SchemaError(f"{path}.{key} is missing")
    return _list(obj[key], f"{path}.{key}", length=count)


def _locate_metrics(obj: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if isinstance(obj, list):
        if len(obj) != 1:
            raise SchemaError(
                "the viewer accepts one response at a time; received "
                f"{len(obj)} responses"
            )
        obj = obj[0]
    response = _object(obj, "response")
    if {"status", "mode", "summary"}.issubset(response):
        return response, {}
    direct = response.get("constraint_metrics")
    if isinstance(direct, dict):
        return direct, response
    meta_info = response.get("meta_info")
    if not isinstance(meta_info, dict):
        raise SchemaError(
            "response.meta_info is missing; expected "
            "response.meta_info.constraint_metrics"
        )
    metrics = meta_info.get("constraint_metrics")
    if not isinstance(metrics, dict):
        raise SchemaError(
            "response.meta_info.constraint_metrics is missing; enable the "
            "constraint_metrics plugin and opt the request in"
        )
    return metrics, response


def _parse_summary(value: Any) -> Summary:
    obj = _object(value, "constraint_metrics.summary")
    prefix = "constraint_metrics.summary"

    def integer(key: str) -> int:
        if key not in obj:
            raise SchemaError(f"{prefix}.{key} is missing")
        return _integer(obj[key], f"{prefix}.{key}")

    def nonnegative_integer(key: str) -> int:
        result = integer(key)
        if result < 0:
            raise SchemaError(f"{prefix}.{key} must be non-negative")
        return result

    def nonnegative_number(key: str) -> float:
        if key not in obj:
            raise SchemaError(f"{prefix}.{key} is missing")
        return _nonnegative_number(obj[key], f"{prefix}.{key}")

    labels = _list(
        obj.get("effective_legal_token_count_histogram_labels"),
        f"{prefix}.effective_legal_token_count_histogram_labels",
    )
    counts = _list(
        obj.get("effective_legal_token_count_histogram"),
        f"{prefix}.effective_legal_token_count_histogram",
        length=len(labels),
    )
    histogram = []
    for index, (label, count) in enumerate(zip(labels, counts, strict=True)):
        if not isinstance(label, str):
            raise SchemaError(
                f"{prefix}.effective_legal_token_count_histogram_labels"
                f"[{index}] must be a string"
            )
        histogram.append(
            (
                label,
                _nonnegative_integer(
                    count,
                    f"{prefix}.effective_legal_token_count_histogram[{index}]",
                ),
            )
        )
    raw_events = _list(obj.get("top_pressure_events"), f"{prefix}.top_pressure_events")
    events = []
    for index, event_value in enumerate(raw_events):
        event = _object(event_value, f"{prefix}.top_pressure_events[{index}]")
        path = f"{prefix}.top_pressure_events[{index}]"
        raw_flags = _list(event.get("flags"), f"{path}.flags")
        if not all(isinstance(flag, str) for flag in raw_flags):
            raise SchemaError(f"{path}.flags must contain only strings")
        source = event.get("spec_source")
        if source is not None and not isinstance(source, str):
            raise SchemaError(f"{path}.spec_source must be a string or null")
        events.append(
            SummaryEvent(
                output_index=_nonnegative_integer(
                    event.get("output_index"), f"{path}.output_index"
                ),
                constraint_pressure_nats=_nonnegative_number(
                    event.get("constraint_pressure_nats"),
                    f"{path}.constraint_pressure_nats",
                ),
                effective_legal_token_count=_nonnegative_integer(
                    event.get("effective_legal_token_count"),
                    f"{path}.effective_legal_token_count",
                ),
                emitted_token_id=_integer(
                    event.get("emitted_token_id"), f"{path}.emitted_token_id"
                ),
                unconstrained_top1_id=_integer(
                    event.get("unconstrained_top1_id"),
                    f"{path}.unconstrained_top1_id",
                ),
                top1_logit_gap_to_top2_nats=_nonnegative_number(
                    event.get("top1_logit_gap_to_top2_nats"),
                    f"{path}.top1_logit_gap_to_top2_nats",
                ),
                argmax_masked=_boolean(
                    event.get("argmax_masked"), f"{path}.argmax_masked"
                ),
                flags=tuple(raw_flags),
                spec_source=source,
            )
        )
    return Summary(
        output_token_count=nonnegative_integer("output_token_count"),
        grammar_active_token_count=nonnegative_integer("grammar_active_token_count"),
        finite_active_token_count=nonnegative_integer("finite_active_token_count"),
        cumulative_constraint_pressure_nats=nonnegative_number(
            "cumulative_constraint_pressure_nats"
        ),
        mean_constraint_pressure_nats=nonnegative_number(
            "mean_constraint_pressure_nats"
        ),
        max_constraint_pressure_nats=nonnegative_number("max_constraint_pressure_nats"),
        max_pressure_output_index=integer("max_pressure_output_index"),
        argmax_masked_count=nonnegative_integer("argmax_masked_count"),
        mean_top1_logit_gap_to_top2_nats=nonnegative_number(
            "mean_top1_logit_gap_to_top2_nats"
        ),
        max_top1_logit_gap_to_top2_nats=nonnegative_number(
            "max_top1_logit_gap_to_top2_nats"
        ),
        high_pressure_low_entropy_count=nonnegative_integer(
            "high_pressure_low_entropy_count"
        ),
        longest_high_pressure_run=nonnegative_integer("longest_high_pressure_run"),
        empty_support_count=nonnegative_integer("empty_support_count"),
        eos_only_support_count=nonnegative_integer("eos_only_support_count"),
        has_non_finite_count=nonnegative_integer("has_non_finite_count"),
        effective_legal_token_count_histogram=tuple(histogram),
        top_pressure_events=tuple(events),
    )


def _parse_candidates(
    tokens: Mapping[str, Any],
    *,
    table_name: str,
    probability_name: str,
    row_count: int,
    legal_masses: Sequence[float],
) -> list[tuple[Candidate, ...]]:
    table = _object(tokens.get(table_name), f"constraint_metrics.tokens.{table_name}")
    path = f"constraint_metrics.tokens.{table_name}"
    width = _integer(table.get("width"), f"{path}.width")
    if width < 0:
        raise SchemaError(f"{path}.width must be non-negative")
    counts = _column(table, "count", path, row_count)
    token_ids = _list(
        table.get("token_id"), f"{path}.token_id", length=row_count * width
    )
    probabilities = _list(
        table.get(probability_name),
        f"{path}.{probability_name}",
        length=row_count * width,
    )
    rows: list[tuple[Candidate, ...]] = []
    for row_index, raw_count in enumerate(counts):
        count = _integer(raw_count, f"{path}.count[{row_index}]")
        if count < 0 or count > width:
            raise SchemaError(
                f"{path}.count[{row_index}]={count} is outside [0, {width}]"
            )
        start = row_index * width
        row: list[Candidate] = []
        for offset in range(count):
            flat_index = start + offset
            token_id = _integer(token_ids[flat_index], f"{path}.token_id[{flat_index}]")
            probability = _probability(
                probabilities[flat_index], f"{path}.{probability_name}[{flat_index}]"
            )
            if probability_name == "constrained_prob":
                row.append(
                    Candidate(
                        token_id=token_id,
                        model_probability=probability * legal_masses[row_index],
                        conditioned_probability=probability,
                    )
                )
            else:
                row.append(Candidate(token_id, probability))
        rows.append(tuple(row))
    return rows


def _parse_tokens(value: Any, mode: str) -> tuple[TokenMetric, ...]:
    if value is None and mode == "triggered":
        return ()
    if value == [] and mode == "summary":
        return ()
    tokens = _object(value, "constraint_metrics.tokens")
    count = _nonnegative_integer(tokens.get("count"), "constraint_metrics.tokens.count")
    path = "constraint_metrics.tokens"

    output_indices = _column(tokens, "output_index", path, count)
    token_ids = _column(tokens, "token_id", path, count)
    grammar_active = _column(tokens, "grammar_active", path, count)
    legal_counts = _column(tokens, "effective_legal_token_count", path, count)
    pressures = _column(tokens, "constraint_pressure_nats", path, count)
    argmax_masked = _column(tokens, "argmax_masked", path, count)
    entropy = _column(tokens, "entropy_nats", path, count)
    spec_sources = _column(tokens, "spec_source", path, count)

    top1 = _object(tokens.get("unconstrained_top1"), f"{path}.unconstrained_top1")
    top1_ids = _column(top1, "token_id", f"{path}.unconstrained_top1", count)
    top1_probs = _column(top1, "prob", f"{path}.unconstrained_top1", count)
    top1_gaps = _column(
        top1, "logit_gap_to_top2_nats", f"{path}.unconstrained_top1", count
    )

    emitted = _object(tokens.get("emitted"), f"{path}.emitted")
    emitted_allowed = _column(emitted, "allowed", f"{path}.emitted", count)
    emitted_log_probs = _column(emitted, "model_log_prob", f"{path}.emitted", count)
    if mode == "forensic":
        emitted_ranks: list[Any | None] = _column(
            emitted, "rank", f"{path}.emitted", count
        )
    else:
        emitted_ranks = [None] * count

    flags = _object(tokens.get("flags"), f"{path}.flags")
    flag_columns = {
        name: _column(flags, name, f"{path}.flags", count) for name in _FLAG_NAMES
    }

    parsed_pressures = [
        _nonnegative_number(value, f"{path}.constraint_pressure_nats[{index}]")
        for index, value in enumerate(pressures)
    ]
    legal_masses = [
        0.0 if pressure >= INF_PRESSURE_SENTINEL else math.exp(-pressure)
        for pressure in parsed_pressures
    ]
    if mode == "forensic":
        top_masked = _parse_candidates(
            tokens,
            table_name="top_masked",
            probability_name="prob",
            row_count=count,
            legal_masses=legal_masses,
        )
        top_legal = _parse_candidates(
            tokens,
            table_name="top_legal_after_conditioning",
            probability_name="constrained_prob",
            row_count=count,
            legal_masses=legal_masses,
        )
    else:
        top_masked = [()] * count
        top_legal = [()] * count

    rows: list[TokenMetric] = []
    for index in range(count):
        source = spec_sources[index]
        if source is not None and not isinstance(source, str):
            raise SchemaError(f"{path}.spec_source[{index}] must be a string or null")
        enabled_flags = tuple(
            name
            for name in _FLAG_NAMES
            if _boolean(flag_columns[name][index], f"{path}.flags.{name}[{index}]")
        )
        rank = emitted_ranks[index]
        parsed_top1_probability = _probability(
            top1_probs[index], f"{path}.unconstrained_top1.prob[{index}]"
        )
        parsed_top1_gap = _nonnegative_number(
            top1_gaps[index],
            f"{path}.unconstrained_top1.logit_gap_to_top2_nats[{index}]",
        )
        parsed_entropy = _nonnegative_number(
            entropy[index], f"{path}.entropy_nats[{index}]"
        )
        parsed_emitted_allowed = _boolean(
            emitted_allowed[index], f"{path}.emitted.allowed[{index}]"
        )
        parsed_emitted_log_probability = _log_probability(
            emitted_log_probs[index],
            f"{path}.emitted.model_log_prob[{index}]",
        )
        if (
            parsed_emitted_allowed
            and parsed_pressures[index] < INF_PRESSURE_SENTINEL
            and parsed_emitted_log_probability + parsed_pressures[index] > 1.0e-4
        ):
            raise SchemaError(
                f"{path}.emitted.model_log_prob[{index}] is inconsistent with "
                f"{path}.constraint_pressure_nats[{index}]"
            )
        rows.append(
            TokenMetric(
                output_index=_nonnegative_integer(
                    output_indices[index], f"{path}.output_index[{index}]"
                ),
                token_id=_integer(token_ids[index], f"{path}.token_id[{index}]"),
                grammar_active=_boolean(
                    grammar_active[index], f"{path}.grammar_active[{index}]"
                ),
                effective_legal_token_count=_nonnegative_integer(
                    legal_counts[index], f"{path}.effective_legal_token_count[{index}]"
                ),
                constraint_pressure_nats=parsed_pressures[index],
                argmax_masked=_boolean(
                    argmax_masked[index], f"{path}.argmax_masked[{index}]"
                ),
                unconstrained_top1_token_id=_integer(
                    top1_ids[index], f"{path}.unconstrained_top1.token_id[{index}]"
                ),
                unconstrained_top1_probability=parsed_top1_probability,
                top1_logit_gap_to_top2_nats=parsed_top1_gap,
                entropy_nats=parsed_entropy,
                emitted_allowed=parsed_emitted_allowed,
                emitted_model_log_probability=parsed_emitted_log_probability,
                emitted_rank=(
                    None
                    if rank is None
                    else _integer(rank, f"{path}.emitted.rank[{index}]")
                ),
                flags=enabled_flags,
                spec_source=source,
                top_masked=top_masked[index],
                top_legal_after_conditioning=top_legal[index],
            )
        )
    return tuple(rows)


def parse_response_obj(
    obj: Any,
    *,
    token_labels: Mapping[int, str] | None = None,
    source: str | None = None,
    endpoint: str | None = None,
) -> ConstraintMetrics:
    """Parse a bare metric object, meta_info, or one `/generate` response."""

    metrics, response = _locate_metrics(obj)
    status = metrics.get("status")
    if status not in _STATUSES:
        raise SchemaError(
            f"constraint_metrics.status must be one of {sorted(_STATUSES)}"
        )
    mode = metrics.get("mode")
    if mode not in _MODES:
        raise SchemaError(f"constraint_metrics.mode must be one of {sorted(_MODES)}")
    warnings = _list(metrics.get("warnings"), "constraint_metrics.warnings")
    if not all(isinstance(warning, str) for warning in warnings):
        raise SchemaError("constraint_metrics.warnings must contain only strings")
    if "summary" not in metrics:
        raise SchemaError("constraint_metrics.summary is missing")
    summary = _parse_summary(metrics["summary"])
    tokens = _parse_tokens(metrics.get("tokens"), mode)
    output_text = response.get("text") if response else None
    if output_text is not None and not isinstance(output_text, str):
        output_text = None
    triggered_token_count = None
    if mode == "triggered":
        triggered_token_count = _nonnegative_integer(
            metrics.get("triggered_token_count"),
            "constraint_metrics.triggered_token_count",
        )
    input_token_count = _nonnegative_integer(
        metrics.get("input_token_count"), "constraint_metrics.input_token_count"
    )
    output_token_count = _nonnegative_integer(
        metrics.get("output_token_count"), "constraint_metrics.output_token_count"
    )
    if status == "complete" and summary.output_token_count != output_token_count:
        raise SchemaError(
            "constraint_metrics.summary.output_token_count does not match "
            "constraint_metrics.output_token_count"
        )
    if (
        status == "partially_dropped"
        and summary.output_token_count >= output_token_count
    ):
        raise SchemaError(
            "partially dropped constraint metrics must summarize fewer tokens "
            "than constraint_metrics.output_token_count"
        )
    return ConstraintMetrics(
        status=status,
        mode=mode,
        warnings=tuple(warnings),
        input_token_count=input_token_count,
        output_token_count=output_token_count,
        summary=summary,
        tokens=tokens,
        token_labels=dict(token_labels or {}),
        triggered_token_count=triggered_token_count,
        output_text=output_text,
        source=source,
        endpoint=endpoint,
    )
