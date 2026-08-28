"""Normal-decoding observer for constraint metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import torch

from sglang.srt.sampling.sampling_observer import CommittedTokens
from sglang_constraint_metrics import layout
from sglang_constraint_metrics.capture import (
    CaptureInputs,
    CaptureScratch,
    ConstraintMetricsCapturePlan,
    build_capture_plan,
    grammar_has_constraint_metrics,
    select_capture_inputs,
)
from sglang_constraint_metrics.config import (
    CONSTRAINT_METRICS_STREAM_ENABLED_KEY,
    CONSTRAINT_METRICS_STREAM_KEY,
    MODE_SUMMARY,
    MODE_TRIGGERED,
    TOP_K_COMPUTE_BUCKETS,
    ConstraintMetricsConfig,
)
from sglang_constraint_metrics.mask import MASK_PACKED_ALLOWED, grammar_mask_kind
from sglang_constraint_metrics.records import (
    CommittedRecordSlice,
    ConstraintMetricsPackedRows,
    ConstraintMetricsRecordPayload,
    ConstraintMetricsTensorBatch,
    _compute_record_tensor_batch,
    archive_record_payloads,
    build_record_payload,
    compact_record_payloads,
    copy_record_payload_rows,
    move_record_payloads_to_host,
    packed_kernel_row_to_record,
    uniform_top_k_ranges,
)
from sglang_constraint_metrics.schema import (
    STATUS_COMPLETE,
    STATUS_PARTIALLY_DROPPED,
    ConstraintMetricsSummaryAccumulator,
    extend_kernel_row_segments,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


@dataclass
class _ObserverState:
    plan: ConstraintMetricsCapturePlan
    inputs: CaptureInputs
    grammar_mask_kind: int


@dataclass(frozen=True, slots=True)
class ConstraintMetricsStreamRecord:
    rows: ConstraintMetricsPackedRows
    row_index: int
    triggered: bool

    def materialize(self) -> dict[str, Any]:
        row_index = self.row_index
        record = packed_kernel_row_to_record(
            self.rows.floats[row_index],
            self.rows.ints[row_index],
            grammar_active=self.rows.grammar_active[row_index],
            output_index=self.rows.output_indices[row_index],
        )
        record["triggered"] = self.triggered
        return record


def _publish_stream_records(
    commits: Sequence[tuple[Req, ConstraintMetricsState, CommittedRecordSlice]],
    rows: ConstraintMetricsPackedRows,
    triggered_by_commit: Sequence[Sequence[bool]],
) -> None:
    streamed_row_indices = []
    row_cursor = 0
    for (_, state, record_slice), triggered_rows in zip(
        commits, triggered_by_commit, strict=True
    ):
        if state.streaming:
            streamed_row_indices.extend(
                row_cursor + offset
                for offset, triggered in enumerate(triggered_rows)
                if triggered or state.config.stream_all_token_metrics
            )
        row_cursor += record_slice.count
    streamed_rows = rows.select(streamed_row_indices) if streamed_row_indices else None

    row_cursor = 0
    streamed_index = 0
    for (req, state, record_slice), triggered_rows in zip(
        commits, triggered_by_commit, strict=True
    ):
        row_end = row_cursor + record_slice.count
        if state.streaming:
            records = req.customized_info[CONSTRAINT_METRICS_STREAM_KEY]
            if record_slice.count:
                last_output_index = int(rows.output_indices[row_end - 1])
                if len(records) <= last_output_index:
                    records.extend([None] * (last_output_index + 1 - len(records)))
            for offset, triggered in enumerate(triggered_rows):
                if not triggered and not state.config.stream_all_token_metrics:
                    continue
                row_index = row_cursor + offset
                if streamed_rows is None:
                    raise RuntimeError("streamed constraint metric rows are missing")
                output_index = int(rows.output_indices[row_index])
                if records[output_index] is not None:
                    raise RuntimeError(
                        "streamed constraint metric position was already populated"
                    )
                records[output_index] = ConstraintMetricsStreamRecord(
                    rows=streamed_rows,
                    row_index=streamed_index,
                    triggered=triggered,
                )
                streamed_index += 1
        row_cursor = row_end
    if streamed_index != len(streamed_row_indices):
        raise RuntimeError("streamed constraint metric rows are misaligned")


@dataclass
class ConstraintMetricsDeviceOutput:
    record_payloads: list[ConstraintMetricsRecordPayload] = field(default_factory=list)

    @staticmethod
    def _copy_payload_to_host(
        payload: ConstraintMetricsRecordPayload,
        copy_tensor: Callable[[torch.Tensor], torch.Tensor],
        *,
        scalars_only: bool,
    ) -> ConstraintMetricsRecordPayload:
        device_batch = payload.tensor_batch
        floats = device_batch.floats
        ints = device_batch.ints
        top_ks = list(device_batch.top_ks)
        if scalars_only:
            floats = floats[:, : layout.KERNEL_ROW_FLOAT_COUNT]
            ints = ints[:, : layout.KERNEL_ROW_INT_COUNT]
            top_ks = [0] * len(top_ks)
        return ConstraintMetricsRecordPayload(
            tensor_batch=ConstraintMetricsTensorBatch(
                floats=copy_tensor(floats),
                ints=copy_tensor(ints),
                top_ks=top_ks,
                grammar_active=list(device_batch.grammar_active),
                output_indices=list(device_batch.output_indices),
                vocab_size=device_batch.vocab_size,
            ),
            row_spans=list(payload.row_spans),
            speculative=payload.speculative,
            defer_commit_validation=(
                payload.defer_commit_validation if not scalars_only else False
            ),
        )

    def copy_to_host(
        self, copy_tensor: Callable[[torch.Tensor], torch.Tensor]
    ) -> ConstraintMetricsHostOutput:
        host_payloads = []
        stream_payloads = []
        for payload in self.record_payloads:
            if payload.device_retained:
                host_payloads.append(payload)
                if payload.stream_to_host:
                    stream_payloads.append(
                        self._copy_payload_to_host(
                            payload,
                            copy_tensor,
                            scalars_only=True,
                        )
                    )
                continue
            host_payloads.append(
                self._copy_payload_to_host(
                    payload,
                    copy_tensor,
                    scalars_only=False,
                )
            )
        return ConstraintMetricsHostOutput(
            record_payloads=host_payloads,
            stream_record_payloads=stream_payloads,
        )


@dataclass
class ConstraintMetricsHostOutput:
    record_payloads: list[ConstraintMetricsRecordPayload] = field(default_factory=list)
    stream_record_payloads: list[ConstraintMetricsRecordPayload] = field(
        default_factory=list
    )

    def consume(
        self,
        batch: ScheduleBatch,
        commits: Sequence[Optional[CommittedTokens]],
    ) -> None:
        if len(commits) != len(batch.reqs):
            raise ValueError("constraint metric commits do not match the batch")
        payload_by_batch_index = {}
        for payload in self.record_payloads:
            for batch_index, _, _ in payload.row_spans:
                if batch_index in payload_by_batch_index:
                    raise ValueError(
                        "constraint metric request appears in multiple payloads"
                    )
                payload_by_batch_index[batch_index] = payload
        stream_payload_by_batch_index = {}
        for payload in self.stream_record_payloads:
            for batch_index, _, _ in payload.row_spans:
                if batch_index in stream_payload_by_batch_index:
                    raise ValueError(
                        "constraint metric request appears in multiple stream payloads"
                    )
                stream_payload_by_batch_index[batch_index] = payload
        aggregate_commits = {}
        retained_payloads = set()
        for batch_index, payload in payload_by_batch_index.items():
            req = batch.reqs[batch_index]
            commit = commits[batch_index]
            if commit is None or req.is_retracted:
                continue
            state = constraint_metrics_state_for_req(req)
            if state is None:
                raise RuntimeError("constraint metrics request state is missing")
            stream_payload = stream_payload_by_batch_index.get(batch_index)
            validation_payload = None
            if payload.defer_commit_validation and state.streaming:
                if stream_payload is None:
                    raise RuntimeError("streamed forensic scalar payload is missing")
                validation_payload = stream_payload
            record_slice = state.commit(
                payload,
                batch_index,
                commit,
                validation_payload=validation_payload,
            )
            if record_slice is None:
                continue
            aggregate_payload = None
            aggregate_slice = record_slice
            if state.config.mode in (MODE_SUMMARY, MODE_TRIGGERED):
                aggregate_payload = payload
            else:
                retained_payloads.add(id(payload))
                if state.streaming:
                    aggregate_payload = stream_payload
                    if aggregate_payload is None:
                        if payload.device_retained:
                            raise RuntimeError(
                                "streamed forensic scalar payload is missing"
                            )
                        aggregate_payload = payload
                    aggregate_slice = CommittedRecordSlice(
                        payload=aggregate_payload,
                        start=record_slice.start,
                        count=record_slice.count,
                        output_index_start=record_slice.output_index_start,
                    )
            if aggregate_payload is not None:
                payload_entry = aggregate_commits.setdefault(
                    id(aggregate_payload), (aggregate_payload, [])
                )
                payload_entry[1].append((req, state, aggregate_slice))

        for payload, commits_for_payload in aggregate_commits.values():
            packed_rows = payload.packed_rows_for_slices(
                [record_slice for _, _, record_slice in commits_for_payload]
            )
            triggered_by_commit = extend_kernel_row_segments(
                [state.summary_accumulator for _, state, _ in commits_for_payload],
                packed_rows.floats,
                packed_rows.ints,
                packed_rows.grammar_active,
                packed_rows.output_indices,
                [record_slice.count for _, _, record_slice in commits_for_payload],
                [
                    state.config.mode == MODE_TRIGGERED
                    or state.config.stream_all_token_metrics
                    for _, state, _ in commits_for_payload
                ],
            )
            triggered_locations = [
                (
                    commit_index,
                    record_slice.start + offset,
                    record_slice.output_index_start + offset,
                )
                for commit_index, (
                    (_, state, record_slice),
                    triggered_rows,
                ) in enumerate(
                    zip(commits_for_payload, triggered_by_commit, strict=True)
                )
                for offset, triggered in enumerate(triggered_rows)
                if triggered and not state.streaming
            ]
            triggered_payload = (
                copy_record_payload_rows(
                    payload,
                    [source_index for _, source_index, _ in triggered_locations],
                    [output_index for _, _, output_index in triggered_locations],
                )
                if triggered_locations
                else None
            )
            triggered_slices_by_commit = [[] for _ in range(len(commits_for_payload))]
            if triggered_payload is not None:
                for selected_index, (
                    commit_index,
                    _,
                    output_index,
                ) in enumerate(triggered_locations):
                    triggered_slices_by_commit[commit_index].append(
                        CommittedRecordSlice(
                            payload=triggered_payload,
                            start=selected_index,
                            count=1,
                            output_index_start=output_index,
                        )
                    )
            _publish_stream_records(
                commits_for_payload,
                packed_rows,
                triggered_by_commit,
            )
            for commit_index, (_, state, _) in enumerate(commits_for_payload):
                if state.config.mode != MODE_TRIGGERED:
                    continue
                if state.streaming:
                    state.triggered_event_count += sum(
                        triggered_by_commit[commit_index]
                    )
                else:
                    state.retain_triggered_slices(
                        triggered_slices_by_commit[commit_index]
                    )

        if retained_payloads:
            with torch.profiler.record_function("_store_constraint_metrics_pageable"):
                for payload in self.record_payloads:
                    if id(payload) in retained_payloads and not payload.device_retained:
                        payload.move_to_pageable_storage()
        for payload in self.record_payloads:
            payload.release_device_storage_if_unreferenced()
        self.record_payloads.clear()
        self.stream_record_payloads.clear()


@dataclass(eq=False)
class ConstraintMetricsState:
    config: ConstraintMetricsConfig
    streaming: bool = False
    record_slices: list[CommittedRecordSlice] = field(default_factory=list, init=False)
    summary_accumulator: ConstraintMetricsSummaryAccumulator = field(
        default_factory=ConstraintMetricsSummaryAccumulator
    )
    captured_prefix_length: int = field(default=0, init=False)
    capture_failed_at: Optional[int] = field(default=None, init=False)
    triggered_event_count: int = field(default=0, init=False)
    stream_release_offset: int = field(default=0, init=False)

    def commit(
        self,
        payload: ConstraintMetricsRecordPayload,
        batch_index: int,
        commit: CommittedTokens,
        *,
        validation_payload: Optional[ConstraintMetricsRecordPayload] = None,
    ) -> Optional[CommittedRecordSlice]:
        token_ids = commit.token_ids
        if not token_ids or self.capture_failed_at is not None:
            return None
        if commit.output_index != self.captured_prefix_length:
            self.capture_failed_at = self.captured_prefix_length
            return None
        committed_payload = validation_payload or payload
        record_slice, first_mismatch = committed_payload.commit(
            batch_index,
            output_index=commit.output_index,
            token_ids=token_ids,
        )
        if record_slice is not None and validation_payload is not None:
            if payload.span_for(batch_index) != validation_payload.span_for(
                batch_index
            ):
                raise RuntimeError(
                    "constraint metrics stream and retained payload spans differ"
                )
            record_slice = CommittedRecordSlice(
                payload=payload,
                start=record_slice.start,
                count=record_slice.count,
                output_index_start=record_slice.output_index_start,
            )
        if record_slice is not None:
            self.captured_prefix_length += record_slice.count
            if self.config.mode not in (MODE_SUMMARY, MODE_TRIGGERED):
                self.record_slices.append(record_slice)
                record_slice.payload.acquire_committed_slice()
        if first_mismatch is not None:
            if first_mismatch != self.captured_prefix_length:
                raise RuntimeError("constraint metric mismatch is not at prefix end")
            self.capture_failed_at = first_mismatch
        return record_slice

    def clear_record_slices(self) -> None:
        record_slices = self.record_slices
        self.record_slices = []
        for record_slice in record_slices:
            record_slice.payload.release_committed_slice()

    def retain_triggered_slices(
        self, event_slices: Sequence[CommittedRecordSlice]
    ) -> None:
        if not event_slices:
            return
        if self.config.mode != MODE_TRIGGERED:
            raise ValueError("only triggered mode can retain sparse event slices")
        for event_slice in event_slices:
            self.record_slices.append(event_slice)
            event_slice.payload.acquire_committed_slice()
        self.triggered_event_count += len(event_slices)

    def snapshot(
        self,
        req: Req,
        record_slices: list[CommittedRecordSlice],
        *,
        validate_terminal_records: bool = False,
    ) -> ConstraintMetricsResponse:
        input_token_count = len(req.origin_input_ids)
        output_token_count = len(req.output_ids_through_stop)
        return ConstraintMetricsResponse(
            config=self.config,
            record_slices=record_slices,
            validation_token_ids=(
                list(req.output_ids_through_stop) if validate_terminal_records else []
            ),
            input_token_count=input_token_count,
            output_token_count=output_token_count,
            summary=self.summary_accumulator.to_summary(),
            captured_prefix_length=self.captured_prefix_length,
            streamed_events=self.streaming,
            triggered_event_count=self.triggered_event_count,
            validate_terminal_records=validate_terminal_records,
        )


@dataclass
class ConstraintMetricsResponse:
    config: ConstraintMetricsConfig
    record_slices: list[CommittedRecordSlice]
    validation_token_ids: list[int]
    input_token_count: int
    output_token_count: int
    summary: dict[str, Any]
    captured_prefix_length: int
    streamed_events: bool = False
    triggered_event_count: int = 0
    validate_terminal_records: bool = False

    def _validated_prefix_length(self) -> int:
        if not self.validate_terminal_records:
            return self.captured_prefix_length
        expected_output_index = 0
        for record_slice in self.record_slices:
            if record_slice.output_index_start != expected_output_index:
                raise RuntimeError(
                    "constraint metric records are not a contiguous prefix"
                )
            validated_end = record_slice.validated_output_index_end(
                self.validation_token_ids
            )
            if validated_end < record_slice.output_index_start + record_slice.count:
                return validated_end
            expected_output_index = validated_end
        return expected_output_index

    def _materialize_forensic_tokens(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any], int]:
        validated_prefix_length = self._validated_prefix_length()
        summary_accumulator = ConstraintMetricsSummaryAccumulator()
        if len(self.record_slices) == 1:
            record_slice = self.record_slices[0]
            if (
                record_slice.output_index_start != 0
                or record_slice.count != self.captured_prefix_length
            ):
                raise RuntimeError(
                    "constraint metric records are not a contiguous prefix"
                )
            batch = record_slice.payload.tensor_batch
            materialize_kwargs = {
                "row_start": record_slice.start,
                "row_count": validated_prefix_length,
                "output_index_start": record_slice.output_index_start,
                "summary_accumulator": summary_accumulator,
            }
            tokens = batch.to_columns(**materialize_kwargs)
            captured = int(tokens["count"])
            if captured != validated_prefix_length:
                raise RuntimeError("constraint metric record prefix is inconsistent")
            return tokens, summary_accumulator.to_summary(), captured
        packed_records = compact_record_payloads(
            self.record_slices, max_top_k=self.config.kernel_top_k
        )
        if packed_records is None:
            if self.captured_prefix_length:
                raise RuntimeError("constraint metric record prefix is missing")
            return (
                ConstraintMetricsTensorBatch.empty_columns(),
                summary_accumulator.to_summary(),
                0,
            )
        expected_indices = list(range(self.captured_prefix_length))
        if packed_records.output_indices != expected_indices:
            raise RuntimeError("constraint metric records are not a contiguous prefix")
        tokens = packed_records.to_columns(
            row_count=validated_prefix_length,
            summary_accumulator=summary_accumulator,
        )
        captured = int(tokens["count"])
        if captured != validated_prefix_length:
            raise RuntimeError("constraint metric record prefix is inconsistent")
        return tokens, summary_accumulator.to_summary(), captured

    def _materialize_triggered_tokens(self) -> dict[str, Any]:
        packed_records = compact_record_payloads(
            sorted(
                self.record_slices,
                key=lambda record_slice: record_slice.output_index_start,
            ),
            max_top_k=self.config.kernel_top_k,
        )
        if packed_records is None:
            return ConstraintMetricsTensorBatch.empty_columns(
                width=self.config.kernel_top_k,
                include_tables=False,
            )
        return packed_records.to_columns(include_tables=False)

    def materialize_constraint_metrics(self) -> dict[str, Any]:
        expected_records = self.output_token_count

        if self.config.mode == MODE_SUMMARY:
            tokens = []
            captured_records = self.captured_prefix_length
            if int(self.summary["output_token_count"]) != captured_records:
                raise RuntimeError("constraint metric summary prefix is inconsistent")
            summary = self.summary
        elif self.config.mode == MODE_TRIGGERED:
            tokens = self._materialize_triggered_tokens()
            captured_records = self.captured_prefix_length
            summary = self.summary
        else:
            tokens, summary, captured_records = self._materialize_forensic_tokens()
        if captured_records > expected_records:
            raise RuntimeError("constraint metrics exceed the terminal output prefix")
        missing = expected_records - captured_records
        complete = missing == 0
        warnings = (
            []
            if complete
            else [f"constraint metrics are missing {missing} committed token record(s)"]
        )
        if self.config.mode != MODE_SUMMARY and (
            not isinstance(tokens, dict) or not isinstance(tokens.get("count"), int)
        ):
            raise ValueError("token-emitting constraint metrics require token columns")
        response = {
            "status": STATUS_COMPLETE if complete else STATUS_PARTIALLY_DROPPED,
            "mode": self.config.mode,
            "warnings": warnings,
            "summary": dict(summary),
            "input_token_count": int(self.input_token_count),
            "output_token_count": int(self.output_token_count),
        }
        if not (self.config.mode == MODE_TRIGGERED and self.streamed_events):
            response["tokens"] = tokens
        if self.config.mode == MODE_TRIGGERED:
            response["triggered_token_count"] = self.triggered_event_count
        return response


_REQUEST_STATE_ATTR = "_constraint_metrics_state"


def bind_constraint_metrics_state(
    req: Req,
    *,
    config: Optional[ConstraintMetricsConfig],
) -> Optional[ConstraintMetricsState]:
    if config is None:
        return None
    sampling_params = req.sampling_params
    grammar_present = any(
        value is not None
        for value in (
            sampling_params.json_schema,
            sampling_params.regex,
            sampling_params.ebnf,
            sampling_params.structural_tag,
        )
    )
    if not grammar_present:
        raise RuntimeError(
            "constraint metrics request reached the scheduler without a grammar"
        )
    state = ConstraintMetricsState(
        config=config,
        streaming=(config.mode == MODE_TRIGGERED or config.stream_all_token_metrics)
        and req.stream,
    )
    if _REQUEST_STATE_ATTR in req.__dict__:
        raise RuntimeError("constraint metrics state was already bound")
    req.__dict__[_REQUEST_STATE_ATTR] = state
    if state.streaming:
        if req.customized_info is None:
            req.customized_info = {}
        for key in (
            CONSTRAINT_METRICS_STREAM_KEY,
            CONSTRAINT_METRICS_STREAM_ENABLED_KEY,
        ):
            if key in req.customized_info:
                raise ValueError(f"duplicate {key} response")
        req.customized_info[CONSTRAINT_METRICS_STREAM_KEY] = []
        req.customized_info[CONSTRAINT_METRICS_STREAM_ENABLED_KEY] = []
    return state


def constraint_metrics_state_for_req(req: Req) -> Optional[ConstraintMetricsState]:
    return req.__dict__.get(_REQUEST_STATE_ATTR)


def discard_constraint_metrics_state_for_req(req: Req) -> None:
    state = req.__dict__.pop(_REQUEST_STATE_ATTR, None)
    if state is None:
        return
    state.clear_record_slices()
    if state.streaming:
        for key in (
            CONSTRAINT_METRICS_STREAM_KEY,
            CONSTRAINT_METRICS_STREAM_ENABLED_KEY,
        ):
            values = req.customized_info.get(key)
            if values is not None:
                values.clear()


def prepare_streamed_constraint_metrics(reqs: Sequence[Req]) -> None:
    """Pad capture gaps before the output streamer slices token metadata."""

    for req in reqs:
        state = constraint_metrics_state_for_req(req)
        if state is None or not state.streaming:
            continue
        records = req.customized_info[CONSTRAINT_METRICS_STREAM_KEY]
        enabled = req.customized_info[CONSTRAINT_METRICS_STREAM_ENABLED_KEY]
        output_length = len(req.output_ids_through_stop)
        if len(records) > output_length or len(enabled) > output_length:
            raise RuntimeError("streamed constraint metrics exceed visible output")
        records.extend([None] * (output_length - len(records)))
        enabled.extend([True] * (output_length - len(enabled)))


def release_streamed_constraint_metrics(reqs: Sequence[Req]) -> None:
    """Release scheduler-owned records after the output streamer has sliced them."""

    for req in reqs:
        state = constraint_metrics_state_for_req(req)
        if state is None or not state.streaming:
            continue
        end = req.send_token_offset
        start = state.stream_release_offset
        if end < start:
            raise RuntimeError("streamed constraint metric offset moved backwards")
        records = req.customized_info[CONSTRAINT_METRICS_STREAM_KEY]
        enabled = req.customized_info[CONSTRAINT_METRICS_STREAM_ENABLED_KEY]
        for index in range(start, end):
            records[index] = None
            enabled[index] = None
        state.stream_release_offset = end


def snapshot_constraint_metrics(
    reqs: Sequence[Req],
) -> list[Optional[ConstraintMetricsResponse]]:
    snapshots: list[Optional[ConstraintMetricsResponse]] = [None] * len(reqs)
    forensic = []
    for index, req in enumerate(reqs):
        state = constraint_metrics_state_for_req(req)
        if state is None or not req.finished():
            continue
        if state.config.mode == MODE_SUMMARY:
            snapshots[index] = state.snapshot(req, [])
        elif state.config.mode == MODE_TRIGGERED:
            snapshots[index] = state.snapshot(
                req,
                [] if state.streaming else state.record_slices,
            )
            state.clear_record_slices()
        else:
            forensic.append((index, req, state))
    if forensic:
        with torch.profiler.record_function("_archive_constraint_metrics_terminal"):
            terminal_validation = [
                any(
                    record_slice.payload.defer_commit_validation
                    for record_slice in state.record_slices
                )
                for _, _, state in forensic
            ]
            histories = archive_record_payloads(
                [state.record_slices for _, _, state in forensic]
            )
            move_record_payloads_to_host(histories)
            prepared_snapshots = []
            for (index, req, state), history, validation in zip(
                forensic, histories, terminal_validation, strict=True
            ):
                prepared_snapshots.append(
                    (
                        index,
                        state.snapshot(
                            req,
                            history,
                            validate_terminal_records=validation,
                        ),
                    )
                )
            for _, _, state in forensic:
                state.clear_record_slices()
            for index, snapshot in prepared_snapshots:
                snapshots[index] = snapshot
    return snapshots


class ConstraintMetricsObserver:
    """Sampling observer installed when the plugin is selected."""

    def __init__(self, *, model_config: Any):
        self._eos_token_ids = list(model_config.hf_eos_token_id)
        self._capture_scratch: Optional[CaptureScratch] = None
        self._capture_scratch_is_bounded = False

    def reserve_capture_scratch(
        self,
        *,
        max_rows: int,
        vocab_size: int,
        device: torch.device | str,
    ) -> int:
        if max_rows <= 0 or vocab_size <= 0:
            raise ValueError("constraint metrics scratch shape must be positive")
        self._capture_scratch = CaptureScratch.allocate(
            max_rows,
            vocab_size,
            device,
        )
        self._capture_scratch_is_bounded = True
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self._capture_scratch.pre,
                self._capture_scratch.emitted_ids,
                self._capture_scratch.spec_sources,
                self._capture_scratch.row_enabled,
            )
        )

    def warmup_kernels(
        self,
        *,
        max_top_k: int,
        vocab_size: int,
        device: torch.device | str,
        filter_rows: bool,
    ) -> None:
        """Load every request-visible kernel bucket before serving starts."""
        scratch = self._capture_scratch
        if scratch is None:
            raise RuntimeError(
                "reserve constraint metrics scratch before kernel warmup"
            )

        pre = scratch.pre[:1]
        pre.zero_()
        grammar_mask = torch.full(
            (1, (vocab_size + 31) // 32),
            -1,
            dtype=torch.int32,
            device=device,
        )
        temperatures = torch.ones(1, dtype=torch.float32, device=device)
        emitted_ids = torch.zeros(1, dtype=torch.int64, device=device)
        spec_sources = torch.zeros(1, dtype=torch.int64, device=device)
        row_enabled = (
            torch.ones(1, dtype=torch.bool, device=device) if filter_rows else None
        )

        for bucket in TOP_K_COMPUTE_BUCKETS:
            _compute_record_tensor_batch(
                pre,
                grammar_mask,
                emitted_ids,
                temperatures=temperatures,
                top_ks=[bucket],
                grammar_active=[True],
                output_indices=[0],
                eos_token_ids=self._eos_token_ids,
                spec_sources=spec_sources,
                row_enabled=row_enabled,
                max_top_k=bucket,
                grammar_mask_kind=MASK_PACKED_ALLOWED,
            )
            if max_top_k <= bucket:
                break

    def prepare_runtime_workspace(
        self,
        *,
        max_rows: int,
        max_top_k: int,
        vocab_size: int,
        device: torch.device | str,
        linear_speculation_width: Optional[int] = None,
    ) -> int:
        """Reserve device scratch and compile kernels before KV-cache sizing."""
        capture_bytes = self.reserve_capture_scratch(
            max_rows=max_rows,
            vocab_size=vocab_size,
            device=device,
        )
        from sglang_constraint_metrics.kernel import (
            reserve_constraint_metrics_kernel_scratch,
        )

        kernel_bytes = reserve_constraint_metrics_kernel_scratch(
            rows=max_rows,
            vocab_size=vocab_size,
            max_top_k=max_top_k,
            device=device,
        )
        self.warmup_kernels(
            max_top_k=max_top_k,
            vocab_size=vocab_size,
            device=device,
            filter_rows=False,
        )
        if linear_speculation_width is not None:
            self.warmup_kernels(
                max_top_k=max_top_k,
                vocab_size=vocab_size,
                device=device,
                filter_rows=True,
            )
            from sglang_constraint_metrics.linear_speculation import (
                warmup_linear_record_metadata,
            )

            warmup_linear_record_metadata(
                width=linear_speculation_width,
                device=device,
                scratch=self._capture_scratch,
            )
        return capture_bytes + kernel_bytes

    def capture_scratch(
        self,
        *,
        rows: int,
        vocab_size: int,
        logits: torch.Tensor,
    ) -> CaptureScratch:
        scratch = self._capture_scratch
        incompatible = (
            scratch is None
            or scratch.pre.device != logits.device
            or scratch.pre.shape[1] != vocab_size
        )
        if incompatible:
            if self._capture_scratch_is_bounded:
                raise RuntimeError(
                    "constraint metrics scratch does not match the model"
                )
            scratch = CaptureScratch.allocate(rows, vocab_size, logits.device)
            self._capture_scratch = scratch
        elif rows > scratch.capacity:
            if self._capture_scratch_is_bounded:
                raise RuntimeError(
                    "constraint metrics batch exceeds the startup-reserved workspace"
                )
            scratch = CaptureScratch.allocate(rows, vocab_size, logits.device)
            self._capture_scratch = scratch
        return scratch

    @staticmethod
    def is_active(sampling_info: SamplingBatchInfo) -> bool:
        return any(
            grammar_has_constraint_metrics(grammar)
            for grammar in (sampling_info.grammars or ())
        )

    @staticmethod
    def is_linear_speculation_active(batch: ScheduleBatch) -> bool:
        """Check request grammars without mutating shared sampling state."""
        return any(grammar_has_constraint_metrics(req.grammar) for req in batch.reqs)

    def before_grammar(
        self,
        logits: torch.Tensor,
        sampling_info: SamplingBatchInfo,
    ) -> Optional[_ObserverState]:
        grammars = sampling_info.grammars or []
        plan = build_capture_plan(grammars)
        if plan is None or not plan.diagnostic_batch_indices:
            return None
        diagnostic_indices = plan.diagnostic_batch_indices
        scratch = self.capture_scratch(
            rows=len(diagnostic_indices),
            vocab_size=logits.shape[1],
            logits=logits,
        )
        grammar_mask = sampling_info.grammar_mask
        if grammar_mask is None:
            raise RuntimeError(
                "constraint metrics require the filled grammar mask before sampling"
            )
        inputs = select_capture_inputs(
            logits,
            sampling_info.temperatures,
            grammar_mask.vocab_mask,
            diagnostic_indices,
            scratch,
            logit_bias=sampling_info.logit_bias,
        )

        return _ObserverState(
            plan=plan,
            inputs=inputs,
            grammar_mask_kind=grammar_mask_kind(inputs.grammar_mask),
        )

    def before_linear_speculation(
        self,
        target_logits: torch.Tensor,
        batch: ScheduleBatch,
        vocab_mask: Optional[torch.Tensor],
    ):
        from sglang_constraint_metrics.linear_speculation import (
            begin_linear_speculation_capture,
        )

        return begin_linear_speculation_capture(
            target_logits,
            batch,
            vocab_mask,
            eos_token_ids=self._eos_token_ids,
            scratch=self.capture_scratch(
                rows=target_logits.shape[0] * target_logits.shape[1],
                vocab_size=target_logits.shape[2],
                logits=target_logits,
            ),
        )

    def after_sample(
        self, state: _ObserverState, token_ids: torch.Tensor
    ) -> ConstraintMetricsDeviceOutput:
        payloads = []
        inputs = state.inputs
        emitted = (
            token_ids
            if inputs.diagnostic_indices_device is None
            else token_ids.index_select(0, inputs.diagnostic_indices_device)
        )
        plan = state.plan
        for start, end, max_top_k in uniform_top_k_ranges(plan.top_ks):
            tensor_batch = _compute_record_tensor_batch(
                inputs.pre[start:end],
                inputs.grammar_mask[start:end],
                emitted[start:end],
                temperatures=inputs.temperatures[start:end],
                top_ks=plan.top_ks[start:end],
                grammar_active=plan.grammar_active[start:end],
                output_indices=[0] * (end - start),
                eos_token_ids=self._eos_token_ids,
                grammar_mask_kind=state.grammar_mask_kind,
                max_top_k=max_top_k,
            )
            payloads.append(
                build_record_payload(
                    tensor_batch,
                    plan.diagnostic_batch_indices[start:end],
                    rows_per_request=1,
                    retention_flags=plan.retain_on_device[start:end],
                    streaming_flags=plan.stream_all_token_metrics[start:end],
                )
            )
        return ConstraintMetricsDeviceOutput(record_payloads=payloads)
