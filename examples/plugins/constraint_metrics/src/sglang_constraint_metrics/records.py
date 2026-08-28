"""Packed fused-kernel records for constraint metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from sglang_constraint_metrics import layout
from sglang_constraint_metrics.config import TOP_K_COMPUTE_BUCKETS
from sglang_constraint_metrics.mask import MASK_DENSE_ALLOWED
from sglang_constraint_metrics.schema import (
    INF_PRESSURE_SENTINEL,
    ConstraintMetricsSummaryAccumulator,
    extend_kernel_row_segments,
)

_SANITIZE_FILL = 1.0e30


def async_index_to_device(
    values: Sequence[int], device: Any, *, dtype: torch.dtype = torch.long
) -> torch.Tensor:
    """Copy a small host index list without a pageable CUDA stream drain."""
    device = torch.device(device)
    is_cuda = device.type == "cuda"
    host = torch.tensor(values, dtype=dtype, pin_memory=is_cuda)
    return host.to(device, non_blocking=is_cuda)


def async_to_device(
    t: torch.Tensor, device: Any, *, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Move a tensor to ``device`` without a pageable CUDA stream drain."""
    device = torch.device(device)
    if t.device == device:
        return t if (dtype is None or t.dtype == dtype) else t.to(dtype=dtype)
    if device.type == "cuda" and t.device.type == "cpu":
        return t.pin_memory().to(device=device, dtype=dtype, non_blocking=True)
    return t.to(device=device, dtype=dtype)


# Dense speculative deferral avoids a host synchronization by staging every bounded chain
# position. The kernel skips metric math for uncommitted positions, but its scratch
# allocation still needs an explicit envelope independent of adaptive chain width.
_SPECULATIVE_DENSE_DEFER_MAX_WORK_ELEMENTS = 96 * 1024 * 1024
# Copying up to 2x selected rows avoids one index_select per source payload while
# bounding the extra memory traffic for staggered completion cohorts.
_ARCHIVE_FULL_PAYLOAD_MAX_ROW_RATIO = 2
# Terminal request-major packing can temporarily hold source, concatenated,
# gathered, and narrowed device tensors. Bound retained sources to 64 MiB so
# that worst-case archive workspace remains within a 256 MiB envelope.
_DEVICE_RETENTION_MAX_BYTES = 64 * 1024 * 1024
_DEVICE_ARCHIVE_HEADROOM_BYTES = 4 * _DEVICE_RETENTION_MAX_BYTES
_DEVICE_RETENTION_BYTES = 0
_DEVICE_RETENTION_LOCK = Lock()


def device_archive_headroom_bytes() -> int:
    """Return the bounded peak GPU workspace for terminal record packing."""
    return _DEVICE_ARCHIVE_HEADROOM_BYTES


def reserve_device_archive_headroom(device: Any) -> torch.Tensor:
    """Hold the bounded archival peak while the KV pool is sized."""
    return torch.empty(
        device_archive_headroom_bytes(),
        dtype=torch.uint8,
        device=device,
    )


def _move_to_pageable_storage(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu" or not tensor.is_pinned():
        return tensor
    pageable = torch.empty(tensor.shape, dtype=tensor.dtype)
    pageable.copy_(tensor)
    return pageable


class _DeviceRetentionLease:
    __slots__ = ("size", "released")

    def __init__(self, size: int):
        self.size = size
        self.released = False

    def release(self) -> None:
        global _DEVICE_RETENTION_BYTES
        if self.released:
            return
        with _DEVICE_RETENTION_LOCK:
            if not self.released:
                _DEVICE_RETENTION_BYTES -= self.size
                self.released = True

    def __del__(self):
        self.release()


def _reserve_device_retention(size: int) -> Optional[_DeviceRetentionLease]:
    global _DEVICE_RETENTION_BYTES
    with _DEVICE_RETENTION_LOCK:
        if _DEVICE_RETENTION_BYTES + size > _DEVICE_RETENTION_MAX_BYTES:
            return None
        _DEVICE_RETENTION_BYTES += size
    return _DeviceRetentionLease(size)


def top_k_compute_bucket(top_k: int) -> int:
    """Round a requested candidate K up to one of four compute widths."""

    requested = int(top_k)
    if requested < 0:
        raise ValueError("constraint metric top_k must be non-negative")
    for bucket in TOP_K_COMPUTE_BUCKETS:
        if requested <= bucket:
            return bucket
    raise ValueError(
        f"constraint metric top_k {requested} exceeds maximum compute width "
        f"{TOP_K_COMPUTE_BUCKETS[-1]}"
    )


def uniform_top_k_ranges(top_ks: Sequence[int]) -> List[tuple[int, int, int]]:
    """Return contiguous ``[start, end)`` ranges with one bucketed width.

    Callers sort request rows by requested ``kernel_top_k`` before capture. Rows
    retain those requested values, while each returned width is rounded to
    ``{0, 8, 16, 32}`` for at most four fused launches and two D2H copies per
    active bucket. Final row materialization still truncates candidates to each
    row's requested K.
    """
    if not top_ks:
        return []
    ranges: List[tuple[int, int, int]] = []
    start = 0
    previous_requested = int(top_ks[0])
    current = top_k_compute_bucket(previous_requested)
    for index, value in enumerate(top_ks[1:], start=1):
        value = int(value)
        if value < previous_requested:
            raise ValueError("constraint metric rows must be sorted by top_k")
        previous_requested = value
        bucket = top_k_compute_bucket(value)
        if bucket == current:
            continue
        ranges.append((start, index, current))
        start = index
        current = bucket
    ranges.append((start, len(top_ks), current))
    return ranges


def _resolve_max_top_k(top_ks: Sequence[int], max_top_k: Optional[int]) -> int:
    """Resolve compute width, trusting explicit widths from uniform row ranges."""

    if max_top_k is None:
        return max((int(top_k) for top_k in top_ks), default=0)

    requested_max_top_k = int(max_top_k)
    actual_max_top_k = int(top_ks[-1]) if top_ks else 0
    if requested_max_top_k < actual_max_top_k:
        raise ValueError(
            f"compute max_top_k {requested_max_top_k} is smaller than requested "
            f"row top_k {actual_max_top_k}"
        )
    return requested_max_top_k


@dataclass
class ConstraintMetricsPackedRows:
    """Host scalar rows before expansion into the public record layout."""

    floats: Sequence[Sequence[Any]]
    ints: Sequence[Sequence[Any]]
    grammar_active: List[bool]
    output_indices: List[int]

    def select(self, row_indices: Sequence[int]) -> ConstraintMetricsPackedRows:
        indices = np.asarray(row_indices, dtype=np.int64)
        return ConstraintMetricsPackedRows(
            floats=np.asarray(self.floats)[indices],
            ints=np.asarray(self.ints)[indices],
            grammar_active=[self.grammar_active[index] for index in row_indices],
            output_indices=[self.output_indices[index] for index in row_indices],
        )


@dataclass
class ConstraintMetricsTensorBatch:
    """Packed device/host tensor columns for a batch of forensic records.

    Runtime keeps these tensors on device until ``GenerationBatchResult`` uses its
    existing async result-copy stream. Float metrics and top-k probabilities share
    one tensor; integer metrics, booleans, and top-k ids share another. This keeps
    the hot-path transfer to two D2H operations. ``to_packed_rows`` keeps the
    scheduler path flat; ``to_columns`` crosses directly into public response
    columns without first allocating an intermediate row tree.
    """

    floats: torch.Tensor
    ints: torch.Tensor
    top_ks: List[int]
    grammar_active: List[bool]
    output_indices: List[int]
    vocab_size: int

    def _ensure_host(self) -> None:
        if self.floats.device.type == "cpu":
            return
        self.floats = self.floats.to("cpu")
        self.ints = self.ints.to("cpu")

    def to_packed_rows(
        self,
        row_indices: Sequence[int],
        *,
        output_indices: Sequence[int],
    ) -> ConstraintMetricsPackedRows:
        """Select scalar host rows for summary and streaming consumers."""

        self._ensure_host()
        if len(row_indices) == 0 or len(output_indices) != len(row_indices):
            raise ValueError(
                "constraint metric row selections must be non-empty and aligned"
            )
        index = torch.tensor(row_indices, dtype=torch.long)
        float_tensor = self.floats[:, : layout.KERNEL_ROW_FLOAT_COUNT].index_select(
            0, index
        )
        int_tensor = self.ints[:, : layout.KERNEL_ROW_INT_COUNT].index_select(0, index)
        return ConstraintMetricsPackedRows(
            floats=float_tensor.numpy(),
            ints=int_tensor.numpy(),
            grammar_active=[self.grammar_active[i] for i in row_indices],
            output_indices=list(output_indices),
        )

    @staticmethod
    def empty_columns(*, width: int = 0, include_tables: bool = True) -> Dict[str, Any]:
        """Return an empty token-metric column object."""

        columns = {
            "count": 0,
            "output_index": [],
            "token_id": [],
            "grammar_active": [],
            "effective_legal_token_count": [],
            "constraint_pressure_nats": [],
            "argmax_masked": [],
            "unconstrained_top1": {
                "token_id": [],
                "prob": [],
                "logit_gap_to_top2_nats": [],
            },
            "entropy_nats": [],
            "emitted": {
                "allowed": [],
                "model_log_prob": [],
            },
            "flags": {
                "grammar_inactive": [],
                "empty_support": [],
                "all_allow": [],
                "eos_only_support": [],
                "has_non_finite_logits": [],
            },
            "spec_source": [],
        }
        if include_tables:
            columns["emitted"]["rank"] = []
            columns["top_masked"] = {
                "width": width,
                "count": [],
                "token_id": [],
                "prob": [],
            }
            columns["top_legal_after_conditioning"] = {
                "width": width,
                "count": [],
                "token_id": [],
                "constrained_prob": [],
            }
        return columns

    def to_columns(
        self,
        *,
        row_start: int = 0,
        row_count: Optional[int] = None,
        output_index_start: Optional[int] = None,
        summary_accumulator: Optional[ConstraintMetricsSummaryAccumulator] = None,
        include_tables: bool = True,
    ) -> Dict[str, Any]:
        """Materialize forensic token metrics as JSON-friendly columns.

        Candidate tables use ``width`` as a fixed row-major stride. For row
        ``i``, only ``[i * width : i * width + count[i]]`` is semantic; callers
        must ignore the remaining stride slots. Flag columns use
        ``CONSTRAINT_METRIC_FLAG_ORDER`` when reconstructed as ordered names.
        """

        self._ensure_host()
        available_rows = self.floats.shape[0]
        if row_start < 0 or row_start > available_rows:
            raise ValueError("constraint metric row_start is out of range")
        if row_count is None:
            selected_rows = available_rows - row_start
        else:
            if row_count < 0:
                raise ValueError("constraint metric row_count must be non-negative")
            selected_rows = int(row_count)
            if selected_rows > available_rows - row_start:
                raise ValueError("constraint metric row_count is out of range")

        max_k = (
            self.floats.shape[1] - layout.KERNEL_ROW_FLOAT_COUNT
        ) // layout.TOP_K_FLOAT_GROUP_COUNT
        if selected_rows == 0:
            return self.empty_columns(width=max_k, include_tables=include_tables)

        row_end = row_start + selected_rows
        selected_floats = self.floats[row_start:row_end]
        selected_ints = self.ints[row_start:row_end]
        float_columns = selected_floats[:, : layout.KERNEL_ROW_FLOAT_COUNT].T.tolist()
        int_columns = selected_ints[:, : layout.KERNEL_ROW_INT_COUNT].T.tolist()
        ids_start = layout.KERNEL_ROW_INT_COUNT
        receiver_ids_start = ids_start + max_k
        masked_probs_start = layout.KERNEL_ROW_FLOAT_COUNT
        receiver_cprobs_start = masked_probs_start + max_k
        output_indices = (
            list(self.output_indices[row_start:row_end])
            if output_index_start is None
            else list(range(output_index_start, output_index_start + selected_rows))
        )
        token_ids = int_columns[layout.KERNEL_INT_TOKEN_ID]
        grammar_active = [
            bool(value) for value in self.grammar_active[row_start:row_end]
        ]
        effective_legal_token_count = int_columns[
            layout.KERNEL_INT_EFFECTIVE_LEGAL_TOKEN_COUNT
        ]
        blocked_token_count = int_columns[layout.KERNEL_INT_BLOCKED_TOKEN_COUNT]
        empty_support = [
            bool(value) for value in int_columns[layout.KERNEL_INT_EMPTY_SUPPORT]
        ]
        pressure = [
            INF_PRESSURE_SENTINEL if empty else float(value)
            for empty, value in zip(
                empty_support,
                float_columns[layout.FLOAT_PRESSURE],
                strict=True,
            )
        ]
        argmax_masked = [
            bool(value) for value in int_columns[layout.KERNEL_INT_ARGMAX_MASKED]
        ]
        top1_token_ids = int_columns[layout.KERNEL_INT_UNCONSTRAINED_TOP1_TOKEN_ID]
        top1_logit_gap = float_columns[layout.FLOAT_TOP1_LOGIT_GAP]
        entropy = float_columns[layout.FLOAT_ENTROPY]
        eos_only_support = [
            bool(value) for value in int_columns[layout.KERNEL_INT_EOS_ONLY_SUPPORT]
        ]
        has_non_finite_logits = [
            bool(value)
            for value in int_columns[layout.KERNEL_INT_HAS_NON_FINITE_LOGITS]
        ]
        all_allow = [
            active and not empty and blocked == 0
            for active, empty, blocked in zip(
                grammar_active,
                empty_support,
                blocked_token_count,
                strict=True,
            )
        ]
        spec_sources = [
            layout.SPEC_SOURCE_NAMES.get(int(value))
            for value in int_columns[layout.KERNEL_INT_SPEC_SOURCE]
        ]
        requested_top_ks = self.top_ks[row_start:row_end]
        masked_counts = [
            min(int(top_k), int(blocked), max_k)
            for top_k, blocked in zip(
                requested_top_ks,
                blocked_token_count,
                strict=True,
            )
        ]
        legal_counts = [
            min(int(top_k), int(legal), max_k)
            for top_k, legal in zip(
                requested_top_ks,
                effective_legal_token_count,
                strict=True,
            )
        ]
        flag_columns = {
            "grammar_inactive": [not active for active in grammar_active],
            "empty_support": empty_support,
            "all_allow": all_allow,
            "eos_only_support": eos_only_support,
            "has_non_finite_logits": has_non_finite_logits,
        }

        if summary_accumulator is not None:
            extend_kernel_row_segments(
                [summary_accumulator],
                selected_floats[:, : layout.KERNEL_ROW_FLOAT_COUNT].numpy(),
                selected_ints[:, : layout.KERNEL_ROW_INT_COUNT].numpy(),
                grammar_active,
                output_indices,
                [selected_rows],
                [False],
            )

        columns = {
            "count": selected_rows,
            "output_index": output_indices,
            "token_id": token_ids,
            "grammar_active": grammar_active,
            "effective_legal_token_count": effective_legal_token_count,
            "constraint_pressure_nats": pressure,
            "argmax_masked": argmax_masked,
            "unconstrained_top1": {
                "token_id": top1_token_ids,
                "prob": float_columns[layout.FLOAT_UNCONSTRAINED_TOP1_PROB],
                "logit_gap_to_top2_nats": top1_logit_gap,
            },
            "entropy_nats": entropy,
            "emitted": {
                "allowed": [
                    bool(value)
                    for value in int_columns[layout.KERNEL_INT_EMITTED_ALLOWED]
                ],
                "model_log_prob": float_columns[layout.FLOAT_EMITTED_MODEL_LOG_PROB],
            },
            "flags": flag_columns,
            "spec_source": spec_sources,
        }
        if include_tables:
            columns["emitted"]["rank"] = int_columns[layout.KERNEL_INT_EMITTED_RANK]
            columns["top_masked"] = {
                "width": max_k,
                "count": masked_counts,
                "token_id": selected_ints[:, ids_start:receiver_ids_start]
                .reshape(-1)
                .tolist(),
                "prob": selected_floats[:, masked_probs_start:receiver_cprobs_start]
                .reshape(-1)
                .tolist(),
            }
            columns["top_legal_after_conditioning"] = {
                "width": max_k,
                "count": legal_counts,
                "token_id": selected_ints[
                    :, receiver_ids_start : receiver_ids_start + max_k
                ]
                .reshape(-1)
                .tolist(),
                "constrained_prob": selected_floats[
                    :, receiver_cprobs_start : receiver_cprobs_start + max_k
                ]
                .reshape(-1)
                .tolist(),
            }
        return columns


@dataclass
class ConstraintMetricsRecordPayload:
    """Tensor batch plus contiguous record spans for each request batch row."""

    tensor_batch: ConstraintMetricsTensorBatch
    row_spans: List[tuple[int, int, int]]
    speculative: bool = False
    defer_commit_validation: bool = False
    stream_to_host: bool = False
    _span_by_batch_index: Optional[Dict[int, tuple[int, int]]] = field(
        default=None, init=False, repr=False
    )
    _commit_columns: Optional[tuple[List[int], List[int]]] = field(
        default=None, init=False, repr=False
    )
    _device_retention: Optional[_DeviceRetentionLease] = field(
        default=None, init=False, repr=False
    )
    # Scheduler result objects can outlive terminal response assembly. Track the
    # request histories that still need this source so VRAM release is deterministic.
    _committed_slice_count: int = field(default=0, init=False, repr=False)

    @property
    def device_retained(self) -> bool:
        return self._device_retention is not None

    def move_to_pageable_storage(self) -> None:
        batch = self.tensor_batch
        batch.floats = _move_to_pageable_storage(batch.floats)
        batch.ints = _move_to_pageable_storage(batch.ints)

    def acquire_committed_slice(self) -> None:
        self._committed_slice_count += 1

    def release_committed_slice(self) -> None:
        if self._committed_slice_count <= 0:
            raise RuntimeError("constraint metric payload slice count underflow")
        self._committed_slice_count -= 1
        self.release_device_storage_if_unreferenced()

    def release_device_storage_if_unreferenced(self) -> None:
        if self._committed_slice_count or self._device_retention is None:
            return
        self._device_retention.release()
        self._device_retention = None
        batch = self.tensor_batch
        batch.floats = torch.empty((0, batch.floats.shape[1]), dtype=batch.floats.dtype)
        batch.ints = torch.empty((0, batch.ints.shape[1]), dtype=batch.ints.dtype)
        batch.top_ks.clear()
        batch.grammar_active.clear()
        batch.output_indices.clear()

    def span_for(self, batch_index: int) -> Optional[tuple[int, int]]:
        if self._span_by_batch_index is None:
            self._span_by_batch_index = {
                candidate: (start, count) for candidate, start, count in self.row_spans
            }
            if len(self._span_by_batch_index) != len(self.row_spans):
                raise ValueError("constraint metric payload has duplicate row spans")
        return self._span_by_batch_index.get(batch_index)

    def commit(
        self,
        batch_index: int,
        *,
        output_index: int,
        token_ids: Sequence[int],
    ) -> tuple[Optional[CommittedRecordSlice], Optional[int]]:
        span = self.span_for(batch_index)
        if span is None:
            return None, output_index
        start, available = span
        count = len(token_ids)
        if count > available:
            raise ValueError("committed rows exceed the captured row span")
        if count == 0:
            return None, None

        matching = count
        if not self.defer_commit_validation:
            if self._commit_columns is None:
                self.tensor_batch._ensure_host()
                self._commit_columns = (
                    self.tensor_batch.ints[:, layout.KERNEL_INT_TOKEN_ID].tolist(),
                    self.tensor_batch.ints[:, layout.KERNEL_INT_SPEC_SOURCE].tolist(),
                )
            captured_token_ids, spec_sources = self._commit_columns
            matching = 0
            for captured_token_id, token_id in zip(
                captured_token_ids[start : start + count], token_ids, strict=True
            ):
                if captured_token_id != int(token_id):
                    break
                matching += 1
            if (
                self.speculative
                and layout.SPEC_SOURCE_NORMAL in spec_sources[start : start + matching]
            ):
                raise ValueError("committed speculative rows are not a prefix")
        record_slice = (
            CommittedRecordSlice(
                payload=self,
                start=start,
                count=matching,
                output_index_start=output_index,
            )
            if matching
            else None
        )
        first_mismatch = output_index + matching if matching < count else None
        return record_slice, first_mismatch

    def packed_rows_for_slices(
        self,
        record_slices: Sequence[CommittedRecordSlice],
    ) -> ConstraintMetricsPackedRows:
        row_indices = []
        output_indices = []
        for record_slice in record_slices:
            if record_slice.payload is not self:
                raise ValueError("record slice belongs to a different payload")
            row_indices.extend(range(record_slice.start, record_slice.end))
            output_indices.extend(
                range(
                    record_slice.output_index_start,
                    record_slice.output_index_start + record_slice.count,
                )
            )
        return self.tensor_batch.to_packed_rows(
            row_indices,
            output_indices=output_indices,
        )


def copy_record_payload_rows(
    payload: ConstraintMetricsRecordPayload,
    row_indices: Sequence[int],
    output_indices: Sequence[int],
) -> ConstraintMetricsRecordPayload:
    """Copy selected host rows into an independently owned compact payload."""

    if not row_indices or len(row_indices) != len(output_indices):
        raise ValueError(
            "selected constraint metric rows must be non-empty and aligned"
        )
    batch = payload.tensor_batch
    batch._ensure_host()
    index = torch.tensor(row_indices, dtype=torch.long)
    return ConstraintMetricsRecordPayload(
        tensor_batch=ConstraintMetricsTensorBatch(
            floats=batch.floats.index_select(0, index),
            ints=batch.ints.index_select(0, index),
            top_ks=[batch.top_ks[row] for row in row_indices],
            grammar_active=[batch.grammar_active[row] for row in row_indices],
            output_indices=list(output_indices),
            vocab_size=batch.vocab_size,
        ),
        row_spans=[],
        speculative=payload.speculative,
    )


def retain_record_payload_on_device(
    payload: ConstraintMetricsRecordPayload,
    *,
    enabled: bool,
) -> ConstraintMetricsRecordPayload:
    """Retain one fresh kernel output in bounded request-lifetime device storage."""

    payload.defer_commit_validation = False
    batch = payload.tensor_batch
    if not enabled or not batch.floats.is_cuda:
        return payload
    retained_bytes = (
        batch.floats.numel() * batch.floats.element_size()
        + batch.ints.numel() * batch.ints.element_size()
    )
    lease = _reserve_device_retention(retained_bytes)
    if lease is None:
        return payload
    payload._device_retention = lease
    payload.defer_commit_validation = True
    return payload


def build_record_payload(
    tensor_batch: ConstraintMetricsTensorBatch,
    batch_indices: Sequence[int],
    *,
    rows_per_request: int,
    retention_flags: Sequence[bool],
    streaming_flags: Sequence[bool],
    speculative: bool = False,
) -> ConstraintMetricsRecordPayload:
    payload = ConstraintMetricsRecordPayload(
        tensor_batch=tensor_batch,
        row_spans=[
            (batch_index, row_index * rows_per_request, rows_per_request)
            for row_index, batch_index in enumerate(batch_indices)
        ],
        speculative=speculative,
        stream_to_host=any(streaming_flags),
    )
    return retain_record_payload_on_device(payload, enabled=all(retention_flags))


@dataclass(frozen=True)
class CommittedRecordSlice:
    payload: ConstraintMetricsRecordPayload
    start: int
    count: int
    output_index_start: int

    @property
    def end(self) -> int:
        return self.start + self.count

    def validated_output_index_end(self, expected_token_ids: Sequence[int]) -> int:
        batch = self.payload.tensor_batch
        batch._ensure_host()
        row_count = batch.floats.shape[0]
        if (
            batch.ints.shape[0] != row_count
            or len(batch.top_ks) != row_count
            or len(batch.grammar_active) != row_count
            or len(batch.output_indices) != row_count
        ):
            raise ValueError("constraint metric payload row metadata is inconsistent")
        if self.start < 0 or self.count < 0 or self.end > row_count:
            raise ValueError("constraint metric record slice is out of range")
        token_ids = batch.ints[
            self.start : self.end, layout.KERNEL_INT_TOKEN_ID
        ].tolist()
        spec_sources = (
            batch.ints[self.start : self.end, layout.KERNEL_INT_SPEC_SOURCE].tolist()
            if self.payload.speculative
            else None
        )
        for offset, captured_token_id in enumerate(token_ids):
            output_index = self.output_index_start + offset
            if output_index >= len(expected_token_ids) or captured_token_id != int(
                expected_token_ids[output_index]
            ):
                return output_index
            if (
                spec_sources is not None
                and spec_sources[offset] == layout.SPEC_SOURCE_NORMAL
            ):
                raise ValueError("committed speculative rows are not a prefix")
        return self.output_index_start + self.count


def _narrow_candidate_columns(
    floats: torch.Tensor,
    ints: torch.Tensor,
    *,
    target_top_k: int,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_float_top_k = (
        floats.shape[1] - layout.KERNEL_ROW_FLOAT_COUNT
    ) // layout.TOP_K_FLOAT_GROUP_COUNT
    source_int_top_k = (
        ints.shape[1] - layout.KERNEL_ROW_INT_COUNT
    ) // layout.TOP_K_INT_GROUP_COUNT
    if source_float_top_k != source_int_top_k:
        raise ValueError("packed constraint metric candidate widths do not match")
    target_top_k = min(int(target_top_k), vocab_size)
    if source_float_top_k < target_top_k:
        raise ValueError("packed constraint metric payload is narrower than requested")
    if source_float_top_k == target_top_k:
        return floats, ints

    float_base = layout.KERNEL_ROW_FLOAT_COUNT
    int_base = layout.KERNEL_ROW_INT_COUNT
    return (
        torch.cat(
            (
                floats[:, :float_base],
                floats[:, float_base : float_base + target_top_k],
                floats[
                    :,
                    float_base
                    + source_float_top_k : float_base
                    + source_float_top_k
                    + target_top_k,
                ],
            ),
            dim=1,
        ),
        torch.cat(
            (
                ints[:, :int_base],
                ints[:, int_base : int_base + target_top_k],
                ints[
                    :,
                    int_base
                    + source_int_top_k : int_base
                    + source_int_top_k
                    + target_top_k,
                ],
            ),
            dim=1,
        ),
    )


def compact_record_payloads(
    payloads: Sequence[CommittedRecordSlice],
    *,
    max_top_k: int,
) -> Optional[ConstraintMetricsTensorBatch]:
    """Pack one request's rows without crossing the tensor-to-Python boundary."""

    float_parts = []
    int_parts = []
    top_ks: List[int] = []
    grammar_active: List[bool] = []
    output_indices: List[int] = []
    vocab_size = None

    for record_slice in payloads:
        payload = record_slice.payload
        start = record_slice.start
        count = record_slice.count
        batch = payload.tensor_batch
        batch._ensure_host()
        row_count = batch.floats.shape[0]
        if (
            batch.ints.shape[0] != row_count
            or len(batch.top_ks) != row_count
            or len(batch.grammar_active) != row_count
            or len(batch.output_indices) != row_count
        ):
            raise ValueError("constraint metric payload row metadata is inconsistent")
        if start < 0 or count < 0 or start + count > row_count:
            raise ValueError("constraint metric record slice is out of range")
        if vocab_size is None:
            vocab_size = batch.vocab_size
        elif vocab_size != batch.vocab_size:
            raise ValueError("constraint metric payload vocab sizes do not match")

        if count == 0:
            continue

        end = start + count
        selected_floats, selected_ints = _narrow_candidate_columns(
            batch.floats[start:end],
            batch.ints[start:end],
            target_top_k=max_top_k,
            vocab_size=batch.vocab_size,
        )

        indices = range(start, end)
        float_parts.append(selected_floats)
        int_parts.append(selected_ints)
        top_ks.extend(batch.top_ks[index] for index in indices)
        grammar_active.extend(batch.grammar_active[index] for index in indices)
        output_indices.extend(
            range(
                record_slice.output_index_start,
                record_slice.output_index_start + count,
            )
        )

    if not float_parts:
        return None
    return ConstraintMetricsTensorBatch(
        floats=float_parts[0] if len(float_parts) == 1 else torch.cat(float_parts),
        ints=int_parts[0] if len(int_parts) == 1 else torch.cat(int_parts),
        top_ks=top_ks,
        grammar_active=grammar_active,
        output_indices=output_indices,
        vocab_size=int(vocab_size),
    )


def _archive_segment_tensors(
    segments: Sequence[tuple[int, int, list[CommittedRecordSlice]]],
    *,
    target_top_k: int,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    payload_entries: Dict[
        int, tuple[ConstraintMetricsRecordPayload, list[CommittedRecordSlice]]
    ] = {}
    selected_rows = 0
    full_rows = 0
    for _, _, record_slices in segments:
        for record_slice in record_slices:
            payload = record_slice.payload
            entry = payload_entries.get(id(payload))
            if entry is None:
                entry = (payload, [])
                payload_entries[id(payload)] = entry
                full_rows += payload.tensor_batch.floats.shape[0]
            entry[1].append(record_slice)
            selected_rows += record_slice.count
    use_full_payloads = full_rows <= _ARCHIVE_FULL_PAYLOAD_MAX_ROW_RATIO * selected_rows

    float_parts = []
    int_parts = []
    slice_source_starts: Dict[int, int] = {}
    source_row_offset = 0
    for payload, record_slices in payload_entries.values():
        batch = payload.tensor_batch
        if use_full_payloads:
            float_parts.append(batch.floats)
            int_parts.append(batch.ints)
            for record_slice in record_slices:
                slice_source_starts[id(record_slice)] = (
                    source_row_offset + record_slice.start
                )
            source_row_offset += batch.floats.shape[0]
            continue

        sorted_slices = sorted(record_slices, key=lambda item: item.start)
        row_indices = [
            row_index
            for record_slice in sorted_slices
            for row_index in range(record_slice.start, record_slice.end)
        ]
        index = async_index_to_device(
            row_indices,
            batch.floats.device,
            dtype=torch.long,
        )
        float_parts.append(batch.floats.index_select(0, index))
        int_parts.append(batch.ints.index_select(0, index))
        selected_offset = 0
        for record_slice in sorted_slices:
            slice_source_starts[id(record_slice)] = source_row_offset + selected_offset
            selected_offset += record_slice.count
        source_row_offset += selected_offset

    source_floats = torch.cat(float_parts, dim=0)
    source_ints = torch.cat(int_parts, dim=0)
    selected_indices = [
        row_index
        for _, _, record_slices in segments
        for record_slice in record_slices
        for row_index in range(
            slice_source_starts[id(record_slice)],
            slice_source_starts[id(record_slice)] + record_slice.count,
        )
    ]
    index = async_index_to_device(
        selected_indices,
        source_floats.device,
        dtype=torch.long,
    )
    archived_floats = source_floats.index_select(0, index)
    archived_ints = source_ints.index_select(0, index)
    return _narrow_candidate_columns(
        archived_floats,
        archived_ints,
        target_top_k=target_top_k,
        vocab_size=vocab_size,
    )


def archive_record_payloads(
    histories: Sequence[Sequence[CommittedRecordSlice]],
) -> list[list[CommittedRecordSlice]]:
    """Archive the union of finishing-request spans once for terminal IPC."""

    referenced_slices: Dict[int, list[CommittedRecordSlice]] = {}
    segments_by_key: Dict[
        tuple[Any, ...],
        list[tuple[int, int, list[CommittedRecordSlice]]],
    ] = {}
    segment_counts = []
    for history_index, history in enumerate(histories):
        segments: list[tuple[tuple[Any, ...], list[CommittedRecordSlice]]] = []
        for record_slice in history:
            payload = record_slice.payload
            batch = payload.tensor_batch
            start = record_slice.start
            end = record_slice.end
            target_top_k = max(batch.top_ks[start:end], default=0)
            key = (
                batch.floats.shape[1],
                batch.ints.shape[1],
                batch.vocab_size,
                payload.speculative,
                target_top_k,
                str(batch.floats.device),
                str(batch.ints.device),
                batch.floats.dtype,
                batch.ints.dtype,
            )
            if not segments or segments[-1][0] != key:
                segments.append((key, []))
            segments[-1][1].append(record_slice)
            referenced = referenced_slices.get(id(payload))
            if referenced is None:
                referenced_slices[id(payload)] = [record_slice]
            else:
                referenced.append(record_slice)
        segment_counts.append(len(segments))
        for segment_index, (key, record_slices) in enumerate(segments):
            segments_by_key.setdefault(key, []).append(
                (history_index, segment_index, record_slices)
            )

    for spans in referenced_slices.values():
        previous_end = None
        for record_slice in sorted(spans, key=lambda item: item.start):
            if previous_end is not None and record_slice.start < previous_end:
                raise ValueError("constraint metric payload spans overlap")
            previous_end = record_slice.end

    archived_segments: Dict[tuple[int, int], CommittedRecordSlice] = {}
    for key, segments in segments_by_key.items():
        top_ks: List[int] = []
        grammar_active: List[bool] = []
        output_indices: List[int] = []
        archived_locations = []
        row_offset = 0
        for history_index, segment_index, record_slices in segments:
            output_index_start = record_slices[0].output_index_start
            expected_output_index = output_index_start
            segment_count = 0
            for record_slice in record_slices:
                if record_slice.output_index_start != expected_output_index:
                    raise ValueError("constraint metric history is not contiguous")
                batch = record_slice.payload.tensor_batch
                start = record_slice.start
                end = record_slice.end
                top_ks.extend(batch.top_ks[start:end])
                grammar_active.extend(batch.grammar_active[start:end])
                output_indices.extend(
                    range(
                        record_slice.output_index_start,
                        record_slice.output_index_start + record_slice.count,
                    )
                )
                segment_count += record_slice.count
                expected_output_index += record_slice.count
            archived_locations.append(
                (
                    history_index,
                    segment_index,
                    row_offset,
                    segment_count,
                    output_index_start,
                )
            )
            row_offset += segment_count
        archived_floats, archived_ints = _archive_segment_tensors(
            segments,
            target_top_k=int(key[4]),
            vocab_size=int(key[2]),
        )
        archived_payload = ConstraintMetricsRecordPayload(
            tensor_batch=ConstraintMetricsTensorBatch(
                floats=archived_floats,
                ints=archived_ints,
                top_ks=top_ks,
                grammar_active=grammar_active,
                output_indices=output_indices,
                vocab_size=int(key[2]),
            ),
            row_spans=[],
            speculative=bool(key[3]),
        )
        for (
            history_index,
            segment_index,
            start,
            count,
            output_index_start,
        ) in archived_locations:
            archived_segments[(history_index, segment_index)] = CommittedRecordSlice(
                payload=archived_payload,
                start=start,
                count=count,
                output_index_start=output_index_start,
            )

    return [
        [archived_segments[(history_index, index)] for index in range(segment_count)]
        for history_index, segment_count in enumerate(segment_counts)
    ]


def move_record_payloads_to_host(
    histories: Sequence[Sequence[CommittedRecordSlice]],
) -> None:
    """Move each unique terminal archive to owned CPU storage before IPC."""

    seen = set()
    pending = []
    streams = {}
    try:
        for history in histories:
            for record_slice in history:
                payload = record_slice.payload
                if id(payload) in seen:
                    continue
                seen.add(id(payload))
                batch = payload.tensor_batch
                host_floats = None
                host_ints = None
                if batch.floats.is_cuda:
                    host_floats = torch.empty(
                        batch.floats.shape,
                        dtype=batch.floats.dtype,
                        pin_memory=True,
                    )
                    stream = torch.cuda.current_stream(batch.floats.device)
                    streams[batch.floats.device] = stream
                    host_floats.copy_(batch.floats, non_blocking=True)
                if batch.ints.is_cuda:
                    host_ints = torch.empty(
                        batch.ints.shape,
                        dtype=batch.ints.dtype,
                        pin_memory=True,
                    )
                    stream = torch.cuda.current_stream(batch.ints.device)
                    streams[batch.ints.device] = stream
                    host_ints.copy_(batch.ints, non_blocking=True)
                if host_floats is not None or host_ints is not None:
                    pending.append((batch, host_floats, host_ints))
    finally:
        for stream in streams.values():
            stream.synchronize()
    for batch, host_floats, host_ints in pending:
        if host_floats is not None:
            batch.floats = _move_to_pageable_storage(host_floats)
        if host_ints is not None:
            batch.ints = _move_to_pageable_storage(host_ints)


def _compute_record_tensor_batch(
    pre_logits: torch.Tensor,
    grammar_mask: torch.Tensor,
    emitted_ids: torch.Tensor,
    *,
    temperatures: Optional[torch.Tensor] = None,
    top_ks: List[int],
    grammar_active: List[bool],
    output_indices: List[int],
    eos_token_ids: Optional[Sequence[int]] = None,
    spec_sources: Optional[torch.Tensor] = None,
    row_enabled: Optional[torch.Tensor] = None,
    grammar_active_device: Optional[torch.Tensor] = None,
    max_top_k: Optional[int] = None,
    grammar_mask_kind: int = MASK_DENSE_ALLOWED,
) -> ConstraintMetricsTensorBatch:
    """Compute packed forensic columns with the fused Triton implementation."""
    n_rows, vocab_size = pre_logits.shape
    if n_rows == 0:
        raise ValueError("record tensor batch must contain at least one row")
    from sglang_constraint_metrics.kernel import (
        compute_constraint_metrics,
    )

    device = pre_logits.device
    if device.type != "cuda":
        raise ValueError("constraint-metrics record math requires CUDA tensors")
    if temperatures is None:
        temperatures_device = torch.ones(n_rows, dtype=torch.float32, device=device)
    else:
        temperatures_device = async_to_device(
            temperatures, device, dtype=torch.float32
        ).reshape(n_rows)
    requested_max_top_k = _resolve_max_top_k(top_ks, max_top_k)
    global_max_top_k = min(requested_max_top_k, vocab_size)
    emitted_device = async_to_device(emitted_ids, device, dtype=torch.long)
    if grammar_active_device is None:
        grammar_active_device = async_index_to_device(
            grammar_active, device, dtype=torch.bool
        )
    elif grammar_active_device.device != device:
        raise ValueError("grammar_active_device must match the logits device")
    grammar_active_device = grammar_active_device.reshape(n_rows)
    if spec_sources is None:
        spec_sources_device = torch.zeros(n_rows, dtype=torch.long, device=device)
    else:
        spec_sources_device = async_to_device(
            spec_sources, device, dtype=torch.long
        ).reshape(n_rows)
    floats, ints = compute_constraint_metrics(
        pre_logits.detach().to(torch.float32),
        grammar_mask,
        temperatures_device,
        emitted_device,
        grammar_active_device,
        spec_sources_device,
        row_enabled=row_enabled,
        max_top_k=global_max_top_k,
        eos_token_ids=eos_token_ids,
        sanitize_fill=_SANITIZE_FILL,
        inf_pressure=INF_PRESSURE_SENTINEL,
        grammar_mask_kind=grammar_mask_kind,
    )
    return ConstraintMetricsTensorBatch(
        floats=floats,
        ints=ints,
        top_ks=top_ks,
        grammar_active=grammar_active,
        output_indices=output_indices,
        vocab_size=vocab_size,
    )


def _compute_record_tensor_batch_chunked(
    pre_logits: torch.Tensor,
    grammar_mask: torch.Tensor,
    emitted_ids: torch.Tensor,
    *,
    temperatures: Optional[torch.Tensor] = None,
    top_ks: List[int],
    grammar_active: List[bool],
    output_indices: List[int],
    eos_token_ids: Optional[Sequence[int]] = None,
    spec_sources: Optional[torch.Tensor] = None,
    row_enabled: Optional[torch.Tensor] = None,
    grammar_active_device: Optional[torch.Tensor] = None,
    max_vocab_elements: int = _SPECULATIVE_DENSE_DEFER_MAX_WORK_ELEMENTS,
    max_top_k: Optional[int] = None,
    grammar_mask_kind: int = MASK_DENSE_ALLOWED,
) -> ConstraintMetricsTensorBatch:
    """Run dense record math in bounded contiguous chunks.

    Speculative decoding stages every verify position without reading acceptance lengths on the
    host, then the fused kernel skips uncommitted rows. This bound prevents large
    verify batches from growing the scratch cache without limit.
    """
    total_rows, vocab_size = pre_logits.shape
    if total_rows == 0:
        raise ValueError("record tensor batch must contain at least one row")
    if not (len(top_ks) == len(grammar_active) == len(output_indices) == total_rows):
        raise ValueError("row metadata must match pre_logits")
    rows_per_chunk = max(1, int(max_vocab_elements) // max(1, vocab_size))
    compute_max_top_k = _resolve_max_top_k(top_ks, max_top_k)
    if pre_logits.device.type != "cuda":
        raise ValueError("constraint-metrics record math requires CUDA tensors")
    storage_top_k = min(compute_max_top_k, vocab_size)
    if total_rows <= rows_per_chunk:
        return _compute_record_tensor_batch(
            pre_logits,
            grammar_mask,
            emitted_ids,
            temperatures=temperatures,
            top_ks=top_ks,
            grammar_active=grammar_active,
            output_indices=output_indices,
            eos_token_ids=eos_token_ids,
            spec_sources=spec_sources,
            row_enabled=row_enabled,
            grammar_active_device=grammar_active_device,
            max_top_k=compute_max_top_k,
            grammar_mask_kind=grammar_mask_kind,
        )

    device = pre_logits.device
    floats = torch.empty(
        total_rows,
        layout.KERNEL_ROW_FLOAT_COUNT + layout.TOP_K_FLOAT_GROUP_COUNT * storage_top_k,
        dtype=torch.float32,
        device=device,
    )
    ints = torch.empty(
        total_rows,
        layout.KERNEL_ROW_INT_COUNT + layout.TOP_K_INT_GROUP_COUNT * storage_top_k,
        dtype=torch.int64,
        device=device,
    )
    for start in range(0, total_rows, rows_per_chunk):
        end = min(start + rows_per_chunk, total_rows)
        chunk_spec_sources = None if spec_sources is None else spec_sources[start:end]
        chunk_row_enabled = None if row_enabled is None else row_enabled[start:end]
        chunk_grammar_active_device = (
            None if grammar_active_device is None else grammar_active_device[start:end]
        )
        chunk_temperatures = None if temperatures is None else temperatures[start:end]
        batch = _compute_record_tensor_batch(
            pre_logits[start:end],
            grammar_mask[start:end],
            emitted_ids[start:end],
            temperatures=chunk_temperatures,
            top_ks=top_ks[start:end],
            grammar_active=grammar_active[start:end],
            output_indices=output_indices[start:end],
            eos_token_ids=eos_token_ids,
            spec_sources=chunk_spec_sources,
            row_enabled=chunk_row_enabled,
            grammar_active_device=chunk_grammar_active_device,
            max_top_k=compute_max_top_k,
            grammar_mask_kind=grammar_mask_kind,
        )
        floats[start:end].copy_(batch.floats)
        ints[start:end].copy_(batch.ints)
    return ConstraintMetricsTensorBatch(
        floats=floats,
        ints=ints,
        top_ks=top_ks,
        grammar_active=grammar_active,
        output_indices=output_indices,
        vocab_size=vocab_size,
    )


def packed_kernel_row_to_record(
    float_values: Sequence[Any],
    tensor_int_values: Sequence[Any],
    *,
    grammar_active: bool,
    output_index: int,
) -> Dict[str, Any]:
    """Expand one fused-kernel row after scheduler transport."""

    token_id = int(tensor_int_values[layout.KERNEL_INT_TOKEN_ID])
    effective_legal_token_count = int(
        tensor_int_values[layout.KERNEL_INT_EFFECTIVE_LEGAL_TOKEN_COUNT]
    )
    blocked_token_count = int(tensor_int_values[layout.KERNEL_INT_BLOCKED_TOKEN_COUNT])
    unconstrained_top1_token_id = int(
        tensor_int_values[layout.KERNEL_INT_UNCONSTRAINED_TOP1_TOKEN_ID]
    )
    empty_support = bool(tensor_int_values[layout.KERNEL_INT_EMPTY_SUPPORT])
    emitted = {
        "allowed": bool(tensor_int_values[layout.KERNEL_INT_EMITTED_ALLOWED]),
        "model_log_prob": float(float_values[layout.FLOAT_EMITTED_MODEL_LOG_PROB]),
    }

    flags: List[str] = []
    if not grammar_active:
        flags.append("grammar_inactive")
    if empty_support:
        flags.append("empty_support")
    elif grammar_active and blocked_token_count == 0:
        flags.append("all_allow")
    if bool(tensor_int_values[layout.KERNEL_INT_EOS_ONLY_SUPPORT]):
        flags.append("eos_only_support")
    if bool(tensor_int_values[layout.KERNEL_INT_HAS_NON_FINITE_LOGITS]):
        flags.append("has_non_finite_logits")

    record = {
        "output_index": int(output_index),
        "token_id": token_id,
        "grammar_active": bool(grammar_active),
        "effective_legal_token_count": effective_legal_token_count,
        # Preserve the exact public sentinel rather than its float32 rounding.
        "constraint_pressure_nats": (
            INF_PRESSURE_SENTINEL
            if empty_support
            else float(float_values[layout.FLOAT_PRESSURE])
        ),
        "argmax_masked": bool(tensor_int_values[layout.KERNEL_INT_ARGMAX_MASKED]),
        "unconstrained_top1": {
            "token_id": unconstrained_top1_token_id,
            "prob": float(float_values[layout.FLOAT_UNCONSTRAINED_TOP1_PROB]),
            "logit_gap_to_top2_nats": float(float_values[layout.FLOAT_TOP1_LOGIT_GAP]),
        },
        "entropy_nats": float(float_values[layout.FLOAT_ENTROPY]),
        "emitted": emitted,
    }
    record["flags"] = flags
    spec_source = layout.SPEC_SOURCE_NAMES.get(
        int(tensor_int_values[layout.KERNEL_INT_SPEC_SOURCE])
    )
    if spec_source is not None:
        record["spec_source"] = spec_source
    return record
