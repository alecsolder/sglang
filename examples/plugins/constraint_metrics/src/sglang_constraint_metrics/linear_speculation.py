"""Constraint-metrics capture for linear speculative decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from sglang_constraint_metrics import layout
from sglang_constraint_metrics.capture import (
    CaptureInputs,
    CaptureScratch,
    ConstraintMetricsCapturePlan,
    build_capture_plan,
    select_capture_inputs,
)
from sglang_constraint_metrics.mask import grammar_mask_kind
from sglang_constraint_metrics.observer import ConstraintMetricsDeviceOutput
from sglang_constraint_metrics.records import (
    _compute_record_tensor_batch_chunked,
    async_index_to_device,
    build_record_payload,
    uniform_top_k_ranges,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


@triton.jit(
    do_not_specialize=[
        "token_row_stride",
        "total_positions",
        "width",
    ],
    do_not_specialize_on_alignment=[
        "committed_token_ids_ptr",
        "token_row_stride",
        "total_positions",
        "width",
    ],
)
def _build_linear_record_metadata_kernel(
    committed_token_ids_ptr,
    commit_lens_ptr,
    emitted_ids_ptr,
    spec_sources_ptr,
    row_enabled_ptr,
    token_row_stride,
    total_positions,
    width,
    accepted_proposal_source: tl.constexpr,
    bonus_source: tl.constexpr,
    normal_source: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    flat_position = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = flat_position < total_positions
    row = flat_position // width
    position = flat_position - row * width
    commit_len = tl.load(commit_lens_ptr + row, mask=valid, other=0).to(tl.int32)
    is_draft = position + 1 < commit_len
    is_bonus = position + 1 == commit_len
    emitted_id = tl.load(
        committed_token_ids_ptr + row * token_row_stride + position,
        mask=valid & (position < commit_len),
        other=0,
    )
    spec_source = tl.where(
        is_draft,
        accepted_proposal_source,
        tl.where(is_bonus, bonus_source, normal_source),
    )
    tl.store(emitted_ids_ptr + flat_position, emitted_id, mask=valid)
    tl.store(spec_sources_ptr + flat_position, spec_source, mask=valid)
    tl.store(row_enabled_ptr + flat_position, position < commit_len, mask=valid)


def _build_linear_record_metadata(
    accept_tokens: torch.Tensor,
    num_accept_tokens: torch.Tensor,
    emitted_ids: torch.Tensor,
    spec_sources: torch.Tensor,
    row_enabled: torch.Tensor,
    *,
    width: int,
) -> None:
    rows = num_accept_tokens.shape[0]
    total_positions = rows * width
    if accept_tokens.device.type != "cuda":
        raise ValueError(
            "linear-speculation constraint-metrics metadata requires CUDA tensors"
        )
    if accept_tokens.shape != (rows, width):
        raise ValueError("committed tokens must match the speculative batch shape")
    if accept_tokens.stride(1) != 1:
        raise ValueError("committed tokens must be contiguous within each row")
    _build_linear_record_metadata_kernel[(triton.cdiv(total_positions, 256),)](
        accept_tokens,
        num_accept_tokens,
        emitted_ids,
        spec_sources,
        row_enabled,
        accept_tokens.stride(0),
        total_positions,
        width=width,
        accepted_proposal_source=layout.SPEC_SOURCE_ACCEPTED_DRAFT,
        bonus_source=layout.SPEC_SOURCE_BONUS,
        normal_source=layout.SPEC_SOURCE_NORMAL,
        BLOCK_SIZE=256,
    )


def warmup_linear_record_metadata(
    *, width: int, device: torch.device | str, scratch: CaptureScratch
) -> None:
    """Load the linear-speculation metadata kernel before serving starts."""
    if width <= 0:
        raise ValueError("linear speculation width must be positive")
    device = torch.device(device)
    accept_tokens = torch.zeros((1, width), dtype=torch.int64, device=device)
    _build_linear_record_metadata(
        accept_tokens,
        torch.ones(1, dtype=torch.int32, device=device),
        scratch.emitted_ids[:width],
        scratch.spec_sources[:width],
        scratch.row_enabled[:width],
        width=width,
    )


@dataclass
class LinearSpeculationCapture:
    plan: ConstraintMetricsCapturePlan
    inputs: CaptureInputs
    grammar_mask_kind: int
    grammar_active_device: torch.Tensor
    eos_token_ids: list[int]
    emitted_ids: torch.Tensor
    spec_sources: torch.Tensor
    row_enabled: torch.Tensor

    def finalize(
        self,
        *,
        accept_tokens: torch.Tensor,
        num_accept_tokens: torch.Tensor,
    ) -> ConstraintMetricsDeviceOutput:
        payloads = []
        plan = self.plan
        inputs = self.inputs
        _, width, vocab_size = inputs.pre.shape
        batch_size = num_accept_tokens.shape[0]
        if accept_tokens.shape != (batch_size, width):
            raise ValueError(
                "committed tokens must match the captured speculative batch"
            )
        diagnostic_indices_device = inputs.diagnostic_indices_device
        if diagnostic_indices_device is None:
            selected_accept_tokens = accept_tokens
            selected_num_accept_tokens = num_accept_tokens
        else:
            selected_accept_tokens = accept_tokens.index_select(
                0, diagnostic_indices_device
            )
            selected_num_accept_tokens = num_accept_tokens.index_select(
                0, diagnostic_indices_device
            )
        _build_linear_record_metadata(
            selected_accept_tokens,
            selected_num_accept_tokens,
            self.emitted_ids,
            self.spec_sources,
            self.row_enabled,
            width=width,
        )
        for start, end, max_top_k in uniform_top_k_ranges(plan.top_ks):
            group_rows = end - start
            position_start = start * width
            position_end = end * width
            tensor_batch = _compute_record_tensor_batch_chunked(
                inputs.pre[start:end].reshape(group_rows * width, vocab_size),
                inputs.grammar_mask[position_start:position_end],
                self.emitted_ids[position_start:position_end],
                temperatures=inputs.temperatures[start:end]
                .reshape(group_rows, 1)
                .expand(group_rows, width)
                .reshape(-1),
                top_ks=[
                    top_k for top_k in plan.top_ks[start:end] for _ in range(width)
                ],
                grammar_active=[
                    active
                    for active in plan.grammar_active[start:end]
                    for _ in range(width)
                ],
                grammar_active_device=self.grammar_active_device[
                    position_start:position_end
                ],
                output_indices=[0] * (group_rows * width),
                eos_token_ids=self.eos_token_ids,
                spec_sources=self.spec_sources[position_start:position_end],
                row_enabled=self.row_enabled[position_start:position_end],
                grammar_mask_kind=self.grammar_mask_kind,
                max_top_k=max_top_k,
            )
            payloads.append(
                build_record_payload(
                    tensor_batch,
                    plan.diagnostic_batch_indices[start:end],
                    rows_per_request=width,
                    retention_flags=plan.retain_on_device[start:end],
                    streaming_flags=plan.stream_all_token_metrics[start:end],
                    speculative=True,
                )
            )
        return ConstraintMetricsDeviceOutput(record_payloads=payloads)


def begin_linear_speculation_capture(
    target_logits: torch.Tensor,
    batch: ScheduleBatch,
    vocab_mask: Optional[torch.Tensor],
    *,
    eos_token_ids: list[int],
    scratch: CaptureScratch,
) -> Optional[LinearSpeculationCapture]:
    """Capture target distributions after penalties/bias and before grammar."""
    plan = build_capture_plan([req.grammar for req in batch.reqs])
    if plan is None or not plan.diagnostic_batch_indices:
        return None
    diagnostic_indices = plan.diagnostic_batch_indices
    rows = len(diagnostic_indices)
    _, width, vocab_size = target_logits.shape
    positions = rows * width
    if scratch.capacity < positions or scratch.pre.shape[1] != vocab_size:
        raise RuntimeError(
            "constraint metrics scratch is smaller than the speculative batch"
        )
    if vocab_mask is None:
        raise RuntimeError(
            "linear-speculation constraint metrics require the packed grammar mask"
        )
    inputs = select_capture_inputs(
        target_logits,
        batch.sampling_info.temperatures,
        vocab_mask,
        diagnostic_indices,
        scratch,
    )
    grammar_active_cache_key = (tuple(plan.grammar_active), width)
    if scratch.grammar_active_cache_key != grammar_active_cache_key:
        scratch.grammar_active_device = async_index_to_device(
            [active for active in plan.grammar_active for _ in range(width)],
            target_logits.device,
            dtype=torch.bool,
        )
        scratch.grammar_active_cache_key = grammar_active_cache_key

    return LinearSpeculationCapture(
        plan=plan,
        inputs=inputs,
        grammar_mask_kind=grammar_mask_kind(inputs.grammar_mask),
        grammar_active_device=scratch.grammar_active_device[:positions],
        eos_token_ids=eos_token_ids,
        emitted_ids=scratch.emitted_ids[:positions],
        spec_sources=scratch.spec_sources[:positions],
        row_enabled=scratch.row_enabled[:positions],
    )
