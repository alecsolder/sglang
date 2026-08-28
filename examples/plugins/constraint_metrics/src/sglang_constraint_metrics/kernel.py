from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from sglang_constraint_metrics import layout
from sglang_constraint_metrics.config import (
    MAX_TOP_K,
    TOP_K_COMPUTE_BUCKETS,
)
from sglang_constraint_metrics.mask import (
    MASK_DENSE_ALLOWED,
    MASK_DENSE_BLOCKED,
    MASK_PACKED_ALLOWED,
)

# K=0 favors smaller partials; K<=8 favors fewer candidate reductions on GB300.
_BASE_TILE_SIZE = 4096
_TOP_K_TILE_SIZE = 8192
_PARTIAL_FLOAT_COUNT = 5
_PARTIAL_INT_COUNT = 4


@functools.cache
def _is_sm103(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) == (10, 3)


@triton.jit
def _constraint_metrics_partial_row(
    pre_ptr,
    grammar_mask_ptr,
    temperatures_ptr,
    emitted_ptr,
    partial_floats_ptr,
    partial_ints_ptr,
    top2_values_ptr,
    top2_ids_ptr,
    masked_values_ptr,
    masked_ids_ptr,
    receiver_values_ptr,
    receiver_ids_ptr,
    vocab_size,
    num_tiles,
    pre_row_stride,
    grammar_mask_row_stride,
    grammar_mask_cols,
    partial_row_stride,
    partial_tile_stride,
    partial_int_row_stride,
    partial_int_tile_stride,
    top2_row_stride,
    top2_tile_stride,
    candidate_row_stride,
    candidate_tile_stride,
    sanitize_fill: tl.constexpr,
    grammar_mask_kind: tl.constexpr,
    max_top_k: tl.constexpr,
    tile_size: tl.constexpr,
):
    work_id = tl.program_id(0)
    row = work_id // num_tiles
    tile = work_id - row * num_tiles
    lane = tl.arange(0, tile_size)
    token = tile * tile_size + lane
    valid = token < vocab_size

    raw = tl.load(pre_ptr + row * pre_row_stride + token, mask=valid, other=0.0).to(
        tl.float32
    )
    temperature = tl.load(temperatures_ptr + row).to(tl.float32)
    finite = (
        (raw == raw)  # noqa: PLR0124
        & (raw != float("inf"))
        & (raw != -float("inf"))
        & valid
    )
    sanitized = tl.where(
        raw != raw,  # noqa: PLR0124
        -sanitize_fill,
        tl.where(
            raw == float("inf"),
            sanitize_fill,
            tl.where(raw == -float("inf"), -sanitize_fill, raw),
        ),
    )
    x = sanitized / temperature
    x = tl.where(valid, x, -float("inf"))
    # Keep these as literals: Triton cannot close over ordinary Python globals.
    if grammar_mask_kind == 2:  # packed allowed bits
        mask_col = token // 32
        mask_word = tl.load(
            grammar_mask_ptr + row * grammar_mask_row_stride + mask_col,
            mask=valid & (mask_col < grammar_mask_cols),
            other=-1,
        ).to(tl.int32)
        grammar_allowed = (((mask_word >> (token % 32)) & 1) != 0) & valid
    else:
        dense_value = (
            tl.load(
                grammar_mask_ptr + row * grammar_mask_row_stride + token,
                mask=valid,
                other=0,
            )
            != 0
        )
        if grammar_mask_kind == 1:  # dense blocked booleans
            grammar_allowed = ~dense_value & valid
        else:
            grammar_allowed = dense_value & valid
    effective_allowed = grammar_allowed

    local_max = tl.max(x, axis=0)
    weight = tl.where(valid, tl.exp(x - local_max), 0.0)
    local_z = tl.sum(weight, axis=0)
    shifted_x = tl.where(valid, x - local_max, 0.0)
    local_mean_offset = tl.sum(weight * shifted_x, axis=0) / local_z

    legal_x = tl.where(effective_allowed, x, -float("inf"))
    legal_max = tl.max(legal_x, axis=0)
    legal_weight = tl.where(effective_allowed, tl.exp(legal_x - legal_max), 0.0)
    legal_z = tl.sum(legal_weight, axis=0)

    emitted_id = tl.load(emitted_ptr + row).to(tl.int64)
    emitted_index = tl.maximum(0, tl.minimum(emitted_id, vocab_size - 1))
    emitted_raw = tl.load(pre_ptr + row * pre_row_stride + emitted_index).to(tl.float32)
    emitted_sanitized = tl.where(
        emitted_raw != emitted_raw,  # noqa: PLR0124
        -sanitize_fill,
        tl.where(
            emitted_raw == float("inf"),
            sanitize_fill,
            tl.where(emitted_raw == -float("inf"), -sanitize_fill, emitted_raw),
        ),
    )
    emitted_x = emitted_sanitized / temperature

    partial_float_base = (
        partial_floats_ptr + row * partial_row_stride + tile * partial_tile_stride
    )
    tl.store(partial_float_base + 0, local_max)
    tl.store(partial_float_base + 1, local_z)
    tl.store(partial_float_base + 2, local_mean_offset)
    tl.store(partial_float_base + 3, legal_max)
    tl.store(partial_float_base + 4, legal_z)

    partial_int_base = (
        partial_ints_ptr + row * partial_int_row_stride + tile * partial_int_tile_stride
    )
    tl.store(partial_int_base + 0, tl.sum(effective_allowed.to(tl.int32), axis=0))
    blocked = valid & ~effective_allowed
    tl.store(partial_int_base + 1, tl.sum(blocked.to(tl.int32), axis=0))
    tl.store(
        partial_int_base + 2, tl.sum((valid & (x > emitted_x)).to(tl.int32), axis=0)
    )
    tl.store(partial_int_base + 3, tl.sum((valid & ~finite).to(tl.int32), axis=0))

    # K>=2 already retains the global top two in the union of the blocked and
    # legal candidate sets. Reuse that bounded union for the production K8 bucket;
    # larger buckets keep the small dedicated merge instead of widening finalize.
    if max_top_k < 2 or max_top_k > 8:
        top2_remaining = x
        for slot in tl.static_range(0, 2):
            selected_value, selected_lane = tl.max(
                top2_remaining, axis=0, return_indices=True
            )
            top2_base = (
                top2_values_ptr + row * top2_row_stride + tile * top2_tile_stride
            )
            top2_id_base = (
                top2_ids_ptr + row * top2_row_stride + tile * top2_tile_stride
            )
            tl.store(top2_base + slot, selected_value)
            tl.store(top2_id_base + slot, tile * tile_size + selected_lane)
            top2_remaining = tl.where(
                lane == selected_lane, -float("inf"), top2_remaining
            )

    masked_remaining = tl.where(blocked, x, -float("inf"))
    receiver_remaining = legal_x
    for slot in tl.static_range(0, max_top_k):
        masked_value, masked_lane = tl.max(
            masked_remaining, axis=0, return_indices=True
        )
        receiver_value, receiver_lane = tl.max(
            receiver_remaining, axis=0, return_indices=True
        )
        masked_base = (
            masked_values_ptr
            + row * candidate_row_stride
            + tile * candidate_tile_stride
        )
        masked_id_base = (
            masked_ids_ptr + row * candidate_row_stride + tile * candidate_tile_stride
        )
        receiver_base = (
            receiver_values_ptr
            + row * candidate_row_stride
            + tile * candidate_tile_stride
        )
        receiver_id_base = (
            receiver_ids_ptr + row * candidate_row_stride + tile * candidate_tile_stride
        )
        tl.store(masked_base + slot, masked_value)
        tl.store(masked_id_base + slot, tile * tile_size + masked_lane)
        tl.store(receiver_base + slot, receiver_value)
        tl.store(receiver_id_base + slot, tile * tile_size + receiver_lane)
        masked_remaining = tl.where(
            lane == masked_lane, -float("inf"), masked_remaining
        )
        receiver_remaining = tl.where(
            lane == receiver_lane, -float("inf"), receiver_remaining
        )


@triton.jit(
    do_not_specialize_on_alignment=[
        "pre_ptr",
        "grammar_mask_ptr",
        "temperatures_ptr",
        "emitted_ptr",
        "row_enabled_ptr",
    ]
)
def _constraint_metrics_partial_kernel(
    pre_ptr,
    grammar_mask_ptr,
    temperatures_ptr,
    emitted_ptr,
    row_enabled_ptr,
    partial_floats_ptr,
    partial_ints_ptr,
    top2_values_ptr,
    top2_ids_ptr,
    masked_values_ptr,
    masked_ids_ptr,
    receiver_values_ptr,
    receiver_ids_ptr,
    vocab_size,
    num_tiles,
    pre_row_stride,
    grammar_mask_row_stride,
    grammar_mask_cols,
    partial_row_stride,
    partial_tile_stride,
    partial_int_row_stride,
    partial_int_tile_stride,
    top2_row_stride,
    top2_tile_stride,
    candidate_row_stride,
    candidate_tile_stride,
    sanitize_fill: tl.constexpr,
    grammar_mask_kind: tl.constexpr,
    max_top_k: tl.constexpr,
    tile_size: tl.constexpr,
    filter_rows: tl.constexpr,
):
    row = tl.program_id(0) // num_tiles
    enabled = True
    if filter_rows:
        enabled = tl.load(row_enabled_ptr + row)
    if enabled:
        _constraint_metrics_partial_row(
            pre_ptr,
            grammar_mask_ptr,
            temperatures_ptr,
            emitted_ptr,
            partial_floats_ptr,
            partial_ints_ptr,
            top2_values_ptr,
            top2_ids_ptr,
            masked_values_ptr,
            masked_ids_ptr,
            receiver_values_ptr,
            receiver_ids_ptr,
            vocab_size,
            num_tiles,
            pre_row_stride,
            grammar_mask_row_stride,
            grammar_mask_cols,
            partial_row_stride,
            partial_tile_stride,
            partial_int_row_stride,
            partial_int_tile_stride,
            top2_row_stride,
            top2_tile_stride,
            candidate_row_stride,
            candidate_tile_stride,
            sanitize_fill,
            grammar_mask_kind,
            max_top_k,
            tile_size,
        )


@triton.jit
def _constraint_metrics_finalize_row(
    pre_ptr,
    grammar_mask_ptr,
    temperatures_ptr,
    emitted_ptr,
    grammar_active_ptr,
    spec_source_ptr,
    eos_ids_ptr,
    partial_floats_ptr,
    partial_ints_ptr,
    top2_values_ptr,
    top2_ids_ptr,
    masked_values_ptr,
    masked_ids_ptr,
    receiver_values_ptr,
    receiver_ids_ptr,
    out_floats_ptr,
    out_ints_ptr,
    vocab_size,
    num_tiles,
    pre_row_stride,
    grammar_mask_row_stride,
    grammar_mask_cols,
    partial_row_stride,
    partial_tile_stride,
    partial_int_row_stride,
    partial_int_tile_stride,
    top2_row_stride,
    candidate_row_stride,
    candidate_tile_stride,
    out_float_row_stride,
    out_int_row_stride,
    sanitize_fill: tl.constexpr,
    grammar_mask_kind: tl.constexpr,
    inf_pressure: tl.constexpr,
    max_top_k: tl.constexpr,
    out_float_candidate_start: tl.constexpr,
    out_int_candidate_start: tl.constexpr,
    num_eos_ids: tl.constexpr,
    tile_merge_size: tl.constexpr,
    top2_merge_size: tl.constexpr,
    candidate_merge_size: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.arange(0, tile_merge_size)
    tile_valid = tile < num_tiles

    partial_base = partial_floats_ptr + row * partial_row_stride
    local_max = tl.load(
        partial_base + tile * partial_tile_stride + 0,
        mask=tile_valid,
        other=-float("inf"),
    )
    global_max = tl.max(local_max, axis=0)
    local_z = tl.load(
        partial_base + tile * partial_tile_stride + 1,
        mask=tile_valid,
        other=0.0,
    )
    local_mean_offset = tl.load(
        partial_base + tile * partial_tile_stride + 2,
        mask=tile_valid,
        other=0.0,
    )
    scale = tl.where(tile_valid, tl.exp(local_max - global_max), 0.0)
    scaled_z = local_z * scale
    z = tl.sum(scaled_z, axis=0)
    # Mask padded merge lanes before multiplying by zero. Without this guard,
    # 0 * (-inf) on lanes beyond num_tiles poisons the cross-tile moments.
    tile_mean_offset = tl.where(
        tile_valid, local_mean_offset + (local_max - global_max), 0.0
    )
    mean_offset = tl.sum(scaled_z * tile_mean_offset, axis=0) / z
    log_z = tl.log(z)
    entropy = tl.maximum(log_z - mean_offset, 0.0)

    partial_int_base = partial_ints_ptr + row * partial_int_row_stride
    n_legal = tl.sum(
        tl.load(
            partial_int_base + tile * partial_int_tile_stride + 0,
            mask=tile_valid,
            other=0,
        ),
        axis=0,
    )
    n_blocked = tl.sum(
        tl.load(
            partial_int_base + tile * partial_int_tile_stride + 1,
            mask=tile_valid,
            other=0,
        ),
        axis=0,
    )
    emitted_rank = 1 + tl.sum(
        tl.load(
            partial_int_base + tile * partial_int_tile_stride + 2,
            mask=tile_valid,
            other=0,
        ),
        axis=0,
    )
    has_non_finite = (
        tl.sum(
            tl.load(
                partial_int_base + tile * partial_int_tile_stride + 3,
                mask=tile_valid,
                other=0,
            ),
            axis=0,
        )
        > 0
    )
    legal_max = tl.load(
        partial_base + tile * partial_tile_stride + 3,
        mask=tile_valid,
        other=-float("inf"),
    )
    empty_support = n_legal == 0
    merged_legal_max = tl.max(legal_max, axis=0)
    safe_legal_max = tl.where(empty_support, 0.0, merged_legal_max)
    local_legal_z = tl.load(
        partial_base + tile * partial_tile_stride + 4,
        mask=tile_valid,
        other=0.0,
    )
    legal_scale = tl.where(
        tile_valid & ~empty_support, tl.exp(legal_max - safe_legal_max), 0.0
    )
    legal_z = tl.sum(local_legal_z * legal_scale, axis=0)
    safe_legal_z = tl.where(empty_support, 1.0, legal_z)
    legal_log_z = tl.log(safe_legal_z)
    finite_pressure = tl.maximum(
        (global_max - safe_legal_max) + log_z - legal_log_z, 0.0
    )
    pressure = tl.where(empty_support, inf_pressure, finite_pressure)
    top2_lane = tl.arange(0, top2_merge_size)
    if max_top_k >= 2 and max_top_k <= 8:
        partition_size = num_tiles * max_top_k
        top2_valid = top2_lane < partition_size * 2
        candidate_lane = tl.where(
            top2_lane < partition_size, top2_lane, top2_lane - partition_size
        )
        candidate_tile = candidate_lane // max_top_k
        candidate_slot = candidate_lane - candidate_tile * max_top_k
        candidate_offset = (
            row * candidate_row_stride
            + candidate_tile * candidate_tile_stride
            + candidate_slot
        )
        use_masked = top2_lane < partition_size
        masked_top2_values = tl.load(
            masked_values_ptr + candidate_offset,
            mask=top2_valid & use_masked,
            other=-float("inf"),
        )
        receiver_top2_values = tl.load(
            receiver_values_ptr + candidate_offset,
            mask=top2_valid & ~use_masked,
            other=-float("inf"),
        )
        top2_values = tl.where(use_masked, masked_top2_values, receiver_top2_values)
        masked_top2_ids = tl.load(
            masked_ids_ptr + candidate_offset,
            mask=top2_valid & use_masked,
            other=0,
        )
        receiver_top2_ids = tl.load(
            receiver_ids_ptr + candidate_offset,
            mask=top2_valid & ~use_masked,
            other=0,
        )
        top2_ids = tl.where(use_masked, masked_top2_ids, receiver_top2_ids)
        top1_value = tl.max(top2_values, axis=0)
        # Match the sampler's deterministic lowest-token-id tie break.
        top1_id = tl.min(
            tl.where(top2_valid & (top2_values == top1_value), top2_ids, 2147483647),
            axis=0,
        )
        top2_value = tl.max(
            tl.where(
                top2_valid & (top2_ids != top1_id),
                top2_values,
                -float("inf"),
            ),
            axis=0,
        )
    else:
        top2_valid = top2_lane < num_tiles * 2
        top2_values = tl.load(
            top2_values_ptr + row * top2_row_stride + top2_lane,
            mask=top2_valid,
            other=-float("inf"),
        )
        top2_ids = tl.load(
            top2_ids_ptr + row * top2_row_stride + top2_lane,
            mask=top2_valid,
            other=0,
        )
        top1_value, top1_slot = tl.max(top2_values, axis=0, return_indices=True)
        top1_id = tl.sum(tl.where(top2_lane == top1_slot, top2_ids, 0), axis=0)
        remaining_top2 = tl.where(top2_lane == top1_slot, -float("inf"), top2_values)
        top2_value = tl.max(remaining_top2, axis=0)
    logit_gap = top1_value - top2_value
    logit_gap_finite = (
        (logit_gap == logit_gap)  # noqa: PLR0124
        & (logit_gap != float("inf"))
        & (logit_gap != -float("inf"))
    )
    logit_gap = tl.where(logit_gap_finite, logit_gap, 0.0)
    top1_prob = tl.exp((top1_value - global_max) - log_z)
    top1_prob = tl.minimum(1.0, tl.maximum(0.0, top1_prob))

    emitted_id = tl.load(emitted_ptr + row).to(tl.int64)
    emitted_index = tl.maximum(0, tl.minimum(emitted_id, vocab_size - 1))
    emitted_raw = tl.load(pre_ptr + row * pre_row_stride + emitted_index).to(tl.float32)
    temperature = tl.load(temperatures_ptr + row).to(tl.float32)
    emitted_sanitized = tl.where(
        emitted_raw != emitted_raw,  # noqa: PLR0124
        -sanitize_fill,
        tl.where(
            emitted_raw == float("inf"),
            sanitize_fill,
            tl.where(emitted_raw == -float("inf"), -sanitize_fill, emitted_raw),
        ),
    )
    emitted_x = emitted_sanitized / temperature
    if grammar_mask_kind == 2:  # packed allowed bits
        emitted_mask_col = emitted_index // 32
        emitted_word = tl.load(
            grammar_mask_ptr + row * grammar_mask_row_stride + emitted_mask_col,
            mask=emitted_mask_col < grammar_mask_cols,
            other=-1,
        ).to(tl.int32)
        emitted_allowed = ((emitted_word >> (emitted_index % 32)) & 1) != 0
    else:
        emitted_mask_value = (
            tl.load(grammar_mask_ptr + row * grammar_mask_row_stride + emitted_index)
            != 0
        )
        if grammar_mask_kind == 1:  # dense blocked booleans
            emitted_allowed = ~emitted_mask_value
        else:
            emitted_allowed = emitted_mask_value
    emitted_model_log_prob = (emitted_x - global_max) - log_z

    if grammar_mask_kind == 2:  # packed allowed bits
        top1_mask_col = top1_id // 32
        top1_word = tl.load(
            grammar_mask_ptr + row * grammar_mask_row_stride + top1_mask_col,
            mask=top1_mask_col < grammar_mask_cols,
            other=-1,
        ).to(tl.int32)
        top1_allowed = ((top1_word >> (top1_id % 32)) & 1) != 0
    else:
        top1_mask_value = (
            tl.load(grammar_mask_ptr + row * grammar_mask_row_stride + top1_id) != 0
        )
        if grammar_mask_kind == 1:  # dense blocked booleans
            top1_allowed = ~top1_mask_value
        else:
            top1_allowed = top1_mask_value
    grammar_active = tl.load(grammar_active_ptr + row) != 0
    argmax_masked = grammar_active & ~top1_allowed

    eos_allowed = 0
    for eos_slot in tl.static_range(0, num_eos_ids):
        eos_id = tl.load(eos_ids_ptr + eos_slot).to(tl.int64)
        if grammar_mask_kind == 2:  # packed allowed bits
            eos_mask_col = eos_id // 32
            eos_word = tl.load(
                grammar_mask_ptr + row * grammar_mask_row_stride + eos_mask_col,
                mask=eos_mask_col < grammar_mask_cols,
                other=-1,
            ).to(tl.int32)
            eos_is_allowed = ((eos_word >> (eos_id % 32)) & 1) != 0
        else:
            eos_mask_value = (
                tl.load(grammar_mask_ptr + row * grammar_mask_row_stride + eos_id) != 0
            )
            if grammar_mask_kind == 1:  # dense blocked booleans
                eos_is_allowed = ~eos_mask_value
            else:
                eos_is_allowed = eos_mask_value
        eos_allowed += eos_is_allowed.to(tl.int32)
    eos_only = (n_legal > 0) & (n_legal <= num_eos_ids) & (n_legal == eos_allowed)

    float_base = out_floats_ptr + row * out_float_row_stride
    tl.store(float_base + 0, pressure)
    tl.store(float_base + 1, entropy)
    tl.store(float_base + 2, top1_prob)
    tl.store(float_base + 3, logit_gap)
    tl.store(float_base + 4, emitted_model_log_prob)
    int_base = out_ints_ptr + row * out_int_row_stride
    tl.store(int_base + 0, top1_id)
    tl.store(int_base + 1, emitted_rank)
    tl.store(int_base + 2, n_legal)
    tl.store(int_base + 3, n_blocked)
    tl.store(int_base + 4, emitted_id)
    tl.store(int_base + 5, tl.load(spec_source_ptr + row))
    tl.store(int_base + 6, argmax_masked)
    tl.store(int_base + 7, empty_support)
    tl.store(int_base + 8, eos_only)
    tl.store(int_base + 9, has_non_finite)
    tl.store(int_base + 10, emitted_allowed)

    candidate_lane = tl.arange(0, candidate_merge_size)
    candidate_valid = candidate_lane < num_tiles * max_top_k
    candidate_width = max_top_k + (max_top_k == 0)
    candidate_tile = candidate_lane // candidate_width
    candidate_slot = candidate_lane - candidate_tile * candidate_width
    candidate_offset = (
        row * candidate_row_stride
        + candidate_tile * candidate_tile_stride
        + candidate_slot
    )
    masked_candidates = tl.load(
        masked_values_ptr + candidate_offset,
        mask=candidate_valid,
        other=-float("inf"),
    )
    masked_candidate_ids = tl.load(
        masked_ids_ptr + candidate_offset,
        mask=candidate_valid,
        other=0,
    )
    receiver_candidates = tl.load(
        receiver_values_ptr + candidate_offset,
        mask=candidate_valid,
        other=-float("inf"),
    )
    receiver_candidate_ids = tl.load(
        receiver_ids_ptr + candidate_offset,
        mask=candidate_valid,
        other=0,
    )
    for slot in tl.static_range(0, max_top_k):
        masked_value, masked_slot = tl.max(
            masked_candidates, axis=0, return_indices=True
        )
        masked_id = tl.sum(
            tl.where(candidate_lane == masked_slot, masked_candidate_ids, 0), axis=0
        )
        receiver_value, receiver_slot = tl.max(
            receiver_candidates, axis=0, return_indices=True
        )
        receiver_id = tl.sum(
            tl.where(candidate_lane == receiver_slot, receiver_candidate_ids, 0),
            axis=0,
        )
        masked_prob = tl.exp((masked_value - global_max) - log_z)
        masked_prob = tl.minimum(1.0, tl.maximum(0.0, masked_prob))
        constrained_prob = tl.where(
            empty_support,
            0.0,
            tl.exp((receiver_value - safe_legal_max) - legal_log_z),
        )
        constrained_prob = tl.minimum(1.0, tl.maximum(0.0, constrained_prob))
        tl.store(float_base + out_float_candidate_start + slot, masked_prob)
        tl.store(
            float_base + out_float_candidate_start + max_top_k + slot,
            constrained_prob,
        )
        tl.store(int_base + out_int_candidate_start + slot, masked_id)
        tl.store(int_base + out_int_candidate_start + max_top_k + slot, receiver_id)
        masked_candidates = tl.where(
            candidate_lane == masked_slot, -float("inf"), masked_candidates
        )
        receiver_candidates = tl.where(
            candidate_lane == receiver_slot, -float("inf"), receiver_candidates
        )


@triton.jit(
    do_not_specialize_on_alignment=[
        "pre_ptr",
        "grammar_mask_ptr",
        "temperatures_ptr",
        "emitted_ptr",
        "row_enabled_ptr",
        "grammar_active_ptr",
        "spec_source_ptr",
    ]
)
def _constraint_metrics_finalize_kernel(
    pre_ptr,
    grammar_mask_ptr,
    temperatures_ptr,
    emitted_ptr,
    row_enabled_ptr,
    grammar_active_ptr,
    spec_source_ptr,
    eos_ids_ptr,
    partial_floats_ptr,
    partial_ints_ptr,
    top2_values_ptr,
    top2_ids_ptr,
    masked_values_ptr,
    masked_ids_ptr,
    receiver_values_ptr,
    receiver_ids_ptr,
    out_floats_ptr,
    out_ints_ptr,
    vocab_size,
    num_tiles,
    pre_row_stride,
    grammar_mask_row_stride,
    grammar_mask_cols,
    partial_row_stride,
    partial_tile_stride,
    partial_int_row_stride,
    partial_int_tile_stride,
    top2_row_stride,
    candidate_row_stride,
    candidate_tile_stride,
    out_float_row_stride,
    out_int_row_stride,
    sanitize_fill: tl.constexpr,
    grammar_mask_kind: tl.constexpr,
    inf_pressure: tl.constexpr,
    max_top_k: tl.constexpr,
    out_float_candidate_start: tl.constexpr,
    out_int_candidate_start: tl.constexpr,
    num_eos_ids: tl.constexpr,
    tile_merge_size: tl.constexpr,
    top2_merge_size: tl.constexpr,
    candidate_merge_size: tl.constexpr,
    filter_rows: tl.constexpr,
):
    row = tl.program_id(0)
    enabled = True
    if filter_rows:
        enabled = tl.load(row_enabled_ptr + row)
    if enabled:
        _constraint_metrics_finalize_row(
            pre_ptr,
            grammar_mask_ptr,
            temperatures_ptr,
            emitted_ptr,
            grammar_active_ptr,
            spec_source_ptr,
            eos_ids_ptr,
            partial_floats_ptr,
            partial_ints_ptr,
            top2_values_ptr,
            top2_ids_ptr,
            masked_values_ptr,
            masked_ids_ptr,
            receiver_values_ptr,
            receiver_ids_ptr,
            out_floats_ptr,
            out_ints_ptr,
            vocab_size,
            num_tiles,
            pre_row_stride,
            grammar_mask_row_stride,
            grammar_mask_cols,
            partial_row_stride,
            partial_tile_stride,
            partial_int_row_stride,
            partial_int_tile_stride,
            top2_row_stride,
            candidate_row_stride,
            candidate_tile_stride,
            out_float_row_stride,
            out_int_row_stride,
            sanitize_fill,
            grammar_mask_kind,
            inf_pressure,
            max_top_k,
            out_float_candidate_start,
            out_int_candidate_start,
            num_eos_ids,
            tile_merge_size,
            top2_merge_size,
            candidate_merge_size,
        )


@dataclass
class _ReductionScratch:
    rows: int
    tiles: int
    partial_floats: torch.Tensor
    partial_ints: torch.Tensor
    top2_values: torch.Tensor
    top2_ids: torch.Tensor


@dataclass
class _CandidateScratch:
    rows: int
    tiles: int
    top_k: int
    masked_values: torch.Tensor
    masked_ids: torch.Tensor
    receiver_values: torch.Tensor
    receiver_ids: torch.Tensor


_REDUCTION_SCRATCH: dict[tuple[str, int | None, int], _ReductionScratch] = {}
_CANDIDATE_SCRATCH: dict[tuple[str, int | None, int, int], _CandidateScratch] = {}
_EOS_IDS: dict[tuple[str, int | None, int, tuple[int, ...]], torch.Tensor] = {}


def _scratch(
    rows: int, tiles: int, top_k: int, device: torch.device
) -> tuple[_ReductionScratch, _CandidateScratch]:
    stream_id = int(torch.cuda.current_stream(device).cuda_stream)
    key = (device.type, device.index, stream_id)
    reduction = _REDUCTION_SCRATCH.get(key)
    if reduction is None or reduction.rows < rows or reduction.tiles < tiles:
        rows_cap = max(rows, 1 if reduction is None else reduction.rows)
        tiles_cap = max(tiles, 1 if reduction is None else reduction.tiles)
        reduction = _ReductionScratch(
            rows=rows_cap,
            tiles=tiles_cap,
            partial_floats=torch.empty(
                rows_cap,
                tiles_cap,
                _PARTIAL_FLOAT_COUNT,
                dtype=torch.float32,
                device=device,
            ),
            partial_ints=torch.empty(
                rows_cap,
                tiles_cap,
                _PARTIAL_INT_COUNT,
                dtype=torch.int32,
                device=device,
            ),
            top2_values=torch.empty(
                rows_cap, tiles_cap, 2, dtype=torch.float32, device=device
            ),
            top2_ids=torch.empty(
                rows_cap, tiles_cap, 2, dtype=torch.int32, device=device
            ),
        )
        _REDUCTION_SCRATCH[key] = reduction

    candidate_key = (*key, top_k)
    candidate = _CANDIDATE_SCRATCH.get(candidate_key)
    if top_k == 0:
        rows_cap = tiles_cap = candidate_width = 1
    else:
        rows_cap = max(rows, 1 if candidate is None else candidate.rows)
        tiles_cap = max(tiles, 1 if candidate is None else candidate.tiles)
        candidate_width = top_k
    if candidate is None or candidate.rows < rows_cap or candidate.tiles < tiles_cap:
        candidate = _CandidateScratch(
            rows=rows_cap,
            tiles=tiles_cap,
            top_k=top_k,
            masked_values=torch.empty(
                rows_cap,
                tiles_cap,
                candidate_width,
                dtype=torch.float32,
                device=device,
            ),
            masked_ids=torch.empty(
                rows_cap, tiles_cap, candidate_width, dtype=torch.int32, device=device
            ),
            receiver_values=torch.empty(
                rows_cap,
                tiles_cap,
                candidate_width,
                dtype=torch.float32,
                device=device,
            ),
            receiver_ids=torch.empty(
                rows_cap, tiles_cap, candidate_width, dtype=torch.int32, device=device
            ),
        )
        _CANDIDATE_SCRATCH[candidate_key] = candidate
    return reduction, candidate


def reserve_constraint_metrics_kernel_scratch(
    *, rows: int, vocab_size: int, max_top_k: int, device: torch.device | str
) -> int:
    """Allocate every request-visible scratch bucket before KV-cache sizing."""
    device = torch.device(device)
    if rows <= 0 or vocab_size <= 0:
        raise ValueError("constraint metrics scratch shape must be positive")

    buckets = []
    for bucket in TOP_K_COMPUTE_BUCKETS:
        buckets.append(bucket)
        if max_top_k <= bucket:
            break

    # Allocate the largest tile count first so later buckets do not replace the
    # shared reduction scratch with a larger allocation.
    max_tiles = triton.cdiv(vocab_size, _BASE_TILE_SIZE)
    reduction, _ = _scratch(rows, max_tiles, 0, device)
    candidates = []
    for bucket in buckets[1:]:
        tile_size = _TOP_K_TILE_SIZE if bucket <= 8 else _BASE_TILE_SIZE
        _, candidate = _scratch(
            rows,
            triton.cdiv(vocab_size, tile_size),
            bucket,
            device,
        )
        candidates.append(candidate)

    tensors = (
        reduction.partial_floats,
        reduction.partial_ints,
        reduction.top2_values,
        reduction.top2_ids,
        *(
            tensor
            for candidate in candidates
            for tensor in (
                candidate.masked_values,
                candidate.masked_ids,
                candidate.receiver_values,
                candidate.receiver_ids,
            )
        ),
    )
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _eos_ids_device(eos_ids: Sequence[int], device: torch.device) -> torch.Tensor:
    values = tuple(eos_ids)
    stream_id = int(torch.cuda.current_stream(device).cuda_stream)
    key = (device.type, device.index, stream_id, values)
    cached = _EOS_IDS.get(key)
    if cached is None:
        host = torch.tensor(values, dtype=torch.int64, pin_memory=device.type == "cuda")
        cached = host.to(device=device, non_blocking=device.type == "cuda")
        _EOS_IDS[key] = cached
    return cached


def compute_constraint_metrics(
    pre_logits: torch.Tensor,
    grammar_mask: torch.Tensor,
    temperatures: torch.Tensor,
    emitted_ids: torch.Tensor,
    grammar_active: torch.Tensor,
    spec_sources: torch.Tensor,
    *,
    row_enabled: torch.Tensor | None = None,
    max_top_k: int,
    eos_token_ids: Sequence[int] | None,
    sanitize_fill: float,
    inf_pressure: float,
    grammar_mask_kind: int = MASK_DENSE_ALLOWED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute record columns from raw pre-temperature logits.

    Nonfinite logits are sanitized before the per-row temperature division, matching
    the production sampler's operation order.
    """
    if pre_logits.device.type != "cuda":
        raise ValueError("fused constraint metrics require CUDA tensors")
    if pre_logits.dtype != torch.float32 or pre_logits.ndim != 2:
        raise ValueError("pre_logits must be a 2D float32 tensor")
    if grammar_mask.device != pre_logits.device:
        raise ValueError("grammar_mask must be on the same device as pre_logits")
    if temperatures.device != pre_logits.device:
        raise ValueError("temperatures must be on the same device as pre_logits")
    if not torch.is_floating_point(temperatures):
        raise ValueError("temperatures must be floating point")
    if temperatures.numel() != pre_logits.shape[0]:
        raise ValueError("temperatures must contain one value per logits row")
    if grammar_mask_kind not in {
        MASK_DENSE_ALLOWED,
        MASK_DENSE_BLOCKED,
        MASK_PACKED_ALLOWED,
    }:
        raise ValueError(f"unknown grammar mask kind {grammar_mask_kind}")
    if grammar_mask_kind == MASK_PACKED_ALLOWED:
        max_cols = (pre_logits.shape[1] + 31) // 32
        if (
            grammar_mask.dtype != torch.int32
            or grammar_mask.ndim != 2
            or grammar_mask.shape[0] != pre_logits.shape[0]
            or not 0 < grammar_mask.shape[1] <= max_cols
        ):
            raise ValueError(
                "packed grammar mask must be int32 with one row per logits row and "
                f"between 1 and {max_cols} columns, got {grammar_mask.dtype} "
                f"{tuple(grammar_mask.shape)}"
            )
    elif grammar_mask.dtype != torch.bool or grammar_mask.shape != pre_logits.shape:
        raise ValueError("dense grammar mask must be bool and match pre_logits")
    if max_top_k < 0 or max_top_k > MAX_TOP_K:
        raise ValueError(f"max_top_k must be in [0, {MAX_TOP_K}], got {max_top_k}")

    pre_logits = pre_logits.contiguous()
    if grammar_mask.stride(1) != 1:
        grammar_mask = grammar_mask.contiguous()
    temperatures = temperatures.to(dtype=torch.float32).reshape(-1).contiguous()
    # Values come from production SamplingBatchInfo, where temperature has
    # already been normalized. Do not add a device reduction/assert here: this
    # recorder intentionally remains exactly two fused launches per K bucket.
    emitted_ids = emitted_ids.to(device=pre_logits.device, dtype=torch.int64).reshape(
        -1
    )
    grammar_active = grammar_active.to(
        device=pre_logits.device, dtype=torch.bool
    ).reshape(-1)
    spec_sources = spec_sources.to(device=pre_logits.device, dtype=torch.int64).reshape(
        -1
    )
    filter_rows = row_enabled is not None
    if row_enabled is None:
        row_enabled = grammar_active
    else:
        row_enabled = row_enabled.to(
            device=pre_logits.device, dtype=torch.bool
        ).reshape(-1)
    rows, vocab_size = pre_logits.shape
    if rows == 0 or vocab_size == 0:
        raise ValueError("fused constraint metrics require non-empty logits")
    if emitted_ids.numel() != rows or grammar_active.numel() != rows:
        raise ValueError("row metadata must match pre_logits")
    if spec_sources.numel() != rows:
        raise ValueError("spec_sources must match pre_logits")
    if row_enabled.numel() != rows:
        raise ValueError("row_enabled must match pre_logits")

    eos_ids = sorted(
        {
            int(token_id)
            for token_id in (eos_token_ids or ())
            if 0 <= int(token_id) < vocab_size
        }
    )
    eos_device = _eos_ids_device(eos_ids, pre_logits.device)

    tile_size = _TOP_K_TILE_SIZE if 0 < max_top_k <= 8 else _BASE_TILE_SIZE
    num_tiles = triton.cdiv(vocab_size, tile_size)
    reduction, candidates = _scratch(rows, num_tiles, max_top_k, pre_logits.device)
    partial_floats = reduction.partial_floats[:rows, :num_tiles]
    partial_ints = reduction.partial_ints[:rows, :num_tiles]
    top2_values = reduction.top2_values[:rows, :num_tiles]
    top2_ids = reduction.top2_ids[:rows, :num_tiles]
    candidate_rows = rows if max_top_k else 1
    candidate_tiles = num_tiles if max_top_k else 1
    masked_values = candidates.masked_values[
        :candidate_rows, :candidate_tiles, : max(1, max_top_k)
    ]
    masked_ids = candidates.masked_ids[
        :candidate_rows, :candidate_tiles, : max(1, max_top_k)
    ]
    receiver_values = candidates.receiver_values[
        :candidate_rows, :candidate_tiles, : max(1, max_top_k)
    ]
    receiver_ids = candidates.receiver_ids[
        :candidate_rows, :candidate_tiles, : max(1, max_top_k)
    ]

    partial_launch_options = {}
    device_index = pre_logits.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    # The uncapped K=8 specialization uses 237 registers on SM103. Capping it at
    # 128 improves occupancy and measured kernel latency by 8-10% without spills.
    if max_top_k == 8 and _is_sm103(device_index):
        partial_launch_options["maxnreg"] = 128
    _constraint_metrics_partial_kernel[(rows * num_tiles,)](
        pre_logits,
        grammar_mask,
        temperatures,
        emitted_ids,
        row_enabled,
        partial_floats,
        partial_ints,
        top2_values,
        top2_ids,
        masked_values,
        masked_ids,
        receiver_values,
        receiver_ids,
        vocab_size,
        num_tiles,
        pre_logits.stride(0),
        grammar_mask.stride(0),
        grammar_mask.shape[1],
        partial_floats.stride(0),
        partial_floats.stride(1),
        partial_ints.stride(0),
        partial_ints.stride(1),
        top2_values.stride(0),
        top2_values.stride(1),
        masked_values.stride(0),
        masked_values.stride(1),
        sanitize_fill=sanitize_fill,
        grammar_mask_kind=grammar_mask_kind,
        max_top_k=max_top_k,
        tile_size=tile_size,
        filter_rows=filter_rows,
        num_warps=8,
        num_stages=1,
        **partial_launch_options,
    )

    floats = torch.empty(
        rows,
        layout.KERNEL_ROW_FLOAT_COUNT + layout.TOP_K_FLOAT_GROUP_COUNT * max_top_k,
        dtype=torch.float32,
        device=pre_logits.device,
    )
    ints = torch.empty(
        rows,
        layout.KERNEL_ROW_INT_COUNT + layout.TOP_K_INT_GROUP_COUNT * max_top_k,
        dtype=torch.int64,
        device=pre_logits.device,
    )
    _constraint_metrics_finalize_kernel[(rows,)](
        pre_logits,
        grammar_mask,
        temperatures,
        emitted_ids,
        row_enabled,
        grammar_active,
        spec_sources,
        eos_device,
        partial_floats,
        partial_ints,
        top2_values,
        top2_ids,
        masked_values,
        masked_ids,
        receiver_values,
        receiver_ids,
        floats,
        ints,
        vocab_size,
        num_tiles,
        pre_logits.stride(0),
        grammar_mask.stride(0),
        grammar_mask.shape[1],
        partial_floats.stride(0),
        partial_floats.stride(1),
        partial_ints.stride(0),
        partial_ints.stride(1),
        top2_values.stride(0),
        masked_values.stride(0),
        masked_values.stride(1),
        floats.stride(0),
        ints.stride(0),
        sanitize_fill=sanitize_fill,
        grammar_mask_kind=grammar_mask_kind,
        inf_pressure=inf_pressure,
        max_top_k=max_top_k,
        out_float_candidate_start=layout.KERNEL_ROW_FLOAT_COUNT,
        out_int_candidate_start=layout.KERNEL_ROW_INT_COUNT,
        num_eos_ids=len(eos_ids),
        tile_merge_size=triton.next_power_of_2(num_tiles),
        top2_merge_size=triton.next_power_of_2(
            num_tiles * (2 * max_top_k if 2 <= max_top_k <= 8 else 2)
        ),
        candidate_merge_size=triton.next_power_of_2(max(1, num_tiles * max_top_k)),
        filter_rows=filter_rows,
        num_warps=8,
        num_stages=1,
    )
    return floats, ints
