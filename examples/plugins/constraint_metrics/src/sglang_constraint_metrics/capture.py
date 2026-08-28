"""Shared request selection for constraint-metrics capture."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from sglang.srt.constrained.llguidance_backend import GuidanceGrammar
from sglang_constraint_metrics.config import MODE_FORENSIC
from sglang_constraint_metrics.records import async_index_to_device


def grammar_has_constraint_metrics(grammar: Any) -> bool:
    return (
        isinstance(grammar, GuidanceGrammar)
        and getattr(grammar, "constraint_metrics_config", None) is not None
    )


@dataclass(frozen=True, slots=True)
class ConstraintMetricsCapturePlan:
    diagnostic_batch_indices: list[int]
    top_ks: list[int]
    grammar_active: list[bool]
    retain_on_device: list[bool]
    stream_all_token_metrics: list[bool]


@dataclass
class CaptureScratch:
    capacity: int
    pre: torch.Tensor
    emitted_ids: torch.Tensor
    spec_sources: torch.Tensor
    row_enabled: torch.Tensor
    grammar_active_cache_key: tuple[tuple[bool, ...], int] | None = None
    grammar_active_device: torch.Tensor | None = None

    @classmethod
    def allocate(
        cls,
        rows: int,
        vocab_size: int,
        device: torch.device | str,
    ) -> CaptureScratch:
        return cls(
            capacity=rows,
            pre=torch.empty(rows, vocab_size, dtype=torch.float32, device=device),
            emitted_ids=torch.empty(rows, dtype=torch.long, device=device),
            spec_sources=torch.empty(rows, dtype=torch.long, device=device),
            row_enabled=torch.empty(rows, dtype=torch.bool, device=device),
        )


@dataclass(frozen=True, slots=True)
class CaptureInputs:
    diagnostic_indices_device: torch.Tensor | None
    pre: torch.Tensor
    temperatures: torch.Tensor
    grammar_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class _StaticCapturePlan:
    diagnostic_batch_indices: tuple[int, ...]
    top_ks: tuple[int, ...]
    retain_on_device: tuple[bool, ...]
    stream_all_token_metrics: tuple[bool, ...]


@functools.lru_cache(maxsize=1)
def _static_capture_plan(grammars: tuple[Any, ...]) -> _StaticCapturePlan | None:
    """Cache batch-invariant selection while the active grammar batch is stable."""

    configs = {
        batch_index: grammar.constraint_metrics_config
        for batch_index, grammar in enumerate(grammars)
        if grammar_has_constraint_metrics(grammar)
    }
    if not configs:
        return None
    diagnostic_batch_indices = sorted(
        configs,
        key=lambda batch_index: (
            configs[batch_index].kernel_top_k,
            batch_index,
        ),
    )
    return _StaticCapturePlan(
        diagnostic_batch_indices=tuple(diagnostic_batch_indices),
        top_ks=tuple(
            configs[batch_index].kernel_top_k
            for batch_index in diagnostic_batch_indices
        ),
        retain_on_device=tuple(
            configs[batch_index].mode == MODE_FORENSIC
            for batch_index in diagnostic_batch_indices
        ),
        stream_all_token_metrics=tuple(
            configs[batch_index].stream_all_token_metrics
            for batch_index in diagnostic_batch_indices
        ),
    )


def build_capture_plan(
    grammars: Sequence[Any],
) -> ConstraintMetricsCapturePlan | None:
    grammar_tuple = tuple(grammars)
    static_plan = _static_capture_plan(grammar_tuple)
    if static_plan is None:
        return None

    diagnostic_batch_indices = list(static_plan.diagnostic_batch_indices)
    top_ks = list(static_plan.top_ks)
    retain_on_device = list(static_plan.retain_on_device)
    stream_all_token_metrics = list(static_plan.stream_all_token_metrics)
    grammar_active = [
        not grammar_tuple[batch_index].finished
        for batch_index in diagnostic_batch_indices
    ]
    return ConstraintMetricsCapturePlan(
        diagnostic_batch_indices=diagnostic_batch_indices,
        top_ks=top_ks,
        grammar_active=grammar_active,
        retain_on_device=retain_on_device,
        stream_all_token_metrics=stream_all_token_metrics,
    )


def select_capture_inputs(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    grammar_mask: torch.Tensor,
    diagnostic_batch_indices: list[int],
    scratch: CaptureScratch,
    *,
    logit_bias: torch.Tensor | None = None,
) -> CaptureInputs:
    """Copy selected normal or linear-speculation inputs into capture scratch."""
    positions_per_request = logits.shape[1] if logits.ndim == 3 else 1
    selected_positions = len(diagnostic_batch_indices) * positions_per_request
    pre = scratch.pre[:selected_positions].view(
        len(diagnostic_batch_indices), *logits.shape[1:]
    )
    full_batch = diagnostic_batch_indices == list(range(logits.shape[0]))
    if full_batch:
        diagnostic_indices_device = None
        pre.copy_(logits)
        if logit_bias is not None:
            pre.add_(logit_bias)
        selected_temperatures = temperatures.float()
        selected_grammar_mask = grammar_mask
    else:
        diagnostic_indices_device = async_index_to_device(
            diagnostic_batch_indices, logits.device, dtype=torch.long
        )
        if logits.dtype == torch.float32:
            torch.index_select(logits, 0, diagnostic_indices_device, out=pre)
        else:
            pre.copy_(torch.index_select(logits, 0, diagnostic_indices_device))
        if logit_bias is not None:
            pre.add_(logit_bias.index_select(0, diagnostic_indices_device))
        selected_temperatures = temperatures.index_select(
            0, diagnostic_indices_device
        ).float()
        mask_indices = diagnostic_indices_device
        if positions_per_request > 1:
            mask_indices = (
                diagnostic_indices_device[:, None] * positions_per_request
                + torch.arange(
                    positions_per_request,
                    device=diagnostic_indices_device.device,
                )[None, :]
            ).reshape(-1)
        selected_grammar_mask = grammar_mask.index_select(0, mask_indices)

    return CaptureInputs(
        diagnostic_indices_device=diagnostic_indices_device,
        pre=pre,
        temperatures=selected_temperatures,
        grammar_mask=selected_grammar_mask,
    )
