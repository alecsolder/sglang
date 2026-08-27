"""Extension contracts for sampling-time auxiliary response metadata.

Out-of-tree integrations can install these hooks from an SGLang plugin by
extending ``ModelRunner``, ``Scheduler``, and ``TokenizerManager`` through the
plugin hook registry. The model runner installs a ``SamplingObserver``; the
scheduler selects a ``SchedulerOutputStreamer`` subclass; and the tokenizer
manager consumes the resulting customized response fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, Sequence, cast

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


@dataclass(frozen=True)
class CommittedTokens:
    output_index: int
    token_ids: tuple[int, ...]


class DeviceAuxiliaryOutput(Protocol):
    """Device output copied later by the scheduler.

    Tensors must not alias CUDA-graph static buffers that a later replay can
    overwrite before the scheduler-side copy completes.
    """

    def copy_to_host(
        self, copy_tensor: Callable[[torch.Tensor], torch.Tensor]
    ) -> HostAuxiliaryOutput: ...


class HostAuxiliaryOutput(Protocol):
    """Scheduler-side result produced by ``DeviceAuxiliaryOutput``.

    ``consume`` runs after sampled tokens have been committed to each request
    and immediately before response streaming. ``commits`` is aligned with
    ``batch.reqs`` and identifies only the newly visible tokens. Implementations
    can buffer per-request values for a ``SchedulerOutputStreamer`` subclass to
    expose through customized response metadata.
    """

    def consume(
        self,
        batch: ScheduleBatch,
        commits: Sequence[Optional[CommittedTokens]],
    ) -> None: ...


class SamplingObserver(Protocol):
    """Invocation-scoped hooks around the production grammar mask and sampler.

    Returning ``None`` from ``before_grammar`` skips ``after_sample``. Install
    an observer through ``ModelRunner.sampling_observer`` from a model-runner
    subclass or plugin hook. Specialized sampling paths must override
    ``ModelRunner.supports_sampling_observer`` and publish equivalent auxiliary
    output before installing one.
    """

    def is_active(self, sampling_info: SamplingBatchInfo) -> bool: ...

    def before_grammar(
        self,
        logits: torch.Tensor,
        sampling_info: SamplingBatchInfo,
    ) -> Any: ...

    def after_sample(
        self, state: Any, token_ids: torch.Tensor
    ) -> Optional[DeviceAuxiliaryOutput]:
        """Return graph-safe device output for later scheduler-side copying."""
        ...


class LinearSpeculationObservation(Protocol):
    """Deferred observer state for one linear speculative verify step.

    ``accept_tokens`` has shape ``[batch, width]`` and
    ``num_accept_tokens`` has shape ``[batch]``. Only the leading
    ``num_accept_tokens[row]`` entries in each row are valid, including the
    bonus token. Scheduler-side stop handling may expose a shorter prefix.
    """

    def finalize(
        self,
        *,
        accept_tokens: torch.Tensor,
        num_accept_tokens: torch.Tensor,
    ) -> Optional[DeviceAuxiliaryOutput]: ...


class LinearSpeculationSamplingObserver(SamplingObserver, Protocol):
    """Sampling observer extension for linear speculative verification.

    ``target_logits`` has shape ``[batch, width, vocab]`` after sampling
    penalties and bias but before the grammar mask. ``vocab_mask`` has
    ``batch * width`` rows in the same row-major order, or is ``None`` for an
    unconstrained batch. These are borrowed device tensors: implementations
    must consume or copy needed values before this method returns. In
    particular, the caller applies the grammar mask to ``target_logits``
    immediately afterward, and either tensor may alias replayed buffers.
    """

    def is_linear_speculation_active(self, batch: ScheduleBatch) -> bool: ...

    def before_linear_speculation(
        self,
        target_logits: torch.Tensor,
        batch: ScheduleBatch,
        vocab_mask: Optional[torch.Tensor],
    ) -> Optional[LinearSpeculationObservation]: ...


def is_linear_speculation_observer(observer: SamplingObserver) -> bool:
    """Whether an observer implements the linear-speculation extension."""
    return callable(
        getattr(observer, "is_linear_speculation_active", None)
    ) and callable(getattr(observer, "before_linear_speculation", None))


def get_active_linear_speculation_observer(
    observer: Optional[SamplingObserver],
    batch: ScheduleBatch,
) -> Optional[LinearSpeculationSamplingObserver]:
    """Return an installed observer when it supports this verify step."""
    if observer is None:
        return None
    if not is_linear_speculation_observer(observer):
        raise TypeError(
            "linear speculative decoding requires an observer that implements "
            "is_linear_speculation_active and before_linear_speculation"
        )
    linear_observer = cast(LinearSpeculationSamplingObserver, observer)
    return (
        linear_observer if linear_observer.is_linear_speculation_active(batch) else None
    )
