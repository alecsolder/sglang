"""Grammar-mask formats consumed by the constraint-metrics kernel."""

from __future__ import annotations

import torch

MASK_DENSE_ALLOWED = 0
MASK_DENSE_BLOCKED = 1
MASK_PACKED_ALLOWED = 2


def grammar_mask_kind(mask: torch.Tensor) -> int:
    if mask.dtype == torch.bool:
        return MASK_DENSE_BLOCKED
    if mask.dtype == torch.int32:
        return MASK_PACKED_ALLOWED
    raise TypeError(
        "constraint metrics require a dense bool or packed int32 grammar mask, "
        f"got {mask.dtype}"
    )
