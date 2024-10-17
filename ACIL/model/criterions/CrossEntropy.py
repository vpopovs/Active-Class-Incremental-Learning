from __future__ import annotations

from typing import TYPE_CHECKING

from torch.nn import CrossEntropyLoss

if TYPE_CHECKING:
    import torch


class CrossEntropy(CrossEntropyLoss):
    """Interface for the CrossEntropyLoss criterion."""

    def post_forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Post forward pass function."""
        return logits
