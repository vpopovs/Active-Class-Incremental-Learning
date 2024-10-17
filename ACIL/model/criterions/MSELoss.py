from __future__ import annotations

from typing import TYPE_CHECKING

from torch.nn import MSELoss as torch_MSELoss

if TYPE_CHECKING:
    import torch


class MSELoss(torch_MSELoss):
    """Interface for the MSELoss criterion."""

    def post_forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Post forward pass function."""
        return logits
