import numpy as np
from pytorch_ood.detector.odin import ODIN as odin

from ACIL.data.query.strategy.base import BaseStrategy


class ODIN(odin, BaseStrategy):
    """Interface for ODIN strategy."""

    use_model = True

    def __init__(self, model, name, **cfg) -> None:
        """Initializes the strategy."""
        criterion = None
        norm_std = None
        super().__init__(model=model, criterion=criterion, norm_std=norm_std, **cfg)

    def select(self, scores: np, n_samples: int) -> np:
        """Returning negative values so higher values indicate greater outlierness"""
        return sorted(scores, key=scores.get, reverse=True)[:n_samples]
