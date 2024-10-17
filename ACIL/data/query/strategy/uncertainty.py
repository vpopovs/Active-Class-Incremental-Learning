import numpy as np
from pytorch_ood.api import ModelNotSetException
from torch import Tensor

from ACIL.data.query.strategy.base import BaseStrategy


class Uncertainty(BaseStrategy):
    """Custom strategy for Uncertainty."""

    def __init__(self, model, **kwargs) -> None:  # pylint: disable=super-init-not-called
        """Initialize the strategy"""
        self.model = model

    def predict(self, x: Tensor) -> Tensor:
        """Caclulate uncertainty for inputs"""
        if self.model is None:
            raise ModelNotSetException

        return self.score(self.model(x))

    def predict_features(self, logits: Tensor) -> Tensor:
        """Predict features."""
        return self.score(logits)

    def score(self, logits: Tensor) -> Tensor:
        """Scores the logits."""
        prob, _ = logits.softmax(dim=1).max(dim=1)
        return prob * -1

    def select(self, scores: np, n_samples: int) -> np:
        """
        Select n samples with lowest probability, are the most uncertain samples.
        Returns negative values.
        """
        return sorted(scores, key=scores.get, reverse=True)[:n_samples]
