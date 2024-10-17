import numpy as np
from pytorch_ood.detector.entropy import Entropy as entropy

from ACIL.data.query.strategy.base import BaseStrategy


class Entropy(entropy, BaseStrategy):
    """Interface for Entropy strategy."""

    def __init__(self, model, name, **cfg) -> None:
        """Initializes the strategy."""
        super().__init__(model, **cfg)

    def select(self, scores: np, n_samples: int) -> np:
        """Higher entropy is better."""
        return sorted(scores, key=scores.get, reverse=True)[:n_samples]

    def score(self, logits) -> np:
        """Scores the logits."""
        return entropy.score(logits)
