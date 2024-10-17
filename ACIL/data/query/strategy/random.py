import numpy as np
from torch import Tensor

from ACIL.data.query.strategy.base import BaseStrategy


class Random(BaseStrategy):
    """Custom random strategy."""

    def __init__(self, *args, **kwargs) -> None:  # pylint: disable=super-init-not-called
        """Initialize the strategy."""

    def predict(self, x: Tensor) -> Tensor:
        """Caclulate uncertainty for inputs."""
        return self.score(x)

    def predict_features(self, logits: Tensor) -> Tensor:
        """Predict features."""
        return self.score(logits)

    @staticmethod
    def score(logits: Tensor) -> Tensor:
        """Scores the logits."""
        return Tensor(np.random.rand(logits.size(0)))

    def select(self, scores: np, n_samples: int) -> np:
        """Select n random samples."""
        return sorted(scores, key=scores.get)[:n_samples]
