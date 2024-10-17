import numpy as np
from pytorch_ood.detector.rmd import RMD as rmd

from ACIL.data.query.strategy.base import BaseStrategy


class RMD(rmd, BaseStrategy):
    """Interface for RMD strategy."""

    use_fit = True
    use_features = True
    use_model = True

    def __init__(self, model, name, **cfg) -> None:
        """Initializes the strategy."""
        model = model.network.forward_backbone
        super().__init__(model=model, **cfg)

    def fit_features(self, features, labels, device):
        """Fits the features."""
        return super().fit_features(z=features, y=labels, device=device)

    def predict_features(self, features):
        """Predicts the features."""
        return super().predict_features(z=features)

    def select(self, scores: np, n_samples: int) -> np:
        """
        Higher dist is better.
        """
        return sorted(scores, key=scores.get, reverse=True)[:n_samples]
