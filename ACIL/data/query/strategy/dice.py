import numpy as np
import torch
from pytorch_ood.detector.dice import DICE as dice

from ACIL.data.query.strategy.base import BaseStrategy


class DICE(dice, BaseStrategy):
    """Interface for DICE strategy."""

    use_fit = True
    use_features = True

    def __init__(self, model, name, **cfg) -> None:
        """Initializes the strategy."""
        backbone = model.network.forward_backbone
        weights = (
            model.network.fc.weight if isinstance(model.network.fc, torch.nn.Linear) else model.network.fc[0].weight
        )
        bias = model.network.fc.bias if isinstance(model.network.fc, torch.nn.Linear) else model.network.fc[0].bias
        super().__init__(model=backbone, w=weights, b=bias, **cfg)

    def fit_features(self, features, labels):
        """Fits the features."""
        return super().fit_features(z=features, y=labels)

    def predict_features(self, features):
        """Predicts the features."""
        return super().predict_features(x=features)

    def select(self, scores: np, n_samples: int) -> np:
        """
        Source paper: where samples with higher scores S(x) are classified as ID and vice versa...
        Higher energy is better.
        """
        return sorted(scores, key=scores.get, reverse=True)[:n_samples]
