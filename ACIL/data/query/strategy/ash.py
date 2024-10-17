import numpy as np
from pytorch_ood.detector.ash import ASH as ash

from ACIL.data.query.strategy.base import BaseStrategy


class ASH(ash, BaseStrategy):
    """Interface for Simple Activation Shaping (ASH) strategy."""

    use_model = True

    def __init__(self, model, name, variant, **cfg) -> None:
        """Initializes the strategy."""
        self.model = model
        backbone = model.network.forward_backbone_before_pool
        head = self.head
        ash_variant = f"ash-{variant}"
        super().__init__(backbone=backbone, head=head, variant=ash_variant, **cfg)

    def predict_features(self, features):
        """Predicts the features."""
        x = self.ash(features, self.percentile)
        x = self.head(x)
        return self.detector(x)

    def fit_features(self, *args, **kwargs):
        """Not required."""
        return self

    def head(self, x):
        """Head function for the model."""
        x = self.model.network.forward_backbone_pool(x)
        x = self.model.network.fc(x)
        return x

    def select(self, scores: np, n_samples: int) -> np:
        """Higher is better."""
        return sorted(scores, key=scores.get, reverse=True)[:n_samples]
