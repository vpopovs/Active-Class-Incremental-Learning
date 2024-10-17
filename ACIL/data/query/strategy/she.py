import numpy as np
import torch
from pytorch_ood.detector.she import SHE as she
from pytorch_ood.utils import is_known

from ACIL.data.query.strategy.base import BaseStrategy


class SHE(she, BaseStrategy):
    """Interface for SHE strategy."""

    use_fit = True
    use_features = True

    def __init__(self, model, name, **cfg) -> None:
        backbone = model.network.forward_backbone
        head = model.network.fc
        super().__init__(model=backbone, head=head, **cfg)

    def fit_features(self, features, labels, device):
        """
        Fixes the issue where score could be NaN
        """
        z = features.to(device)
        y = labels.to(device)
        known = is_known(y)

        if not known.any():
            raise ValueError("No IN samples")

        y = y[known]
        z = z[known]
        classes = y.unique()

        # assume all classes are present
        assert len(classes) == classes.max().item() + 1

        # select correctly classified
        y_hat = self.head(z).argmax(dim=1)
        z = z[y_hat == y]
        y = y[y_hat == y]

        m = []
        for clazz in classes:
            class_instances = z[y == clazz]
            if len(class_instances) > 0:
                mav = class_instances.mean(dim=0)
                m.append(mav)
            else:
                # If no instances of the class, replace NaN with a large negative number
                m.append(torch.full_like(z[0], -3000))  # Or any other logical value

        self.patterns = torch.stack(m)
        return self

    def predict_features(self, features):
        """Predicts the features."""
        return super().predict_features(z=features)

    def select(self, scores: np, n_samples: int) -> np:
        """
        Higher is better.
        """
        return sorted(scores, key=scores.get, reverse=True)[:n_samples]
