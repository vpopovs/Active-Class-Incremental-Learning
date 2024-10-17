import numpy as np
from pytorch_ood.detector.knn import KNN as knn
from sklearn.neighbors import NearestNeighbors
from torch import Tensor

from ACIL.data.query.strategy.base import BaseStrategy


class KNN(knn, BaseStrategy):
    """Interface for KNN strategy."""

    use_latent = True
    use_fit = True

    def __init__(self, model, name, k, **cfg) -> None:  # pylint: disable=super-init-not-called
        """Initializes the strategy."""
        assert model.cfg.network.extra_fc is not None, "KNN requires extra_fc (latent) layer."
        self.model = model.network.forward_latent
        self.k = k
        self._is_fitted = False
        self.knn: NearestNeighbors = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1, **cfg)

    def predict_features(self, z: Tensor) -> Tensor:
        """Predicts the features."""
        dist, _ = self.knn.kneighbors(z.detach().cpu().numpy(), return_distance=True)

        if self.k > 0:
            return Tensor(np.mean(dist[:, 1:], axis=1))

        return Tensor(dist)

    def select(self, scores: np, n_samples: int) -> np:
        """
        Higher dist is better.
        dist, idx = self.knn.kneighbors(z.detach().cpu().numpy(), n_neighbors=1, return_distance=True)
        """
        return sorted(scores, key=scores.get, reverse=True)[:n_samples]
