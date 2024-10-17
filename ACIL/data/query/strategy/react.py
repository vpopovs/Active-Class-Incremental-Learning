import importlib

from pytorch_ood.detector.react import ReAct as react

from ACIL.data.query.strategy.base import BaseStrategy


class ReAct(react, BaseStrategy):
    """Interface for ReAct strategy."""

    use_features = True

    def __init__(self, model, name, detector, **cfg) -> None:
        """Initializes the strategy."""
        backbone = model.network.forward_backbone
        head = model.network.fc

        detector_module, detector_class = detector["_target_"].rsplit(".", 1)
        detector_module = importlib.import_module(detector_module)
        detector_class = getattr(detector_module, detector_class)(model=None, name=None)
        detector = getattr(detector_class, "score")
        self.select = getattr(detector_class, "select")
        self.feedback = getattr(detector_class, "feedback")

        super().__init__(backbone=backbone, head=head, detector=detector, **cfg)

    def predict_features(self, features):
        """Predicts the features."""
        x = features.clip(max=self.threshold)
        x = self.head(x)
        return self.detector(x)

    def fit_features(self, *args, **kwargs):
        """Not required (Orginial raises instead of return)"""
        return self
