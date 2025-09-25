from __future__ import annotations

from typing import Optional, Tuple

import torch

from ACIL.utils.base import Base


class VisionTransformer(torch.nn.Module, Base):
    """Wrapper around torchvision VisionTransformer to match the ResNet API used in this repo."""

    def __init__(self, n_classes: int, device: torch.device, **cfg: dict):
        """
        Initialize ViT model, optionally load weights, optionally freeze, and construct classification head.

        Args:
            n_classes: number of classes
            device: torch device
            **cfg: hydra config
        """
        super().__init__()
        Base.__init__(self, cfg)

        self.device = device
        self._classifier_attr = "heads"  # torchvision VisionTransformer uses `heads`

        self._build_network()

        if self.cfg.freeze:
            for param in self.network.parameters():
                param.requires_grad = False

        self.make_fc(n_classes, transfer_weights=False)
        self.network.to(device)

    def _build_network(self) -> None:
        self._get_weights()
        self.network = self.model(weights=self.weights)

    def _get_weights(self) -> None:
        if getattr(self.cfg, "pretrained", None) == "DEFAULT":
            assert self.default_weights is not None, "Default weights are not defined"
            self.log.debug(f"Using default weights ({self.default_weights})")
            self.weights = self.default_weights
        elif getattr(self.cfg, "pretrained", None):
            # Load external weights file/url using torch.load when provided a local path
            self.weights = None
            self.log.info(f"Loading pretrained weights from: {self.cfg.pretrained}")
            # weights loaded later via load_state_dict (after constructing model)
        else:
            self.weights = None

    def get_params(self, split: Optional[str] = None):  # noqa: A003
        """
        Return parameters; if split provided, split into backbone vs classifier (fc/heads).
        """
        if not split:
            return self.network.parameters()

        classifier_name = self._classifier_attr
        split_a = [param for name, param in self.network.named_parameters() if classifier_name not in name]
        split_b = [param for name, param in self.network.named_parameters() if classifier_name in name]
        return split_a, split_b

    def _current_classifier(self) -> torch.nn.Module:
        return getattr(self.network, self._classifier_attr)

    def _classifier_in_features(self) -> int:
        classifier = self._current_classifier()
        if isinstance(classifier, torch.nn.Linear):
            return classifier.in_features
        # Try to find the last Linear child
        last_linear = None
        for child in classifier.children():
            if isinstance(child, torch.nn.Linear):
                last_linear = child
        if last_linear is None:
            # Fallback to attribute often present in torchvision ViT
            if hasattr(self.network, "hidden_dim"):
                return int(self.network.hidden_dim)
            raise AttributeError("Could not determine classifier in_features for VisionTransformer.")
        return last_linear.in_features

    def make_fc(self, n_classes: int, transfer_weights: bool = True) -> None:
        if self.cfg.extra_fc:
            in_features = self._classifier_in_features()
            fc = torch.nn.Sequential(
                torch.nn.Linear(in_features, self.cfg.extra_fc),
                torch.nn.Linear(self.cfg.extra_fc, n_classes),
            )
        else:
            in_features = self._classifier_in_features()
            fc = torch.nn.Linear(in_features, n_classes)

        if transfer_weights:
            old = self._current_classifier()
            with torch.no_grad():
                if isinstance(fc, torch.nn.Sequential) and isinstance(old, torch.nn.Sequential):
                    old_linears = [m for m in old.children() if isinstance(m, torch.nn.Linear)]
                    new_linears = [m for m in fc.children() if isinstance(m, torch.nn.Linear)]
                    for i in range(min(len(old_linears), len(new_linears))):
                        new_linears[i].weight[: old_linears[i].out_features] = old_linears[i].weight
                        if old_linears[i].bias is not None and new_linears[i].bias is not None:
                            new_linears[i].bias[: old_linears[i].out_features] = old_linears[i].bias
                elif isinstance(fc, torch.nn.Linear) and isinstance(old, torch.nn.Linear):
                    fc.weight[: old.out_features] = old.weight
                    if old.bias is not None and fc.bias is not None:
                        fc.bias[: old.out_features] = old.bias

        fc = fc.to(self.device)
        setattr(self.network, self._classifier_attr, fc)
        self.fc = fc
        self.log.info(f"fc layer:\n{self.fc}")

        # If external state dict path was given, load now (after fc set)
        if getattr(self.cfg, "pretrained", None) not in (None, "DEFAULT"):
            try:
                dict_state = torch.load(self.cfg.pretrained, map_location="cpu")
                if "state_dict" in dict_state:
                    dict_state = dict_state["state_dict"]
                msg = self.network.load_state_dict(dict_state, strict=False)
                if msg and (set(getattr(msg, "missing_keys", [])) - {f"{self._classifier_attr}.weight", f"{self._classifier_attr}.bias"}):
                    self.log.error(msg)
            except Exception as exc:  # pylint: disable=broad-except
                self.log.error(f"Failed to load external pretrained weights: {exc}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def forward_latent(self, x: torch.Tensor) -> torch.Tensor:
        if not self.cfg.extra_fc:
            return self.network(x)
        features = self.forward_backbone(x)
        z = self.fc[0](features)  # type: ignore[index]
        return z

    def forward_with_latent(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.cfg.extra_fc:
            return self.network(x)
        z = self.forward_latent(x)
        return z, self.fc[1](z)  # type: ignore[index]

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.network, "forward_features"):
            return self.network.forward_features(x)
        raise AttributeError("VisionTransformer model does not expose forward_features in this torchvision version.")

    def forward_head(self, features: torch.Tensor) -> torch.Tensor:
        classifier = self._current_classifier()
        return classifier(features)

    def forward_head_with_latent(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.cfg.extra_fc:
            return self.forward_head(features)
        z = self.fc[0](features)  # type: ignore[index]
        return z, self.fc[1](z)  # type: ignore[index]


class ViT_B_16(VisionTransformer):
    """Torchvision ViT-B/16 wrapper."""

    def __init__(self, *args, **kwargs):
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        self.model = vit_b_16
        self.default_weights = ViT_B_16_Weights.DEFAULT
        super().__init__(*args, **kwargs)


