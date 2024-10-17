from __future__ import annotations

import importlib
from typing import Optional, Tuple

import torch

from ACIL.utils.base import Base


class ResNet(torch.nn.Module, Base):
    """ResNet model class."""

    def __init__(self, n_classes: int, device: torch.device, **cfg: dict):
        """
        Initializes the ResNet model. Loads the weights if provided. Freezes the model if specified.

        Args:
            n_classes (int): Number of classes.
            device (torch.device): Device on which the model will be trained.
            **cfg (dict): Hydra configuration dictionary.
        """
        super().__init__()
        Base.__init__(self, cfg)
        self.get_weights()

        self.network = self.model(weights=self.weights)

        if self.cfg.pretrained and self.weights is None:
            self.log.info(f"Loading pretrained weights: {self.cfg.pretrained}")
            dict_state = torch.load(self.cfg.pretrained)
            if "state_dict" in dict_state:
                dict_state = dict_state["state_dict"]
            msg = self.network.load_state_dict(dict_state, strict=False)
            if msg:
                if not set(msg.missing_keys) == {"fc.weight", "fc.bias"}:
                    self.log.error(msg)
        if self.cfg.freeze:
            for param in self.network.parameters():
                param.requires_grad = False

        self.device = device
        self.make_fc(n_classes, transfer_weights=False)
        self.network.to(device)

    def get_params(self, split: Optional[str] = None):  # noqa: A003
        """
        Returns the parameters of the model.
        If split is provided, if will return [without split, with split].

        Args:
            split (str, optional): Split the parameters based on the name. Defaults to None.
        Returns:
            list: List of parameters
        """
        if split:
            split_a = [param for name, param in self.network.named_parameters() if split not in name]
            split_b = [param for name, param in self.network.named_parameters() if split in name]
            return split_a, split_b
        return self.network.parameters()

    def make_fc(self, n_classes: int, transfer_weights: bool = True) -> None:
        """
        Adds a fully connected layer to the model.

        Args:
            n_classes (int): Number of classes.
            transfer_weights (bool, optional): Transfer the weights from the previous fc layer. Defaults to True.
        """
        if self.cfg.extra_fc:
            in_features = (
                self.network.fc[0].in_features
                if isinstance(self.network.fc, torch.nn.Sequential)
                else self.network.fc.in_features
            )
            fc = torch.nn.Sequential(
                torch.nn.Linear(in_features, self.cfg.extra_fc),
                torch.nn.Linear(self.cfg.extra_fc, n_classes),
            )
        else:
            fc = torch.nn.Linear(self.network.fc.in_features, n_classes)

        if transfer_weights:
            if isinstance(fc, torch.nn.Sequential):
                with torch.no_grad():
                    fc[0].weight[: self.network.fc[0].out_features] = self.network.fc[0].weight
                    fc[0].bias[: self.network.fc[0].out_features] = self.network.fc[0].bias
                    fc[1].weight[: self.network.fc[1].out_features] = self.network.fc[1].weight
                    fc[1].bias[: self.network.fc[1].out_features] = self.network.fc[1].bias
            else:
                with torch.no_grad():
                    fc.weight[: self.network.fc.out_features] = self.network.fc.weight
                    fc.bias[: self.network.fc.out_features] = self.network.fc.bias
        fc = fc.to(self.device)
        self.network.fc = fc
        self.fc = self.network.fc
        self.log.info(f"fc layer:\n{self.fc}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)

    def forward_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with latent"""
        if not self.cfg.extra_fc:
            return self.network(x)
        x = self.forward_backbone(x)
        z = self.network.fc[0](x)
        return z

    def forward_with_latent(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with latent"""
        if not self.cfg.extra_fc:
            return self.network(x)
        z = self.forward_latent(x)
        return z, self.network.fc[1](z)

    def forward_backbone_before_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone before the pooling layer."""
        x = self.network.conv1(x)
        x = self.network.bn1(x)
        x = self.network.relu(x)
        x = self.network.maxpool(x)

        x = self.network.layer1(x)
        x = self.network.layer2(x)
        x = self.network.layer3(x)
        x = self.network.layer4(x)
        return x

    def forward_backbone_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the pooling layer, after `forward_backbone_before_pool`."""
        x = self.network.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone."""
        x = self.forward_backbone_before_pool(x)
        x = self.forward_backbone_pool(x)
        return x

    def forward_head(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the head."""
        return self.network.fc(features)

    def forward_head_with_latent(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the head with latent."""
        if not self.cfg.extra_fc:
            return self.forward_head(features)
        z = self.network.fc[0](features)
        return z, self.network.fc[1](z)

    def get_weights(self) -> None:
        """Get the weights for the model."""
        if self.cfg.pretrained == "DEFAULT":
            assert self.default_weights is not None, "Default weights are not defined"
            self.log.debug(f"Using default weights ({self.default_weights})")
            self.weights = self.default_weights
        else:
            if not self.model.__name__ in self.cfg.pretrained:
                self.log.error(
                    "Pretrained weights arch might not match model arch: "
                    + f"{self.model.__name__} and {self.cfg.pretrained}"
                )
                try:
                    pretrained_file = self.cfg.pretrained.split("/")[-1]
                    pretrained_arch = [x.lower() for x in pretrained_file.split("_") if "resnet" in x.lower()][0]
                    self.log.info(
                        f"Assuming pretrained weights are for {pretrained_arch} model, changing model to match."
                    )
                    self.model = getattr(importlib.import_module("torchvision.models"), pretrained_arch)
                except IndexError:
                    self.log.error("Could not determine model architecture from pretrained url.")
            self.weights = None


class ResNet18(ResNet):
    """ResNet18 model class."""

    def __init__(self, *args, **kwargs):
        """Initialize the ResNet18 model."""
        from torchvision.models import ResNet18_Weights, resnet18

        self.model = resnet18
        self.default_weights = ResNet18_Weights.DEFAULT
        super().__init__(*args, **kwargs)


class ResNet34(ResNet):
    """ResNet34 model class."""

    def __init__(self, *args, **kwargs):
        """Initialize the ResNet34 model."""
        from torchvision.models import ResNet34_Weights, resnet34

        self.model = resnet34
        self.default_weights = ResNet34_Weights.DEFAULT
        super().__init__(*args, **kwargs)


class ResNet50(ResNet):
    """ResNet50 model class."""

    def __init__(self, *args, **kwargs):
        """Initialize the ResNet50 model."""
        from torchvision.models import ResNet50_Weights, resnet50

        self.model = resnet50
        self.default_weights = ResNet50_Weights.DEFAULT
        super().__init__(*args, **kwargs)


class ResNet101(ResNet):
    """ResNet101 model class."""

    def __init__(self, *args, **kwargs):
        """Initialize the ResNet101 model."""
        from torchvision.models import ResNet101_Weights, resnet101

        self.model = resnet101
        self.default_weights = ResNet101_Weights.DEFAULT
        super().__init__(*args, **kwargs)


class ResNet152(ResNet):
    """ResNet152 model class."""

    def __init__(self, *args, **kwargs):
        """Initialize the ResNet152 model."""
        from torchvision.models import ResNet152_Weights, resnet152

        self.model = resnet152
        self.default_weights = ResNet152_Weights.DEFAULT
        super().__init__(*args, **kwargs)


class WideResNet50(ResNet):
    """WideResNet50 model class."""

    def __init__(self, *args, **kwargs):
        """Initialize the WideResNet50 model."""
        from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

        self.model = wide_resnet50_2
        self.default_weights = Wide_ResNet50_2_Weights.DEFAULT
        super().__init__(*args, **kwargs)


class WideResNet101(ResNet):
    """WideResNet101 model class."""

    def __init__(self, *args, **kwargs):
        """Initialize the WideResNet101 model."""
        from torchvision.models import Wide_ResNet101_2_Weights, wide_resnet101_2

        self.model = wide_resnet101_2
        self.default_weights = Wide_ResNet101_2_Weights.DEFAULT
        super().__init__(*args, **kwargs)


class ResNext50_32x4d(ResNet):
    """ResNext50_32x4d model class."""

    def __init__(self, *args, **kwargs):
        """Initialize the ResNext50_32x4d model."""
        from torchvision.models import ResNeXt50_32X4D_Weights, resnext50_32x4d

        self.model = resnext50_32x4d
        self.default_weights = ResNeXt50_32X4D_Weights.DEFAULT
        super().__init__(*args, **kwargs)


class ResNext101_32x8d(ResNet):
    """ResNext101_32x8d model class."""

    def __init__(self, *args, **kwargs):
        """Initialize the ResNext101_32x8d model."""
        from torchvision.models import ResNeXt101_32X8D_Weights, resnext101_32x8d

        self.model = resnext101_32x8d
        self.default_weights = ResNeXt101_32X8D_Weights.DEFAULT
        super().__init__(*args, **kwargs)


class ResNext101_64x4d(ResNet):
    """ResNext101_64x4d model class."""

    def __init__(self, *args, **kwargs):
        """Initialize the ResNext101_64x4d model."""
        from torchvision.models import ResNeXt101_64X4D_Weights, resnext101_64x4d

        self.model = resnext101_64x4d
        self.default_weights = ResNeXt101_64X4D_Weights.DEFAULT
        super().__init__(*args, **kwargs)
