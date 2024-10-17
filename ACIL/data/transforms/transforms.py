from __future__ import annotations

from typing import TYPE_CHECKING

from torchvision import transforms

if TYPE_CHECKING:
    import torch


class make_grayscale_3channels:
    """Converts a grayscale image to 3 channels. Hacky solution to make grayscale images compatible with ResNet."""

    @staticmethod
    def make_grayscale_3channels(img: torch.Tensor) -> torch.Tensor:
        """
        Converts a grayscale image to 3 channels."""
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return img

    def get(self) -> transforms.Lambda:
        """Returns the transform."""
        return transforms.Lambda(self.make_grayscale_3channels)


class target_class_mapping:
    """Maps the target to a new target based on a class mapping."""

    def __init__(self, class_mapping: dict):
        """Initializes the target_class_mapping object."""
        self.class_mapping = class_mapping

    @staticmethod
    def target_class_mapping(target: int, class_mapping: dict) -> int:
        """
        Maps the target to a new target based on a class mapping.

        Args:
            target (int): Target to map.
            class_mapping (dict): Class mapping.
        Returns:
            int: Mapped target.
        """
        return class_mapping[target]

    def get(self) -> transforms.Lambda:
        """Returns the transform."""
        return transforms.Lambda(lambda x: self.target_class_mapping(x, self.class_mapping))
