from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional, Union

from PIL import Image
from torch.utils.data import Dataset

logging.getLogger("PIL").setLevel(logging.WARNING)

if TYPE_CHECKING:
    import torch
    from torchvision.transforms import transforms


class TXTDataset(Dataset):
    """Allows to load a dataset from a txt file."""

    def __init__(
        self, root: str, txt: str, transform: Optional[transforms] = None, target_transform: Optional[transforms] = None
    ):
        """
        Initializes the TXTDataset object.

        Args:
            root (str): Root directory of the dataset.
            txt (str): Path to the txt file containing the dataset.
            transform (Optional[transforms]): Transformation to apply to the dataset.
            target_transform (Optional[transforms]): Transformation to apply to the targets.
        """
        self.root = root
        self.img_path = []
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform
        with open(os.path.join(root, txt), encoding="utf-8") as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Union[int, torch.Tensor]]:
        """
        Get the item at the given index.

        Args:
            index (int): Index of the item.
        Returns:
            tuple[torch.Tensor, Union[int, torch.Tensor]]: Image and target.
        """
        path = self.img_path[index]
        target = self.targets[index]

        with open(path, "rb") as f:
            data = Image.open(f).convert("RGB")

        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target
