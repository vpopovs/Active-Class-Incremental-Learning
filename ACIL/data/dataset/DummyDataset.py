from typing import Optional, Tuple, Union
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch

class DummyDataset(Dataset):
    """
    A dummy dataset mimicking TXTDataset, generating random images and targets.
    Returns (img, target) or (img, target, path) if return_path=True.
    """

    def __init__(
        self,
        root: str = "",
        txt: str = "",
        transform=None,
        target_transform=None,
        n_samples: int = 100,
        image_size: Tuple[int, int, int] = (3, 224, 224),  # (C,H,W)
        n_classes: int = 10,
        return_path: bool = False,
        target_dtype: torch.dtype = torch.long,  # downstream-safe dtype
    ):
        self.root = root
        self.img_path = [f"dummy_image_{i}" for i in range(n_samples)]

        # Long-tailed (exponential decay) class distribution
        exp_decay = np.exp(-np.arange(n_classes))
        probs = exp_decay / exp_decay.sum()

        # Ensure every class appears at least once
        if n_samples < n_classes:
            # If not enough samples to cover all classes, cap to n_samples unique classes
            base = np.arange(n_samples, dtype=int)
        else:
            base = np.arange(n_classes, dtype=int)
        remaining = n_samples - len(base)
        if remaining > 0:
            tail = np.random.choice(n_classes, size=remaining, p=probs)
            targets = np.concatenate([base, tail])
        else:
            targets = base
        self.targets = np.random.permutation(targets).astype(int)

        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        self.return_path = return_path
        self.target_dtype = target_dtype

    def __len__(self) -> int:
        return len(self.targets)

    def _to_safe_target(self, t) -> torch.Tensor:
        """
        Convert target to a torch tensor with a stable dtype (default: long).
        This avoids numpy.bool_ / numpy scalar issues during collation.
        """
        # If target_transform produced a numpy scalar or bool, normalize it:
        if isinstance(t, (np.bool_, bool)):
            t = int(bool(t))  # -> 0/1
        elif isinstance(t, (np.integer,)):
            t = int(t)
        # If it's an ndarray of bools, cast to uint8 or int before tensor
        if isinstance(t, np.ndarray) and t.dtype == np.bool_:
            t = t.astype(np.uint8)

        # Finally return as a torch tensor (many heads expect tensor targets)
        return torch.tensor(t, dtype=self.target_dtype)

    def __getitem__(self, index: int):
        # Make a random image as uint8
        arr = np.random.randint(0, 256, self.image_size, dtype=np.uint8)
        if arr.shape[0] == 3:  # (C,H,W) -> (H,W,C) for PIL
            arr = np.transpose(arr, (1, 2, 0))
        img = Image.fromarray(arr)

        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = self._to_safe_target(target)

        if self.return_path:
            return img, target, self.img_path[index]
        return img, target
