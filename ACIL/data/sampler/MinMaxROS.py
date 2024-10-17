from __future__ import annotations

import math
from copy import deepcopy
from typing import TYPE_CHECKING, Union

import numpy as np
import torch
from torch.utils.data import Sampler

from ACIL.utils.base import Base
from ACIL.utils.subsets import Subset

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from torch.utils.data import Subset as TorchSubset


class MinMaxROS(Sampler, Base):
    """
    Uses Random Over Sampling to balance the classes
    And filters out classes with too few and too many samples.
    """

    def __init__(self, dataset: Union[Dataset, Subset, TorchSubset], **cfg: dict):
        """
        Initializes the MinMaxROS object.

        Args:
            dataset (Union[Dataset, Subset, TorchSubset]): Dataset to balance.
            **cfg (dict): Hydra configuration.
        """
        Base.__init__(self, cfg)

        self.dataset = dataset
        if isinstance(dataset, Subset) and hasattr(dataset.dataset, "target_transform"):
            self.transform = dataset.dataset.target_transform
        else:
            self.transform = None
        self.__post_init__()

    def __post_init__(self):
        """Creates new data for epoch."""
        n_min = 0 if not self.cfg.n_min else self.cfg.n_min
        n_max = math.inf if not self.cfg.n_max else self.cfg.n_max

        self._idx_dataset = deepcopy(self.dataset.indices)
        self.ignored_classes = []
        self._n_ignored_classes = []
        self.cls_data_list = {}
        classes, counts = np.unique(self.dataset.targets, return_counts=True)
        for label, cnt in zip(classes, counts):
            if self.transform is not None:
                label = self.transform(label)
            if n_min <= cnt <= n_max:
                self.cls_data_list[label] = []
            else:
                self.ignored_classes.append(label)
                self._n_ignored_classes.append(cnt)
        self.n_classes = len(self.cls_data_list)

        for i, label in enumerate(self.dataset.targets):
            if isinstance(label, torch.Tensor):
                label = label.item()
            if self.transform is not None:
                label = self.transform(label)
            if label in self.cls_data_list:
                self.cls_data_list[label].append(i)
        if self.ignored_classes:
            self.log.info(f"Only sampling from {self.n_classes}/{self.n_classes+len(self.ignored_classes)} classes.")

    def __iter__(self):
        """Iterates over the dataset."""
        if not np.array_equal(self._idx_dataset, self.dataset.indices):
            if self.cfg.verbose:
                self.log.debug("Reinitializing Sampler..")
            self.__post_init__()

        classes = list(self.cls_data_list.keys())
        if self.cfg.shuffle:
            for label in self.cls_data_list:
                np.random.shuffle(self.cls_data_list[label])
            np.random.shuffle(classes)

        for i in range(self.cfg.n_per_class):
            for label in classes:
                yield self.cls_data_list[label][i % len(self.cls_data_list[label])]

    def __len__(self):
        """Returns the length of the sampler."""
        return self.cfg.n_per_class * self.n_classes
