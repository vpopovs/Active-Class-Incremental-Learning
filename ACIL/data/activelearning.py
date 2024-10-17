from __future__ import annotations

import os
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Tuple, Union

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import Dataset

from ACIL.data.data import Data
from ACIL.utils.subsets import Subset

if TYPE_CHECKING:
    from torchvision import transforms

    from ACIL.data.query.query import Query
    from ACIL.model.model import BaseModel as Model


class ActiveLearning(Data):
    """Variants of the data module that are used for active learning."""

    def __init__(
        self,
        device: torch.device,
        **cfg: dict,
    ):
        """
        Initializes the ActiveLearning object.

        Args:
            device (torch.device): Device to run the data on.
            dataset_transforms (list): List of dataset transforms to apply.
            model_transforms (list): List of model transforms to apply.
            training_transforms (list): List of training transforms to apply.
            **cfg (dict): Hydra configuration.
        """
        _cfg = DictConfig(cfg)
        assert (
            not _cfg.dataset.tail or _cfg.dataset.get("tail", {}).get("val", False) is False
        ), "Validation data is not supported."
        super().__init__(device, **cfg)

    def get_dataset(self) -> None:
        """Builds the dataset."""
        self._init_dataset()

        self.n_classes = (
            self.cfg.dataset.n_classes if self.cfg.dataset.n_classes > 0 else len(np.unique(self.train_data.targets))
        )

        self.labelled_idx = None
        self.unlabelled_idx = None

        self.log.debug(f"Dataset contains {len(self.train_data)} train & {len(self.test_data)} test samples")
        self._make_tailed_dataset()
        if not isinstance(self.train_data, Subset):
            self.train_data = Subset(self.train_data, indices=list(range(len(self.train_data))))
        self._label_initial_data()
        self._make_open_dataset()
        self._limit_dataset()

    def _label_initial_data(self) -> None:
        """
        Label the initial data D_I.
        """
        np.random.seed(self.cfg.seed)
        if self.cfg.initial.type == "random":
            if str(self.cfg.initial.size).endswith("%"):
                self.cfg.initial.size = int(len(self.train_data) * int(self.cfg.initial.size[:-1]) / 100)

            if self.cfg.initial.size >= len(self.train_data):
                self.log.warning(
                    "Initial size is larger than the dataset size. Labelling the whole dataset, excl a few sample."
                )
                self.cfg.initial.size = len(self.train_data) - 1
            self.labelled_idx = set(np.random.choice(range(len(self.train_data)), self.cfg.initial.size, replace=False))
            self.unlabelled_idx = set(range(len(self.train_data))) - self.labelled_idx
        elif self.cfg.initial.type == "balanced":
            raise NotImplementedError

        self.all_train_data = deepcopy(self.train_data)

        self.train_data = Subset(self.all_train_data, list(self.labelled_idx), transform=self.training_transform)
        self.plain_data = Subset(self.all_train_data, list(self.labelled_idx), transform=self.plain_transform)
        self.unlabelled_data = Subset(self.all_train_data, list(self.unlabelled_idx), transform=self.plain_transform)
        self.unlabelled_data.dataset = GenericDatasetWrapper(self.unlabelled_data.dataset)
        self.log.info(
            f"Initially {len(self.labelled_idx)} "
            + f"({len(self.labelled_idx)/(len(self.labelled_idx)+len(self.unlabelled_idx)):.1%}) labelled "
            + f"and {len(self.unlabelled_idx)} unlabelled samples."
        )

    def _make_open_dataset(self) -> None:
        """Make the open dataset based on the configuration."""
        if "open" in self.cfg:
            self.log.warning("Open classes are not supported.")

        train_classes = np.unique(self.train_data.targets)
        if self.cfg.extra_classes:
            all_classes = np.arange(int(self.n_classes * (1 + self.cfg.extra_classes)))
            self.log.info(
                f"Adding {int(self.n_classes * self.cfg.extra_classes)}"
                + f"{self.cfg.extra_classes} extra classes to the output."
            )
        else:
            all_classes = np.arange(self.n_classes)
        self.open_classes = set(np.setdiff1d(all_classes, train_classes))
        self.closed_classes = set(all_classes) - self.open_classes
        self.class_mapping = {cls: cls for cls in all_classes}
        for open_class in self.open_classes:
            self.class_mapping[open_class] = -1
        if self.cfg.initial.reorder:
            self._reorder_classes()

    def get_loaders(self, dataloader_cfg: DictConfig = None) -> None:
        """Builds the dataloaders."""
        dataloader_cfg = dataloader_cfg or self.cfg.dataloaders
        self.train_plain_data = self.plain_data
        if not isinstance(self.val_data, (Subset, Dataset)):
            self.val_data = Subset(self.plain_data, [])
        super().get_loaders(dataloader_cfg)
        self.labelled_loader = instantiate(dataloader_cfg.labelled_loader, dataset=self.train_data)
        self.unlabelled_loader = instantiate(dataloader_cfg.unlabelled_loader, dataset=self.unlabelled_data)

    def get_query(self, model: Model) -> Query:
        """
        Initializes the query object.

        Args:
            model (Model): Model to use for querying.
        Returns:
            Query: Query object.
        """
        if not hasattr(self, "query_counter"):
            self.query_counter = 2
        self.query = instantiate(self.cfg.query, data=self, model=model, counter=self.query_counter, _recursive_=False)
        return self.query

    def label_idx(self, idx: Union[int, list], subset_index: bool = False) -> list:
        """
        Labels the given indices and returns the feedback.

        Args:
            idx (Union[int, list]): Index or list of indices to label.
            subset_index (bool): Whether the indices are subset indices.
        Returns:
            list: Feedback for the queried indices.
        """
        if isinstance(idx, int):
            idx = [idx]

        if subset_index:
            subset_idx = idx
        else:
            if isinstance(self.all_train_data.indices, torch.Tensor):
                subset_idx = torch.where(torch.isin(self.all_train_data.indices, torch.Tensor(idx)))[0]
            elif isinstance(self.all_train_data.indices, (list, np.ndarray)):
                subset_idx = [
                    i for i in range(len(self.all_train_data.indices)) if self.all_train_data.indices[i] in idx
                ]
            else:
                raise ValueError(f"Unknown type {type(self.all_train_data.indices)}")

        assert len(subset_idx) == len(idx)
        feedback = []
        _, training_counts = np.unique(self.train_data.targets, return_counts=True)

        set_subset_idx = set(subset_idx.tolist())
        self.labelled_idx = self.labelled_idx | set_subset_idx
        self.unlabelled_idx = self.unlabelled_idx - set_subset_idx

        idx_labels = [self.all_train_data.targets[idx] for idx in subset_idx.tolist()]
        for label in idx_labels:
            if self.open_classes:
                training_label = self.class_mapping[label]
                if isinstance(label, torch.Tensor):
                    label = label.item()

                if label in self.open_classes:
                    self.discover_class(label)
                    feedback.append(0)
                    training_counts = np.append(training_counts, 1)
                else:
                    feedback.append(training_counts[training_label])
                    training_counts[training_label] += 1

        self.train_data.indices = self.all_train_data.indices[list(self.labelled_idx)]
        self.plain_data.indices = self.train_data.indices
        self.unlabelled_data.indices = self.all_train_data.indices[list(self.unlabelled_idx)]
        self.log.info(f"Labelled {len(subset_idx)} new samples. ({len(self.labelled_idx)}/{len(self.all_train_data)})")
        self.get_dataset_info(info=False)
        return feedback

    def indexs_to_class_counts(self, idx: list) -> list:
        """
        Convert the indices to the class count.

        Args:
            idx (list): List of indices.
        Returns:
            list: List of class counts
        """
        _, training_counts = np.unique(self.train_data.targets, return_counts=True)
        counts = []
        for i in idx:
            training_label = self.class_mapping[i]
            counts.append(training_counts[training_label])
        return counts

    def discover_class(self, open_class: int) -> None:
        """
        Discover open class, i.e. remove it from open classes and add it to class mapping.

        Args:
            open_class (int): Open class to discover.
        """
        assert open_class in self.open_classes
        assert open_class not in self.closed_classes
        assert self.class_mapping[open_class] == -1

        self.open_classes.discard(open_class)
        self.closed_classes.add(open_class)
        self._n_classes = len(self.closed_classes)
        if self._class_reordered:
            self.class_mapping[open_class] = len(set(self.class_mapping.values()) - {-1})
        else:
            self.class_mapping[open_class] = open_class
        self.log.info(f"Discovered open class {open_class}, now {self.class_mapping[open_class]}.")

    def is_idx_ood(self, idx: int) -> bool:
        """
        Check if the given index is out-of-distribution (novel class).

        Args:
            idx (int): Index to check.
        Returns:
            bool: Whether the index is out-of-distribution.
        """
        return self.all_train_data.dataset.targets[idx] in self.open_classes

    def write_counts_to_file(self):
        """Write the class counts to a file."""
        try:
            path = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "results.txt")
        except AttributeError:
            self.log.error("Could not find hydra output directory. Writing to results.txt in current directory.")
            path = "results.txt"
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    f"Dataset: {self.dataset_name}, "
                    + f"Open_classes: {self.open_classes} "
                    + f"({self.n_open_classes/(self.n_open_classes+self.n_classes if self.n_classes else -1):.2%})\n\n"
                    + f"Additional classes: {self.cfg.extra_classes:.1%}\n\n"
                )
                f.write(f"Classes {' '.join([str(i) for i in self.train_classes])}\n")
                _, all_train_data_counts = np.unique(self.all_train_data.targets, return_counts=True)
                f.write(f"All ({sum(all_train_data_counts)}) {' '.join([str(i) for i in all_train_data_counts])}\n")
        with open(path, "a", encoding="utf-8") as f:
            counts = []
            for cls in range(self.n_classes):
                if cls in self.train_classes:
                    index = list(self.train_classes).index(cls)
                    counts.append(self.class_counts[index])
                else:
                    counts.append(0)
            f.write(f"Counts ({sum(counts)}) {' '.join([str(i) for i in counts])}\n")


class GenericDatasetWrapper(Dataset):
    """Wrapper for a dataset that adds the index to the output. Used by D_U for querying."""

    def __init__(
        self, dataset: Dataset, transform: Optional[transforms] = None, target_transform: Optional[transforms] = None
    ):
        """
        Initializes the GenericDatasetWrapper.

        Args:
            dataset (Dataset): Dataset to wrap.
            transform (Optional[transforms]): Transform to apply.
            target_transform (Optional[transforms]): Target transform to apply.
        """
        self.dataset = dataset
        self.targets = dataset.targets
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, bool, int]:
        """
        Get the item at the given index. Returns the image, label and sample index.
        Label represents whether the class was known or not.

        Args:
            index (int): Index of the item.
        Returns:
            Tuple[torch.Tensor, bool, int]: Image, label and index.
        """
        original_output = self.dataset.__getitem__(index)
        img, label = original_output

        return img, label != -1, index

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.dataset)
