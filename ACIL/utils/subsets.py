from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset as TorchSubset
from torch.utils.data import random_split

from ACIL.utils.cache import cache_file

if TYPE_CHECKING:
    from logging import Logger

    from torchvision import transforms


class Subset(TorchSubset):
    """Created so Subsets can be created from Subsets, which is not possible with Torch's Subset."""

    def __init__(
        self,
        dataset: Union[Dataset, Subset, TorchSubset],
        indices: Sequence[int] = None,
        transform: Optional[transforms] = None,
        *arg,
        **kwargs,
    ):
        """
        Initializes the Subset object.

        Args:
            dataset (Union[Dataset, Subset, TorchSubset]): Dataset to create the subset from.
            indices (Sequence[int]): Indices to include in the subset.
            transform (Optional[transforms]): Transformation to apply to the subset.
        """
        if isinstance(dataset, Subset):
            indices = [dataset.indices[i] for i in indices] if indices is not None else dataset.indices
            self.class_mapping = dataset.class_mapping
            dataset = dataset.dataset
            self.root = dataset.root
        elif isinstance(dataset, Dataset):
            indices = indices if indices is not None else list(range(len(dataset)))
            classes = np.unique(dataset.targets)
            self.root = dataset.root
            self.class_mapping = {_class: _class for _class in classes}
        else:
            raise ValueError(f"Unknown dataset type: {type(dataset)}")
        self._transform = transform if transform else dataset.transform
        if isinstance(indices, list):
            indices = torch.tensor(indices)

        super().__init__(dataset, indices, *arg, **kwargs)

    @property
    def transform(self):
        """Getter for the transform property."""
        return self._transform

    @transform.setter
    def transform(self, transform):
        """Setter for the transform property."""
        self._transform = transform
        self.dataset.transform = transform

    @property
    def targets(self):
        """Getter for the targets property."""
        return [self.dataset.targets[i] for i in self.indices]


def split_dataset_class_balanced(
    dataset: Union[Dataset, Subset],
    n_samples: int,
    seed: int,
    minimum: int = 3,
) -> tuple[Subset, Subset]:
    """
    Splits dataset into A and B, where B has n_samples per class and A has the rest.

    Args:
        dataset: Dataset to split.
        n_samples: Number of samples per class in B.
        seed: Seed for random generator.
        minimum: Minimum number of samples per class left in A.
    Returns:
        tuple[Subset, Subset]: A and B subsets.
    """
    np.random.seed(seed)

    classes = np.unique(dataset.targets)
    split_A, split_B = [], []
    split_A_counts = dict(zip(classes, np.zeros(len(classes))))
    split_B_counts = dict(zip(classes, np.zeros(len(classes))))

    targets = dataset.targets

    for i, target in enumerate(targets):
        if isinstance(target, torch.Tensor):
            target = target.item()
        if split_B_counts[target] < n_samples and split_A_counts[target] >= minimum:
            split_B.append(i)
            split_B_counts[target] += 1
        else:
            split_A.append(i)
            split_A_counts[target] += 1

    for idx, count in split_B_counts.items():
        if count < minimum:
            print(f"Class {idx} has less than {minimum}*2 samples. Split A: {split_A_counts[idx]}, Split B: {count}")

    return Subset(dataset, split_A), Subset(dataset, split_B)


def split_holdout(
    dataset: Union[Dataset, Subset],
    n_samples: int,
    seed: int = None,
    probability: float = 0.5,
) -> tuple[Subset, Subset]:
    """
    Makes an holdout split of the dataset. It returns, A (train) and B (val), where B has n_samples and A the rest.

    Args:
        dataset: Dataset to split.
        n_samples: Number of samples in B.
        seed: Seed for random generator.
        probability: Propability of a sample to be in B if less then n_samples are available.
    Returns:
        tuple[Subset, Subset]: A and B subsets.
    """
    if seed:
        np.random.seed(seed)
    else:
        np.random.seed()

    classes = np.unique(dataset.targets)
    split_A, split_B = [], []
    split_B_counts = dict(zip(classes, np.zeros(len(classes))))

    targets = np.array(dataset.targets)

    for i, target in enumerate(targets):
        if isinstance(target, torch.Tensor):
            target = target.item()

        if split_B_counts[target] < n_samples:
            if np.random.rand() < probability:
                split_B.append(i)
                split_B_counts[target] += 1
            else:
                split_A.append(i)
        else:
            split_A.append(i)

    A_indices = np.array([dataset.indices[i] for i in split_A])
    B_indices = np.array([dataset.indices[i] for i in split_B])
    return A_indices, B_indices


def split_dataset(
    dataset: Union[Dataset, Subset],
    split_n: int,
    seed: int,
    random: bool = True,
    transform: transforms = None,
) -> tuple[Subset, Subset]:
    """
    Randomly splits the dataset into two subsets, A and B, where A has split_n samples and B the rest.

    Args:
        dataset: Dataset to split.
        split_n: Number of samples in A.
        seed: Seed for random generator.
        random: If True, the split is random, otherwise it is class balanced.
        transform: Transformation to apply to the subset.
    Returns:
        tuple[Subset, Subset]: A and B subsets.
    """

    if random:
        split_A, split_B = random_split(
            dataset,
            [_t := split_n, len(dataset.targets) - _t],
            generator=torch.Generator().manual_seed(seed),
        )
        return Subset(dataset, split_A.indices), Subset(dataset, split_B.indices, transform=transform)

    classes = np.unique(dataset.targets)
    split_n_class = int(split_n / len(classes))
    split_A, split_B = [], []
    count_classes = dict(zip(classes, np.zeros(len(classes))))

    if isinstance(dataset, Subset):
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        targets = dataset.targets

    for i, target in enumerate(targets):
        if isinstance(target, torch.Tensor):
            target = target.item()
        if count_classes[target] < split_n_class:
            split_A.append(i)
        else:
            split_B.append(i)
        count_classes[target] += 1

    return Subset(dataset, split_A), Subset(dataset, split_B, transform=transform)


def make_LT_dataset(
    dataset: Dataset,
    alpha: float,
    logger: Logger,
    seed: int,
    minimal: int = 1,
    shuffle: bool = True,
    distribution: str = "pareto",
    **kwargs,
) -> Subset:
    """
    Returns a tailed subset of the dataset following a selected distribution.

    Args:
        dataset: Dataset to create the tailed dataset from.
        alpha: Parameter for the distribution.
        logger: Logger to log information.
        seed: Seed for random generator.
        minimal: Minimum number of samples per class.
        shuffle: If True, the classes are shuffled.
        distribution: Distribution to follow for the tailing.
        **kwargs: Additional arguments for the distribution
    Returns:
        Subset: Tailed dataset.
    """

    @cache_file
    def select_indices(_self: dict, dataset: Dataset, indices: list[int]) -> Subset:
        for key, value in new_class_count.items():
            indices += np.random.choice(np.where(dataset.targets == key)[0], value, replace=False).tolist()
        return np.random.permutation(indices) if shuffle else indices

    classes, counts = np.unique(dataset.targets, return_counts=True)
    max_count = max(counts)
    if shuffle:
        class_count = dict(zip(classes, counts))
    else:
        # TODO call best_fit
        class_count = dict(sorted(dict(zip(classes, counts)).items(), key=lambda item: item[1]))

    if distribution == "pareto":
        from scipy.stats import pareto

        logger.info(f"Creating tailed dataset with Pareto's distribution alpha={alpha}")
        xm = len(class_count)
        pareto_dist = pareto(alpha, loc=0, scale=xm)
        x_min = xm  # Minimum x value
        x_max = xm + xm  # Maximum x value
        step = 1  # Step size
        x_values = np.arange(x_min, x_max, step)

        pdf_values = pareto_dist.pdf(x_values)
        rescale = max_count / pdf_values[0]
        pdf_values = [int(max(round(value * rescale), minimal)) for value in pdf_values]

        new_class_count = dict(zip(class_count.keys(), pdf_values))
    elif distribution == "exp":
        logger.info(f"Creating tailed dataset with Exponential distribution alpha={alpha}")
        img_num_per_cls = []
        for cls_idx in range(len(class_count)):
            num = max_count * (alpha ** (cls_idx / (len(class_count) - 1.0)))
            img_num_per_cls.append(int(num))
        new_class_count = dict(zip(class_count.keys(), img_num_per_cls))
    elif distribution == "step":
        logger.info(f"Creating tailed dataset with Step distribution alpha={alpha}")
        img_num_per_cls = []
        for cls_idx in range(len(class_count) // 2):
            img_num_per_cls.append(max_count)
        for cls_idx in range(len(class_count) // 2):
            img_num_per_cls.append(max_count * alpha)
        new_class_count = dict(zip(class_count.keys(), img_num_per_cls))
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    if shuffle:
        np.random.seed(seed)
        new_class_count = dict(zip(list(new_class_count.keys()), np.random.permutation(list(new_class_count.values()))))
    cropped_classes = {}
    for key in new_class_count.keys():
        if new_class_count[key] > class_count[key]:
            cropped_classes[key] = new_class_count[key] - class_count[key]
            new_class_count[key] = class_count[key]
    if cropped_classes:
        logger.warning(
            f"During tailing process, shortend classes by a total of {sum(cropped_classes.values())} samples:\n"
            + f"{cropped_classes}"
        )
    indices = []
    logger.debug("Selecting indices for the new dataset... (This might take a while)")
    indices = select_indices(
        {
            "cache": os.path.join(dataset.root, ".cache", f"{seed}_{distribution}_{alpha}_{minimal}_{shuffle}.pkl"),
            "log": logger,
        },
        dataset,
        indices,
    )
    return Subset(dataset, indices)
