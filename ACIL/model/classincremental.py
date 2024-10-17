from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Generator, Tuple

from hydra.utils import instantiate
from omegaconf import DictConfig

from ACIL.model.model import BaseModel
from ACIL.utils.subsets import split_holdout

if TYPE_CHECKING:
    from ACIL.data.data import Data
    from ACIL.trainer.trainer import BaseTrainer as Trainer


class ClassIncrementalModel(BaseModel):
    """Variants of the model that are trained in a class incremental, where the model query the data after training."""

    def pretrain(self, trainer: Trainer) -> None:
        """Pretrain step for the model."""
        trainer.data.get_query(self)

    def postepoch(self, trainer: Trainer) -> None:
        """Postepoch step for the model."""
        trainer.data.query(trainer)
        self.update_fc(trainer)

    # pylint: disable=protected-access
    def update_fc(self, trainer: Trainer) -> None:
        """Update the fully connected layer of the model, if novel classes are found."""
        if self.cfg.fc.update and trainer.data._n_classes > self.n_classes:
            self.log.info(f"Adding {trainer.data._n_classes - self.n_classes} new classes to fc.")
            self.n_classes = trainer.data._n_classes
            if self.cfg.reload_model_on_new_classes:
                self._init_network()
            else:
                self.network.make_fc(self.n_classes, transfer_weights=self.cfg.fc.transfer_weights)
            if self.cfg.fc.reinit_optimizer or self.cfg.reload_model_on_new_classes:
                trainer.init_optimizer()


class QueryAfterTrain(ClassIncrementalModel):
    """Model that queries the data after training."""

    def postepoch(self, trainer: Trainer) -> None:
        """Override postepoch to query the data."""

    def posttrain(self, trainer: Trainer) -> None:
        """Posttrain query the data."""
        if not self.cfg.query:
            return
        trainer.data.query(trainer, force=True)


class SteppedFrozenModel(ClassIncrementalModel):
    """Model that trains the backbone and the fc in different steps."""

    def __init__(self, *args, **kwargs):
        """Initialize the model."""
        super().__init__(*args, **kwargs)
        self._last_step = -1

    def preepoch(self, trainer: Trainer) -> None:
        """Preepoch step for the model. Freezes the backbone and trains the fc depending on the step configurations."""
        step = trainer.step
        if self._last_step == step:
            return
        self._last_step = step
        backbone_params, fc_params = self.network.get_params(split="fc")
        if step == self.cfg.steps[0]:
            for param in backbone_params:
                param.requires_grad = False
            for param in fc_params:
                param.requires_grad = True
            self.log.debug("Training FC.")
        elif step == self.cfg.steps[1]:
            for param in backbone_params:
                param.requires_grad = True
            for param in fc_params:
                param.requires_grad = True
            self.log.debug("Training Backbone and FC.")
        else:
            raise ValueError(f"Step {step} not implemented.")

    def postepoch(self, trainer: Trainer) -> None:
        """Override postepoch to query the data."""

    def posttrain(self, trainer: Trainer) -> None:
        """Posttrain query the data."""
        if not self.cfg.query:
            return
        trainer.data.query(trainer, force=True)
        self.update_fc(trainer)


class HoldoutModel(SteppedFrozenModel):
    """Model that uses a holdout set for validation."""

    def preepoch(self, trainer: Trainer) -> None:
        """Preepoch step for the model."""
        super().preepoch(trainer)
        if not hasattr(self, "_train_data_indices") or len(trainer.data.train_data.indices) > len(
            self._train_data_indices  # pylint: disable=E0203
        ):
            self._train_data_indices = deepcopy(trainer.data.train_data.indices)  # make copy
        trainer.data.train_data.indices = deepcopy(self._train_data_indices)
        trainer.data.train_loader.dataset.indices, trainer.data.val_loader.dataset.indices = split_holdout(
            trainer.data.train_data,
            self.cfg.holdout,
        )

    def postepoch(self, trainer: Trainer) -> None:
        """Postepoch step for the model."""
        trainer.data.train_data.indices = deepcopy(self._train_data_indices)


class KFoldModel(SteppedFrozenModel):
    """Model that uses KFold for validation."""

    class KFold:
        """KFold class that wraps the sklearn KFold."""

        def __init__(self, cfg: DictConfig):
            """
            Initialize the KFold object.

            Args:
                cfg (DictConfig): Hydra configuration.
            """
            self.kfold = instantiate(cfg)

        def split(self, dataset) -> Generator[Tuple[list, list], None, None]:
            """
            Split the dataset into train and validation indices.

            Args:
                dataset: Dataset to split.
            Returns:
                Tuple[list, list]: Train and validation indices.
            """
            for train_idx, val_idx in self.kfold.split(dataset):
                yield train_idx, val_idx

    def preepoch(self, trainer: Trainer) -> None:
        """Preepoch step for the model."""
        super().preepoch(trainer)
        if not hasattr(self, "_kfold"):
            self._kfold = self.KFold(self.cfg.kfold)
            self._n_fold = 0

        if not hasattr(self, "_train_data_indices") or len(trainer.data.train_data.indices) > len(
            self._train_data_indices  # pylint: disable=E0203
        ):
            self._train_data_indices = deepcopy(trainer.data.train_data.indices)  # make copy
        trainer.data.train_data.indices = deepcopy(self._train_data_indices)
        if self._n_fold == 0:
            self._kfold_generator = self._kfold.split(trainer.data.train_data)

        train_idx, val_idx = next(self._kfold_generator)
        trainer.data.train_loader.dataset.indices = [self._train_data_indices[i] for i in train_idx]
        trainer.data.val_loader.dataset.indices = [self._train_data_indices[i] for i in val_idx]

        self._n_fold += 1

        if self._n_fold == self.cfg.kfold.n_splits:
            self._n_fold = 0

    def postepoch(self, trainer: Trainer) -> None:
        """Postepoch step for the model."""
        trainer.data.train_data.indices = deepcopy(self._train_data_indices)

    def posttrain(self, trainer: Trainer) -> None:
        """
        If stopped early, the postepoch will not trigger. This will make sure the data is queried after training.
        """
        self.postepoch(trainer)
        return super().posttrain(trainer)
