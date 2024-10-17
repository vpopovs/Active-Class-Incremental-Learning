from __future__ import annotations

import time
from typing import TYPE_CHECKING, Tuple, Union

import numpy as np
import torch
import wandb
from hydra.utils import instantiate
from pytorch_ood.utils import TensorBuffer, is_known

from ACIL.data.query.strategy.multiood import MultiOOD
from ACIL.data.query.strategy.random import Random
from ACIL.utils.base import Base
from ACIL.utils.metrics import class_coverage, openset_recognition
from ACIL.utils.parse import match_parameters

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from ACIL.data.activelearning import ActiveLearning
    from ACIL.model.model import BaseModel as Model
    from ACIL.trainer.trainer import BaseTrainer as Trainer


class Query(Base):
    """Query module for active learning."""

    def __init__(self, data: ActiveLearning, model: torch.nn.Module, counter: int, **cfg: dict):
        """
        Initializes the Query module.

        Args:
            data (ActiveLearning): ActiveLearning object.
            model (torch.nn.Module): Model object.
            counter (int): Query counter.
            **cfg (dict): Hydra configuration
        """
        super().__init__(cfg)
        self.data = data
        self.model = model

        self.strategy = instantiate(self.cfg.strategy, model=self.model, _recursive_=False)
        self._random = isinstance(self.strategy, Random)

        if self.cfg.osr and not hasattr(data, "open_loader"):
            self.log.warning("Open-set recognition enabled but no open_loader found in data. Disabling OSR.")
            self.cfg.osr = False

        self.counter = counter

    def __call__(self, trainer: Trainer, force: bool = False, *args, **kwargs) -> Union[None, list]:
        """
        Queries the data for new samples.

        Args:
            trainer (Trainer): Trainer object.
            force (bool): Whether to force query, ignoring the query_every and warmup settings.
        Returns:
            Union[None, list]: List of indices that have been queried.
        """
        if not force:
            if self.cfg.warmup and not self.cfg.inference_before_warmup and self.counter < self.cfg.warmup:
                return None
            if self.cfg.query_every and self.counter % self.cfg.query_every != 0:
                return None
        if len(self.data.unlabelled_idx) == 0:
            self.log.debug(f"No more unlabelled data. ({self.counter}).")
            return None
        trainer.eval()

        if hasattr(self.strategy, "reset"):
            self.strategy.reset()
        query = self.query(*args, **kwargs)
        self.log.info(
            "Currently coverage:"
            + f"{class_coverage(self.data.train_data.targets, self.data.n_classes, thresholds=[5, 10]):.1%}"
        )
        self.data.query_counter += 1
        return query

    def query(self) -> Union[None, list]:
        """
        Queries the data for new samples.

        Returns:
            Union[None, list]: List of indices that have been queried.
        """
        start = time.time()
        self.n_samples = self.cfg.n_samples * self.counter if self.cfg.incremental else self.cfg.n_samples
        self.log.debug(f"Querying data for {self.n_samples} samples...")

        if self._random:
            if len(self.data.unlabelled_loader.dataset.indices) < self.n_samples:
                self.data.label_idx(self.data.unlabelled_loader.dataset.indices)
            else:
                self.data.label_idx(
                    np.random.choice(self.data.unlabelled_loader.dataset.indices, self.n_samples, replace=False)
                )
            return None

        scores = self.inference()
        queries = []
        self.log.debug("Selecting samples..")
        for _ in range(self.cfg.strategy.steps) if hasattr(self.cfg.strategy, "steps") else range(1):
            query = self.strategy.select(scores, self.n_samples)
            if query is None:
                break
            feedback = self.data.label_idx(query)
            self.strategy.feedback(feedback)
            queries.append(query)
            for idx in query:
                scores.pop(idx)
        wandb.log({"OOD/time/query": time.time() - start}, commit=False)
        return [q for query in queries for q in query]

    def inference(self) -> dict:
        """
        Infers the unlabelled data and returns the scores.

        Returns:
            dict: Scores for the unlabelled data.
        """
        scores = {}
        inference_counter = 0
        self.model.eval()
        with torch.no_grad():
            if self.strategy.use_fit:
                self.log.debug("Fitting on labelled data..")
                features, z, logits, y = extract_features(
                    model=self.model.network,
                    data_loader=self.data.labelled_loader,
                    device=self.model.device,
                )
                args = {
                    "logits": logits,
                    "labels": y,
                    "features": features,
                    "z": z,
                    "device": self.model.device,
                }

                match_parameters(self.strategy.fit_features, args)

            if self.cfg.osr:
                self.log.debug("Inferencing for open-set recognition..")
                closed_scores = []
                open_scores = []
                with torch.no_grad():
                    for inputs, labels in self.data.labelled_loader:
                        inputs = inputs.to(self.model.device)
                        predictions = self.strategy.predict(inputs).detach().cpu().numpy()
                        closed_scores.extend(predictions)

                    for inputs, labels in self.data.open_loader:
                        inputs = inputs.to(self.model.device)
                        predictions = self.strategy.predict(inputs).detach().cpu().numpy()
                        open_scores.extend(predictions)

                osr_scores = np.concatenate([closed_scores, open_scores], axis=0)
                osr_labels = np.concatenate([np.zeros(len(closed_scores)), np.ones(len(open_scores))], axis=0)

                if isinstance(self.strategy, MultiOOD):
                    for i, name in enumerate(self.strategy.strategies.keys()):
                        openset_recognition(osr_scores[:, i], osr_labels, tag=f"D_open {name}", q_i=self.n_samples)
                else:
                    openset_recognition(osr_scores, osr_labels, tag="D_open", q_i=self.n_samples)

            self.log.debug("Inferencing on unlabelled data..")
            with torch.no_grad():
                for data in self.data.unlabelled_loader:
                    if len(data) == 2:  # Open-set data
                        inputs, _ = data
                        inputs = inputs.to(self.model.device)
                        labels = torch.full((len(inputs),), False)
                        index = torch.arange(len(scores), len(scores) + len(inputs))

                    elif len(data) == 3:  # Unlabelled data
                        inputs, labels, index = data
                        inputs, labels, index = inputs.to(self.model.device), labels.cpu(), index.cpu()

                    predictions = self.strategy.predict(inputs).detach().cpu().numpy()
                    for i, idx in enumerate(index):
                        scores[idx.item()] = predictions[i], labels[i].item()
                    inference_counter += len(inputs)
                    if self.cfg.limit_inferences and inference_counter >= self.cfg.limit_inferences:
                        break

            if self.data.open_classes:
                unlabelled_scores, unlabelled_labels = zip(*list(scores.values()))
                unlabelled_scores, unlabelled_labels = np.array(unlabelled_scores), np.array(unlabelled_labels)
                targets = np.array(self.data.all_train_data.dataset.targets)[list(scores.keys())]
                if isinstance(self.strategy, MultiOOD):
                    for i, name in enumerate(self.strategy.strategies.keys()):
                        openset_recognition(
                            unlabelled_scores[:, i],
                            unlabelled_labels,
                            tag=f"D_U {name}",
                            q_i=self.cfg.n_samples,
                            targets=targets,
                        )
                else:
                    openset_recognition(
                        unlabelled_scores,
                        unlabelled_labels,
                        tag="D_U",
                        q_i=self.cfg.n_samples,
                        targets=targets,
                    )
            else:
                self.log.debug("No open classes, skipping unlabelled OSR..")

            scores = {k: v[0] for k, v in scores.items()}
            return scores


def extract_features(
    data_loader: DataLoader, model: Model, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Based on pytorchood.utils.extract_features
    Extracts features and embeddings from the model.

    Args:
        data_loader (DataLoader): DataLoader.
        model (Model): Model.
        device (torch.device): Device.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Features, embeddings, logits, labels.
    """
    buffer = TensorBuffer()
    with torch.no_grad():
        for batch in data_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            known = is_known(y)
            if known.any():
                features = model.forward_backbone(x[known])
                z, logits = model.forward_head_with_latent(features)
                z = z.view(known.sum(), -1)
                buffer.append("features", features)
                buffer.append("embedding", z)
                buffer.append("logits", logits)
                buffer.append("label", y[known])

        if buffer.is_empty():
            raise ValueError("No IN instances in loader")

    features = buffer.get("features")
    z = buffer.get("embedding")
    logits = buffer.get("logits")
    y = buffer.get("label")
    return features, z, logits, y
