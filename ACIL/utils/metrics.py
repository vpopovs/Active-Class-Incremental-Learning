from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import numpy as np
import torch
import wandb
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score,
)

if TYPE_CHECKING:
    from ACIL.data.data import Data


np.seterr(divide="ignore", invalid="ignore")


@dataclass
class Metrics:
    """Class to store and calculate metrics for a classification task."""

    name: str
    y_true: np.ndarray
    y_out: np.ndarray
    y_pred: np.ndarray
    labels: List[int]

    def __post_init__(self):
        """Calculates metrics for the classification task."""
        matrix = confusion_matrix(self.y_true, self.y_pred)
        self.per_class_accuracy = matrix.diagonal() / matrix.sum(axis=1)
        self.accuracy = accuracy_score(self.y_true, self.y_pred)
        self.top_3_accuracy = top_k_accuracy_score(self.y_true, self.y_out, k=3, labels=self.labels)
        self.top_5_accuracy = top_k_accuracy_score(self.y_true, self.y_out, k=5, labels=self.labels)
        self.top_10_accuracy = top_k_accuracy_score(self.y_true, self.y_out, k=10, labels=self.labels)
        self.f1 = f1_score(self.y_true, self.y_pred, average="macro", zero_division=0)
        self.precision = precision_score(self.y_true, self.y_pred, average="macro", zero_division=0)
        self.recall = recall_score(self.y_true, self.y_pred, average="macro", zero_division=0)
        self.per_class_accuracy = self.per_class_accuracy
        self.per_class_f1 = f1_score(self.y_true, self.y_pred, average=None)
        self.confusion_matrix = matrix

    def results(self):
        """Returns the results of the metrics."""
        return [
            f"{self.name} {'-' * (14 - len(self.name))}",
            f"Acc:     {self.accuracy:.2%}",
            f"Top-3:   {self.top_3_accuracy:.2%}",
            f"Top-5:   {self.top_5_accuracy:.2%}",
            f"Top-10:  {self.top_10_accuracy:.2%}",
            f"F1:      {self.f1:.2%}",
            f"Prec.:   {self.precision:.2%}",
            f"Recall:  {self.recall:.2%}",
        ]

    def wandb_out(self, epoch, openset=False):
        """Returns loggable metrics for wandb."""
        data = {
            f"eval/{self.name}_Accuracy": self.accuracy,
            f"eval/{self.name}_F1": self.f1,
            f"eval/{self.name}_Precision": self.precision,
            f"eval/{self.name}_Recall": self.recall,
            "eval/epoch": epoch,
        }
        if not openset:
            data[f"eval/{self.name}_top_3_acc"] = self.top_3_accuracy
            data[f"eval/{self.name}_top_5_acc"] = self.top_5_accuracy
            data[f"eval/{self.name}_top_10_acc"] = self.top_10_accuracy
        return data


MANY_THRES = "Many", [False, 100]
MID_THRES = "Med", [100, 20]
FEW_THRES = "Few", [20, 5]
XLT_THRES = "XTL", [5, 0]


def tailed_metrics(
    class_mapping: dict,
    training_class_counts: dict,
    y_true: np.ndarray,
    y_out: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int],
) -> List[Metrics]:
    """
    Calculates metrics for the tailed classes.

    Args:
        class_mapping (dict): Mapping of classes.
        training_class_counts (dict): Training class counts.
        y_true (np.ndarray): True labels.
        y_out (np.ndarray): Predicted outputs.
        y_pred (np.ndarray): Predicted labels.
        labels (List[int]): List of labels.
    Returns:
        List[Metrics]: List of metrics for the tailed classes.
    """
    mapped_training_class_counts = {class_mapping[k]: v for k, v in training_class_counts.items()}

    metrics = []
    for name, thres in [MANY_THRES, MID_THRES, FEW_THRES, XLT_THRES]:
        thres = list(thres)
        thres[0] = thres[0] if thres[0] else np.inf
        thres[1] = thres[1] if thres[1] else 0
        thres_classes = [k for k, v in mapped_training_class_counts.items() if thres[1] < v <= thres[0]]
        thres_mask = np.isin(y_true, thres_classes)
        wandb.log({f"Distribution/{name}": len(thres_classes)}, commit=False)
        metrics.append(
            Metrics(
                name,
                y_true[thres_mask],
                y_out[thres_mask],
                y_pred[thres_mask],
                labels,
            )
            if np.any(thres_mask)
            else None
        )
    return metrics


def class_metrics(
    y_true: torch.Tensor, y_out: torch.Tensor, training_epoch: int = None, data: Data = None, tag: str = "Test"
) -> tuple[Metrics, Metrics]:
    """
    Calculates accuracy, f1-score, and confusion matrix for a given classification task.

    Args:
        y_true (torch.Tensor): true labels
        y_out (torch.Tensor): predicted outputs (plain model outputs)
        training_epoch (int): training epoch
        data (Data): data object
        tag (str): tag for logging
    Returns:
        Metrics: Metrics for the classification task
    """

    closed_results, open_results, tailed_results = None, None, None

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_out, torch.Tensor):
        y_out = y_out.detach().cpu().numpy()

    y_pred = np.argmax(y_out, axis=1)
    true_classes = np.arange(y_out.shape[1])

    closed_mask = y_true != -1
    closed_results = Metrics(
        name="Closed",
        y_true=y_true[closed_mask],
        y_out=y_out[closed_mask],
        y_pred=y_pred[closed_mask],
        labels=true_classes[true_classes != -1],
    )

    if -1 in true_classes:
        open_results = Metrics(
            name="Open",
            y_true=y_true,
            y_out=np.concatenate(
                [np.full((len(y_out), 1), np.finfo(y_out.dtype).min), y_out], axis=1
            ),  # Adds -inf to -1 label
            y_pred=y_pred,
            labels=true_classes,
        )

    if data:
        classes, counts = np.unique(data.train_data.targets, return_counts=True)
        training_class_counts = dict(zip(classes, counts))

        tailed_results = tailed_metrics(
            class_mapping=data.class_mapping,
            training_class_counts=training_class_counts,
            y_true=y_true,
            y_out=y_out,
            y_pred=y_pred,
            labels=true_classes[true_classes != -1],  # Without -1
        )

    data = {}
    results = []
    for result in [closed_results, *tailed_results, open_results]:
        if result:
            data.update(result.wandb_out(training_epoch))
            results.append(result.results())
    if tag == "val":
        data = {"val": data}
    else:
        print("----- Evaluation -----")
        print("\n".join(["\t".join(a) for a in zip(*results)]))
    wandb.log(data, commit=False)

    return closed_results, open_results


def class_coverage(targets: torch.Tensor, n_classes, thresholds: list = [5]) -> float:  # pylint: disable=W0102
    """
    Calculates the class coverage.

    Args:
        targets (torch.Tensor): true labels
        n_classes (int): number of classes
        thresholds (list): thresholds for class coverage, excluding 1
    Returns:
        float: class coverage
    """
    classes, counts = np.unique(targets, return_counts=True)
    coverage = len(classes) / n_classes if n_classes else None
    data = {"Class/Coverage": coverage, "Labels/Total": len(targets), "Labels/Unique": len(classes)}
    if thresholds:
        coverage = [coverage]
        for threshold in thresholds:
            thres_coverage = len(classes[counts > threshold]) / n_classes if n_classes else None
            data[f"Class/Coverage_N_{threshold}"] = thres_coverage
            coverage = thres_coverage
    wandb.log(data, commit=False)
    return coverage


def openset_recognition(score, labels, tag: str, q_i: int, targets=None) -> tuple[float, float, float, float]:
    """
    Calculates openset recognition metrics.

    Args:
        score (np.ndarray): model output scores
        labels (np.ndarray): true labels
        tag (str): tag for logging
        q_i (int): number of samples to consider
        targets (np.ndarray): targets
    Returns:
        tuple[float, float, float, float]: FPR95, AUROC, AUPR_IN, AUPR_OUT
    """
    try:
        if not isinstance(score, np.ndarray):
            score = np.array(score)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)

        if np.isnan(score).any():
            print(f"Found NaN values in score for {tag}, removing them.")
            mask = ~np.isnan(score)
            score = score[mask]
            labels = labels[mask]

        auroc = roc_auc_score(labels, score)
        precision_in, recall_in, _ = precision_recall_curve(labels, score)
        aupr_in = auc(recall_in, precision_in)
        labels_out = 1 - labels  # Reverse the labels
        precision_out, recall_out, _ = precision_recall_curve(labels_out, score)
        aupr_out = auc(recall_out, precision_out)
        fpr, tpr, _ = roc_curve(labels, score)
        fpr95 = fpr[np.argmax(tpr >= 0.95)]

        combined = np.array([score.copy(), labels_out.copy(), targets.copy()])
        combined = combined[:, np.argsort(combined[0])]
        combined = combined[:, -q_i:]
        novel_samples = sum(combined[1])

        tag = "/".join(tag.split(" "))
        data = {
            f"osr/{tag}/FPR95": fpr95,
            f"osr/{tag}/AUROC": auroc,
            f"osr/{tag}/AUPR_IN": aupr_in,
            f"osr/{tag}/AUPR_OUT": aupr_out,
            f"osr/{tag}/Novel_samples": novel_samples,
        }

        if targets is not None:
            novel_class = np.unique([_cls for _ood, _cls in zip(combined[1], combined[2]) if _ood])
            data[f"osr/{tag}/Novel_classes"] = len(novel_class)

        wandb.log(data, commit=False)

        return fpr95, auroc, aupr_in, aupr_out
    except Exception as e:
        print(f"Error in openset_recognition, {tag}\n {e}")
        return None
