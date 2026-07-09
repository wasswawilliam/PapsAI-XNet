# src/evaluation/metrics.py

"""
Core evaluation metrics for PapsAI XNet.

This module implements the metrics reported in the manuscript:
- Accuracy
- Sensitivity / Recall
- Specificity
- Precision
- F1-score
- Class-wise metrics
- Confusion matrix
- One-vs-rest AUC
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)


ArrayLike = Union[List[int], np.ndarray]


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(accuracy_score(y_true, y_pred))


def precision(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str = "macro",
) -> float:
    return float(
        precision_score(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        )
    )


def sensitivity(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str = "macro",
) -> float:
    return float(
        recall_score(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        )
    )


def recall(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str = "macro",
) -> float:
    return sensitivity(y_true, y_pred, average=average)


def f1(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str = "macro",
) -> float:
    return float(
        f1_score(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        )
    )


def confusion(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def specificity(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str = "macro",
) -> Union[float, np.ndarray]:
    cm = confusion_matrix(y_true, y_pred)
    specificities = []

    for class_index in range(cm.shape[0]):
        tp = cm[class_index, class_index]
        fn = np.sum(cm[class_index, :]) - tp
        fp = np.sum(cm[:, class_index]) - tp
        tn = np.sum(cm) - tp - fn - fp

        value = tn / (tn + fp + 1e-12)
        specificities.append(value)

    specificities = np.array(specificities, dtype=float)

    if average == "none":
        return specificities

    return float(np.mean(specificities))


def auc_ovr(
    y_true: ArrayLike,
    y_prob: np.ndarray,
    average: str = "macro",
) -> float:
    return float(
        roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average=average,
        )
    )


def classwise_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    results: Dict[str, Dict[str, float]] = {}

    for i, label in enumerate(labels):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fn - fp

        sens = tp / (tp + fn + 1e-12)
        spec = tn / (tn + fp + 1e-12)
        prec = tp / (tp + fp + 1e-12)
        f1_val = 2 * prec * sens / (prec + sens + 1e-12)

        if class_names is not None:
            class_label = class_names[int(label)]
        else:
            class_label = str(label)

        results[class_label] = {
            "sensitivity": float(sens),
            "specificity": float(spec),
            "precision": float(prec),
            "f1_score": float(f1_val),
        }

        if y_prob is not None:
            binary_true = (y_true == label).astype(int)
            try:
                class_auc = roc_auc_score(binary_true, y_prob[:, int(label)])
            except ValueError:
                class_auc = np.nan

            results[class_label]["auc"] = float(class_auc)

    return results


def overall_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    results = {
        "accuracy": accuracy(y_true, y_pred),
        "sensitivity": sensitivity(y_true, y_pred, average="macro"),
        "specificity": specificity(y_true, y_pred, average="macro"),
        "precision": precision(y_true, y_pred, average="macro"),
        "f1_score": f1(y_true, y_pred, average="macro"),
    }

    if y_prob is not None:
        results["auc"] = auc_ovr(y_true, y_prob, average="macro")

    return results


def full_evaluation_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, object]:
    return {
        "overall_metrics": overall_metrics(y_true, y_pred, y_prob),
        "classwise_metrics": classwise_metrics(
            y_true,
            y_pred,
            y_prob,
            class_names,
        ),
        "confusion_matrix": confusion(y_true, y_pred),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
    }


def print_overall_metrics(metrics: Dict[str, float]) -> None:
    print("\nOverall Performance")
    print("-" * 40)

    for key, value in metrics.items():
        if key == "auc":
            print(f"{key.upper():<15}: {value:.4f}")
        else:
            print(f"{key:<15}: {value * 100:.2f}%")


def print_classwise_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    print("\nClass-wise Performance")
    print("-" * 80)

    header = f"{'Class':<35} {'Sens':>8} {'Spec':>8} {'Prec':>8} {'F1':>8} {'AUC':>8}"
    print(header)
    print("-" * 80)

    for class_name, values in metrics.items():
        auc_value = values.get("auc", np.nan)

        print(
            f"{class_name:<35} "
            f"{values['sensitivity'] * 100:>7.2f}% "
            f"{values['specificity'] * 100:>7.2f}% "
            f"{values['precision'] * 100:>7.2f}% "
            f"{values['f1_score'] * 100:>7.2f}% "
            f"{auc_value:>8.3f}"
        )
