# src/evaluation/confusion.py

"""
Confusion matrix utilities for PapsAI XNet.

This module supports:
- Raw confusion matrix computation
- Normalized confusion matrix computation
- Class-wise error analysis
- Identification of major misclassification pairs
- Publication-quality confusion matrix plots
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


DEFAULT_CLASS_NAMES = [
    "SSE",
    "ISE",
    "CE",
    "Mild",
    "Moderate",
    "Severe",
    "CIS",
]


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Compute raw confusion matrix.
    """

    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

    return confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )


def normalize_confusion_matrix(
    cm: np.ndarray,
    mode: str = "true",
) -> np.ndarray:
    """
    Normalize confusion matrix.

    Parameters
    ----------
    cm:
        Raw confusion matrix.

    mode:
        "true" normalizes by row.
        "pred" normalizes by column.
        "all" normalizes by total count.
    """

    cm = cm.astype(float)

    if mode == "true":
        denominator = cm.sum(axis=1, keepdims=True)

    elif mode == "pred":
        denominator = cm.sum(axis=0, keepdims=True)

    elif mode == "all":
        denominator = cm.sum()

    else:
        raise ValueError("mode must be one of: true, pred, all")

    return np.divide(
        cm,
        denominator,
        out=np.zeros_like(cm, dtype=float),
        where=denominator != 0,
    )


def confusion_matrix_to_dataframe(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convert confusion matrix to pandas DataFrame.
    """

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    return pd.DataFrame(
        cm,
        index=[f"True {name}" for name in class_names],
        columns=[f"Pred {name}" for name in class_names],
    )


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    dpi: int = 300,
    figsize: Tuple[int, int] = (8, 7),
    cmap: str = "Blues",
    show: bool = True,
) -> None:
    """
    Plot publication-quality confusion matrix.
    """

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    display_cm = normalize_confusion_matrix(cm, mode="true") if normalize else cm

    if title is None:
        title = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    image = ax.imshow(display_cm, interpolation="nearest", cmap=cmap)

    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = display_cm.max() / 2.0

    for i in range(display_cm.shape[0]):
        for j in range(display_cm.shape[1]):
            if normalize:
                text = f"{display_cm[i, j] * 100:.1f}%"
            else:
                text = str(int(display_cm[i, j]))

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if display_cm[i, j] > threshold else "black",
                fontsize=9,
            )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def class_error_summary(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Summarize correct predictions and errors for each class.
    """

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    rows = []

    for i, class_name in enumerate(class_names):
        total = int(cm[i, :].sum())
        correct = int(cm[i, i])
        errors = total - correct
        error_rate = errors / total if total > 0 else 0.0

        rows.append(
            {
                "Class": class_name,
                "Total Samples": total,
                "Correct": correct,
                "Errors": errors,
                "Error Rate (%)": round(error_rate * 100, 2),
            }
        )

    return pd.DataFrame(rows)


def misclassification_pairs(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Identify the largest off-diagonal misclassification pairs.
    """

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    rows = []

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                rows.append(
                    {
                        "True Class": class_names[i],
                        "Predicted Class": class_names[j],
                        "Count": int(cm[i, j]),
                    }
                )

    table = pd.DataFrame(rows)

    if table.empty:
        return pd.DataFrame(
            columns=["True Class", "Predicted Class", "Count"]
        )

    table = table.sort_values(
        by="Count",
        ascending=False,
    ).head(top_k)

    return table.reset_index(drop=True)


def save_confusion_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str | Path = "outputs/confusion",
    class_names: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Compute and save all confusion matrix outputs.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    cm = compute_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=list(range(len(class_names))),
    )

    cm_norm = normalize_confusion_matrix(cm, mode="true")

    raw_df = confusion_matrix_to_dataframe(cm, class_names)
    norm_df = confusion_matrix_to_dataframe(cm_norm, class_names)

    error_df = class_error_summary(cm, class_names)
    pair_df = misclassification_pairs(cm, class_names)

    raw_df.to_csv(output_dir / "confusion_matrix_raw.csv")
    norm_df.to_csv(output_dir / "confusion_matrix_normalized.csv")
    error_df.to_csv(output_dir / "class_error_summary.csv", index=False)
    pair_df.to_csv(output_dir / "misclassification_pairs.csv", index=False)

    plot_confusion_matrix(
        cm,
        class_names=class_names,
        normalize=False,
        title="PapsAI XNet Confusion Matrix",
        save_path=output_dir / "confusion_matrix_raw.png",
        show=False,
    )

    plot_confusion_matrix(
        cm,
        class_names=class_names,
        normalize=True,
        title="PapsAI XNet Normalized Confusion Matrix",
        save_path=output_dir / "confusion_matrix_normalized.png",
        show=False,
    )

    return {
        "confusion_matrix": cm,
        "normalized_confusion_matrix": cm_norm,
        "raw_dataframe": raw_df,
        "normalized_dataframe": norm_df,
        "class_error_summary": error_df,
        "misclassification_pairs": pair_df,
    }
