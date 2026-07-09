# src/evaluation/comparison.py

"""
Model comparison utilities for PapsAI XNet.

This module compares PapsAI XNet against baseline architectures such as
ResNet-18 and MobileNetV2, as described in the manuscript.

It supports:
- Overall metric comparison
- Class-wise comparison
- McNemar statistical testing
- Manuscript-ready summary tables
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import overall_metrics, classwise_metrics
from src.evaluation.statistics import compare_models_mcnemar


def compare_overall_performance(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    probabilities: Optional[Dict[str, np.ndarray]] = None,
) -> pd.DataFrame:
    """
    Compare overall performance of multiple models.

    Parameters
    ----------
    y_true:
        Ground-truth labels.

    predictions:
        Dictionary mapping model name to predicted labels.

    probabilities:
        Optional dictionary mapping model name to class probabilities.

    Returns
    -------
    DataFrame with overall metrics for each model.
    """

    rows = []

    for model_name, y_pred in predictions.items():
        y_prob = None

        if probabilities is not None and model_name in probabilities:
            y_prob = probabilities[model_name]

        metrics = overall_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
        )

        row = {
            "Model": model_name,
            "Accuracy (%)": round(metrics["accuracy"] * 100, 2),
            "Sensitivity (%)": round(metrics["sensitivity"] * 100, 2),
            "Specificity (%)": round(metrics["specificity"] * 100, 2),
            "Precision (%)": round(metrics["precision"] * 100, 2),
            "F1-score (%)": round(metrics["f1_score"] * 100, 2),
        }

        if "auc" in metrics:
            row["AUC"] = round(metrics["auc"], 4)

        rows.append(row)

    return pd.DataFrame(rows)


def compare_classwise_performance(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    probabilities: Optional[Dict[str, np.ndarray]] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Generate class-wise comparison tables for multiple models.

    Returns
    -------
    Dictionary where each key is a model name and each value is a DataFrame.
    """

    output = {}

    for model_name, y_pred in predictions.items():
        y_prob = None

        if probabilities is not None and model_name in probabilities:
            y_prob = probabilities[model_name]

        metrics = classwise_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            class_names=class_names,
        )

        rows = []

        for class_name, values in metrics.items():
            row = {
                "Class": class_name,
                "Sensitivity (%)": round(values["sensitivity"] * 100, 2),
                "Specificity (%)": round(values["specificity"] * 100, 2),
                "Precision (%)": round(values["precision"] * 100, 2),
                "F1-score (%)": round(values["f1_score"] * 100, 2),
            }

            if "auc" in values:
                row["AUC"] = round(values["auc"], 4)

            rows.append(row)

        output[model_name] = pd.DataFrame(rows)

    return output


def statistical_comparison_against_reference(
    y_true: np.ndarray,
    reference_model_name: str,
    predictions: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Compare a reference model, usually PapsAI XNet, against all baselines
    using McNemar's test.
    """

    if reference_model_name not in predictions:
        raise ValueError(
            f"Reference model '{reference_model_name}' not found in predictions."
        )

    reference_predictions = predictions[reference_model_name]

    baseline_predictions = {
        model_name: y_pred
        for model_name, y_pred in predictions.items()
        if model_name != reference_model_name
    }

    mcnemar_results = compare_models_mcnemar(
        y_true=y_true,
        reference_predictions=reference_predictions,
        baseline_predictions=baseline_predictions,
    )

    rows = []

    for baseline_name, result in mcnemar_results.items():
        rows.append(
            {
                "Comparison": f"{reference_model_name} vs {baseline_name}",
                "b": result["b_model_a_correct_model_b_wrong"],
                "c": result["c_model_a_wrong_model_b_correct"],
                "McNemar Statistic": round(result["statistic"], 4),
                "p-value": round(result["p_value"], 4),
                "Significant (p < 0.05)": result["p_value"] < 0.05,
            }
        )

    return pd.DataFrame(rows)


def create_manuscript_baseline_summary() -> pd.DataFrame:
    """
    Create the baseline comparison table reported in the manuscript.
    """

    rows = [
        {
            "Model": "ResNet-18",
            "Accuracy (%)": 95.4,
            "Sensitivity (%)": 93.1,
            "Specificity (%)": 96.2,
            "Precision (%)": 94.0,
            "F1-score (%)": 93.5,
        },
        {
            "Model": "MobileNetV2",
            "Accuracy (%)": 94.8,
            "Sensitivity (%)": 92.4,
            "Specificity (%)": 95.3,
            "Precision (%)": 93.1,
            "F1-score (%)": 92.7,
        },
        {
            "Model": "PapsAI XNet",
            "Accuracy (%)": 97.8,
            "Sensitivity (%)": 96.4,
            "Specificity (%)": 98.6,
            "Precision (%)": 97.1,
            "F1-score (%)": 96.7,
        },
    ]

    return pd.DataFrame(rows)


def create_conceptual_comparison_table() -> pd.DataFrame:
    """
    Create the conceptual comparison table positioning PapsAI XNet
    against recent deep learning and explainable AI approaches.
    """

    rows = [
        {
            "Model/Approach": "ResNet/VGG-based CNNs",
            "Core Method": "Deep convolutional feature extraction",
            "Main Strength": "Strong local morphology learning",
            "Key Limitation": "Limited interpretability and weaker performance in borderline classes",
            "Relevance to PapsAI XNet": "Baseline CNN comparison",
        },
        {
            "Model/Approach": "MobileNetV2",
            "Core Method": "Lightweight CNN",
            "Main Strength": "Efficient and suitable for low-resource deployment",
            "Key Limitation": "Lower sensitivity in borderline dysplasia",
            "Relevance to PapsAI XNet": "Lightweight baseline comparison",
        },
        {
            "Model/Approach": "CerviFormer",
            "Core Method": "Cross-attention and latent transformer",
            "Main Strength": "Captures long-range spatial dependencies",
            "Key Limitation": "Computationally demanding and limited explicit rule-based explanation",
            "Relevance to PapsAI XNet": "Contemporary transformer benchmark",
        },
        {
            "Model/Approach": "CNN–Transformer hybrids",
            "Core Method": "Combined local and global feature learning",
            "Main Strength": "Improved representation of complex image patterns",
            "Key Limitation": "Requires larger data and offers limited transparent reasoning",
            "Relevance to PapsAI XNet": "Recent hybrid deep learning comparison",
        },
        {
            "Model/Approach": "PapsAI XNet",
            "Core Method": "Attention-enhanced CNN + feature normalization + ANFIS",
            "Main Strength": "Combines high performance with explicit fuzzy-rule interpretability",
            "Key Limitation": "Requires further validation on additional datasets",
            "Relevance to PapsAI XNet": "Proposed explainable hybrid model",
        },
    ]

    return pd.DataFrame(rows)


def save_comparison_tables(
    output_dir: str | Path = "outputs/tables",
    overall_table: Optional[pd.DataFrame] = None,
    classwise_tables: Optional[Dict[str, pd.DataFrame]] = None,
    statistical_table: Optional[pd.DataFrame] = None,
) -> None:
    """
    Save comparison tables to CSV files.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if overall_table is not None:
        overall_table.to_csv(
            output_dir / "overall_model_comparison.csv",
            index=False,
        )

    if classwise_tables is not None:
        for model_name, table in classwise_tables.items():
            safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
            table.to_csv(
                output_dir / f"classwise_{safe_name}.csv",
                index=False,
            )

    if statistical_table is not None:
        statistical_table.to_csv(
            output_dir / "mcnemar_statistical_comparison.csv",
            index=False,
        )

    create_manuscript_baseline_summary().to_csv(
        output_dir / "manuscript_baseline_summary.csv",
        index=False,
    )

    create_conceptual_comparison_table().to_csv(
        output_dir / "conceptual_comparison_table.csv",
        index=False,
    )


def run_complete_model_comparison(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    probabilities: Optional[Dict[str, np.ndarray]] = None,
    class_names: Optional[List[str]] = None,
    reference_model_name: str = "PapsAI XNet",
    output_dir: str | Path = "outputs/tables",
) -> Dict[str, object]:
    """
    Run complete model comparison and save all outputs.
    """

    overall_table = compare_overall_performance(
        y_true=y_true,
        predictions=predictions,
        probabilities=probabilities,
    )

    classwise_tables = compare_classwise_performance(
        y_true=y_true,
        predictions=predictions,
        probabilities=probabilities,
        class_names=class_names,
    )

    statistical_table = statistical_comparison_against_reference(
        y_true=y_true,
        reference_model_name=reference_model_name,
        predictions=predictions,
    )

    save_comparison_tables(
        output_dir=output_dir,
        overall_table=overall_table,
        classwise_tables=classwise_tables,
        statistical_table=statistical_table,
    )

    return {
        "overall_table": overall_table,
        "classwise_tables": classwise_tables,
        "statistical_table": statistical_table,
        "manuscript_baseline_summary": create_manuscript_baseline_summary(),
        "conceptual_comparison_table": create_conceptual_comparison_table(),
    }
