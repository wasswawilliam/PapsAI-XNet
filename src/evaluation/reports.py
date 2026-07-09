# src/evaluation/reports.py

"""
Reporting utilities for PapsAI XNet.

This module generates manuscript-ready tables and CSV summaries for:
- Overall model performance
- Class-wise performance
- ROC/AUC analysis
- Computational complexity
- Quantitative explainability
- McNemar statistical comparison
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd


def ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def percentage(value: float, decimals: int = 1) -> float:
    return round(float(value) * 100, decimals)


def create_overall_performance_table(
    model_results: Dict[str, Dict[str, float]],
    save_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Create a table comparing overall model performance.

    Expected input:
    {
        "PapsAI XNet": {
            "accuracy": 0.978,
            "sensitivity": 0.964,
            "specificity": 0.986,
            "precision": 0.971,
            "f1_score": 0.967,
            "auc": 0.985
        }
    }
    """

    rows = []

    for model_name, metrics in model_results.items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy (%)": percentage(metrics.get("accuracy", np.nan)),
                "Sensitivity (%)": percentage(metrics.get("sensitivity", np.nan)),
                "Specificity (%)": percentage(metrics.get("specificity", np.nan)),
                "Precision (%)": percentage(metrics.get("precision", np.nan)),
                "F1-score (%)": percentage(metrics.get("f1_score", np.nan)),
                "AUC": round(float(metrics.get("auc", np.nan)), 3),
            }
        )

    table = pd.DataFrame(rows)

    if save_path is not None:
        save_table(table, save_path)

    return table


def create_classwise_performance_table(
    classwise_results: Dict[str, Dict[str, float]],
    save_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Create manuscript-ready class-wise performance table.
    """

    rows = []

    for class_name, values in classwise_results.items():
        rows.append(
            {
                "Class": class_name,
                "Sensitivity (%)": percentage(values.get("sensitivity", np.nan)),
                "Specificity (%)": percentage(values.get("specificity", np.nan)),
                "Precision (%)": percentage(values.get("precision", np.nan)),
                "F1-score (%)": percentage(values.get("f1_score", np.nan)),
                "AUC": round(float(values.get("auc", np.nan)), 3),
            }
        )

    table = pd.DataFrame(rows)

    if save_path is not None:
        save_table(table, save_path)

    return table


def create_auc_table(
    auc_results: Dict[str, Any],
    save_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Create AUC table from ROC results.
    """

    rows = []

    class_names = auc_results.get("class_names", [])

    for i, class_name in enumerate(class_names):
        rows.append(
            {
                "Class": class_name,
                "AUC": round(float(auc_results["auc"][i]), 3),
            }
        )

    if "macro" in auc_results.get("auc", {}):
        rows.append(
            {
                "Class": "Macro Average",
                "AUC": round(float(auc_results["auc"]["macro"]), 3),
            }
        )

    if "micro" in auc_results.get("auc", {}):
        rows.append(
            {
                "Class": "Micro Average",
                "AUC": round(float(auc_results["auc"]["micro"]), 3),
            }
        )

    table = pd.DataFrame(rows)

    if save_path is not None:
        save_table(table, save_path)

    return table


def create_computational_complexity_table(
    complexity_results: Optional[List[Dict[str, float]]] = None,
    save_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Create computational complexity table.

    If no values are supplied, the manuscript values are used.
    """

    if complexity_results is None:
        complexity_results = [
            {
                "Model": "MobileNetV2",
                "Parameters (M)": 3.5,
                "FLOPs (G)": 0.30,
                "Inference Time (ms/image)": 7.2,
            },
            {
                "Model": "ResNet-18",
                "Parameters (M)": 11.7,
                "FLOPs (G)": 1.81,
                "Inference Time (ms/image)": 12.8,
            },
            {
                "Model": "PapsAI XNet",
                "Parameters (M)": 2.0,
                "FLOPs (G)": 0.32,
                "Inference Time (ms/image)": 7.0,
            },
        ]

    table = pd.DataFrame(complexity_results)

    if save_path is not None:
        save_table(table, save_path)

    return table


def create_explainability_table(
    explainability_results: Optional[Dict[str, float]] = None,
    save_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Create quantitative explainability metrics table.

    If no values are supplied, the manuscript values are used.
    """

    if explainability_results is None:
        explainability_results = {
            "Rule Coverage": "96.8%",
            "Rule Utilization": "82.4%",
            "Rule Consistency Score": 0.89,
            "Average Active Rules per Sample": 4.7,
            "Mean Rule Firing Strength": 0.74,
        }

    rows = [
        {"Metric": key, "Value": value}
        for key, value in explainability_results.items()
    ]

    table = pd.DataFrame(rows)

    if save_path is not None:
        save_table(table, save_path)

    return table


def create_mcnemar_table(
    mcnemar_results: Dict[str, Dict[str, float]],
    save_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Create McNemar statistical significance table.
    """

    rows = []

    for comparison_name, values in mcnemar_results.items():
        rows.append(
            {
                "Comparison": f"PapsAI XNet vs {comparison_name}",
                "b": values.get("b_model_a_correct_model_b_wrong", np.nan),
                "c": values.get("c_model_a_wrong_model_b_correct", np.nan),
                "McNemar Statistic": round(float(values.get("statistic", np.nan)), 4),
                "p-value": round(float(values.get("p_value", np.nan)), 4),
                "Significant (p < 0.05)": bool(values.get("p_value", 1.0) < 0.05),
            }
        )

    table = pd.DataFrame(rows)

    if save_path is not None:
        save_table(table, save_path)

    return table


def create_baseline_comparison_table(
    save_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Create the conceptual comparison table used in the manuscript.
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
            "Key Limitation": "Computationally demanding with limited explicit rule-based explanation",
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

    table = pd.DataFrame(rows)

    if save_path is not None:
        save_table(table, save_path)

    return table


def save_table(table: pd.DataFrame, save_path: str | Path) -> None:
    """
    Save table as CSV, Excel, Markdown, or LaTeX based on file extension.
    """

    save_path = Path(save_path)
    ensure_directory(save_path.parent)

    suffix = save_path.suffix.lower()

    if suffix == ".csv":
        table.to_csv(save_path, index=False)

    elif suffix in {".xlsx", ".xls"}:
        table.to_excel(save_path, index=False)

    elif suffix == ".md":
        save_path.write_text(table.to_markdown(index=False), encoding="utf-8")

    elif suffix == ".tex":
        save_path.write_text(table.to_latex(index=False), encoding="utf-8")

    else:
        raise ValueError(
            "Unsupported file format. Use .csv, .xlsx, .md, or .tex"
        )


def generate_all_manuscript_tables(
    output_dir: str | Path = "outputs/tables",
    overall_results: Optional[Dict[str, Dict[str, float]]] = None,
    classwise_results: Optional[Dict[str, Dict[str, float]]] = None,
    auc_results: Optional[Dict[str, Any]] = None,
    mcnemar_results: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Generate all standard manuscript tables used by the PapsAI XNet paper.
    """

    output_dir = ensure_directory(output_dir)
    tables = {}

    tables["computational_complexity"] = create_computational_complexity_table(
        output_dir / "computational_complexity.csv"
    )

    tables["explainability"] = create_explainability_table(
        save_path=output_dir / "explainability_metrics.csv"
    )

    tables["baseline_comparison"] = create_baseline_comparison_table(
        save_path=output_dir / "baseline_comparison.csv"
    )

    if overall_results is not None:
        tables["overall_performance"] = create_overall_performance_table(
            overall_results,
            save_path=output_dir / "overall_performance.csv",
        )

    if classwise_results is not None:
        tables["classwise_performance"] = create_classwise_performance_table(
            classwise_results,
            save_path=output_dir / "classwise_performance.csv",
        )

    if auc_results is not None:
        tables["auc"] = create_auc_table(
            auc_results,
            save_path=output_dir / "auc_table.csv",
        )

    if mcnemar_results is not None:
        tables["mcnemar"] = create_mcnemar_table(
            mcnemar_results,
            save_path=output_dir / "mcnemar_test.csv",
        )

    return tables
