# src/evaluation/statistics.py

"""
Statistical validation utilities for PapsAI XNet.

Implements:
- Bootstrap confidence intervals
- Metric standard deviation estimation
- McNemar's test for paired model comparison
- Statistical comparison between PapsAI XNet and baseline models
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
from scipy.stats import chi2


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> Tuple[float, float, float, float]:
    """
    Compute bootstrap mean, standard deviation, and confidence interval.

    Parameters
    ----------
    y_true:
        Ground-truth labels.

    y_pred:
        Predicted labels.

    metric_fn:
        Function that accepts y_true and y_pred.

    n_bootstrap:
        Number of bootstrap samples.

    confidence:
        Confidence level.

    random_seed:
        Random seed.

    Returns
    -------
    mean_metric, std_metric, lower_ci, upper_ci
    """

    rng = np.random.default_rng(random_seed)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_samples = len(y_true)
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)

        score = metric_fn(
            y_true[indices],
            y_pred[indices],
        )

        bootstrap_scores.append(score)

    bootstrap_scores = np.asarray(bootstrap_scores)

    alpha = 1.0 - confidence

    lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
    upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))

    return (
        float(np.mean(bootstrap_scores)),
        float(np.std(bootstrap_scores, ddof=1)),
        float(lower),
        float(upper),
    )


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_model_a: np.ndarray,
    y_pred_model_b: np.ndarray,
    continuity_correction: bool = True,
) -> Dict[str, float]:
    """
    Perform McNemar's test for paired classification outcomes.

    Model A is usually PapsAI XNet.
    Model B is usually ResNet-18 or MobileNetV2.

    Returns
    -------
    Dictionary with:
        b:
            Model A correct, Model B incorrect.

        c:
            Model A incorrect, Model B correct.

        statistic:
            McNemar chi-square statistic.

        p_value:
            Statistical significance value.
    """

    y_true = np.asarray(y_true)
    y_pred_model_a = np.asarray(y_pred_model_a)
    y_pred_model_b = np.asarray(y_pred_model_b)

    correct_a = y_pred_model_a == y_true
    correct_b = y_pred_model_b == y_true

    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)

    denominator = b + c

    if denominator == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        if continuity_correction:
            statistic = (abs(b - c) - 1) ** 2 / denominator
        else:
            statistic = (b - c) ** 2 / denominator

        p_value = 1.0 - chi2.cdf(statistic, df=1)

    return {
        "b_model_a_correct_model_b_wrong": int(b),
        "c_model_a_wrong_model_b_correct": int(c),
        "statistic": float(statistic),
        "p_value": float(p_value),
    }


def compare_models_mcnemar(
    y_true: np.ndarray,
    reference_predictions: np.ndarray,
    baseline_predictions: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Compare PapsAI XNet against multiple baselines using McNemar's test.
    """

    results = {}

    for model_name, predictions in baseline_predictions.items():
        results[model_name] = mcnemar_test(
            y_true=y_true,
            y_pred_model_a=reference_predictions,
            y_pred_model_b=predictions,
        )

    return results


def format_confidence_interval(
    mean: float,
    std: float,
    lower: float,
    upper: float,
    percentage: bool = True,
) -> str:
    """
    Format metric result for manuscript tables.
    """

    if percentage:
        return (
            f"{mean * 100:.2f} ± {std * 100:.2f} "
            f"({lower * 100:.2f}–{upper * 100:.2f})"
        )

    return f"{mean:.4f} ± {std:.4f} ({lower:.4f}–{upper:.4f})"
