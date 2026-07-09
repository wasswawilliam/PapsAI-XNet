# src/evaluation/evaluator.py

"""
Complete evaluation pipeline for PapsAI XNet.

This module evaluates a trained PapsAI XNet model on a test set and generates:
- Overall performance metrics
- Class-wise metrics
- Confusion matrix outputs
- ROC and AUC analysis
- Manuscript-ready tables
- Prediction arrays for statistical comparison
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import (
    full_evaluation_report,
    overall_metrics,
    classwise_metrics,
)

from src.evaluation.confusion import save_confusion_outputs

from src.evaluation.roc import (
    compute_multiclass_roc,
    plot_roc_curves,
    save_auc_table,
)

from src.evaluation.reports import (
    create_overall_performance_table,
    create_classwise_performance_table,
)


DEFAULT_CLASS_NAMES = [
    "Superficial Squamous Epithelial",
    "Intermediate Squamous Epithelial",
    "Columnar Epithelial",
    "Mild Dysplasia",
    "Moderate Dysplasia",
    "Severe Dysplasia",
    "Carcinoma in Situ",
]


class PapsAIXNetEvaluator:
    """
    Evaluation engine for PapsAI XNet.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        class_names: Optional[List[str]] = None,
        output_dir: str | Path = "outputs/evaluation",
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.class_names = class_names or DEFAULT_CLASS_NAMES

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def collect_predictions(self) -> Dict[str, np.ndarray]:
        """
        Run model inference and collect predictions, probabilities, and labels.
        """

        self.model.eval()

        all_labels = []
        all_predictions = []
        all_probabilities = []
        all_logits = []

        for images, labels in tqdm(self.dataloader, desc="Evaluating", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

        return {
            "y_true": np.concatenate(all_labels),
            "y_pred": np.concatenate(all_predictions),
            "y_prob": np.concatenate(all_probabilities),
            "logits": np.concatenate(all_logits),
        }

    def evaluate(self) -> Dict[str, object]:
        """
        Run complete evaluation and save outputs.
        """

        predictions = self.collect_predictions()

        y_true = predictions["y_true"]
        y_pred = predictions["y_pred"]
        y_prob = predictions["y_prob"]

        overall = overall_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
        )

        classwise = classwise_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            class_names=self.class_names,
        )

        full_report = full_evaluation_report(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            class_names=self.class_names,
        )

        confusion_outputs = save_confusion_outputs(
            y_true=y_true,
            y_pred=y_pred,
            output_dir=self.output_dir / "confusion",
            class_names=self.class_names,
        )

        roc_results = compute_multiclass_roc(
            y_true=y_true,
            probabilities=y_prob,
            class_names=self.class_names,
        )

        plot_roc_curves(
            roc_results=roc_results,
            save_path=self.output_dir / "roc" / "multiclass_roc.png",
            show=False,
        )

        save_auc_table(
            roc_results=roc_results,
            output_file=self.output_dir / "roc" / "auc_table.csv",
        )

        overall_table = create_overall_performance_table(
            {"PapsAI XNet": overall},
            save_path=self.output_dir / "tables" / "overall_performance.csv",
        )

        classwise_table = create_classwise_performance_table(
            classwise,
            save_path=self.output_dir / "tables" / "classwise_performance.csv",
        )

        self.save_predictions(predictions)

        return {
            "predictions": predictions,
            "overall_metrics": overall,
            "classwise_metrics": classwise,
            "full_report": full_report,
            "confusion_outputs": confusion_outputs,
            "roc_results": roc_results,
            "overall_table": overall_table,
            "classwise_table": classwise_table,
        }

    def save_predictions(self, predictions: Dict[str, np.ndarray]) -> None:
        """
        Save prediction arrays for reproducibility and statistical comparison.
        """

        output_path = self.output_dir / "predictions"
        output_path.mkdir(parents=True, exist_ok=True)

        np.save(output_path / "y_true.npy", predictions["y_true"])
        np.save(output_path / "y_pred.npy", predictions["y_pred"])
        np.save(output_path / "y_prob.npy", predictions["y_prob"])
        np.save(output_path / "logits.npy", predictions["logits"])

    def print_summary(self, results: Dict[str, object]) -> None:
        """
        Print compact evaluation summary.
        """

        metrics = results["overall_metrics"]

        print("\nPapsAI XNet Evaluation Summary")
        print("=" * 50)
        print(f"Accuracy     : {metrics['accuracy'] * 100:.2f}%")
        print(f"Sensitivity  : {metrics['sensitivity'] * 100:.2f}%")
        print(f"Specificity  : {metrics['specificity'] * 100:.2f}%")
        print(f"Precision    : {metrics['precision'] * 100:.2f}%")
        print(f"F1-score     : {metrics['f1_score'] * 100:.2f}%")

        if "auc" in metrics:
            print(f"AUC          : {metrics['auc']:.4f}")

        print("=" * 50)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    output_dir: str | Path = "outputs/evaluation",
) -> Dict[str, object]:
    """
    Convenience function for evaluating a trained model.
    """

    evaluator = PapsAIXNetEvaluator(
        model=model,
        dataloader=dataloader,
        device=device,
        class_names=class_names,
        output_dir=output_dir,
    )

    results = evaluator.evaluate()
    evaluator.print_summary(results)

    return results
