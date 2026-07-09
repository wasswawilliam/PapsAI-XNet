# src/evaluation/roc.py

"""
===========================================================
PapsAI XNet ROC Analysis
===========================================================

Receiver Operating Characteristic (ROC) analysis for
multi-class cervical cytology classification.

Implements:

• Class-wise ROC curves
• Class-wise AUC
• Micro-average ROC
• Macro-average ROC
• Publication-quality ROC plots

Compatible with the evaluation protocol reported in the
PapsAI XNet manuscript.

===========================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize


##########################################################################
# Default Herlev class names
##########################################################################

DEFAULT_CLASS_NAMES = [

    "Superficial",

    "Intermediate",

    "Columnar",

    "Mild Dysplasia",

    "Moderate Dysplasia",

    "Severe Dysplasia",

    "Carcinoma In Situ"

]


##########################################################################
# ROC computation
##########################################################################

def compute_multiclass_roc(
        y_true,
        probabilities,
        class_names: Optional[List[str]] = None
):

    """
    Compute One-vs-Rest ROC curves.

    Parameters
    ----------

    y_true

    probabilities

        Shape:

        (N, number_of_classes)

    Returns
    -------

    Dictionary containing

    FPR

    TPR

    AUC

    Micro ROC

    Macro ROC
    """

    if class_names is None:

        class_names = DEFAULT_CLASS_NAMES

    number_of_classes = len(class_names)

    y_true = np.asarray(y_true)

    y_true = label_binarize(

        y_true,

        classes=np.arange(number_of_classes)

    )

    fpr = {}

    tpr = {}

    roc_auc = {}

    ############################################################
    # Individual ROC
    ############################################################

    for i in range(number_of_classes):

        fpr[i], tpr[i], _ = roc_curve(

            y_true[:, i],

            probabilities[:, i]

        )

        roc_auc[i] = auc(

            fpr[i],

            tpr[i]

        )

    ############################################################
    # Micro Average
    ############################################################

    fpr["micro"], tpr["micro"], _ = roc_curve(

        y_true.ravel(),

        probabilities.ravel()

    )

    roc_auc["micro"] = auc(

        fpr["micro"],

        tpr["micro"]

    )

    ############################################################
    # Macro Average
    ############################################################

    all_fpr = np.unique(

        np.concatenate(

            [fpr[i] for i in range(number_of_classes)]

        )

    )

    mean_tpr = np.zeros_like(all_fpr)

    for i in range(number_of_classes):

        mean_tpr += np.interp(

            all_fpr,

            fpr[i],

            tpr[i]

        )

    mean_tpr /= number_of_classes

    fpr["macro"] = all_fpr

    tpr["macro"] = mean_tpr

    roc_auc["macro"] = auc(

        all_fpr,

        mean_tpr

    )

    return {

        "fpr": fpr,

        "tpr": tpr,

        "auc": roc_auc,

        "class_names": class_names

    }


##########################################################################
# Plot ROC
##########################################################################

def plot_roc_curves(
        roc_results,
        figsize=(8,8),
        dpi=300,
        save_path=None,
        show=True
):

    """
    Plot publication-quality ROC curves.
    """

    plt.figure(

        figsize=figsize,

        dpi=dpi

    )

    ############################################################

    plt.plot(

        roc_results["fpr"]["macro"],

        roc_results["tpr"]["macro"],

        linewidth=3,

        label=f"Macro Average (AUC={roc_results['auc']['macro']:.3f})"

    )

    ############################################################

    plt.plot(

        roc_results["fpr"]["micro"],

        roc_results["tpr"]["micro"],

        linewidth=3,

        linestyle="--",

        label=f"Micro Average (AUC={roc_results['auc']['micro']:.3f})"

    )

    ############################################################

    for i, name in enumerate(

            roc_results["class_names"]

    ):

        plt.plot(

            roc_results["fpr"][i],

            roc_results["tpr"][i],

            linewidth=2,

            label=f"{name} (AUC={roc_results['auc'][i]:.3f})"

        )

    ############################################################

    plt.plot(

        [0,1],

        [0,1],

        "k--",

        linewidth=1

    )

    ############################################################

    plt.xlabel(

        "False Positive Rate",

        fontsize=12

    )

    plt.ylabel(

        "True Positive Rate",

        fontsize=12

    )

    plt.title(

        "Multi-Class ROC Curves",

        fontsize=14,

        weight="bold"

    )

    plt.legend(

        fontsize=8,

        loc="lower right"

    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    ############################################################

    if save_path is not None:

        save_path = Path(save_path)

        save_path.parent.mkdir(

            parents=True,

            exist_ok=True

        )

        plt.savefig(

            save_path,

            dpi=dpi,

            bbox_inches="tight"

        )

    if show:

        plt.show()

    plt.close()


##########################################################################
# Summary Table
##########################################################################

def roc_summary(roc_results):

    """
    Print ROC summary.
    """

    print()

    print("="*60)

    print("ROC ANALYSIS")

    print("="*60)

    for i, name in enumerate(

            roc_results["class_names"]

    ):

        print(

            f"{name:<30}"

            f"AUC = "

            f"{roc_results['auc'][i]:.4f}"

        )

    print("-"*60)

    print(

        f"Macro Average AUC : "

        f"{roc_results['auc']['macro']:.4f}"

    )

    print(

        f"Micro Average AUC : "

        f"{roc_results['auc']['micro']:.4f}"

    )

    print("="*60)


##########################################################################
# Save ROC Results
##########################################################################

def save_auc_table(
        roc_results,
        output_file
):

    """
    Save AUC values to CSV.
    """

    import pandas as pd

    rows = []

    for i, name in enumerate(

            roc_results["class_names"]

    ):

        rows.append({

            "Class":name,

            "AUC":roc_results["auc"][i]

        })

    rows.append({

        "Class":"Macro Average",

        "AUC":roc_results["auc"]["macro"]

    })

    rows.append({

        "Class":"Micro Average",

        "AUC":roc_results["auc"]["micro"]

    })

    df = pd.DataFrame(rows)

    df.to_csv(

        output_file,

        index=False

    )

    return df
