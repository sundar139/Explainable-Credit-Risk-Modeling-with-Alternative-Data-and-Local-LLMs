"""Plot helpers for modeling evaluation artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()


def save_roc_curve_plot(
    *,
    fpr: NDArray[np.float64],
    tpr: NDArray[np.float64],
    auc_value: float,
    output_path: Path,
) -> Path:
    """Save ROC curve plot to PNG."""
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc_value:.4f}", color="#1f77b4")
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#666666")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    _save_figure(output_path)
    return output_path


def save_pr_curve_plot(
    *,
    recall: NDArray[np.float64],
    precision: NDArray[np.float64],
    average_precision: float,
    output_path: Path,
) -> Path:
    """Save precision-recall curve plot to PNG."""
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR AUC = {average_precision:.4f}", color="#2ca02c")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="best")
    _save_figure(output_path)
    return output_path


def save_calibration_curve_plot(
    *,
    prob_pred: NDArray[np.float64],
    prob_true: NDArray[np.float64],
    output_path: Path,
) -> Path:
    """Save calibration curve plot to PNG."""
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", color="#ff7f0e", label="Model")
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#666666", label="Perfect")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title("Calibration Curve")
    plt.legend(loc="best")
    _save_figure(output_path)
    return output_path


def save_probability_distribution_plot(
    *,
    probabilities: NDArray[np.float64],
    output_path: Path,
) -> Path:
    """Save prediction-probability histogram to PNG."""
    plt.figure(figsize=(6, 5))
    plt.hist(probabilities, bins=30, color="#9467bd", alpha=0.85, edgecolor="#ffffff")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Predicted Probability Distribution")
    _save_figure(output_path)
    return output_path
