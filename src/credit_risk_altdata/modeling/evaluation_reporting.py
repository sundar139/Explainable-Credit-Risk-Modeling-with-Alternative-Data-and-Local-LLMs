"""Evaluation artifact generation for tuned modeling workflows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.calibration import calibration_curve  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from credit_risk_altdata.config import Settings
from credit_risk_altdata.modeling.metrics import compute_classification_metrics
from credit_risk_altdata.modeling.plots import (
    save_calibration_curve_plot,
    save_pr_curve_plot,
    save_probability_distribution_plot,
    save_roc_curve_plot,
)
from credit_risk_altdata.utils.filesystem import ensure_directories


def build_threshold_grid(settings: Settings) -> NDArray[np.float64]:
    """Build threshold grid from settings."""
    thresholds = np.arange(
        settings.modeling_threshold_grid_min,
        settings.modeling_threshold_grid_max + (settings.modeling_threshold_grid_step / 2.0),
        settings.modeling_threshold_grid_step,
        dtype=float,
    )
    thresholds = np.unique(np.round(np.clip(thresholds, 0.0, 1.0), 6))
    if thresholds.size == 0:
        raise ValueError("Threshold grid is empty; check MODELING_THRESHOLD_GRID_* settings")
    return thresholds.astype(np.float64)


def generate_threshold_analysis(
    *,
    y_true: NDArray[np.int_],
    y_prob: NDArray[np.float64],
    thresholds: NDArray[np.float64],
) -> DataFrame:
    """Generate threshold-level metric table."""
    records: list[dict[str, float]] = []
    for threshold in thresholds:
        metrics = compute_classification_metrics(
            y_true=y_true,
            y_prob=y_prob,
            threshold=float(threshold),
        )
        positive_rate = float(np.mean(y_prob >= threshold))
        metrics["positive_prediction_rate"] = positive_rate
        records.append(metrics)

    threshold_table = pd.DataFrame(records)
    return threshold_table.sort_values("threshold").reset_index(drop=True)


def build_gain_lift_summary(
    *,
    y_true: NDArray[np.int_],
    y_prob: NDArray[np.float64],
    n_bins: int = 10,
) -> DataFrame:
    """Build gain/lift style summary table from ranked predictions."""
    if y_true.size == 0:
        return pd.DataFrame(
            columns=[
                "bin",
                "population_count",
                "event_count",
                "event_rate",
                "lift",
                "cumulative_population_share",
                "cumulative_event_share",
            ]
        )

    ranked = pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).sort_values(
        "y_prob", ascending=False
    )
    ranked = ranked.reset_index(drop=True)

    effective_bins = int(min(n_bins, len(ranked)))
    ranked["bin"] = pd.qcut(
        np.arange(len(ranked)),
        q=effective_bins,
        labels=False,
        duplicates="drop",
    )
    ranked["bin"] = ranked["bin"].astype(int) + 1

    grouped = ranked.groupby("bin", as_index=False).agg(
        population_count=("y_true", "size"),
        event_count=("y_true", "sum"),
    )
    grouped = grouped.sort_values("bin").reset_index(drop=True)

    total_events = float(grouped["event_count"].sum())
    total_population = float(grouped["population_count"].sum())
    base_event_rate = float(total_events / total_population) if total_population else 0.0

    grouped["event_rate"] = grouped["event_count"] / grouped["population_count"]
    grouped["lift"] = grouped["event_rate"] / base_event_rate if base_event_rate > 0 else 0.0
    grouped["cumulative_population_share"] = grouped["population_count"].cumsum() / total_population
    grouped["cumulative_event_share"] = (
        grouped["event_count"].cumsum() / total_events if total_events > 0 else 0.0
    )
    return grouped


def build_class_distribution_summary(y_true: NDArray[np.int_]) -> dict[str, float | int]:
    """Summarize class balance for evaluation sample."""
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    total = int(y_true.size)
    positive_rate = float(positives / total) if total > 0 else 0.0
    return {
        "total_count": total,
        "positive_count": positives,
        "negative_count": negatives,
        "positive_rate": positive_rate,
    }


def build_probability_distribution_summary(
    probabilities: NDArray[np.float64],
    *,
    n_bins: int = 20,
) -> DataFrame:
    """Build histogram table for predicted probabilities."""
    hist, bin_edges = np.histogram(probabilities, bins=n_bins, range=(0.0, 1.0))
    return pd.DataFrame(
        {
            "bin_left": bin_edges[:-1],
            "bin_right": bin_edges[1:],
            "count": hist,
        }
    )


def generate_evaluation_artifacts(
    *,
    y_true: NDArray[np.int_],
    y_prob: NDArray[np.float64],
    thresholds: NDArray[np.float64],
    evaluation_dir: Path,
) -> tuple[dict[str, float | int], dict[str, Path]]:
    """Generate evaluation curves, plots, and summary tables."""
    ensure_directories([evaluation_dir])

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = float(roc_auc_score(y_true, y_prob))
    roc_curve_frame = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    roc_curve_data_path = evaluation_dir / "roc_curve_data.csv"
    roc_curve_frame.to_csv(roc_curve_data_path, index=False)
    roc_curve_plot_path = save_roc_curve_plot(
        fpr=np.asarray(fpr, dtype=np.float64),
        tpr=np.asarray(tpr, dtype=np.float64),
        auc_value=roc_auc,
        output_path=evaluation_dir / "roc_curve.png",
    )

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(average_precision_score(y_true, y_prob))
    pr_curve_frame = pd.DataFrame({"recall": recall, "precision": precision})
    pr_curve_data_path = evaluation_dir / "pr_curve_data.csv"
    pr_curve_frame.to_csv(pr_curve_data_path, index=False)
    pr_curve_plot_path = save_pr_curve_plot(
        recall=np.asarray(recall, dtype=np.float64),
        precision=np.asarray(precision, dtype=np.float64),
        average_precision=pr_auc,
        output_path=evaluation_dir / "pr_curve.png",
    )

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    calibration_curve_frame = pd.DataFrame(
        {
            "mean_predicted_probability": prob_pred,
            "observed_frequency": prob_true,
        }
    )
    calibration_curve_data_path = evaluation_dir / "calibration_curve_data.csv"
    calibration_curve_frame.to_csv(calibration_curve_data_path, index=False)
    calibration_curve_plot_path = save_calibration_curve_plot(
        prob_pred=np.asarray(prob_pred, dtype=np.float64),
        prob_true=np.asarray(prob_true, dtype=np.float64),
        output_path=evaluation_dir / "calibration_curve.png",
    )

    threshold_analysis = generate_threshold_analysis(
        y_true=y_true,
        y_prob=y_prob,
        thresholds=thresholds,
    )
    threshold_analysis_path = evaluation_dir / "threshold_analysis.csv"
    threshold_analysis.to_csv(threshold_analysis_path, index=False)

    gain_lift_summary = build_gain_lift_summary(y_true=y_true, y_prob=y_prob)
    gain_lift_summary_path = evaluation_dir / "gain_lift_summary.csv"
    gain_lift_summary.to_csv(gain_lift_summary_path, index=False)

    class_distribution = build_class_distribution_summary(y_true)
    class_distribution_path = evaluation_dir / "class_distribution_summary.json"
    class_distribution_path.write_text(json.dumps(class_distribution, indent=2), encoding="utf-8")

    probability_distribution = build_probability_distribution_summary(y_prob)
    probability_distribution_path = evaluation_dir / "probability_distribution_summary.csv"
    probability_distribution.to_csv(probability_distribution_path, index=False)
    probability_distribution_plot_path = save_probability_distribution_plot(
        probabilities=y_prob,
        output_path=evaluation_dir / "probability_distribution.png",
    )

    summary = {
        "row_count": int(len(y_true)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "positive_rate": float(class_distribution["positive_rate"]),
    }
    artifact_paths = {
        "roc_curve_data": roc_curve_data_path,
        "roc_curve_plot": roc_curve_plot_path,
        "pr_curve_data": pr_curve_data_path,
        "pr_curve_plot": pr_curve_plot_path,
        "calibration_curve_data": calibration_curve_data_path,
        "calibration_curve_plot": calibration_curve_plot_path,
        "threshold_analysis": threshold_analysis_path,
        "gain_lift_summary": gain_lift_summary_path,
        "class_distribution": class_distribution_path,
        "probability_distribution": probability_distribution_path,
        "probability_distribution_plot": probability_distribution_plot_path,
    }
    return summary, artifact_paths
