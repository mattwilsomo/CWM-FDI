#!/usr/bin/env python3
"""Create the main visualisations for the ML sustainability mini-project.

This script reads the profiled-model CSV produced by the benchmarking stage and
saves a small set of plots that tell the project story:

1. A model comparison table in image form.
2. Bar charts for accuracy, timing, and energy.
3. Pareto-style scatter plots showing tradeoffs.
4. Workload-sensitivity curves showing how total cost changes as the number of
   predictions increases.

The script is intentionally verbose and heavily commented so that the reasoning
is easy to follow and so it is straightforward to adapt if the CSV column names
change slightly.

Expected CSV columns (the script tries common variants):
- model / Model
- ap_score / AP score / AUPRC
- train_time_s / train_time / Train time
- inference_time_s / inference_time / Inference time
- train_energy_j / train_energy / Train energy
- inference_energy_j / inference_energy / Inference energy
- carbon_footprint (optional)

Outputs are written to a 'figures' directory next to this script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

# Use a clean, consistent look for all plots.
plt.style.use("seaborn-v0_8-whitegrid")

# Base paths are derived from the location of this script so that the file works
# no matter where it is launched from, as long as the CSV is in the same folder.
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "model_profiles.csv"
FIG_DIR = BASE_DIR / "figures"

# How many prediction counts we want to show in the workload-sensitivity plot.
# These values are chosen to show the “small workload vs large workload” effect.
WORKLOAD_POINTS = [1, 10, 100, 1_000, 10_000, 100_000]


def get_col(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    """Return the first matching column name from a list of alternatives."""
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"Missing one of these columns: {list(candidates)}")


def load_profiles(csv_path: Path) -> pd.DataFrame:
    """Load the profiling CSV and normalise a few known column names.

    The profiler output has changed a few times while the project was being
    built, so this function accepts a few common variants and reduces the rest
    of the script to a stable internal schema.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV file: {csv_path}")

    df = pd.read_csv(csv_path)

    # Map the CSV into a stable set of column names used by the plotting code.
    model_col = get_col(df, ["model", "Model"])
    ap_col = get_col(df, ["ap_score", "AP score", "AUPRC"])
    train_time_col = get_col(df, ["train_time_s", "train_time", "Train time"])
    inference_time_col = get_col(df, ["inference_time_s", "inference_time", "Inference time"])
    train_energy_col = get_col(df, ["train_energy_j", "train_energy", "Train energy"])
    inference_energy_col = get_col(df, ["inference_energy_j", "inference_energy", "Inference energy"])

    out = pd.DataFrame(
        {
            "model": df[model_col],
            "ap_score": pd.to_numeric(df[ap_col], errors="coerce"),
            "train_time_s": pd.to_numeric(df[train_time_col], errors="coerce"),
            "inference_time_s": pd.to_numeric(df[inference_time_col], errors="coerce"),
            "train_energy_j": pd.to_numeric(df[train_energy_col], errors="coerce"),
            "inference_energy_j": pd.to_numeric(df[inference_energy_col], errors="coerce"),
        }
    )

    # Carbon is optional in the CSV. If it exists, keep it. If not, create an
    # empty column so the rest of the script can rely on it safely.
    if "carbon_footprint" in df.columns:
        out["carbon_footprint"] = pd.to_numeric(df["carbon_footprint"], errors="coerce")
    else:
        out["carbon_footprint"] = pd.NA

    # Derived quantities are what we actually need for the workload tradeoff.
    out["total_time_s"] = out["train_time_s"] + out["inference_time_s"]
    out["total_energy_j"] = out["train_energy_j"] + out["inference_energy_j"]

    return out


def ensure_output_dir() -> None:
    """Create the figures directory if it does not exist."""
    FIG_DIR.mkdir(exist_ok=True)


def save_table_image(df: pd.DataFrame) -> Path:
    """Save a compact table as an image for the report.

    This is helpful because a single table can summarise the entire profiling
    stage and can be dropped straight into the report.
    """
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.axis("off")

    display_df = df.copy()
    for col in ["ap_score", "train_time_s", "inference_time_s", "train_energy_j", "inference_energy_j"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4g}" if pd.notna(x) else "N/A")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    out_path = FIG_DIR / "01_profile_table.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def bar_plot(df: pd.DataFrame, value_col: str, title: str, ylabel: str, filename: str) -> Path:
    """Simple bar chart across models for one metric.

    Bar charts are best for direct comparison of the profiled values.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["model"], df[value_col])
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)

    out_path = FIG_DIR / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def scatter_pareto(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, filename: str) -> Path:
    """Scatter plot to show a tradeoff frontier.

    This is the key plot for the project because it shows that no single model
    dominates every metric.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df[x_col], df[y_col], s=80)

    for _, row in df.iterrows():
        ax.annotate(
            row["model"],
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=9,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    out_path = FIG_DIR / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def add_workload_totals(df: pd.DataFrame, workload: int) -> pd.DataFrame:
    """Compute workload-dependent totals.

    The recommendation engine uses these exact formulas:
        total_time = train_time + workload * inference_time
        total_energy = train_energy + workload * inference_energy

    The visualisation uses the same formulas so the plots and the recommender are
    aligned.
    """
    out = df.copy()
    out["workload"] = workload
    out["workload_total_time_s"] = out["train_time_s"] + workload * out["inference_time_s"]
    out["workload_total_energy_j"] = out["train_energy_j"] + workload * out["inference_energy_j"]
    return out


def workload_sensitivity_plot(df: pd.DataFrame, metric: str, title: str, ylabel: str, filename: str) -> Path:
    """Plot a metric as a function of the expected number of inferences.

    This plot makes the key idea of the recommender visible: a model with a
    larger training cost can become better when the user will make enough
    predictions.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for _, row in df.iterrows():
        series = []
        for n in WORKLOAD_POINTS:
            if metric == "time":
                value = row["train_time_s"] + n * row["inference_time_s"]
            elif metric == "energy":
                value = row["train_energy_j"] + n * row["inference_energy_j"]
            else:
                raise ValueError("metric must be 'time' or 'energy'")
            series.append(value)

        ax.plot(WORKLOAD_POINTS, series, marker="o", label=row["model"])

    ax.set_xscale("log")
    ax.set_xlabel("Expected number of inferences (log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    out_path = FIG_DIR / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ensure_output_dir()
    df = load_profiles(DEFAULT_CSV)

    # Sort models by AP score so the table and bar charts are easy to read.
    df = df.sort_values("ap_score", ascending=False).reset_index(drop=True)

    print("Loaded model profiles:")
    print(df.to_string(index=False))

    outputs = []
    outputs.append(save_table_image(df[["model", "ap_score", "train_time_s", "inference_time_s", "train_energy_j", "inference_energy_j"]]))

    # Core comparison charts.
    outputs.append(bar_plot(df, "ap_score", "AUPRC / Average Precision by Model", "AP score", "02_ap_score_bar.png"))
    outputs.append(bar_plot(df, "train_time_s", "Training Time by Model", "Train time (s)", "03_train_time_bar.png"))
    outputs.append(bar_plot(df, "inference_time_s", "Inference Time by Model", "Inference time (s)", "04_inference_time_bar.png"))
    outputs.append(bar_plot(df, "train_energy_j", "Training Energy by Model", "Energy (J)", "05_train_energy_bar.png"))
    outputs.append(bar_plot(df, "inference_energy_j", "Inference Energy by Model", "Energy (J)", "06_inference_energy_bar.png"))

    # Tradeoff scatter plots.
    outputs.append(scatter_pareto(
        df,
        x_col="total_energy_j",
        y_col="ap_score",
        title="Accuracy vs Total Energy",
        xlabel="Total energy (J)",
        ylabel="AP score",
        filename="07_accuracy_vs_total_energy.png",
    ))
    outputs.append(scatter_pareto(
        df,
        x_col="total_time_s",
        y_col="ap_score",
        title="Accuracy vs Total Time",
        xlabel="Total time (s)",
        ylabel="AP score",
        filename="08_accuracy_vs_total_time.png",
    ))

    # Workload sensitivity: show how total runtime/energy changes when the user
    # plans to make many predictions after training.
    outputs.append(workload_sensitivity_plot(
        df,
        metric="time",
        title="Workload Sensitivity: Total Time vs Number of Inferences",
        ylabel="Total time (s)",
        filename="09_workload_time_sensitivity.png",
    ))
    outputs.append(workload_sensitivity_plot(
        df,
        metric="energy",
        title="Workload Sensitivity: Total Energy vs Number of Inferences",
        ylabel="Total energy (J)",
        filename="10_workload_energy_sensitivity.png",
    ))

    # A small summary file is handy when writing the report.
    summary = df[["model", "ap_score", "train_time_s", "inference_time_s", "train_energy_j", "inference_energy_j", "total_time_s", "total_energy_j"]].copy()
    summary.to_csv(FIG_DIR / "model_summary_for_report.csv", index=False)

    print("\nSaved figures:")
    for p in outputs:
        print(f"  {p}")
    print(f"\nSummary CSV: {FIG_DIR / 'model_summary_for_report.csv'}")


if __name__ == "__main__":
    main()
