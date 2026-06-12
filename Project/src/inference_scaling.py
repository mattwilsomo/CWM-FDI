#!/usr/bin/env python3
"""Measure how inference time scales with the number of predictions.

This script is intended for the ML sustainability mini-project. It trains each
chosen model once, then measures prediction time on increasing batch sizes built
from the held-out test set.

Why this matters:
- A model may be cheap to train but expensive to run many times.
- For large workloads, inference cost can dominate total cost.
- This lets you test the linear-scaling assumption used by the recommender.

Outputs:
- CSV file with all raw measurements
- A combined plot of batch size vs median inference time
- One plot per model

Run from the same folder as profiling.py / modelling.py / data_loading.py.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from profiling import Model


DEFAULT_MODELS = ["lr", "dt", "rf", "svm", "mlp"]
DEFAULT_BATCH_SIZES = [1, 10, 100, 1_000, 5_000, 10_000, 20_000, 56_962]
DEFAULT_REPEATS = 5
DEFAULT_WARMUP_ROWS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure inference scaling for the ML sustainability project."
    )
    parser.add_argument(
        "--data",
        default="../data/creditcard.csv",
        help="Path to the credit card fraud CSV.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Model types to test: lr dt rf svm mlp",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="*",
        type=int,
        default=DEFAULT_BATCH_SIZES,
        help="Batch sizes to evaluate.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Number of timings per batch size.",
    )
    parser.add_argument(
        "--warmup-rows",
        type=int,
        default=DEFAULT_WARMUP_ROWS,
        help="Small warm-up batch used to trigger any one-off overhead before measuring.",
    )
    parser.add_argument(
        "--outdir",
        default="scaling_results",
        help="Directory where CSVs and plots will be saved.",
    )
    return parser.parse_args()


def normalise_model_name(modeltype: str) -> str:
    key = modeltype.lower().strip()
    aliases = {
        "lr": "lr",
        "logistic regression": "lr",
        "dt": "dt",
        "decision tree": "dt",
        "rf": "rf",
        "random forest": "rf",
        "svm": "svm",
        "linear svm": "svm",
        "mlp": "mlp",
    }
    if key not in aliases:
        raise ValueError(f"Unknown model type: {modeltype}")
    return aliases[key]


def build_batch(X_test: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Create a batch of exactly n_rows from the held-out test set.

    We avoid modifying the model or the original test set. For small batches,
    we sample from the test set. For larger batches, we repeat the test set
    until we have enough rows, then truncate.
    """
    if n_rows <= len(X_test):
        # Use a random sample so the batch is not biased by row order.
        return X_test.sample(n=n_rows, random_state=42).copy()

    # Repeat the test set enough times to reach the target size.
    copies = math.ceil(n_rows / len(X_test))
    repeated = pd.concat([X_test] * copies, ignore_index=True)
    return repeated.iloc[:n_rows].copy()


def score_batch(model, X_batch: pd.DataFrame):
    """Return model scores for AP-style inference.

    Logistic Regression / Tree / RF / MLP expose predict_proba.
    LinearSVC exposes decision_function.
    """
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_batch)
        # For binary classification, keep the positive class score.
        if getattr(scores, "ndim", 1) == 2:
            return scores[:, 1]
        return scores

    if hasattr(model, "decision_function"):
        return model.decision_function(X_batch)

    # Fallback for any other model type.
    return model.predict(X_batch)


def measure_one_batch(model, X_batch: pd.DataFrame, repeats: int) -> list[float]:
    """Time one batch several times and return the timings.

    We use the median later because runtime measurements are noisy.
    """
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = score_batch(model, X_batch)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
    return timings


def profile_model(modeltype: str, data_path: str, batch_sizes: Iterable[int], repeats: int, warmup_rows: int) -> tuple[pd.DataFrame, Model]:
    """Train one model and collect inference-scaling measurements."""
    m = Model(modeltype, data_path)
    m.train()

    # A tiny warm-up call helps absorb one-off overheads such as imports,
    # lazy internal setup, or cache effects before timed measurements start.
    warmup_batch = build_batch(m.X_test, min(warmup_rows, len(m.X_test)))
    _ = score_batch(m.model, warmup_batch)

    rows = []
    for batch_size in batch_sizes:
        # Ensure batch size is at least 1 and not absurd.
        batch_size = max(1, int(batch_size))
        batch = build_batch(m.X_test, batch_size)
        timings = measure_one_batch(m.model, batch, repeats)

        rows.append(
            {
                "model": normalise_model_name(modeltype),
                "batch_size": batch_size,
                "repeat_count": repeats,
                "min_time_s": min(timings),
                "median_time_s": statistics.median(timings),
                "mean_time_s": statistics.mean(timings),
                "stdev_time_s": statistics.pstdev(timings) if len(timings) > 1 else 0.0,
            }
        )

    return pd.DataFrame(rows), m


def plot_combined(df: pd.DataFrame, outpath: Path) -> None:
    """Plot all models on one figure for easy comparison."""
    plt.figure(figsize=(10, 6))

    for model_name in df["model"].dropna().unique():
        sub = df[df["model"] == model_name].sort_values("batch_size")
        plt.plot(
            sub["batch_size"],
            sub["median_time_s"],
            marker="o",
            linewidth=2,
            label=model_name,
        )

    plt.xscale("log")
    plt.xlabel("Number of predictions in batch (log scale)")
    plt.ylabel("Median inference time (s)")
    plt.title("Inference scaling with batch size")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_per_model(df: pd.DataFrame, outdir: Path) -> None:
    """Create one plot per model so each curve is easy to read."""
    for model_name in df["model"].dropna().unique():
        sub = df[df["model"] == model_name].sort_values("batch_size")

        plt.figure(figsize=(8, 5))
        plt.plot(
            sub["batch_size"],
            sub["median_time_s"],
            marker="o",
            linewidth=2,
        )
        plt.xscale("log")
        plt.xlabel("Number of predictions in batch (log scale)")
        plt.ylabel("Median inference time (s)")
        plt.title(f"Inference scaling for {model_name}")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(outdir / f"{model_name}_scaling.png", dpi=200)
        plt.close()


def main() -> None:
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_results = []
    trained_models = {}

    for modeltype in args.models:
        print(f"Profiling {modeltype}...")
        model_df, model_obj = profile_model(
            modeltype=modeltype,
            data_path=args.data,
            batch_sizes=args.batch_sizes,
            repeats=args.repeats,
            warmup_rows=args.warmup_rows,
        )
        all_results.append(model_df)
        trained_models[normalise_model_name(modeltype)] = model_obj

    results = pd.concat(all_results, ignore_index=True)

    csv_path = outdir / "inference_scaling_results.csv"
    results.to_csv(csv_path, index=False)

    combined_plot = outdir / "inference_scaling_combined.png"
    plot_combined(results, combined_plot)
    plot_per_model(results, outdir)

    print("\nSaved results to:")
    print(csv_path)
    print(combined_plot)
    print(outdir)
    print("\nPreview:")
    print(results.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
