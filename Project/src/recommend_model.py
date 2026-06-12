#!/usr/bin/env python3
"""Interactive recommender for the ML sustainability mini-project.

Reads a CSV of profiled models and recommends the best feasible model under
user-specified constraints.

Expected CSV columns:
- model
- ap_score
- train_time_s
- inference_time_s
- train_energy_j
- inference_energy_j
- carbon_footprint (optional / can be empty)
- profiled_inference_count (optional; defaults to 56962 if missing)

The inference metrics in the CSV are assumed to be measured on a fixed number
of predictions. The recommender scales those metrics to the user-specified
workload.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

CSV_PATH = Path(__file__).resolve().with_name("model_profiles.csv")
JOULES_PER_KWH = 3_600_000.0
DEFAULT_PROFILED_INFERENCE_COUNT = 56962.0

Mode = Literal["accuracy", "speed", "energy", "carbon", "balanced"]


def _prompt_float(prompt: str, *, allow_blank: bool = False) -> float | None:
    while True:
        raw = input(prompt).strip()
        if raw == "" and allow_blank:
            return None
        try:
            return float(raw)
        except ValueError:
            print("Please enter a number.")


def _prompt_int(prompt: str, *, allow_blank: bool = False) -> int | None:
    while True:
        raw = input(prompt).strip()
        if raw == "" and allow_blank:
            return None
        try:
            value = int(raw)
            if value < 0:
                print("Please enter a non-negative integer.")
                continue
            return value
        except ValueError:
            print("Please enter a whole number.")


def _prompt_mode(prompt: str) -> Mode:
    allowed = {"accuracy", "speed", "energy", "carbon", "balanced"}
    while True:
        raw = input(prompt).strip().lower()
        if raw in allowed:
            return raw  # type: ignore[return-value]
        print(f"Choose one of: {', '.join(sorted(allowed))}")


def load_profiles(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")

    df = pd.read_csv(csv_path)

    required = {
        "model",
        "ap_score",
        "train_time_s",
        "inference_time_s",
        "train_energy_j",
        "inference_energy_j",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    out = df.copy()
    if "profiled_inference_count" not in out.columns:
        out["profiled_inference_count"] = DEFAULT_PROFILED_INFERENCE_COUNT

    return out


def add_derived_metrics(
    df: pd.DataFrame,
    workload: int,
    carbon_intensity_g_per_kwh: float | None,
) -> pd.DataFrame:
    out = df.copy()

    out["inference_scale"] = workload / out["profiled_inference_count"].astype(float)
    out["scaled_inference_time_s"] = out["inference_time_s"] * out["inference_scale"]
    out["total_time_s"] = out["train_time_s"] + out["scaled_inference_time_s"]

    out["scaled_inference_energy_j"] = out["inference_energy_j"] * out["inference_scale"]
    out["total_energy_j"] = out["train_energy_j"] + out["scaled_inference_energy_j"]

    out["train_energy_kwh"] = out["train_energy_j"] / JOULES_PER_KWH
    out["inference_energy_kwh"] = out["scaled_inference_energy_j"] / JOULES_PER_KWH
    out["total_energy_kwh"] = out["total_energy_j"] / JOULES_PER_KWH

    if carbon_intensity_g_per_kwh is not None:
        out["carbon_footprint_g"] = out["total_energy_kwh"] * carbon_intensity_g_per_kwh
    else:
        out["carbon_footprint_g"] = pd.NA

    return out


def filter_feasible(
    df: pd.DataFrame,
    min_ap: float | None,
    max_train_time: float | None,
    max_inference_time: float | None,
    max_train_energy: float | None,
    max_inference_energy: float | None,
    max_total_energy: float | None,
    max_carbon_g: float | None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    if min_ap is not None:
        mask &= df["ap_score"] >= min_ap
    if max_train_time is not None:
        mask &= df["train_time_s"] <= max_train_time
    if max_inference_time is not None:
        mask &= df["scaled_inference_time_s"] <= max_inference_time
    if max_train_energy is not None:
        mask &= df["train_energy_j"] <= max_train_energy
    if max_inference_energy is not None:
        mask &= df["scaled_inference_energy_j"] <= max_inference_energy
    if max_total_energy is not None:
        mask &= df["total_energy_j"] <= max_total_energy
    if max_carbon_g is not None:
        mask &= df["carbon_footprint_g"].notna() & (df["carbon_footprint_g"] <= max_carbon_g)

    return df.loc[mask].copy()


def choose_recommendation(df: pd.DataFrame, mode: Mode) -> pd.Series:
    # Sort by the primary objective, then use secondary criteria to break ties.
    if mode == "accuracy":
        sort_cols = ["ap_score", "total_energy_j", "total_time_s"]
        ascending = [False, True, True]
    elif mode == "speed":
        sort_cols = ["total_time_s", "ap_score", "total_energy_j"]
        ascending = [True, False, True]
    elif mode == "energy":
        sort_cols = ["total_energy_j", "ap_score", "total_time_s"]
        ascending = [True, False, True]
    elif mode == "carbon":
        if df["carbon_footprint_g"].notna().any():
            sort_cols = ["carbon_footprint_g", "ap_score", "total_time_s"]
            ascending = [True, False, True]
        else:
            sort_cols = ["total_energy_j", "ap_score", "total_time_s"]
            ascending = [True, False, True]
    else:  # balanced
        # Keep the model with the best AP among the more efficient half of the set.
        median_energy = df["total_energy_j"].median()
        subset = df[df["total_energy_j"] <= median_energy].copy()
        if subset.empty:
            subset = df.copy()
        df = subset
        sort_cols = ["ap_score", "total_time_s", "total_energy_j"]
        ascending = [False, True, True]

    return df.sort_values(by=sort_cols, ascending=ascending, kind="mergesort").iloc[0]


def _fmt_number(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def main() -> int:
    df = load_profiles()

    print("Loaded profiles:\n")
    print(
        df[
            [
                "model",
                "ap_score",
                "train_time_s",
                "inference_time_s",
                "train_energy_j",
                "inference_energy_j",
            ]
        ].to_string(index=False)
    )
    print()

    print("Enter constraints. Press Enter to skip a constraint.")
    workload = _prompt_int("How many predictions will you run? ", allow_blank=True)
    if workload is None:
        workload = 1
    if workload == 0:
        workload = 1

    min_ap = _prompt_float("Minimum AP score: ", allow_blank=True)
    max_train_time = _prompt_float("Maximum training time (s): ", allow_blank=True)
    max_inference_time = _prompt_float("Maximum inference time for the workload (s): ", allow_blank=True)
    max_train_energy = _prompt_float("Maximum training energy (J): ", allow_blank=True)
    max_inference_energy = _prompt_float("Maximum inference energy for the workload (J): ", allow_blank=True)
    max_total_energy = _prompt_float("Maximum total energy (J): ", allow_blank=True)

    carbon_intensity = _prompt_float(
        "Carbon intensity in gCO2e/kWh (optional, used to compute carbon): ",
        allow_blank=True,
    )
    max_carbon_g = _prompt_float("Maximum carbon footprint (gCO2e): ", allow_blank=True)

    mode = _prompt_mode("Optimisation mode [accuracy/speed/energy/carbon/balanced]: ")

    df = add_derived_metrics(df, workload, carbon_intensity)
    feasible = filter_feasible(
        df,
        min_ap=min_ap,
        max_train_time=max_train_time,
        max_inference_time=max_inference_time,
        max_train_energy=max_train_energy,
        max_inference_energy=max_inference_energy,
        max_total_energy=max_total_energy,
        max_carbon_g=max_carbon_g,
    )

    print()
    if feasible.empty:
        print("No model satisfies all constraints.")
        print("Try relaxing one or more limits.")
        return 1

    recommended = choose_recommendation(feasible, mode)

    print("Feasible models:\n")
    cols = [
        "model",
        "ap_score",
        "train_time_s",
        "scaled_inference_time_s",
        "total_time_s",
        "train_energy_j",
        "scaled_inference_energy_j",
        "total_energy_j",
        "carbon_footprint_g",
    ]
    display = feasible[cols].sort_values(by=["ap_score"], ascending=False).copy()
    if carbon_intensity is None:
        display["carbon_footprint_g"] = display["carbon_footprint_g"].apply(lambda x: "N/A")
    print(display.to_string(index=False))

    print("\nRecommended model:")
    print(f"  model: {recommended['model']}")
    print(f"  AP score: {recommended['ap_score']:.6f}")
    print(f"  training time (s): {recommended['train_time_s']:.6f}")
    print(f"  inference time for workload (s): {recommended['scaled_inference_time_s']:.6f}")
    print(f"  total time (s): {recommended['total_time_s']:.6f}")
    print(f"  training energy (J): {recommended['train_energy_j']:.2f}")
    print(f"  inference energy for workload (J): {recommended['scaled_inference_energy_j']:.2f}")
    print(f"  total energy (J): {recommended['total_energy_j']:.2f}")

    if carbon_intensity is not None:
        print(f"  carbon footprint (gCO2e): {_fmt_number(recommended['carbon_footprint_g'], 4)}")
    else:
        print("  carbon footprint (gCO2e): N/A (no carbon intensity provided)")

    print(f"  workload used for decision: {workload} predictions")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
