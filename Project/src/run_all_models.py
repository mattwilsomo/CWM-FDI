from __future__ import annotations

from typing import Iterable

import pandas as pd

from profiling import Model


MODELS: tuple[str, ...] = (
    "lr",
    "dt",
    "rf",
    "svm",
    "mlp",
)

FILEPATH = "../data/creditcard.csv"
OUTPUT_CSV = "model_profiles.csv"


def profile_one(modeltype: str, filepath: str) -> dict:
    m = Model(modeltype, filepath)
    m.train()
    m.evaluate()

    return {
        "model": modeltype,
        "ap_score": m.ap_score,
        "train_time_s": m.train_time,
        "inference_time_s": m.inference_time,
        "train_energy_j": m.train_energy,
        "inference_energy_j": m.inference_energy,
        "carbon_footprint": m.carbon_footprint,
    }


def profile_all(modeltypes: Iterable[str], filepath: str) -> pd.DataFrame:
    rows = []
    for modeltype in modeltypes:
        print(f"Profiling {modeltype}...")
        rows.append(profile_one(modeltype, filepath))
    return pd.DataFrame(rows)


def main() -> None:
    df = profile_all(MODELS, FILEPATH)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved profile table to {OUTPUT_CSV}")
    print("\nPreview:")
    print(df)


if __name__ == "__main__":
    main()
