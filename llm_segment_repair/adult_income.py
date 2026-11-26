from __future__ import annotations

from pathlib import Path

import pandas as pd

INPUT_PATH = Path("dataset/adult/adult.data")
OUTPUT_PATH = Path("dataset/adult/adult_clean.csv")
MIN_SEGMENT_SIZE = 25

SEGMENT_COLUMNS = ["age_group", "sex", "race", "workclass"]

CATEGORICAL_COLUMNS = [
    "age_group",
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
    "income",
]

NUMERICAL_COLUMNS = [
    "capital_gain",
    "capital_loss",
]


def _age_group(value: float | int | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    age = int(value)
    if age < 25:
        return "18-24"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    if age < 65:
        return "55-64"
    return "65+"


def prepare_adult_income(path: Path) -> pd.DataFrame:
    """Read the headerless UCI Adult training file in its original column order."""

    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income",
    ]

    df = pd.read_csv(
        path,
        header=None,
        names=columns,
        na_values="?",
        skipinitialspace=True,
    )

    df["age_group"] = df["age"].apply(_age_group)

    # Drop rows missing segment columns.
    df = df.dropna(subset=SEGMENT_COLUMNS).copy()

    # Keep only segments with enough rows.
    counts = df.groupby(SEGMENT_COLUMNS, dropna=False).size()
    keep_keys = counts[counts >= MIN_SEGMENT_SIZE].index
    if len(keep_keys) > 0:
        keep_mask = pd.MultiIndex.from_frame(df[SEGMENT_COLUMNS]).isin(keep_keys)
        df = df[keep_mask].copy()
    else:
        df = df.iloc[0:0].copy()

    # Final column set.
    final_cols = [*CATEGORICAL_COLUMNS, *NUMERICAL_COLUMNS]
    return df[final_cols].copy()


def main() -> None:
    df = prepare_adult_income(INPUT_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
