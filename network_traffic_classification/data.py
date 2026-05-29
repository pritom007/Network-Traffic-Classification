"""Data loading helpers for nDPI flow CSV exports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATASET_COLUMNS, FEATURE_COLUMNS


def load_labeled_flows(csv_path: str | Path) -> pd.DataFrame:
    """Load a labeled flow CSV and normalize its expected schema."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if not set(FEATURE_COLUMNS + ["class"]).issubset(df.columns):
        df = pd.read_csv(path, names=DATASET_COLUMNS)
        if str(df.iloc[0]["class"]).lower() == "class":
            df = df.iloc[1:].reset_index(drop=True)

    missing = sorted(set(FEATURE_COLUMNS + ["class"]) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    return df


def split_features_and_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return numeric model features and the raw class labels."""
    features = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    invalid_rows = features.isna().any(axis=1)
    if invalid_rows.any():
        count = int(invalid_rows.sum())
        raise ValueError(f"{count} rows contain non-numeric feature values.")

    labels = df["class"].astype(str)
    return features, labels


def encode_labels(labels: pd.Series) -> tuple[pd.Series, list[str]]:
    """Encode string labels as integers while preserving class order."""
    class_names = list(dict.fromkeys(labels.tolist()))
    class_to_index = {name: index for index, name in enumerate(class_names)}
    encoded = labels.map(class_to_index)
    return encoded, class_names
