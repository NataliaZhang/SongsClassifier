from __future__ import annotations

import os
import pandas as pd

from .config import Paths, TARGET_COL


def _read_csv(data_dir: str, filename: str) -> pd.DataFrame:
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _encode_target(y_raw: pd.Series) -> pd.Series:
    """
    Kaggle label is 'High'/'Low'. If already numeric, keep it.
    """
    if pd.api.types.is_numeric_dtype(y_raw):
        return y_raw.astype(int)

    y_raw = y_raw.astype(str)
    label_map = {"Low": 0, "High": 1}
    unknown = set(y_raw.unique()) - set(label_map.keys())
    if unknown:
        raise ValueError(f"Unexpected labels in {TARGET_COL}: {sorted(unknown)}")

    return y_raw.map(label_map).astype(int)


def _add_date_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Convert a date column into year/month/day numeric features.
    Drops the original column.
    """
    if col not in df.columns:
        return df

    dt = pd.to_datetime(df[col], errors="coerce")  # handles YYYY-MM-DD and similar
    df = df.copy()
    df[f"{col}_year"] = dt.dt.year
    df[f"{col}_month"] = dt.dt.month
    df[f"{col}_day"] = dt.dt.day
    df = df.drop(columns=[col])
    return df


def _choose_categorical_cols(X: pd.DataFrame, high_cardinality_frac: float = 0.5) -> tuple[list[str], list[str]]:
    """
    Split object columns into:
      - low/medium-cardinality categorical cols to keep
      - high-cardinality cols to drop (often IDs/URLs)
    Heuristic: drop if nunique > high_cardinality_frac * n_rows
    """
    obj_cols = [c for c in X.columns if X[c].dtype == "object"]
    n = max(len(X), 1)

    keep, drop = [], []
    for c in obj_cols:
        nunique = X[c].nunique(dropna=True)
        if nunique > high_cardinality_frac * n:
            drop.append(c)
        else:
            keep.append(c)
    return keep, drop


def load_train_test(paths: Paths) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train_df = _read_csv(paths.data_dir, paths.train_csv)
    test_df = _read_csv(paths.data_dir, paths.test_csv)

    if TARGET_COL not in train_df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in train.csv")

    y = _encode_target(train_df[TARGET_COL])

    X = train_df.drop(columns=[TARGET_COL]).copy()
    X_test = test_df.copy()

    # Date feature engineering
    date_col = "track_album_release_date"
    X = _add_date_features(X, date_col)
    X_test = _add_date_features(X_test, date_col)

    # Decide which categorical columns to keep vs drop
    cat_keep, cat_drop = _choose_categorical_cols(X, high_cardinality_frac=0.5)

    # Drop high-cardinality object columns (IDs/URLs)
    if cat_drop:
        X = X.drop(columns=cat_drop)
        X_test = X_test.drop(columns=[c for c in cat_drop if c in X_test.columns])

    # Ensure test has same columns in same order
    for c in X.columns:
        if c not in X_test.columns:
            X_test[c] = pd.NA
    X_test = X_test[X.columns]

    return X, y, X_test


def load_sample_submission(paths: Paths) -> pd.DataFrame:
    return _read_csv(paths.data_dir, paths.sample_sub_csv)