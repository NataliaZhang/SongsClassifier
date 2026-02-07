from __future__ import annotations
import os
import pandas as pd
from .config import Paths, TARGET_COL

def _read_csv(data_dir: str, filename: str) -> pd.DataFrame:
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def load_train_test(paths: Paths) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train_df = _read_csv(paths.data_dir, paths.train_csv)
    test_df = _read_csv(paths.data_dir, paths.test_csv)

    if TARGET_COL not in train_df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in train.csv")

    y = train_df[TARGET_COL].astype(int)
    X = train_df.drop(columns=[TARGET_COL])

    return X, y, test_df

def load_sample_submission(paths: Paths) -> pd.DataFrame:
    return _read_csv(paths.data_dir, paths.sample_sub_csv)