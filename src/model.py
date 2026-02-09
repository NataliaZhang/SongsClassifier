from __future__ import annotations

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class ModelSpec:
    name: str  # "logreg" or "hgb"


class AugmentFeatures(BaseEstimator, TransformerMixin):
    """
    Adds:
      - song_age = current_year - release_year
      - cyclical month enc: sin/cos(2π*month/12)
      - a handful of interaction features
    Works on pandas DataFrame; returns a DataFrame with extra columns appended.
    """
    def __init__(self, current_year: int = 2026):
        self.current_year = current_year

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        # ---- Season / age features ----
        year_col = "track_album_release_date_year"
        month_col = "track_album_release_date_month"

        if year_col in X.columns:
            # print('here')
            year = pd.to_numeric(X[year_col], errors="coerce")
            X["song_age"] = self.current_year - year

        if month_col in X.columns:
            # print('here 2')
            month = pd.to_numeric(X[month_col], errors="coerce")
            # map invalid months to NaN; we'll impute later
            month = month.where(month.between(1, 12))
            angle = 2.0 * np.pi * (month / 12.0)
            X["release_month_sin"] = np.sin(angle)
            X["release_month_cos"] = np.cos(angle)

        # # ---- Musically meaningful interactions ----
        # def _mul(a, b, out):
        #     if a in X.columns and b in X.columns:
        #         X[out] = pd.to_numeric(X[a], errors="coerce") * pd.to_numeric(X[b], errors="coerce")

        # _mul("energy", "loudness", "energy_x_loudness")
        # _mul("danceability", "tempo", "danceability_x_tempo")
        # _mul("acousticness", "instrumentalness", "acoustic_x_instr")
        # _mul("speechiness", "duration_ms", "speech_x_duration")
        # _mul("valence", "danceability", "valence_x_danceability")

        # ---- Fix potentially skewed distributions ----
        def _log1p(col, out_col, shift=0.0, clip_min=0.0):
            if col not in X.columns:
                return
            v = pd.to_numeric(X[col], errors="coerce")
            v = v + shift
            # log1p requires >= -1; we enforce >= 0 for stability
            v = v.clip(lower=clip_min)
            X[out_col] = np.log1p(v + 1e-9)

        _log1p("duration_ms", "log1p_duration_ms")
        _log1p("tempo", "log1p_tempo")
        _log1p("loudness", "log1p_loudness_shifted", shift=getattr(self, "_loud_shift_", 0.0))

        return X


def _preprocess(scale_numeric: bool) -> ColumnTransformer:
    num_pipe_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        num_pipe_steps.append(("scaler", StandardScaler()))

    num_pipe = Pipeline(num_pipe_steps)

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, selector(dtype_include=["number"])),
            ("cat", cat_pipe, selector(dtype_exclude=["number"])),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


def build_model(spec: ModelSpec) -> Pipeline:
    if spec.name == "logreg":
        return Pipeline(steps=[
            ("pre", _preprocess(scale_numeric=True)),
            ("clf", LogisticRegression(max_iter=5000, solver="lbfgs")),
        ])

    if spec.name == "hgb":
        # HGB doesn't need scaling; but it needs dense numeric features.
        # OneHotEncoder above outputs dense (sparse_output=False).
        return Pipeline(steps=[
            ("pre", _preprocess(scale_numeric=False)),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=6,
                max_iter=400,
                random_state=0,
            )),
        ])

    if spec.name == "xgb":
        # XGBoost handles unscaled numeric features well.
        return Pipeline(steps=[
            ("pre", _preprocess(scale_numeric=False)),
            ("clf", XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                random_state=0,
                n_jobs=-1,
            )),
        ])

    if spec.name == "xgb_feat":
        return Pipeline(steps=[
        ("feat", AugmentFeatures(current_year=2026)),
        ("pre", _preprocess(scale_numeric=False)),
        ("clf", XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=0,
            n_jobs=-1,
        )),
    ])

    raise ValueError(f"Unknown model spec: {spec.name}")
