from __future__ import annotations

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


@dataclass(frozen=True)
class ModelSpec:
    name: str  # "logreg" or "hgb"


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

    raise ValueError(f"Unknown model spec: {spec.name}")