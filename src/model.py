from __future__ import annotations
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

@dataclass(frozen=True)
class ModelSpec:
    name: str  # "logreg" or "hgb"

def build_model(spec: ModelSpec) -> Pipeline:
    if spec.name == "logreg":
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=5000,
                solver="lbfgs",
                n_jobs=None,  # lbfgs ignores n_jobs
            )),
        ])

    if spec.name == "hgb":
        # HistGradientBoosting doesn't need scaling; it handles nonlinearity well.
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=6,
                max_iter=400,
                random_state=0,
            )),
        ])

    raise ValueError(f"Unknown model spec: {spec.name}")