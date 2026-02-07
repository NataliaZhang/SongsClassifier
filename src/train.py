from __future__ import annotations
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

def cv_auc(model, X, y, n_splits: int = 5, seed: int = 0) -> tuple[float, list[float]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores: list[float] = []

    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        m = clone(model)
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_va)[:, 1]
        scores.append(float(roc_auc_score(y_va, p)))

    return float(np.mean(scores)), scores

def fit_full(model, X, y):
    model.fit(X, y)
    return model