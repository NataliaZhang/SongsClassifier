from __future__ import annotations
import numpy as np

def predict_proba_1(model, X) -> np.ndarray:
    p = model.predict_proba(X)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("Model predict_proba must return (n, 2+) probabilities.")
    return p[:, 1]