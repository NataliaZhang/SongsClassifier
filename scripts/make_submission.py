import os
import pandas as pd

from src.config import Paths
from src.data import load_train_test, load_sample_submission
from src.model import build_model, ModelSpec
from src.train import fit_full
from src.predict import predict_proba_1

def main():
    paths = Paths()
    X, y, X_test = load_train_test(paths)

    # Compute scale_pos_weight for XGBoost, which helps with class imbalance.
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    scale_pos_weight = (n_neg / max(n_pos, 1))

    # Pick your current best
    model = build_model(ModelSpec(name="xgb"), scale_pos_weight)
    model = fit_full(model, X, y)

    p_test = predict_proba_1(model, X_test)

    sub = load_sample_submission(paths)
    # Assumes sample_submission has exactly the right columns.
    target_col = sub.columns[-1]
    sub[target_col] = p_test

    os.makedirs(paths.output_dir, exist_ok=True)
    out_path = os.path.join(paths.output_dir, paths.submission_csv)
    sub.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()