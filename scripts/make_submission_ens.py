import os
import numpy as np
import pandas as pd

from src.config import Paths, TARGET_COL
from src.data import load_train_test, load_sample_submission
from src.model import build_model, ModelSpec
from src.train import fit_full
from src.predict import predict_proba_1


def main():
    paths = Paths()

    X, y, X_test = load_train_test(paths)

    xgb = build_model(ModelSpec(name="xgb"))
    xgb = fit_full(xgb, X, y)
    p_xgb = predict_proba_1(xgb, X_test)

    cat = build_model(ModelSpec(name="cat"))
    cat = fit_full(cat, X, y)
    p_cat = predict_proba_1(cat, X_test)

    w_xgb, w_cat = 0.6, 0.4   # start here; also try 0.5/0.5
    p_ens = w_xgb * p_xgb + w_cat * p_cat

    test_path = os.path.join(paths.data_dir, paths.test_csv)
    test_df = pd.read_csv(test_path)

    sub = pd.DataFrame({
        "ID": test_df["ID"].values,
        TARGET_COL: p_ens
    })

    # Write submission
    os.makedirs(paths.output_dir, exist_ok=True)
    out_path = os.path.join(paths.output_dir, paths.submission_csv)
    sub.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()