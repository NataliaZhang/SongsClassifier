import os
import pandas as pd

from src.config import Paths, TARGET_COL
from src.data import load_train_test, load_sample_submission
from src.model import build_model, ModelSpec
from src.train import fit_full
from src.predict import predict_proba_1

def main():
    paths = Paths()
    X, y, X_test = load_train_test(paths)

    # Pick your current best
    model = build_model(ModelSpec(name="xgb"))
    model = fit_full(model, X, y)

    p_test = predict_proba_1(model, X_test)

    test_path = os.path.join(paths.data_dir, paths.test_csv)
    test_df = pd.read_csv(test_path)
    sub = pd.DataFrame({"ID": test_df["ID"].values, TARGET_COL: p_test})


    os.makedirs(paths.output_dir, exist_ok=True)
    out_path = os.path.join(paths.output_dir, paths.submission_csv)
    sub.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()