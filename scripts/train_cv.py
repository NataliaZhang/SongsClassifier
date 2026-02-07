from src.config import Paths
from src.data import load_train_test
from src.model import build_model, ModelSpec
from src.train import cv_auc

def main():
    paths = Paths()
    X, y, _ = load_train_test(paths)

    for name in ["logreg", "hgb", "xgb"]:
        model = build_model(ModelSpec(name=name))
        mean_auc, folds = cv_auc(model, X, y, n_splits=5, seed=0)
        print(f"{name}: mean AUC={mean_auc:.5f} folds={['%.5f'%s for s in folds]}")

if __name__ == "__main__":
    main()
