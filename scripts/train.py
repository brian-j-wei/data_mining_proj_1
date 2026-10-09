
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from models.linear import MultivariateLinearModel, UnivariateLinearModel
from utils.data import FEATURES, load_data, to_arrays, train_test_split
from utils.metrics import mse, r2
from utils.transforms import Standardizer, log1p_transform


def build_features(X_train, X_test, transform):
    """Fit the transform on train, then apply to train and test."""
    if transform == "none":
        return X_train, X_test
    if transform == "standardized":
        scaler = Standardizer().fit(X_train)
        return scaler.transform(X_train), scaler.transform(X_test)
    if transform == "log":
        return log1p_transform(X_train), log1p_transform(X_test)
    raise ValueError(f"unknown transform: {transform}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--transform", default="none",
                   choices=["none", "standardized", "log"])
    p.add_argument("--feature", default=None,
                   help="single feature name for a univariate fit")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--tol", type=float, default=1e-8)
    return p.parse_args()


def main():
    args = parse_args()

    train_df, test_df = train_test_split(load_data())
    features = [args.feature] if args.feature else FEATURES
    X_train, y_train = to_arrays(train_df, features)
    X_test, y_test = to_arrays(test_df, features)
    X_train, X_test = build_features(X_train, X_test, args.transform)

    cls = UnivariateLinearModel if args.feature else MultivariateLinearModel
    model = cls(lr=args.lr, max_iters=args.iters, tol=args.tol)
    model.fit(X_train, y_train)

    for name, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        pred = model.predict(X)
        print(f"{name:5s}  MSE={mse(y, pred):10.4f}  R2={r2(y, pred):8.4f}")

    print(f"iters run: {len(model.loss_history_)}")
    print(f"theta: {np.array2string(model.theta_, precision=4)}")


if __name__ == "__main__":
    main()
