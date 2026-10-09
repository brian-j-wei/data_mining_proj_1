import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from utils.data import FEATURES, add_intercept, load_data, to_arrays, train_test_split
from utils.metrics import mse, r2

from scripts.train import build_features


def fit_sklearn(X_train, y_train, X_test, y_test):
    """scikit learn linear regression"""
    model = LinearRegression().fit(X_train, y_train)
    scores = {}
    for name, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        pred = model.predict(X)
        scores[name] = (mse(y, pred), r2(y, pred))
    return model, scores


def fit_ols(X_train, y_train):
    """statsmodels OLS"""
    return sm.OLS(y_train, add_intercept(X_train)).fit() # add intercept as per statsmodels' api


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--transform", default="none",
                   choices=["none", "standardized", "log"])
    return p.parse_args()


def main():
    args = parse_args()

    train_df, test_df = train_test_split(load_data())
    X_train, y_train = to_arrays(train_df)
    X_test, y_test = to_arrays(test_df)
    X_train, X_test = build_features(X_train, X_test, args.transform)

    model, scores = fit_sklearn(X_train, y_train, X_test, y_test)
    for name, (m, r) in scores.items():
        print(f"{name:5s}  MSE={m:10.4f}  R2={r:8.4f}")

    print(f"\nintercept: {model.intercept_:.4f}")
    for feat, coef in zip(FEATURES, model.coef_):
        print(f"  {feat:18s} {coef:10.4f}")

    ols = fit_ols(X_train, y_train)
    print("\np-values:")
    for feat, pval in zip(["intercept"] + FEATURES, ols.pvalues):
        print(f"  {feat:18s} {pval:.4g}")


if __name__ == "__main__":
    main()
