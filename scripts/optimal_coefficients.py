"""Part 3: closed-form coefficients of the untransformed, standardized and
log-transformed multivariate models (scikit-learn LinearRegression).

Standardized weights divided by each feature's training std recover the
untransformed weights exactly, which is the point of the table.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from utils.data import FEATURES, load_data, to_arrays, train_test_split
from utils.report_style import DISPLAY_NAMES, save_table
from utils.transforms import Standardizer

from scripts.train import build_features


def main():
    train_df, test_df = train_test_split(load_data())
    X_train_raw, y_train = to_arrays(train_df)
    X_test_raw, _ = to_arrays(test_df)

    models, test_preds = {}, {}
    for label, transform in [("Untransformed", "none"),
                             ("Standardized", "standardized"),
                             ("Log-transformed", "log")]:
        X_train, X_test = build_features(X_train_raw, X_test_raw, transform)
        models[label] = LinearRegression().fit(X_train, y_train)
        test_preds[label] = models[label].predict(X_test)

    sigma = Standardizer().fit(X_train_raw).std_
    table = pd.DataFrame({
        "Untransformed": models["Untransformed"].coef_,
        "Standardized": models["Standardized"].coef_,
        "Standardized / std": models["Standardized"].coef_ / sigma,
        "Log-transformed": models["Log-transformed"].coef_,
    }, index=[DISPLAY_NAMES[f] for f in FEATURES])
    table.loc["Intercept"] = [
        models["Untransformed"].intercept_,
        models["Standardized"].intercept_,
        np.nan,
        models["Log-transformed"].intercept_,
    ]
    table.index.name = "Feature"

    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    gap = np.abs(test_preds["Untransformed"] - test_preds["Standardized"]).max()
    print(f"\nmax |prediction difference| untransformed vs standardized (test): {gap:.2e}")

    save_table(table, "optimal_coefficients.csv", float_format="%.6g")


if __name__ == "__main__":
    main()
