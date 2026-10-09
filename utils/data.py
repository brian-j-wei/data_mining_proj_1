"""Dataset loading and the fixed 900/130 split."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "concrete+compressive+strength"
    / "Concrete_Data.xls"
)

FEATURES = [
    "Cement",
    "BlastFurnaceSlag",
    "FlyAsh",
    "Water",
    "Superplasticizer",
    "CoarseAggregate",
    "FineAggregate",
    "Age",
]
TARGET = "CompressiveStrength"


def load_data(path=DATA_PATH):
    """Load the UCI xls and rename columns to short names."""
    df = pd.read_excel(path)
    df.columns = FEATURES + [TARGET]
    return df


def train_test_split(df):
    """Rows 501-630 (1-indexed) are the test set; the other 900 rows train."""
    test = df.iloc[500:630]
    train = pd.concat([df.iloc[:500], df.iloc[630:]])
    return train, test


def to_arrays(df, features=None):
    features = FEATURES if features is None else features
    return df[features].to_numpy(dtype=float), df[TARGET].to_numpy(dtype=float)


def add_intercept(X):
    """Prepend a column of ones. Accepts (n,) or (n, p)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.hstack([np.ones((X.shape[0], 1)), X])
