"""Evaluation metrics."""

import numpy as np


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def r2(y_true, y_pred):
    """1 - MSE / Var(observed)"""
    y_true = np.asarray(y_true)
    return 1.0 - mse(y_true, y_pred) / float(np.var(y_true))
