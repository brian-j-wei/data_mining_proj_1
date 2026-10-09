"""Feature transformations"""

import numpy as np


class Standardizer:
    """Zero mean, unit variance per feature."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def log1p_transform(X):
    """log(x + 1), elementwise."""
    return np.log1p(np.asarray(X, dtype=float))
