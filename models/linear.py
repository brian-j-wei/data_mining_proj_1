"""Linear models fit with our own gradient descent."""

import numpy as np

from utils.data import add_intercept
from utils.grad_descent import gradient_descent


class MultivariateLinearModel:
    """y = x @ w + b, fit by batch gradient descent on MSE."""

    def __init__(self, lr=1e-3, max_iters=1000, tol=1e-8):
        self.lr = lr
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X, y):
        self.theta_, self.loss_history_ = gradient_descent(
            add_intercept(X), np.asarray(y, dtype=float),
            lr=self.lr, max_iters=self.max_iters, tol=self.tol,
        )
        return self

    def predict(self, X):
        return add_intercept(X) @ self.theta_

    @property
    def intercept_(self):
        return self.theta_[0]

    @property
    def coef_(self):
        return self.theta_[1:]


class UnivariateLinearModel(MultivariateLinearModel):
    """y = m*x + b. Same solver, single predictor."""

    def fit(self, x, y):
        return super().fit(np.asarray(x, dtype=float).reshape(-1, 1), y)

    def predict(self, x):
        return super().predict(np.asarray(x, dtype=float).reshape(-1, 1))

    @property
    def slope_(self):
        return self.theta_[1]
