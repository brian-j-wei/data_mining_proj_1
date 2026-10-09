"""Batch gradient descent on MSE."""

import numpy as np


def gradient_descent(X, y, lr=1e-3, max_iters=1000, tol=1e-8, theta0=None):
    """Minimize MSE for y = X @ theta."""
    n, p = X.shape
    theta = np.zeros(p) if theta0 is None else np.asarray(theta0, dtype=float).copy()
    history = []

    for _ in range(max_iters):
        residual = X @ theta - y
        loss = np.mean(residual**2)
        history.append(loss)

        # d/dtheta of (1/n)||X@theta - y||^2
        grad = (2.0 / n) * (X.T @ residual)
        theta = theta - lr * grad

        # tolerance-based stopping
        if len(history) > 1 and abs(history[-2] - history[-1]) < tol:
            break

    return theta, np.array(history)
