"""Q6 (gradient descent half): eight univariate models y = m*x + b, one per
untransformed predictor, trained with our own gradient descent.

Each model uses the largest learning rate on a 1-2-5 grid below its stability
limit 2 / lambda_max. The assignment budget is 5,000 iterations; an extended
50,000-iteration run shows the slower models eventually crossing R^2 = 0. For
models that need more than that, the crossing point is computed exactly from
the closed-form gradient descent recursion on the quadratic MSE surface.

Outputs
  assets/tables/univariate_gradient_descent.csv
  assets/figures/univariate_gradient_descent.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from utils.data import FEATURES, add_intercept, load_data, to_arrays, train_test_split
from utils.grad_descent import gradient_descent
from utils.metrics import mse, r2
from utils.report_style import (
    AXIS, BLUE, DISPLAY_NAMES, INK_2, MUTED, save_figure, save_table, use_report_style,
)

BUDGET = 5_000
EXTENDED = 50_000


def grid_learning_rate(limit):
    """Largest value of the form {1, 2, 5} x 10^k strictly below limit."""
    k = int(np.floor(np.log10(limit)))
    candidates = [m * 10.0 ** e for e in (k - 1, k) for m in (1, 2, 5)]
    return max(c for c in candidates if c < limit)


def updates_to_positive_r2(Z, y, lr):
    """Exact number of updates until MSE < Var(y), from the GD recursion.

    On a quadratic, theta_k - theta* = (I - lr H)^k (theta_0 - theta*), so the
    loss after k updates is f* + 1/2 sum_i lam_i (1 - lr lam_i)^(2k) d_i^2.
    Each term shrinks monotonically when lr < 2 / lam_max, so bisection works.
    """
    n = len(y)
    lam, V = np.linalg.eigh(2.0 / n * Z.T @ Z)
    theta_star = np.linalg.lstsq(Z, y, rcond=None)[0]
    f_star = np.mean((Z @ theta_star - y) ** 2)
    d = V.T @ (np.zeros(Z.shape[1]) - theta_star)

    def loss(k):
        return f_star + 0.5 * np.sum(lam * (1 - lr * lam) ** (2 * k) * d**2)

    target = np.var(y)
    hi = 1
    while loss(hi) >= target:
        hi *= 2
    lo = 0
    while lo < hi:
        mid = (lo + hi) // 2
        if loss(mid) < target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def fit_feature(i, X_train, y_train, X_test, y_test):
    x_train, x_test = X_train[:, i], X_test[:, i]
    Z_train, Z_test = add_intercept(x_train), add_intercept(x_test)
    lam = np.linalg.eigvalsh(2.0 / len(y_train) * Z_train.T @ Z_train)
    limit = 2.0 / lam.max()
    lr = grid_learning_rate(limit)

    # history[k] is the training MSE after k updates
    _, history = gradient_descent(Z_train, y_train, lr=lr, max_iters=EXTENDED + 1, tol=0.0)
    theta, _ = gradient_descent(Z_train, y_train, lr=lr, max_iters=BUDGET, tol=0.0)
    r2_curve = 1.0 - history / np.var(y_train)

    exact = updates_to_positive_r2(Z_train, y_train, lr)
    observed = np.flatnonzero(r2_curve > 0)
    if observed.size:
        assert observed[0] == exact, (FEATURES[i], observed[0], exact)

    opt = LinearRegression().fit(x_train.reshape(-1, 1), y_train)
    row = {
        "Feature": DISPLAY_NAMES[FEATURES[i]],
        "Mean": x_train.mean(),
        "Std": x_train.std(),
        "Condition number": lam.max() / lam.min(),
        "Stability limit 2/L": limit,
        "Learning rate": lr,
        f"Train MSE after {BUDGET}": mse(y_train, Z_train @ theta),
        f"Train R2 after {BUDGET}": r2(y_train, Z_train @ theta),
        f"Test R2 after {BUDGET}": r2(y_test, Z_test @ theta),
        f"Train R2 after {EXTENDED}": r2_curve[EXTENDED],
        "Optimal train MSE": mse(y_train, opt.predict(x_train.reshape(-1, 1))),
        "Optimal train R2": r2(y_train, opt.predict(x_train.reshape(-1, 1))),
        "Updates to R2 > 0": exact,
        "Count source": "observed" if observed.size else "GD recursion",
        f"GD slope after {BUDGET}": theta[1],
        "Optimal slope": opt.coef_[0],
        f"GD intercept after {BUDGET}": theta[0],
        "Optimal intercept": opt.intercept_,
    }
    return row, r2_curve


def plot_small_multiples(rows, curves):
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.4), sharex=True, sharey=True)
    iters = np.arange(len(curves[0]))

    for ax, row, curve in zip(axes.flat, rows, curves):
        ax.axhline(0, color=AXIS, linewidth=0.9, zorder=1)
        ax.axvline(BUDGET, color=AXIS, linewidth=0.9, zorder=1)
        ax.axhline(row["Optimal train R2"], color=MUTED, linewidth=0.8, zorder=1)
        ax.plot(iters[1:], curve[1:], color=BLUE)

        k = row["Updates to R2 > 0"]
        if k <= EXTENDED:
            ax.plot(k, curve[k], "o", color=BLUE, markersize=4.5,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=3)
            note = f"$R^2>0$ after\n{k:,} iterations"
        else:
            note = f"$R^2>0$ after\n{k / 1e6:.1f} million (off-chart)"
        # late crossers occupy the lower right, so their note goes left
        x_note, ha = (0.04, "left") if 1_000 < k <= EXTENDED else (0.96, "right")
        ax.text(x_note, 0.05, note, transform=ax.transAxes, fontsize=7,
                color=INK_2, ha=ha)

        mantissa, exponent = f"{row['Condition number']:.1e}".split("e")
        ax.set_title(row["Feature"], loc="left", fontsize=8.5, pad=13)
        ax.text(0, 1.03, rf"$\kappa = {mantissa}\times10^{{{int(exponent)}}}$",
                transform=ax.transAxes, fontsize=7, color=INK_2, va="bottom")

    ax0 = axes[0, 0]
    ax0.set_xscale("log")
    ax0.set_xlim(1, EXTENDED)
    ax0.set_ylim(-2.0, 0.6)
    for ax in axes[1]:
        ax.set_xlabel("Iteration")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Training $R^2$")

    fig.text(0.5, -0.02,
             "Light vertical line: 5,000-iteration budget.  Gray horizontal line: "
             "optimal (closed-form) R².  Panels ordered from easiest to hardest.",
             ha="center", fontsize=7, color=INK_2)
    fig.tight_layout()
    return save_figure(fig, "univariate_gradient_descent.png")


def main():
    use_report_style()
    train_df, test_df = train_test_split(load_data())
    X_train, y_train = to_arrays(train_df)
    X_test, y_test = to_arrays(test_df)

    results = [fit_feature(i, X_train, y_train, X_test, y_test)
               for i in range(len(FEATURES))]
    results.sort(key=lambda r: r[0]["Updates to R2 > 0"])
    rows = [r for r, _ in results]
    curves = [c for _, c in results]

    table = pd.DataFrame(rows).set_index("Feature")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.T.to_string(float_format=lambda v: f"{v:.5g}"))
    positive = table[f"Train R2 after {BUDGET}"] > 0
    print(f"\npositive R2 within {BUDGET} iterations: {list(table.index[positive])}")

    save_table(table, "univariate_gradient_descent.csv", float_format="%.6g")
    plot_small_multiples(rows, curves)


if __name__ == "__main__":
    main()
