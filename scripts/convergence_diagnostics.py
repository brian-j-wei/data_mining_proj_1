"""Part 3: conditioning, learning-rate stability and convergence of the three
multivariate gradient descent runs (untransformed, standardized, log).

Outputs
  assets/tables/convergence_diagnostics.csv
  assets/tables/learning_rate_sweep.csv
  assets/figures/convergence_scaling.png          untransformed vs. standardized
  assets/figures/convergence_log_transform.png    all three feature sets
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from utils.data import add_intercept, load_data, to_arrays, train_test_split
from utils.grad_descent import gradient_descent
from utils.metrics import mse, r2
from utils.report_style import (
    AQUA, AXIS, BLUE, INK_2, MUTED, ORANGE, save_figure, save_table, use_report_style,
)

from scripts.train import build_features

ITERS = 5000

# Same learning rates as the self-contained Q2/Q5 scripts
RUNS = [
    ("Untransformed", "none", 3e-7, BLUE),
    ("Standardized", "standardized", 0.2, ORANGE),
    ("Log-transformed", "log", 5e-3, AQUA),
]

# Learning rates tried in the stability sweep, as multiples of 2 / lambda_max
SWEEP_FACTORS = [0.5, 0.9, 0.99, 1.01, 1.1]


def hessian_eigenvalues(X):
    """Eigenvalues of the constant MSE Hessian H = (2/n) Z^T Z, Z = [1 | X]."""
    Z = add_intercept(X)
    return np.linalg.eigvalsh(2.0 / len(Z) * Z.T @ Z)


def updates_until_below(history, threshold):
    """history[k] is the loss after k updates; first k with loss < threshold."""
    hits = np.flatnonzero(np.asarray(history) < threshold)
    return int(hits[0]) if hits.size else None


def fmt_lr(lr):
    """3e-07 -> $3\\times10^{-7}$ for plot labels."""
    if lr >= 0.01:
        return f"{lr:g}"
    mantissa, exponent = f"{lr:.0e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def run_config(transform, lr, X_train_raw, X_test_raw, y_train, y_test):
    X_train, X_test = build_features(X_train_raw, X_test_raw, transform)
    eig = hessian_eigenvalues(X_train)
    limit = 2.0 / eig.max()

    # closed-form optimum from the external solver
    opt = LinearRegression().fit(X_train, y_train)
    opt_train_mse = mse(y_train, opt.predict(X_train))

    # tol=0 so every run uses the full iteration budget
    Z_train, Z_test = add_intercept(X_train), add_intercept(X_test)
    theta, history = gradient_descent(Z_train, y_train, lr=lr, max_iters=ITERS, tol=0.0)
    gd_train_mse = mse(y_train, Z_train @ theta)

    row = {
        "Learning rate": lr,
        "Stability limit 2/L": limit,
        "Condition number": eig.max() / eig.min(),
        "Updates to R2 > 0": updates_until_below(history, np.var(y_train)),
        "Updates to within 1% of optimum": updates_until_below(history, 1.01 * opt_train_mse),
        "GD train MSE": gd_train_mse,
        "Optimal train MSE": opt_train_mse,
        "GD excess MSE (%)": 100 * (gd_train_mse / opt_train_mse - 1),
        "GD train R2": r2(y_train, Z_train @ theta),
        "GD test MSE": mse(y_test, Z_test @ theta),
        "GD test R2": r2(y_test, Z_test @ theta),
        "Optimal train R2": r2(y_train, opt.predict(X_train)),
        "Optimal test R2": r2(y_test, opt.predict(X_test)),
        "GD intercept": theta[0],
        "Optimal intercept": opt.intercept_,
    }

    sweep = []
    for factor in SWEEP_FACTORS:
        with np.errstate(over="ignore", invalid="ignore"):
            _, h = gradient_descent(Z_train, y_train, lr=factor * limit,
                                    max_iters=ITERS, tol=0.0)
        stable = bool(np.isfinite(h[-1]) and h[-1] < h[0])
        sweep.append({
            "Learning rate / limit": factor,
            "Learning rate": factor * limit,
            "Final train MSE": h[-1],
            "Outcome": "converges" if stable else "diverges",
        })

    return row, sweep, np.asarray(history), opt_train_mse


def plot_convergence(results, var_y, name):
    """(a) training MSE on log-log axes; (b) excess MSE over each optimum."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    iters = np.arange(1, ITERS + 1)

    for label, lr, color, history, opt_mse in results:
        legend = f"{label} (α = {fmt_lr(lr)})"
        ax_a.plot(iters, history, color=color, label=legend)
        # below ~1e-10 the gap is floating-point noise, so stop drawing there
        excess = np.where(history - opt_mse > 1e-10, history - opt_mse, np.nan)
        ax_b.plot(iters, excess, color=color)
        floor = np.flatnonzero(np.isnan(excess))
        if floor.size:
            ax_b.annotate(f"machine precision\nafter {floor[0]:,} iterations",
                          xy=(floor[0], 1e-10), xytext=(floor[0] + 300, 1e-8),
                          color=INK_2, fontsize=7.5,
                          arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

    # reference lines stay a step lighter than the data
    ax_a.axhline(var_y, color=AXIS, linewidth=0.9, zorder=1)
    ax_a.text(300, var_y * 1.07, r"$R^2 = 0$", color=INK_2, fontsize=7.5)
    optima = sorted({round(r[4], 6) for r in results}, reverse=True)
    for i, opt_mse in enumerate(optima):
        ax_a.axhline(opt_mse, color=AXIS, linewidth=0.9, zorder=1)
        offset, va = (0.93, "top") if i == 0 else (1.07, "bottom")
        ax_a.text(1.3, opt_mse * offset, f"optimum {opt_mse:.1f}", color=INK_2,
                  fontsize=7.5, va=va)

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_ylim(min(optima) * 0.8, None)
    ax_a.set_xlabel("Iteration (log scale)")
    ax_a.set_ylabel("Training MSE")
    ax_a.set_title("(a) Training loss", loc="left")

    ax_b.set_yscale("log")
    ax_b.set_ylim(1e-11, 1e4)
    ax_b.set_xlabel("Iteration")
    ax_b.set_ylabel(r"MSE $-$ optimal MSE")
    ax_b.set_title("(b) Distance from the closed-form optimum", loc="left")

    fig.legend(loc="upper center", ncol=len(results), bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    return save_figure(fig, name)


def main():
    use_report_style()
    train_df, test_df = train_test_split(load_data())
    X_train_raw, y_train = to_arrays(train_df)
    X_test_raw, y_test = to_arrays(test_df)

    rows, sweeps, curves = [], [], []
    for label, transform, lr, color in RUNS:
        row, sweep, history, opt_mse = run_config(
            transform, lr, X_train_raw, X_test_raw, y_train, y_test)
        rows.append({"Model": label, **row})
        sweeps += [{"Model": label, **s} for s in sweep]
        curves.append((label, lr, color, history, opt_mse))

    diagnostics = pd.DataFrame(rows).set_index("Model")
    sweep = pd.DataFrame(sweeps)

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(diagnostics.T.to_string(float_format=lambda v: f"{v:.6g}"))
        print()
        print(sweep.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print()

    save_table(diagnostics, "convergence_diagnostics.csv", float_format="%.6g")
    save_table(sweep, "learning_rate_sweep.csv", index=False, float_format="%.6g")

    var_y = float(np.var(y_train))
    plot_convergence(curves[:2], var_y, "convergence_scaling.png")
    plot_convergence(curves, var_y, "convergence_log_transform.png")


if __name__ == "__main__":
    main()
