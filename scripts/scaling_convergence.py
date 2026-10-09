import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.linear import MultivariateLinearModel
from utils.data import load_data, to_arrays, train_test_split
from utils.metrics import mse, r2
from utils.plotting import plot_loss_curves

from scripts.train import build_features

FIG_DIR = Path(__file__).resolve().parents[1] / "assets" / "figures"

# Largest stable rate for each: untransformed diverges above ~5e-7, while
# standardized features tolerate ~1e6x larger steps.
RUNS = [
    ("Untransformed", "none", 5e-7),
    ("Standardized", "standardized", 0.1),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--out", default=str(FIG_DIR / "scaling_convergence.png"))
    return p.parse_args()


def main():
    args = parse_args()

    train_df, test_df = train_test_split(load_data())
    X_train_raw, y_train = to_arrays(train_df)
    X_test_raw, y_test = to_arrays(test_df)

    histories = []
    for label, transform, lr in RUNS:
        X_train, X_test = build_features(X_train_raw, X_test_raw, transform)

        # tol=0 so every run uses the full iteration budget, keeping curves comparable
        model = MultivariateLinearModel(lr=lr, max_iters=args.iters, tol=0.0)
        model.fit(X_train, y_train)
        histories.append((f"{label} (lr={lr:g})", model.loss_history_))

        tr_pred, te_pred = model.predict(X_train), model.predict(X_test)
        print(f"{label:15s} lr={lr:<8g} "
              f"train MSE={mse(y_train, tr_pred):9.4f} R2={r2(y_train, tr_pred):7.4f}  "
              f"test MSE={mse(y_test, te_pred):9.4f} R2={r2(y_test, te_pred):7.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plot_loss_curves(
        histories,
        args.out,
        title="Gradient descent convergence: untransformed vs. standardized features",
        inset_iters=200,
    )
    print(f"\nfigure: {args.out}")


if __name__ == "__main__":
    main()
