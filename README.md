# CSE5104 FL26 Project 1: Concrete Compressive Strength

Linear regression analysis of the UCI Concrete Compressive Strength dataset
(1030 instances), with gradient descent implemented from scratch.

**Data split:** rows 501–630 (1-indexed) are the 130-sample test set; the
remaining 900 rows are the training set. Test data is never used to update
model parameters, and all feature transformations are fit on the training set
only.

## Setup

Requires Python 3.9+. No GPU needed; the full analysis runs on CPU in seconds.

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Using conda instead? A venv inside an activated conda env works fine, or:

```bash
conda create -n cse5104 python=3.12 --file requirements.txt
conda activate cse5104
```

Avoid installing into conda `base` — mixing pip and conda there is what breaks
numpy/pandas ABI compatibility.

## Reproducing the results

Run from the repository root, in this order. Each script is independent.

```bash
# Q2/Q3: gradient descent, scaling & convergence speed
#   -> assets/figures/gradient_descent_convergence.png
python scripts/gradient_descent_self_contained.py

# Q5: log transformation, all three loss curves
#   -> assets/figures/log_transformation_convergence.png
python scripts/log_transformation.py

# Q5: summary table across the three optimal models
#   -> assets/summary_table.csv
python scripts/summary_table.py

# Q6: univariate vs. multivariate coefficients
#   -> assets/univariate_vs_multivariate.csv, assets/feature_correlations.csv
python scripts/univariate_vs_multivariate.py

# Q7: p-values across the three configurations
#   -> assets/pvalues.csv
python scripts/pvalues.py
```

The two gradient descent scripts are self-contained: they depend only on the
listed packages and can be run from any directory, or copied out of the repo
alongside `Concrete_Data.xls`.

### Exploratory interface

`train.py` and `eval.py` are a parameterized version of the same analysis,
useful for testing other learning rates:

```bash
python -m scripts.train --transform standardized --lr 0.01 --iters 5000
python -m scripts.train --feature Cement --lr 1e-6 --iters 5000
python -m scripts.eval --transform log
python -m scripts.scaling_convergence
```

## Hyperparameters

Each learning rate is the largest stable value for that feature scaling, found
by testing powers of ten and confirmed against the theoretical limit 2/L from
the Hessian's largest eigenvalue. All runs use 5000 iterations.

| Model | Learning rate | Diverges above | Condition number |
|---|---|---|---|
| Untransformed | 3e-7 | ~6.0e-7 | 1.2e10 |
| Standardized | 0.2 | ~0.43 | 8.7e1 |
| Log-transformed | 5e-3 | ~5.7e-3 | 1.6e7 |

## Layout

- `data/` — UCI dataset and its original readme
- `scripts/` — one script per assignment question (see above)
- `utils/` — `grad_descent.py`, `metrics.py`, `data.py`, `transforms.py`, `plotting.py`
- `models/` — `linear.py` (univariate + multivariate wrappers)
- `assets/` — assignment PDF, generated figures and tables

## Dependencies

numpy, pandas, matplotlib, scikit-learn, statsmodels, xlrd

`xlrd` is required for `pandas.read_excel()` to open the `.xls` file.
scikit-learn and statsmodels are used only for `StandardScaler`, the external
optimal-solution baselines (`LinearRegression`), and p-values (`statsmodels.OLS`)
— the gradient descent optimizer is implemented from scratch in
`scripts/gradient_descent_self_contained.py` and `utils/grad_descent.py`.
