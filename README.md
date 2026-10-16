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

`bash run_all.sh` runs every analysis script below in order (well under a
minute); `bash run_all.sh --reports` also rebuilds the Part 2/3 reports. To
run scripts individually, run them from the repository root. Each script is
independent. Figures go to `assets/figures/`, tables to `assets/tables/`.

```bash
# Q2/Q3: gradient descent, scaling & convergence speed
#   -> figures/gradient_descent_convergence.png
python scripts/gradient_descent_self_contained.py

# Q5: log transformation, all three loss curves
#   -> figures/log_transformation_convergence.png
python scripts/log_transformation.py

# Q5: summary table across the three optimal models
#   -> tables/summary_table.csv
python scripts/summary_table.py

# Q6: univariate vs. multivariate coefficients, omitted-variable decomposition
#   -> tables/univariate_vs_multivariate.csv, tables/feature_correlations.csv,
#      figures/feature_correlations.png
python scripts/univariate_vs_multivariate.py

# Q6: eight univariate models trained with our gradient descent
#   -> tables/univariate_gradient_descent.csv, figures/univariate_gradient_descent.png
python scripts/univariate_gradient_descent.py

# Q7: p-values across the three configurations
#   -> tables/pvalues.csv
python scripts/pvalues.py

# Part 3 support: conditioning, learning-rate sweep, GD vs. closed form
#   -> tables/convergence_diagnostics.csv, tables/learning_rate_sweep.csv,
#      figures/convergence_scaling.png, figures/convergence_log_transform.png
python scripts/convergence_diagnostics.py

# Part 3 support: closed-form coefficients for all three feature sets
#   -> tables/optimal_coefficients.csv
python scripts/optimal_coefficients.py

# Part 2 support: training-set statistics, client figures and tables
#   -> tables/descriptive_stats.csv, tables/model_reliability.csv,
#      tables/mix_adjustment_effects.csv, tables/water_cement_bands.csv, ...,
#      figures/client_*.png
python scripts/descriptive_stats.py
python scripts/client_insights.py
```

## Reports (Parts 2 and 3)

The client report and the technical appendix are written in Markdown under
`reports/` and built into formatted Word documents (Calibri 11 pt, 1-inch
margins, 1.15 spacing, numbered tables and figures, native Word equations):

```bash
conda install pandoc                  # or https://pandoc.org/installing.html
pip install -r reports/requirements.txt
python reports/build_reports.py       # -> reports/output/*.docx, prints word counts
python reports/build_reports.py --pdf # macOS + Microsoft Word: also export PDFs
```

The reports cite figures and tables from `assets/`, so run the analysis
scripts first. Edit the `.md` files, not the generated `.docx`.

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

```
data/                     UCI dataset and its original readme
scripts/                  one script per analysis (see above)
utils/                    grad_descent.py, metrics.py, data.py, transforms.py,
                          plotting.py, report_style.py (shared figure style)
models/                   linear.py (univariate + multivariate wrappers)
assets/
  CSE5104 ... .pdf        assignment
  figures/                generated figures (PNG, 300 dpi for the reports)
  tables/                 generated tables (CSV)
reports/
  part2_client_report.md       Part 2 source
  part3_technical_appendix.md  Part 3 source
  build_reports.py             Markdown -> Word (and PDF) build
  templates/reference.docx     generated Word style template
  output/                      built .docx / .pdf deliverables
run_all.sh                reproduce every figure and table
```

## Dependencies

numpy, pandas, matplotlib, scikit-learn, statsmodels, xlrd

`xlrd` is required for `pandas.read_excel()` to open the `.xls` file.
scikit-learn and statsmodels are used only for `StandardScaler`, the external
optimal-solution baselines (`LinearRegression`), and p-values (`statsmodels.OLS`)
— the gradient descent optimizer is implemented from scratch in
`scripts/gradient_descent_self_contained.py` and `utils/grad_descent.py`.
