#!/usr/bin/env bash
# Reproduce every figure and table, in order. Run from anywhere:
#   bash run_all.sh            analysis only
#   bash run_all.sh --reports  analysis, then rebuild the Part 2/3 Word reports
set -euo pipefail
cd "$(dirname "$0")"

# matplotlib would otherwise open a window for the two scripts that call plt.show()
export MPLBACKEND=Agg
export PYTHONWARNINGS="ignore:FigureCanvasAgg is non-interactive"

scripts=(
  # Part 1 analyses
  scripts/gradient_descent_self_contained.py   # Q2/Q3 loss curves
  scripts/log_transformation.py                # Q5 loss curves
  scripts/summary_table.py                     # Q5 summary table
  scripts/univariate_vs_multivariate.py        # Q6 coefficients, correlations
  scripts/pvalues.py                           # Q7 p-values
  # Part 3 supporting analyses
  scripts/convergence_diagnostics.py
  scripts/optimal_coefficients.py
  scripts/univariate_gradient_descent.py
  # Part 2 client report
  scripts/descriptive_stats.py
  scripts/client_insights.py
)

for script in "${scripts[@]}"; do
  echo "==> $script"
  python "$script" > /dev/null
done

if [[ "${1:-}" == "--reports" ]]; then
  echo "==> reports/build_reports.py"
  python reports/build_reports.py
fi

echo "done: figures in assets/figures/, tables in assets/tables/"
