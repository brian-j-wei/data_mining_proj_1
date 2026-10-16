from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LinearRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Falls back to the script's own folder if the repo layout isn't present
DATA = ROOT / "data/concrete+compressive+strength/Concrete_Data.xls"
if not DATA.exists():
    DATA = HERE / "Concrete_Data.xls"
OUT_DIR = ROOT / "assets/tables" if (ROOT / "assets").is_dir() else HERE
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT / "assets/figures" if (ROOT / "assets/figures").is_dir() else HERE

FEATURES = [
    "Cement", "Blast Furnace Slag", "Fly Ash", "Water",
    "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age",
]

# Load the dataset
df = pd.read_excel(DATA)

# Split into training and testing data, where rows 501-630 are testing data
train = pd.concat([df.iloc[:500], df.iloc[630:]])

X_train = train.iloc[:, :8].to_numpy()
y_train = train.iloc[:, 8].to_numpy()

# Optimal multivariate model: all eight predictors at once
multi = LinearRegression().fit(X_train, y_train)

# Omitted-variable decomposition. Because the multivariate residuals are
# uncorrelated with every predictor, each univariate slope splits exactly into
#   b_j = beta_j + sum_{k != j} beta_k * Cov(x_j, x_k) / Var(x_j)
# where the sum is the bias picked up from the predictors left out.
cov = np.cov(X_train, rowvar=False)
beta = multi.coef_

rows = []
for i, name in enumerate(FEATURES):
    # Optimal univariate model: this predictor alone
    uni = LinearRegression().fit(X_train[:, [i]], y_train)

    contributions = beta * cov[i] / cov[i, i]
    contributions[i] = 0.0
    assert np.isclose(uni.coef_[0], beta[i] + contributions.sum())
    largest = np.argmax(np.abs(contributions))

    rows.append({
        "Feature": name,
        "Univariate Coef": round(uni.coef_[0], 4),
        "Multivariate Coef": round(beta[i], 4),
        "Omitted-Variable Term": round(contributions.sum(), 4),
        "Largest Contributor": FEATURES[largest],
        "Largest Contribution": round(contributions[largest], 4),
    })

table = pd.DataFrame(rows)
print(table.to_string(index=False))
print(f"\nMultivariate intercept: {multi.intercept_:.4f}")

out_path = OUT_DIR / "univariate_vs_multivariate.csv"
table.to_csv(out_path, index=False)
print(f"saved: {out_path}")

# Correlations among predictors explain why the two sets of coefficients differ
corr = pd.DataFrame(X_train, columns=FEATURES).corr().round(3)
print("\nPredictor correlation matrix:")
print(corr.to_string())
corr.to_csv(OUT_DIR / "feature_correlations.csv")

# Heatmap of the lower triangle: blue = negative, red = positive, gray = none
cmap = LinearSegmentedColormap.from_list(
    "diverging", ["#184f95", "#6da7ec", "#f0efec", "#ec8b8a", "#b83232"])
masked = np.where(np.tril(np.ones_like(corr, dtype=bool), k=-1), corr, np.nan)

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
fig, ax = plt.subplots(figsize=(5.6, 4.6))
image = ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1)
for r in range(len(FEATURES)):
    for c in range(r):
        value = corr.iloc[r, c]
        ax.text(c, r, f"{value:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(value) > 0.45 else "#0b0b0b")

ax.set_xticks(range(len(FEATURES) - 1), FEATURES[:-1], rotation=40, ha="right", fontsize=8)
ax.set_yticks(range(1, len(FEATURES)), FEATURES[1:], fontsize=8)
ax.set_xlim(-0.5, len(FEATURES) - 1.5)
ax.set_ylim(len(FEATURES) - 0.5, 0.5)
for side in ax.spines.values():
    side.set_visible(False)
ax.tick_params(length=0, colors="#52514e")

bar = fig.colorbar(image, ax=ax, shrink=0.75, pad=0.02)
bar.set_label("Pearson correlation (training set)", fontsize=8, color="#52514e")
bar.ax.tick_params(labelsize=7, colors="#52514e", length=0)
bar.outline.set_visible(False)

fig.tight_layout()
fig_path = FIG_DIR / "feature_correlations.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"saved: {fig_path}")
