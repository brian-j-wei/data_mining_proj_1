from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Falls back to the script's own folder if the repo layout isn't present
DATA = ROOT / "data/concrete+compressive+strength/Concrete_Data.xls"
if not DATA.exists():
    DATA = HERE / "Concrete_Data.xls"
OUT_DIR = ROOT / "assets" if (ROOT / "assets").is_dir() else HERE

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

rows = []
for i, name in enumerate(FEATURES):
    # Optimal univariate model: this predictor alone
    uni = LinearRegression().fit(X_train[:, [i]], y_train)

    rows.append({
        "Feature": name,
        "Univariate Coef": round(uni.coef_[0], 4),
        "Multivariate Coef": round(multi.coef_[i], 4),
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
