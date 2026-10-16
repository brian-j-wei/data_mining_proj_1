from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Falls back to the script's own folder if the repo layout isn't present
DATA = ROOT / "data/concrete+compressive+strength/Concrete_Data.xls"
if not DATA.exists():
    DATA = HERE / "Concrete_Data.xls"
OUT_DIR = ROOT / "assets/tables" if (ROOT / "assets").is_dir() else HERE
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

# Standardize features (fit on training data only)
X_train_scaled = StandardScaler().fit_transform(X_train)

# Log transform features: log(x + 1)
X_train_log = np.log1p(X_train)

configs = [
    ("Untransformed", X_train),
    ("Standardized", X_train_scaled),
    ("Log-Transformed", X_train_log),
]

pvals = {}
for name, X in configs:
    # add_constant gives the intercept term; statsmodels does not add one
    model = sm.OLS(y_train, sm.add_constant(X)).fit()
    # params[0] is the intercept, so feature p-values start at index 1
    pvals[name] = model.pvalues[1:]

table = pd.DataFrame(pvals, index=FEATURES)
table.index.name = "Feature"

print(table.to_string(float_format=lambda v: f"{v:.4g}"))

out_path = OUT_DIR / "pvalues.csv"
table.to_csv(out_path, float_format="%.6g")
print(f"\nsaved: {out_path}")

# Significance at alpha = 0.05, for reference
print("\nSignificant at alpha=0.05:")
print((table < 0.05).to_string())
