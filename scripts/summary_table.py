from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Falls back to the script's own folder if the repo layout isn't present
DATA = ROOT / "data/concrete+compressive+strength/Concrete_Data.xls"
if not DATA.exists():
    DATA = HERE / "Concrete_Data.xls"
OUT_DIR = ROOT / "assets" if (ROOT / "assets").is_dir() else HERE

# Load the dataset
df = pd.read_excel(DATA)

# Split into training and testing data, where rows 501-630 are testing data
test = df.iloc[500:630]
train = pd.concat([df.iloc[:500], df.iloc[630:]])

# Separate input features (X) and output variable (y)
X_train = train.iloc[:, :8].to_numpy()
y_train = train.iloc[:, 8].to_numpy()

X_test = test.iloc[:, :8].to_numpy()
y_test = test.iloc[:, 8].to_numpy()

# Standardize features (fit on training data only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Log transform features: log(x + 1)
X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)


def evaluate(model, X, y):
    predictions = model.predict(X)
    mse = np.mean((y - predictions) ** 2)
    r2 = 1 - mse / np.var(y)

    return mse, r2


# Fit each model to its optimum with sklearn's closed-form solver
configs = [
    ("Untransformed", X_train, X_test),
    ("Standardized", X_train_scaled, X_test_scaled),
    ("Log-Transformed", X_train_log, X_test_log),
]

rows = []
for name, Xtr, Xte in configs:
    model = LinearRegression().fit(Xtr, y_train)
    train_mse, train_r2 = evaluate(model, Xtr, y_train)
    test_mse, test_r2 = evaluate(model, Xte, y_test)
    rows.append({
        "Model": name,
        "Training MSE": round(train_mse, 4),
        "Testing MSE": round(test_mse, 4),
        "Training R2": round(train_r2, 4),
        "Testing R2": round(test_r2, 4),
    })

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))

out_path = OUT_DIR / "summary_table.csv"
summary.to_csv(out_path, index=False)
print(f"\nsaved: {out_path}")
