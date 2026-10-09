from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Falls back to the script's own folder if the repo layout isn't present
DATA = ROOT / "data/concrete+compressive+strength/Concrete_Data.xls"
if not DATA.exists():
    DATA = HERE / "Concrete_Data.xls"
FIG_DIR = ROOT / "assets/figures" if (ROOT / "assets/figures").is_dir() else HERE

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

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def add_intercept(X):
    # Prepend a column of ones so the intercept is just another weight
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.hstack([np.ones((X.shape[0], 1)), X])


# Gradient descent
def gradient_descent(X, y, learning_rate, iterations):
    X = add_intercept(X)

    # theta[0] is the intercept, theta[1:] are the feature weights
    theta = np.zeros(X.shape[1])
    # Record MSE at each iteration to plot losses
    losses = []

    for i in range(iterations):
        # Implements y = X @ theta, where the first column of X is all ones
        predictions = X @ theta

        # Calculate errors and MSE
        errors = predictions - y
        mse = np.mean(errors ** 2)
        losses.append(mse)

        # One gradient for all parameters, including the intercept
        gradient = (2 / len(y)) * (X.T @ errors)

        # Update parameters
        theta = theta - learning_rate * gradient

    return theta, losses


# Calculate MSE and R squared
def evaluate(X, y, theta):
    predictions = add_intercept(X) @ theta
    mse = np.mean((y - predictions) ** 2)
    r2 = 1 - mse / np.var(y)

    return mse, r2


# Train both models
theta1, loss1 = gradient_descent(X_train, y_train, 0.0000003, 5000)
theta2, loss2 = gradient_descent(X_train_scaled, y_train, 0.2, 5000)

# Evaluate unscaled model
print("Unscaled Model")
print("Training", evaluate(X_train, y_train, theta1))
print("Testing", evaluate(X_test, y_test, theta1))
print("\n")

# Evaluate scaled model
print("Scaled Model")
print("Training", evaluate(X_train_scaled, y_train, theta2))
print("Testing", evaluate(X_test_scaled, y_test, theta2))

# Plot losses
plt.plot(loss1, label="Unscaled (lr=3e-7)")
plt.plot(loss2, label="Scaled (lr=0.2)")

plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.yscale("log")
plt.title("Gradient Descent Convergence")
plt.legend()
plt.savefig(FIG_DIR / "gradient_descent_convergence.png", dpi=200,
            bbox_inches="tight")
plt.show()
