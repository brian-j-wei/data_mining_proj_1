"""Part 2, Table 1: descriptive statistics of the 900-sample training set."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data import FEATURES, TARGET, load_data, train_test_split
from utils.report_style import DISPLAY_NAMES, save_table, unit


def main():
    train, _ = train_test_split(load_data())
    columns = FEATURES + [TARGET]

    # pandas std() is the sample standard deviation (ddof=1)
    stats = train[columns].agg(["mean", "std", "min", "max"]).T
    stats.columns = ["Mean", "Std. Dev.", "Min", "Max"]
    stats.insert(0, "Unit", [unit(c) for c in columns])
    stats.index = [DISPLAY_NAMES[c] for c in columns]
    stats.index.name = "Variable"
    stats = stats.round(2)

    print(f"training rows: {len(train)}\n")
    print(stats.to_string())
    save_table(stats, "descriptive_stats.csv")


if __name__ == "__main__":
    main()
