"""Part 2: figures and tables for the client report.

The "final model" is the closed-form multivariate regression on log(x + 1)
features, which generalizes best (see summary_table.py). Effects are reported
for practical changes to a typical mix: every ingredient at its training
median, cured 28 days.

Outputs
  assets/tables/model_reliability.csv
  assets/tables/mix_adjustment_effects.csv
  assets/tables/typical_mix_strength_by_age.csv
  assets/tables/water_cement_bands.csv
  assets/tables/superplasticizer_comparison.csv
  assets/figures/client_strength_by_age.png
  assets/figures/client_strength_vs_water_cement.png
  assets/figures/client_predicted_vs_measured.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess

from utils.data import FEATURES, TARGET, load_data, to_arrays, train_test_split
from utils.metrics import mse, r2
from utils.report_style import (
    AXIS, BLUE, INK_2, MUTED, ORANGE, POINT_GRAY, save_figure, save_table,
    use_report_style,
)
from utils.transforms import log1p_transform

AGE_TICKS = [1, 3, 7, 14, 28, 56, 90, 180, 365]
WC_BANDS = [0, 0.40, 0.50, 0.60, 0.80, np.inf]
WC_LABELS = ["0.40 or less", "0.40 to 0.50", "0.50 to 0.60", "0.60 to 0.80", "above 0.80"]

# (description, feature, change); change is added to the typical-mix value
ADJUSTMENTS = [
    ("Add 50 kg/m³ cement", "Cement", 50),
    ("Remove 10 kg/m³ water", "Water", -10),
    ("Remove 15 kg/m³ water", "Water", -15),
    ("Remove 26 kg/m³ water (typical saving with superplasticizer)", "Water", -26),
    ("Add 50 kg/m³ blast furnace slag", "BlastFurnaceSlag", 50),
    ("Add 5 kg/m³ superplasticizer", "Superplasticizer", 5),
    ("Add 50 kg/m³ fly ash", "FlyAsh", 50),
    ("Add 50 kg/m³ coarse aggregate", "CoarseAggregate", 50),
    ("Add 50 kg/m³ fine aggregate", "FineAggregate", 50),
]


def evidence(p):
    if p < 0.001:
        return "Strong"
    if p < 0.05:
        return "Moderate"
    return "Weak (not reliable)"


def typical_mix(X_train):
    mix = np.median(X_train, axis=0)
    mix[FEATURES.index("Age")] = 28
    return mix


def predict_one(model, mix):
    return float(model.predict(log1p_transform(mix.reshape(1, -1)))[0])


def reliability_table(X_train, y_train, X_test, y_test):
    rows = []
    for label, transform in [("Simple straight-line model", None),
                             ("Final model (log-transformed)", log1p_transform)]:
        f = transform or (lambda X: X)
        model = LinearRegression().fit(f(X_train), y_train)
        pred_train, pred_test = model.predict(f(X_train)), model.predict(f(X_test))
        rows.append({
            "Model": label,
            "Training R2": r2(y_train, pred_train),
            "Testing R2": r2(y_test, pred_test),
            "Training typical error (MPa)": np.sqrt(mse(y_train, pred_train)),
            "Testing typical error (MPa)": np.sqrt(mse(y_test, pred_test)),
            "Testing average over-prediction (MPa)": float(np.mean(pred_test - y_test)),
            "Testing share within 10 MPa": float(np.mean(np.abs(pred_test - y_test) <= 10)),
        })
    return pd.DataFrame(rows).set_index("Model")


def effects_table(model, pvalues, mix):
    base = predict_one(model, mix)
    rows = []
    for description, feature, change in ADJUSTMENTS:
        i = FEATURES.index(feature)
        new = mix.copy()
        new[i] += change
        rows.append({
            "Adjustment": description,
            "From": mix[i],
            "To": new[i],
            "Strength change (MPa)": predict_one(model, new) - base,
            "Evidence": evidence(pvalues[i]),
        })

    age = FEATURES.index("Age")
    for start, end in [(3, 7), (7, 28), (28, 56), (28, 90)]:
        early, late = mix.copy(), mix.copy()
        early[age], late[age] = start, end
        rows.append({
            "Adjustment": f"Cure {end} days instead of {start}",
            "From": start,
            "To": end,
            "Strength change (MPa)": predict_one(model, late) - predict_one(model, early),
            "Evidence": evidence(pvalues[age]),
        })

    swap = mix.copy()
    swap[FEATURES.index("Cement")] -= 50
    swap[FEATURES.index("BlastFurnaceSlag")] += 50
    rows.append({
        "Adjustment": "Replace 50 kg/m³ cement with slag",
        "From": np.nan,
        "To": np.nan,
        "Strength change (MPa)": predict_one(model, swap) - base,
        "Evidence": evidence(max(pvalues[0], pvalues[1])),
    })
    return pd.DataFrame(rows).set_index("Adjustment"), base


def plot_strength_by_age(train, model, mix):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    rng = np.random.default_rng(0)
    jitter = np.exp(rng.uniform(-0.06, 0.06, len(train)))
    ax.scatter(train["Age"] * jitter, train[TARGET], s=7, color=POINT_GRAY,
               alpha=0.6, linewidths=0, label="Individual training mixes")

    medians = train.groupby("Age")[TARGET].agg(["median", "size"])
    medians = medians[medians["size"] >= 10]
    ax.plot(medians.index, medians["median"], "o", color=ORANGE, markersize=5.5,
            markeredgecolor="white", markeredgewidth=1.2,
            label="Median of mixes tested at that age")

    # past ~90 days the data thins out and the log curve keeps climbing while
    # measured medians level off, so the estimate is drawn only where it holds
    ages = np.geomspace(3, 90, 200)
    grid = np.tile(mix, (len(ages), 1))
    grid[:, FEATURES.index("Age")] = ages
    curve = model.predict(log1p_transform(grid))
    ax.plot(ages, curve, color=BLUE, label="Model estimate for a typical mix (3–90 days)")

    for day in (7, 28):
        value = curve[np.argmin(np.abs(ages - day))]
        ax.annotate(f"{value:.0f} MPa at {day} days", xy=(day, value),
                    xytext=(day * 0.42, value + 15), fontsize=7.5, color=INK_2,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9),
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

    ax.set_xscale("log")
    ax.set_xticks(AGE_TICKS, [str(a) for a in AGE_TICKS])
    ax.minorticks_off()
    ax.set_xlabel("Curing age (days, log scale)")
    ax.set_ylabel("Compressive strength (MPa)")
    ax.set_ylim(0, 90)
    ax.legend(loc="upper left", ncol=1)
    fig.tight_layout()
    return save_figure(fig, "client_strength_by_age.png")


def plot_strength_vs_wc(train28):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    with_sp = train28["Superplasticizer"] > 0
    ax.scatter(train28.loc[with_sp, "wc"], train28.loc[with_sp, TARGET], s=14,
               color=BLUE, alpha=0.75, linewidths=0.6, edgecolors="white",
               label="With superplasticizer")
    ax.scatter(train28.loc[~with_sp, "wc"], train28.loc[~with_sp, TARGET], s=14,
               color=ORANGE, alpha=0.85, linewidths=0.6, edgecolors="white",
               label="Without superplasticizer")

    trend = lowess(train28[TARGET], train28["wc"], frac=0.5)
    ax.plot(trend[:, 0], trend[:, 1], color=INK_2, linewidth=1.4, label="Overall trend")

    ax.axvline(0.50, color=AXIS, linewidth=0.9, zorder=1)
    ax.text(0.51, 84, "w/c = 0.50", fontsize=7.5, color=INK_2, va="top")

    ax.set_xlabel("Water-to-cement ratio (by weight)")
    ax.set_ylabel("28-day compressive strength (MPa)")
    ax.set_ylim(0, 90)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return save_figure(fig, "client_strength_vs_water_cement.png")


def plot_predicted_vs_measured(model, X_train, y_train, X_test, y_test, band):
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    lims = np.array([0, 90])
    ax.fill_between(lims, lims - band, lims + band, color=BLUE, alpha=0.10,
                    linewidth=0, label=f"Within ±{band:.1f} MPa (typical error)")
    ax.plot(lims, lims, color=INK_2, linewidth=1.0, label="Perfect prediction")
    ax.scatter(y_train, model.predict(log1p_transform(X_train)), s=7,
               color=POINT_GRAY, alpha=0.6, linewidths=0, label="Training mixes (900)")
    ax.scatter(y_test, model.predict(log1p_transform(X_test)), s=16, color=BLUE,
               linewidths=0.6, edgecolors="white", label="Unseen test mixes (130)")

    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect("equal")
    ax.set_xlabel("Measured strength (MPa)")
    ax.set_ylabel("Predicted strength (MPa)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return save_figure(fig, "client_predicted_vs_measured.png")


def main():
    use_report_style()
    train, test = train_test_split(load_data())
    X_train, y_train = to_arrays(train)
    X_test, y_test = to_arrays(test)

    model = LinearRegression().fit(log1p_transform(X_train), y_train)
    pvalues = sm.OLS(y_train, sm.add_constant(log1p_transform(X_train))).fit().pvalues[1:]
    mix = typical_mix(X_train)
    print("typical mix:", dict(zip(FEATURES, np.round(mix, 2))))

    reliability = reliability_table(X_train, y_train, X_test, y_test)
    effects, base = effects_table(model, pvalues, mix)
    print(f"\ntypical mix, 28-day prediction: {base:.2f} MPa\n")

    age = FEATURES.index("Age")
    by_age = []
    for day in [3, 7, 14, 28, 56, 90]:
        m = mix.copy()
        m[age] = day
        by_age.append({"Age (days)": day, "Predicted strength (MPa)": predict_one(model, m)})
    by_age = pd.DataFrame(by_age).set_index("Age (days)")
    by_age["Share of 28-day strength"] = by_age["Predicted strength (MPa)"] / base

    t = train.copy()
    t["wc"] = t["Water"] / t["Cement"]
    t28 = t[t["Age"] == 28]
    grouped = t28.groupby(pd.cut(t28["wc"], WC_BANDS, labels=WC_LABELS), observed=True)[TARGET]
    bands = pd.DataFrame({
        "Mixes": grouped.size(),
        "Median (MPa)": grouped.median(),
        "Lowest 10% below (MPa)": grouped.quantile(0.10),
        "Share reaching 30 MPa": grouped.apply(lambda s: (s >= 30).mean()),
        "Share reaching 40 MPa": grouped.apply(lambda s: (s >= 40).mean()),
    })
    bands.index.name = "Water-to-cement ratio"

    sp = t.groupby(np.where(t["Superplasticizer"] > 0, "With", "Without"))
    sp_table = pd.DataFrame({
        "Mixes": sp.size(),
        "Mean water (kg/m³)": sp["Water"].mean(),
        "Mean strength, all ages (MPa)": sp[TARGET].mean(),
        "Mean 28-day strength (MPa)": t28.groupby(
            np.where(t28["Superplasticizer"] > 0, "With", "Without"))[TARGET].mean(),
    })
    sp_table.index.name = "Superplasticizer"

    age_group = pd.cut(t["Age"], [0, 7, 28, 365], labels=["7 days or less", "14 to 28 days", "over 28 days"])
    slag = t.pivot_table(index=age_group, columns=np.where(t["BlastFurnaceSlag"] > 0, "With slag", "No slag"),
                         values=TARGET, aggfunc="mean", observed=True)
    slag["Gap (MPa)"] = slag["With slag"] - slag["No slag"]
    slag.index.name = "Curing age"

    # The blind test rows differ from the training rows, which explains the bias
    composition = pd.DataFrame({
        split: {
            "Mixes": len(df),
            "Share using superplasticizer": float((df["Superplasticizer"] > 0).mean()),
            "Mean superplasticizer (kg/m³)": df["Superplasticizer"].mean(),
            "Mean strength (MPa)": df[TARGET].mean(),
        }
        for split, df in [("Training", train), ("Testing", test)]
    })

    binder = t["Cement"] + t["BlastFurnaceSlag"] + t["FlyAsh"]
    corr = pd.Series({
        "water / cement": t["wc"].corr(t[TARGET]),
        "water / (cement + slag + fly ash)": (t["Water"] / binder).corr(t[TARGET]),
    }, name="Correlation with strength")

    fmt = lambda v: f"{v:.3f}"
    for title, df in [("Model reliability", reliability), ("Mix adjustments", effects),
                      ("Typical mix by age", by_age), ("28-day strength by w/c", bands),
                      ("Superplasticizer", sp_table), ("Slag by curing age", slag),
                      ("Training vs. testing composition", composition),
                      ("Ratios", corr.to_frame())]:
        print(f"{title}\n{df.to_string(float_format=fmt)}\n")

    save_table(reliability, "model_reliability.csv", float_format="%.4f")
    save_table(effects, "mix_adjustment_effects.csv", float_format="%.3f")
    save_table(by_age, "typical_mix_strength_by_age.csv", float_format="%.3f")
    save_table(bands, "water_cement_bands.csv", float_format="%.3f")
    save_table(sp_table, "superplasticizer_comparison.csv", float_format="%.3f")
    save_table(slag, "slag_by_curing_age.csv", float_format="%.3f")
    save_table(composition, "train_test_composition.csv", float_format="%.3f")
    save_table(corr.to_frame(), "ratio_correlations.csv", float_format="%.3f")

    band = reliability.loc["Final model (log-transformed)", "Testing typical error (MPa)"]
    plot_strength_by_age(train, model, mix)
    plot_strength_vs_wc(t28)
    plot_predicted_vs_measured(model, X_train, y_train, X_test, y_test, band)


if __name__ == "__main__":
    main()
