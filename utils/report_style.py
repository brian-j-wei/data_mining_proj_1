"""Shared output paths, display names, and figure style for the report figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "assets" / "figures"
TABLE_DIR = ROOT / "assets" / "tables"

DISPLAY_NAMES = {
    "Cement": "Cement",
    "BlastFurnaceSlag": "Blast Furnace Slag",
    "FlyAsh": "Fly Ash",
    "Water": "Water",
    "Superplasticizer": "Superplasticizer",
    "CoarseAggregate": "Coarse Aggregate",
    "FineAggregate": "Fine Aggregate",
    "Age": "Age",
    "CompressiveStrength": "Compressive Strength",
}

UNITS = {
    "Age": "days",
    "CompressiveStrength": "MPa",
}  # every other column is kg per m^3 of mix

# Categorical slots 1-3 of the reference palette; these three pass the
# all-pairs colorblind checks on a light surface, so no chart uses more.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
POINT_GRAY = "#b9b8b2"


def unit(column):
    return UNITS.get(column, "kg/m³")


def use_report_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelsize": 9,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": INK_2,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def save_figure(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"figure: {path.relative_to(ROOT)}")
    return path


def save_table(df, name, **kwargs):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / name
    df.to_csv(path, **kwargs)
    print(f"table:  {path.relative_to(ROOT)}")
    return path
