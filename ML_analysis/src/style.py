"""Shared publication-quality plot style: large fonts, thick axes, one
consistent color per model, applied uniformly across every figure so a
reader can compare plots at a glance."""
import matplotlib.pyplot as plt

MODEL_COLORS = {
    "GBR": "#1b7837",
    "XGBR": "#d73027",
    "RandomForest": "#4575b4",
    "DTR": "#f46d43",
    "CatBoost": "#762a83",
}

# Fixed, high-contrast pair for train-vs-test scatter (parity) plots -- kept
# constant across every property panel so a reader learns the color coding
# once, rather than a per-model color that changes meaning panel to panel.
TRAIN_COLOR = "#F2A900"   # amber -- warm, reads clearly against white
TEST_COLOR = "#0B3D91"    # deep blue -- strong contrast against amber and gray

PROPERTY_LABELS = {
    "seebeck_coefficient": "Seebeck coefficient",
    "electrical_conductivity": "Electrical conductivity",
    "thermal_conductivity": "Thermal conductivity",
    "power_factor": "Power factor",
    "ZT": "ZT",
}

ACCENT = "#2166ac"
ACCENT2 = "#b2182b"


def apply_publication_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 18,
        "font.family": "sans-serif",
        "axes.titlesize": 22,
        "axes.titleweight": "bold",
        "axes.labelsize": 20,
        "axes.labelweight": "bold",
        "axes.linewidth": 2.2,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "xtick.major.width": 2.2,
        "ytick.major.width": 2.2,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "legend.fontsize": 14,
        "legend.frameon": False,
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
