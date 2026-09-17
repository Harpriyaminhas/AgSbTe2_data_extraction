"""
Step 5b: publication-styled summary figures --
  (1) model-comparison bar chart (test R^2 for all 5 algorithms per property,
      best model highlighted)
  (2) parity plot (predicted vs. actual, test set) for the best model per
      property

Usage: python3 src/make_summary_figures.py
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from macos_omp_fix import ensure_xgboost_can_load
ensure_xgboost_can_load()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

from style import apply_publication_style, MODEL_COLORS, PROPERTY_LABELS, ACCENT, TRAIN_COLOR, TEST_COLOR

LOG_AXIS_PROPERTIES = {"electrical_conductivity"}

ML_RUN_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ML_RUN_DIR / "outputs" / "models"
FIG_DIR = ML_RUN_DIR / "outputs" / "figures"
PROPERTIES = ["seebeck_coefficient", "electrical_conductivity", "thermal_conductivity", "power_factor", "ZT"]

PROPERTY_UNITS = {
    "seebeck_coefficient": "uV/K", "electrical_conductivity": "S/cm",
    "thermal_conductivity": "W/m.K", "power_factor": "uW/cm.K2", "ZT": "dimensionless",
}

apply_publication_style()


def model_comparison_figure():
    metrics = pd.read_csv(MODELS_DIR / "all_model_metrics.csv")
    n = len(PROPERTIES)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 7.5), sharey=False)
    for ax, prop in zip(axes, PROPERTIES):
        # Selection metric must match train_models.py's actual choice: for
        # log-scale properties (electrical conductivity, power factor) the
        # best model is chosen by log-space R^2 (robust to single high-
        # leverage points in a small test set), not linear-space R^2 -- read
        # the actual saved choice rather than re-deriving it, so this figure
        # never contradicts best_model.txt / the parity plots.
        best_name = (MODELS_DIR / prop / "best_model.txt").read_text().strip()
        sub = metrics[metrics["property"] == prop].sort_values("test_r2", ascending=False)
        colors = [MODEL_COLORS.get(m, "#888888") for m in sub["model"]]
        edgecolors = ["black" if m == best_name else "none" for m in sub["model"]]
        linewidths = [3.0 if m == best_name else 0 for m in sub["model"]]
        bars = ax.bar(sub["model"], sub["test_r2"], color=colors, edgecolor=edgecolors, linewidth=linewidths)
        ax.set_title(PROPERTY_LABELS.get(prop, prop), fontsize=18, fontweight="bold")
        ax.set_ylabel("Test R$^2$" if prop == PROPERTIES[0] else "")
        ax.axhline(0, color="black", linewidth=1.5)
        ax.set_ylim(min(-0.1, sub["test_r2"].min() - 0.1), 1.0)
        ax.tick_params(axis="x", rotation=35, labelsize=13)
        for b, v in zip(bars, sub["test_r2"]):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.02 if v >= 0 else -0.06),
                     f"{v:.2f}", ha="center", fontsize=12, fontweight="bold")
    fig.suptitle("Model comparison across 5 thermoelectric properties (test-set R$^2$; best model outlined)\n"
                  "Note: for electrical conductivity and power factor, the outlined model is chosen by "
                  "log-space R$^2$ (Table 3/6), not by the linear-space bars shown here",
                  fontsize=16, fontweight="bold", y=1.06)
    plt.tight_layout()
    out = FIG_DIR / "model_comparison_all_properties.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def parity_plots():
    metrics = pd.read_csv(MODELS_DIR / "all_model_metrics.csv")
    for prop in PROPERTIES:
        prop_dir = MODELS_DIR / prop
        best_name = (prop_dir / "best_model.txt").read_text().strip()
        row = metrics[(metrics["property"] == prop) & (metrics["model"] == best_name)].iloc[0]

        with open(prop_dir / f"best_model_{best_name}.pkl", "rb") as f:
            payload = pickle.load(f)
        model, feature_cols, log_target = payload["model"], payload["features"], payload["log_target"]

        train_df = pd.read_csv(prop_dir / "train.csv")
        test_df = pd.read_csv(prop_dir / "test.csv")
        y_train_true = train_df["target_value"]
        y_test_true = test_df["target_value"]
        pred_train = model.predict(train_df[feature_cols])
        pred_test = model.predict(test_df[feature_cols])
        if log_target:
            pred_train = 10 ** pred_train
            pred_test = 10 ** pred_test

        train_r2 = r2_score(y_train_true, pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train_true, pred_train))

        use_log_axis = prop in LOG_AXIS_PROPERTIES
        fig, ax = plt.subplots(figsize=(8, 8))
        lo = min(y_train_true.min(), y_test_true.min(), pred_train.min(), pred_test.min())
        hi = max(y_train_true.max(), y_test_true.max(), pred_train.max(), pred_test.max())
        if use_log_axis:
            lo = max(lo, 1e-3)
            lo_lim, hi_lim = lo / 1.5, hi * 1.5
        else:
            pad = 0.05 * (hi - lo if hi > lo else 1)
            lo_lim, hi_lim = lo - pad, hi + pad
        ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim], "--", color="gray", linewidth=2, zorder=1)
        ax.scatter(y_train_true, pred_train, s=70, color=TRAIN_COLOR, edgecolor="#7a5200",
                   linewidth=1.0, alpha=0.85, zorder=2, label=f"Train (n={len(y_train_true)}, R$^2$={train_r2:.3f})")
        ax.scatter(y_test_true, pred_test, s=90, color=TEST_COLOR, edgecolor="black",
                   linewidth=1.2, alpha=0.9, zorder=3,
                   label=f"Test (n={len(y_test_true)}, R$^2$={row['test_r2']:.3f})")
        if use_log_axis:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlim(lo_lim, hi_lim)
        ax.set_ylim(lo_lim, hi_lim)
        ax.set_xlabel(f"Actual {PROPERTY_LABELS.get(prop, prop)} ({PROPERTY_UNITS[prop]})")
        ax.set_ylabel(f"Predicted {PROPERTY_LABELS.get(prop, prop)} ({PROPERTY_UNITS[prop]})")
        ax.set_title(f"{PROPERTY_LABELS.get(prop, prop)}: {best_name} (best model)\n"
                      f"Train RMSE={train_rmse:.3g}, Test RMSE={row['test_rmse']:.3g}", fontsize=17, fontweight="bold")
        ax.legend(loc="upper left", fontsize=12, frameon=True)
        ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()
        out = FIG_DIR / prop / f"parity_{prop}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}  (train R2={train_r2:.3f}, test R2={row['test_r2']:.3f})")


if __name__ == "__main__":
    model_comparison_figure()
    parity_plots()
