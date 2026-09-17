"""
Step 5: SHAP global-importance bar plot + beeswarm plot for the best model
of each property, publication-styled (large fonts, thick axes, consistent
colors, saved at high DPI).

Usage: python3 src/shap_analysis.py
"""
import sys
import pickle
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from macos_omp_fix import ensure_xgboost_can_load
ensure_xgboost_can_load()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from style import apply_publication_style, PROPERTY_LABELS, ACCENT

warnings.filterwarnings("ignore")

ML_RUN_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ML_RUN_DIR / "outputs" / "models"
FIG_DIR = ML_RUN_DIR / "outputs" / "figures"
PROPERTIES = ["seebeck_coefficient", "electrical_conductivity", "thermal_conductivity", "power_factor", "ZT"]

apply_publication_style()


import re

_CUSTOM_NAMES = {
    "dopant_host_charge_mismatch": "Dopant-host charge mismatch",
    "dopant_host_radius_mismatch": "Dopant-host ionic-radius mismatch",
    "dopant_concentration": "Dopant concentration",
    "temperature_K": "Temperature (K)",
    "mixing_entropy": "Mixing entropy",
    "ionic_character": "Ionic character",
    "is_codoped": "Co-doped (0/1)",
    "num_dopants": "Number of dopants",
    "frac_Ag": "Ag site fraction", "frac_Sb": "Sb site fraction", "frac_Te": "Te site fraction",
}

_MAGPIE_PROPERTY_NAMES = {
    "AtomicWeight": "atomic mass", "MeltingT": "melting point", "Column": "periodic-table column",
    "Row": "periodic-table row", "CovalentRadius": "covalent radius", "Electronegativity": "electronegativity",
    "NsValence": "s-valence electron count", "NpValence": "p-valence electron count",
    "NdValence": "d-valence electron count", "NfValence": "f-valence electron count",
    "NValence": "valence electron count", "NsUnfilled": "unfilled s-valence states",
    "NpUnfilled": "unfilled p-valence states", "NdUnfilled": "unfilled d-valence states",
    "NfUnfilled": "unfilled f-valence states", "NUnfilled": "unfilled valence states",
    "GSvolume_pa": "ground-state volume/atom", "GSbandgap": "ground-state bandgap",
    "GSmagmom": "ground-state magnetic moment", "SpaceGroupNumber": "space group number",
    "MendeleevNumber": "Mendeleev number", "Number": "atomic number",
}
_STAT_NAMES = {
    "mean": "mean", "avg_dev": "avg. deviation", "range": "range",
    "minimum": "min", "maximum": "max", "mode": "mode",
}


def pretty_feature_name(name: str) -> str:
    if name in _CUSTOM_NAMES:
        return _CUSTOM_NAMES[name]
    if name.startswith("frac ") or name.startswith("avg "):
        return name[0].upper() + name[1:]
    if name.startswith("MagpieData "):
        rest = name[len("MagpieData "):]
        for stat, stat_label in _STAT_NAMES.items():
            if rest.startswith(stat + " "):
                prop = rest[len(stat) + 1:]
                prop_label = _MAGPIE_PROPERTY_NAMES.get(prop, re.sub(r"(?<!^)(?=[A-Z])", " ", prop).lower())
                return f"{prop_label.capitalize()} ({stat_label})"
        return rest
    return name


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for prop in PROPERTIES:
        prop_dir = MODELS_DIR / prop
        best_name = (prop_dir / "best_model.txt").read_text().strip()
        with open(prop_dir / f"best_model_{best_name}.pkl", "rb") as f:
            payload = pickle.load(f)
        model = payload["model"]
        features = payload["features"]

        train_df = pd.read_csv(prop_dir / "train.csv")
        X = train_df[features]

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        except Exception as e:
            print(f"{prop}: SHAP failed ({e}), skipping.")
            continue

        pretty_cols = [pretty_feature_name(c) for c in X.columns]
        X_pretty = X.copy()
        X_pretty.columns = pretty_cols

        prop_fig_dir = FIG_DIR / prop
        prop_fig_dir.mkdir(parents=True, exist_ok=True)
        label = PROPERTY_LABELS.get(prop, prop)

        # --- global bar plot ---
        fig = plt.figure(figsize=(11, 8))
        shap.summary_plot(shap_values, X_pretty, plot_type="bar", show=False, color=ACCENT, max_display=12)
        plt.title(f"{label}\nSHAP global feature importance ({best_name})", fontsize=20, fontweight="bold", pad=15)
        plt.xlabel("mean(|SHAP value|)", fontsize=18, fontweight="bold")
        ax = plt.gca()
        ax.tick_params(labelsize=14)
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        plt.tight_layout()
        plt.savefig(prop_fig_dir / f"shap_bar_{prop}.png", dpi=300)
        plt.close(fig)

        # --- beeswarm plot ---
        fig = plt.figure(figsize=(11, 8))
        shap.summary_plot(shap_values, X_pretty, plot_type="dot", show=False, max_display=12)
        plt.title(f"{label}\nSHAP beeswarm ({best_name})", fontsize=20, fontweight="bold", pad=15)
        plt.xlabel("SHAP value (impact on model output)", fontsize=18, fontweight="bold")
        ax = plt.gca()
        ax.tick_params(labelsize=14)
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        plt.tight_layout()
        plt.savefig(prop_fig_dir / f"shap_beeswarm_{prop}.png", dpi=300)
        plt.close(fig)

        mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns).sort_values(ascending=False)
        mean_abs_shap.to_csv(prop_fig_dir / f"shap_importance_{prop}.csv", header=["mean_abs_shap"])

        print(f"{prop:24s}: SHAP plots saved for best model {best_name}. Top feature: {mean_abs_shap.index[0]}")


if __name__ == "__main__":
    main()
