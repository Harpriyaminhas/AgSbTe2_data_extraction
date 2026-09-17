"""
Step 2: featurize each property's raw dataset -- parse full_formula into a
pymatgen Composition, compute matminer (magpie + valence-orbital) descriptors
plus custom descriptors from the descriptor doc that matminer can't provide
directly (mixing entropy, ionic character, dopant concentration), and merge
with temperature/co-doping metadata.

Usage: python3 src/make_features.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty, ValenceOrbital

sys.path.insert(0, str(Path(__file__).resolve().parent))
from composition_utils import (
    formula_to_composition, mixing_entropy, ionic_character,
    dopant_site_fractions, parse_fraction, dopant_host_mismatch,
    reconstruct_formula_from_fields,
)
from unit_convert import CONVERTERS

warnings.filterwarnings("ignore")

ML_RUN_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = ML_RUN_DIR / "outputs" / "datasets"
PROPERTIES = ["seebeck_coefficient", "electrical_conductivity", "thermal_conductivity", "power_factor", "ZT"]

EP = ElementProperty.from_preset("magpie")
VO = ValenceOrbital()
MAGPIE_LABELS = EP.feature_labels()
VO_LABELS = VO.feature_labels()

CUSTOM_LABELS = [
    "mixing_entropy", "ionic_character", "dopant_concentration",
    "num_dopants", "is_codoped", "frac_Ag", "frac_Sb", "frac_Te", "temperature_K",
    "dopant_host_charge_mismatch", "dopant_host_radius_mismatch",
]
ALL_FEATURE_LABELS = MAGPIE_LABELS + VO_LABELS + CUSTOM_LABELS


def featurize_row(row) -> dict:
    comp = formula_to_composition(row["full_formula"])
    reconstructed = False
    if comp is None:
        comp = reconstruct_formula_from_fields(
            row.get("base_composition"), row.get("primary_dopant_element"),
            row.get("primary_dopant_pct"), row.get("primary_dopant_site"),
            row.get("codopant_elements"),
        )
        reconstructed = comp is not None
    if comp is None:
        return None
    try:
        magpie_vals = EP.featurize(comp)
        vo_vals = VO.featurize(comp)
    except Exception:
        return None

    feats = dict(zip(MAGPIE_LABELS, magpie_vals))
    feats.update(dict(zip(VO_LABELS, vo_vals)))
    feats["mixing_entropy"] = mixing_entropy(comp)
    feats["ionic_character"] = ionic_character(comp)
    feats["dopant_concentration"] = parse_fraction(row.get("primary_dopant_pct"))
    feats["num_dopants"] = float(row.get("num_dopants") or 0)
    feats["is_codoped"] = 1.0 if str(row.get("is_codoped")).strip().lower() == "yes" else 0.0
    feats.update(dopant_site_fractions(comp))
    feats["temperature_K"] = row.get("temperature_K")
    charge_mm, radius_mm = dopant_host_mismatch(row.get("primary_dopant_element"), row.get("primary_dopant_site"))
    feats["dopant_host_charge_mismatch"] = charge_mm
    feats["dopant_host_radius_mismatch"] = radius_mm
    feats["_formula_reconstructed"] = reconstructed
    return feats


def main():
    summary = []
    for prop in PROPERTIES:
        raw_path = DATASETS_DIR / f"{prop}_raw.csv"
        df = pd.read_csv(raw_path, dtype=str)
        df["temperature_K"] = pd.to_numeric(df["temperature_K"], errors="coerce")
        df["target_value"] = pd.to_numeric(df["target_value"], errors="coerce")

        feature_rows = []
        parse_failures = 0
        for _, row in df.iterrows():
            feats = featurize_row(row)
            if feats is None:
                parse_failures += 1
                continue
            feature_rows.append({**row.to_dict(), **feats})

        fdf = pd.DataFrame(feature_rows)
        # keep rows with a usable temperature (impute missing T with the
        # property's own median rather than dropping -- composition features
        # are still valid without it, and T is one feature among ~140)
        if fdf["temperature_K"].isna().any():
            med_T = fdf["temperature_K"].median()
            fdf["temperature_K"] = fdf["temperature_K"].fillna(med_T)

        meta_cols = [c for c in df.columns if c not in ALL_FEATURE_LABELS] + ["_formula_reconstructed"]
        ordered_cols = meta_cols + [c for c in ALL_FEATURE_LABELS if c in fdf.columns] + ["target_value"]
        # target_value is already in meta_cols (from df) -- drop the duplicate, keep at end
        ordered_cols = [c for i, c in enumerate(ordered_cols) if c not in ordered_cols[:i]]
        fdf = fdf[[c for c in ordered_cols if c in fdf.columns]]

        out_path = DATASETS_DIR / f"{prop}_featurized.csv"
        fdf.to_csv(out_path, index=False)

        n_features = len([c for c in ALL_FEATURE_LABELS if c in fdf.columns])
        n_reconstructed = int(fdf["_formula_reconstructed"].sum()) if "_formula_reconstructed" in fdf.columns else 0
        summary.append({
            "property": prop, "input_rows": len(df), "parse_failures": parse_failures,
            "output_rows": len(fdf), "n_reconstructed_from_fields": n_reconstructed, "n_features": n_features,
        })
        print(f"{prop:24s}: {len(df):4d} rows -> {parse_failures:3d} unparseable -> {len(fdf):4d} featurized rows "
              f"({n_reconstructed} reconstructed from dopant/site/ratio fields), {n_features} features")

    pd.DataFrame(summary).to_csv(DATASETS_DIR / "featurize_summary.csv", index=False)


if __name__ == "__main__":
    main()
