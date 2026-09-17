"""
For every top candidate found by the inverse-design ZT screen (best single
dopant, best novel dopant, best co-doped pair, and the top-20 single
dopant/site list), predict ALL FIVE thermoelectric properties (not just ZT)
using each property's own best model -- giving a full optimum-property
table per candidate, at that candidate's own best temperature.

Usage: python3 src/predict_all_properties_for_candidates.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from macos_omp_fix import ensure_xgboost_can_load
ensure_xgboost_can_load()

import pandas as pd

from inverse_design import (
    load_best_model, build_doped_formula, featurize_hypothetical, predict, TEMPERATURES,
)

ML_RUN_DIR = Path(__file__).resolve().parent.parent
PRED_DIR = ML_RUN_DIR / "outputs" / "predictions"
PROPERTIES = ["seebeck_coefficient", "electrical_conductivity", "thermal_conductivity", "power_factor", "ZT"]
PROPERTY_UNITS = {
    "seebeck_coefficient": "uV/K", "electrical_conductivity": "S/cm",
    "thermal_conductivity": "W/m.K", "power_factor": "uW/cm.K2", "ZT": "dimensionless",
}


def predict_all_properties(dopants: list, site_for_mismatch: str, ratio_for_conc: float,
                            temperature_K: float, num_dopants: int, is_codoped: int, formula: str = None):
    """dopants: list of (element, site, ratio) -- for single-site doping all
    share one site; for multi-site co-doping, formula must be pre-built."""
    if formula is None:
        site = dopants[0][1]
        formula = build_doped_formula(site, [(el, r) for el, _, r in dopants])
    primary_el, primary_site = dopants[0][0], site_for_mismatch
    result = {}
    for prop in PROPERTIES:
        model, features, log_target, model_name = load_best_model(prop)
        feats = featurize_hypothetical(formula, primary_el, primary_site, temperature_K, num_dopants, is_codoped)
        if feats is None:
            result[prop] = None
            continue
        pred = predict(model, features, log_target, feats, ratio_for_conc)
        result[prop] = pred
        result[f"{prop}_model"] = model_name
    return result


def main():
    rows = []

    summary = pd.read_csv(PRED_DIR / "inverse_design_answer_summary.csv", index_col=0)["value"]

    # 1. best single dopant
    el, site, ratio, T = summary["best_single_dopant_element"], summary["best_single_dopant_site"], \
        float(summary["best_single_dopant_ratio"]), float(summary["best_single_dopant_temperature_K"])
    preds = predict_all_properties([(el, site, ratio)], site, ratio, T, 1, 0)
    rows.append({"candidate": "Best single dopant", "dopant_1": el, "site_1": site, "ratio_1": ratio,
                 "dopant_2": "", "site_2": "", "ratio_2": "", "temperature_K": T, **preds})

    # 2. best novel dopant
    el, site, ratio, T = summary["best_novel_dopant_element"], summary["best_novel_dopant_site"], \
        float(summary["best_novel_dopant_ratio"]), TEMPERATURES[3]
    best_row = pd.read_csv(PRED_DIR / "single_dopant_screen_full.csv")
    nov = best_row[(best_row.dopant_element == el) & (best_row.doped_site == site) & (best_row.dopant_ratio == ratio)]
    T = float(nov.loc[nov["predicted_ZT"].idxmax(), "temperature_K"]) if len(nov) else T
    preds = predict_all_properties([(el, site, ratio)], site, ratio, T, 1, 0)
    rows.append({"candidate": "Best novel/unexplored dopant", "dopant_1": el, "site_1": site, "ratio_1": ratio,
                 "dopant_2": "", "site_2": "", "ratio_2": "", "temperature_K": T, **preds})

    # 3. best co-doped pair (if present)
    if "best_codopant_1" in summary.index:
        e1, s1, r1 = summary["best_codopant_1"], summary["best_codopant_1_site"], float(summary["best_codopant_1_ratio"])
        e2, s2, r2 = summary["best_codopant_2"], summary["best_codopant_2_site"], float(summary["best_codopant_2_ratio"])
        T = float(summary["best_codopant_temperature_K"])
        if s1 == s2:
            formula = build_doped_formula(s1, [(e1, r1 / 2), (e2, r2 / 2)])
        else:
            base = {"Ag": 1.0, "Sb": 1.0, "Te": 2.0}
            base[s1] *= (1 - r1)
            base[s2] *= (1 - r2)
            parts = [f"Ag{base['Ag']:.4f}", f"Sb{base['Sb']:.4f}", f"Te{base['Te']:.4f}"]
            amt1 = {"Ag": 1.0, "Sb": 1.0, "Te": 2.0}[s1] * r1
            amt2 = {"Ag": 1.0, "Sb": 1.0, "Te": 2.0}[s2] * r2
            parts += [f"{e1}{amt1:.4f}", f"{e2}{amt2:.4f}"]
            formula = "".join(parts)
        actual_r1 = r1 / 2 if s1 == s2 else r1
        actual_r2 = r2 / 2 if s1 == s2 else r2
        preds = predict_all_properties([(e1, s1, actual_r1)], s1, actual_r1 + actual_r2, T, 2, 1, formula=formula)
        rows.append({"candidate": "Best co-doped pair", "dopant_1": e1, "site_1": s1, "ratio_1": actual_r1,
                     "dopant_2": e2, "site_2": s2, "ratio_2": actual_r2, "temperature_K": T, **preds})

    # 4. top-20 single dopant/site combos (each at its own best ratio+temperature)
    top20 = pd.read_csv(PRED_DIR / "top20_single_dopant_site_combos.csv")
    for r in top20.itertuples():
        preds = predict_all_properties([(r.dopant_element, r.doped_site, r.dopant_ratio)],
                                        r.doped_site, r.dopant_ratio, r.temperature_K, 1, 0)
        rows.append({"candidate": f"Top-20 screen: {r.dopant_element}@{r.doped_site}",
                     "dopant_1": r.dopant_element, "site_1": r.doped_site, "ratio_1": r.dopant_ratio,
                     "dopant_2": "", "site_2": "", "ratio_2": "", "temperature_K": r.temperature_K, **preds})

    out = pd.DataFrame(rows)
    ordered_cols = ["candidate", "dopant_1", "site_1", "ratio_1", "dopant_2", "site_2", "ratio_2", "temperature_K"]
    for prop in PROPERTIES:
        ordered_cols += [prop, f"{prop}_model"]
    out = out[[c for c in ordered_cols if c in out.columns]]
    out = out.rename(columns={p: f"{p}_predicted ({PROPERTY_UNITS[p]})" for p in PROPERTIES})

    out_path = PRED_DIR / "optimum_predicted_TE_properties.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(out)} candidate rows, all 5 properties each)")
    print(out.head(3).to_string())


if __name__ == "__main__":
    main()
