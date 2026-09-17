"""
Step 6: inverse design -- use the trained best ZT model (and, for context,
the best power-factor/Seebeck/conductivity models) to screen dopant
element x doped-site x doping-ratio x temperature combinations for
AgSbTe2, answering:
  1. Best dopant element
  2. Single-doped vs co-doped, and which site(s)
  3. Optimum dopant ratio/percentage
  4. Whether an element NEVER seen in the training literature (not just the
     ~40 elements already reported as AgSbTe2 dopants) is predicted to do
     even better -- i.e. genuine exploration, not interpolation only.

This is only as reliable as the training data lets it be (a few hundred
literature points per property) -- report predictions as *model-guided
hypotheses to test experimentally*, not guaranteed outcomes.

Usage: python3 src/inverse_design.py
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
from pymatgen.core.periodic_table import Element

from composition_utils import (
    formula_to_composition, mixing_entropy, ionic_character,
    dopant_site_fractions, dopant_host_mismatch,
)
from make_features import EP, VO, MAGPIE_LABELS, VO_LABELS

warnings.filterwarnings("ignore")

ML_RUN_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ML_RUN_DIR / "outputs" / "models"
DATASETS_DIR = ML_RUN_DIR / "outputs" / "datasets"
PRED_DIR = ML_RUN_DIR / "outputs" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

RATIOS = np.round(np.arange(0.01, 0.21, 0.01), 2)
TEMPERATURES = [300, 373, 473, 573, 673, 773]
SITES = ["Ag", "Sb", "Te"]

EXCLUDE_ELEMENTS = {
    "Ag", "Sb", "Te",  # host elements, not dopants
    "He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og",  # noble gases
    "Tc", "Pm",  # no stable isotopes / not realistic dopants
    "Po", "At", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",  # radioactive/actinides, impractical
}
CANDIDATE_ELEMENTS = [
    el.symbol for el in Element
    if el.symbol not in EXCLUDE_ELEMENTS and el.Z <= 83  # up to Bi, stable/practical range
]


def load_best_model(prop: str):
    prop_dir = MODELS_DIR / prop
    best_name = (prop_dir / "best_model.txt").read_text().strip()
    with open(prop_dir / f"best_model_{best_name}.pkl", "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["features"], payload["log_target"], best_name


def build_doped_formula(site: str, dopants: list):
    """dopants: list of (element, ratio) substituting the given site.
    Returns a formula string for (AgSbTe2) with those substitutions."""
    base = {"Ag": 1.0, "Sb": 1.0, "Te": 2.0}
    total_ratio = sum(r for _, r in dopants)
    base[site] = base[site] * (1 - total_ratio)
    parts = [f"Ag{base['Ag']:.4f}", f"Sb{base['Sb']:.4f}", f"Te{base['Te']:.4f}"]
    for el, r in dopants:
        amt = {"Ag": 1.0, "Sb": 1.0, "Te": 2.0}[site] * r
        parts.append(f"{el}{amt:.4f}")
    return "".join(parts)


def featurize_hypothetical(formula: str, primary_element: str, primary_site: str, temperature_K: float,
                            num_dopants: int, is_codoped: int):
    comp = formula_to_composition(formula)
    if comp is None:
        return None
    try:
        magpie_vals = dict(zip(MAGPIE_LABELS, EP.featurize(comp)))
        vo_vals = dict(zip(VO_LABELS, VO.featurize(comp)))
    except Exception:
        return None
    feats = {**magpie_vals, **vo_vals}
    feats["mixing_entropy"] = mixing_entropy(comp)
    feats["ionic_character"] = ionic_character(comp)
    feats["dopant_concentration"] = np.nan  # filled per-call by caller if needed
    feats["num_dopants"] = float(num_dopants)
    feats["is_codoped"] = float(is_codoped)
    feats.update(dopant_site_fractions(comp))
    feats["temperature_K"] = temperature_K
    charge_mm, radius_mm = dopant_host_mismatch(primary_element, primary_site)
    feats["dopant_host_charge_mismatch"] = charge_mm
    feats["dopant_host_radius_mismatch"] = radius_mm
    return feats


def predict(model, features, log_target, feats: dict, ratio_for_conc: float):
    feats = dict(feats)
    feats["dopant_concentration"] = ratio_for_conc
    row = pd.DataFrame([{k: feats.get(k, np.nan) for k in features}])
    if row.isna().any(axis=None):
        return None
    pred = model.predict(row)[0]
    return 10 ** pred if log_target else pred


def main():
    zt_model, zt_features, zt_log, zt_name = load_best_model("ZT")
    print(f"Using best ZT model: {zt_name} ({len(zt_features)} features)")

    known_dopants = set(pd.read_csv(DATASETS_DIR / "ZT_raw.csv")["primary_dopant_element"].dropna().unique())
    known_dopants = {e for e in known_dopants if e in CANDIDATE_ELEMENTS}

    # ---- 1-3: single-dopant screen over element x site x ratio x temperature ----
    rows = []
    for el in CANDIDATE_ELEMENTS:
        for site in SITES:
            for ratio in RATIOS:
                formula = build_doped_formula(site, [(el, ratio)])
                feats = featurize_hypothetical(formula, el, site, TEMPERATURES[0], 1, 0)
                if feats is None:
                    continue
                for T in TEMPERATURES:
                    feats["temperature_K"] = T
                    pred_zt = predict(zt_model, zt_features, zt_log, feats, ratio)
                    if pred_zt is None:
                        continue
                    rows.append({
                        "dopant_element": el, "doped_site": site, "dopant_ratio": ratio,
                        "temperature_K": T, "predicted_ZT": pred_zt,
                        "seen_in_training_data": el in known_dopants,
                        "formula": formula,
                    })
    screen_df = pd.DataFrame(rows)
    screen_df.to_csv(PRED_DIR / "single_dopant_screen_full.csv", index=False)

    best_row = screen_df.loc[screen_df["predicted_ZT"].idxmax()]
    print("\n=== BEST SINGLE-DOPANT PREDICTION ===")
    print(best_row.to_string())

    top20 = screen_df.sort_values("predicted_ZT", ascending=False).drop_duplicates(
        subset=["dopant_element", "doped_site"]
    ).head(20)
    top20.to_csv(PRED_DIR / "top20_single_dopant_site_combos.csv", index=False)

    novel_only = screen_df[~screen_df["seen_in_training_data"]]
    best_novel = novel_only.loc[novel_only["predicted_ZT"].idxmax()]
    print("\n=== BEST NOVEL (never reported as an AgSbTe2 dopant in our corpus) ===")
    print(best_novel.to_string())

    # ---- 4: co-doping screen among the top single-dopant candidates ----
    top_candidates = top20.head(8)[["dopant_element", "doped_site", "dopant_ratio"]].to_dict("records")
    co_rows = []
    for i in range(len(top_candidates)):
        for j in range(i + 1, len(top_candidates)):
            a, b = top_candidates[i], top_candidates[j]
            if a["dopant_element"] == b["dopant_element"]:
                continue
            # co-dope on whichever single site each prefers; if same site, split it
            if a["doped_site"] == b["doped_site"]:
                site = a["doped_site"]
                ra, rb = a["dopant_ratio"] / 2, b["dopant_ratio"] / 2
                dopants = [(a["dopant_element"], ra), (b["dopant_element"], rb)]
                formula = build_doped_formula(site, dopants)
                primary_site = site
            else:
                # two different sites: build sequentially (Ag then Sb then Te ordering)
                base = {"Ag": 1.0, "Sb": 1.0, "Te": 2.0}
                base[a["doped_site"]] *= (1 - a["dopant_ratio"])
                base[b["doped_site"]] *= (1 - b["dopant_ratio"])
                parts = [f"Ag{base['Ag']:.4f}", f"Sb{base['Sb']:.4f}", f"Te{base['Te']:.4f}"]
                amt_a = {"Ag": 1.0, "Sb": 1.0, "Te": 2.0}[a["doped_site"]] * a["dopant_ratio"]
                amt_b = {"Ag": 1.0, "Sb": 1.0, "Te": 2.0}[b["doped_site"]] * b["dopant_ratio"]
                parts.append(f"{a['dopant_element']}{amt_a:.4f}")
                parts.append(f"{b['dopant_element']}{amt_b:.4f}")
                formula = "".join(parts)
                primary_site = a["doped_site"]

            feats = featurize_hypothetical(formula, a["dopant_element"], primary_site, TEMPERATURES[0], 2, 1)
            if feats is None:
                continue
            for T in TEMPERATURES:
                feats["temperature_K"] = T
                pred_zt = predict(zt_model, zt_features, zt_log, feats, (a["dopant_ratio"] + b["dopant_ratio"]) / 2)
                if pred_zt is None:
                    continue
                co_rows.append({
                    "dopant_1": a["dopant_element"], "site_1": a["doped_site"], "ratio_1": a["dopant_ratio"],
                    "dopant_2": b["dopant_element"], "site_2": b["doped_site"], "ratio_2": b["dopant_ratio"],
                    "temperature_K": T, "predicted_ZT": pred_zt, "formula": formula,
                })
    co_df = pd.DataFrame(co_rows)
    co_df.to_csv(PRED_DIR / "codopant_screen_full.csv", index=False)
    best_co = co_df.loc[co_df["predicted_ZT"].idxmax()] if len(co_df) else None

    print("\n=== BEST CO-DOPED PREDICTION ===")
    if best_co is not None:
        print(best_co.to_string())

    # ---- final answer summary ----
    single_best_zt = best_row["predicted_ZT"]
    co_best_zt = best_co["predicted_ZT"] if best_co is not None else -np.inf
    recommendation = "co-doped" if co_best_zt > single_best_zt else "single-doped"

    summary = {
        "recommendation_single_vs_codoped": recommendation,
        "best_single_dopant_element": best_row["dopant_element"],
        "best_single_dopant_site": best_row["doped_site"],
        "best_single_dopant_ratio": best_row["dopant_ratio"],
        "best_single_dopant_temperature_K": best_row["temperature_K"],
        "best_single_dopant_predicted_ZT": round(float(single_best_zt), 3),
        "best_single_seen_in_training_data": bool(best_row["seen_in_training_data"]),
        "best_novel_dopant_element": best_novel["dopant_element"],
        "best_novel_dopant_site": best_novel["doped_site"],
        "best_novel_dopant_ratio": best_novel["dopant_ratio"],
        "best_novel_predicted_ZT": round(float(best_novel["predicted_ZT"]), 3),
    }
    if best_co is not None:
        summary.update({
            "best_codopant_1": best_co["dopant_1"], "best_codopant_1_site": best_co["site_1"],
            "best_codopant_1_ratio": best_co["ratio_1"],
            "best_codopant_2": best_co["dopant_2"], "best_codopant_2_site": best_co["site_2"],
            "best_codopant_2_ratio": best_co["ratio_2"],
            "best_codopant_temperature_K": best_co["temperature_K"],
            "best_codopant_predicted_ZT": round(float(co_best_zt), 3),
        })

    pd.Series(summary).to_csv(PRED_DIR / "inverse_design_answer_summary.csv", header=["value"])
    print("\n=== FINAL RECOMMENDATION ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
