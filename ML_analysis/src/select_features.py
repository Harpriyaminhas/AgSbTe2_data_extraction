"""
Step 3: reduce the ~151-feature matminer+custom feature set down to a
minimum required feature set per property:
  1. drop near-constant features (variance threshold)
  2. drop highly-correlated features (|r| > 0.95, keep one per cluster --
     protected/forced features are never the one dropped)
  3. rank remaining features by Random Forest permutation importance, keep
     the smallest top-K (K >= MIN_K) that retains >=95% of the full-feature
     CV R^2, then force-include the Biswas-mechanism cation-disorder
     descriptors (dopant/host charge + ionic-radius mismatch) if not
     already present, since they are a deliberate physics-informed
     descriptor set we always want represented, not left to chance ranking.

Usage: python3 src/select_features.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_features import ALL_FEATURE_LABELS

warnings.filterwarnings("ignore")

ML_RUN_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = ML_RUN_DIR / "outputs" / "datasets"
PROPERTIES = ["seebeck_coefficient", "electrical_conductivity", "thermal_conductivity", "power_factor", "ZT"]

MIN_K = 10
MAX_K = 25
CORR_THRESHOLD = 0.95
RANDOM_STATE = 42

# Always keep these represented in the selected feature set, regardless of
# where the importance ranking happens to place them: temperature_K because
# every one of these properties is fundamentally temperature-dependent, and
# the two dopant/host mismatch descriptors because they operationalize the
# Biswas-group AgSbTe2 cation-disorder mechanism (e.g. Science 2021
# 10.1126/science.abb3517; Hg-doping JACS 2023 10.1021/jacs.3c09643).
FORCE_INCLUDE = ["temperature_K", "dopant_host_charge_mismatch", "dopant_host_radius_mismatch"]


def drop_correlated(X: pd.DataFrame, threshold=CORR_THRESHOLD, protect=()):
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = []
    for col in upper.columns:
        if col in protect:
            continue
        partners = upper.index[upper[col] > threshold].tolist()
        if partners:
            to_drop.append(col)
    return X.drop(columns=to_drop), to_drop


def main():
    summary = []
    for prop in PROPERTIES:
        path = DATASETS_DIR / f"{prop}_featurized.csv"
        df = pd.read_csv(path)
        feat_cols = [c for c in ALL_FEATURE_LABELS if c in df.columns]
        X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
        y = df["target_value"]
        mask = y.notna() & X.notna().all(axis=1)
        X, y = X[mask], y[mask]

        # 1. variance threshold
        variances = X.var()
        X = X[variances[variances > 1e-8].index]

        # 2. correlation pruning (never drop the forced Biswas-mechanism features)
        present_force = [c for c in FORCE_INCLUDE if c in X.columns]
        X, dropped_corr = drop_correlated(X, protect=present_force)

        # 3. importance ranking via Random Forest + permutation importance
        rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X, y)
        perm = permutation_importance(rf, X, y, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
        importance = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        full_score = cross_val_score(rf, X, y, cv=kf, scoring="r2").mean()

        min_k = min(MIN_K, len(importance))
        best_k, best_score, best_cols = len(X.columns), full_score, list(X.columns)
        for k in range(min_k, min(MAX_K, len(importance)) + 1):
            cols_k = importance.index[:k].tolist()
            score_k = cross_val_score(rf, X[cols_k], y, cv=kf, scoring="r2").mean()
            if score_k >= 0.95 * full_score or (full_score < 0 and score_k >= full_score - 0.02):
                best_k, best_score, best_cols = k, score_k, cols_k
                break
        else:
            best_k = min(MAX_K, len(importance))
            best_cols = importance.index[:best_k].tolist()
            best_score = cross_val_score(rf, X[best_cols], y, cv=kf, scoring="r2").mean()

        # force-include the Biswas cation-disorder descriptors even if their
        # own importance ranking fell outside the chosen top-K
        for c in present_force:
            if c not in best_cols:
                best_cols.append(c)
        best_k = len(best_cols)

        selected = pd.concat([X[best_cols], y.rename("target_value")], axis=1)
        out_path = DATASETS_DIR / f"{prop}_selected_features.csv"
        selected.to_csv(out_path, index=False)

        importance.to_csv(DATASETS_DIR / f"{prop}_feature_importance_full.csv", header=["importance"])

        summary.append({
            "property": prop, "n_features_before": len(feat_cols),
            "n_after_variance_corr": len(X.columns) if best_k == len(X.columns) else None,
            "n_dropped_correlated": len(dropped_corr),
            "n_selected": best_k, "cv_r2_full_features": round(full_score, 3),
            "cv_r2_selected_features": round(best_score, 3),
        })
        print(f"{prop:24s}: {len(feat_cols)} -> after var/corr {len(X.columns)} -> selected {best_k} "
              f"(CV R2 full={full_score:.3f}, selected={best_score:.3f})")
        print(f"    top features: {best_cols}")

    pd.DataFrame(summary).to_csv(DATASETS_DIR / "feature_selection_summary.csv", index=False)


if __name__ == "__main__":
    main()
