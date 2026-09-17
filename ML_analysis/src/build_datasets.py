"""
Step 1: build 5 raw per-property CSVs from the AgSbTe2 long-format extracted
dataset. One row per (paper, sample, temperature) data point for that
property, keeping every composition/dopant column needed for featurization.

Usage: python3 src/build_datasets.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from numeric_parse import parse_numeric_value
from composition_utils import parse_temperature_to_kelvin
from unit_convert import CONVERTERS, CANONICAL_UNIT, PLAUSIBLE_RANGE

ML_RUN_DIR = Path(__file__).resolve().parent.parent
LONG_CSV = ML_RUN_DIR.parent / "AgSbTe2_work" / "output" / "AgSbTe2_Thermoelectric_Extracted_Data_long.csv"
OUT_DIR = ML_RUN_DIR / "outputs" / "datasets"

# property_key (lowercased, matched against the 'property' column) -> output name
PROPERTY_GROUPS = {
    "seebeck_coefficient": {"seebeck coefficient"},
    "electrical_conductivity": {"electrical conductivity"},
    "thermal_conductivity": {"thermal conductivity", "total thermal conductivity"},
    "power_factor": {"power factor"},
    "ZT": {"zt"},
}

KEEP_COLS = [
    "source_pdf", "doi", "base_composition", "sample_id", "full_formula",
    "num_dopants", "is_codoped", "primary_dopant_element", "primary_dopant_pct",
    "primary_dopant_site", "codopant_elements", "all_dopants_detail",
    "source_type", "source_detail", "notes",
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(LONG_CSV, dtype=str)
    df["property_key"] = df["property"].fillna("").str.strip().str.lower()

    for col in KEEP_COLS:
        if col not in df.columns:
            df[col] = ""

    summary = []
    for prop_name, keys in PROPERTY_GROUPS.items():
        sub = df[df["property_key"].isin(keys)].copy()
        sub["target_value"] = sub["value"].apply(parse_numeric_value)
        sub["temperature_K"] = [
            parse_temperature_to_kelvin(v, u)
            for v, u in zip(sub["temperature_value"], sub["temperature_unit"])
        ]
        before = len(sub)
        sub = sub[sub["target_value"].notna() & sub["full_formula"].notna() & (sub["full_formula"] != "")]
        after_value = len(sub)

        converter = CONVERTERS[prop_name]
        sub["target_value"] = [
            converter(v, u) for v, u in zip(sub["target_value"], sub["unit"])
        ]
        sub["unit"] = CANONICAL_UNIT[prop_name]
        sub = sub[sub["target_value"].notna()]
        after_unit_convert = len(sub)

        lo, hi = PLAUSIBLE_RANGE[prop_name]
        implausible = sub[~sub["target_value"].between(lo, hi)]
        if len(implausible):
            implausible.to_csv(OUT_DIR / f"{prop_name}_dropped_implausible.csv", index=False)
        sub = sub[sub["target_value"].between(lo, hi)]
        after_plausibility = len(sub)

        sub = sub.drop_duplicates(
            subset=["source_pdf", "sample_id", "full_formula", "temperature_K", "target_value"]
        )
        after_dedup = len(sub)

        out_cols = KEEP_COLS + ["temperature_K", "target_value", "unit"]
        out_path = OUT_DIR / f"{prop_name}_raw.csv"
        sub[out_cols].to_csv(out_path, index=False)

        summary.append({
            "property": prop_name,
            "raw_rows_matched": before,
            "rows_with_numeric_value_and_formula": after_value,
            "rows_with_convertible_unit": after_unit_convert,
            "rows_dropped_implausible": after_unit_convert - after_plausibility,
            "rows_after_dedup": after_dedup,
            "output_file": str(out_path),
        })
        print(f"{prop_name:24s}: {before:5d} matched -> {after_value:5d} numeric+formula -> "
              f"{after_unit_convert:5d} unit-converted -> {after_plausibility:5d} plausible -> {after_dedup:5d} after dedup")

    pd.DataFrame(summary).to_csv(OUT_DIR / "build_summary.csv", index=False)
    print(f"\nSaved per-property raw datasets to: {OUT_DIR}")


if __name__ == "__main__":
    main()
