"""
Builds a refined, wide-format CSV from the long-format extracted dataset:
one row per (paper, sample, temperature), with power factor, Seebeck
coefficient, electrical conductivity, and ZT as separate columns instead of
stacked rows.

Run any time after (re-)running extract_thermoelectric_data.py /
backfill_doi_and_rebuild.py:

    python3 build_refined_csv.py

Careful-handling notes (read before trusting a cell blindly):
- Property names are matched case-insensitively (the LLM occasionally
  returned "Electrical conductivity" instead of "electrical conductivity",
  etc.) but NOT unit-converted -- unit is kept alongside every value.
- Temperature is combined as "value unit" exactly as reported (K, degC, C
  all occur in the source data) -- rows with no reported temperature are
  bucketed as "not reported" rather than silently dropped or merged into
  the wrong temperature.
- If a paper reports the SAME property at the SAME temperature for the SAME
  sample more than once (e.g. once in body text and again read off a
  figure), text/table values are preferred over figure-digitized ones
  (figure reads are visual estimates); if more than one non-duplicate value
  still remains at the same priority level, ALL of them are kept, joined
  with " | ", rather than silently picking one and discarding data.
- Rows where NONE of the four target properties were found for that
  (sample, temperature) are dropped from this refined file (they still
  exist in the full long-format CSV, e.g. thermal-conductivity-only rows).
- `primary_dopant_ratio` is the doping level/percentage exactly as reported
  (e.g. "2 at%", "x=0.02") -- it is a ratio/fraction label, not a unit-less
  number.
"""
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
LONG_CSV = BASE_DIR / "output" / "AgSbTe2_Thermoelectric_Extracted_Data_long.csv"
REFINED_CSV = BASE_DIR / "output" / "AgSbTe2_Thermoelectric_Refined_Data.csv"

PROPERTY_MAP = {
    "power factor": "power_factor",
    "seebeck coefficient": "seebeck_coefficient",
    "electrical conductivity": "electrical_conductivity",
    "zt": "ZT",
}

GROUP_COLS = [
    "source_pdf", "doi", "base_composition", "sample_id", "full_formula",
    "num_dopants", "is_codoped", "primary_dopant_element", "primary_dopant_pct",
    "primary_dopant_site", "codopant_elements", "all_dopants_detail",
]

FINAL_COLUMNS = [
    "doi", "source_pdf", "base_composition", "sample_id", "full_formula",
    "num_dopants", "is_codoped", "primary_dopant_element", "primary_dopant_ratio",
    "primary_dopant_site", "codopant_elements", "all_dopants_detail",
    "power_factor", "seebeck_coefficient", "electrical_conductivity", "ZT",
    "temperature",
]


def source_priority(source_type: str) -> int:
    s = (source_type or "").lower()
    if "table" in s or "text" in s:
        return 0
    if "figure" in s:
        return 2
    return 1  # unknown/blank source_type


def normalize_for_dedup(s: str) -> str:
    """Collapse cosmetic differences (mu-sign spelling, unicode minus, caret
    exponents, spacing) so the SAME reported value isn't kept twice just
    because text-pass and figure-pass rendered its unit glyphs differently."""
    s = s.strip().lower()
    s = s.replace("µ", "u").replace("μ", "u")  # µ / μ -> u
    s = s.replace("−", "-")  # unicode minus -> hyphen
    s = s.replace("^", "")
    s = re.sub(r"\s+", "", s)  # ignore spacing differences entirely for comparison
    return s


def format_value(value, unit) -> str:
    value = "" if pd.isna(value) else str(value).strip()
    unit = "" if pd.isna(unit) else str(unit).strip()
    if not value:
        return ""
    return f"{value} {unit}".strip()


def format_temperature(t_value, t_unit) -> str:
    if pd.isna(t_value) or str(t_value).strip() == "":
        return "not reported"
    unit = "" if pd.isna(t_unit) else str(t_unit).strip()
    return f"{str(t_value).strip()} {unit}".strip()


def main():
    if not LONG_CSV.exists():
        print(f"ERROR: {LONG_CSV} not found -- run extract_thermoelectric_data.py first.")
        return

    df = pd.read_csv(LONG_CSV, dtype=str)
    df["property_key"] = df["property"].fillna("").str.strip().str.lower()
    df = df[df["property_key"].isin(PROPERTY_MAP.keys())].copy()
    if df.empty:
        print("No rows matched the target properties (power factor / Seebeck / electrical conductivity / ZT).")
        return

    df["temperature_str"] = [
        format_temperature(v, u) for v, u in zip(df["temperature_value"], df["temperature_unit"])
    ]
    df["value_str"] = [format_value(v, u) for v, u in zip(df["value"], df["unit"])]
    df["priority"] = df["source_type"].apply(source_priority)

    for col in GROUP_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    records = {}
    dropped_no_value = 0
    for _, row in df.iterrows():
        if not row["value_str"]:
            dropped_no_value += 1
            continue
        key = tuple(row[c] for c in GROUP_COLS) + (row["temperature_str"],)
        prop_col = PROPERTY_MAP[row["property_key"]]
        rec = records.setdefault(key, {})
        slot = rec.setdefault(prop_col, {})  # priority -> {normalized_key: original value_str}
        bucket = slot.setdefault(row["priority"], {})
        norm_key = normalize_for_dedup(row["value_str"])
        bucket.setdefault(norm_key, row["value_str"])

    rows_out = []
    for key, props in records.items():
        base = dict(zip(GROUP_COLS, key[:-1]))
        base["temperature"] = key[-1]
        for prop_col in PROPERTY_MAP.values():
            slot = props.get(prop_col)
            if not slot:
                base[prop_col] = ""
                continue
            best_priority = min(slot.keys())
            base[prop_col] = " | ".join(sorted(slot[best_priority].values()))
        rows_out.append(base)

    out = pd.DataFrame(rows_out)
    out = out.rename(columns={"primary_dopant_pct": "primary_dopant_ratio"})
    for col in FINAL_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out = out[FINAL_COLUMNS]

    # sort for readability: paper, then sample, then temperature (numeric where possible)
    def temp_sort_key(t):
        try:
            return float(str(t).split()[0])
        except (ValueError, IndexError):
            return float("inf")

    out["_tsort"] = out["temperature"].apply(temp_sort_key)
    out = out.sort_values(["source_pdf", "sample_id", "_tsort"]).drop(columns="_tsort")
    out = out.drop_duplicates()

    out.to_csv(REFINED_CSV, index=False)

    n_papers = out["source_pdf"].nunique()
    n_samples = out.drop_duplicates(subset=["source_pdf", "sample_id", "full_formula"]).shape[0]
    print(f"Refined CSV written: {REFINED_CSV}")
    print(f"  rows (sample x temperature): {len(out)}")
    print(f"  distinct papers            : {n_papers}")
    print(f"  distinct samples           : {n_samples}")
    print(f"  long-format rows with no numeric value (skipped): {dropped_no_value}")
    for col in ["power_factor", "seebeck_coefficient", "electrical_conductivity", "ZT"]:
        filled = (out[col] != "").sum()
        print(f"  {col:24s}: {filled} / {len(out)} rows populated")


if __name__ == "__main__":
    main()
