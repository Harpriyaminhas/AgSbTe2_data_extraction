"""
Adds a `doi` field to every cached per-PDF raw_json record (regex lookup on
the PDF's own first pages -- no LLM calls, so this is fast/free and safe to
run anytime, including while extract_thermoelectric_data.py is still running
on the remaining PDFs) and rebuilds the long-format CSV + Excel workbook from
all cached raw_json records currently on disk.

Run again any time (idempotent) -- e.g. once more after the main extraction
run finishes, to get the final DOI-complete dataset.

    python3 backfill_doi_and_rebuild.py
"""
import json

import pandas as pd

from extract_thermoelectric_data import (
    PDF_DIR,
    RAW_JSON_DIR,
    CSV_PATH,
    XLSX_PATH,
    RUN_LOG_PATH,
    LONG_COLUMNS,
    safe_name,
    extract_doi,
    flatten_one,
)


def main():
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    n_processed = 0
    n_doi_filled_now = 0
    n_doi_missing = 0
    all_rows = []
    run_log_rows = []

    for pdf_path in pdfs:
        name = safe_name(pdf_path.stem)
        raw_path = RAW_JSON_DIR / f"{name}.json"
        if not raw_path.exists():
            continue  # not processed by the main pipeline yet
        n_processed += 1

        try:
            record = json.loads(raw_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  ! could not read {raw_path.name}: {e}")
            continue

        if not record.get("doi"):
            doi = extract_doi(pdf_path)
            record["doi"] = doi
            raw_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            if doi:
                n_doi_filled_now += 1
            else:
                n_doi_missing += 1
        elif record.get("doi"):
            pass  # already had one from a previous backfill run
        else:
            n_doi_missing += 1

        all_rows.extend(flatten_one(record))
        run_log_rows.append({
            "source_pdf": record.get("source_pdf", pdf_path.name),
            "doi": record.get("doi", ""),
            "status": record.get("status", "unknown"),
            "n_pages": record.get("n_pages"),
            "n_materials": len(record.get("materials", [])),
            "extraction_notes": (record.get("extraction_notes", "") or "")[:500],
        })

    if not all_rows:
        print("No processed PDFs found yet (output/raw_json is empty) -- nothing to rebuild.")
        return

    df = pd.DataFrame(all_rows, columns=LONG_COLUMNS).drop_duplicates()
    df.to_csv(CSV_PATH, index=False)

    run_log_df = pd.DataFrame(run_log_rows)
    run_log_df.to_csv(RUN_LOG_PATH, index=False)

    materials_summary = (
        df[df["sample_id"] != ""]
        .drop_duplicates(subset=["source_pdf", "sample_id", "full_formula"])
        [["source_pdf", "doi", "base_composition", "sample_id", "full_formula", "num_dopants",
          "is_codoped", "primary_dopant_element", "primary_dopant_pct", "primary_dopant_site",
          "codopant_elements", "all_dopants_detail"]]
    )

    with pd.ExcelWriter(XLSX_PATH, engine="openpyxl") as writer_x:
        df.to_excel(writer_x, sheet_name="Extracted_Data", index=False)
        materials_summary.to_excel(writer_x, sheet_name="Materials_Summary", index=False)
        run_log_df.to_excel(writer_x, sheet_name="Run_Log", index=False)

    print(f"PDFs processed so far : {n_processed} / {len(pdfs)}")
    print(f"DOIs filled in this pass: {n_doi_filled_now}")
    print(f"DOIs still not found   : {n_doi_missing} (not printed on the first pages / not a regular DOI-bearing PDF)")
    print(f"Rebuilt: {CSV_PATH} ({len(df)} rows), {XLSX_PATH}")


if __name__ == "__main__":
    main()
