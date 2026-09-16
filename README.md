# AgSbTe2 Thermoelectric Dataset Extraction + Viewer

Pipeline and dataset for doping/thermoelectric-property data extracted from
173 journal-article PDFs on AgSbTe2-family thermoelectrics (Seebeck
coefficient, electrical conductivity, thermal conductivity, lattice thermal
conductivity, power factor, ZT), with dopant/co-dopant/doping-site and DOI
provenance for every data point.

**What's in this repo, and what's deliberately NOT:**
- Included: the extraction pipeline code, the Streamlit viewer app, the
  extracted dataset (CSV/Excel), and per-paper raw extraction JSON.
- **Not included**: the 173 source PDFs, or rendered page images of them.
  Those are copyrighted journal articles — this repo only ever contains
  *extracted facts* (compositions, dopants, property values), never the
  articles' own text/figures/pages. If you're re-running the pipeline
  yourself, supply your own `PDFs/` folder locally (see below); it's
  git-ignored here on purpose.

## Viewing the data (no setup needed for this part)

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deploy the viewer Community Cloud

https://agsbte2.streamlit.app/

## Re-running the extraction pipeline yourself

This needs your own Gemini API key and your own copy of the source PDFs
(not included here — see above).

```bash
pip install -r requirements-pipeline.txt
mkdir PDFs                      # put your own source PDFs in here
echo "GEMINI_API_KEY=your_key_here" > .env   # never commit this file
python3 extract_thermoelectric_data.py
python3 backfill_doi_and_rebuild.py   # adds/repairs the doi column
python3 build_refined_csv.py          # wide-format PF/Seebeck/σ/ZT-vs-T table
```

`extract_thermoelectric_data.py` flags: `--limit N`, `--no-figures`,
`--workers N`, `--redo "substring"`, `--redo-failed` (see the script's
docstring/argparse help for details). Each PDF's result is cached in
`output/raw_json/<name>.json`, so re-running is resumable and safe to
interrupt.

**Note on API quota:** free-tier Gemini keys are rate-limited (and appear to
have a daily cap too, not just per-minute) — the pipeline paces itself under
the documented per-minute limit automatically, but a stalled run after many
requests may just mean the daily quota needs to reset.

## Data files

- `output/AgSbTe2_Thermoelectric_Extracted_Data_long.csv` — one row per
  (composition, property, temperature) data point. Also mirrored in
  `output/AgSbTe2_Thermoelectric_Extracted_Data.xlsx` (+ Materials_Summary
  and Run_Log sheets).
- `output/AgSbTe2_Thermoelectric_Refined_Data.csv` — wide format: one row
  per (paper, sample, temperature), with `power_factor`,
  `seebeck_coefficient`, `electrical_conductivity`, `ZT` as columns.
- `output/raw_json/<pdf_name>.json` — the raw structured-extraction result
  per paper (audit trail / pipeline cache).
- `output/logs/run_log.csv` — per-paper extraction status.

Units and values are kept exactly as reported in each source paper (not
converted/normalized) — always check the `unit` column before comparing
rows. Rows with `source_type = figure` are visually estimated from a plot,
not read from exact text/table values.

This is a snapshot as of the last `git push`.
