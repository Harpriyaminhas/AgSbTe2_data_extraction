# AgSbTe2 Thermoelectric Dataset Viewer

Interactive dashboard for a dataset of doping/thermoelectric-property data
extracted from 173 journal-article PDFs on AgSbTe2-family thermoelectrics
(Seebeck coefficient, electrical conductivity, thermal conductivity, lattice
thermal conductivity, power factor, ZT), with dopant/co-dopant/doping-site
and DOI provenance for every data point.

This repo contains only the **extracted dataset and the viewer app** — not
the source PDFs (copyrighted journal articles) or the extraction pipeline
itself, which live in a private working folder.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. "New app" → select this repo → branch `main` → main file `app.py`.
3. Deploy. No secrets/API key needed — the app only reads the CSV files
   already committed in `output/`.

## Data

- `output/AgSbTe2_Thermoelectric_Extracted_Data_long.csv` — one row per
  (composition, property, temperature) data point.
- `output/logs/run_log.csv` — per-source-PDF extraction status.

Units and values are kept exactly as reported in each source paper (not
converted/normalized) — always check the `unit` column before comparing
rows. Rows with `source_type = figure` are visually estimated from a plot,
not read from exact text/table values.

This is a snapshot of the extraction pipeline's output as of the last
`git push` — see the main project for the up-to-date/regenerated data.
