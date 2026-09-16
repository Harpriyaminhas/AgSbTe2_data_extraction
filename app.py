"""
AgSbTe2 Thermoelectric Doping Dataset -- viewer app.

Run from VS Code's integrated terminal (in this folder):
    streamlit run app.py

Reads output/AgSbTe2_Thermoelectric_Extracted_Data_long.csv (written
incrementally by extract_thermoelectric_data.py, so this works even while
that script is still running on the remaining PDFs).
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "output" / "AgSbTe2_Thermoelectric_Extracted_Data_long.csv"
RUN_LOG_PATH = BASE_DIR / "output" / "logs" / "run_log.csv"

st.set_page_config(page_title="AgSbTe2 Thermoelectric Dataset", layout="wide")

PROPERTY_UNITS_HINT = {
    "Seebeck coefficient": "µV/K",
    "electrical conductivity": "S/cm",
    "thermal conductivity": "W/m·K",
    "lattice thermal conductivity": "W/m·K",
    "power factor": "µW/cm·K² or mW/m·K²",
    "ZT": "dimensionless",
}


@st.cache_data(ttl=30)
def load_data():
    if not CSV_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(CSV_PATH, dtype=str)
    for col in ("value", "temperature_value"):
        if col in df.columns:
            df[col + "_num"] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=30)
def load_run_log():
    if not RUN_LOG_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(RUN_LOG_PATH)


st.title("AgSbTe2-family Thermoelectric Doping Dataset")
st.caption(
    "Extracted from 173 journal-article PDFs (text, tables, and figures) via an "
    "LLM structured-extraction pipeline. Every row is one (composition, property, "
    "temperature) data point, with dopant / co-dopant / doping-site provenance."
)

if st.button("Refresh data (extraction may still be running)"):
    st.cache_data.clear()

df = load_data()
run_log = load_run_log()

if df.empty:
    st.warning("No extracted data yet -- run `python3 extract_thermoelectric_data.py` first, or wait for it to finish a few PDFs.")
    st.stop()

# ---------------------------------------------------------------------------
# Top-line stats
# ---------------------------------------------------------------------------
n_pdfs = df["source_pdf"].nunique()
n_materials = df.drop_duplicates(subset=["source_pdf", "sample_id", "full_formula"]).shape[0]
n_datapoints = df[df["property"].notna() & (df["property"] != "")].shape[0]
n_codoped = df[df["is_codoped"] == "Yes"].drop_duplicates(subset=["source_pdf", "sample_id"]).shape[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Papers with extracted data", n_pdfs)
c2.metric("Distinct compositions/samples", n_materials)
c3.metric("Property data points", n_datapoints)
c4.metric("Co-doped samples", n_codoped)

if not run_log.empty:
    total_pdfs = len(run_log)
    done = run_log[run_log["status"].isin(["ok", "partial-error"])].shape[0]
    failed = run_log[run_log["status"] == "failed"].shape[0]
    st.progress(done / total_pdfs if total_pdfs else 0, text=f"Extraction progress: {done}/{total_pdfs} PDFs processed ({failed} failed)")

st.divider()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

properties = sorted([p for p in df["property"].dropna().unique() if p])
sel_properties = st.sidebar.multiselect("Property", properties, default=properties)

dopant_elements = sorted(set(
    list(df["primary_dopant_element"].dropna().unique())
    + [e.split(" ")[0] for s in df["codopant_elements"].dropna() for e in s.split(";") if e.strip()]
))
dopant_elements = [d for d in dopant_elements if d and d.lower() != "nan"]
sel_dopants = st.sidebar.multiselect("Dopant element (primary or co-dopant)", dopant_elements)

sites = sorted([s for s in df["primary_dopant_site"].dropna().unique() if s])
sel_sites = st.sidebar.multiselect("Doped site", sites)

codope_choice = st.sidebar.radio("Co-doping", ["All", "Single-dopant only", "Co-doped only"], index=0)

source_filter = st.sidebar.text_input("Source PDF contains...", "")

if "temperature_value_num" in df.columns and df["temperature_value_num"].notna().any():
    tmin, tmax = float(df["temperature_value_num"].min()), float(df["temperature_value_num"].max())
    if tmin < tmax:
        sel_temp = st.sidebar.slider("Temperature range (K)", tmin, tmax, (tmin, tmax))
    else:
        sel_temp = (tmin, tmax)
else:
    sel_temp = None

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------
fdf = df.copy()
if sel_properties:
    fdf = fdf[fdf["property"].isin(sel_properties) | fdf["property"].isna() | (fdf["property"] == "")]
if sel_dopants:
    mask = fdf["primary_dopant_element"].isin(sel_dopants)
    for d in sel_dopants:
        mask = mask | fdf["codopant_elements"].fillna("").str.contains(rf"\b{d}\b", regex=True)
    fdf = fdf[mask]
if sel_sites:
    fdf = fdf[fdf["primary_dopant_site"].isin(sel_sites)]
if codope_choice == "Single-dopant only":
    fdf = fdf[fdf["is_codoped"] == "No"]
elif codope_choice == "Co-doped only":
    fdf = fdf[fdf["is_codoped"] == "Yes"]
if source_filter:
    fdf = fdf[fdf["source_pdf"].str.contains(source_filter, case=False, na=False)]
if sel_temp:
    fdf = fdf[
        fdf["temperature_value_num"].isna()
        | fdf["temperature_value_num"].between(sel_temp[0], sel_temp[1])
    ]

st.subheader(f"Data table ({len(fdf)} rows)")
st.dataframe(
    fdf.drop(columns=["value_num", "temperature_value_num"], errors="ignore"),
    use_container_width=True,
    height=380,
)
st.download_button(
    "Download filtered data as CSV",
    fdf.to_csv(index=False).encode("utf-8"),
    file_name="AgSbTe2_filtered_extract.csv",
    mime="text/csv",
)

st.divider()

# ---------------------------------------------------------------------------
# Property vs temperature plot
# ---------------------------------------------------------------------------
st.subheader("Property vs. temperature")
plot_props = [p for p in sel_properties if p in fdf["property"].unique()]
if plot_props:
    chosen_prop = st.selectbox("Plot property", plot_props)
    pdata = fdf[
        (fdf["property"] == chosen_prop)
        & fdf["value_num"].notna()
        & fdf["temperature_value_num"].notna()
    ].copy()
    if pdata.empty:
        st.info("No numeric (value, temperature) pairs for this property under the current filters.")
    else:
        pdata["series"] = (
            pdata["source_pdf"].str.slice(0, 25) + " | " + pdata["sample_id"].fillna("").str.slice(0, 30)
        )
        fig = px.scatter(
            pdata,
            x="temperature_value_num",
            y="value_num",
            color="series",
            symbol="source_type",
            hover_data=["source_pdf", "doi", "sample_id", "full_formula", "all_dopants_detail", "unit", "source_type", "source_detail", "notes"],
            labels={"temperature_value_num": "Temperature (K, as reported)", "value_num": f"{chosen_prop} ({PROPERTY_UNITS_HINT.get(chosen_prop, 'as reported')})"},
        )
        fig.update_layout(legend=dict(font=dict(size=9)), height=550)
        fig.update_traces(marker=dict(size=9))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Marker shape = source type (text / table / figure). Units are as reported in each paper and are NOT converted, so compare carefully across series.")
else:
    st.info("Select at least one property in the sidebar to plot.")

st.divider()

# ---------------------------------------------------------------------------
# Dopant / co-doping overview
# ---------------------------------------------------------------------------
st.subheader("Dopant coverage")
col_a, col_b = st.columns(2)

with col_a:
    dopant_counts = (
        df[df["primary_dopant_element"].notna() & (df["primary_dopant_element"] != "")]
        .drop_duplicates(subset=["source_pdf", "sample_id"])["primary_dopant_element"]
        .value_counts()
        .reset_index()
    )
    dopant_counts.columns = ["dopant_element", "n_samples"]
    if not dopant_counts.empty:
        fig2 = px.bar(dopant_counts.head(25), x="dopant_element", y="n_samples", title="Samples per primary dopant element")
        st.plotly_chart(fig2, use_container_width=True)

with col_b:
    site_counts = (
        df[df["primary_dopant_site"].notna() & (df["primary_dopant_site"] != "")]
        .drop_duplicates(subset=["source_pdf", "sample_id"])["primary_dopant_site"]
        .value_counts()
        .reset_index()
    )
    site_counts.columns = ["doped_site", "n_samples"]
    if not site_counts.empty:
        fig3 = px.pie(site_counts, names="doped_site", values="n_samples", title="Samples by doped site")
        st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Run log / provenance
# ---------------------------------------------------------------------------
with st.expander("Per-PDF extraction status (run log)"):
    if run_log.empty:
        st.write("No run log yet.")
    else:
        st.dataframe(run_log, use_container_width=True, height=300)
