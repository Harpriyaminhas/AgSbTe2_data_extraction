"""
AgSbTe2 ML results viewer -- model performance, SHAP interpretability, and
inverse-design predictions.

Run locally:
    streamlit run ml_app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"
PRED_DIR = OUT_DIR / "predictions"

PROPERTIES = ["seebeck_coefficient", "electrical_conductivity", "thermal_conductivity", "power_factor", "ZT"]
PROPERTY_LABELS = {
    "seebeck_coefficient": "Seebeck coefficient", "electrical_conductivity": "Electrical conductivity",
    "thermal_conductivity": "Thermal conductivity", "power_factor": "Power factor", "ZT": "ZT",
}

st.set_page_config(page_title="AgSbTe2 ML Results", layout="wide")
st.title("AgSbTe2-family Thermoelectric ML Results")
st.caption(
    "Composition-based ML models (Decision Tree, Random Forest, Gradient Boosting, XGBoost, CatBoost) trained on "
    "the LLM-extracted literature dataset, with SHAP interpretability and a dopant/site/ratio/temperature "
    "inverse-design screen. See the full Word report for methodology details."
)

report_path = BASE_DIR / "AgSbTe2_ML_Report.docx"
if report_path.exists():
    with open(report_path, "rb") as f:
        st.download_button("Download full Word report", f, file_name="AgSbTe2_ML_Report.docx")

tabs = st.tabs(["Model performance", "Per-property details", "Inverse design"])

# ---------------------------------------------------------------------------
# Tab 1: model performance
# ---------------------------------------------------------------------------
with tabs[0]:
    metrics_path = MODELS_DIR / "all_model_metrics.csv"
    best_path = MODELS_DIR / "best_model_summary.csv"
    if metrics_path.exists():
        st.subheader("Best model per property")
        best_df = pd.read_csv(best_path)
        best_df["property"] = best_df["property"].map(lambda p: PROPERTY_LABELS.get(p, p))
        st.dataframe(best_df, use_container_width=True)

        comp_fig = FIG_DIR / "model_comparison_all_properties.png"
        if comp_fig.exists():
            st.image(str(comp_fig), use_container_width=True)

        with st.expander("All models, all properties (full metrics + hyperparameters)"):
            full_df = pd.read_csv(metrics_path)
            full_df["property"] = full_df["property"].map(lambda p: PROPERTY_LABELS.get(p, p))
            st.dataframe(full_df, use_container_width=True)
    else:
        st.warning("No model metrics found -- run the training pipeline first (see README.md).")

# ---------------------------------------------------------------------------
# Tab 2: per-property SHAP + parity
# ---------------------------------------------------------------------------
with tabs[1]:
    prop_choice = st.selectbox("Property", PROPERTIES, format_func=lambda p: PROPERTY_LABELS.get(p, p))
    prop_dir = MODELS_DIR / prop_choice
    fig_dir = FIG_DIR / prop_choice
    best_model_file = prop_dir / "best_model.txt"
    if best_model_file.exists():
        st.markdown(f"**Best model: {best_model_file.read_text().strip()}**")

    col1, col2 = st.columns(2)
    with col1:
        parity = fig_dir / f"parity_{prop_choice}.png"
        if parity.exists():
            st.image(str(parity), caption="Train/test parity", use_container_width=True)
    with col2:
        bar = fig_dir / f"shap_bar_{prop_choice}.png"
        if bar.exists():
            st.image(str(bar), caption="SHAP global importance", use_container_width=True)

    beeswarm = fig_dir / f"shap_beeswarm_{prop_choice}.png"
    if beeswarm.exists():
        st.image(str(beeswarm), caption="SHAP beeswarm", use_container_width=True)

    imp_csv = fig_dir / f"shap_importance_{prop_choice}.csv"
    if imp_csv.exists():
        with st.expander("Full SHAP importance ranking"):
            st.dataframe(pd.read_csv(imp_csv), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: inverse design
# ---------------------------------------------------------------------------
with tabs[2]:
    summary_path = PRED_DIR / "inverse_design_answer_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path, index_col=0)["value"]
        st.subheader("Answers to the four design questions")
        c1, c2, c3 = st.columns(3)
        c1.metric("Best single dopant", f"{summary['best_single_dopant_element']} @ {summary['best_single_dopant_site']}",
                   f"{float(summary['best_single_dopant_ratio'])*100:.0f}% @ {int(float(summary['best_single_dopant_temperature_K']))}K")
        c1.metric("Predicted ZT", f"{float(summary['best_single_dopant_predicted_ZT']):.2f}")
        if "best_codopant_1" in summary.index:
            c2.metric("Best co-doped pair", f"{summary['best_codopant_1']}+{summary['best_codopant_2']} @ {summary['best_codopant_1_site']}")
            c2.metric("Predicted ZT", f"{float(summary['best_codopant_predicted_ZT']):.2f}")
        c3.metric("Best novel dopant", f"{summary['best_novel_dopant_element']} @ {summary['best_novel_dopant_site']}",
                   f"{float(summary['best_novel_dopant_ratio'])*100:.0f}%")
        c3.metric("Predicted ZT", f"{float(summary['best_novel_predicted_ZT']):.2f}")

        st.info(
            "The single-dopant answer sits close to chemistry already well-represented in the training corpus "
            "(Cd-on-Sb doping, reported across 5 papers) -- read it as an internal consistency check, not "
            "external validation. The co-doped and novel-dopant answers are genuine extrapolations: real "
            "hypotheses to test experimentally, not validated results."
        )

        top_fig = FIG_DIR / "inverse_design_top_candidates.png"
        if top_fig.exists():
            st.image(str(top_fig), use_container_width=True)

    opt_path = PRED_DIR / "optimum_predicted_TE_properties.csv"
    if opt_path.exists():
        st.subheader("All 5 properties predicted for the top candidates")
        st.dataframe(pd.read_csv(opt_path), use_container_width=True)
        with open(opt_path, "rb") as f:
            st.download_button("Download full candidate predictions (CSV)", f, file_name="optimum_predicted_TE_properties.csv")

    top20_path = PRED_DIR / "top20_single_dopant_site_combos.csv"
    if top20_path.exists():
        with st.expander("Top 20 single-dopant/site screen (full table)"):
            st.dataframe(pd.read_csv(top20_path), use_container_width=True)
