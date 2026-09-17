# AgSbTe2 Thermoelectric ML Pipeline

Machine-learning pipeline that takes the extracted AgSbTe2-family dataset
from `../AgSbTe2_work/output/AgSbTe2_Thermoelectric_Extracted_Data_long.csv`
and, for five thermoelectric (TE) properties (Seebeck coefficient, electrical
conductivity, thermal conductivity, power factor, ZT), builds composition
descriptors, trains/benchmarks 5 regression models, runs SHAP interpretability
analysis, and screens dopant/site/ratio/temperature combinations to answer:
which dopant, single- or co-doped, what ratio, and whether an unexplored
dopant could do better for AgSbTe2.

**Main deliverable:** `AgSbTe2_ML_Report.docx` -- a full write-up (methods,
per-property model comparison + hyperparameters, SHAP figures, inverse-design
results) generated from everything in `outputs/`.

## Setup

```bash
pip install -r requirements.txt
```

**macOS + XGBoost:** the XGBoost wheel needs `libomp.dylib`, which normally
requires Homebrew. `src/macos_omp_fix.py` works around this by pointing
XGBoost at the copy scikit-learn already bundles -- no Homebrew/sudo needed.
It's called automatically at the top of every script that imports XGBoost.

## Pipeline (run in order)

```bash
python3 src/build_datasets.py          # 1. filter+unit-harmonize the 5 properties from the long CSV
python3 src/make_features.py           # 2. matminer/magpie + physics-informed descriptors per sample
python3 src/select_features.py         # 3. reduce to a minimal (>=10) feature set per property
python3 src/train_models.py            # 4. train/tune DTR, RandomForest, GBR, XGBoost, CatBoost
python3 src/shap_analysis.py           # 5. SHAP bar + beeswarm for each property's best model
python3 src/make_summary_figures.py    # 6. model-comparison bars + train/test parity plots
python3 src/inverse_design.py          # 7. dopant x site x ratio x temperature screen (best ZT model)
python3 src/predict_all_properties_for_candidates.py   # 8. all 5 properties for the top candidates
python3 src/make_inverse_design_figure.py              # 9. top-15 candidates figure
python3 src/build_word_report.py       # 10. assemble the final .docx from everything above
```

Each script reads the previous step's output from `outputs/` and can be
re-run independently once its inputs exist. Re-running step 1 after the
extraction pipeline produces new/updated papers picks up the larger dataset
automatically -- just re-run the whole chain in order.

## Feature engineering

- **matminer** `ElementProperty` (132 "magpie" elemental-property statistics:
  mean/avg-deviation/range/min/max/mode across 22 properties) + `ValenceOrbital`
  (8 features), computed on the parsed composition.
- **Custom descriptors** from `Thermoelectric_ML_Descriptors.docx` that
  matminer doesn't provide directly: mixing entropy, electronegativity-based
  ionic character, dopant concentration, co-doping flag, per-sublattice
  (Ag/Sb/Te) atomic fractions, temperature.
- **Cation-disorder descriptors** (`dopant_host_charge_mismatch`,
  `dopant_host_radius_mismatch`): dopant-vs-host-ion charge and Shannon
  ionic-radius mismatch, operationalizing the Ag+/Sb3+ site-disorder
  mechanism from Kanishka Biswas's AgSbTe2 work (Science 2021
  10.1126/science.abb3517; JACS 2023 10.1021/jacs.3c09643). Always
  force-included in the selected feature set regardless of automatic
  importance ranking, along with temperature.
- **Formula parsing fallback**: if a paper's reported formula string can't be
  parsed directly (templated `x=` notation, stray arithmetic, OCR artifacts),
  `composition_utils.reconstruct_formula_from_fields` rebuilds the
  composition from the already-extracted dopant element/site/ratio fields
  instead of dropping the row -- but only when the site is an unambiguous
  Ag/Sb/Te substitution, never guessed for "unclear"/"multiple sites".

## Modeling notes

- Electrical conductivity, power factor, and thermal conductivity are
  modeled in log10 space (they span multiple orders of magnitude / benefit
  from it given a modest sample count); predictions are back-transformed
  before any reported metric, so R2/RMSE in the report are in physical units.
- Electrical conductivity is additionally scoped to <=500 S/cm (drops ~4% of
  rows -- near-metallic composite/alloy samples too sparse to learn
  reliably and that dominated test error). Predictions above ~500 S/cm are
  outside this model's intended scope.
- Train/test split (80/20) and the internal cross-validation folds during
  hyperparameter search are **quantile-stratified on the target value**, not
  a plain random split -- necessary for heavy-tailed properties, where an
  unstratified split can strand nearly all high-magnitude samples on one
  side and produce a model that looks badly overfit (or falsely excellent)
  purely from an unlucky/lucky split.
- Hyperparameters are tuned per (property, model) with `RandomizedSearchCV`
  (50 iterations, 5-fold CV, R2 scoring).

## Current results (re-run after the extraction corpus reached 173/173 papers)

| Property | Best model | Test R2 | Test RMSE | Log-space test R2 |
|---|---|---|---|---|
| Seebeck coefficient | CatBoost | 0.77 | 48.2 uV/K | -- |
| Electrical conductivity (<=500 S/cm) | CatBoost | 0.30 | 79.8 S/cm | 0.93 |
| Thermal conductivity | CatBoost | 0.84 | 0.25 W/m.K | 0.64 |
| Power factor | CatBoost | 0.54 | 4.2 uW/cm.K2 | 0.92 |
| ZT | CatBoost | 0.83 | 0.27 | -- |

Full per-model metrics and every algorithm's tuned hyperparameters are in
`outputs/models/all_model_metrics.csv` and the Word report's Section 4/5.

## Inverse-design result

Screening dopant element (all elements up to Bi, excluding noble/radioactive
elements and the host Ag/Sb/Te) x site (Ag/Sb/Te) x ratio (1-20%) x
temperature (300-773 K) with the best ZT model:

- **Best single dopant:** Cd on Sb, 8%, 573 K -- predicted ZT = 2.67. This
  chemistry is heavily represented in training (5 papers, ratios 2/4/6%
  reported), so treat this as the model correctly recovering/mildly
  interpolating a known optimum, not a blind prediction.
- **Best co-doped pair:** Cd (4%) + Hg (2%) on Sb, 773 K -- predicted
  ZT = 2.60. A genuine extrapolation: both dopants are individually
  well-documented, but this combination has not been tested together.
- **Best novel/unexplored dopant:** Be on Te, 5%, 773 K -- predicted
  ZT = 2.28. Genuinely out-of-sample (Be appears zero times as an AgSbTe2
  dopant in the corpus). Be compounds are toxic -- a practical caveat before
  pursuing this experimentally, independent of the model's uncertainty.

See `outputs/predictions/optimum_predicted_TE_properties.csv` for all five
properties predicted for each top candidate, and
`outputs/predictions/top20_single_dopant_site_combos.csv` for the full
ranked list.

## Output layout

```
outputs/
├── datasets/<property>_{raw,featurized,selected_features}.csv
├── models/<property>/{train.csv, test.csv, model_*.pkl, best_model.txt}
├── models/all_model_metrics.csv, best_model_summary.csv
├── figures/<property>/{shap_bar,shap_beeswarm,parity}_<property>.png
├── figures/model_comparison_all_properties.png, inverse_design_top_candidates.png
└── predictions/{single_dopant_screen_full, codopant_screen_full,
                 top20_single_dopant_site_combos, optimum_predicted_TE_properties,
                 inverse_design_answer_summary}.csv
```

## Limitations

See the Word report's Section 8 for the full list -- in short: dataset size
is modest (~250-800 rows/property after de-duplication) for a ~10-12-feature
regression; figure-sourced data points are visual estimates; several
descriptors from the reference list need crystal-structure/DFT/Materials
Project data not available per-sample here; properties are modeled
independently (not under a joint PF = S2*sigma constraint); and inverse-design
predictions for genuinely novel chemistry are hypotheses to test
experimentally, not guaranteed outcomes.
