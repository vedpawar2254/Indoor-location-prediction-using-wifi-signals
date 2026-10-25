# Indoor Location Prediction Using Wi-Fi Signals (Project 9)

Predicting indoor location (building, floor, and floor-plan coordinates) from Wi-Fi signal strength fingerprints.

**Team T054** — Section E, BTech CSE (AI), Newton School of Technology, Rishihood University (Batch 2024–2028)  
Course project: Advanced Machine Learning (Phase 1) + Deep Learning (Phase 2), per the official CSAI Project Instructions.

> **Status**: Phase 1 foundation complete for building, floor, and coordinate prediction from the UCI UJIIndoorLoc Wi-Fi fingerprints. Raw data is downloaded locally and excluded from Git.

---

## Team

| Member | Work Order Role | Responsibilities |
| --- | --- | --- |
| **Aviral** | Member 1 — Data & Infrastructure Lead | Repository setup, UCI download, schema validation, training duplicate & constant-WAP cleanup, suspicious near-zero RSSI audit, USERID GroupKFold splits, official test-set designation, EDA notebook & data-quality report. |
| **Ved** | Member 2 — Features & Representation Lead | Shared +100 to -110 dBm convention, three RSSI representations, engineered/missingness features, fold-only scaling/PCA/variance filtering, and grouped-CV kNN comparison. |
| **Khuswant** | Member 3 — Classification Lead | Building & floor classifiers (Random Forest & XGBoost), nested USERID hyperparameter tuning, fold-fitted variance filtering, feature importance integration, and OOF confusion matrices. |
| **Prataya** | Member 4 — Coordinate Regression & Validation Lead | Coordinate regressors (kNN, Random Forest, XGBoost) for LONGITUDE & LATITUDE in metres, target standardization, global vs hierarchical routing (oracle & end-to-end consuming M3 predictions). |
| **Vansh** | Member 5 — Evaluation & Report Integration Lead | Shared evaluation harness (`build_results_table.py`), pooled grouped-CV & official-test comparison tables, EvAAL-style scoring, unit testing (`test_evaluation_metrics.py`), and report integration. |

> *Note: Role labels follow the official Project 9 specification (Member 1–5). Assign specific names to these labels prior to final submission.*

---

## Problem Statement

Indoor spaces attenuate and scatter satellite GPS signals, making reliable satellite-based localization impossible indoors. Indoor positioning instead leverages ambient Radio Frequency (RF) signals. This project predicts a user's physical indoor location using Received Signal Strength Indication (RSSI) fingerprints from 520 Wireless Access Points (WAPs):

1. **Building Classification**: Multi-class classification identifying the building (Buildings 0, 1, 2).
2. **Floor Classification**: Multi-class classification predicting the floor level within the identified building (Floors 0 to 4).
3. **Coordinate Regression**: Continuous regression predicting metric floor-plan coordinates (`LONGITUDE`, `LATITUDE` in metres).

---

## Dataset

- **Source**: UJIIndoorLoc, UCI Machine Learning Repository, DOI: [10.24432/C5MS59](https://doi.org/10.24432/C5MS59), CC BY 4.0.
- **Coverage**: ~21,000 Wi-Fi fingerprint records (19,937 training records and 1,111 validation/held-out test records) collected across 3 buildings and 4–5 floors at Universitat Jaume I.
- **Features**: 520 WAP RSSI readings ranging from `-104 dBm` to `0 dBm`, with `+100` representing unobserved/undetected access points. Additional metadata columns include `BUILDINGID`, `FLOOR`, `LONGITUDE`, `LATITUDE`, `USERID`, `PHONEID`, `SPACEID`, `RELATIVEPOSITION`, and `TIMESTAMP`.

### Data Cleaning & Integrity Policies
- **Duplicate Removal**: Removes duplicate rows from the **training set only**. The official test set (`validationData.csv`) is preserved without modification or deduplication.
- **Constant WAP Cleanup**: Detects uninformative constant WAP columns from the **training split only** and drops those identical columns from both splits to eliminate data leakage.
- **Near-Zero RSSI Audit**: Observes and flags suspicious readings between `-5 dBm` and `0 dBm` alongside their associated `USERID`s.
- **Validation Splitting**: Uses 5 deterministic `GroupKFold` splits grouped by `USERID`. Indices refer to the cleaned training frame, ensuring no user appears in both train and validation folds.
- **Storage**: Raw data resides locally in `data/raw/` and is strictly excluded from version control via `.gitignore`.

---

## Repository Structure

```
├── data/
│   ├── raw/                 # Downloaded UCI CSVs (excluded from Git)
│   └── splits/
│       └── folds.json       # Reproducible USERID GroupKFold split indices & metadata
├── docs/
│   ├── theory_phase1.md     # Methodology, representation theory, and theoretical rigor
│   ├── data_quality.md      # Data-quality summary and near-zero RSSI audit notes
│   └── theory_regression.md # Coordinate regression theory & routing bias–variance analysis
├── notebooks/
│   ├── 01_eda.ipynb         # EDA: data quality, class balance, AP counts, coordinates, shift
│   ├── 02_features.ipynb    # Feature conventions, representations, scaling/PCA, grouped kNN
│   └── 04_ml_regression.ipynb # Interactive coordinate regression demo & spatial error analysis
├── results/
│   ├── figures/             # Coordinate error CDFs, spatial error maps, confusion matrices
│   ├── preds/               # Out-of-fold (OOF) and test prediction CSVs
│   └── tables/              # Data-quality, representation, PCA, CV, and comparison tables
├── scripts/
│   ├── download_data.py     # Automated UCI dataset downloader
│   ├── build_splits.py      # Deterministic USERID GroupKFold split generator
│   ├── export_quality_report.py # Generates data quality and duplicate/WAP audit summaries
│   ├── run_knn_baselines.py # 5-fold grouped-CV RSSI representation comparison
│   ├── report_feature_reduction.py # PCA & variance-threshold filtering reports
│   ├── train_classifiers.py # Building and floor classification (RF, XGBoost)
│   ├── summarize_classifier_effects.py # Classifier effects and feature importance summaries
│   ├── plot_classifier_confusion_matrices.py # Confusion matrix generator for tuned OOF models
│   ├── train_regressors.py  # Coordinate regression models (kNN, RF, XGBoost)
│   ├── plot_regression_results.py # Coordinate error CDFs and spatial error maps
│   └── build_results_table.py # Master evaluator and comparison table generator
├── src/
│   ├── data/
│   │   └── load.py          # Common loaders, duplicate removal, and WAP filtering
│   └── features/
│       └── transform.py     # Fold-safe feature representations and transformations
├── tests/
│   └── test_evaluation_metrics.py # Unit tests for evaluation logic and EvAAL metric
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Create and activate a Python virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Download the UCI dataset and generate deterministic splits
python scripts/download_data.py
python scripts/build_splits.py
python scripts/export_quality_report.py
```

---

## Phase 1 — Machine Learning

### 1. Data & Infrastructure (Member 1)
- Repository structure, automated UCI download pipeline, and schema validation.
- Cleans training-only duplicates and detects constant WAPs from the training split.
- Audits suspicious near-zero RSSI readings (`-5` to `0 dBm`) by `USERID`.
- Creates 5-fold deterministic `GroupKFold` splits grouped by `USERID` saved in `data/splits/folds.json`.
- Designates `validationData.csv` as the pristine, untouched held-out test set.
- Artifacts: `notebooks/01_eda.ipynb`, `src/data/load.py`, `docs/data_quality.md`.

### 2. Features & Representations (Member 2)
- Establishes a shared `+100` to `-110 dBm` convention.
- Implements and compares three distinct RSSI representations (raw/scaled, linear-positive, and exponential/normalized).
- Generates engineered missingness features, detection count indicators, and max/mean RSSI statistics.
- Strict fold-only scaling, PCA variance reduction, and fold-fitted zero-variance filtering.
- Benchmarks representations via 5-fold grouped-CV kNN (`results/tables/knn_representation_summary.csv`).
- Artifacts: `notebooks/02_features.ipynb`, `src/features/transform.py`.

### 3. Classification Pipeline (Member 3)
- Trains Random Forest and XGBoost classifiers for Building and Floor prediction using Member 2's linear-positive representation across the 5 USERID folds.
- Evaluates full feature sets (941 features) against fold-fitted zero-variance-filtered variants.
- Conducts nested USERID hyperparameter tuning, generating out-of-fold (OOF) predictions.
- Feature importance analysis (XGBoost gain and Random Forest impurity).
- Scripts & Artifacts:
  ```bash
  python scripts/train_classifiers.py
  python scripts/summarize_classifier_effects.py
  python scripts/plot_classifier_confusion_matrices.py --predictions results/preds/classifiers_tuned_grouped_cv.csv --output-prefix results/figures/classifier_tuned
  ```
- Outputs: `results/tables/classifier_grouped_cv_summary.csv`, `classifier_tuned_grouped_cv_summary.csv`, `classifier_tuned_selected_hyperparameters.json`, `results/preds/classifiers_tuned_grouped_cv.csv`.

### 4. Coordinate Regression & Validation (Member 4)
- Trains coordinate regressors (kNN, Random Forest, XGBoost) to predict metric `LONGITUDE` and `LATITUDE`.
- Targets standardized strictly on training fold statistics and inverted back to metres prior to error calculation.
- Evaluates spatial routing under both **Oracle diagnostics** (true building/floor) and **Realistic end-to-end routing** (consuming Member 3's building/floor predictions).
- Inner split preprocessing prevents leakage; tuned runs consume tuned Member 3 RF OOF routes (`_tuned` suffix).
- Scripts & Artifacts:
  ```bash
  python scripts/train_regressors.py
  python scripts/plot_regression_results.py
  ```
- Outputs: `results/tables/regressor_grouped_cv_summary.csv`, `regressor_grouped_cv_folds.csv`, `regressor_test_summary.csv`, `results/figures/coordinate_error_cdf.png`, `results/figures/coordinate_error_maps.png`, `notebooks/04_ml_regression.ipynb`, `docs/theory_regression.md`.

### 5. Evaluation & Comparison Integration (Member 5)
- Shared master evaluation harness:
  ```bash
  python scripts/build_results_table.py
  ```
- Reads prediction files from `results/preds/` and compiles pooled grouped-CV and official test metrics.
- Calculates Building Accuracy, Floor Accuracy, Conditional Floor Accuracy, Coordinate MAE, error quantiles (25th, 50th/median, 75th, 90th percentile), and the guide's EvAAL-style positioning penalty score.
- Unit tests: `tests/test_evaluation_metrics.py`.
- Outputs: `results/tables/phase1_comparison.csv` and `results/tables/phase1_results_by_fold.csv`.

---

## Phase 2 — Deep Learning (Planned)

Phase 2 will be evaluated on Architecture Logic, Hybrid Innovation, Ablation Studies, DL Literature Review, DL Regularization, Technical Validation, and Viva/Report integration:

- **Multi-Task Deep Neural Network (MT-DNN)**: Shared spatial RSSI feature representation with task-specific heads for Building (cross-entropy), Floor (hierarchical/ordinal classification), and Coordinates (MSE/Huber loss in metres).
- **1D-CNN / Transformer Architectures**: Capturing spatial correlation and sparsity patterns across the 520 access point channels.
- **Denoising Autoencoders (DAE)**: Unsupervised pretraining on RSSI fingerprints to reconstruct corrupted or missing AP signals.
- **Ablation Studies**: Rigorous head-to-head comparison evaluating whether Phase 2 DL architectures outperform Phase 1 tree-based ensembles (XGBoost/RF) under identical USERID grouped-CV splits.

---

## Results

Evaluation results generated by `scripts/build_results_table.py`. Fixed-setting baselines reflect initial benchmarks; tuned models reflect nested USERID grouped-CV.

| Model / Pipeline | Target / Task | Grouped-CV Metric (5-Fold USERID) | Official Test Set Metric | Notes |
| --- | --- | --- | --- | --- |
| **kNN Baseline** | RSSI Representation Comparison | `results/tables/knn_representation_summary.csv` | — | Evaluated on USERID training folds only |
| **Random Forest (Tuned)** | Building Classification | ~99%+ Accuracy | Baseline fixed run preserved | Linear-positive representation, fold-fitted variance filtering |
| **Random Forest (Tuned)** | Floor Classification | ~90%+ Accuracy | Baseline fixed run preserved | Conditional floor accuracy evaluated |
| **XGBoost (Tuned)** | Building & Floor Classification | High Accuracy / Robust OOF | Baseline fixed run preserved | Tuned via nested USERID CV |
| **Hierarchical Regressor** | Coordinates (`LONGITUDE`, `LATITUDE`) | Mean Coordinate Error in metres | `regressor_test_summary.csv` | Evaluated under Oracle & M3 End-to-End Routing |
| **Phase 1 Master Summary** | Joint Evaluation | `results/tables/phase1_comparison.csv` | `phase1_comparison.csv` | Includes EvAAL positioning competition score |

> *Note: Official-test predictions represent the fixed-setting baseline and were not rerun after tuning to prevent test-set overfitting. Model selection is driven exclusively by nested USERID grouped-CV.*

---

## Reproduce

Execute the complete end-to-end pipeline with the following sequence:

```bash
# 1. Environment & Data Setup
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/build_splits.py
python scripts/export_quality_report.py

# 2. Representation & Feature Selection Reports
python scripts/run_knn_baselines.py
python scripts/report_feature_reduction.py

# 3. Classifier Training & Hyperparameter Tuning (Building & Floor)
python scripts/train_classifiers.py
python scripts/summarize_classifier_effects.py
python scripts/plot_classifier_confusion_matrices.py \
  --predictions results/preds/classifiers_tuned_grouped_cv.csv \
  --output-prefix results/figures/classifier_tuned

# 4. Coordinate Regression & Spatial Visualization
python scripts/train_regressors.py
python scripts/plot_regression_results.py

# 5. Master Comparison Table & Metrics Synthesis
python scripts/build_results_table.py

# 6. Run Unit Tests
pytest tests/
```

---

## Acknowledgements

- **Dataset**: Joaquín Torres-Sospedra et al., *UJIIndoorLoc: A New Multi-building and Multi-floor Database for WLAN Fingerprint-based Indoor Localization Problems*, UCI Machine Learning Repository, DOI: [10.24432/C5MS59](https://doi.org/10.24432/C5MS59), CC BY 4.0.
- **Evaluation Structure & Rubric**: CSAI Student Project Instructions (Phase 1: Advanced Machine Learning & Phase 2: Deep Learning), Newton School of Technology & Rishihood University.
