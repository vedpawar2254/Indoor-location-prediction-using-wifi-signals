# Indoor Location Prediction Using Wi-Fi Signals (Project 9)

Predicting indoor location (**building**, **floor**, and **continuous floor-plan coordinates**) from Wi-Fi signal strength (RSSI) fingerprints.

**Team T054** — Section E, BTech CSE (AI), Newton School of Technology, Rishihood University (Batch 2024–2028)  
Course project: Advanced Machine Learning (Phase 1 AML Baseline) + Deep Learning (Phase 2), per the official CSAI Project Instructions.

GitHub Repository: [vedpawar2254/Indoor-location-prediction-using-wifi-signals](https://github.com/vedpawar2254/Indoor-location-prediction-using-wifi-signals)

---

## Phase 1 Ownership & Team Contributions

| Member | Work Order Role | Phase 1 Responsibility | Main Deliverables & Code |
| :--- | :--- | :--- | :--- |
| **Aviral Mishra** | **Member 1 — Data & Infrastructure Lead** | Data setup, automated ingestion, quality checks, EDA, and USERID-grouped folds. | `scripts/download_data.py`, `scripts/build_splits.py`, `src/part1_eda.py`, `data/splits/folds.json`, `data/splits/folds_summary.csv`, EDA plots (`eda_spatial_layout.png`, `eda_floor_per_building.png`, `eda_aps_per_sample.png`). |
| **Ved Pawar** | **Member 2 — Features & Representation Lead** | Missing-signal handling (+100 dBm to -105 dBm floor), 3 RSSI representations, 936 engineered features, fold-safe transforms, and kNN feature baselines. | `src/part2_preprocess.py`, `src/part3_features.py`, `scripts/run_knn_baselines.py`, `outputs/feature_ablation_results.csv`, `outputs/knn_baseline_results.csv`. |
| **Khuswant** | **Member 3 — Classification Lead** | Independent building and floor classification with Random Forest and XGBoost. | `src/part4_train.py` (building & floor classifiers), out-of-fold confusion matrix evaluations. |
| **Prataya** | **Member 4 — Coordinate Regression Lead** | Multi-output 2D coordinate regression (LONGITUDE, LATITUDE) and spatial error analysis. | `src/part4_train.py` (MultiOutputRegressor), `src/part6_visualize.py` (spatial error maps & displacement vectors). |
| **Vansh** | **Member 5 — Evaluation & Integration Lead** | Unified evaluation harness, coordinate MAE reporting, EvAAL-style combined error, and device robustness. | `src/part5_evaluate.py`, `src/part7_robustness.py`, `outputs/metrics.csv`, `outputs/robustness_by_phone.csv`. |

---

## Problem Statement

Satellite GPS signals attenuate and scatter heavily when penetrating indoor environments, rendering satellite navigation ineffective indoors. Ambient Radio Frequency (RF) signals from existing Wi-Fi infrastructure provide an attractive alternative.

Given a fingerprint vector of **520 Wireless Access Point (WAP)** RSSI observations, this project solves a coupled multitask localization problem:
1. **Building Classification**: 3 classes (Buildings 0, 1, 2).
2. **Floor Classification**: 5 classes (Floors 0 to 4).
3. **Coordinate Regression**: Continuous floor-plan metric coordinates (`LONGITUDE`, `LATITUDE` in meters).

---

## Dataset & Integrity Policies

- **Source**: UJIIndoorLoc, UCI Machine Learning Repository, DOI: [10.24432/C5MS59](https://doi.org/10.24432/C5MS59), CC BY 4.0.
- **Coverage**: 19,937 training records and 1,111 validation/held-out test records collected across 3 buildings and 4–5 floors at Universitat Jaume I.
- **Signal Sparsity**: Wi-Fi fingerprints are **96.5% sparse**. The median fingerprint observes only 17 access points; 55 access points are never observed in training data.

### Integrity & Preprocessing Policies
- **Deduplication**: 637 exact-duplicate scans in the training set are dropped (`19,937 -> 19,300`). The official validation set (`validationData.csv`) is preserved without modification.
- **Missing-Signal Sentinel Remapping**: The dataset uses `+100 dBm` as a non-reception sentinel. We remap undetected APs to a fixed floor `NO_SIGNAL_FILL = -105 dBm`, just below the weakest observed real reading (-104 dBm).
- **Constant AP Removal**: 55 unobserved WAPs in training are pruned, yielding 465 active AP features.
- **Leakage-Free Scaling**: `StandardScaler` is fitted strictly on training data and applied unchanged to validation.
- **Deterministic No-Signal Guard (G1)**: `detect_no_signal()` flags scans where zero APs are detected, preventing arbitrary spatial predictions.
- **USERID GroupKFold Splits**: 5-fold cross-validation partitioned strictly by `USERID` on the training set (`data/splits/folds.json`), ensuring zero user leakage.

---

## Feature Engineering & Representations (Member 2 Deliverable)

The feature matrix constructs **936 features**:
1. **465 Scaled RSSI Features**: Standardized continuous readings for all active APs.
2. **465 Binary Absence Indicators (`WAP###_absent`)**: Explicit missingness representation (1.0 if WAP <= -105 dBm, 0.0 otherwise). While standard scaling is an invertible linear transform, decision trees make orthogonal axis-aligned splits on single features. Standardized missing values fall within continuous floats; explicit binary flags enable trees to isolate AP visibility in a single split.
3. **5 Physical Signal Aggregates**: Standardized summary statistics computed from raw dBm:
   - `signal_count`: Total APs heard
   - `strongest_signal`: Maximum received dBm
   - `mean_rssi`: Average signal over heard APs
   - `std_rssi`: Standard deviation of heard signals
   - `range_rssi`: Dynamic range (`strongest - weakest`)
4. **1 Discrete Location Fingerprint**: `strongest_wap_id` (argmax AP index, -1 if no signal heard).

### Empirical Feature Ablation Study (kNN, k=5)

| Representation Mode | Features | Building Acc | Floor Acc | Coord MAE (Avg) | Coord Mean Error | P95 Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A1: Scaled RSSI only** | 465 | 99.01% | 80.65% | 7.82 m | 12.06 m | 39.17 m |
| **A2: RSSI + Absence flags** | 930 | 98.92% | 80.56% | 8.03 m | 12.35 m | 39.35 m |
| **A3: Full Engineered Set** | **936** | **99.01%** | **84.34%** | **7.59 m** | **11.77 m** | **37.75 m** |

*Result:* The engineered aggregate features provide high-signal shortcuts that boost floor classification by **+3.69%** and reduce 95th-percentile localization error.

---

## Verified Phase 1 Baseline Results

Evaluated on the official 1,111-row validation set:

| Model | Building Acc | Floor Acc | Floor Macro-F1 | Lon MAE | Lat MAE | Avg MAE | Euclidean Mean | Euclidean Median | P90 Error | EvAAL Custom Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **99.91%** | 90.73% | 0.8927 | **8.19 m** | **6.80 m** | **7.49 m** | **11.58 m** | **6.86 m** | **24.99 m** | 12.00 m |
| **XGBoost** | 99.73% | **91.99%** | **0.9159** | 9.82 m | 8.54 m | 9.18 m | 14.52 m | 10.06 m | 30.38 m | 14.97 m |
| **Combined (Post-Hoc)** | **99.91%** | **91.99%** | **0.9159** | **8.19 m** | **6.80 m** | **7.49 m** | **11.58 m** | **6.86 m** | **24.99 m** | **11.95 m** |

*Notes on Evaluation Conventions:*
- **Combined Score Formula**: `Euclidean_error + 4.0 * (floor != true_floor) + 50.0 * (building != true_building)` (EvAAL competition penalty).
- **Combined Row Status**: Post-hoc exploratory combination routing each target to its stronger model family (RF for building and coordinates, XGBoost for floor).
- **Target Coupling Evidence**: Positioning error increases from **10.08 m** when building and floor are correct to **28.57 m** when misclassified.

---

## Device Robustness: Seen vs. Unseen Handset Analysis

Validation data consists of **11 distinct handsets**, providing a realistic test of antenna calibration shifts:
- **Shared (Seen) Handsets**: Phones 13 and 14 appear in both training and validation (**397 validation samples**).
- **Unseen Handsets**: 9 phones (0, 2, 4, 5, 9, 12, 15, 20, 21) are absent from training (**714 validation samples**).

| Model Family | Handset Group | Samples | Floor Accuracy | Mean Coord Error | Median Coord Error |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | **Seen Handsets (13 & 14)** | 397 | 91.69% | 10.13 m | 5.14 m |
| **Random Forest** | **Unseen Handsets (9 phones)**| 714 | 90.20% | 12.39 m | 7.88 m |
| **XGBoost** | **Seen Handsets (13 & 14)** | 397 | 93.95% | 14.14 m | 8.82 m |
| **XGBoost** | **Unseen Handsets (9 phones)**| 714 | 90.90% | 14.73 m | 10.99 m |

---

## Repository Structure

```
├── data/
│   ├── splits/
│   │   ├── folds.json             # 5-fold USERID GroupKFold split indices (Member 1)
│   │   └── folds_summary.csv      # Sample and class distributions per fold
│   └── UJIndoorLoc/               # Dataset CSVs (excluded from Git; downloaded via script)
├── outputs/
│   ├── build_notebook.py          # Automated Phase 1 Jupyter notebook compiler
│   ├── confusion_floor_*.png      # Floor classification confusion matrices
│   ├── eda_*.png                  # Spatial layout, floor distribution, sparsity histograms
│   ├── error_cdf_comparison.png   # Positioning error CDF comparison (RF vs XGBoost)
│   ├── error_map_*.png            # Validation points colored by spatial error
│   ├── error_vectors_*.png        # Worst 10% displacement vectors (true -> predicted)
│   ├── feature_ablation_results.csv # Empirical A1/A2/A3 representation comparison (Member 2)
│   ├── knn_baseline_results.csv   # kNN feature baseline metrics (Member 2)
│   ├── metrics.csv                # Headline metrics including explicit Coordinate MAE
│   ├── predictions.npz            # Cached ground truth and out-of-fold validation predictions
│   ├── robustness_by_phone.csv    # Per-handset and seen/unseen error table
│   └── robustness_by_phone.png    # Per-handset mean coordinate error bar chart
├── scripts/
│   ├── download_data.py           # Automated UCI dataset downloader (Member 1)
│   ├── build_splits.py            # Reproducible USERID GroupKFold split generator (Member 1)
│   ├── run_knn_baselines.py       # Feature ablation & kNN baseline harness (Member 2)
│   └── run_pipeline.py            # Master one-command end-to-end pipeline runner
├── src/
│   ├── config.py                  # Paths, constants, and evaluation penalty definitions
│   ├── part1_eda.py               # EDA, data quality checks, near-zero RSSI audit (Member 1)
│   ├── part2_preprocess.py        # Deduplication, sentinel remapping, train-only scaling (M1/M2)
│   ├── part3_features.py          # 936 engineered features & ablation modes (Member 2)
│   ├── part4_train.py             # 6 tabular baseline estimators (RF + XGBoost)
│   ├── part5_evaluate.py          # Coordinate MAE, Euclidean metrics, and EvAAL score
│   ├── part6_visualize.py         # Confusion matrices, error maps, and CDF plots
│   └── part7_robustness.py        # Hardware domain shift: seen vs unseen handset analysis
├── Phase1_AML_Baseline.ipynb      # Complete executed Phase 1 deliverable notebook
├── requirements.txt               # Locked dependencies
├── .gitignore                     # Git exclusions (raw data and large model binaries)
└── README.md                      # Project documentation and reproduction guide
```

---

## Quickstart & Execution

### 1. Environment Setup
```bash
git clone https://github.com/vedpawar2254/Indoor-location-prediction-using-wifi-signals.git
cd Indoor-location-prediction-using-wifi-signals

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Dataset
Automatically downloads and extracts the official UJIIndoorLoc dataset from the UCI ML Repository:
```bash
python scripts/download_data.py
```

### 3. Run Entire Pipeline (One Command)
Runs the complete lifecycle end-to-end, executing data quality audits, split generation, feature engineering, model training, evaluation, visualization, robustness analysis, and notebook compilation:
```bash
python scripts/run_pipeline.py
```

### 4. Run Individual Modules
```bash
# Member 1: EDA & USERID GroupKFold generation
python src/part1_eda.py
python scripts/build_splits.py

# Member 2: Feature Engineering & kNN Representation Ablation
python src/part3_features.py
python scripts/run_knn_baselines.py

# Model Training & Unified Evaluation
python src/part4_train.py
python src/part5_evaluate.py
python src/part6_visualize.py
python src/part7_robustness.py
```

---

## Transition to Phase 2 (Multitask Deep Learning)

The Phase 1 AML baseline establishes:
1. **Target Coupling**: Independent models cannot exploit spatial correlations between building, floor, and continuous coordinates.
2. **Complementary Strengths**: Random Forest provides superior coordinate regression (7.49 m MAE), while XGBoost yields superior floor classification (91.99%).
3. **Phase 2 Plan**: In Phase 2, a shared deep MLP trunk with auxiliary classification heads (building, floor) and a regression head (coordinates) will be trained jointly using this exact verified feature pipeline and split protocol for rigorous, fair comparison.
