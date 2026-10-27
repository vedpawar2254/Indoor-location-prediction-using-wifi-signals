# Predicting Indoor Location from Wi-Fi Signal Strength: A Phase 1 Baseline on UJIIndoorLoc

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/Dataset-UJIIndoorLoc%20(CC%20BY%204.0)-green.svg)](https://doi.org/10.24432/C5MS59)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Phase%201%20Baseline-orange.svg)](https://github.com/vedpawar2254/Indoor-location-prediction-using-wifi-signals)

**Course:** Applied Machine Learning (Project 9 — Phase 1 Baseline)  
**Team T054** — Section E, BTech CSE (AI), Newton School of Technology, Rishihood University (Batch 2024–2028)  
**Authors:** Aviral Mishra, Ved Pawar, Khuswant Rajpurohit, Prataya Silla, Vansh Jain  
**Repository:** [vedpawar2254/Indoor-location-prediction-using-wifi-signals](https://github.com/vedpawar2254/Indoor-location-prediction-using-wifi-signals)  
**Deliverables:** [IEEE Report (PDF)](report/Phase1_Report.pdf) | [Presentation Deck (PDF)](presentation/Phase1_Presentation.pdf) | [Deliverable Notebook](Phase1_AML_Baseline.ipynb)

---

## Abstract

We estimate where a phone is — which building, which floor, and its coordinates — from Wi-Fi signal strengths. On the UJIIndoorLoc benchmark, we build separate tree models for each target using the supplied train/validation split. **Random Forest** reads the building almost perfectly (99.91%) and gives the tightest coordinates (6.86 m median); **XGBoost** is the better floor classifier (91.99%). Routing each task to whichever model wins it yields a post-hoc combined baseline at **11.58 m mean / 6.86 m median Euclidean positioning error** (11.95 m mean / 7.00 m median under our penalized combined metric). Wrong-floor cases have larger coordinate errors, and errors vary more than three-fold across handsets. These results establish the baseline performance for comparison with the proposed shared multitask network in Phase 2.

---

## Team Contributions & Work Order

| Member | Work Order Role | Responsibilities & Deliverables | Core Artifacts |
| :--- | :--- | :--- | :--- |
| **Aviral Mishra** | **Member 1 — Data & Infrastructure Lead** | Led data ingestion, quality audits, deduplication, and USERID-grouped fold creation. | `scripts/download_data.py`, `scripts/build_splits.py`, `src/part1_eda.py`, `data/splits/folds.json`, `data/splits/folds_summary.csv` |
| **Ved Pawar** | **Member 2 — Features & Representation Lead** | Designed missing-signal remapping, feature representations, and kNN ablations. | `src/part2_preprocess.py`, `src/part3_features.py`, `scripts/run_knn_baselines.py`, `outputs/feature_ablation_results.csv` |
| **Khuswant Rajpurohit** | **Member 3 — Classification Lead** | Trained building and floor classifiers with Random Forest and XGBoost. | `src/part4_train.py` (building & floor models), `outputs/confusion_floor_*.png` |
| **Prataya Silla** | **Member 4 — Coordinate Regression Lead** | Implemented multi-output coordinate regressors and spatial error diagnostics. | `src/part4_train.py` (MultiOutputRegressor), `src/part6_visualize.py` (`error_map_*.png`, `error_vectors_*.png`) |
| **Vansh Jain** | **Member 5 — Evaluation & Integration Lead** | Integrated unified metrics, coordinate MAE reporting, and device robustness evaluations. | `src/part5_evaluate.py`, `src/part7_robustness.py`, `outputs/metrics.csv`, `outputs/robustness_by_phone.csv` |

---

## Problem Formulation & Dataset

Satellite positioning fails indoors due to concrete structural attenuation and multi-path scattering. Indoor positioning instead exploits ambient Radio Frequency (RF) signals from nearby Wi-Fi infrastructure.

Given a fingerprint vector of **520 Wireless Access Point (WAP)** RSSI observations, this project solves a coupled multitask localization problem:
1. **Building Classification**: 3 classes (Buildings 0, 1, 2).
2. **Floor Classification**: 5 classes (Floors 0 to 4). Floor 4 exists only in Building 2 (~1,000 scans, introducing class imbalance).
3. **Coordinate Regression**: Continuous metric floor-plan coordinates (`LONGITUDE`, `LATITUDE` in meters).

### Dataset & Preprocessing Integrity
- **Source**: UJIIndoorLoc, UCI Machine Learning Repository, DOI: [10.24432/C5MS59](https://doi.org/10.24432/C5MS59), CC BY 4.0.
- **Coverage**: 19,937 training records and 1,111 validation records collected across 3 buildings and 4–5 floors at Universitat Jaume I.
- **Sparsity**: Wi-Fi fingerprints are **96.5% sparse**. The median fingerprint observes only 17 access points; 55 access points are never observed anywhere in training.
- **Deduplication**: 637 exact-duplicate scans in the training set are dropped (`19,937 -> 19,300 unique scans`). The validation set is preserved without modification.
- **Missing-Signal Sentinel Remapping**: The dataset uses `+100 dBm` for undetected APs. We remap undetected APs to a fixed floor `NO_SIGNAL_FILL = -105 dBm`, just below the weakest observed real reading (-104 dBm).
- **Pruning Dead APs**: 55 unobserved WAPs in training are pruned, yielding 465 active AP columns.
- **Leakage-Free Scaling**: `StandardScaler` is fitted strictly on training data and applied unchanged to validation.
- **Deterministic No-Signal Guard (G1)**: `detect_no_signal()` flags scans where zero APs are detected, preventing arbitrary spatial predictions.
- **USERID GroupKFold Splits**: 5-fold cross-validation partitioned strictly by `USERID` on the cleaned training set (`data/splits/folds.json`), ensuring zero user leakage.

---

## 936-Feature Representation (Member 2 Deliverable)

The feature matrix constructs **936 columns**:
1. **465 Scaled RSSI Readings**: Standardized continuous readings for all active APs.
2. **465 Binary Absence Flags (`WAP###_absent`)**: Marking whether each access point was heard. Standard scaling is invertible: weak-but-heard and absent RSSI remain distinct. However, decision trees use axis-aligned splits; binary flags explicitly expose AP visibility through a boolean threshold. They are a representation choice that simplifies tree branching.
3. **5 Physical Signal Aggregates**: Computed from raw dBm readings before scaling:
   - `signal_count`: Number of APs heard (RSSI > -105 dBm)
   - `strongest_signal`: Maximum received dBm (closest AP)
   - `mean_rssi`: Ambient average over heard APs
   - `std_rssi`: Spread of heard signals (indicates environment complexity/multipath)
   - `range_rssi`: Dynamic range (`strongest - weakest`)
4. **1 Discrete Location Fingerprint**: `strongest_wap_id` (argmax AP index, -1 if no signal heard).

### Empirical Feature Ablation Study (kNN, k=5)

| Representation Mode | Features | Building Acc | Floor Acc | Coord MAE (Avg) | Coord Mean Error | P95 Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A1: Scaled RSSI only** | 465 | 99.01% | 80.65% | 7.82 m | 12.06 m | 39.17 m |
| **A2: RSSI + Absence flags** | 930 | 98.92% | 80.56% | 8.03 m | 12.35 m | 39.35 m |
| **A3: Full Engineered Set** | **936** | **99.01%** | **84.34%** | **7.59 m** | **11.77 m** | **37.75 m** |

*Finding:* Signal aggregates provide critical summary statistics that boost floor classification by **+3.69%** and reduce 95th-percentile error.

---

## Verified Phase 1 Baseline Results

Evaluated on the official 1,111-row validation set:

| Model | Building Acc | Floor Acc | Floor Macro-F1 | Lon MAE | Lat MAE | Avg MAE | Euclidean Mean | Euclidean Median | P90 Error | Penalized Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **99.91%** | 90.73% | 0.8927 | **8.19 m** | **6.80 m** | **7.49 m** | **11.58 m** | **6.86 m** | **24.99 m** | 12.00 m |
| **XGBoost** | 99.73% | **91.99%** | **0.9159** | 9.82 m | 8.54 m | 9.18 m | 14.52 m | 10.06 m | 30.38 m | 14.97 m |
| **Combined (Post-Hoc)** | **99.91%** | **91.99%** | **0.9159** | **8.19 m** | **6.80 m** | **7.49 m** | **11.58 m** | **6.86 m** | **24.99 m** | **11.95 m** |

### Key Evaluation Insights:
- **Coordinate MAE**: In addition to 2D Euclidean distance, we explicitly compute coordinate MAE. RF achieves **8.19 m** on Longitude and **6.80 m** on Latitude (**7.49 m average MAE**), substantially outperforming XGBoost's 9.82 m and 8.54 m (9.18 m average MAE).
- **Euclidean Positioning Error**: Random Forest gives noticeably tighter coordinates (**6.86 m median** vs XGBoost 10.06 m).
- **Floor Accuracy & Confusion**: XGBoost reaches **91.99% floor accuracy** (macro-F1 0.9159), outperforming Random Forest (90.73%). Mistakes cluster on adjacent floors (off-by-one errors), consistent with vertical signal propagation.
- **Rare Floor 4 Recall**: The rare Floor 4 (~1,000 scans, only in Building 2) is recalled **79.5% of the time by XGBoost** and **71.8% by Random Forest**.
- **Combined Post-Hoc Baseline**: Routing building and coordinates from RF and floor from XGBoost yields **11.58 m mean / 6.86 m median Euclidean error** (P90 24.99 m). Under our penalized combined metric, this scores **11.95 m mean / 7.00 m median**. This combination is selected post-hoc on the same validation set, not a fresh test.
- **Penalized Combined Metric**: $\text{Score} = \text{Euclidean Error} + 4 \times \mathbb{I}(\text{floor wrong}) + 50 \times \mathbb{I}(\text{building wrong})$. A wrong floor incurs 4 m once, regardless of the number of floors apart.
- **Target Coupling (Association, not Causation)**: With RF floor predictions, mean RF coordinate error is **10.19 m** when the floor is correct and jumps to **25.21 m** when it is wrong. This is an association, not evidence that a classification error causes regression error (the separate baseline estimators do not pass predictions to each other). Joint multitask learning is a Phase 2 hypothesis to test.

---

## Device Robustness: Shared vs. Unseen Handsets

Averages hide substantial device variance. The validation set uses **11 phone models**, of which two (**Phones 13 and 14**, comprising 397 scans) were seen in training, while nine (**714 scans**) are entirely unseen:

| Model Family | Handset Partition | Scans | Floor Accuracy | Mean Coord Error | Median Coord Error |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | **Shared Handsets (Phones 13 & 14)** | 397 | 91.69% | 10.13 m | 5.14 m |
| **Random Forest** | **Unseen Handsets (9 phones)** | 714 | 90.20% | 12.39 m | 7.88 m |
| **XGBoost** | **Shared Handsets (Phones 13 & 14)** | 397 | 93.95% | 14.14 m | 8.82 m |
| **XGBoost** | **Unseen Handsets (9 phones)** | 714 | 90.90% | 14.73 m | 10.99 m |

- **Handset Error Range**: Per-phone RF mean errors range from under 8 m (**Phone 9 at 7.65 m**) to about 31 m (**Phone 14 at 30.45 m**), a spread of roughly 4:1.
- **Takeaway**: XGBoost is slightly more uniform across devices, while RF yields lower errors on typical handsets. Device sensitivity matters for real deployment and is not revealed by an aggregate mean error alone.

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
├── presentation/
│   ├── Phase1_Presentation.pdf    # Complete 11-slide presentation deck
│   └── README.md                  # Slide-by-slide agenda and overview
├── report/
│   ├── Phase1_Report.pdf          # Finalized 3-page IEEE two-column report
│   ├── report.tex                 # IEEE LaTeX source (synchronized with report)
│   ├── make_pdf.py                # Standalone ReportLab generator
│   └── figs/                      # Six figures used in the report
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
├── requirements.txt               # Pinned dependencies
├── .gitignore                     # Git exclusions (raw data, personal docs, large model binaries)
└── README.md                      # Project documentation and reproduction guide
```

---

## Quickstart & Reproducibility

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
Runs the complete lifecycle end-to-end (audits, splits, feature engineering, baselines, evaluation, visualizations, robustness, and notebook assembly):
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

## Discussion & Transition to Phase 2 (Multitask Deep Learning)

The baseline does its job: building is essentially solved, floors land in the low nineties, and median positioning sits around 7 m, all from models that are easy to reason about and quick to retrain. The honest limits are also clear:
1. We report on one supplied split, not repeated folds, so the figures are point estimates, not confidence intervals.
2. We did no hyperparameter search.
3. Saved USERID-grouped folds were created but are not consumed by the current training script.
4. The three target models never talk to each other.

**Bridge to Phase 2:** The association between floor and coordinate errors (10.19 m when floor is correct vs 25.21 m when wrong), and the uneven behaviour across devices, motivate testing a shared representation: **one deep MLP trunk feeding building, floor, and coordinate heads**. Whether shared learning improves over these separate baselines remains to be measured. We will retain the exact same preprocessing, 936 features, split, and metric conventions so Phase 2 results can be compared directly and fairly.

---

## References

1. J. Torres-Sospedra et al., "UJIIndoorLoc: A new multi-building and multi-floor database for WLAN fingerprint-based indoor localization problems," in *Proc. IPIN*, 2014. [DOI: 10.1109/IPIN.2014.7275492](https://doi.org/10.1109/IPIN.2014.7275492)
2. R. Caruana, "Multitask learning," *Machine Learning*, vol. 28, no. 1, pp. 41–75, 1997. [DOI: 10.1023/A:1007379606734](https://doi.org/10.1023/A:1007379606734)
3. UCI Machine Learning Repository: [UJIIndoorLoc Dataset](https://archive.ics.uci.edu/dataset/310/ujiindoorloc)
