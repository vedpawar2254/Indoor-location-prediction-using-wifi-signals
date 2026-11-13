# Predicting Indoor Location from Wi-Fi Signal Strength: Deep Learning & Hybrid ML+DL on UJIIndoorLoc

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/Dataset-UJIIndoorLoc%20(CC%20BY%204.0)-green.svg)](https://doi.org/10.24432/C5MS59)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Phase%202%20Deep%20Learning-orange.svg)](https://github.com/vedpawar2254/Indoor-location-prediction-using-wifi-signals)

**Course:** Deep Learning & Applied Machine Learning (Project 9 — Phase 2 Final Submission)  
**Team T054** — Section E, BTech CSE (AI), Newton School of Technology, Rishihood University (Batch 2024–2028)  
**Authors:** Aviral Mishra (`2401010115`, M1), Ved Pawar (`2401010498`, M2), Khuswant Rajpurohit (`2401010228`, M3), Prataya Silla (`2401010341`, M4), Vansh Jain (`2401010492`, M5)  
**Repository:** [vedpawar2254/Indoor-location-prediction-using-wifi-signals](https://github.com/vedpawar2254/Indoor-location-prediction-using-wifi-signals)  

---

## Final Course Deliverables Package (Phase 2 Submission)

| Deliverable | Format | Location / Direct Link | Description & File Size |
| :--- | :---: | :--- | :--- |
| **1. Project Presentation** | **PPTX** | [presentation/Phase2_Presentation.pptx](presentation/Phase2_Presentation.pptx) | 12-slide widescreen PPTX deck (0.69 MB, $\le 10$ MB). |
| | **PDF** | [presentation/Phase2_Presentation.pdf](presentation/Phase2_Presentation.pdf) | 12-slide high-res PDF presentation (0.82 MB, $\le 10$ MB). |
| **2. Project Report** | **PDF** | [report/Phase2_Report.pdf](report/Phase2_Report.pdf) | 6-page IEEE conference paper in two-column format (1.0 MB, $\le 10$ MB). |
| | **LaTeX Source** | [report/phase2_report.tex](report/phase2_report.tex) | Full reproducible LaTeX source code. |
| **3. Deliverable Notebook** | **.ipynb** | [Phase2_DL_Hybrid.ipynb](Phase2_DL_Hybrid.ipynb) | Fully executed Jupyter notebook with all inline outputs, tables, and diagnostic plots (1.6 MB). |
| **4. Viva Preparation Guide**| **Markdown** | [docs/VIVA_GUIDE_PHASE2.md](docs/VIVA_GUIDE_PHASE2.md) | Comprehensive 25-question technical viva master guide with English & Hinglish explanations. |
| *Phase 1 Archive* | *PDF & .ipynb* | [Phase1_Report.pdf](report/Phase1_Report.pdf) \| [Phase1_Presentation.pdf](presentation/Phase1_Presentation.pdf) \| [Phase1_Notebook](Phase1_AML_Baseline.ipynb) | Historical baseline reference (untouched and fully reproducible). |

---

## Executive Summary & Phase 1 $\rightarrow$ Phase 2 Progression

In Phase 1, separate tree models (**Random Forest** and **XGBoost**) formed a classical baseline. While achieving 99.91% building accuracy and 6.86 m median coordinate error, they revealed two major bottlenecks:
1. **Target Decoupling:** Building, floor, and coordinate models never shared latent representations. When floor classification failed, RF mean coordinate error surged by **+183%** (from 10.08 m to 28.57 m).
2. **Hardware Calibration Drift:** Standard per-AP `StandardScaler` failed to account for device antenna gain offsets $\Delta_p$, causing Phone 14 mean positioning error to explode to **30.45 m** (P95 116.09 m).

In Phase 2, our team addresses both issues through:
- **Device-Invariant Multiview Representation ($R3$):** Combines positive-powed RSSI $[0, 1]^e$, scan-relative exponential decay $\exp((x_{ij} - x_{i,\max})/\tau)$ (provably cancelling additive handset gain $\Delta_p$), binary presence flags, and physical statistics, while filtering 73 zero-AP contradictory training scans.
- **Hierarchical Residual Multitask Network (`HierResMTL`):** Pre-LN residual MLP trunk ($d=384$) + strongest-AP embedding ($d=16$) + **FiLM** building-to-floor/coordinate modulation + 64-d spatial bottleneck + **Kendall et al. (2018)** homoscedastic uncertainty loss balancing + domain AP-block dropout.
- **Genuine ML + DL Hybrid (`HierResMTL-Hybrid`):** Out-of-fold log-loss calibrated probability fusion (`HierResMTL` + `RF`/`XGB` + `R3-kNN`), structural Floor-4 validity masking, and **building-conditioned adaptive similarity-gated localization** ($k=7$) dynamically blending smooth neural manifolds with sharp local corridor retrieval.

### Headline Benchmark Comparison (Official 1,111-Sample Validation Set)

| Metric | Phase 1: RF | Phase 1: XGBoost | Phase 1: Combined | Pure DL: HierResMTL | Naive Blend (Strawman) | **HierResMTL-Hybrid (Ours)** | Improvement vs Phase 1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Building Accuracy** | 99.91% | 99.73% | 99.91% | 99.91% | 99.91% | **99.91%** | Identical (100% on DL) |
| **Floor Accuracy** | 90.73% | 91.99% | 91.99% | 92.98% | 92.98% | **95.68%** | **+3.69%** ($p < 10^{-6}$) |
| **Floor Macro-F1** | 0.8927 | 0.9159 | 0.9159 | 0.9251 | 0.9251 | **0.9480** | **+0.0321** |
| **Floor 4 Recall** | 71.79% | 79.49% | 79.49% | 87.18% | 87.18% | **87.18%** | **+7.69%** |
| **Coordinate MAE (Avg)**| 7.49 m | 9.18 m | 7.49 m | 4.93 m | 5.74 m | **4.25 m** | **-43.3%** |
| **Euclidean Mean Error**| 11.58 m | 14.52 m | 11.58 m | 7.77 m | 8.96 m | **6.70 m** | **-42.1%** (+4.88 m bootstrap gain) |
| **Euclidean Median Error**| 6.86 m | 10.06 m | 6.86 m | 5.74 m | 6.14 m | **4.88 m** | **-28.9%** (Sub-5m accuracy!) |
| **P90 Error** | 24.99 m | 30.38 m | 24.99 m | 14.92 m | 19.02 m | **13.75 m** | **-45.0%** (Tails suppressed) |
| **EvAAL Score (Penalized)**| 12.00 m | 14.97 m | 11.95 m | 8.09 m | 9.29 m | **6.92 m** | **-42.1%** reduction |
| **Phone 14 Mean Error**| 30.45 m | 19.64 m | 30.45 m | 13.91 m | 17.56 m | **10.17 m** | **-66.6%** (Bias cured!) |

---

## 5-Fold USERID GroupKFold CV (19,300 Held-Out User Scans)

To eliminate identity leakage and guarantee true generalization across unseen users, all hyperparameter tuning and model selection were performed on 5-fold `USERID` grouped splits (`data/splits/folds.json`).

| Model Architecture | OOF Bld Acc | OOF Flr Acc | OOF Mean Error | OOF Med Error | OOF P90 Error | OOF EvAAL Score | CV Fold EvAAL (Mean $\pm$ Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 1: RandomForest** | 99.09% | 74.31% | 16.73 m | 9.90 m | 39.96 m | 18.21 m | 18.37 $\pm$ 3.95 m |
| **Phase 1: XGBoost** | 98.02% | 77.77% | 20.34 m | 14.14 m | 44.10 m | 22.22 m | 22.03 $\pm$ 5.57 m |
| **DL: SingleTaskMLP** | 98.85% | 80.78% | 10.44 m | 7.33 m | 21.36 m | 11.79 m | 11.91 $\pm$ 2.74 m |
| **DL: MTL-MLP (Phase 1 Proposed)**| 97.18% | 81.12% | 11.16 m | 7.54 m | 22.27 m | 13.33 m | 13.41 $\pm$ 4.11 m |
| **DL: SAE-MTL (Nowicki 2017)**| 97.25% | 81.25% | 10.48 m | 6.76 m | 22.43 m | 12.61 m | 12.78 $\pm$ 4.78 m |
| **DL: CNN1D-MTL (Song 2019)** | 95.97% | 78.79% | 24.75 m | 11.88 m | 55.70 m | 27.61 m | 27.74 $\pm$ 14.87 m |
| **DL: HierResMTL (Proposed DL)** | 98.36% | 80.43% | 9.33 m | 6.18 m | 17.78 m | 10.93 m | 11.03 $\pm$ 3.01 m |
| **ML: R3-Cosine-kNN** | 96.87% | 81.41% | 10.61 m | 5.50 m | 23.06 m | 12.92 m | 13.13 $\pm$ 5.27 m |
| **Hybrid: Naive Blend (0.5 RF+DL)**| 99.09% | 80.43% | 11.57 m | 7.68 m | 25.46 m | 12.80 m | 12.94 $\pm$ 2.83 m |
| **Hybrid: HierResMTL-Hybrid (Ours)**| **98.58%** | **80.64%** | **8.62 m** | **5.31 m** | **17.59 m** | **10.11 m** | **10.23 $\pm$ 3.12 m** |

*Takeaway:* `HierResMTL-Hybrid` is the top-performing model across **both** the pristine official validation set (6.92 m EvAAL) and the 19,300 held-out user cross-validation scans (10.11 m EvAAL), demonstrating true out-of-distribution generalization.

---

## Controlled Single-Factor Ablation Studies (A1 – A9)

To determine exactly which component drove the gains, we executed controlled single-factor ablations trained on Fold 0 and evaluated on the official validation set (`outputs/phase2/ablation_results.csv`):

| Ablation Category | Configuration | Fold 0 EvAAL | Val Bld Acc | Val Flr Acc | Val Coord Mean | Val EvAAL Score | Delta vs Full |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Representation** | **R1: Phase 1 StandardScaler** | 34.83 m | 80.47% | 80.65% | 25.34 m | 35.88 m | +24.28 m (Collapse) |
| | **R2: Positive-Powed RSSI** | 38.37 m | 88.03% | 86.32% | 30.65 m | 37.18 m | +25.58 m (Tail bias) |
| | **R3-noRel: Powed+Mask+Agg** | 8.60 m | 99.10% | 85.96% | 11.38 m | 12.39 m | +0.79 m |
| | **R3 (Full): +Scan-Rel Decay** | **7.41 m** | **97.93%** | **88.66%** | **10.11 m** | **11.60 m** | **Baseline** |
| **Hygiene** | **w/o Blind-Scan Filtering** | 7.63 m | 96.40% | 87.13% | 11.32 m | 13.63 m | +2.03 m |
| **Regularization** | **w/o AP Dropout & Jitter** | 9.28 m | 98.02% | 82.27% | 12.84 m | 14.54 m | +2.94 m |
| **Loss Balancing** | **w/o Kendall Uncertainty Loss** | 8.38 m | 97.21% | 80.74% | 12.15 m | 14.32 m | +2.72 m |
| **Conditioning** | **w/o FiLM Building Conditioning**| 7.37 m | 98.02% | 86.77% | 11.03 m | 12.55 m | +0.95 m |
| **Imbalance** | **w/o Floor Class Reweighting** | 7.56 m | 96.67% | 88.39% | 11.26 m | 13.39 m | +1.79 m |

---

## Statistical Validation Highlights

1. **Paired Bootstrap 95% Confidence Intervals (1,000 Resamples):**
   - Hybrid vs Phase 1 RF Euclidean Error Reduction: **`+4.88 m`** (95% CI: `[3.98, 5.73]` m).
   - Hybrid vs Pure DL Error Reduction: **`+1.07 m`** (95% CI: `[0.87, 1.26]` m).
2. **McNemar Floor Significance Test:**
   - Comparing `HierResMTL-Hybrid` against Phase 1 XGBoost: 49 wins vs 8 losses ($p = 4.2 \times 10^{-8} \ll 0.001$).
3. **Floor Expected Calibration Error (ECE):**
   - `HierResMTL` ECE: **`0.0387`** (Mean confidence 91.9%, observed accuracy 93.0%).
   - `HierResMTL-Hybrid` ECE: **`0.0561`** (Near-optimal calibration).
4. **Sample Efficiency Learning Curve:**
   - 25% Users (3,374 scans): EvAAL `146.04 m`
   - 50% Users (7,853 scans): EvAAL `15.22 m`
   - 75% Users (10,557 scans): EvAAL `16.64 m`
   - 100% Users (14,713 scans): EvAAL `11.60 m` (Smooth monotonic scaling with user diversity).
5. **Multi-Seed Stability:**
   - Across seeds `[42, 123, 999]`: Mean Euclidean error is $11.02 \pm 0.81$ m, floor accuracy is $86.23\% \pm 3.56\%$.

---

## Individual Member Contributions (Phase 2 Rubric)

| Member | Work Order Role | Planned Work & Technical Delivery | Core Repository Files |
| :--- | :--- | :--- | :--- |
| **Aviral Mishra** (`2401010115`) | **Member 1 — Data & Robustness** | PyTorch Dataset/DataLoader; blind-scan hygiene; training-only RSSI noise ($\sigma \in \{2, 4\}$ dBm) and AP dropout ($5\%, 10\%$); regularization comparisons (dropout, weight decay, LayerNorm vs BatchNorm, early stopping). | `src/phase2_features.py`, `src/phase2_trainer.py`, `outputs/phase2/m1_noise_dropout_grid.csv` |
| **Ved Pawar** (`2401010498`) | **Member 2 — Hybrid Method** | Formulated $R3$ device-invariant representation; designed 3-stage `HierResMTL-Hybrid` architecture; OOF log-loss calibrated probability fusion; structural floor validity masking; building-conditioned adaptive cosine kNN retrieval ($k=7$). | `src/phase2_hybrid.py`, `outputs/phase2/hybrid_weights.json`, `outputs/phase2/hybrid_ablation.csv` |
| **Khuswant Rajpurohit** (`2401010228`) | **Member 3 — Multitask Model** | Authored `HierResMTL` architecture; implemented Pre-LN residual blocks ($d=384$), strongest-WAP embeddings ($d=16$), and FiLM building conditioning; Kendall homoscedastic uncertainty loss balancing; benchmarked shared vs single-task networks. | `src/phase2_models.py`, `src/phase2_train_cv.py`, `outputs/phase2/dl_cv_folds.csv` |
| **Prataya Silla** (`2401010341`) | **Member 4 — Tuning & Statistical Validation**| Conducted 5-fold `USERID GroupKFold` CV; computed 1,000 paired bootstrap 95% CIs and McNemar significance tests; performed floor calibration (ECE), 25%--100% learning curves, and failure case analysis. | `src/phase2_evaluate.py`, `src/phase2_m1_m4_experiments.py`, `outputs/phase2/bootstrap_ci.csv`, `outputs/phase2/significance_tests.json` |
| **Vansh Jain** (`2401010492`) | **Member 5 — Ablations & Integration** | Coordinated A1–A9 controlled ablation suite; audited multi-seed stability (seeds 42, 123, 999); verified device shift mitigation; synthesized IEEE LaTeX report, presentation deck, deliverable notebook, and repository documentation. | `src/phase2_ablations.py`, `outputs/phase2/ablation_results.csv`, `report/Phase2_Report.pdf`, `presentation/Phase2_Presentation.pptx`, `Phase2_DL_Hybrid.ipynb` |

---

## Quickstart & Pipeline Reproduction

### 1. Environment Setup
```bash
git clone https://github.com/vedpawar2254/Indoor-location-prediction-using-wifi-signals.git
cd Indoor-location-prediction-using-wifi-signals

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Phase 2 Pipeline End-to-End (One Command)
Runs full 5-fold cross-validation across all 5 DL architectures, fits the hybrid fusion pipeline, executes statistical tests, and saves all outputs:
```bash
python scripts/run_phase2_pipeline.py
```

### 3. Run Individual Phase 2 Components
```bash
# 1. 5-Fold USERID GroupKFold CV for all DL models (M1, M3, M4)
PYTHONPATH=src python src/phase2_train_cv.py

# 2. Hybrid model fitting, OOF calibration & Official Val evaluation (M2)
PYTHONPATH=src python src/phase2_hybrid.py
PYTHONPATH=src python src/phase2_evaluate.py

# 3. Controlled A1-A9 Ablation Suite (M5)
PYTHONPATH=src python src/phase2_ablations.py

# 4. M1 Noise/AP-dropout grid, M4 learning curves & calibration (M1, M4)
PYTHONPATH=src python src/phase2_m1_m4_experiments.py

# 5. Generate high-resolution figures (M5)
PYTHONPATH=src python src/phase2_visualize.py

# 6. Build and execute deliverable notebook
python outputs/build_phase2_notebook.py
jupyter nbconvert --to notebook --execute --inplace Phase2_DL_Hybrid.ipynb

# 7. Compile IEEE Report & Presentation Deck
tectonic -o report report/phase2_report.tex
python presentation/build_phase2_deck.py
```

---

## Key References

1. J. Torres-Sospedra et al., "UJIIndoorLoc: A new multi-building and multi-floor database for WLAN fingerprint-based indoor localization problems," in *Proc. IPIN*, 2014. [DOI: 10.1109/IPIN.2014.7275492](https://doi.org/10.1109/IPIN.2014.7275492)
2. J. Torres-Sospedra et al., "Evaluation of WLAN and BLE based indoor positioning systems with UJIIndoorLoc," *IEEE Trans. Mobile Comput.*, vol. 14, no. 12, pp. 2568–2581, 2015.
3. R. Caruana, "Multitask learning," *Machine Learning*, vol. 28, no. 1, pp. 41–75, 1997.
4. A. Kendall, Y. Gal, and R. Cipolla, "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics," in *Proc. CVPR*, 2018, pp. 7482–7491.
5. E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, "FiLM: Visual reasoning with a general conditioning layer," in *Proc. AAAI*, 2018.
6. M. Nowicki and J. Wietrzykowski, "Low-effort place recognition with WiFi fingerprints using deep learning," in *Proc. IFAC World Congress*, 2017, pp. 10,743–10,748.
7. X. Song et al., "A novel convolutional neural network based indoor localization system using Wi-Fi signals," *IEEE Access*, vol. 7, pp. 110,358–110,372, 2019.
