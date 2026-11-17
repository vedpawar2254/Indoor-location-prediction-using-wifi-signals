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

## Executive Summary & Phase 1 $\rightarrow$ Phase 2 Progression

In Phase 1, separate tree models (**Random Forest** and **XGBoost**) formed a classical baseline. While achieving 99.91% building accuracy and 6.86 m median coordinate error, they revealed two major bottlenecks:
1. **Target Decoupling:** Building, floor, and coordinate models never shared latent representations. When floor classification failed, RF mean coordinate error surged by **+183%** (from 10.08 m to 28.57 m).
2. **Hardware Calibration Drift:** Standard per-AP `StandardScaler` failed to account for device antenna gain offsets $\Delta_p$, causing Phone 14 mean positioning error to explode to **30.45 m** (P95 116.09 m).

In Phase 2, our team addresses both issues through:
- **Device-Invariant Multiview Representation ($R3$):** Combines positive-powed RSSI $[0, 1]^e$, scan-relative exponential decay $\exp((x_{ij} - x_{i,\max})/\tau)$ (provably cancelling additive handset gain $\Delta_p$), binary presence flags, and physical statistics, while filtering 73 zero-AP contradictory training scans.
- **Hierarchical Residual Multitask Network (`HierResMTL`):** Pre-LN residual MLP trunk ($d=384$) + strongest-AP embedding ($d=16$) + **FiLM** building-to-floor/coordinate modulation + 64-d spatial bottleneck + **Kendall et al. (2018)** homoscedastic uncertainty loss balancing + domain AP-block dropout.
- **Genuine ML + DL Hybrid (`HierResMTL-Hybrid`):** Out-of-fold log-loss calibrated probability fusion (`HierResMTL` + `RF`/`XGB` + `R3-kNN`), structural Floor-4 validity masking, and **building-conditioned adaptive similarity-gated localization** ($k=7$) dynamically blending smooth neural manifolds with sharp local corridor retrieval.

### Benchmark Comparison (Official 1,111-Sample Validation Set)

| Metric | Phase 1: RF | Phase 1: XGBoost | Best-per-Task Oracle | Pure DL: HierResMTL | Naive Blend (Strawman) | **HierResMTL-Hybrid (Ours)** | Improvement vs Phase 1 RF |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Building Accuracy** | 99.91% | 99.73% | 99.91% | 99.82% | 99.91% | **99.91%** | Identical |
| **Floor Accuracy** | 90.73% | 91.99% | 91.99% | 93.16% | 93.16% | **95.50%** | **+4.77%** ($p < 10^{-5}$) |
| **Floor Macro-F1** | 0.8927 | 0.9159 | 0.9159 | 0.9267 | 0.9267 | **0.9469** | **+0.0542** |
| **Floor 4 Recall** | 71.79% | 79.49% | 79.49% | 87.18% | 87.18% | **87.18%** | **+15.39%** |
| **Coordinate MAE (Avg)**| 7.49 m | 9.18 m | 7.49 m | 5.13 m | 5.81 m | **4.29 m** | **-42.7%** |
| **Euclidean Mean Error**| 11.58 m | 14.52 m | 11.58 m | 8.05 m | 9.07 m | **6.76 m** | **-41.6%** (paired gain 4.82 m, CI95 [3.92, 5.66]) |
| **Euclidean Median Error**| 6.86 m | 10.06 m | 6.86 m | 5.81 m | 6.28 m | **4.98 m** | **-27.4%** (Sub-5m median) |
| **P90 Error** | 24.99 m | 30.38 m | 24.99 m | 16.21 m | 19.53 m | **13.88 m** | **-44.5%** (Tails suppressed) |
| **EvAAL Score (Penalized)**| 12.00 m | 14.97 m | 11.95 m | 8.41 m | 9.39 m | **6.99 m** | **-41.8%** reduction |
| **Phone 14 Mean Error**| 30.45 m | 29.56 m | 30.45 m | 11.74 m | — | **9.70 m** | **-68.1%** (device drift mitigated) |

---

## 5-Fold USERID GroupKFold CV (19,300 Held-Out User Scans)

To eliminate identity leakage and guarantee true generalization across unseen users, all hyperparameter tuning and model selection were performed on 5-fold `USERID` grouped splits (`data/splits/folds.json`).

| Model Architecture | OOF Bld Acc | OOF Flr Acc | OOF Mean Error | OOF Med Error | OOF P90 Error | OOF EvAAL Score | CV Fold EvAAL (Mean $\pm$ Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 1: RandomForest** | 99.09% | 74.31% | 16.67 m | 9.86 m | 39.70 m | 18.15 m | 18.31 $\pm$ 3.90 m |
| **Phase 1: XGBoost** | 98.02% | 77.77% | 20.34 m | 14.14 m | 44.10 m | 22.22 m | 22.03 $\pm$ 5.57 m |
| **DL: SingleTaskMLP** | 98.92% | 80.10% | 10.72 m | 7.55 m | 21.58 m | 12.05 m | 12.17 $\pm$ 3.16 m |
| **DL: MTL-MLP (Phase 1 Proposed)**| 97.61% | 80.77% | 10.61 m | 7.34 m | 21.43 m | 12.57 m | 12.71 $\pm$ 4.01 m |
| **DL: SAE-MTL (Nowicki 2017)**| 98.85% | 82.16% | 10.25 m | 7.40 m | 20.77 m | 11.54 m | 11.65 $\pm$ 3.31 m |
| **DL: CNN1D-MTL (Song 2019)** | 97.67% | 77.77% | 22.91 m | 12.09 m | 56.97 m | 24.96 m | 24.86 $\pm$ 6.82 m |
| **DL: HierResMTL (Proposed DL)** | 97.80% | 80.34% | 9.65 m | 6.23 m | 19.21 m | 11.53 m | 11.63 $\pm$ 3.86 m |
| **ML: R3-Cosine-kNN** | 96.87% | 81.41% | 10.61 m | 5.50 m | 23.06 m | 12.92 m | 13.13 $\pm$ 5.27 m |
| **Hybrid: Naive Blend (0.5 RF+DL)**| 99.09% | 80.34% | 11.63 m | 7.66 m | 25.99 m | 12.87 m | 12.99 $\pm$ 2.82 m |
| **Hybrid: HierResMTL-Hybrid (Ours)**| **98.12%** | **81.09%** | **8.86 m** | **5.33 m** | **18.45 m** | **10.55 m** | **10.68 $\pm$ 3.74 m** |

*Takeaway:* `HierResMTL-Hybrid` is the strongest model across **both** the official validation set (6.99 m EvAAL) and the 19,300 held-out-user cross-validation scans (10.55 m EvAAL), demonstrating out-of-distribution generalization to unseen users.

---

## Controlled Single-Factor Ablation Studies (A1 – A9)

To determine exactly which component drove the gains, we executed controlled single-factor ablations trained on Fold 0 and evaluated on the official validation set (`outputs/phase2/ablation_results.csv`):

| Ablation Category | Configuration | Val Flr Acc | Val Coord Mean | Val Coord Median | Val EvAAL Score | Delta vs Full |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Representation** | **R1: Phase 1 StandardScaler** | 79.30% | 28.99 m | 17.73 m | 36.48 m | +23.46 m (Collapse) |
| | **R2: Positive-Powed RSSI** | 81.80% | 38.38 m | 13.09 m | 44.50 m | +31.48 m (Tail bias) |
| | **R3-noRel: Powed+Mask+Agg** | 81.90% | 11.69 m | 7.69 m | 13.45 m | +0.43 m |
| | **R3 (Full): +Scan-Rel Decay** | **82.40%** | **11.19 m** | **8.08 m** | **13.02 m** | **Baseline** |
| **Hygiene** | **w/o Blind-Scan Filtering** | 87.00% | 10.29 m | 7.77 m | 11.53 m | -1.49 m |
| **Regularization** | **w/o AP Dropout & Jitter** | 89.40% | 11.53 m | 7.79 m | 13.08 m | +0.06 m |
| **Loss Balancing** | **w/o Kendall Uncertainty Loss** | 89.70% | 11.37 m | 7.40 m | 13.09 m | +0.07 m |
| **Conditioning** | **w/o FiLM Building Conditioning**| 87.40% | 11.59 m | 8.05 m | 13.31 m | +0.29 m |
| **Imbalance** | **w/o Floor Class Reweighting** | 87.00% | 11.48 m | 7.47 m | 13.80 m | +0.78 m |

*Note:* these are single-fold (Fold 0) single-run numbers; differences smaller than roughly 1 m are within single-run noise (see the multi-seed spread below), so the representation jump (R1/R2 → R3) is the only large, unambiguous effect. The component ablations (FiLM, Kendall, dropout) move EvAAL by well under a metre on Fold 0 alone; their value shows more clearly in the full 5-fold CV than in this single fold.

---

## Statistical Validation Highlights

1. **Paired Bootstrap 95% Confidence Intervals (1,000 Resamples):**
   - Hybrid vs Phase 1 RF Euclidean error reduction: **`4.82 m`** (95% CI `[3.92, 5.66]` m).
   - Hybrid vs Pure DL error reduction: **`1.29 m`** (95% CI `[1.06, 1.52]` m).
   - Both CIs exclude zero, so each improvement is statistically significant.
2. **McNemar Floor Significance Test:**
   - `HierResMTL-Hybrid` vs Phase 1 XGBoost: 48 wins vs 9 losses ($p \approx 0$, well below 0.001).
   - `HierResMTL-Hybrid` vs Pure DL: 33 wins vs 7 losses ($p = 4.2 \times 10^{-5}$).
3. **Floor Expected Calibration Error (ECE), 10-bin:**
   - `HierResMTL` ECE: **`0.0260`** (mean confidence 92.0%, accuracy 93.2%).
   - `HierResMTL-Hybrid` ECE: **`0.0476`** (mean confidence 90.7%, accuracy 95.5%); slightly under-confident, computed on properly normalized floor probabilities.
4. **Sample Efficiency Learning Curve (user-grouped):**
   - 25% users (2,703 scans): val EvAAL `107.42 m`
   - 50% users (5,728 scans): val EvAAL `98.01 m`
   - 75% users (9,284 scans): val EvAAL `89.65 m`
   - 100% users (14,713 scans): val EvAAL `13.68 m`
   - With only a handful of training users the user-grouped coordinate estimates are highly unstable; performance is usable only once the full user population is seen. This exposes how much the model depends on spatial user diversity rather than raw sample count.
5. **Multi-Seed Stability (single-fold HierResMTL, seeds 42/123/999):**
   - Mean Euclidean error $12.03 \pm 1.55$ m, floor accuracy $85.69\% \pm 1.28\%$ — the run-to-run spread of one single-fold model, distinct from the 5-fold ensembled hybrid (6.76 m).

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
