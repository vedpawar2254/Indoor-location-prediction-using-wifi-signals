# Presentation Decks (Phase 1 & Phase 2)

This directory contains the complete presentation decks for **Project 9 — Deep Learning & Applied Machine Learning (UJIIndoorLoc Benchmark)**.

---

## Phase 2 Deliverables (Deep Learning & Hybrid ML+DL)

- **[Phase2_Presentation.pptx](Phase2_Presentation.pptx)**: 12-slide widescreen PPTX deck (0.69 MB, $\le 10$ MB requirement).
- **[Phase2_Presentation.pdf](Phase2_Presentation.pdf)**: 12-slide high-resolution PDF presentation (0.82 MB, $\le 10$ MB requirement).
- **Builder Script**: `build_phase2_deck.py` (reproducibly generates both formats).

### Phase 2 Slide Structure & Agenda

| Slide | Topic | Content Summary |
| :---: | :--- | :--- |
| **01** | **Title & Key Metrics** | Team T054, course info, verified metrics (99.91% building, 95.50% floor, 4.29 m MAE, 6.76 m mean / 4.98 m median Euclidean error, 6.99 m EvAAL score). |
| **02** | **Phase 1 Recap & Motivation** | Review of Phase 1 tree baseline and two structural bottlenecks: target decoupling (error cascading from 10.08 m to 28.57 m) and hardware drift (Phone 14 error 30.45 m). |
| **03** | **Data Hygiene & Grouped Folds** | 637 duplicates removed, 73 zero-AP blind scans filtered strictly for DL training, fold-safe USERID GroupKFold CV. |
| **04** | **Device-Invariant Representation (R3)** | Mathematical formulation of Positive-Powed RSSI, scan-relative exponential decay $\exp((x - x_{\max})/\tau)$ cancelling device gain $\Delta_p$, binary presence flags, and physical statistics. |
| **05** | **HierResMTL Architecture** | Deep Pre-LN residual trunk, strongest-WAP embeddings, FiLM building-to-floor/coord modulation, 64-d bottleneck, Kendall uncertainty loss balancing. |
| **06** | **HierResMTL-Hybrid Method** | Stage 1 DL trunk; Stage 2 OOF-calibrated probability fusion + valid floor mask; Stage 3 building-conditioned adaptive cosine kNN retrieval ($k=7$) gated by similarity confidence. |
| **07** | **Official Validation Benchmark** | Complete Table I benchmark across RF, XGB, SingleTaskMLP, MTL-MLP, SAE-MTL, CNN1D-MTL, HierResMTL, and HierResMTL-Hybrid. |
| **08** | **5-Fold USERID Cross-Validation** | Generalization on 19,300 held-out user scans. Hybrid achieves 10.55 m EvAAL score (CV fold mean $10.68 \pm 3.74$ m vs Phase 1 RF 18.15 m). |
| **09** | **Statistical Validation & Ablations** | Paired 1,000 bootstrap 95% CIs (4.82 m over RF, 1.29 m over DL), McNemar test ($p \ll 10^{-3}$), controlled A1-A9 single-factor ablations. |
| **10** | **Spatial Diagnostics & Confusion** | Empirical error CDF dominating at every percentile; floor confusion matrix eliminating off-by-one errors. |
| **11** | **Device Shift & Hardware Invariance** | Handset generalization across 11 phones: Phone 14 error reduced from 30.45 m to 9.70 m (-68.1%); seen vs unseen phone floor accuracy (95.97% vs 95.24%). |
| **12** | **Team Work & Deliverables** | Individual member breakdown (M1-M5) per Section 7 of instructions, project checklist. |

---

## Phase 1 Baseline Deck
- **[Phase1_Presentation.pdf](Phase1_Presentation.pdf)**: 11-slide presentation deck covering the original Phase 1 classical baseline.
