# Phase 1 Presentation Deck

This directory contains the presentation slides for **Project 9 — Applied Machine Learning (Phase 1 Baseline)**.

## Presentation File
- **[Phase1_Presentation.pdf](Phase1_Presentation.pdf)**: 11-slide presentation deck covering the full Phase 1 baseline methodology, data engineering, modeling results, failure analysis, and Phase 2 transition.

---

## Slide Structure & Agenda

| Slide | Topic | Content Summary |
| :---: | :--- | :--- |
| **01** | **Title & Headline** | Team T054, course info, and verified headline metrics (99.91% building, 91.99% floor, 6.86 m median coord). |
| **02** | **Problem Formulation** | Wi-Fi fingerprinting breakdown into 3 coupled tasks: Building (3-way), Floor (5-way), Coordinates (continuous metric 2D). |
| **03** | **Dataset & EDA** | 19,937 train & 1,111 validation scans; 96.5% signal sparsity audit; floor imbalance (Floor 4 rare). |
| **04** | **Data Cleaning & 936 Features** | -105 dBm floor remapping, deduplication (19,300 unique scans), 465 RSSI + 465 absence flags + 5 physical aggregates + 1 argmax AP. |
| **05** | **Model Architecture** | Tabular tree-based baselines: Bagging (Random Forest) vs Boosting (XGBoost). |
| **06** | **Evaluation Metrics** | Coordinate MAE, 2D Euclidean distance, and custom penalized combined score (EvAAL rules: +4m floor penalty, +50m building penalty). |
| **07** | **Verified Results** | Random Forest wins coordinates (7.49 m avg MAE, 6.86 m median error); XGBoost wins floor classification (91.99%). Post-hoc combined score: 11.95 m. |
| **08** | **Error Diagnostics & Spatial Patterns** | Floor confusion off-by-one clustering; coordinate errors concentrated at building perimeter/corners where signal density thins. |
| **09** | **Device Robustness** | Antenna shift evaluation across 11 handsets: 2 shared handsets (397 scans, RF 10.13 m) vs 9 unseen handsets (714 scans, RF 12.39 m). |
| **10** | **Limitations & Phase 2 Roadmap** | Independent estimators cannot share representations; sets up Phase 2 shared multi-task deep neural network (one trunk feeding three heads). |
| **11** | **Conclusion & References** | Final takeaways and academic references (Torres-Sospedra et al., Caruana). |
