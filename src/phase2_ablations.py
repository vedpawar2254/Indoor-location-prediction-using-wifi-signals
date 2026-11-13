"""Phase 2 Controlled Ablation Studies (Representation, Dataset Hygiene, Regularization, Architecture).

Runs controlled single-factor ablations on USERID GroupKFold Fold 0 (early stopping
strictly on Fold 0's held-out USERID split) and measures zero-leakage transfer to
both Fold 0 unseen users and the Official 1,111-row Validation Set.

Ablation Dimensions:
  1. Input Representation:
     - R1: Phase 1 StandardScaler RSSI + absence flags + aggregates
     - R2: Positive-Powed RSSI [0, 1] (Torres-Sospedra 2015 / CNNLoc)
     - R3_no_rel: Positive-Powed + Presence + Aggregates (removes scan-relative decay)
     - R3 (Full): Device-Invariant Multiview (+ scan-relative exponential decay)
  2. Dataset Hygiene & DL Regularization (on R3 + HierResMTL):
     - No Blind-Scan Filter (retains 73 zero-AP contradictory train rows)
     - No Domain Augmentation (ap_drop_p=0, noise_std=0)
     - No Kendall Loss (fixed 1:1:2 task weights)
     - No FiLM Conditioning (removes building->floor/coord FiLM modulation)
     - No Floor Class Weights (uniform CE instead of inverse-sqrt Floor 4 balancing)

Outputs `outputs/phase2/ablation_results.csv`.
"""
import json
import time
import numpy as np
import pandas as pd

from config import OUT_DIR, FOLDS_JSON, RANDOM_STATE
from part2_preprocess import load_raw, Preprocessor, targets
from phase2_features import filter_non_blind, extract_strongest_wap_idx, build_dl_features
from phase2_models import HierResMTL
from phase2_trainer import train_dl_model, predict_dl_model
from phase2_train_cv import compute_metrics_dict

P2_OUT_DIR = OUT_DIR / "phase2"


def run_ablation_case(
    case_name: str,
    group_name: str,
    train_df: pd.DataFrame,
    es_df: pd.DataFrame,
    val_df: pd.DataFrame,
    kept_waps: list,
    rep_mode: str = "R3",
    drop_blind: bool = True,
    ap_drop_p: float = 0.12,
    noise_std: float = 0.02,
    use_kendall: bool = True,
    use_film: bool = True,
    use_floor_weights: bool = True,
    epochs: int = 45,
):
    tr_use = filter_non_blind(train_df) if drop_blind else train_df.reset_index(drop=True)
    X_tr, X_es = build_dl_features(tr_use, es_df, kept_waps, mode=rep_mode)
    _, X_va = build_dl_features(tr_use, val_df, kept_waps, mode=rep_mode)

    wid_tr = extract_strongest_wap_idx(tr_use, kept_waps)
    wid_es = extract_strongest_wap_idx(es_df, kept_waps)
    wid_va = extract_strongest_wap_idx(val_df, kept_waps)
    ytr, yes, yva = targets(tr_use), targets(es_df), targets(val_df)

    model = HierResMTL(X_tr.shape[1], len(kept_waps), use_film=use_film)
    model, cscaler, hist = train_dl_model(
        model,
        X_tr, wid_tr, ytr["building"], ytr["floor"], ytr["coords"],
        X_es, wid_es, yes["building"], yes["floor"], yes["coords"],
        n_waps=len(kept_waps),
        rep_mode=rep_mode,
        epochs=epochs,
        ap_drop_p=ap_drop_p,
        noise_std=noise_std,
        use_kendall=use_kendall,
        use_floor_weights=use_floor_weights,
        seed=RANDOM_STATE,
    )

    pb_es, pf_es, pc_es, _ = predict_dl_model(model, X_es, wid_es, cscaler)
    pb_va, pf_va, pc_va, _ = predict_dl_model(model, X_va, wid_va, cscaler)

    m_es = compute_metrics_dict(case_name, pb_es.argmax(1), pf_es.argmax(1), pc_es,
                                yes["building"], yes["floor"], yes["coords"])
    m_va = compute_metrics_dict(case_name, pb_va.argmax(1), pf_va.argmax(1), pc_va,
                                yva["building"], yva["floor"], yva["coords"])

    row = {
        "group": group_name,
        "ablation": case_name,
        "rep_mode": rep_mode,
        "in_dim": X_tr.shape[1],
        "fold0_bld_acc": m_es["building_acc"],
        "fold0_flr_acc": m_es["floor_acc"],
        "fold0_coord_mean_m": m_es["coord_mean_m"],
        "fold0_evaal_m": m_es["combined_mean_m"],
        "val_bld_acc": m_va["building_acc"],
        "val_flr_acc": m_va["floor_acc"],
        "val_flr_f1": m_va["floor_macroF1"],
        "val_mae_avg_m": m_va["coord_mae_avg_m"],
        "val_coord_mean_m": m_va["coord_mean_m"],
        "val_coord_median_m": m_va["coord_median_m"],
        "val_coord_p90_m": m_va["coord_p90_m"],
        "val_evaal_m": m_va["combined_mean_m"],
        "best_epoch": hist["best_epoch"],
        "train_sec": hist["train_sec"],
    }
    print(
        f"[{group_name:14s}] {case_name:32s} | "
        f"F0 EvAAL={row['fold0_evaal_m']:5.2f}m | "
        f"Val Bld={row['val_bld_acc']*100:.2f}% Flr={row['val_flr_acc']*100:.2f}% "
        f"Mean={row['val_coord_mean_m']:5.2f}m Med={row['val_coord_median_m']:5.2f}m EvAAL={row['val_evaal_m']:5.2f}m"
    )
    return row


def main():
    P2_OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    train, val = load_raw(dedup=True)
    with open(FOLDS_JSON) as f:
        f0 = json.load(f)["folds"][0]

    pre_global = Preprocessor().fit(train)
    kept_waps = pre_global.kept_waps

    tr_df = train.iloc[f0["train_indices"]].reset_index(drop=True)
    es_df = train.iloc[f0["val_indices"]].reset_index(drop=True)

    print("=" * 80)
    print("PHASE 2 CONTROLLED ABLATION STUDIES (Fold 0 USERID Early-Stopping -> Official Val)")
    print("=" * 80)

    rows = []
    # 1. Input Representation Ablation
    rows.append(run_ablation_case("R1: Phase 1 StandardScaler", "Representation", tr_df, es_df, val, kept_waps, rep_mode="R1"))
    rows.append(run_ablation_case("R2: Positive-Powed RSSI", "Representation", tr_df, es_df, val, kept_waps, rep_mode="R2"))
    rows.append(run_ablation_case("R3-noRel: Powed+Mask+Agg", "Representation", tr_df, es_df, val, kept_waps, rep_mode="R3_no_rel"))
    rows.append(run_ablation_case("R3 (Full): +Scan-Relative Decay", "Representation", tr_df, es_df, val, kept_waps, rep_mode="R3"))

    # 2. Regularization & Mechanism Ablations (subtracting one component at a time from Full)
    rows.append(run_ablation_case("w/o Blind-Scan Filtering", "Regularization", tr_df, es_df, val, kept_waps, drop_blind=False))
    rows.append(run_ablation_case("w/o AP-Dropout & Jitter", "Regularization", tr_df, es_df, val, kept_waps, ap_drop_p=0.0, noise_std=0.0))
    rows.append(run_ablation_case("w/o Kendall Uncertainty Loss", "Mechanism", tr_df, es_df, val, kept_waps, use_kendall=False))
    rows.append(run_ablation_case("w/o FiLM Building Conditioning", "Mechanism", tr_df, es_df, val, kept_waps, use_film=False))
    rows.append(run_ablation_case("w/o Floor Class Reweighting", "Regularization", tr_df, es_df, val, kept_waps, use_floor_weights=False))

    df = pd.DataFrame(rows)
    out_csv = P2_OUT_DIR / "ablation_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✓ Saved ablation table to {out_csv} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()

