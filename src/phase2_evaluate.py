"""Phase 2 Unified Evaluation, 5-Fold CV Statistics, Bootstrap CIs & Device Robustness.

Reads `outputs/phase2/all_predictions.npz` and computes:
  1. Official 1,111 Validation Set comprehensive comparison (`metrics_official_val.csv`)
     including explicit Coordinate MAE, Floor 4 recall, target-coupling breakdown, and EvAAL score.
  2. 5-Fold USERID GroupKFold CV Mean +/- Std (`metrics_5fold_cv.csv`).
  3. Statistical significance tests (`statistical_tests.csv`):
     - 1,000-resample Bootstrap 95% CIs on Mean/Median Euclidean error and EvAAL
     - Paired difference 95% CIs & McNemar exact floor test vs Phase 1 RF/XGBoost
  4. Device Hardware Shift Robustness (`robustness_by_phone.csv`):
     - Seen (Phones 13 & 14, 397 scans) vs Unseen (9 phones, 714 scans) + per-phone spread.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, recall_score, mean_absolute_error

from config import OUT_DIR, FOLDS_JSON, VAL_CSV, TRAIN_CSV, FLOOR_PENALTY_METERS, BUILDING_PENALTY_METERS
from part2_preprocess import load_raw
from phase2_train_cv import compute_metrics_dict

P2_OUT_DIR = OUT_DIR / "phase2"


def euclidean(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=1))


def mcnemar_pvalue(y_true, pred_a, pred_b):
    ok_a = pred_a == y_true
    ok_b = pred_b == y_true
    n01 = int((~ok_a & ok_b).sum())  # A wrong, B right
    n10 = int((ok_a & ~ok_b).sum())  # A right, B wrong
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    return float(binomtest(min(n01, n10), n=n, p=0.5, alternative="two-sided").pvalue), n01, n10


def bootstrap_ci(arr, n_boot=1000, stat_fn=np.mean, seed=42):
    rng = np.random.default_rng(seed)
    n = len(arr)
    vals = [stat_fn(arr[rng.integers(0, n, n)]) for _ in range(n_boot)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    d = np.load(P2_OUT_DIR / "all_predictions.npz")
    tb_tr, tf_tr, tc_tr = d["train_building"], d["train_floor"], d["train_coords"]
    tb_va, tf_va, tc_va = d["val_building"], d["val_floor"], d["val_coords"]

    with open(FOLDS_JSON) as f:
        folds = json.load(f)["folds"]

    # Map every model cleanly to its (val_b, val_f, val_c) and (oof_b, oof_f, oof_c)
    models = [
        ("Phase 1: RandomForest", d["RandomForest_val_b"], d["RandomForest_val_f"], d["RandomForest_val_c"],
         d["RandomForest_oof_pb"].argmax(1), d["RandomForest_oof_pf"].argmax(1), d["RandomForest_oof_pc"]),
        ("Phase 1: XGBoost", d["XGBoost_val_b"], d["XGBoost_val_f"], d["XGBoost_val_c"],
         d["XGBoost_oof_pb"].argmax(1), d["XGBoost_oof_pf"].argmax(1), d["XGBoost_oof_pc"]),
        # Per-task oracle: picks the best Phase-1 learner for EACH task independently
        # (RF building, XGB floor, RF coords). Not a single deployable model — it is an
        # upper-bound reference for what Phase-1 components could do if an oracle routed
        # each task to its strongest baseline. Labelled as such so it is never read as a
        # trained end-to-end baseline.
        ("Phase 1: Best-per-Task Oracle", d["RandomForest_val_b"], d["XGBoost_val_f"], d["RandomForest_val_c"],
         d["RandomForest_oof_pb"].argmax(1), d["XGBoost_oof_pf"].argmax(1), d["RandomForest_oof_pc"]),
        ("DL: SingleTaskMLP", d["SingleTaskMLP_val_pb"].argmax(1), d["SingleTaskMLP_val_pf"].argmax(1), d["SingleTaskMLP_val_pc"],
         d["SingleTaskMLP_oof_pb"].argmax(1), d["SingleTaskMLP_oof_pf"].argmax(1), d["SingleTaskMLP_oof_pc"]),
        ("DL: MTL-MLP (Phase 1 Plan)", d["MTL_MLP_val_pb"].argmax(1), d["MTL_MLP_val_pf"].argmax(1), d["MTL_MLP_val_pc"],
         d["MTL_MLP_oof_pb"].argmax(1), d["MTL_MLP_oof_pf"].argmax(1), d["MTL_MLP_oof_pc"]),
        ("DL: SAE-MTL (Nowicki 2017)", d["SAE_MTL_val_pb"].argmax(1), d["SAE_MTL_val_pf"].argmax(1), d["SAE_MTL_val_pc"],
         d["SAE_MTL_oof_pb"].argmax(1), d["SAE_MTL_oof_pf"].argmax(1), d["SAE_MTL_oof_pc"]),
        ("DL: CNN1D-MTL (CNNLoc 2019)", d["CNN1D_MTL_val_pb"].argmax(1), d["CNN1D_MTL_val_pf"].argmax(1), d["CNN1D_MTL_val_pc"],
         d["CNN1D_MTL_oof_pb"].argmax(1), d["CNN1D_MTL_oof_pf"].argmax(1), d["CNN1D_MTL_oof_pc"]),
        ("DL: HierResMTL (Proposed)", d["HierResMTL_val_pb"].argmax(1), d["HierResMTL_val_pf"].argmax(1), d["HierResMTL_val_pc"],
         d["HierResMTL_oof_pb"].argmax(1), d["HierResMTL_oof_pf"].argmax(1), d["HierResMTL_oof_pc"]),
        ("ML: R3-Cosine-kNN", d["R3_kNN_val_pb"].argmax(1), d["R3_kNN_val_pf"].argmax(1), d["R3_kNN_val_pc"],
         d["R3_kNN_oof_pb"].argmax(1), d["R3_kNN_oof_pf"].argmax(1), d["R3_kNN_oof_pc"]),
        ("Hybrid: Naive Blend (0.5 RF+DL)", d["RandomForest_val_b"], d["HierResMTL_val_pf"].argmax(1), d["NaiveBlend_val_pc"],
         d["RandomForest_oof_pb"].argmax(1), d["HierResMTL_oof_pf"].argmax(1), d["NaiveBlend_oof_pc"]),
        ("Hybrid: HierResMTL-Hybrid (Ours)", d["Hybrid_val_pb"].argmax(1), d["Hybrid_val_pf"].argmax(1), d["Hybrid_val_pc"],
         d["Hybrid_oof_pb"].argmax(1), d["Hybrid_oof_pf"].argmax(1), d["Hybrid_oof_pc"]),
    ]

    # ---- 1. Official Validation Table ------------------------------------
    val_rows = []
    for name, pb, pf, pc, _, _, _ in models:
        m = compute_metrics_dict(name, pb, pf, pc, tb_va, tf_va, tc_va)
        err = euclidean(pc, tc_va)
        both_ok = (pb == tb_va) & (pf == tf_va)
        f4_mask = tf_va == 4
        m["floor4_recall"] = round(float((pf[f4_mask] == 4).mean()), 4)
        m["coord_p95_m"] = round(float(np.percentile(err, 95)), 2)
        m["err_when_bf_ok_m"] = round(float(err[both_ok].mean()), 2) if both_ok.any() else 0.0
        m["err_when_bf_wrong_m"] = round(float(err[~both_ok].mean()), 2) if (~both_ok).any() else 0.0
        m["n_bf_wrong"] = int((~both_ok).sum())
        val_rows.append(m)

    df_val = pd.DataFrame(val_rows)
    df_val.to_csv(P2_OUT_DIR / "metrics_official_val.csv", index=False)

    print("=" * 95)
    print("OFFICIAL 1,111 VALIDATION SET SUMMARY (Phase 1 vs Phase 2 DL vs Hybrid)")
    print("=" * 95)
    cols_show = ["model", "building_acc", "floor_acc", "floor_macroF1", "floor4_recall",
                 "coord_mae_avg_m", "coord_mean_m", "coord_median_m", "coord_p90_m", "combined_mean_m"]
    print(df_val[cols_show].to_string(index=False))

    # ---- 2. 5-Fold USERID GroupKFold CV Table (Mean +/- Std across folds) ---
    cv_summary = []
    for name, _, _, _, ob, of, oc in models:
        fold_stats = []
        for fo in folds:
            vi = np.array(fo["val_indices"])
            fold_stats.append(compute_metrics_dict(name, ob[vi], of[vi], oc[vi], tb_tr[vi], tf_tr[vi], tc_tr[vi]))
        fdf = pd.DataFrame(fold_stats)
        oof_full = compute_metrics_dict(name, ob, of, oc, tb_tr, tf_tr, tc_tr)
        cv_summary.append({
            "model": name,
            "oof_building_acc": oof_full["building_acc"],
            "oof_floor_acc": oof_full["floor_acc"],
            "oof_floor_macroF1": oof_full["floor_macroF1"],
            "oof_mae_avg_m": oof_full["coord_mae_avg_m"],
            "oof_coord_mean_m": oof_full["coord_mean_m"],
            "oof_coord_median_m": oof_full["coord_median_m"],
            "oof_coord_p90_m": oof_full["coord_p90_m"],
            "oof_evaal_m": oof_full["combined_mean_m"],
            "cv_mean_bld_acc": round(float(fdf["building_acc"].mean()), 4),
            "cv_std_bld_acc": round(float(fdf["building_acc"].std()), 4),
            "cv_mean_flr_acc": round(float(fdf["floor_acc"].mean()), 4),
            "cv_std_flr_acc": round(float(fdf["floor_acc"].std()), 4),
            "cv_mean_coord_m": round(float(fdf["coord_mean_m"].mean()), 2),
            "cv_std_coord_m": round(float(fdf["coord_mean_m"].std()), 2),
            "cv_mean_evaal_m": round(float(fdf["combined_mean_m"].mean()), 2),
            "cv_std_evaal_m": round(float(fdf["combined_mean_m"].std()), 2),
        })
    df_cv = pd.DataFrame(cv_summary)
    df_cv.to_csv(P2_OUT_DIR / "metrics_5fold_cv.csv", index=False)

    print("\n" + "=" * 95)
    print("5-FOLD USERID GROUPKFOLD CV SUMMARY (19,300 Unseen-User OOF)")
    print("=" * 95)
    print(df_cv[["model", "oof_building_acc", "oof_floor_acc", "oof_coord_mean_m", "oof_coord_median_m", "oof_evaal_m", "cv_mean_evaal_m", "cv_std_evaal_m"]].to_string(index=False))

    # ---- 3. Bootstrap 95% CIs & Statistical Significance ------------------
    rf_err = euclidean(d["RandomForest_val_c"], tc_va)
    xgb_err = euclidean(d["XGBoost_val_c"], tc_va)
    dl_err = euclidean(d["HierResMTL_val_pc"], tc_va)
    hyb_err = euclidean(d["Hybrid_val_pc"], tc_va)

    rf_ev = rf_err + FLOOR_PENALTY_METERS * (d["RandomForest_val_f"] != tf_va) + BUILDING_PENALTY_METERS * (d["RandomForest_val_b"] != tb_va)
    dl_ev = dl_err + FLOOR_PENALTY_METERS * (d["HierResMTL_val_pf"].argmax(1) != tf_va) + BUILDING_PENALTY_METERS * (d["HierResMTL_val_pb"].argmax(1) != tb_va)
    hyb_ev = hyb_err + FLOOR_PENALTY_METERS * (d["Hybrid_val_pf"].argmax(1) != tf_va) + BUILDING_PENALTY_METERS * (d["Hybrid_val_pb"].argmax(1) != tb_va)

    stat_rows = []
    for label, err_arr, ev_arr in [
        ("Phase 1: RandomForest", rf_err, rf_ev),
        ("DL: HierResMTL", dl_err, dl_ev),
        ("Hybrid: HierResMTL-Hybrid", hyb_err, hyb_ev),
    ]:
        m_lo, m_hi = bootstrap_ci(err_arr, stat_fn=np.mean)
        md_lo, md_hi = bootstrap_ci(err_arr, stat_fn=np.median)
        ev_lo, ev_hi = bootstrap_ci(ev_arr, stat_fn=np.mean)
        stat_rows.append({
            "model": label,
            "mean_err_m": round(float(err_arr.mean()), 2), "mean_ci95_lo": round(m_lo, 2), "mean_ci95_hi": round(m_hi, 2),
            "median_err_m": round(float(np.median(err_arr)), 2), "med_ci95_lo": round(md_lo, 2), "med_ci95_hi": round(md_hi, 2),
            "evaal_mean_m": round(float(ev_arr.mean()), 2), "evaal_ci95_lo": round(ev_lo, 2), "evaal_ci95_hi": round(ev_hi, 2),
        })

    # Paired comparisons
    p_flr_xgb_hyb, n01_x, n10_x = mcnemar_pvalue(tf_va, d["XGBoost_val_f"], d["Hybrid_val_pf"].argmax(1))
    p_flr_rf_hyb, n01_r, n10_r = mcnemar_pvalue(tf_va, d["RandomForest_val_f"], d["Hybrid_val_pf"].argmax(1))
    p_flr_dl_hyb, n01_d, n10_d = mcnemar_pvalue(tf_va, d["HierResMTL_val_pf"].argmax(1), d["Hybrid_val_pf"].argmax(1))
    diff_rf_hyb = rf_err - hyb_err
    d_lo, d_hi = bootstrap_ci(diff_rf_hyb, stat_fn=np.mean)
    diff_dl_hyb = dl_err - hyb_err
    dl_lo, dl_hi = bootstrap_ci(diff_dl_hyb, stat_fn=np.mean)

    pd.DataFrame(stat_rows).to_csv(P2_OUT_DIR / "bootstrap_ci.csv", index=False)
    sig_summary = {
        "paired_rf_minus_hybrid_mean_m": round(float(diff_rf_hyb.mean()), 2),
        "paired_rf_minus_hybrid_ci95": [round(d_lo, 2), round(d_hi, 2)],
        "paired_dl_minus_hybrid_mean_m": round(float(diff_dl_hyb.mean()), 2),
        "paired_dl_minus_hybrid_ci95": [round(dl_lo, 2), round(dl_hi, 2)],
        "mcnemar_xgb_vs_hybrid_floor_p": round(p_flr_xgb_hyb, 6),
        "mcnemar_xgb_vs_hybrid_wins_hyb_vs_xgb": [n01_x, n10_x],
        "mcnemar_rf_vs_hybrid_floor_p": round(p_flr_rf_hyb, 6),
        "mcnemar_dl_vs_hybrid_floor_p": round(p_flr_dl_hyb, 6),
        "mcnemar_dl_vs_hybrid_wins_hyb_vs_dl": [n01_d, n10_d],
    }
    with open(P2_OUT_DIR / "significance_tests.json", "w") as f:
        json.dump(sig_summary, f, indent=2)
    print("\nStatistical Significance Summary:", json.dumps(sig_summary, indent=2))

    # ---- 4. Device Robustness (Seen vs Unseen Handsets) -------------------
    train_raw = pd.read_csv(TRAIN_CSV)
    val_raw = pd.read_csv(VAL_CSV)
    phones = val_raw["PHONEID"].values
    train_phones = set(train_raw["PHONEID"].unique().tolist())
    val_phones = sorted(np.unique(phones).tolist())
    shared_phones = set(val_phones).intersection(train_phones)
    seen_mask = np.isin(phones, list(shared_phones))
    unseen_mask = ~seen_mask

    rob_families = [
        ("RandomForest", d["RandomForest_val_f"], d["RandomForest_val_c"]),
        ("XGBoost", d["XGBoost_val_f"], d["XGBoost_val_c"]),
        ("HierResMTL (DL)", d["HierResMTL_val_pf"].argmax(1), d["HierResMTL_val_pc"]),
        ("HierResMTL-Hybrid", d["Hybrid_val_pf"].argmax(1), d["Hybrid_val_pc"]),
    ]

    rob_rows = []
    group_rows = []
    for fam, pf, pc in rob_families:
        err = euclidean(pc, tc_va)
        for ph in val_phones:
            m = phones == ph
            rob_rows.append({
                "family": fam,
                "phone": int(ph),
                "device_status": "Seen (shared)" if ph in shared_phones else "Unseen",
                "n": int(m.sum()),
                "floor_acc": round(float((pf[m] == tf_va[m]).mean()), 4),
                "coord_mean_m": round(float(err[m].mean()), 2),
                "coord_median_m": round(float(np.median(err[m])), 2),
                "coord_p95_m": round(float(np.percentile(err[m], 95)), 2),
            })
        for grp_name, mask in [("Seen (Phones 13 & 14)", seen_mask), ("Unseen (9 phones)", unseen_mask)]:
            group_rows.append({
                "family": fam,
                "group": grp_name,
                "n_samples": int(mask.sum()),
                "floor_acc": round(float((pf[mask] == tf_va[mask]).mean()), 4),
                "coord_mean_m": round(float(err[mask].mean()), 2),
                "coord_median_m": round(float(np.median(err[mask])), 2),
            })

    pd.DataFrame(rob_rows).to_csv(P2_OUT_DIR / "robustness_by_phone.csv", index=False)
    df_grp = pd.DataFrame(group_rows)
    df_grp.to_csv(P2_OUT_DIR / "robustness_seen_unseen.csv", index=False)

    print("\n" + "=" * 75)
    print("DEVICE ROBUSTNESS: SEEN (397) vs UNSEEN (714) HANDSETS")
    print("=" * 75)
    print(df_grp.to_string(index=False))
    print("\n✓ Saved all evaluation & robustness CSVs in outputs/phase2/")


if __name__ == "__main__":
    main()

