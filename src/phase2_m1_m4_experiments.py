"""Phase 2 Member 1 & Member 4 Targeted Experiments.

Satisfies specific rubric requirements:
  - M1: PyTorch noise augmentation (2 dBm, 4 dBm) and AP dropout (5%, 10%) grid.
        (Regularization comparisons — Dropout / Weight Decay / Kendall / FiLM —
        live in phase2_ablations.py, not here.)
  - M4: Learning curves (25%, 50%, 75%, 100%), Floor ECE / calibration,
        3-seed variance on the single-fold HierResMTL model, and worst-error analysis.

Outputs saved in outputs/phase2/:
  - m1_noise_dropout_grid.csv
  - m4_learning_curve.csv
  - m4_calibration_ece.csv  (+ m4_calibration_bins.json)
  - m4_single_model_seed_variance.csv
  - m4_worst_errors.csv
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


def compute_ece(probs, true_labels, n_bins=10):
    """Computes Expected Calibration Error (ECE) and bin statistics."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == true_labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += float(np.abs(acc_in_bin - conf_in_bin) * prop_in_bin)
            bin_stats.append({
                "bin": int(i),
                "bin_lower": float(bin_boundaries[i]),
                "bin_upper": float(bin_boundaries[i+1]),
                "prop": float(prop_in_bin),
                "accuracy": float(acc_in_bin),
                "confidence": float(conf_in_bin)
            })
    return float(ece), bin_stats


def run_m1_noise_dropout(tr_df, es_df, val, kept_waps):
    """M1: Grid of noise augmentation (std=2, 4 dBm) and AP dropout (5%, 10%)."""
    print("\n--- M1: Noise Augmentation & AP Dropout Grid ---")
    tr_use = filter_non_blind(tr_df)
    X_tr, X_es = build_dl_features(tr_use, es_df, kept_waps, mode="R3")
    _, X_va = build_dl_features(tr_use, val, kept_waps, mode="R3")

    wid_tr = extract_strongest_wap_idx(tr_use, kept_waps)
    wid_es = extract_strongest_wap_idx(es_df, kept_waps)
    wid_va = extract_strongest_wap_idx(val, kept_waps)
    ytr, yes, yva = targets(tr_use), targets(es_df), targets(val)

    # Convert dBm to normalized space (range ~0 to 105 dBm -> std in normalized ~ std_dbm / 105)
    # std 2 dBm ~ 0.019; std 4 dBm ~ 0.038
    grid = [
        {"noise_dbm": 0, "noise_std": 0.00, "ap_drop_pct": 0, "ap_drop_p": 0.00},
        {"noise_dbm": 2, "noise_std": 0.019, "ap_drop_pct": 5, "ap_drop_p": 0.05},
        {"noise_dbm": 2, "noise_std": 0.019, "ap_drop_pct": 10, "ap_drop_p": 0.10},
        {"noise_dbm": 4, "noise_std": 0.038, "ap_drop_pct": 5, "ap_drop_p": 0.05},
        {"noise_dbm": 4, "noise_std": 0.038, "ap_drop_pct": 10, "ap_drop_p": 0.10},
        {"noise_dbm": 2.1, "noise_std": 0.02, "ap_drop_pct": 12, "ap_drop_p": 0.12}, # Default tuned
    ]

    records = []
    for g in grid:
        tag = f"noise_{g['noise_dbm']}dBm_apdrop_{g['ap_drop_pct']}pct"
        model = HierResMTL(X_tr.shape[1], len(kept_waps))
        model, cscaler, hist = train_dl_model(
            model,
            X_tr, wid_tr, ytr["building"], ytr["floor"], ytr["coords"],
            X_es, wid_es, yes["building"], yes["floor"], yes["coords"],
            n_waps=len(kept_waps),
            rep_mode="R3",
            epochs=35,
            ap_drop_p=g["ap_drop_p"],
            noise_std=g["noise_std"],
            seed=RANDOM_STATE,
        )
        pb_es, pf_es, pc_es, _ = predict_dl_model(model, X_es, wid_es, cscaler)
        pb_va, pf_va, pc_va, _ = predict_dl_model(model, X_va, wid_va, cscaler)

        m_es = compute_metrics_dict(tag, pb_es.argmax(1), pf_es.argmax(1), pc_es,
                                    yes["building"], yes["floor"], yes["coords"])
        m_va = compute_metrics_dict(tag, pb_va.argmax(1), pf_va.argmax(1), pc_va,
                                    yva["building"], yva["floor"], yva["coords"])

        rec = {
            "noise_dbm": g["noise_dbm"],
            "noise_std_norm": g["noise_std"],
            "ap_dropout_pct": g["ap_drop_pct"],
            "fold0_coord_mean_m": m_es["coord_mean_m"],
            "val_bld_acc": m_va["building_acc"],
            "val_flr_acc": m_va["floor_acc"],
            "val_flr_f1": m_va["floor_macroF1"],
            "val_coord_mean_m": m_va["coord_mean_m"],
            "val_coord_median_m": m_va["coord_median_m"],
            "val_evaal_m": m_va["combined_mean_m"],
            "best_epoch": hist["best_epoch"],
        }
        records.append(rec)
        print(f"[{tag:28s}] Val Flr={rec['val_flr_acc']*100:.2f}% Mean={rec['val_coord_mean_m']:5.2f}m EvAAL={rec['val_evaal_m']:5.2f}m")

    df = pd.DataFrame(records)
    df.to_csv(P2_OUT_DIR / "m1_noise_dropout_grid.csv", index=False)
    print("✓ Saved m1_noise_dropout_grid.csv")


def run_m4_learning_curve(tr_df, es_df, val, kept_waps):
    """M4: Learning curve across 25%, 50%, 75%, 100% of grouped training users."""
    print("\n--- M4: Learning Curves (25%, 50%, 75%, 100% Training Users) ---")
    tr_use = filter_non_blind(tr_df)
    users = tr_use["USERID"].unique()
    # Isolated RNG so a prior global np.random.seed() call elsewhere can't change
    # which users land in each learning-curve fraction.
    rng = np.random.default_rng(RANDOM_STATE)
    shuffled_users = rng.permutation(users)

    fractions = [0.25, 0.50, 0.75, 1.00]
    records = []

    for frac in fractions:
        n_u = max(1, int(len(users) * frac))
        sub_users = set(shuffled_users[:n_u])
        sub_tr = tr_use[tr_use["USERID"].isin(sub_users)].reset_index(drop=True)

        X_tr, X_es = build_dl_features(sub_tr, es_df, kept_waps, mode="R3")
        _, X_va = build_dl_features(sub_tr, val, kept_waps, mode="R3")

        wid_tr = extract_strongest_wap_idx(sub_tr, kept_waps)
        wid_es = extract_strongest_wap_idx(es_df, kept_waps)
        wid_va = extract_strongest_wap_idx(val, kept_waps)
        ytr, yes, yva = targets(sub_tr), targets(es_df), targets(val)

        model = HierResMTL(X_tr.shape[1], len(kept_waps))
        model, cscaler, hist = train_dl_model(
            model,
            X_tr, wid_tr, ytr["building"], ytr["floor"], ytr["coords"],
            X_es, wid_es, yes["building"], yes["floor"], yes["coords"],
            n_waps=len(kept_waps),
            rep_mode="R3",
            epochs=35,
            seed=RANDOM_STATE,
        )
        pb_es, pf_es, pc_es, _ = predict_dl_model(model, X_es, wid_es, cscaler)
        pb_va, pf_va, pc_va, _ = predict_dl_model(model, X_va, wid_va, cscaler)

        m_es = compute_metrics_dict(f"{frac*100:.0f}%", pb_es.argmax(1), pf_es.argmax(1), pc_es,
                                    yes["building"], yes["floor"], yes["coords"])
        m_va = compute_metrics_dict(f"{frac*100:.0f}%", pb_va.argmax(1), pf_va.argmax(1), pc_va,
                                    yva["building"], yva["floor"], yva["coords"])

        rec = {
            "fraction": frac,
            "train_users": n_u,
            "train_samples": len(sub_tr),
            "fold0_flr_acc": m_es["floor_acc"],
            "fold0_coord_mean_m": m_es["coord_mean_m"],
            "fold0_evaal_m": m_es["combined_mean_m"],
            "val_bld_acc": m_va["building_acc"],
            "val_flr_acc": m_va["floor_acc"],
            "val_flr_f1": m_va["floor_macroF1"],
            "val_coord_mean_m": m_va["coord_mean_m"],
            "val_coord_median_m": m_va["coord_median_m"],
            "val_evaal_m": m_va["combined_mean_m"],
        }
        records.append(rec)
        print(f"[{frac*100:3.0f}% ({len(sub_tr):5d} rows)] Val Flr={rec['val_flr_acc']*100:.2f}% Mean={rec['val_coord_mean_m']:5.2f}m EvAAL={rec['val_evaal_m']:5.2f}m")

    df = pd.DataFrame(records)
    df.to_csv(P2_OUT_DIR / "m4_learning_curve.csv", index=False)
    print("✓ Saved m4_learning_curve.csv")


def run_m4_calibration(val):
    """M4: Expected Calibration Error (ECE) for Floor Classification across models."""
    print("\n--- M4: Floor Calibration & ECE Analysis ---")
    data = np.load(P2_OUT_DIR / "all_predictions.npz")
    y_true_floor = val["FLOOR"].values

    models = [
        ("DL: MTL-MLP", data["MTL_MLP_val_pf"]),
        ("DL: HierResMTL", data["HierResMTL_val_pf"]),
        ("ML: R3-Cosine-kNN", data["R3_kNN_val_pf"]),
        ("Hybrid: HierResMTL-Hybrid", data["Hybrid_val_pf"]),
    ]

    records = []
    full_bin_dict = {}

    for name, probs in models:
        ece, bstats = compute_ece(probs, y_true_floor, n_bins=10)
        top1_conf = float(np.mean(np.max(probs, axis=1)))
        acc = float(np.mean(np.argmax(probs, axis=1) == y_true_floor))
        records.append({
            "model": name,
            "ece": round(ece, 4),
            "mean_confidence": round(top1_conf, 4),
            "accuracy": round(acc, 4),
            "overconfidence_bias": round(top1_conf - acc, 4),
        })
        full_bin_dict[name] = bstats
        print(f"{name:26s} | ECE: {ece:.4f} | Conf: {top1_conf*100:.1f}% | Acc: {acc*100:.1f}% | Gap: {(top1_conf-acc)*100:+.1f}%")

    df = pd.DataFrame(records)
    df.to_csv(P2_OUT_DIR / "m4_calibration_ece.csv", index=False)
    with open(P2_OUT_DIR / "m4_calibration_bins.json", "w") as f:
        json.dump(full_bin_dict, f, indent=2)
    print("✓ Saved m4_calibration_ece.csv and m4_calibration_bins.json")


def run_m4_worst_errors(val, train):
    """M4: Documented Failure Cases / Worst 20 Error Review."""
    print("\n--- M4: Worst-Error Review ---")
    data = np.load(P2_OUT_DIR / "all_predictions.npz")
    true_bld = val["BUILDINGID"].values
    true_flr = val["FLOOR"].values
    true_crd = val[["LONGITUDE", "LATITUDE"]].values
    phones = val["PHONEID"].values
    users = val["USERID"].values

    # Dynamically derive which handsets were seen in training (matches phase2_evaluate.py),
    # instead of hardcoding {13, 14} which only holds for the default UJIIndoorLoc split.
    train_phones = set(train["PHONEID"].unique().tolist())
    val_phones = set(np.unique(phones).tolist())
    seen_phones = val_phones & train_phones

    hyb_bld = np.argmax(data["Hybrid_val_pb"], axis=1)
    hyb_flr = np.argmax(data["Hybrid_val_pf"], axis=1)
    hyb_crd = data["Hybrid_val_pc"]

    eucl = np.sqrt(((true_crd - hyb_crd) ** 2).sum(axis=1))
    evaal = eucl + 50.0 * (true_bld != hyb_bld) + 4.0 * (true_flr != hyb_flr)

    worst_idx = np.argsort(evaal)[::-1][:20]
    records = []

    for rank, idx in enumerate(worst_idx, 1):
        tb, pb = int(true_bld[idx]), int(hyb_bld[idx])
        tf, pf = int(true_flr[idx]), int(hyb_flr[idx])
        tx, ty = true_crd[idx]
        px, py = hyb_crd[idx]
        err_m = float(eucl[idx])
        ev_m = float(evaal[idx])
        phone = int(phones[idx])
        user = int(users[idx])

        # Identify reason
        reasons = []
        if tb != pb:
            reasons.append(f"Wrong Bld ({tb}->{pb})")
        if tf != pf:
            reasons.append(f"Wrong Floor ({tf}->{pf})")
        if phone not in seen_phones:
            reasons.append(f"Unseen Phone {phone}")
        if err_m > 25.0:
            reasons.append("Extreme Distance")
        if not reasons:
            reasons.append("High multipath / edge corridor")

        records.append({
            "rank": rank,
            "sample_index": int(idx),
            "phone_id": phone,
            "user_id": user,
            "is_unseen_phone": phone not in seen_phones,
            "true_building": tb,
            "pred_building": pb,
            "true_floor": tf,
            "pred_floor": pf,
            "euclidean_error_m": round(err_m, 2),
            "evaal_score_m": round(ev_m, 2),
            "failure_mode": "; ".join(reasons),
        })

    df = pd.DataFrame(records)
    df.to_csv(P2_OUT_DIR / "m4_worst_errors.csv", index=False)
    print("✓ Saved m4_worst_errors.csv")


def run_m4_three_seeds(tr_df, es_df, val, kept_waps):
    """M4: Seed variance of a SINGLE-FOLD standalone HierResMTL on Official Val.

    NOTE: This characterizes the run-to-run variance of one HierResMTL trained on
    Fold-0 only (~12-13 m mean error). It is NOT the final HierResMTL-Hybrid, which
    is a 5-fold CV ensemble fused with RF/XGB/kNN (~6.8 m). The two numbers describe
    different estimators and are expected to differ; see the report for the gap.
    """
    print("\n--- M4: Seed Variance (single-fold standalone HierResMTL) -> Official Val ---")
    tr_use = filter_non_blind(tr_df)
    X_tr, X_es = build_dl_features(tr_use, es_df, kept_waps, mode="R3")
    _, X_va = build_dl_features(tr_use, val, kept_waps, mode="R3")

    wid_tr = extract_strongest_wap_idx(tr_use, kept_waps)
    wid_es = extract_strongest_wap_idx(es_df, kept_waps)
    wid_va = extract_strongest_wap_idx(val, kept_waps)
    ytr, yes, yva = targets(tr_use), targets(es_df), targets(val)

    seeds = [42, 123, 999]
    records = []
    for s in seeds:
        model = HierResMTL(X_tr.shape[1], len(kept_waps))
        model, cscaler, hist = train_dl_model(
            model,
            X_tr, wid_tr, ytr["building"], ytr["floor"], ytr["coords"],
            X_es, wid_es, yes["building"], yes["floor"], yes["coords"],
            n_waps=len(kept_waps),
            rep_mode="R3",
            epochs=35,
            seed=s,
        )
        pb_va, pf_va, pc_va, _ = predict_dl_model(model, X_va, wid_va, cscaler)
        m_va = compute_metrics_dict(f"seed_{s}", pb_va.argmax(1), pf_va.argmax(1), pc_va,
                                    yva["building"], yva["floor"], yva["coords"])
        records.append({
            "seed": s,
            "bld_acc": m_va["building_acc"],
            "flr_acc": m_va["floor_acc"],
            "flr_f1": m_va["floor_macroF1"],
            "coord_mean_m": m_va["coord_mean_m"],
            "coord_median_m": m_va["coord_median_m"],
            "coord_p90_m": m_va["coord_p90_m"],
            "evaal_m": m_va["combined_mean_m"],
        })
        print(f"[Seed {s:3d}] Flr={m_va['floor_acc']*100:.2f}% Mean={m_va['coord_mean_m']:.2f}m EvAAL={m_va['combined_mean_m']:.2f}m")

    df = pd.DataFrame(records)
    num_cols = ["bld_acc", "flr_acc", "flr_f1", "coord_mean_m", "coord_median_m", "coord_p90_m", "evaal_m"]
    means = df[num_cols].mean().to_dict()
    means["seed"] = "mean"
    stds = df[num_cols].std().to_dict()
    stds["seed"] = "std"
    df = pd.concat([df, pd.DataFrame([means, stds])], ignore_index=True)
    df.to_csv(P2_OUT_DIR / "m4_single_model_seed_variance.csv", index=False)
    print("✓ Saved m4_single_model_seed_variance.csv (single-fold HierResMTL, not the hybrid)")


def main():
    train, val = load_raw(dedup=True)
    with open(FOLDS_JSON) as f:
        f0 = json.load(f)["folds"][0]

    pre_global = Preprocessor().fit(train)
    kept_waps = pre_global.kept_waps

    tr_df = train.iloc[f0["train_indices"]].reset_index(drop=True)
    es_df = train.iloc[f0["val_indices"]].reset_index(drop=True)

    t0 = time.time()
    if not (P2_OUT_DIR / "m1_noise_dropout_grid.csv").exists():
        run_m1_noise_dropout(tr_df, es_df, val, kept_waps)
    else:
        print("✓ m1_noise_dropout_grid.csv exists, skipping.")

    if not (P2_OUT_DIR / "m4_learning_curve.csv").exists():
        run_m4_learning_curve(tr_df, es_df, val, kept_waps)
    else:
        print("✓ m4_learning_curve.csv exists, skipping.")

    # Calibration and worst-error review read all_predictions.npz (written by
    # phase2_hybrid.py). Skip gracefully if the hybrid stage has not run yet.
    if (P2_OUT_DIR / "all_predictions.npz").exists():
        run_m4_calibration(val)
        run_m4_worst_errors(val, train)
    else:
        print("! all_predictions.npz not found — run phase2_hybrid.py first; "
              "skipping M4 calibration and worst-error review.")

    if not (P2_OUT_DIR / "m4_single_model_seed_variance.csv").exists():
        run_m4_three_seeds(tr_df, es_df, val, kept_waps)
    else:
        print("✓ m4_single_model_seed_variance.csv exists, skipping.")
    print(f"\n✓ Completed all targeted M1 & M4 experiments in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
