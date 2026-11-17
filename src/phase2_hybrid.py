"""Phase 2 ML + DL Hybrid Modeling & Nested OOF Validation.

Runs independently of PyTorch (loads saved DL OOF/Val probabilities and predictions
from `outputs/phase2/dl_predictions.npz`) to cleanly avoid macOS OpenMP runtime clashes
between libtorch and libxgboost.

Hybrid Architecture (`HierResMTL-Hybrid`):
  1. Stage 1 (DL Trunk): `HierResMTL` provides smooth device-invariant multitask
     posteriors (p_b^DL, p_f^DL) and parametric 2D coordinates (y_DL).
  2. Stage 2 (OOF-Calibrated Classification Fusion + Structural Mask):
     Combines `HierResMTL` with orthogonal tree boundary learners (`RF` / `R3-XGBoost`)
     and `R3-Cosine-kNN` via convex weights learned strictly by minimizing OOF log-loss,
     followed by the structural validity constraint (Floor 4 exists only in Building 2).
  3. Stage 3 (Hierarchical DL-Guided Adaptive Kernel Localization):
     Restricts R3-Cosine-kNN fingerprint retrieval to the Hybrid-predicted Building
     and adapts coordinate fusion weight w(s) dynamically based on local neighbor cosine
     similarity `max_sim` (high similarity -> local corridor kNN sharpness; low similarity
     -> smooth parametric DL manifold prevents large jump outliers).

Also logs intermediate Hybrid ablations (Naive Blend Strawman, Unconditioned R3-kNN,
Hybrid-Latent/Prob without Adaptive Gate) into `outputs/phase2/hybrid_ablation.csv`.
"""
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import log_loss
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, XGBRegressor

from config import OUT_DIR, FOLDS_JSON, RANDOM_STATE
from part2_preprocess import load_raw, Preprocessor, targets
from part3_features import FeatureBuilder
from phase2_features import build_dl_features
from phase2_train_cv import compute_metrics_dict

P2_OUT_DIR = OUT_DIR / "phase2"


def build_r3_matrix(train_df: pd.DataFrame, eval_df: pd.DataFrame, kept_waps: list):
    """Build device-invariant R3 features (single source of truth in phase2_features).

    `phase2_features` imports no PyTorch, so it is safe to reuse here without risking
    the macOS libtorch/libxgboost OpenMP clash this module avoids. Delegating keeps the
    tau / gamma constants from drifting between the DL and hybrid pipelines."""
    return build_dl_features(train_df, eval_df, kept_waps, mode="R3")


def smooth_probs(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    k = p.shape[1]
    return (1.0 - alpha) * p + alpha / k


def renorm_rows(p: np.ndarray) -> np.ndarray:
    """Rescale each row to sum to 1 so it stays a valid probability distribution.

    Needed after structurally masking a class (e.g. zeroing Floor-4 outside
    Building 2): the blended rows no longer sum to 1, which corrupts any
    downstream use of the probabilities (calibration / ECE, log-loss)."""
    s = p.sum(axis=1, keepdims=True)
    return p / np.clip(s, 1e-12, None)


def fit_prob_blend(probs_list: list, y_true: np.ndarray, n_classes: int) -> np.ndarray:
    """Optimize simplex convex weights on OOF log-loss."""
    m = len(probs_list)

    def _loss(logits):
        w = np.exp(logits - logits.max())
        w /= w.sum()
        p = sum(wi * pi for wi, pi in zip(w, probs_list))
        return log_loss(y_true, np.clip(p, 1e-6, 1.0), labels=list(range(n_classes)))

    res = minimize(_loss, np.zeros(m), method="Nelder-Mead")
    w = np.exp(res.x - res.x.max())
    return w / w.sum()


def hier_cosine_knn(
    X_r3_tr: np.ndarray,
    tb_tr: np.ndarray,
    tc_tr: np.ndarray,
    X_r3_ev: np.ndarray,
    pb_ev: np.ndarray,
    fallback_pc: np.ndarray,
    n_waps: int,
    k: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Building-conditioned R3 Cosine-kNN localization returning coords and max cosine similarity."""
    spec_dim = 2 * n_waps  # pow + scan-relative channels
    Xtr_n = X_r3_tr[:, :spec_dim].copy()
    Xev_n = X_r3_ev[:, :spec_dim].copy()
    Xtr_n /= np.linalg.norm(Xtr_n, axis=1, keepdims=True) + 1e-8
    Xev_n /= np.linalg.norm(Xev_n, axis=1, keepdims=True) + 1e-8

    preds = np.zeros((len(X_r3_ev), 2), dtype=np.float64)
    max_sims = np.zeros(len(X_r3_ev), dtype=np.float64)
    pred_b = pb_ev.argmax(axis=1)

    for b in range(3):
        ev_mask = pred_b == b
        if not ev_mask.any():
            continue
        tr_mask = tb_tr == b
        if not tr_mask.any():
            preds[ev_mask] = fallback_pc[ev_mask]
            continue

        sims = Xev_n[ev_mask] @ Xtr_n[tr_mask].T
        kk = min(k, sims.shape[1])
        top_idx = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        row_idx = np.arange(sims.shape[0])[:, None]
        top_sims = np.clip(sims[row_idx, top_idx], 0.0, 1.0)
        max_sims[ev_mask] = top_sims.max(axis=1)

        w = (top_sims ** 4) + 1e-6
        w /= w.sum(axis=1, keepdims=True)
        tr_coords = tc_tr[tr_mask]
        preds[ev_mask] = (w[:, :, None] * tr_coords[top_idx]).sum(axis=1)

    return preds, max_sims


def adaptive_gate(y_dl: np.ndarray, y_knn: np.ndarray, max_sim: np.ndarray, threshold: float = 0.5, max_w: float = 0.7) -> np.ndarray:
    """Similarity-gated fusion: high local cosine similarity favors kNN, low favors DL."""
    w = (np.clip((max_sim - threshold) / (1.0 - threshold), 0.0, 1.0) * max_w)[:, None]
    return (1.0 - w) * y_dl + w * y_knn


def main():
    P2_OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    dl = np.load(P2_OUT_DIR / "dl_predictions.npz")
    p1 = np.load(OUT_DIR / "predictions.npz")

    train, val = load_raw(dedup=True)
    with open(FOLDS_JSON) as f:
        folds = json.load(f)["folds"]

    pre = Preprocessor().fit(train)
    kept_waps = pre.kept_waps
    n_waps = len(kept_waps)

    fb = FeatureBuilder(pre, mode="A3").fit(train)
    X_eng_tr, X_eng_va = fb.transform(train), fb.transform(val)
    X_r3_tr, X_r3_va = build_r3_matrix(train, val, kept_waps)

    y_all, y_val = targets(train), targets(val)
    n_train, n_val = len(train), len(val)

    dl_oof_pb, dl_oof_pf, dl_oof_pc = dl["HierResMTL_oof_pb"], dl["HierResMTL_oof_pf"], dl["HierResMTL_oof_pc"]
    dl_val_pb, dl_val_pf, dl_val_pc = dl["HierResMTL_val_pb"], dl["HierResMTL_val_pf"], dl["HierResMTL_val_pc"]

    # ---- 1. Compute 5-fold OOF predictions for ML baselines & components ----
    print("Computing 5-fold USERID OOF predictions for ML components (RF, XGB, R3-kNN, R3-XGB)...")
    rf_oof_pb = np.zeros((n_train, 3), dtype=np.float32)
    rf_oof_pf = np.zeros((n_train, 5), dtype=np.float32)
    rf_oof_pc = np.zeros((n_train, 2), dtype=np.float64)

    xgb_oof_pb = np.zeros((n_train, 3), dtype=np.float32)
    xgb_oof_pf = np.zeros((n_train, 5), dtype=np.float32)
    xgb_oof_pc = np.zeros((n_train, 2), dtype=np.float64)

    knn_oof_pb = np.zeros((n_train, 3), dtype=np.float32)
    knn_oof_pf = np.zeros((n_train, 5), dtype=np.float32)
    knn_oof_pc = np.zeros((n_train, 2), dtype=np.float64)

    r3xgb_oof_pb = np.zeros((n_train, 3), dtype=np.float32)
    r3xgb_oof_pf = np.zeros((n_train, 5), dtype=np.float32)

    spec_dim = 2 * n_waps
    for fo in folds:
        ti, vi = np.array(fo["train_indices"]), np.array(fo["val_indices"])

        rf_kw = dict(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
        rf_oof_pb[vi] = smooth_probs(RandomForestClassifier(**rf_kw).fit(X_eng_tr[ti], y_all["building"][ti]).predict_proba(X_eng_tr[vi]), 0.02)
        rf_oof_pf[vi] = smooth_probs(RandomForestClassifier(**rf_kw).fit(X_eng_tr[ti], y_all["floor"][ti]).predict_proba(X_eng_tr[vi]), 0.02)
        rf_oof_pc[vi] = RandomForestRegressor(**rf_kw).fit(X_eng_tr[ti], y_all["coords"][ti]).predict(X_eng_tr[vi])

        xgb_kw = dict(n_estimators=250, max_depth=8, learning_rate=0.1, subsample=0.9, colsample_bytree=0.8,
                      tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE)
        xgb_oof_pb[vi] = XGBClassifier(eval_metric="mlogloss", **xgb_kw).fit(X_eng_tr[ti], y_all["building"][ti]).predict_proba(X_eng_tr[vi])
        xgb_oof_pf[vi] = XGBClassifier(eval_metric="mlogloss", **xgb_kw).fit(X_eng_tr[ti], y_all["floor"][ti]).predict_proba(X_eng_tr[vi])
        xgb_oof_pc[vi] = MultiOutputRegressor(XGBRegressor(**xgb_kw)).fit(X_eng_tr[ti], y_all["coords"][ti]).predict(X_eng_tr[vi])

        knn_oof_pb[vi] = smooth_probs(
            KNeighborsClassifier(n_neighbors=7, weights="distance", metric="cosine", n_jobs=-1)
            .fit(X_r3_tr[ti, :spec_dim], y_all["building"][ti])
            .predict_proba(X_r3_tr[vi, :spec_dim]),
            0.03,
        )
        knn_oof_pf[vi] = smooth_probs(
            KNeighborsClassifier(n_neighbors=9, weights="distance", metric="cosine", n_jobs=-1)
            .fit(X_r3_tr[ti, :spec_dim], y_all["floor"][ti])
            .predict_proba(X_r3_tr[vi, :spec_dim]),
            0.08,
        )
        pc_knn_raw, _ = hier_cosine_knn(X_r3_tr[ti], y_all["building"][ti], y_all["coords"][ti],
                                        X_r3_tr[vi], knn_oof_pb[vi], dl_oof_pc[vi], n_waps, k=7)
        knn_oof_pc[vi] = pc_knn_raw

        r3_kw = dict(n_estimators=250, max_depth=7, learning_rate=0.07, subsample=0.85, colsample_bytree=0.6,
                     tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE)
        r3xgb_oof_pb[vi] = smooth_probs(XGBClassifier(eval_metric="mlogloss", **r3_kw).fit(X_r3_tr[ti], y_all["building"][ti]).predict_proba(X_r3_tr[vi]), 0.03)
        r3xgb_oof_pf[vi] = smooth_probs(XGBClassifier(eval_metric="mlogloss", **r3_kw).fit(X_r3_tr[ti], y_all["floor"][ti]).predict_proba(X_r3_tr[vi]), 0.03)

    # ---- 2. Nested 5-Fold OOF Evaluation of Adaptive Hybrid ---------------
    hyb_oof_pb = np.zeros((n_train, 3), dtype=np.float32)
    hyb_oof_pf = np.zeros((n_train, 5), dtype=np.float32)
    hyb_oof_pc = np.zeros((n_train, 2), dtype=np.float64)
    hyb_oof_hknn = np.zeros((n_train, 2), dtype=np.float64)
    hyb_oof_sim = np.zeros(n_train, dtype=np.float64)

    for fo in folds:
        ti, vi = np.array(fo["train_indices"]), np.array(fo["val_indices"])
        wb = fit_prob_blend([dl_oof_pb[ti], knn_oof_pb[ti], rf_oof_pb[ti]], y_all["building"][ti], 3)
        hyb_oof_pb[vi] = wb[0] * dl_oof_pb[vi] + wb[1] * knn_oof_pb[vi] + wb[2] * rf_oof_pb[vi]

        wf = fit_prob_blend([dl_oof_pf[ti], knn_oof_pf[ti], r3xgb_oof_pf[ti]], y_all["floor"][ti], 5)
        hyb_oof_pf[vi] = wf[0] * dl_oof_pf[vi] + wf[1] * knn_oof_pf[vi] + wf[2] * r3xgb_oof_pf[vi]
        hyb_oof_pf[vi][hyb_oof_pb[vi].argmax(1) != 2, 4] = 0.0
        hyb_oof_pf[vi] = renorm_rows(hyb_oof_pf[vi])

        pc_hknn, sim_h = hier_cosine_knn(X_r3_tr[ti], y_all["building"][ti], y_all["coords"][ti],
                                         X_r3_tr[vi], hyb_oof_pb[vi], dl_oof_pc[vi], n_waps, k=7)
        hyb_oof_hknn[vi] = pc_hknn
        hyb_oof_sim[vi] = sim_h
        hyb_oof_pc[vi] = adaptive_gate(dl_oof_pc[vi], pc_hknn, sim_h)

    # ---- 3. Full-Train Fit for Official 1,111 Validation Set --------------
    wb_full = fit_prob_blend([dl_oof_pb, knn_oof_pb, rf_oof_pb], y_all["building"], 3)
    wf_full = fit_prob_blend([dl_oof_pf, knn_oof_pf, r3xgb_oof_pf], y_all["floor"], 5)
    weights_info = {
        "building_weights": {"HierResMTL": round(float(wb_full[0]), 4), "R3_Cosine_kNN": round(float(wb_full[1]), 4), "RandomForest": round(float(wb_full[2]), 4)},
        "floor_weights": {"HierResMTL": round(float(wf_full[0]), 4), "R3_Cosine_kNN": round(float(wf_full[1]), 4), "R3_XGBoost": round(float(wf_full[2]), 4)},
        "adaptive_gate": {"threshold": 0.5, "max_knn_weight": 0.7, "k_neighbors": 7},
    }
    with open(P2_OUT_DIR / "hybrid_weights.json", "w") as f:
        json.dump(weights_info, f, indent=2)
    print(f"Learned OOF Building weights: {weights_info['building_weights']}")
    print(f"Learned OOF Floor weights   : {weights_info['floor_weights']}")

    rf_b_full = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE).fit(X_eng_tr, y_all["building"])
    knn_b_full = KNeighborsClassifier(n_neighbors=7, weights="distance", metric="cosine", n_jobs=-1).fit(X_r3_tr[:, :spec_dim], y_all["building"])
    knn_f_full = KNeighborsClassifier(n_neighbors=9, weights="distance", metric="cosine", n_jobs=-1).fit(X_r3_tr[:, :spec_dim], y_all["floor"])
    r3_xf_full = XGBClassifier(n_estimators=250, max_depth=7, learning_rate=0.07, subsample=0.85, colsample_bytree=0.6,
                               tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE, eval_metric="mlogloss").fit(X_r3_tr, y_all["floor"])
    r3_xb_full = XGBClassifier(n_estimators=250, max_depth=7, learning_rate=0.07, subsample=0.85, colsample_bytree=0.6,
                               tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE, eval_metric="mlogloss").fit(X_r3_tr, y_all["building"])

    va_rf_pb = smooth_probs(rf_b_full.predict_proba(X_eng_va), 0.02)
    va_knn_pb = smooth_probs(knn_b_full.predict_proba(X_r3_va[:, :spec_dim]), 0.03)
    va_knn_pf = smooth_probs(knn_f_full.predict_proba(X_r3_va[:, :spec_dim]), 0.08)
    va_r3xgb_pb = smooth_probs(r3_xb_full.predict_proba(X_r3_va), 0.03)
    va_r3xgb_pf = smooth_probs(r3_xf_full.predict_proba(X_r3_va), 0.03)

    va_knn_pc, _ = hier_cosine_knn(X_r3_tr, y_all["building"], y_all["coords"],
                                   X_r3_va, va_knn_pb, dl_val_pc, n_waps, k=7)

    hyb_val_pb = wb_full[0] * dl_val_pb + wb_full[1] * va_knn_pb + wb_full[2] * va_rf_pb
    hyb_val_pf = wf_full[0] * dl_val_pf + wf_full[1] * va_knn_pf + wf_full[2] * va_r3xgb_pf
    hyb_val_pf[hyb_val_pb.argmax(1) != 2, 4] = 0.0
    hyb_val_pf = renorm_rows(hyb_val_pf)

    hyb_val_hknn, hyb_val_sim = hier_cosine_knn(X_r3_tr, y_all["building"], y_all["coords"],
                                                X_r3_va, hyb_val_pb, dl_val_pc, n_waps, k=7)
    hyb_val_pc = adaptive_gate(dl_val_pc, hyb_val_hknn, hyb_val_sim)

    # Naive Blend strawman (simple 0.5 RF + 0.5 HierResMTL)
    naive_oof_pc = 0.5 * rf_oof_pc + 0.5 * dl_oof_pc
    naive_val_pc = 0.5 * p1["RandomForest_coords"] + 0.5 * dl_val_pc

    # ---- 4. Save unified predictions + Hybrid Ablation table --------------
    hyb_ablation = [
        {**compute_metrics_dict("Pure DL (HierResMTL)", dl_val_pb.argmax(1), dl_val_pf.argmax(1), dl_val_pc,
                                y_val["building"], y_val["floor"], y_val["coords"]), "split": "Official Val"},
        {**compute_metrics_dict("R3-Cosine-kNN (Standalone)", va_knn_pb.argmax(1), va_knn_pf.argmax(1), va_knn_pc,
                                y_val["building"], y_val["floor"], y_val["coords"]), "split": "Official Val"},
        {**compute_metrics_dict("R3-XGBoost (Bld/Flr) + DL Coords", va_r3xgb_pb.argmax(1), va_r3xgb_pf.argmax(1), dl_val_pc,
                                y_val["building"], y_val["floor"], y_val["coords"]), "split": "Official Val"},
        {**compute_metrics_dict("Naive Blend Strawman (0.5 RF + 0.5 DL)", p1["RandomForest_building"], dl_val_pf.argmax(1), naive_val_pc,
                                y_val["building"], y_val["floor"], y_val["coords"]), "split": "Official Val"},
        {**compute_metrics_dict("Hybrid B/F Fusion + DL Coords", hyb_val_pb.argmax(1), hyb_val_pf.argmax(1), dl_val_pc,
                                y_val["building"], y_val["floor"], y_val["coords"]), "split": "Official Val"},
        {**compute_metrics_dict("Hybrid B/F + Hier-R3-kNN (No Gate)", hyb_val_pb.argmax(1), hyb_val_pf.argmax(1), hyb_val_hknn,
                                y_val["building"], y_val["floor"], y_val["coords"]), "split": "Official Val"},
        {**compute_metrics_dict("HierResMTL-Hybrid (Adaptive Gate)", hyb_val_pb.argmax(1), hyb_val_pf.argmax(1), hyb_val_pc,
                                y_val["building"], y_val["floor"], y_val["coords"]), "split": "Official Val"},
    ]
    pd.DataFrame(hyb_ablation).to_csv(P2_OUT_DIR / "hybrid_ablation.csv", index=False)

    # Save unified NPZ containing all models across OOF and Official Val
    payload = {k: dl[k] for k in dl.files}
    payload.update({
        "RandomForest_oof_pb": rf_oof_pb, "RandomForest_oof_pf": rf_oof_pf, "RandomForest_oof_pc": rf_oof_pc,
        "RandomForest_val_b": p1["RandomForest_building"], "RandomForest_val_f": p1["RandomForest_floor"], "RandomForest_val_c": p1["RandomForest_coords"],
        "XGBoost_oof_pb": xgb_oof_pb, "XGBoost_oof_pf": xgb_oof_pf, "XGBoost_oof_pc": xgb_oof_pc,
        "XGBoost_val_b": p1["XGBoost_building"], "XGBoost_val_f": p1["XGBoost_floor"], "XGBoost_val_c": p1["XGBoost_coords"],
        "R3_kNN_oof_pb": knn_oof_pb, "R3_kNN_oof_pf": knn_oof_pf, "R3_kNN_oof_pc": knn_oof_pc,
        "R3_kNN_val_pb": va_knn_pb, "R3_kNN_val_pf": va_knn_pf, "R3_kNN_val_pc": va_knn_pc,
        "NaiveBlend_oof_pc": naive_oof_pc, "NaiveBlend_val_pc": naive_val_pc,
        "Hybrid_oof_pb": hyb_oof_pb, "Hybrid_oof_pf": hyb_oof_pf, "Hybrid_oof_pc": hyb_oof_pc,
        "Hybrid_val_pb": hyb_val_pb, "Hybrid_val_pf": hyb_val_pf, "Hybrid_val_pc": hyb_val_pc,
        "Hybrid_val_hknn": hyb_val_hknn, "Hybrid_val_sim": hyb_val_sim,
    })
    np.savez(P2_OUT_DIR / "all_predictions.npz", **payload)

    print("\n=== HYBRID ABLATION SUMMARY (Official 1,111 Val) ===")
    for r in hyb_ablation:
        print(f"  {r['model']:40s} | Bld={r['building_acc']*100:6.2f}% Flr={r['floor_acc']*100:5.2f}% "
              f"Mean={r['coord_mean_m']:5.2f}m Med={r['coord_median_m']:5.2f}m P90={r['coord_p90_m']:5.2f}m "
              f"EvAAL={r['combined_mean_m']:5.2f}m")
    print(f"\n✓ Saved all_predictions.npz, hybrid_weights.json, hybrid_ablation.csv ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()

