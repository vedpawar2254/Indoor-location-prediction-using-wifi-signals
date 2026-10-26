#!/usr/bin/env python3
"""Feature Ablation & kNN Baseline Evaluation.

Member 2 (Features & Representation Lead) Deliverable.
Evaluates the three RSSI feature representations:
  A1: Scaled RSSI only (465 features)
  A2: Scaled RSSI + Absence indicators (930 features)
  A3: Scaled RSSI + Absence + Aggregates (936 features, full set)

Evaluates on kNN and baseline models, computing:
  - Building & Floor accuracy
  - Coordinate MAE (Longitude, Latitude, Average)
  - Euclidean positioning error (Mean, Median, P95)
  - Runtime (seconds)

Saves results to outputs/feature_ablation_results.csv and outputs/knn_baseline_results.csv.
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import OUT_DIR, RANDOM_STATE
from part2_preprocess import Preprocessor, load_raw, targets
from part3_features import FeatureBuilder


def euclidean(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=1))


def run_ablation_and_knn(k_neighbors: int = 5):
    print("=" * 70)
    print("Feature Representation Ablation & kNN Baselines (Member 2)")
    print("=" * 70)

    train, val = load_raw(dedup=True)
    ytr = targets(train)
    yva = targets(val)

    pre = Preprocessor().fit(train)

    modes = [
        ("A1_raw_scaled_rssi", "A1", 465),
        ("A2_rssi_plus_absence", "A2", 930),
        ("A3_full_engineered", "A3", 936),
    ]

    results = []

    for name, mode_code, expected_feats in modes:
        print(f"\n--- Testing Representation {name} (Mode: {mode_code}) ---")
        t0 = time.time()
        fb = FeatureBuilder(pre, mode=mode_code).fit(train)
        Xtr = fb.transform(train)
        Xva = fb.transform(val)
        prep_time = time.time() - t0
        print(f"  Feature shape: train {Xtr.shape}, val {Xva.shape} (built in {prep_time:.2f}s)")

        # 1. kNN Baselines
        print(f"  Fitting kNN (k={k_neighbors}) ...")
        t_fit = time.time()
        knn_bld = KNeighborsClassifier(n_neighbors=k_neighbors, weights="distance", n_jobs=-1)
        knn_flr = KNeighborsClassifier(n_neighbors=k_neighbors, weights="distance", n_jobs=-1)
        knn_crd = KNeighborsRegressor(n_neighbors=k_neighbors, weights="distance", n_jobs=-1)

        knn_bld.fit(Xtr, ytr["building"])
        knn_flr.fit(Xtr, ytr["floor"])
        knn_crd.fit(Xtr, ytr["coords"])
        fit_time = time.time() - t_fit

        pb = knn_bld.predict(Xva)
        pf = knn_flr.predict(Xva)
        pc = knn_crd.predict(Xva)

        b_acc = accuracy_score(yva["building"], pb)
        f_acc = accuracy_score(yva["floor"], pf)

        mae_lon = mean_absolute_error(yva["coords"][:, 0], pc[:, 0])
        mae_lat = mean_absolute_error(yva["coords"][:, 1], pc[:, 1])
        mae_avg = (mae_lon + mae_lat) / 2.0

        err = euclidean(pc, yva["coords"])
        mean_err = err.mean()
        med_err = np.median(err)
        p95_err = np.percentile(err, 95)

        print(f"  Building Acc : {b_acc * 100:.2f}%")
        print(f"  Floor Acc    : {f_acc * 100:.2f}%")
        print(f"  Coord MAE    : Lon {mae_lon:.2f} m, Lat {mae_lat:.2f} m (Avg {mae_avg:.2f} m)")
        print(f"  Pos Error    : Mean {mean_err:.2f} m, Median {med_err:.2f} m, P95 {p95_err:.2f} m")
        print(f"  Fit/Eval Time: {fit_time:.2f}s")

        results.append({
            "representation": name,
            "mode": mode_code,
            "n_features": Xtr.shape[1],
            "model": f"kNN (k={k_neighbors})",
            "building_acc": round(b_acc, 4),
            "floor_acc": round(f_acc, 4),
            "mae_lon_m": round(mae_lon, 2),
            "mae_lat_m": round(mae_lat, 2),
            "mae_avg_m": round(mae_avg, 2),
            "coord_mean_m": round(mean_err, 2),
            "coord_median_m": round(float(med_err), 2),
            "coord_p95_m": round(float(p95_err), 2),
            "runtime_sec": round(fit_time, 2),
        })

    df_results = pd.DataFrame(results)
    out_csv = OUT_DIR / "feature_ablation_results.csv"
    df_results.to_csv(out_csv, index=False)
    knn_csv = OUT_DIR / "knn_baseline_results.csv"
    df_results.to_csv(knn_csv, index=False)

    print("\n" + "=" * 70)
    print("Summary Table of Feature Representations (Member 2)")
    print("=" * 70)
    print(df_results[["representation", "n_features", "building_acc", "floor_acc", "mae_avg_m", "coord_mean_m", "coord_p95_m", "runtime_sec"]].to_string(index=False))
    print(f"\n✓ Saved results to {out_csv} and {knn_csv}")
    return df_results


if __name__ == "__main__":
    run_ablation_and_knn()
