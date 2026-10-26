"""Part 5 of the ML lifecycle: Evaluation.

Reads outputs/predictions.npz (written by Part 4) and computes the full metric
suite on the SUPPLIED validation set. No retraining here -- evaluation only.

Metrics reported per model family:
  classification
    building accuracy
    floor accuracy
    floor macro-F1 + per-class report
  regression (coordinates)
    coordinate MAE (LONGITUDE, LATITUDE, and their average) explicitly in meters
    coordinate positioning error = Euclidean distance in meters
    mean / median / 75th / 90th percentile
  hierarchical error (competition / EvAAL convention)
    custom combined score = Euclidean_2d + 4.0 * (floor_wrong) + 50.0 * (building_wrong)
    (Using standard binary indicator penalty: 4 m for incorrect floor, 50 m for incorrect building)
  breakdown
    coordinate error split by whether building/floor were predicted correctly.

All numbers are printed and also written to outputs/metrics.csv for the report.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, mean_absolute_error,
)

from config import OUT_DIR, FLOOR_PENALTY_METERS, BUILDING_PENALTY_METERS

FAMILIES = ["RandomForest", "XGBoost"]


def euclidean(pred_xy, true_xy):
    return np.sqrt(((pred_xy - true_xy) ** 2).sum(axis=1))


def score_row(name, pb, pf, pc, tb, tf, tc, verbose=True):
    """Compute the full metric row for one set of predictions."""
    b_acc = accuracy_score(tb, pb)
    f_acc = accuracy_score(tf, pf)
    f_f1 = f1_score(tf, pf, average="macro")

    # Coordinate MAE per axis explicitly
    mae_lon = mean_absolute_error(tc[:, 0], pc[:, 0])
    mae_lat = mean_absolute_error(tc[:, 1], pc[:, 1])
    mae_avg = (mae_lon + mae_lat) / 2.0

    # Euclidean distance
    err = euclidean(pc, tc)

    # Custom combined score (indicator penalty: 4 m for wrong floor, 50 m for wrong building)
    combined = err + FLOOR_PENALTY_METERS * (pf != tf) + BUILDING_PENALTY_METERS * (pb != tb)

    if verbose:
        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)
        print(f"  building accuracy : {b_acc:.4f} ({b_acc*100:.2f}%)")
        print(f"  floor    accuracy : {f_acc:.4f} ({f_acc*100:.2f}%)   macro-F1 {f_f1:.4f}")
        print(f"  coordinate MAE (m): lon {mae_lon:6.2f}  lat {mae_lat:6.2f}  avg {mae_avg:6.2f}")
        print(f"  coord error (m)   : mean {err.mean():6.2f}  median {np.median(err):6.2f}"
              f"  p75 {np.percentile(err,75):6.2f}  p90 {np.percentile(err,90):6.2f}")
        print(f"  combined score (m): mean {combined.mean():6.2f}  median {np.median(combined):6.2f}")
        print("\n  floor per-class report:")
        print(classification_report(tf, pf, digits=3,
                                    zero_division=0, target_names=[f"F{i}" for i in range(5)]))
        both_ok = (pb == tb) & (pf == tf)
        print(f"  coord error | building+floor correct : "
              f"mean {err[both_ok].mean():.2f}m  (n={both_ok.sum()})")
        if (~both_ok).any():
            print(f"  coord error | something misclassified: "
                  f"mean {err[~both_ok].mean():.2f}m  (n={(~both_ok).sum()})")

    return dict(
        model=name,
        building_acc=round(b_acc, 4),
        floor_acc=round(f_acc, 4),
        floor_macroF1=round(f_f1, 4),
        coord_mae_lon_m=round(mae_lon, 2),
        coord_mae_lat_m=round(mae_lat, 2),
        coord_mae_avg_m=round(mae_avg, 2),
        coord_mean_m=round(err.mean(), 2),
        coord_median_m=round(float(np.median(err)), 2),
        coord_p75_m=round(float(np.percentile(err, 75)), 2),
        coord_p90_m=round(float(np.percentile(err, 90)), 2),
        combined_mean_m=round(combined.mean(), 2),
        combined_median_m=round(float(np.median(combined)), 2),
    )


def main():
    d = np.load(OUT_DIR / "predictions.npz")
    tb = d["truth_building"]
    tf = d["truth_floor"]
    tc = d["truth_coords"]

    rows = []
    for fam in FAMILIES:
        rows.append(score_row(fam, d[f"{fam}_building"], d[f"{fam}_floor"],
                              d[f"{fam}_coords"], tb, tf, tc))

    # "Combined" row: note that this selects stronger families post-hoc after inspecting
    # validation results (RF for building & coordinates, XGBoost for floor).
    # Labelled explicitly as post-hoc exploratory combination as recommended by review.
    rows.append(score_row(
        "Combined post-hoc (RF bld + XGB floor + RF coords)",
        d["RandomForest_building"], d["XGBoost_floor"], d["RandomForest_coords"],
        tb, tf, tc))

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "metrics.csv", index=False)
    print("\n" + "=" * 70)
    print("Summary Metrics Table (also saved to outputs/metrics.csv)")
    print("=" * 70)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
