"""Part 7 of the ML lifecycle: Robustness analysis (Device Hardware Shift).

The brief instructs us to retain user and phone identifiers for robustness analysis.
Hardware antenna calibration, sensitivity, and form factor introduce substantial
RSSI variation across handsets.

Phone Overlap Analysis:
------------------------
* Training contains 16 distinct PHONEIDs:
    [1, 3, 6, 7, 8, 10, 11, 13, 14, 16, 17, 18, 19, 22, 23, 24]
* Validation contains 11 distinct PHONEIDs:
    [0, 2, 4, 5, 9, 12, 13, 14, 15, 20, 21]
* Overlap: Phones 13 and 14 appear in BOTH training and validation.
    - Shared (seen) phones : 397 validation rows (Phone 13: 138, Phone 14: 259)
    - Unseen phones        : 714 validation rows (9 handsets)

We evaluate both the per-handset breakdown and the aggregate seen vs unseen group
performance to honestly report device generalisation.

Output: printed report + outputs/robustness_by_phone.png and outputs/robustness_by_phone.csv
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import OUT_DIR, VAL_CSV, TRAIN_CSV, TARGET_FLOOR
from part2_preprocess import targets

FAMILIES = ["RandomForest", "XGBoost"]


def euclidean(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=1))


def main():
    # Load fitted models + feature builder (from Part 4)
    with open(OUT_DIR / "models.pkl", "rb") as f:
        bundle = pickle.load(f)
    fb = bundle["feature_builder"]
    models = bundle["models"]

    train = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)
    Xva = fb.transform(val)
    yva = targets(val)
    phones = val["PHONEID"].values

    train_phones = set(train["PHONEID"].unique().tolist())
    val_phones = sorted(np.unique(phones).tolist())
    shared_phones = set(val_phones).intersection(train_phones)
    unseen_phones = set(val_phones) - train_phones

    print("=" * 70)
    print("Device Robustness Analysis: Mixed Seen vs Unseen Handsets")
    print("=" * 70)
    print(f"Validation handsets ({len(val_phones)} total): {val_phones}")
    print(f"  Shared (seen in train) : {sorted(list(shared_phones))} ({len(shared_phones)} phones)")
    print(f"  Unseen in train        : {sorted(list(unseen_phones))} ({len(unseen_phones)} phones)")

    rows = []
    group_rows = []

    for fam in FAMILIES:
        pf = models[fam]["floor"].predict(Xva)
        pc = models[fam]["coords"].predict(Xva)
        err = euclidean(pc, yva["coords"])

        # Per-phone breakdown
        for ph in val_phones:
            m = phones == ph
            is_seen = ph in shared_phones
            rows.append(dict(
                family=fam,
                phone=int(ph),
                device_status="Seen (shared)" if is_seen else "Unseen",
                n=int(m.sum()),
                floor_acc=round((pf[m] == yva["floor"][m]).mean(), 4),
                coord_mean_m=round(err[m].mean(), 2),
                coord_median_m=round(float(np.median(err[m])), 2),
                coord_p95_m=round(float(np.percentile(err[m], 95)), 2),
            ))

        # Seen vs Unseen group comparison
        seen_mask = np.isin(phones, list(shared_phones))
        unseen_mask = ~seen_mask

        group_rows.append(dict(
            family=fam,
            group="Seen (shared handsets 13 & 14)",
            n_samples=int(seen_mask.sum()),
            floor_acc=round((pf[seen_mask] == yva["floor"][seen_mask]).mean(), 4),
            coord_mean_m=round(err[seen_mask].mean(), 2),
            coord_median_m=round(float(np.median(err[seen_mask])), 2),
        ))
        group_rows.append(dict(
            family=fam,
            group="Unseen handsets (9 phones)",
            n_samples=int(unseen_mask.sum()),
            floor_acc=round((pf[unseen_mask] == yva["floor"][unseen_mask]).mean(), 4),
            coord_mean_m=round(err[unseen_mask].mean(), 2),
            coord_median_m=round(float(np.median(err[unseen_mask])), 2),
        ))

    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT_DIR / "robustness_by_phone.csv", index=False)

    df_group = pd.DataFrame(group_rows)
    print("\n" + "-" * 70)
    print("Seen vs Unseen Group Summary:")
    print("-" * 70)
    print(df_group.to_string(index=False))

    for fam in FAMILIES:
        sub = tbl[tbl.family == fam].sort_values("coord_mean_m")
        print(f"\n{fam} -- Coordinate Error by Phone (sorted by mean error):")
        print(sub[["phone", "device_status", "n", "floor_acc", "coord_mean_m", "coord_median_m", "coord_p95_m"]].to_string(index=False))
        best = sub.iloc[0]; worst = sub.iloc[-1]
        print(f"  Best handset : Phone {int(best.phone)} ({best.device_status}) -> {best.coord_mean_m} m")
        print(f"  Worst handset: Phone {int(worst.phone)} ({worst.device_status}) -> {worst.coord_mean_m} m")
        print(f"  Handset performance spread: {worst.coord_mean_m / max(best.coord_mean_m, 1e-9):.1f}x")

    # ---- Figure: Grouped bar of mean coord error per phone ---------------
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = tbl.pivot(index="phone", columns="family", values="coord_mean_m")
    pivot.plot(kind="bar", ax=ax, width=0.8)

    # Highlight shared phones on the x-axis
    labels = []
    for ph in pivot.index:
        tag = " (seen)" if ph in shared_phones else ""
        labels.append(f"Phone {ph}{tag}")
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_title("Robustness: Mean Coordinate Error per Handset (Validation: 397 Seen vs 714 Unseen)", fontsize=12)
    ax.set_xlabel("Device (PHONEID)")
    ax.set_ylabel("Mean Coordinate Error (m)")
    ax.legend(title="Model Family")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "robustness_by_phone.png", dpi=120)
    plt.close(fig)

    print("\n✓ Saved: outputs/robustness_by_phone.csv, outputs/robustness_by_phone.png")


if __name__ == "__main__":
    main()
