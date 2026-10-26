"""Part 1 of the ML lifecycle: Problem framing + Data loading + EDA.

Problem framing
---------------
UJIIndoorLoc: predict a device's indoor location from 520 Wi-Fi RSSI readings.
This is a MULTITASK problem with three coupled targets:
  - BUILDINGID  (3 classes: 0, 1, 2)  -> classification
  - FLOOR       (5 classes: 0 to 4)   -> classification
  - LONGITUDE / LATITUDE (meters)     -> continuous 2D regression

Phase 1 (this project stage) builds the AML baseline: separate tabular models
per target. Phase 2 will share information across targets with a DL trunk.

Evaluation metrics (chosen up front, before any modelling):
  - Building accuracy, Floor accuracy
  - Coordinate error:
      * Mean Absolute Error (MAE) for LONGITUDE and LATITUDE explicitly
      * Euclidean distance in meters (mean, median, P75, P90)
  - Hierarchical combined error (EvAAL / UJIIndoorLoc custom competition penalty):
      distance + 4 m * (wrong floor) + 50 m * (wrong building)
  - Floor confusion matrices + a coordinate error map (Part 6).

Split strategy:
  - Official validation set (validationData.csv): 1,111 samples collected months
    later by different operators and devices -> realistic distribution shift test.
  - USERID GroupKFold: 5-fold cross-validation partitioned strictly by USERID
    on the training set (see scripts/build_splits.py), ensuring zero user leakage.

Member 1 (Data & Infrastructure Lead) Deliverable.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless; we save figures to disk
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    TRAIN_CSV, VAL_CSV, OUT_DIR, WAP_COLS, NO_SIGNAL,
    TARGET_BUILDING, TARGET_FLOOR, TARGET_LON, TARGET_LAT,
)


def load_raw():
    train = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)
    return train, val


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    train, val = load_raw()

    section("1. Shapes & columns")
    print(f"train: {train.shape}   val: {val.shape}")
    print(f"WAP feature columns: {len(WAP_COLS)}  (WAP001..WAP520)")
    print(f"non-WAP columns: {[c for c in train.columns if not c.startswith('WAP')]}")

    section("2. Target distributions (train)")
    print("BUILDINGID counts:\n", train[TARGET_BUILDING].value_counts().sort_index())
    print("\nFLOOR counts:\n", train[TARGET_FLOOR].value_counts().sort_index())
    print("\nFLOOR per building (crosstab):")
    print(pd.crosstab(train[TARGET_BUILDING], train[TARGET_FLOOR]))
    print(f"\nLONGITUDE range: [{train[TARGET_LON].min():.1f}, {train[TARGET_LON].max():.1f}]")
    print(f"LATITUDE  range: [{train[TARGET_LAT].min():.1f}, {train[TARGET_LAT].max():.1f}]")

    section("3. Signal sparsity & Data Quality Audits")
    wap_tr = train[WAP_COLS].values
    not_heard = (wap_tr == NO_SIGNAL)
    print(f"'not heard' sentinel value: +{NO_SIGNAL} dBm")
    print(f"fraction of (row,AP) cells with no signal: {not_heard.mean():.4f} (96.5% sparse)")
    observed = wap_tr[~not_heard]
    print(f"observed RSSI range: [{observed.min()}, {observed.max()}] dBm")
    
    aps_per_sample = (~not_heard).sum(axis=1)
    zero_ap_scans = (aps_per_sample == 0).sum()
    print(f"APs heard per sample -> min {aps_per_sample.min()}, "
          f"median {int(np.median(aps_per_sample))}, max {aps_per_sample.max()}, "
          f"mean {aps_per_sample.mean():.1f}")
    print(f"Scans with ZERO APs heard in train: {zero_ap_scans} / {len(train)}")
    
    dead_aps = (not_heard.all(axis=0)).sum()
    print(f"APs never heard in ANY training sample: {dead_aps} / {len(WAP_COLS)} (uninformative constant columns)")

    # Suspicious Near-Zero RSSI Audit
    near_zero = (observed >= -5) & (observed <= 0)
    print(f"\nNear-zero RSSI readings ([-5, 0] dBm): {near_zero.sum()} occurrences "
          f"({near_zero.mean() * 100:.3f}% of heard readings)")
    if near_zero.sum() > 0:
        nz_mask = (wap_tr >= -5) & (wap_tr <= 0)
        nz_rows = np.where(nz_mask.any(axis=1))[0]
        nz_users = train.iloc[nz_rows]["USERID"].unique()
        print(f"  Affects {len(nz_rows)} training rows across USERIDs: {sorted(nz_users.tolist())}")

    section("4. Split Realism & Phone Overlap Audit (Factual Correction)")
    train_users = sorted(train.USERID.unique().tolist())
    val_users = sorted(val.USERID.unique().tolist())
    train_phones = set(train.PHONEID.unique().tolist())
    val_phones = set(val.PHONEID.unique().tolist())

    print(f"train USERID ({len(train_users)}): {train_users}")
    print(f"val   USERID ({len(val_users)}): {val_users}  (0 = unlabelled in official validation)")
    print(f"train PHONEID ({len(train_phones)}): {sorted(list(train_phones))}")
    print(f"val   PHONEID ({len(val_phones)}): {sorted(list(val_phones))}")

    shared_phones = train_phones.intersection(val_phones)
    unseen_phones = val_phones - train_phones
    print(f"\nPhone overlap breakdown:")
    print(f"  Shared (seen) phones: {sorted(list(shared_phones))} ({len(shared_phones)} phones)")
    print(f"  Unseen phones in val: {sorted(list(unseen_phones))} ({len(unseen_phones)} phones)")

    shared_val_rows = val["PHONEID"].isin(shared_phones).sum()
    unseen_val_rows = val["PHONEID"].isin(unseen_phones).sum()
    print(f"  Validation rows with shared phones: {shared_val_rows} / {len(val)} ({shared_val_rows / len(val) * 100:.1f}%)")
    print(f"  Validation rows with unseen phones: {unseen_val_rows} / {len(val)} ({unseen_val_rows / len(val) * 100:.1f}%)")
    print("=> Validation set is a MIXED seen/unseen handset evaluation, accurately reflecting real deployment.")

    # ---- EDA figures ------------------------------------------------------
    sns.set_theme(style="whitegrid")

    # Fig A: floor distribution per building
    fig, ax = plt.subplots(figsize=(7, 4))
    ct = pd.crosstab(train[TARGET_BUILDING], train[TARGET_FLOOR])
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
    ax.set_title("Train samples: floor distribution per building")
    ax.set_xlabel("Building ID"); ax.set_ylabel("sample count")
    ax.legend(title="Floor", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_floor_per_building.png", dpi=120)
    plt.close(fig)

    # Fig B: APs-heard-per-sample histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(aps_per_sample, bins=40, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_title("Signal sparsity: number of APs heard per sample (train)")
    ax.set_xlabel("APs heard"); ax.set_ylabel("samples")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_aps_per_sample.png", dpi=120)
    plt.close(fig)

    # Fig C: spatial scatter coloured by building
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(train[TARGET_LON], train[TARGET_LAT],
                    c=train[TARGET_BUILDING], cmap="Set1", s=4, alpha=0.5)
    ax.set_title("Train fingerprint locations (coloured by building)")
    ax.set_xlabel("LONGITUDE (m)"); ax.set_ylabel("LATITUDE (m)")
    ax.set_aspect("equal", adjustable="datalim")
    plt.colorbar(sc, ax=ax, label="Building ID")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_spatial_layout.png", dpi=120)
    plt.close(fig)

    section("5. Saved EDA figures")
    for f in ["eda_floor_per_building.png", "eda_aps_per_sample.png", "eda_spatial_layout.png"]:
        print(f"  outputs/{f}")


if __name__ == "__main__":
    main()
