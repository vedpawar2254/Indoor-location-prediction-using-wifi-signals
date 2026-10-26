#!/usr/bin/env python3
"""USERID GroupKFold Split Generator.

Generates reproducible 5-fold cross-validation splits grouped by USERID
on the deduplicated UJIIndoorLoc training set.

Member 1 (Data & Infrastructure Lead) Deliverable.
Ensures zero user leakage between training and validation folds.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import (
    TRAIN_CSV, SPLITS_DIR, FOLDS_JSON, RANDOM_STATE,
    TARGET_BUILDING, TARGET_FLOOR, TARGET_LON, TARGET_LAT
)
from part2_preprocess import load_raw


def generate_folds(n_splits: int = 5):
    print("=" * 70)
    print("Generating USERID GroupKFold Cross-Validation Splits (Member 1)")
    print("=" * 70)

    train, _ = load_raw(dedup=True)
    n_samples = len(train)
    users = train["USERID"].values
    unique_users = np.unique(users)
    print(f"Cleaned training samples: {n_samples}")
    print(f"Unique USERIDs ({len(unique_users)}): {sorted(unique_users.tolist())}")

    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    summary_rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(train, groups=users)):
        train_users = sorted(np.unique(users[train_idx]).tolist())
        val_users = sorted(np.unique(users[val_idx]).tolist())

        # Verify no user overlap
        overlap = set(train_users).intersection(set(val_users))
        assert len(overlap) == 0, f"Fold {fold_idx} has user leakage: {overlap}"

        folds.append({
            "fold": fold_idx,
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "train_users": train_users,
            "val_users": val_users,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
        })

        summary_rows.append({
            "fold": fold_idx,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "val_users": ",".join(map(str, val_users)),
            "val_pct": round(len(val_idx) / n_samples * 100, 2),
            "building_counts": dict(train.iloc[val_idx][TARGET_BUILDING].value_counts().items()),
        })

        print(f"\nFold {fold_idx}:")
        print(f"  Train: {len(train_idx)} samples (users: {train_users})")
        print(f"  Val  : {len(val_idx)} samples (users: {val_users})")

    # Save folds to JSON
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    with open(FOLDS_JSON, "w") as f:
        json.dump({
            "n_splits": n_splits,
            "n_samples": n_samples,
            "folds": folds,
        }, f, indent=2)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = SPLITS_DIR / "folds_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"\n✓ Saved fold definitions to: {FOLDS_JSON}")
    print(f"✓ Saved summary table to: {summary_csv}")
    return folds


if __name__ == "__main__":
    generate_folds()
