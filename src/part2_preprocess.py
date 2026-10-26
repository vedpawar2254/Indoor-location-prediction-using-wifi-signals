"""Part 2 of the ML lifecycle: Preprocessing.

Decisions (and why):

1. "Not heard" convention (+100 dBm) is a sentinel, not a measurement.
   Left as-is it would dominate every distance/split. We remap it to a fixed
   floor value (NO_SIGNAL_FILL = -105 dBm), just below the weakest real reading
   (-104). This is the "impute absent access points consistently" step from the
   AML baseline. We use a FIXED value (not column means) so the mapping is
   identical for train and validation -> no leakage, deterministic.

2. Drop dead APs: 55 WAPs are never heard in ANY training sample. They carry
   zero training signal, so we drop them from the feature set (fit on train,
   applied to val). Keeps the model honest and a little smaller.

3. Scaling: after remapping, RSSI lives in [-105, 0]. We standardise
   (StandardScaler) using statistics computed on TRAIN ONLY, then apply the
   same transform to validation. This honours "do not normalize test data
   separately".

The scaler/column choices are packaged in `Preprocessor` so Parts 3-6 reuse
exactly the same transform.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    TRAIN_CSV, VAL_CSV, WAP_COLS, NO_SIGNAL, NO_SIGNAL_FILL,
    TARGET_BUILDING, TARGET_FLOOR, TARGET_LON, TARGET_LAT, ID_COLS,
)


def load_raw(dedup=True):
    """Load train/val. By default drop exact-duplicate training scans.

    The training file contains 637 rows that are exact copies of another scan
    (all 520 WAPs + all targets identical). Left in, they both waste model
    capacity and quietly inflate validation-independent diagnostics, so we drop
    them (keeping the first copy). Validation is never deduplicated.
    """
    train = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)
    if dedup:
        before = len(train)
        train = train.drop_duplicates().reset_index(drop=True)
        dropped = before - len(train)
        if dropped:
            print(f"[load_raw] dropped {dropped} duplicate train rows "
                  f"({before} -> {len(train)})")
    return train, val


@dataclass
class Preprocessor:
    """Fit on train, transform anything. Holds all learned state."""
    kept_waps: list = field(default_factory=list)
    scaler: StandardScaler = None

    def fit(self, train: pd.DataFrame):
        # remap sentinel so dead-AP detection sees real floor values
        remapped = self._remap(train[WAP_COLS])
        # an AP is "dead" if it never rose above the no-signal floor in train
        heard_any = (remapped.values > NO_SIGNAL_FILL).any(axis=0)
        self.kept_waps = [w for w, keep in zip(WAP_COLS, heard_any) if keep]
        self.scaler = StandardScaler().fit(remapped[self.kept_waps].values)
        return self

    def transform_rssi(self, df: pd.DataFrame) -> np.ndarray:
        """Return scaled RSSI matrix (kept APs only)."""
        remapped = self._remap(df[WAP_COLS])
        return self.scaler.transform(remapped[self.kept_waps].values)

    @staticmethod
    def _remap(wap_df: pd.DataFrame) -> pd.DataFrame:
        return wap_df.replace(NO_SIGNAL, NO_SIGNAL_FILL)


def targets(df: pd.DataFrame) -> dict:
    return {
        "building": df[TARGET_BUILDING].values,
        "floor": df[TARGET_FLOOR].values,
        "coords": df[[TARGET_LON, TARGET_LAT]].values,
    }


def main():
    train, val = load_raw()
    pre = Preprocessor().fit(train)

    print("=" * 70)
    print("Preprocessing summary")
    print("=" * 70)
    print(f"sentinel +{NO_SIGNAL} dBm  ->  fixed floor {NO_SIGNAL_FILL} dBm")
    print(f"WAPs kept: {len(pre.kept_waps)} / {len(WAP_COLS)} "
          f"(dropped {len(WAP_COLS) - len(pre.kept_waps)} dead APs)")

    Xtr = pre.transform_rssi(train)
    Xva = pre.transform_rssi(val)
    print(f"\nscaled train RSSI matrix: {Xtr.shape}")
    print(f"scaled val   RSSI matrix: {Xva.shape}")
    print(f"train scaled mean ~0 ? {Xtr.mean():.4f}   std ~1 ? {Xtr.std():.4f}")
    print(f"val   scaled mean    : {Xva.mean():.4f}   std    : {Xva.std():.4f}  "
          f"(NOT forced to 0/1 -> no separate normalisation, good)")

    yt = targets(train)
    print(f"\ntargets extracted: building{yt['building'].shape}, "
          f"floor{yt['floor'].shape}, coords{yt['coords'].shape}")


if __name__ == "__main__":
    main()
