"""Part 3 of the ML lifecycle: Feature engineering & Representations.

The AML baseline augments the raw RSSI vector with hand-crafted aggregate features
and explicit absence indicators.

Feature representations implemented:
------------------------------------
1. Scaled RSSI: 465 kept WAP columns (StandardScaler fit on train only).
2. Binary Absence Indicators: 465 flags (1.0 if WAP <= -105 dBm / not heard, 0.0 if heard).
   * Scientific justification: While standard scaling is an invertible linear transform,
     decision trees make axis-aligned cuts on individual features. After standardization,
     the -105 dBm sentinel maps to varying continuous values across APs. Explicit binary
     absence flags provide a direct, single-split boolean representation of AP visibility.
3. Signal Aggregates (5 features, standardized with train-only statistics):
   - signal_count: number of APs actually heard (RSSI > -105 dBm)
   - strongest_signal: maximum observed RSSI (loudest AP; floor if none)
   - mean_rssi: ambient mean over heard APs only
   - std_rssi: spread / dispersion of heard RSSI
   - range_rssi: dynamic range (strongest - weakest heard)
4. Discrete Spatial Fingerprint (1 unscaled ordinal feature):
   - strongest_wap_id: index of the loudest AP (-1 if none heard)

Total: 936 features = 465 RSSI + 465 absence + 5 aggregates + 1 ordinal.

Supports feature ablation modes:
  - "A1": Scaled RSSI only (465 features)
  - "A2": Scaled RSSI + Absence indicators (930 features)
  - "A3": Full engineered set (936 features, default)

Deterministic Guard (G1):
  - detect_no_signal(): returns boolean mask of scans hearing 0 APs for fallback handling.

Member 2 (Features & Representation Lead) Deliverable.
"""
from dataclasses import dataclass, field
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import WAP_COLS, NO_SIGNAL, NO_SIGNAL_FILL
from part2_preprocess import Preprocessor, load_raw, targets

AGG_NAMES = ["signal_count", "strongest_signal", "mean_rssi",
             "std_rssi", "range_rssi"]  # scaled block
RAW_EXTRA = ["strongest_wap_id"]        # unscaled ordinal


def _raw_remapped(df: pd.DataFrame, kept_waps) -> np.ndarray:
    """Remapped (sentinel->floor) RSSI for the KEPT APs, as a float matrix."""
    return df[kept_waps].replace(NO_SIGNAL, NO_SIGNAL_FILL).values.astype(float)


def _absence_flags(raw: np.ndarray) -> np.ndarray:
    """Binary missingness indicators: 1.0 where an AP was NOT heard.

    Provides an explicit, un-normalized missingness representation for tree splits.
    """
    return (raw <= NO_SIGNAL_FILL).astype(float)


def _aggregates(raw: np.ndarray) -> np.ndarray:
    """Compute the aggregate feature block from a raw remapped RSSI matrix."""
    heard = raw > NO_SIGNAL_FILL                    # bool mask of real readings
    count = heard.sum(axis=1)

    # strongest: max over all (floor if nothing heard)
    strongest = raw.max(axis=1)

    # mean / std / range over HEARD values only; floor-safe when count==0.
    raw_heard = np.where(heard, raw, np.nan)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(raw_heard, axis=1)
        std = np.nanstd(raw_heard, axis=1)
        weakest = np.nanmin(raw_heard, axis=1)

    no_heard = count == 0
    mean = np.where(no_heard, NO_SIGNAL_FILL, mean)
    std = np.where(no_heard, 0.0, std)
    weakest = np.where(no_heard, NO_SIGNAL_FILL, weakest)
    rng = strongest - weakest

    return np.column_stack([count, strongest, mean, std, rng])


def _strongest_wap_id(raw: np.ndarray) -> np.ndarray:
    """Index of the loudest AP; -1 when nothing heard."""
    heard_any = (raw > NO_SIGNAL_FILL).any(axis=1)
    idx = raw.argmax(axis=1).astype(float)
    idx[~heard_any] = -1
    return idx.reshape(-1, 1)


def detect_no_signal(df: pd.DataFrame, kept_waps=None) -> np.ndarray:
    """Deterministic no-signal guard (G1): flag scans that heard 0 APs."""
    cols = kept_waps if kept_waps is not None else [c for c in df.columns if c.startswith("WAP")]
    raw = df[cols].replace(NO_SIGNAL, NO_SIGNAL_FILL).values
    return (raw <= NO_SIGNAL_FILL).all(axis=1)


@dataclass
class FeatureBuilder:
    pre: Preprocessor
    mode: str = "A3"  # "A1", "A2", or "A3"
    agg_scaler: StandardScaler = None
    feature_names: list = field(default_factory=list)

    def fit(self, train: pd.DataFrame):
        raw = _raw_remapped(train, self.pre.kept_waps)
        self.agg_scaler = StandardScaler().fit(_aggregates(raw))

        rssi_names = list(self.pre.kept_waps)
        absence_names = [f"{w}_absent" for w in self.pre.kept_waps]

        if self.mode == "A1":
            self.feature_names = rssi_names
        elif self.mode == "A2":
            self.feature_names = rssi_names + absence_names
        else:  # "A3"
            self.feature_names = rssi_names + absence_names + AGG_NAMES + RAW_EXTRA
        return self

    def transform(self, df: pd.DataFrame, mode: str = None) -> np.ndarray:
        m = mode or self.mode
        rssi_scaled = self.pre.transform_rssi(df)               # (n, 465)
        if m == "A1":
            return rssi_scaled

        raw = _raw_remapped(df, self.pre.kept_waps)
        absence = _absence_flags(raw)                             # (n, 465)
        if m == "A2":
            return np.hstack([rssi_scaled, absence])

        agg_scaled = self.agg_scaler.transform(_aggregates(raw))  # (n, 5)
        wap_id = _strongest_wap_id(raw)                           # (n, 1)
        return np.hstack([rssi_scaled, absence, agg_scaled, wap_id])

    def fit_transform(self, train: pd.DataFrame) -> np.ndarray:
        return self.fit(train).transform(train)


def build_all(mode: str = "A3"):
    """Convenience: load, preprocess, engineer features for train+val."""
    train, val = load_raw()
    pre = Preprocessor().fit(train)
    fb = FeatureBuilder(pre, mode=mode).fit(train)
    Xtr, Xva = fb.transform(train), fb.transform(val)
    return fb, Xtr, Xva, targets(train), targets(val)


def main():
    fb, Xtr, Xva, ytr, yva = build_all()
    print("=" * 70)
    print("Feature Engineering & Representations (Member 2)")
    print("=" * 70)
    n_kept = len(fb.pre.kept_waps)
    print(f"Active mode: {fb.mode}")
    print(f"Total features: {len(fb.feature_names)}  "
          f"= {n_kept} RSSI + {n_kept} absence flags + {len(AGG_NAMES)} "
          f"aggregates + {len(RAW_EXTRA)} ordinal")
    print(f"Design matrix train: {Xtr.shape}   val: {Xva.shape}")

    train, val = load_raw()
    no_sig_tr = detect_no_signal(train, fb.pre.kept_waps).sum()
    no_sig_va = detect_no_signal(val, fb.pre.kept_waps).sum()
    print(f"\nDeterministic Guard (G1) Check:")
    print(f"  Train scans with no signal heard: {no_sig_tr} / {len(train)}")
    print(f"  Val scans with no signal heard  : {no_sig_va} / {len(val)}")

    # Show engineered block for first 5 rows
    raw = _raw_remapped(train, fb.pre.kept_waps)
    agg = _aggregates(raw)
    wid = _strongest_wap_id(raw)
    demo = pd.DataFrame(np.hstack([agg, wid]),
                        columns=AGG_NAMES + RAW_EXTRA).head()
    print("\nEngineered features (raw scale, first 5 train rows):")
    print(demo.to_string(index=False))

    bad = (~np.isfinite(Xtr)).sum() + (~np.isfinite(Xva)).sum()
    print(f"\nNon-finite cells in design matrices: {bad} (must be 0)")


if __name__ == "__main__":
    main()
