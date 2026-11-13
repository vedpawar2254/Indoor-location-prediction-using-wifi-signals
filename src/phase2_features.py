"""Phase 2 Feature Engineering & Representations for Deep Learning & Hybrid Models.

Extends Phase 1's leakage-free pipeline (`part2_preprocess.py`, `part3_features.py`)
with neural-optimized and device-invariant RSSI representations while preserving
identical train/validation conventions for a strict apples-to-apples comparison.

Representations supported:
  - "R1": Phase 1 StandardScaler RSSI + binary absence flags + scaled aggregates
  - "R2": Positive-Powed RSSI in [0, 1] (Torres-Sospedra 2015 / Song et al. 2019 CNNLoc)
  - "R3_no_rel": Positive-Powed RSSI + binary presence mask + scaled aggregates (ablation)
  - "R3": Device-Invariant Multiview (Positive-Powed RSSI + Scan-Relative Exponential
          Decay + Binary Presence Mask + Train-Scaled Physical Aggregates)

Deterministic Guard (G1) & Blind-Scan Hygiene:
  - Deduplicated training data contains 73 zero-AP blind scans carrying contradictory
    spatial labels across all 3 buildings. `filter_non_blind()` removes these strictly
    from DL training batches while keeping validation untouched and retaining G1 at inference.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import WAP_COLS, NO_SIGNAL_FILL
from part3_features import _raw_remapped, _aggregates, _strongest_wap_id


def filter_non_blind(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """Drop scans with zero heard APs strictly during DL training."""
    target_cols = cols if cols is not None else WAP_COLS
    raw = _raw_remapped(df, target_cols)
    keep_mask = (raw > NO_SIGNAL_FILL).any(axis=1)
    return df.loc[keep_mask].reset_index(drop=True)


def extract_strongest_wap_idx(df: pd.DataFrame, kept_waps) -> np.ndarray:
    """Return 1-based strongest AP index (0 = no AP heard, 1..n_waps = kept AP)."""
    raw = _raw_remapped(df, kept_waps)
    return (_strongest_wap_id(raw).ravel().astype(np.int64) + 1)


def _positive_powed_rssi(raw: np.ndarray, heard: np.ndarray, gamma: float = np.e) -> np.ndarray:
    """Map observed RSSI [-105, 0] -> [0, 1]^gamma; undetected stays exact 0.0."""
    norm = np.clip((raw - NO_SIGNAL_FILL) / (-NO_SIGNAL_FILL), 0.0, 1.0)
    return np.where(heard > 0, norm ** gamma, 0.0).astype(np.float32)


def _scan_relative_decay(raw: np.ndarray, heard: np.ndarray, tau: float = 20.0) -> np.ndarray:
    """Offset-invariant per-scan relative RSSI: exp((x_ij - max_k x_ik) / tau).

    Cancels additive per-handset RSSI sensitivity shifts Delta_p because:
      (x_ij + Delta_p) - (x_i,max + Delta_p) = x_ij - x_i,max.
    """
    max_per_scan = np.max(np.where(heard > 0, raw, float(NO_SIGNAL_FILL)), axis=1, keepdims=True)
    diff = raw - max_per_scan
    return np.where(heard > 0, np.exp(diff / tau), 0.0).astype(np.float32)


def build_dl_features(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    kept_waps: list,
    mode: str = "R3",
) -> tuple[np.ndarray, np.ndarray]:
    """Construct leakage-free DL design matrices fit strictly on `train_df`."""
    raw_tr = _raw_remapped(train_df, kept_waps)
    raw_ev = _raw_remapped(eval_df, kept_waps)

    heard_tr = (raw_tr > NO_SIGNAL_FILL).astype(np.float32)
    heard_ev = (raw_ev > NO_SIGNAL_FILL).astype(np.float32)

    agg_scaler = StandardScaler().fit(_aggregates(raw_tr))
    agg_tr = agg_scaler.transform(_aggregates(raw_tr)).astype(np.float32)
    agg_ev = agg_scaler.transform(_aggregates(raw_ev)).astype(np.float32)

    if mode == "R1":
        rssi_scaler = StandardScaler().fit(raw_tr)
        rssi_tr = rssi_scaler.transform(raw_tr).astype(np.float32)
        rssi_ev = rssi_scaler.transform(raw_ev).astype(np.float32)
        abs_tr = (1.0 - heard_tr).astype(np.float32)
        abs_ev = (1.0 - heard_ev).astype(np.float32)
        return np.hstack([rssi_tr, abs_tr, agg_tr]), np.hstack([rssi_ev, abs_ev, agg_ev])

    pow_tr = _positive_powed_rssi(raw_tr, heard_tr)
    pow_ev = _positive_powed_rssi(raw_ev, heard_ev)

    if mode == "R2":
        return pow_tr, pow_ev

    if mode == "R3_no_rel":
        return np.hstack([pow_tr, heard_tr, agg_tr]), np.hstack([pow_ev, heard_ev, agg_ev])

    if mode == "R3":
        rel_tr = _scan_relative_decay(raw_tr, heard_tr)
        rel_ev = _scan_relative_decay(raw_ev, heard_ev)
        return (
            np.hstack([pow_tr, rel_tr, heard_tr, agg_tr]),
            np.hstack([pow_ev, rel_ev, heard_ev, agg_ev]),
        )

    raise ValueError(f"Unknown representation mode: {mode}")


@dataclass
class CoordScaler:
    """Standardize 2D metric coordinates (LON, LAT) on training data only."""
    mean: np.ndarray = None
    std: np.ndarray = None

    def fit(self, coords: np.ndarray):
        self.mean = coords.mean(axis=0).astype(np.float32)
        self.std = np.maximum(coords.std(axis=0), 1e-6).astype(np.float32)
        return self

    def transform(self, coords: np.ndarray) -> np.ndarray:
        return ((coords - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, coords_scaled: np.ndarray) -> np.ndarray:
        return (coords_scaled * self.std + self.mean).astype(np.float64)

