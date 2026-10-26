"""Shared config + constants for the UJIIndoorLoc indoor-positioning project.

All lifecycle scripts import from here so paths and conventions stay consistent.
"""
from pathlib import Path

# ---- paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

# Check candidate data directories
CANDIDATE_DATA_DIRS = [
    ROOT / "data" / "UJIndoorLoc",
    ROOT / "data" / "raw",
    ROOT / "data",
]

DATA_DIR = CANDIDATE_DATA_DIRS[0]
for d in CANDIDATE_DATA_DIRS:
    if (d / "trainingData.csv").exists():
        DATA_DIR = d
        break

TRAIN_CSV = DATA_DIR / "trainingData.csv"
VAL_CSV = DATA_DIR / "validationData.csv"

SPLITS_DIR = ROOT / "data" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
FOLDS_JSON = SPLITS_DIR / "folds.json"

OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- dataset conventions --------------------------------------------------
N_WAPS = 520
WAP_COLS = [f"WAP{i:03d}" for i in range(1, N_WAPS + 1)]  # WAP001..WAP520

# UJIIndoorLoc uses +100 dBm to mean "access point not heard".
NO_SIGNAL = 100
# Observed RSSI lives in roughly [-104, 0] dBm. We map "not heard" to a value
# just below the weakest real reading so models see a consistent floor.
NO_SIGNAL_FILL = -105

# target columns
TARGET_LON = "LONGITUDE"
TARGET_LAT = "LATITUDE"
TARGET_FLOOR = "FLOOR"
TARGET_BUILDING = "BUILDINGID"

# identifier / metadata columns (kept for robustness analysis, not used as features)
ID_COLS = ["SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP"]

RANDOM_STATE = 42

# Competition evaluation penalty weights
FLOOR_PENALTY_METERS = 4.0     # 4 m penalty per incorrect floor level
BUILDING_PENALTY_METERS = 50.0 # 50 m penalty per incorrect building
