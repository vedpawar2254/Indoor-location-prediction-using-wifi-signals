"""Part 4 of the ML lifecycle: Model training (AML baseline).

Baseline = SEPARATE tabular models, one per target, for two model families:
    families : RandomForest, XGBoost
    tasks    : building (classify), floor (classify), coords (regress)

That is 2 x 3 = 6 fitted estimators. The point of the baseline is precisely
that these models share NO information across targets -- that gap is what
Phase 2's multitask DL trunk will try to close.

Choices
-------
* Coordinates: one MultiOutput regressor predicting (LON, LAT) jointly per
  family. (RF supports multi-output natively; XGBoost is wrapped with
  MultiOutputRegressor since its sklearn API is single-output.)
* Floor/building fit on the full 936-feature matrix. A common stronger variant
  conditions floor on predicted building; we keep the clean independent
  baseline here and leave the hierarchical breakdown to the evaluation (Part 6).
* Hyperparameters: modest, fixed, reproducible. Not tuned -- this is a baseline.
  n_jobs=-1 to use all cores.

Artifacts written to outputs/:
    models.pkl        all 6 fitted estimators + the fitted FeatureBuilder
    predictions.npz   val predictions per family/task + ground truth
These feed Part 5 (metrics) and Part 6 (plots) with no retraining.
"""
import pickle
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBClassifier, XGBRegressor

from config import OUT_DIR, RANDOM_STATE
from part3_features import build_all


def make_models():
    """Factory: fresh estimator per (family, task)."""
    rf_kw = dict(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
    xgb_common = dict(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.8,
        tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE,
    )
    return {
        "RandomForest": {
            "building": RandomForestClassifier(**rf_kw),
            "floor": RandomForestClassifier(**rf_kw),
            "coords": RandomForestRegressor(**rf_kw),  # native multi-output
        },
        "XGBoost": {
            "building": XGBClassifier(eval_metric="mlogloss", **xgb_common),
            "floor": XGBClassifier(eval_metric="mlogloss", **xgb_common),
            "coords": MultiOutputRegressor(XGBRegressor(**xgb_common)),
        },
    }


def main():
    print("Loading + engineering features ...")
    fb, Xtr, Xva, ytr, yva = build_all()
    print(f"train {Xtr.shape}  val {Xva.shape}")

    models = make_models()
    fitted = {}
    preds = {"truth": yva}  # ground truth for the val set

    for family, tasks in models.items():
        fitted[family] = {}
        preds[family] = {}
        print("\n" + "=" * 60)
        print(f"Family: {family}")
        print("=" * 60)
        for task, est in tasks.items():
            y = ytr[task]
            t0 = time.time()
            est.fit(Xtr, y)
            dt = time.time() - t0
            pred = est.predict(Xva)
            fitted[family][task] = est
            preds[family][task] = pred
            shape = pred.shape if pred.ndim > 1 else (pred.shape[0],)
            print(f"  {task:9s} fit in {dt:6.1f}s  -> val pred {shape}")

    # ---- persist ----------------------------------------------------------
    with open(OUT_DIR / "models.pkl", "wb") as f:
        pickle.dump({"feature_builder": fb, "models": fitted}, f)
    np.savez(
        OUT_DIR / "predictions.npz",
        truth_building=yva["building"],
        truth_floor=yva["floor"],
        truth_coords=yva["coords"],
        **{f"{fam}_{task}": preds[fam][task]
           for fam in models for task in models[fam]},
    )
    print("\nsaved: outputs/models.pkl, outputs/predictions.npz")


if __name__ == "__main__":
    main()
