"""Phase 2: 5-Fold USERID GroupKFold Training & Cross-Architecture DL Benchmark.

Trains all five deep learning architectures strictly on the Phase 1 USERID
GroupKFold splits (`data/splits/folds.json`) with zero validation leakage:
  1. SingleTaskMLP  (unshared 3-branch DL baseline)
  2. MTL_MLP        (Phase 1 proposed plain multitask MLP)
  3. SAE_MTL        (Stacked Denoising Autoencoder + MTL, Nowicki & Wietrzykowski 2017)
  4. CNN1D_MTL      (1D CNN over 3-channel AP spectrum, CNNLoc-inspired, Song et al. 2019)
  5. HierResMTL     (Proposed Hierarchical Residual FiLM Multitask Network)

Artifacts written to `outputs/phase2/`:
  - `dl_predictions.npz`: 5-fold OOF and 5-fold ensemble Official Val predictions
  - `dl_cv_folds.csv`: per-fold CV metrics for all 5 architectures
  - `training_histories.json`: epoch-level convergence & Kendall sigmas for HierResMTL
"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from config import OUT_DIR, FOLDS_JSON, FLOOR_PENALTY_METERS, BUILDING_PENALTY_METERS, RANDOM_STATE
from part2_preprocess import load_raw, Preprocessor, targets
from phase2_features import filter_non_blind, extract_strongest_wap_idx, build_dl_features
from phase2_models import SingleTaskMLP, MTL_MLP, SAE_MTL, CNN1D_MTL, HierResMTL
from phase2_trainer import train_dl_model, predict_dl_model

P2_OUT_DIR = OUT_DIR / "phase2"


def compute_metrics_dict(name, pb, pf, pc, tb, tf, tc, fold=None):
    b_acc = float(accuracy_score(tb, pb))
    f_acc = float(accuracy_score(tf, pf))
    f_f1 = float(f1_score(tf, pf, average="macro"))
    mae_lon = float(mean_absolute_error(tc[:, 0], pc[:, 0]))
    mae_lat = float(mean_absolute_error(tc[:, 1], pc[:, 1]))
    mae_avg = (mae_lon + mae_lat) / 2.0
    err = np.sqrt(((pc - tc) ** 2).sum(axis=1))
    comb = err + FLOOR_PENALTY_METERS * (pf != tf) + BUILDING_PENALTY_METERS * (pb != tb)
    d = dict(
        model=name,
        building_acc=round(b_acc, 4),
        floor_acc=round(f_acc, 4),
        floor_macroF1=round(f_f1, 4),
        coord_mae_lon_m=round(mae_lon, 2),
        coord_mae_lat_m=round(mae_lat, 2),
        coord_mae_avg_m=round(mae_avg, 2),
        coord_mean_m=round(float(err.mean()), 2),
        coord_median_m=round(float(np.median(err)), 2),
        coord_p75_m=round(float(np.percentile(err, 75)), 2),
        coord_p90_m=round(float(np.percentile(err, 90)), 2),
        combined_mean_m=round(float(comb.mean()), 2),
        combined_median_m=round(float(np.median(comb)), 2),
    )
    if fold is not None:
        d["fold"] = int(fold)
    return d


def _build_arch(arch_name: str, in_dim: int, n_waps: int):
    if arch_name == "SingleTaskMLP":
        return SingleTaskMLP(in_dim, n_waps)
    if arch_name == "MTL_MLP":
        return MTL_MLP(in_dim, n_waps)
    if arch_name == "SAE_MTL":
        return SAE_MTL(in_dim, n_waps)
    if arch_name == "CNN1D_MTL":
        return CNN1D_MTL(in_dim, n_waps)
    if arch_name == "HierResMTL":
        return HierResMTL(in_dim, n_waps)
    raise ValueError(f"Unknown arch: {arch_name}")


def main():
    P2_OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    train, val = load_raw(dedup=True)
    with open(FOLDS_JSON) as f:
        folds = json.load(f)["folds"]

    pre_global = Preprocessor().fit(train)
    kept_waps = pre_global.kept_waps
    n_waps = len(kept_waps)
    y_all = targets(train)
    y_val = targets(val)

    n_train, n_val = len(train), len(val)
    wid_val = extract_strongest_wap_idx(val, kept_waps)

    # Precompute R3 features per fold once so all 5 DL architectures train on identical inputs
    fold_cache = []
    for fobj in folds:
        fi = fobj["fold"]
        tr_idx = np.array(fobj["train_indices"])
        es_idx = np.array(fobj["val_indices"])

        tr_df = filter_non_blind(train.iloc[tr_idx].reset_index(drop=True))
        es_df = train.iloc[es_idx].reset_index(drop=True)

        X_tr, X_es = build_dl_features(tr_df, es_df, kept_waps, mode="R3")
        _, X_va = build_dl_features(tr_df, val, kept_waps, mode="R3")

        wid_tr = extract_strongest_wap_idx(tr_df, kept_waps)
        wid_es = extract_strongest_wap_idx(es_df, kept_waps)
        ytr, yes = targets(tr_df), targets(es_df)

        fold_cache.append((fi, tr_idx, es_idx, X_tr, wid_tr, ytr, X_es, wid_es, yes, X_va))

    arch_specs = [
        ("SingleTaskMLP", dict(epochs=40, sae_pretrain_epochs=0)),
        ("MTL_MLP", dict(epochs=40, sae_pretrain_epochs=0)),
        ("SAE_MTL", dict(epochs=40, sae_pretrain_epochs=12)),
        ("CNN1D_MTL", dict(epochs=40, sae_pretrain_epochs=0)),
        ("HierResMTL", dict(epochs=55, sae_pretrain_epochs=0)),
    ]

    npz_payload = {
        "train_building": y_all["building"],
        "train_floor": y_all["floor"],
        "train_coords": y_all["coords"],
        "val_building": y_val["building"],
        "val_floor": y_val["floor"],
        "val_coords": y_val["coords"],
    }
    fold_rows = []
    histories = {}

    for arch_name, kw in arch_specs:
        print("\n" + "=" * 75)
        print(f"5-FOLD USERID CV: {arch_name}")
        print("=" * 75)

        oof_pb = np.zeros((n_train, 3), dtype=np.float32)
        oof_pf = np.zeros((n_train, 5), dtype=np.float32)
        oof_pc = np.zeros((n_train, 2), dtype=np.float64)

        val_pb_ens = np.zeros((n_val, 3), dtype=np.float32)
        val_pf_ens = np.zeros((n_val, 5), dtype=np.float32)
        val_pc_ens = np.zeros((n_val, 2), dtype=np.float64)

        arch_hist = []
        for fi, _, es_idx, X_tr, wid_tr, ytr, X_es, wid_es, yes, X_va in fold_cache:
            model = _build_arch(arch_name, X_tr.shape[1], n_waps)
            model, cscaler, hist = train_dl_model(
                model,
                X_tr, wid_tr, ytr["building"], ytr["floor"], ytr["coords"],
                X_es, wid_es, yes["building"], yes["floor"], yes["coords"],
                n_waps=n_waps,
                rep_mode="R3",
                epochs=kw["epochs"],
                sae_pretrain_epochs=kw["sae_pretrain_epochs"],
                seed=RANDOM_STATE + fi,
            )
            pb_es, pf_es, pc_es, _ = predict_dl_model(model, X_es, wid_es, cscaler)
            oof_pb[es_idx] = pb_es
            oof_pf[es_idx] = pf_es
            oof_pc[es_idx] = pc_es

            m = compute_metrics_dict(
                arch_name, pb_es.argmax(1), pf_es.argmax(1), pc_es,
                yes["building"], yes["floor"], yes["coords"], fold=fi
            )
            fold_rows.append(m)
            arch_hist.append(hist)
            print(
                f"  Fold {fi} (ep {hist['best_epoch']:2d}, {hist['train_sec']:5.1f}s) | "
                f"Bld={m['building_acc']*100:.2f}% Flr={m['floor_acc']*100:.2f}% | "
                f"Mean={m['coord_mean_m']:.2f}m Med={m['coord_median_m']:.2f}m EvAAL={m['combined_mean_m']:.2f}m"
            )

            pb_v, pf_v, pc_v, _ = predict_dl_model(model, X_va, wid_val, cscaler)
            val_pb_ens += pb_v / len(fold_cache)
            val_pf_ens += pf_v / len(fold_cache)
            val_pc_ens += pc_v / len(fold_cache)

        histories[arch_name] = arch_hist

        oof_m = compute_metrics_dict(
            f"{arch_name}_OOF", oof_pb.argmax(1), oof_pf.argmax(1), oof_pc,
            y_all["building"], y_all["floor"], y_all["coords"]
        )
        val_m = compute_metrics_dict(
            f"{arch_name}_ValEns", val_pb_ens.argmax(1), val_pf_ens.argmax(1), val_pc_ens,
            y_val["building"], y_val["floor"], y_val["coords"]
        )
        print(
            f"  >> 5-Fold OOF : Bld={oof_m['building_acc']*100:.2f}% Flr={oof_m['floor_acc']*100:.2f}% "
            f"Mean={oof_m['coord_mean_m']:.2f}m EvAAL={oof_m['combined_mean_m']:.2f}m"
        )
        print(
            f"  >> Val 5-Ens  : Bld={val_m['building_acc']*100:.2f}% Flr={val_m['floor_acc']*100:.2f}% "
            f"(F1={val_m['floor_macroF1']:.4f}) MAE={val_m['coord_mae_avg_m']:.2f}m "
            f"Mean={val_m['coord_mean_m']:.2f}m Med={val_m['coord_median_m']:.2f}m "
            f"P90={val_m['coord_p90_m']:.2f}m EvAAL={val_m['combined_mean_m']:.2f}m"
        )

        npz_payload[f"{arch_name}_oof_pb"] = oof_pb
        npz_payload[f"{arch_name}_oof_pf"] = oof_pf
        npz_payload[f"{arch_name}_oof_pc"] = oof_pc
        npz_payload[f"{arch_name}_val_pb"] = val_pb_ens
        npz_payload[f"{arch_name}_val_pf"] = val_pf_ens
        npz_payload[f"{arch_name}_val_pc"] = val_pc_ens

    np.savez(P2_OUT_DIR / "dl_predictions.npz", **npz_payload)
    pd.DataFrame(fold_rows).to_csv(P2_OUT_DIR / "dl_cv_folds.csv", index=False)
    with open(P2_OUT_DIR / "training_histories.json", "w") as f:
        json.dump(histories, f, indent=2)

    print(f"\n✓ Phase 2 DL 5-fold CV completed in {time.time() - t_all:.1f}s")
    print(f"  Saved: outputs/phase2/dl_predictions.npz, dl_cv_folds.csv, training_histories.json")


if __name__ == "__main__":
    main()

