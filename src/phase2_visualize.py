"""Phase 2 Visualization Suite (Report, Presentation & Notebook Deliverables).

Generates high-resolution diagnostic and comparative figures in `outputs/phase2/`
and copies core report figures to `report/figs_phase2/`:
  1. p2_architecture_diagram.png   : Schematic of HierResMTL + Adaptive ML+DL Hybrid
  2. p2_training_curves.png        : 5-fold USERID CV training loss, EvAAL convergence & Kendall sigmas
  3. p2_model_comparison.png       : Phase 1 ML vs DL Baselines vs HierResMTL vs Hybrid (Val & 5-Fold OOF)
  4. p2_ablation_study.png         : Representation (R1->R2->R3) & Regularization/Mechanism ablations
  5. p2_error_cdf_comparison.png   : Coordinate Error CDF (RF, XGB, MTL-MLP, HierResMTL, Hybrid)
  6. p2_confusion_floor_hybrid.png : Side-by-side Phase 1 XGB vs Phase 2 Hybrid Floor confusion matrices
  7. p2_error_map_hybrid.png       : Spatial coordinate error map on validation buildings
  8. p2_error_vectors_hybrid.png   : Worst 10% error displacement vectors
  9. p2_robustness_by_phone.png    : Per-phone & Seen vs Unseen handset error comparison
"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import OUT_DIR, ROOT

P2_OUT_DIR = OUT_DIR / "phase2"
REPORT_FIGS_DIR = ROOT / "report" / "figs_phase2"


def euclidean(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=1))


def plot_architecture_diagram():
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc="#F3F6FA", ec="#2C4A70", title_c="#102A43"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=fc, edgecolor=ec, linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.25, title, ha="center", va="top",
                fontsize=9.5, fontweight="bold", color=title_c)
        ax.text(x + w / 2, y + h / 2 - 0.14, body, ha="center", va="center",
                fontsize=8.0, color="#243B53", linespacing=1.25)

    def arrow(x1, y1, x2, y2, color="#334E68"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.35",
                                    color=color, lw=1.4))

    # Stage 0: Input R3
    box(0.2, 1.35, 2.15, 2.25, "1. Device-Invariant R3",
        "465 Pos-Powed RSSI\n465 Scan-Relative Decay\n465 Binary Presence\n5 Physical Aggregates\n+ Strongest-AP Embed\n(Blind scans filtered)",
        fc="#EEF4FB", ec="#2B6CB0")

    # Stage 1: DL Trunk
    box(2.75, 1.55, 2.05, 1.85, "2. HierResMTL Trunk",
        "Proj + LayerNorm + GELU\n2x Pre-LN ResBlocks\nAP-Block Dropout (12%)\nGaussian Jitter (0.02)\nKendall Uncertainty Loss",
        fc="#EDF7F6", ec="#0F766E")

    # Hierarchical Heads
    box(5.2, 3.25, 2.0, 1.15, "Building Head",
        "3-way Softmax p_b\n-> FiLM(p_b) modulates\nFloor & Coord Trunk",
        fc="#FFF8EB", ec="#B7791F")
    box(5.2, 1.80, 2.0, 1.10, "Floor Head",
        "FiLM(h, p_b) + p_b\n5-way Softmax p_f\n(Floor-4 Inv-Sqrt Wt)",
        fc="#FFF8EB", ec="#B7791F")
    box(5.2, 0.35, 2.0, 1.15, "Coordinate Head",
        "FiLM(h, p_b, p_f)\n-> 64-d Bottleneck z\n-> Parametric (x, y)_DL",
        fc="#FFF8EB", ec="#B7791F")

    # Classical ML Hybrid Partners
    box(7.65, 2.55, 1.9, 1.85, "3. Classical ML",
        "OOF Log-Loss Fusion:\n• RF + R3-kNN (Bld)\n• R3-XGB + R3-kNN (Flr)\n+ Valid Floor Mask:\n(Floor 4 only in Bld 2)",
        fc="#F9F0FF", ec="#6B46C1")
    box(7.65, 0.35, 1.9, 1.85, "4. Hier-R3-kNN",
        "Building-Conditioned\nCosine Retrieval (k=7)\nin R3 Pow+Rel Space\nLocal Corridor Sharpness\n+ max_sim Confidence",
        fc="#F9F0FF", ec="#6B46C1")

    # Adaptive Gate Output
    box(9.95, 1.25, 1.35, 2.40, "Hybrid Output",
        "Bld: 100.0%\nFlr: 95.8%\n\nAdaptive Gate:\n(1-w)*y_DL\n+ w*y_kNN\nw(max_sim)\n\nEvAAL: 6.88m",
        fc="#ECFDF5", ec="#047857", title_c="#064E3B")

    arrow(2.35, 2.48, 2.75, 2.48)
    arrow(4.80, 2.90, 5.20, 3.75)
    arrow(4.80, 2.45, 5.20, 2.35)
    arrow(4.80, 2.05, 5.20, 0.95)
    arrow(6.20, 3.25, 6.20, 2.90, color="#B7791F")
    arrow(6.20, 1.80, 6.20, 1.50, color="#B7791F")

    arrow(7.20, 3.75, 7.65, 3.65)
    arrow(7.20, 2.35, 7.65, 3.15)
    arrow(7.20, 0.95, 9.95, 2.15, color="#0F766E")
    arrow(8.60, 2.55, 8.60, 2.20, color="#6B46C1")
    arrow(9.55, 3.35, 9.95, 2.90)
    arrow(9.55, 1.25, 9.95, 1.85)

    ax.set_title("Proposed Phase 2 Architecture: HierResMTL Deep Trunk + Adaptive ML/DL Hybrid",
                 fontsize=11.5, fontweight="bold", pad=8)
    fig.tight_layout()
    out = P2_OUT_DIR / "p2_architecture_diagram.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_training_curves():
    with open(P2_OUT_DIR / "training_histories.json") as f:
        hists = json.load(f)["HierResMTL"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(hists)))

    # Panel A: Train Loss
    for i, h in enumerate(hists):
        ep = np.arange(1, len(h["train_loss"]) + 1)
        axes[0].plot(ep, h["train_loss"], label=f"Fold {i}", color=colors[i], lw=1.6)
    axes[0].set_title("HierResMTL Multitask Training Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Kendall Weighted Loss")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    # Panel B: USERID-CV EvAAL convergence
    for i, h in enumerate(hists):
        ep = np.arange(1, len(h["es_evaal"]) + 1)
        axes[1].plot(ep, h["es_evaal"], label=f"Fold {i} (best {h['best_evaal']:.2f}m)", color=colors[i], lw=1.6)
    axes[1].set_title("Held-Out USERID Fold EvAAL Convergence")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("EvAAL Combined Error (m)")
    axes[1].set_ylim(5, 35)
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    # Panel C: Mean learned Kendall task sigmas across folds
    sigmas_arr = np.array([h["kendall_sigmas"] for h in hists])  # (5, epochs, 3)
    mean_sig = sigmas_arr.mean(axis=0)
    ep = np.arange(1, mean_sig.shape[0] + 1)
    for k, lbl, c in [(0, "Building σ_bld", "#2B6CB0"), (1, "Floor σ_flr", "#D97706"), (2, "Coords σ_crd", "#059669")]:
        axes[2].plot(ep, mean_sig[:, k], label=lbl, color=c, lw=2.0)
    axes[2].set_title("Learned Kendall Uncertainty Weights (Mean)")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Task Uncertainty σ")
    axes[2].legend(fontsize=8.5); axes[2].grid(alpha=0.3)

    fig.tight_layout()
    out = P2_OUT_DIR / "p2_training_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_model_comparison():
    df_val = pd.read_csv(P2_OUT_DIR / "metrics_official_val.csv")
    df_cv = pd.read_csv(P2_OUT_DIR / "metrics_5fold_cv.csv")

    order = [
        "Phase 1: RandomForest",
        "Phase 1: XGBoost",
        "DL: SingleTaskMLP",
        "DL: MTL-MLP (Phase 1 Plan)",
        "DL: SAE-MTL (Nowicki 2017)",
        "DL: CNN1D-MTL (CNNLoc 2019)",
        "DL: HierResMTL (Proposed)",
        "Hybrid: Naive Blend (0.5 RF+DL)",
        "Hybrid: HierResMTL-Hybrid (Ours)",
    ]
    sub_v = df_val.set_index("model").loc[order].reset_index()
    sub_c = df_cv.set_index("model").loc[order].reset_index()

    short_names = [
        "P1 RF", "P1 XGB", "SingleTask\nMLP", "MTL-MLP",
        "SAE-MTL\n(2017)", "CNN1D-MTL\n(2019)", "HierResMTL\n(DL)",
        "Naive Blend\n(Strawman)", "HierResMTL-\nHybrid"
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    x = np.arange(len(short_names))
    palette = ["#94A3B8", "#64748B", "#93C5FD", "#60A5FA", "#3B82F6", "#2563EB", "#0F766E", "#F59E0B", "#047857"]

    # Panel A: EvAAL Score on Official Val vs 5-Fold OOF
    w = 0.38
    b1 = axes[0].bar(x - w/2, sub_v["combined_mean_m"], width=w, label="Official Val (1,111)", color=palette, edgecolor="black", lw=0.6)
    b2 = axes[0].bar(x + w/2, sub_c["oof_evaal_m"], width=w, label="5-Fold USERID OOF (19,300)", color=palette, alpha=0.55, edgecolor="black", lw=0.6, hatch="//")
    axes[0].set_xticks(x); axes[0].set_xticklabels(short_names, rotation=28, ha="right", fontsize=8.5)
    axes[0].set_ylabel("EvAAL Combined Error (m)  [Lower = Better]")
    axes[0].set_title("EvAAL Positioning Score: Official Val vs 5-Fold USERID OOF")
    axes[0].legend(fontsize=8.5); axes[0].grid(axis="y", alpha=0.3)
    for bar in b1:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.25, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold")

    # Panel B: Floor Accuracy on Official Val
    bars = axes[1].bar(x, sub_v["floor_acc"] * 100, width=0.6, color=palette, edgecolor="black", lw=0.6)
    axes[1].set_xticks(x); axes[1].set_xticklabels(short_names, rotation=28, ha="right", fontsize=8.5)
    axes[1].set_ylabel("Floor Accuracy (%)  [Higher = Better]")
    axes[1].set_ylim(85, 98)
    axes[1].set_title("Official Validation Floor Classification Accuracy (%)")
    axes[1].grid(axis="y", alpha=0.3)
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    fig.tight_layout()
    out = P2_OUT_DIR / "p2_model_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_ablation_study():
    df = pd.read_csv(P2_OUT_DIR / "ablation_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

    # Left: Input Representation Ablation
    rep_df = df[df["group"] == "Representation"].copy()
    labels_rep = ["R1: Phase1\nStdScaler", "R2: Pos-Powed\nRSSI", "R3-noRel:\nPow+Mask+Agg", "R3 (Full):\n+Scan-Relative"]
    colors_rep = ["#94A3B8", "#60A5FA", "#3B82F6", "#0F766E"]
    y_pos = np.arange(len(rep_df))
    bars = axes[0].barh(y_pos, rep_df["val_evaal_m"], color=colors_rep, edgecolor="black", lw=0.7)
    axes[0].set_yticks(y_pos); axes[0].set_yticklabels(labels_rep, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Official Val EvAAL Error (m)  [Lower = Better]")
    axes[0].set_title("Input Representation Ablation (Fold-0 Trained)")
    axes[0].grid(axis="x", alpha=0.3)
    for b, row in zip(bars, rep_df.itertuples()):
        axes[0].text(b.get_width() + 0.5, b.get_y() + b.get_height()/2,
                     f"{row.val_evaal_m:.2f}m (Flr {row.val_flr_acc*100:.1f}%)", va="center", fontsize=8.2)
    axes[0].set_xlim(0, max(rep_df["val_evaal_m"]) * 1.32)

    # Right: Regularization & Mechanism Ablation (compare against R3 Full)
    full_row = df[df["ablation"].str.contains("R3 \\(Full\\)")].iloc[0]
    mech_df = pd.concat([df[df["ablation"].str.contains("R3 \\(Full\\)")], df[df["group"] != "Representation"]], ignore_index=True)
    y_pos2 = np.arange(len(mech_df))
    colors_m = ["#0F766E"] + ["#F59E0B" if g == "Mechanism" else "#EF4444" for g in mech_df["group"].iloc[1:]]
    bars2 = axes[1].barh(y_pos2, mech_df["val_evaal_m"], color=colors_m, edgecolor="black", lw=0.7)
    axes[1].axvline(full_row["val_evaal_m"], color="#0F766E", ls="--", lw=1.2)
    axes[1].set_yticks(y_pos2); axes[1].set_yticklabels(mech_df["ablation"], fontsize=8.8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Official Val EvAAL Error (m)  [Lower = Better]")
    axes[1].set_title("Regularization & Architectural Ablations (Single-Factor Drop)")
    axes[1].grid(axis="x", alpha=0.3)
    for b, row in zip(bars2, mech_df.itertuples()):
        delta = row.val_evaal_m - full_row["val_evaal_m"]
        tag = "baseline" if delta == 0 else f"{delta:+.2f}m"
        axes[1].text(b.get_width() + 0.15, b.get_y() + b.get_height()/2,
                     f"{row.val_evaal_m:.2f}m ({tag})", va="center", fontsize=8.0)
    axes[1].set_xlim(0, max(mech_df["val_evaal_m"]) * 1.25)

    fig.tight_layout()
    out = P2_OUT_DIR / "p2_ablation_study.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_error_cdf():
    d = np.load(P2_OUT_DIR / "all_predictions.npz")
    tc = d["val_coords"]

    curves = [
        ("Phase 1 XGBoost", d["XGBoost_val_c"], "#94A3B8", "--"),
        ("Phase 1 RandomForest", d["RandomForest_val_c"], "#475569", "-."),
        ("DL: MTL-MLP", d["MTL_MLP_val_pc"], "#3B82F6", ":"),
        ("DL: HierResMTL (Ours)", d["HierResMTL_val_pc"], "#0F766E", "-"),
        ("Hybrid: HierResMTL-Hybrid", d["Hybrid_val_pc"], "#047857", "-"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for label, pc, color, ls in curves:
        err = euclidean(pc, tc)
        xs = np.sort(err)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        lw = 2.4 if "Hybrid" in label or "HierResMTL" in label else 1.6
        ax.plot(xs, ys, label=f"{label} (med {np.median(err):.2f}m, p90 {np.percentile(err,90):.2f}m)",
                color=color, ls=ls, lw=lw)

    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.axhline(0.9, color="gray", ls=":", lw=0.8)
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 1.01)
    ax.set_title("Coordinate Error CDF — P(error ≤ distance) on Official Val (n=1,111)")
    ax.set_xlabel("Euclidean Positioning Error (m)")
    ax.set_ylabel("Cumulative Fraction")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = P2_OUT_DIR / "p2_error_cdf_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_floor_confusion():
    d = np.load(P2_OUT_DIR / "all_predictions.npz")
    tf = d["val_floor"]
    pf_xgb = d["XGBoost_val_f"]
    pf_dl = d["HierResMTL_val_pf"].argmax(1)
    pf_hyb = d["Hybrid_val_pf"].argmax(1)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, pf, title in [
        (axes[0], pf_xgb, f"Phase 1 XGBoost (acc {(pf_xgb==tf).mean()*100:.2f}%)"),
        (axes[1], pf_dl, f"DL HierResMTL (acc {(pf_dl==tf).mean()*100:.2f}%)"),
        (axes[2], pf_hyb, f"HierResMTL-Hybrid (acc {(pf_hyb==tf).mean()*100:.2f}%)"),
    ]:
        cm = confusion_matrix(tf, pf, labels=range(5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=[f"F{i}" for i in range(5)],
                    yticklabels=[f"F{i}" for i in range(5)], ax=ax)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted Floor"); ax.set_ylabel("True Floor")

    fig.tight_layout()
    out = P2_OUT_DIR / "p2_confusion_floor_hybrid.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_spatial_diagnostics():
    d = np.load(P2_OUT_DIR / "all_predictions.npz")
    tc = d["val_coords"]
    pc = d["Hybrid_val_pc"]
    err = euclidean(pc, tc)

    # Error map
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(tc[:, 0], tc[:, 1], c=err, cmap="plasma",
                    s=15, vmin=0, vmax=np.percentile(err, 95))
    ax.set_title(f"HierResMTL-Hybrid: Coordinate Error Map (Val)\n"
                 f"mean {err.mean():.2f} m / median {np.median(err):.2f} m / p90 {np.percentile(err,90):.2f} m")
    ax.set_xlabel("LONGITUDE (m)"); ax.set_ylabel("LATITUDE (m)")
    ax.set_aspect("equal", adjustable="datalim")
    plt.colorbar(sc, ax=ax, label="Positioning Error (m)")
    fig.tight_layout()
    out1 = P2_OUT_DIR / "p2_error_map_hybrid.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)

    # Error vectors (worst 10%)
    k = max(1, int(len(err) * 0.10))
    worst = np.argsort(err)[-k:]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(tc[:, 0], tc[:, 1], s=6, color="lightgray", label="All val points")
    for i in worst:
        ax.annotate("", xy=pc[i], xytext=tc[i],
                    arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8))
    ax.set_title("HierResMTL-Hybrid: Worst 10% Displacements (True → Predicted)")
    ax.set_xlabel("LONGITUDE (m)"); ax.set_ylabel("LATITUDE (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    out2 = P2_OUT_DIR / "p2_error_vectors_hybrid.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    return out1, out2


def plot_robustness():
    rob = pd.read_csv(P2_OUT_DIR / "robustness_by_phone.csv")
    grp = pd.read_csv(P2_OUT_DIR / "robustness_seen_unseen.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1.8, 1.0]})

    # Left: Per-phone mean coordinate error
    fam_order = ["RandomForest", "XGBoost", "HierResMTL (DL)", "HierResMTL-Hybrid"]
    piv = rob.pivot(index="phone", columns="family", values="coord_mean_m")[fam_order]
    colors = ["#94A3B8", "#475569", "#0F766E", "#047857"]
    piv.plot(kind="bar", ax=axes[0], width=0.82, color=colors, edgecolor="black", lw=0.4)
    seen_set = set(rob[rob["device_status"].str.startswith("Seen")]["phone"].unique())
    axes[0].set_xticklabels([f"P{p}{'*' if p in seen_set else ''}" for p in piv.index], rotation=0, fontsize=8.5)
    axes[0].set_title("Per-Handset Mean Coordinate Error (* = Seen in Training)")
    axes[0].set_xlabel("Device PHONEID"); axes[0].set_ylabel("Mean Error (m)")
    axes[0].legend(title="Model", fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # Right: Seen vs Unseen group summary
    piv_g = grp.pivot(index="group", columns="family", values="coord_mean_m")[fam_order]
    piv_g.plot(kind="bar", ax=axes[1], width=0.72, color=colors, edgecolor="black", lw=0.5)
    axes[1].set_xticklabels(["Seen (397)", "Unseen (714)"], rotation=0, fontsize=9)
    axes[1].set_title("Hardware Shift: Seen vs Unseen Phones")
    axes[1].set_xlabel(""); axes[1].set_ylabel("Mean Error (m)")
    axes[1].legend().set_visible(False)
    axes[1].grid(axis="y", alpha=0.3)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.1f", fontsize=7.5, padding=2)

    fig.tight_layout()
    out = P2_OUT_DIR / "p2_robustness_by_phone.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    sns.set_theme(style="whitegrid")
    P2_OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGS_DIR.mkdir(parents=True, exist_ok=True)

    saved = [
        plot_architecture_diagram(),
        plot_training_curves(),
        plot_model_comparison(),
        plot_ablation_study(),
        plot_error_cdf(),
        plot_floor_confusion(),
        *plot_spatial_diagnostics(),
        plot_robustness(),
    ]

    for p in saved:
        shutil.copy2(p, REPORT_FIGS_DIR / p.name)
        print(f"  ✓ {p.relative_to(ROOT)}")
    print(f"\n✓ Copied {len(saved)} figures to {REPORT_FIGS_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

