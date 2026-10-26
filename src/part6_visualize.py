"""Part 6 of the ML lifecycle: Visualization (the brief's figure deliverables).

Reads outputs/predictions.npz and produces, per model family:
  1. Floor confusion matrix (counts, with accuracy in title)
  2. Coordinate error map  -- validation points plotted at their TRUE location,
     coloured by positioning error (m); lets us see WHERE the model is weak.
  3. Error vector map       -- true -> predicted arrows for the worst cases.
And one shared figure:
  4. Error CDF comparison   -- P(error <= d) for both families on one axis.

All figures saved to outputs/.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import OUT_DIR

FAMILIES = ["RandomForest", "XGBoost"]
N_FLOORS = 5


def euclidean(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=1))


def plot_confusion(tf, pf, fam):
    cm = confusion_matrix(tf, pf, labels=range(N_FLOORS))
    acc = (tf == pf).mean()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=[f"F{i}" for i in range(N_FLOORS)],
                yticklabels=[f"F{i}" for i in range(N_FLOORS)], ax=ax)
    ax.set_title(f"{fam}: floor confusion (val acc {acc:.3f})")
    ax.set_xlabel("predicted floor"); ax.set_ylabel("true floor")
    fig.tight_layout()
    out = OUT_DIR / f"confusion_floor_{fam}.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


def plot_error_map(true_xy, err, fam):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sc = ax.scatter(true_xy[:, 0], true_xy[:, 1], c=err, cmap="plasma",
                    s=14, vmin=0, vmax=np.percentile(err, 95))
    ax.set_title(f"{fam}: coordinate error map (val)\n"
                 f"mean {err.mean():.1f} m / median {np.median(err):.1f} m")
    ax.set_xlabel("LONGITUDE (m)"); ax.set_ylabel("LATITUDE (m)")
    ax.set_aspect("equal", adjustable="datalim")
    plt.colorbar(sc, ax=ax, label="positioning error (m)")
    fig.tight_layout()
    out = OUT_DIR / f"error_map_{fam}.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


def plot_error_vectors(true_xy, pred_xy, err, fam, worst_frac=0.1):
    """Arrows true->pred for the worst `worst_frac` of predictions."""
    k = max(1, int(len(err) * worst_frac))
    worst = np.argsort(err)[-k:]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(true_xy[:, 0], true_xy[:, 1], s=6, color="lightgray",
               label="all val points")
    for i in worst:
        ax.annotate("", xy=pred_xy[i], xytext=true_xy[i],
                    arrowprops=dict(arrowstyle="->", color="crimson", lw=0.7))
    ax.set_title(f"{fam}: worst {int(worst_frac*100)}% errors (true -> predicted)")
    ax.set_xlabel("LONGITUDE (m)"); ax.set_ylabel("LATITUDE (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / f"error_vectors_{fam}.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


def plot_error_cdf(errs_by_fam):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for fam, err in errs_by_fam.items():
        xs = np.sort(err)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, label=f"{fam} (median {np.median(err):.1f} m)")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xlim(0, np.percentile(np.concatenate(list(errs_by_fam.values())), 98))
    ax.set_title("Coordinate error CDF  —  P(error ≤ distance)")
    ax.set_xlabel("positioning error (m)"); ax.set_ylabel("cumulative fraction")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "error_cdf_comparison.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


def main():
    sns.set_theme(style="whitegrid")
    d = np.load(OUT_DIR / "predictions.npz")
    tf = d["truth_floor"]
    tc = d["truth_coords"]

    saved = []
    errs_by_fam = {}
    for fam in FAMILIES:
        pf = d[f"{fam}_floor"]
        pc = d[f"{fam}_coords"]
        err = euclidean(pc, tc)
        errs_by_fam[fam] = err
        saved += [
            plot_confusion(tf, pf, fam),
            plot_error_map(tc, err, fam),
            plot_error_vectors(tc, pc, err, fam),
        ]
    saved.append(plot_error_cdf(errs_by_fam))

    print("Saved figures:")
    for p in saved:
        print(f"  {p.relative_to(OUT_DIR.parent)}")


if __name__ == "__main__":
    main()
