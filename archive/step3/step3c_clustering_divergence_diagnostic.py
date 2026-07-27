"""
Diagnostic: why does Schaefer-400 mean clustering diverge from the coarser atlases?
Per atlas, across the density sweep: mean degree, weighted vs binary clustering,
and the retained-weight threshold. Aggregated across all subjects (no group split).

Self-validation: recomputed weighted clustering is overlaid with the stored
step3c mean_clustering to confirm the construction matches the productive pipeline.

Outputs:
    - clustering_resolution_diagnostic.csv
    - clustering_resolution_diagnostic.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from comet.graph import bct

# =============================================================
# SETTINGS
# =============================================================
MATRIX_DIRS = {
    "Schaefer-400": config.atlas_dir("schaefer400", "step2_pipeline") / "comet_matrices",
    "Schaefer-100": config.atlas_dir("schaefer100", "step2_pipeline") / "comet_matrices",
    "AAL":          config.atlas_dir("aal",         "step2_pipeline") / "comet_matrices",
}

# stored productive curve for self-validation (from the slide-C plotting script)
STEP3C_CSVS = {
    "Schaefer-400": config.atlas_dir("schaefer400", "step3c_metrics") / "step3c_aggregated.csv",
    "Schaefer-100": config.atlas_dir("schaefer100", "step3c_metrics") / "step3c_aggregated.csv",
    "AAL":          config.atlas_dir("aal",         "step3c_metrics") / "step3c_aggregated.csv",
}

OUT_DIR = config.ensure(config.CROSS_DIRS["step3c_clustering_diagnostic"])

DENSITIES = np.round(np.arange(0.05, 0.501, 0.05), 2)   # 5..50 %, matches the slide
SAMPLE_N = None        # None = all subjects; set e.g. 30 for a fast check
WEIGHT_NORM = "max"    # normalise W -> W/W.max() before clustering_coef_wu (BCT needs [0,1])

ATLAS_COLORS = {"Schaefer-400": "#1f77b4", "Schaefer-100": "#ff7f0e", "AAL": "#2ca02c"}

# =============================================================
# Positive-only proportional threshold
# Local diagnostic copy: operates on raw r, intentionally kept separate from the
# canonical step3a implementation (this is what produces the documented offset
# vs. the stored step3c curve in the self-validation overlay).
# =============================================================
def proportional_threshold(W, density):
    W = W.copy()
    np.fill_diagonal(W, 0.0)
    W[W < 0] = 0.0                       # positive-only
    iu = np.triu_indices_from(W, k=1)
    w = W[iu]
    n_keep = int(round(density * w.size))
    if n_keep < 1:
        return np.zeros_like(W)
    keep = np.argsort(w)[::-1][:n_keep]
    mask = np.zeros(w.size, bool); mask[keep] = True
    Wt = np.zeros_like(W)
    Wt[iu[0][mask], iu[1][mask]] = w[mask]
    return Wt + Wt.T


def load_step3c_wcc(path):
    """Return (density_pct, mean_clustering_mean) from the hybrid-header step3c CSV."""
    df = pd.read_csv(path, header=[0, 1])
    df.columns = [f"{a}_{b}" if not str(b).startswith("Unnamed") else a for a, b in df.columns]
    df = df.rename(columns={"Unnamed: 0_level_0": "density", "Unnamed: 1_level_0": "group"})
    df = df.iloc[1:]
    df["density"] = pd.to_numeric(df["density"], errors="coerce")
    df["mean_clustering_mean"] = pd.to_numeric(df["mean_clustering_mean"], errors="coerce")
    df = df.dropna(subset=["density"]).groupby("density", as_index=False)["mean_clustering_mean"].mean()
    return df["density"].values * 100, df["mean_clustering_mean"].values


# =============================================================
# COMPUTE
# =============================================================
records = []
for atlas, mdir in MATRIX_DIRS.items():
    files = sorted(glob.glob(str(mdir / "CP*_connectivity_comet.npy")))
    subj = files
    if SAMPLE_N:
        subj = subj[:SAMPLE_N]
    if not subj:
        print(f"  [skip] no matrices in {mdir}")
        continue
    print(f"{atlas}: {len(subj)} subjects")

    for dens in DENSITIES:
        wcc, bcc, deg, thr = [], [], [], []
        for f in subj:
            W = np.load(f)
            Wt = proportional_threshold(W, dens)            # positive-only, thresholded
            A = (Wt > 0).astype(float)                       # binary graph
            # weighted clustering (Onnela); normalise to [0,1]
            Wn = Wt / Wt.max() if (WEIGHT_NORM == "none" and Wt.max() > 0) else Wt
            wcc.append(np.nanmean(bct.clustering_coef_wu(Wn)))
            bcc.append(np.nanmean(bct.clustering_coef_bu(A)))
            deg.append(A.sum(1).mean())
            pos = Wt[Wt > 0]
            thr.append(pos.min() if pos.size else np.nan)    # weakest retained edge (r)
        records.append({
            "atlas": atlas, "density": dens,
            "wcc_mean": np.mean(wcc), "bcc_mean": np.mean(bcc),
            "mean_degree": np.mean(deg), "min_retained_r": np.nanmean(thr),
        })
        print(f"  d={dens:.2f}  wCC={np.mean(wcc):.3f}  bCC={np.mean(bcc):.3f}  "
              f"k={np.mean(deg):.1f}  r_min={np.nanmean(thr):.3f}")

res = pd.DataFrame(records)
res.to_csv(OUT_DIR / "clustering_resolution_diagnostic.csv", index=False)
print(f"\nSaved CSV: {OUT_DIR / 'clustering_resolution_diagnostic.csv'}")


# =============================================================
# FIGURE (2 x 2)
# =============================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
panels = [
    ("wcc_mean",     "Weighted clustering (Onnela)", axes[0, 0]),
    ("bcc_mean",     "Binary clustering",            axes[0, 1]),
    ("mean_degree",  "Mean degree  (k ≈ density·(N−1))", axes[1, 0]),
    ("min_retained_r", "Weakest retained edge weight (r)", axes[1, 1]),
]
for col, title, ax in panels:
    for atlas in MATRIX_DIRS:
        d = res[res["atlas"] == atlas]
        ax.plot(d["density"] * 100, d[col], marker="o", lw=2, ms=5,
                color=ATLAS_COLORS[atlas], label=atlas)
    ax.set_xlabel("Density (%)"); ax.set_ylabel(title); ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
axes[0, 0].legend(fontsize=8, title="Atlas")

# self-validation overlay on the weighted-clustering panel
for atlas, p in STEP3C_CSVS.items():
    if Path(p).exists():
        x, y = load_step3c_wcc(p)
        axes[0, 0].plot(x, y, ls="--", lw=1.2, color=ATLAS_COLORS[atlas], alpha=0.7)
axes[0, 0].plot([], [], ls="--", color="grey", label="step3c (stored)")
axes[0, 0].legend(fontsize=8, title="Atlas")

fig.suptitle("Clustering resolution diagnostic — solid: recomputed, dashed: stored step3c",
             fontsize=12, y=1.0)
fig.tight_layout()
fig.savefig(OUT_DIR / "clustering_resolution_diagnostic.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved figure: {OUT_DIR / 'clustering_resolution_diagnostic.png'}")