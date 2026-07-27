"""
Generate the FC matrix figure and the density-sweep figure for the thesis.
Both figures use the same example subject and the same FC matrix.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import datasets, image
from nilearn.maskers import NiftiLabelsMasker

# =============================================================
# SETTINGS
# =============================================================
DATA_DIR   = config.NII_ROOT
OUT_DIR    = config.ensure(config.CROSS_DIRS["step0_subject_data"] / "visualizations" / "vis_fc_matrix_density_sweep_CP0189")
SUBJECT_ID = "CP0189"
DENSITIES  = [0.05, 0.10, 0.25, 0.50]   # boundaries of literature (10-25%) and broad (5-50%) ranges


def yeo_from_label(lab):
    parts = lab.split("_")
    return parts[2] if len(parts) >= 3 else "Unknown"


def proportional_threshold_positive(W, density):
    """
    Retain the top `density` proportion of upper-triangle edges,
    using positive-only strategy (negative values set to zero first).
    Returns a symmetric matrix.
    """
    W = W.copy()
    np.fill_diagonal(W, 0)
    W[W < 0] = 0  # positive-only

    triu_idx = np.triu_indices_from(W, k=1)
    edges    = W[triu_idx]
    n_total  = len(edges)
    n_keep   = int(np.round(density * n_total))

    if n_keep == 0:
        return np.zeros_like(W)

    sorted_edges = np.sort(edges)[::-1]
    threshold    = sorted_edges[n_keep - 1]

    W_thresh = np.where(W >= threshold, W, 0)
    np.fill_diagonal(W_thresh, 0)
    return W_thresh


def main():
    # =============================================================
    # LOAD DATA + ATLAS + EXTRACT TIME SERIES
    # =============================================================
    print(f"Loading BOLD for {SUBJECT_ID} ...")
    subject_dir = DATA_DIR / SUBJECT_ID
    bold_path   = sorted(list(subject_dir.glob("*.nii*")))[0]
    bold_img    = nib.load(str(bold_path))
    TR          = float(bold_img.header.get_zooms()[3])
    print(f"  TR = {TR}s, shape = {bold_img.shape}")

    print("Fetching Schaefer-400 ...")
    schaefer = datasets.fetch_atlas_schaefer_2018(
        n_rois=400, yeo_networks=7, resolution_mm=2
    )
    atlas_img  = nib.load(schaefer.maps)
    roi_labels = [lab.decode() if isinstance(lab, bytes) else lab
                  for lab in schaefer.labels]
    if len(roi_labels) == 401:
        roi_labels = roi_labels[1:]

    atlas_resampled = image.resample_to_img(
        atlas_img, bold_img, interpolation="nearest"
    )

    print("Extracting ROI time series ...")
    masker = NiftiLabelsMasker(
        labels_img=atlas_resampled,
        standardize="zscore_sample",
        detrend=True,
        t_r=TR,
        memory=None,
        verbose=0,
    )
    ts_all = masker.fit_transform(bold_img)
    print(f"  time series shape: {ts_all.shape}")

    # =============================================================
    # SORT ROIs BY YEO NETWORK
    # =============================================================
    yeo_per_roi = np.array([yeo_from_label(l) for l in roi_labels])
    yeo_order   = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
    yeo_colors  = {
        "Vis":         "#781286",
        "SomMot":      "#4682B4",
        "DorsAttn":    "#00760E",
        "SalVentAttn": "#C43AFA",
        "Limbic":      "#DCF8A4",
        "Cont":        "#E69422",
        "Default":     "#CD3E4E",
    }

    sort_idx   = np.argsort(
        [yeo_order.index(y) if y in yeo_order else 99 for y in yeo_per_roi]
    )
    ts_sorted  = ts_all[:, sort_idx]
    yeo_sorted = yeo_per_roi[sort_idx]

    block_bounds = {}
    for y in yeo_order:
        idx = np.where(yeo_sorted == y)[0]
        if len(idx) > 0:
            block_bounds[y] = (idx[0], idx[-1] + 1)

    # =============================================================
    # FC MATRIX (Pearson, sorted by Yeo)
    # =============================================================
    print("Computing FC matrix ...")
    fc_pearson = np.corrcoef(ts_sorted.T)
    np.fill_diagonal(fc_pearson, 0.0)

    print("Applying density thresholds ...")
    thresholded = {d: proportional_threshold_positive(fc_pearson, d) for d in DENSITIES}

    # =============================================================
    # FIGURE 2: FULL FC MATRIX
    # =============================================================
    print("Generating Figure 2 (FC matrix) ...")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(fc_pearson, cmap="RdBu_r", vmin=-0.8, vmax=0.8,
                   interpolation="nearest", aspect="equal")
    ax.set_xlabel("ROI index (sorted by Yeo network)")
    ax.set_ylabel("ROI index (sorted by Yeo network)")

    for y in yeo_order:
        if y not in block_bounds:
            continue
        start, end = block_bounds[y]
        ax.axhline(end - 0.5, color="black", linewidth=0.4, alpha=0.4)
        ax.axvline(end - 0.5, color="black", linewidth=0.4, alpha=0.4)
        mid = (start + end) / 2
        ax.text(mid, -8, y, ha="center", va="bottom",
                fontsize=8, color=yeo_colors[y], fontweight="bold",
                rotation=45)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure2_fc_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {OUT_DIR / 'figure2_fc_matrix.png'}")

    # =============================================================
    # FIGURE 3: DENSITY SWEEP (2x2 panel)
    # =============================================================
    print("Generating Figure 3 (density sweep) ...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes_flat = axes.flatten()

    for ax, d in zip(axes_flat, DENSITIES):
        W = thresholded[d]
        im = ax.imshow(W, cmap="Reds", vmin=0, vmax=0.8,
                       interpolation="nearest", aspect="equal")
        ax.set_title(f"{int(d*100)} % density", fontsize=13)
        ax.set_xlabel("ROI index (sorted)")
        ax.set_ylabel("ROI index (sorted)")

        for y in yeo_order:
            if y not in block_bounds:
                continue
            start, end = block_bounds[y]
            ax.axhline(end - 0.5, color="black", linewidth=0.4, alpha=0.3)
            ax.axvline(end - 0.5, color="black", linewidth=0.4, alpha=0.3)

    # shared colorbar on the right
    cbar = fig.colorbar(im, ax=axes_flat.tolist(), fraction=0.025, pad=0.04)
    cbar.set_label("Pearson r (retained edges)")

    fig.savefig(OUT_DIR / "figure3_density_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {OUT_DIR / 'figure3_density_sweep.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
