"""
Visualisation for REPRESENTATION slide:
4-step sequence showing how 4D BOLD becomes an FC matrix.

Sequence:
1. Mean BOLD volume (input)
2. Schaefer-400 parcellation overlay (brain → 400 ROIs)
3. 7 example ROI time series (one per Yeo network)
4. 400×400 FC matrix sorted by Yeo network

Example subject: CP0189 (representative PASS subject).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from nilearn import datasets, plotting, image
from nilearn.maskers import NiftiLabelsMasker

# =============================================================
# SETTINGS
# =============================================================
DATA_DIR   = config.NII_ROOT
OUT_DIR    = config.ensure(config.CROSS_DIRS["step0_subject_data"] / "visualizations" / "pipeline_BOLD_to_matrix")
SUBJECT_ID = "CP0189"
FIXED_Z    = 40            # axial slice for consistency with QC visuals


def main():
    # =============================================================
    # LOAD DATA
    # =============================================================
    print(f"Loading BOLD for {SUBJECT_ID} ...")
    subject_dir = DATA_DIR / SUBJECT_ID
    bold_path   = sorted(list(subject_dir.glob("*.nii*")))[0]
    bold_img    = nib.load(str(bold_path))
    TR          = float(bold_img.header.get_zooms()[3])
    print(f"  TR = {TR}s, shape = {bold_img.shape}")

    print("Fetching Schaefer-400 (7 networks) ...")
    schaefer = datasets.fetch_atlas_schaefer_2018(
        n_rois=400, yeo_networks=7, resolution_mm=2
    )
    atlas_img    = nib.load(schaefer.maps)
    roi_labels   = [lab.decode() if isinstance(lab, bytes) else lab
                    for lab in schaefer.labels]

    # Schaefer ships with an extra background label at index 0 — remove it,
    # so roi_labels aligns with the 400 ROI columns produced by NiftiLabelsMasker
    if len(roi_labels) == 401:
        roi_labels = roi_labels[1:]

    print(f"  atlas shape: {atlas_img.shape}, n_labels: {len(roi_labels)}")

    # resample atlas to BOLD space (Schaefer is in MNI 2mm, BOLD may differ)
    print("Resampling atlas to BOLD grid ...")
    atlas_resampled = image.resample_to_img(
        atlas_img, bold_img, interpolation="nearest"
    )

    # =============================================================
    # EXTRACT ROI TIME SERIES (all 400 ROIs)
    # =============================================================
    print("Extracting ROI time series via NiftiLabelsMasker ...")
    masker = NiftiLabelsMasker(
        labels_img=atlas_resampled,
        standardize="zscore_sample",
        detrend=True,
        t_r=TR,
        memory=None,
        verbose=0,
    )
    ts_all = masker.fit_transform(bold_img)   # shape: (n_timepoints, 400)
    print(f"  time series shape: {ts_all.shape}")

    # =============================================================
    # YEO NETWORK MAPPING
    # =============================================================
    # Schaefer label format: '7Networks_LH_Vis_1' → network = 'Vis'
    def yeo_from_label(lab):
        parts = lab.split("_")
        return parts[2] if len(parts) >= 3 else "Unknown"

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

    # sort indices by Yeo network for block-structured FC matrix
    sort_idx = np.argsort(
        [yeo_order.index(y) if y in yeo_order else 99 for y in yeo_per_roi]
    )
    ts_sorted        = ts_all[:, sort_idx]
    yeo_sorted       = yeo_per_roi[sort_idx]

    # block boundaries for FC matrix annotation
    block_bounds = {}
    for y in yeo_order:
        idx = np.where(yeo_sorted == y)[0]
        if len(idx) > 0:
            block_bounds[y] = (idx[0], idx[-1] + 1)

    # pick one representative ROI per Yeo network (medoid: closest to network mean ts)
    example_roi_idx_by_yeo = {}
    for y in yeo_order:
        net_idx = np.where(yeo_per_roi == y)[0]
        if len(net_idx) == 0:
            continue
        net_ts   = ts_all[:, net_idx]
        net_mean = net_ts.mean(axis=1, keepdims=True)
        dists    = np.linalg.norm(net_ts - net_mean, axis=0)
        medoid   = net_idx[np.argmin(dists)]
        example_roi_idx_by_yeo[y] = medoid
        print(f"  example ROI {y}: index {medoid} ({roi_labels[medoid]})")

    # =============================================================
    # FC MATRIX (Pearson + Fisher-z, all 400 ROIs sorted by Yeo)
    # =============================================================
    print("Computing FC matrix ...")
    fc_pearson = np.corrcoef(ts_sorted.T)
    np.fill_diagonal(fc_pearson, 0.0)         # zero diagonal for visualisation

    # =============================================================
    # PLOT 1: MEAN BOLD VOLUME
    # =============================================================
    print("\nGenerating plots ...")
    mean_bold = image.mean_img(bold_img, copy_header=True)

    fig = plt.figure(figsize=(6, 5))
    plotting.plot_epi(
        mean_bold, display_mode="z", cut_coords=[FIXED_Z],
        title=f"Step 1: 4D BOLD → mean volume\n({SUBJECT_ID})",
        figure=fig,
    )
    fig.savefig(OUT_DIR / "01_mean_bold.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # =============================================================
    # PLOT 2: ATLAS OVERLAY
    # =============================================================
    fig = plt.figure(figsize=(6, 5))
    plotting.plot_roi(
        atlas_resampled, bg_img=mean_bold,
        display_mode="z", cut_coords=[FIXED_Z],
        title="Step 2: Schaefer-400 parcellation\n(400 cortical ROIs, 7 Yeo networks)",
        figure=fig, alpha=0.7,
    )
    fig.savefig(OUT_DIR / "02_atlas_overlay.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # =============================================================
    # PLOT 3: 7 EXAMPLE ROI TIME SERIES (medoid + all other ROIs in grey)
    # =============================================================
    fig, ax = plt.subplots(figsize=(11, 7))

    n_timepoints = ts_all.shape[0]
    time_axis    = np.arange(n_timepoints) * TR

    offset = 0
    y_ticks, y_tick_labels = [], []
    spacing = 5

    for y in yeo_order:
        if y not in example_roi_idx_by_yeo:
            continue
        medoid_idx = example_roi_idx_by_yeo[y]
        net_idx    = np.where(yeo_per_roi == y)[0]

        # plot all other ROIs of this network in grey, behind the medoid
        for roi_idx in net_idx:
            if roi_idx == medoid_idx:
                continue
            ax.plot(time_axis, ts_all[:, roi_idx] + offset,
                    color="grey", linewidth=0.4, alpha=0.25, zorder=1)

        # plot the medoid on top in network colour
        ax.plot(time_axis, ts_all[:, medoid_idx] + offset,
                color=yeo_colors[y], linewidth=1.4, alpha=1.0, zorder=2)

        y_ticks.append(offset)
        y_tick_labels.append(y)
        offset -= spacing

    ax.set_xlabel("Time (s)")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    ax.set_title(f"Step 3: ROI time series — 1 example (coloured) + all other ROIs (grey)\n"
                 f"{SUBJECT_ID}, z-scored & detrended")
    ax.set_xlim(0, time_axis[-1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_roi_timeseries.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # =============================================================
    # PLOT 4: FC MATRIX (400×400, sorted by Yeo network)
    # =============================================================
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(fc_pearson, cmap="RdBu_r", vmin=-0.8, vmax=0.8,
                   interpolation="nearest", aspect="equal")
    ax.set_title(f"Step 4: 400×400 FC matrix (Pearson r)\n"
                 f"{SUBJECT_ID}, ROIs sorted by Yeo network")
    ax.set_xlabel("ROI index (sorted)")
    ax.set_ylabel("ROI index (sorted)")

    # draw block boundary lines + label networks on top/left
    for y in yeo_order:
        if y not in block_bounds:
            continue
        start, end = block_bounds[y]
        # block boundary
        ax.axhline(end - 0.5, color="black", linewidth=0.4, alpha=0.4)
        ax.axvline(end - 0.5, color="black", linewidth=0.4, alpha=0.4)
        # label at top
        mid = (start + end) / 2
        ax.text(mid, -8, y, ha="center", va="bottom",
                fontsize=8, color=yeo_colors[y], fontweight="bold",
                rotation=45)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_fc_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # =============================================================
    # COMBINED 4-PANEL FIGURE (for the slide)
    # =============================================================
    print("Building combined 4-panel figure ...")
    fig = plt.figure(figsize=(20, 14))
    gs  = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    # top-left: mean BOLD
    ax1 = fig.add_subplot(gs[0, 0])
    disp1 = plotting.plot_epi(
        mean_bold, display_mode="z", cut_coords=[FIXED_Z],
        axes=ax1, title="1. 4D BOLD → mean volume",
        colorbar=False,
    )

    # top-right: atlas overlay
    ax2 = fig.add_subplot(gs[0, 1])
    disp2 = plotting.plot_roi(
        atlas_resampled, bg_img=mean_bold,
        display_mode="z", cut_coords=[FIXED_Z],
        axes=ax2, title="2. Schaefer-400 parcellation",
        alpha=0.7,
        colorbar=False,
    )

    # bottom-left: ROI time series (medoid + all other ROIs in grey)
    ax3 = fig.add_subplot(gs[1, 0])
    offset = 0
    y_ticks, y_tick_labels = [], []
    for y in yeo_order:
        if y not in example_roi_idx_by_yeo:
            continue
        medoid_idx = example_roi_idx_by_yeo[y]
        net_idx    = np.where(yeo_per_roi == y)[0]

        # grey background traces
        for roi_idx in net_idx:
            if roi_idx == medoid_idx:
                continue
            ax3.plot(time_axis, ts_all[:, roi_idx] + offset,
                     color="grey", linewidth=0.3, alpha=0.2, zorder=1)

        # coloured medoid on top
        ax3.plot(time_axis, ts_all[:, medoid_idx] + offset,
                 color=yeo_colors[y], linewidth=1.2, alpha=1.0, zorder=2)

        y_ticks.append(offset)
        y_tick_labels.append(y)
        offset -= spacing

    ax3.set_xlabel("Time (s)")
    ax3.set_yticks(y_ticks)
    ax3.set_yticklabels(y_tick_labels)
    ax3.set_title("3. ROI time series (medoid coloured, other ROIs grey)")
    ax3.set_xlim(0, time_axis[-1])
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # bottom-right: FC matrix
    ax4 = fig.add_subplot(gs[1, 1])
    im = ax4.imshow(fc_pearson, cmap="RdBu_r", vmin=-0.8, vmax=0.8,
                    interpolation="nearest", aspect="equal")
    ax4.set_title("4. 400×400 FC matrix (Pearson r)", pad=50)
    ax4.set_xlabel("ROI index (sorted by Yeo)")
    ax4.set_ylabel("ROI index (sorted by Yeo)")
    for y in yeo_order:
        if y not in block_bounds:
            continue
        start, end = block_bounds[y]
        ax4.axhline(end - 0.5, color="black", linewidth=0.4, alpha=0.4)
        ax4.axvline(end - 0.5, color="black", linewidth=0.4, alpha=0.4)
        mid = (start + end) / 2
        ax4.text(mid, -6, y, ha="center", va="bottom",
                 fontsize=7, color=yeo_colors[y], fontweight="bold",
                 rotation=45)
    fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04, label="Pearson r")

    fig.suptitle(f"REPRESENTATION: From BOLD to FC matrix — example {SUBJECT_ID}",
                 fontsize=15, y=0.92)
    fig.savefig(OUT_DIR / "representation_4panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nAll plots written to: {OUT_DIR}")
    print("Files:")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
