"""
step4h visualization: dropout QC on the group-mean BOLD maps (COVID vs CONTROL).
  Part A: ortho visualization of both groups (shared grayscale).
  Part B: ROI-wise mean intensity per Yeo-7 network on the group-mean maps
          (spatial description, not a statistical test), with focus blocks
          for the networks in config.TARGET_NETWORKS_BY_ARM[FC_METHOD].
  Part C: Yeo-7 network outlines over mean BOLD.
  Part D: ROI-wise intensity boxplot per Yeo-7 network.

In: step4h_signal_dropout_qc/mean_COVID.nii.gz, mean_CONTROL.nii.gz (from
    step4h_schaefer400_signal_dropout_qc.py), step4a_labels/
    schaefer400_yeo7_roi_info.csv.
Out: mean_ortho_COVID_vs_CONTROL.png, yeo7_roi_intensity.csv,
     yeo7_contours_COVID_vs_CONTROL.png, yeo7_intensity_boxplot.png.

Operates on the group-mean maps, not the 4D raw data; per-subject Cohen's d
lives in step4h's network_signal_quality_summary.csv. Yeo assignment is loaded
from the step4a CSV (single source of truth), not re-parsed here.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

FOCUS_NETWORKS = config.TARGET_NETWORKS_BY_ARM[config.FC_METHOD]
if not FOCUS_NETWORKS:
    print(f"step4h viz: no network-level trend to focus on in the {config.FC_METHOD} arm; skipped by design (MD).")
    sys.exit(0)

import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from nilearn import plotting, datasets, image
from nilearn.image import new_img_like

# ============================ SETTINGS ============================
MAP_DIR       = config.atlas_dir("schaefer400", "step4h_signal_dropout_qc")
ROI_INFO_PATH = config.atlas_dir("schaefer400", "step4a_labels") / "schaefer400_yeo7_roi_info.csv"
MEAN_COVID    = os.path.join(MAP_DIR, "mean_COVID.nii.gz")
MEAN_CONTROL  = os.path.join(MAP_DIR, "mean_CONTROL.nii.gz")

OUT_PNG     = os.path.join(MAP_DIR, "mean_ortho_COVID_vs_CONTROL.png")
OUT_CSV     = os.path.join(MAP_DIR, "yeo7_roi_intensity.csv")
OUT_PNG_NET = os.path.join(MAP_DIR, "yeo7_contours_COVID_vs_CONTROL.png")
OUT_PNG_BOX = os.path.join(MAP_DIR, "yeo7_intensity_boxplot.png")

CMAP       = "gray"
CUT_COORDS = (-24, 28, -18)   # OFC / temporal pole (dropout region)
DPI        = 150

YEO_ORDER  = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
YEO_COLORS = {"Vis": "#781286", "SomMot": "#4682B4", "DorsAttn": "#00760E",
              "SalVentAttn": "#C43AFA", "Limbic": "#9ACD32", "Cont": "#E69422",
              "Default": "#CD3E4E"}
# ==================================================================

img_cov = nib.load(MEAN_COVID)
img_con = nib.load(MEAN_CONTROL)
d_cov = img_cov.get_fdata()
d_con = img_con.get_fdata()

# ============================================================
# Part A — ortho visualization (shared scale)
# ============================================================
both = np.concatenate([d_cov[d_cov > 0], d_con[d_con > 0]])
vmin, vmax = np.percentile(both, [2, 98])
print(f"Shared color scale: vmin={vmin:.1f}, vmax={vmax:.1f}")

fig, axes = plt.subplots(2, 1, figsize=(12, 7))
for ax, img, label in [(axes[0], img_cov, "COVID"), (axes[1], img_con, "CONTROL")]:
    plotting.plot_anat(img, display_mode="ortho", cut_coords=CUT_COORDS,
                       vmin=vmin, vmax=vmax, cmap=CMAP,
                       title=f"mean BOLD intensity - {label}", annotate=True, axes=ax)
fig.suptitle("Signal dropout QC: mean BOLD intensity (shared scale)", fontsize=13)
fig.tight_layout(); fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
print(f"Saved: {OUT_PNG}")

# ============================================================
# Part B — ROI-wise mean intensity per Yeo-7 (Yeo from step4a CSV)
# ============================================================
roi_info = pd.read_csv(ROI_INFO_PATH)
assert len(roi_info) == 400 and list(roi_info["roi_idx"]) == list(range(400)), \
    "step4a ROI info not aligned 0..399"
yeo = roi_info["yeo_network"].values          # ROI i (0-based) -> network
labels = roi_info["full_label"].values

atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=2)
atlas_rs = image.resample_to_img(atlas.maps, img_cov, interpolation="nearest")
atlas_data = np.asarray(atlas_rs.get_fdata()).round().astype(int)   # labels 1..400

rows = []
for roi_val in range(1, 401):
    mask = atlas_data == roi_val
    n_vox = int(mask.sum())
    rows.append({
        "roi_value": roi_val, "label": labels[roi_val - 1], "yeo": yeo[roi_val - 1],
        "n_voxels": n_vox,
        "mean_COVID":   float(d_cov[mask].mean()) if n_vox else np.nan,
        "mean_CONTROL": float(d_con[mask].mean()) if n_vox else np.nan,
    })
df = pd.DataFrame(rows)
df["diff_COVID_minus_CONTROL"] = df["mean_COVID"] - df["mean_CONTROL"]
df.to_csv(OUT_CSV, index=False)

yeo_summary = (df.groupby("yeo")[["mean_COVID", "mean_CONTROL"]].mean()
                 .assign(diff=lambda x: x["mean_COVID"] - x["mean_CONTROL"])
                 .reindex(YEO_ORDER))
wb_cov, wb_con = df["mean_COVID"].mean(), df["mean_CONTROL"].mean()

print("\n=== mean BOLD intensity per Yeo-7 network (ROI mean of group maps) ===")
print(yeo_summary.round(1).to_string())
print(f"\nWhole-brain ROI mean: COVID={wb_cov:.1f}  CONTROL={wb_con:.1f}")

print("\n=== Focus networks (descriptive, on group-mean maps) ===")
for net in FOCUS_NETWORKS:
    sub = df[df["yeo"] == net]
    print(f"\n  {net} (n={len(sub)} ROIs):")
    print(f"    mean_COVID   = {sub['mean_COVID'].mean():.1f}  "
          f"({100*sub['mean_COVID'].mean()/wb_cov:.1f}% of whole-brain)")
    print(f"    mean_CONTROL = {sub['mean_CONTROL'].mean():.1f}  "
          f"({100*sub['mean_CONTROL'].mean()/wb_con:.1f}% of whole-brain)")
    print(f"    diff (COVID-CONTROL) = {sub['diff_COVID_minus_CONTROL'].mean():+.1f} a.u.")
print("\n  NOTE: descriptive spatial values on group-mean maps; the per-subject "
      "between-group effect sizes (Cohen's d) are in step4h "
      "network_signal_quality_summary.csv.")
print(f"\nROI table saved: {OUT_CSV}")

# ============================================================
# Part C — Yeo-7 network outlines over mean BOLD
# ============================================================
net_masks = {}
for net in YEO_ORDER:
    roi_vals = np.where(yeo == net)[0] + 1
    net_masks[net] = new_img_like(atlas_rs, np.isin(atlas_data, roi_vals).astype(np.int8))

fig_c, axes_c = plt.subplots(2, 1, figsize=(12, 7))
for ax, img, label in [(axes_c[0], img_cov, "COVID"), (axes_c[1], img_con, "CONTROL")]:
    disp = plotting.plot_anat(img, display_mode="ortho", cut_coords=CUT_COORDS,
                              vmin=vmin, vmax=vmax, cmap=CMAP,
                              title=f"Yeo-7 networks over mean BOLD - {label}",
                              annotate=True, axes=ax)
    for net in YEO_ORDER:
        disp.add_contours(net_masks[net], levels=[0.5], colors=[YEO_COLORS[net]], linewidths=1.2)
handles = [Patch(color=YEO_COLORS[n], label=n) for n in YEO_ORDER]
fig_c.legend(handles=handles, loc="center left", fontsize=9, title="Yeo-7",
             frameon=False, bbox_to_anchor=(0.0, 0.5))
fig_c.suptitle("Signal dropout QC: Yeo-7 network outlines on mean BOLD intensity", fontsize=13)
fig_c.subplots_adjust(left=0.15, right=0.9)
fig_c.savefig(OUT_PNG_NET, dpi=DPI)
print(f"Saved: {OUT_PNG_NET}")

# ============================================================
# Part D — ROI-wise intensity boxplot per Yeo-7 network
# (spread = variability BETWEEN ROIs within a network, NOT between subjects;
#  descriptive, no group test)
# ============================================================
order = (df[df["yeo"].isin(YEO_ORDER)].groupby("yeo")["mean_COVID"]
           .median().sort_values().index.tolist())
fig_d, ax_d = plt.subplots(figsize=(11, 6))
width = 0.35
for i, net in enumerate(order):
    cov = df.loc[df["yeo"] == net, "mean_COVID"].dropna()
    con = df.loc[df["yeo"] == net, "mean_CONTROL"].dropna()
    bp_c = ax_d.boxplot(cov, positions=[i - width/2], widths=width*0.9,
                        patch_artist=True, showfliers=False, medianprops=dict(color="black"))
    bp_n = ax_d.boxplot(con, positions=[i + width/2], widths=width*0.9,
                        patch_artist=True, showfliers=False, medianprops=dict(color="black"))
    for box in bp_c["boxes"]: box.set(facecolor="#4C72B0", alpha=0.85)
    for box in bp_n["boxes"]: box.set(facecolor="#DD8452", alpha=0.85)
ax_d.set_xticks(range(len(order))); ax_d.set_xticklabels(order, rotation=20, ha="right")
ax_d.set_ylabel("mean intensity (preprocessed BOLD, a.u.)")
ax_d.set_title("ROI-wise mean intensity per Yeo-7 network (COVID vs CONTROL)\n"
               "spread = between-ROI variability within a network (descriptive, no group test)")
ax_d.legend(handles=[Patch(facecolor="#4C72B0", label="COVID", alpha=0.85),
                     Patch(facecolor="#DD8452", label="CONTROL", alpha=0.85)],
            loc="upper left", frameon=False)
ax_d.grid(axis="y", alpha=0.3)
fig_d.tight_layout(); fig_d.savefig(OUT_PNG_BOX, dpi=DPI)
print(f"Saved: {OUT_PNG_BOX}")
plt.close("all")