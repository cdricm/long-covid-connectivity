"""
Step 5b: NBS visualization (Family C) — reads step5a cache only, no NBS rerun.

In: step5_nbs/nbs_null_<model>_<contrast>_<thr>.npz,
    nbs_components_<model>_<contrast>_<thr>.csv,
    nbs_edges_<model>_<contrast>_<thr>.csv, step4a_labels/schaefer400_yeo7_roi_info.csv.
    <model> = age_sex_residualised; <contrast> = COVID_gt_CONTROL / COVID_lt_CONTROL.
Out: figures/fig_nbs_null_distributions.png (permutation null distributions,
     observed overlaid, log-y, 3 thresholds), figures/
     fig_nbs_component_size_and_pfwer_vs_threshold.png, plus a descriptive
     (non-figure) Yeo endpoint normalization CSV of the largest primary-
     threshold component.

The Yeo endpoint normalization is console/CSV only, explicitly not rendered as
a thesis figure — it describes a non-significant null cluster, and plotting it
would imply network structure the inference does not support.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================
NBS_DIR  = config.atlas_dir("schaefer400", "step5_nbs")
ROI_INFO = config.atlas_dir("schaefer400", "step4a_labels") / "schaefer400_yeo7_roi_info.csv"
FIG_DIR  = os.path.join(NBS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

THRESHOLDS = config.NBS_THRESHOLDS
PRIMARY    = config.NBS_PRIMARY_THRESHOLD
MODEL_TAG  = "age_sex_residualised"
MODEL_LABEL = "Freedman-Lane (age + sex residualised edges)"
CONTRASTS  = config.NBS_CONTRASTS   # [(label, tail), ...]: COVID_gt_CONTROL, COVID_lt_CONTROL
NBS_ALPHA  = config.NBS_ALPHA       # 0.025 per directional contrast
CONTRAST_LABELS = {"COVID_gt_CONTROL": "COVID > CONTROL",
                   "COVID_lt_CONTROL": "COVID < CONTROL"}
CONTRAST_COLORS = {"COVID_gt_CONTROL": "#c0392b", "COVID_lt_CONTROL": "#2c3e50"}
DPI = 300

def tag_of(thr):
    return f"t{str(thr).replace('.', '')}"

# ============================================================
# LOAD CACHE (per directional contrast)
# ============================================================
def load_contrast(model_tag, contrast_label):
    null_data, comp_data = {}, {}
    for thr in THRESHOLDS:
        tag = tag_of(thr)
        stem = f"{model_tag}_{contrast_label}_{tag}"
        npz_path  = NBS_DIR / f"nbs_null_{stem}.npz"
        comp_path = NBS_DIR / f"nbs_components_{stem}.csv"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing NBS cache: {npz_path}")
        if not comp_path.exists():
            raise FileNotFoundError(f"Missing components CSV: {comp_path}")
        null_data[thr] = np.load(npz_path)["null"]
        comp_data[thr] = pd.read_csv(comp_path)
    return null_data, comp_data

def largest_comp(comp_df):
    """Largest component by edge count; return (n_edges, p_fwer)."""
    if len(comp_df) == 0:
        return 0, np.nan
    row = comp_df.sort_values("n_edges", ascending=False).iloc[0]
    return int(row["n_edges"]), float(row["p_fwer"])

# Load both contrasts up front.
contrast_names = [c[0] for c in CONTRASTS]
null_by_c, comp_by_c, obs_by_c = {}, {}, {}
for cname in contrast_names:
    nd, cd = load_contrast(MODEL_TAG, cname)
    null_by_c[cname] = nd
    comp_by_c[cname] = cd
    obs_by_c[cname]  = {thr: largest_comp(cd[thr]) for thr in THRESHOLDS}
    print(f"[loaded] {MODEL_TAG} / {cname}")
    for thr in THRESHOLDS:
        n_e, p = obs_by_c[cname][thr]
        print(f"    t={thr}: largest={n_e} edges, p_fwer={p:.4f}, "
              f"null max={null_by_c[cname][thr].max():.0f}")

# ============================================================
# FIG 1 — Null distributions, observed overlaid (log-y).
# Rows = directional contrasts (COVID>CONTROL, COVID<CONTROL), cols = thresholds.
# ============================================================
print("\n[Fig 1] Null distributions (log-y), both directional contrasts")
n_c = len(contrast_names)
fig, axes = plt.subplots(n_c, 3, figsize=(15, 4.3 * n_c), squeeze=False)
for row_i, cname in enumerate(contrast_names):
    null_d = null_by_c[cname]
    obs = obs_by_c[cname]
    for col_i, thr in enumerate(THRESHOLDS):
        ax = axes[row_i][col_i]
        null = null_d[thr]
        obs_size, obs_p = obs[thr]
        upper = max(obs_size * 3, np.percentile(null, 99), 10)
        bins = np.linspace(0, upper, 60)
        ax.hist(np.clip(null, 0, upper), bins=bins, color="#b0b0b0",
                edgecolor="none", alpha=0.85)
        p95 = np.percentile(null, 95)
        ax.axvline(p95, color="#2c3e50", lw=1.0, ls=":", label=f"null 95th = {p95:.0f}")
        ax.axvline(obs_size, color=CONTRAST_COLORS[cname], lw=2,
                   label=f"observed = {obs_size}")
        ax.set_yscale("log")
        ax.set_xlabel("Max. component size (edges)")
        if col_i == 0:
            ax.set_ylabel(f"{CONTRAST_LABELS[cname]}\nPermutation count (log)", fontsize=9)
        primary_tag = "  (primary)" if thr == PRIMARY else ""
        ax.set_title(f"t = {thr}{primary_tag}   $p_{{FWER}}$ = {obs_p:.3f}", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, upper)
fig.suptitle("NBS permutation null distribution of maximal component size "
             f"(10,000 permutations, one-sided directional contrasts, FWE {NBS_ALPHA}, "
             "log-scaled count)\n"
             "observed largest component lies within the null at every threshold "
             "in both directions — no FWER-significant component",
             fontsize=11, y=1.01)
fig.tight_layout()
f1 = os.path.join(FIG_DIR, "fig_nbs_null_distributions.png")
fig.savefig(f1, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"    -> {f1}")

# ============================================================
# FIG 2 — Largest-component size + p_FWER vs threshold (dual axis)
#   Left  (navy) : largest component size, log scale
#   Right (red)  : p_FWER of that largest component, with alpha=0.05 reference
# The null story in one figure: the component shrinks with rising t, but p_FWER
# stays flat and far above alpha at every threshold (no significant component).
# ============================================================
print("\n[Fig 2] Largest-component size + p_FWER vs threshold (dual axis, both contrasts)")
fig, ax_size = plt.subplots(figsize=(8.5, 5.4))
ax_p = ax_size.twinx()

for cname in contrast_names:
    obs = obs_by_c[cname]
    sizes = [obs[thr][0] for thr in THRESHOLDS]
    pvals = [obs[thr][1] for thr in THRESHOLDS]
    col = CONTRAST_COLORS[cname]
    lbl = CONTRAST_LABELS[cname]
    # left axis: component size (log), solid line + circles
    ax_size.plot(THRESHOLDS, sizes, marker="o", color=col, lw=1.8, markersize=8,
                 zorder=3, label=f"{lbl} — size")
    for thr, s in zip(THRESHOLDS, sizes):
        ax_size.annotate(f"{s}", (thr, s), textcoords="offset points", xytext=(0, 9),
                         ha="center", fontsize=8, color=col)
    # right axis: p_FWER (linear), dashed line + squares
    ax_p.plot(THRESHOLDS, pvals, marker="s", color=col, lw=1.5, ls="--",
              markersize=7, zorder=3, alpha=0.75, label=f"{lbl} — $p_{{FWER}}$")

ax_size.set_yscale("log")
ax_size.set_xticks(THRESHOLDS)
ax_size.set_xlabel("Cluster-forming threshold (t)")
ax_size.set_ylabel("Largest component size (edges)")
ax_size.spines["top"].set_visible(False)

ax_p.axhline(NBS_ALPHA, color="grey", lw=1.0, ls=":", alpha=0.8)
ax_p.text(THRESHOLDS[-1], NBS_ALPHA, rf"$\alpha = {NBS_ALPHA}$", fontsize=8,
          color="grey", ha="right", va="bottom")
ax_p.set_ylabel(r"$p_{\mathrm{FWER}}$ (largest component)")
all_p = [obs_by_c[c][thr][1] for c in contrast_names for thr in THRESHOLDS]
ax_p.set_ylim(0, max(0.35, max(all_p) * 1.15))
ax_p.spines["top"].set_visible(False)

# combined legend (both axes)
h1, l1 = ax_size.get_legend_handles_labels()
h2, l2 = ax_p.get_legend_handles_labels()
ax_size.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right", framealpha=0.95)
ax_size.set_title(r"$p_{\mathrm{FWER}}$ stays above $\alpha$ in both directions "
                  "(no significant component)", fontsize=11)
fig.tight_layout()
f2 = os.path.join(FIG_DIR, "fig_nbs_component_size_and_pfwer_vs_threshold.png")
fig.savefig(f2, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"    -> {f2}")

# ============================================================
# Yeo endpoint normalization — DESCRIPTIVE ONLY (t=PRIMARY), per contrast.
# Non-significant null clusters: console + CSV, NO thesis figure.
# ============================================================
print(f"\n[Descriptive] Yeo endpoint normalization — largest t={PRIMARY} component per contrast")
roi_df  = pd.read_csv(ROI_INFO)
yeo_col = "yeo_network"
for cname in contrast_names:
    print(f"\n  --- {CONTRAST_LABELS[cname]} ---")
    edge_path = NBS_DIR / f"nbs_edges_{MODEL_TAG}_{cname}_{tag_of(PRIMARY)}.csv"
    if not edge_path.exists():
        print(f"    [skip] edge CSV not found: {edge_path}")
        continue
    edge_df = pd.read_csv(edge_path)
    obs_size, obs_p = obs_by_c[cname][PRIMARY]
    if len(edge_df) == 0:
        print("    [skip] no edges in any component at this threshold")
        continue
    top_id = edge_df.groupby("comp_id").size().sort_values(ascending=False).index[0]
    top = edge_df[edge_df["comp_id"] == top_id].copy()
    yeo_labels = roi_df[yeo_col].values
    top["yeo_i"] = top["roi_i"].map(lambda i: yeo_labels[i])
    top["yeo_j"] = top["roi_j"].map(lambda j: yeo_labels[j])
    endpoints = pd.concat([top["yeo_i"], top["yeo_j"]]).value_counts()
    net_size  = roi_df[yeo_col].value_counts()
    norm = pd.DataFrame({"endpoints": endpoints, "n_rois": net_size}).fillna(0)
    norm["endpoints_per_roi"] = norm["endpoints"] / norm["n_rois"]
    norm = norm.sort_values("endpoints_per_roi", ascending=False)
    pct_neg = 100 * (top["t"] < 0).mean()
    print(norm.round(3).to_string())
    print(f"\n    Largest component: id={top_id}, {len(top)} edges, p_fwer={obs_p:.3f}")
    print(f"    Direction: {(top['t']<0).sum()}/{len(top)} edges "
          f"COVID<CONTROL ({pct_neg:.0f}%)")
    print(f"    NOTE: NOT FWER-significant (p_fwer={obs_p:.3f}); descriptive only, "
          f"no network-level interpretation of a null cluster is warranted.")
    norm_csv = NBS_DIR / f"nbs_top_component_yeo_normalized_{cname}.csv"
    norm.to_csv(norm_csv)
    print(f"    -> {norm_csv}")

print("\n" + "=" * 60)
print("DONE (step5b). Figures in:", FIG_DIR)
print("=" * 60)