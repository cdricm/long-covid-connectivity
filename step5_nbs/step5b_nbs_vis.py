"""
Step 5b: NBS visualization (Family C) — reads step5a cache only, no NBS rerun.

In: step5_nbs/nbs_null_<model>_<thr>.npz, nbs_components_<model>_<thr>.csv,
    nbs_edges_<model>_<thr>.csv, step4a_labels/schaefer400_yeo7_roi_info.csv.
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
MODEL_TAG  = "naive_no_covariates"
MODEL_LABEL = "Naive label permutation (no covariates)"
DPI = 300

def tag_of(thr):
    return f"t{str(thr).replace('.', '')}"

# ============================================================
# LOAD CACHE (single model)
# ============================================================
def load_model(model_tag):
    null_data, comp_data = {}, {}
    for thr in THRESHOLDS:
        tag = tag_of(thr)
        npz_path  = NBS_DIR / f"nbs_null_{model_tag}_{tag}.npz"
        comp_path = NBS_DIR / f"nbs_components_{model_tag}_{tag}.csv"
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

null_d, comp_d = load_model(MODEL_TAG)
obs = {thr: largest_comp(comp_d[thr]) for thr in THRESHOLDS}
print(f"[loaded] {MODEL_TAG}")
for thr in THRESHOLDS:
    n_e, p = obs[thr]
    print(f"    t={thr}: largest={n_e} edges, p_fwer={p:.4f}, "
          f"null max={null_d[thr].max():.0f}")

# ============================================================
# FIG 1 — Null distributions, observed overlaid (log-y)
# ============================================================
print("\n[Fig 1] Null distributions (log-y)")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
for col_i, thr in enumerate(THRESHOLDS):
    ax = axes[0][col_i]
    null = null_d[thr]
    obs_size, obs_p = obs[thr]
    upper = max(obs_size * 3, np.percentile(null, 99), 10)
    bins = np.linspace(0, upper, 60)
    ax.hist(np.clip(null, 0, upper), bins=bins, color="#b0b0b0",
            edgecolor="none", alpha=0.85)
    p95 = np.percentile(null, 95)
    ax.axvline(p95, color="#2c3e50", lw=1.0, ls=":", label=f"null 95th = {p95:.0f}")
    ax.axvline(obs_size, color="#c0392b", lw=2, label=f"observed = {obs_size}")
    ax.set_yscale("log")
    ax.set_xlabel("Max. component size (edges)")
    if col_i == 0:
        ax.set_ylabel("Permutation count (log)", fontsize=9)
    primary_tag = "  (primary threshold)" if thr == PRIMARY else ""
    ax.set_title(f"t = {thr}{primary_tag}   $p_{{FWER}}$ = {obs_p:.3f}", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, upper)
fig.suptitle("NBS permutation null distribution of maximal component size "
             "(10,000 permutations, tail='both', log-scaled count)\n"
             "observed largest component (red) lies within the null at every "
             "threshold — no FWER-significant component",
             fontsize=11, y=1.02)
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
print("\n[Fig 2] Largest-component size + p_FWER vs threshold (dual axis)")
obs_sizes = [obs[thr][0] for thr in THRESHOLDS]
obs_pvals = [obs[thr][1] for thr in THRESHOLDS]

fig, ax_size = plt.subplots(figsize=(8, 5.2))

# left axis: component size (log)
c_size = "#2c3e50"
ax_size.plot(THRESHOLDS, obs_sizes, marker="o", color=c_size, lw=1.8, markersize=9,
             zorder=3)
ax_size.set_yscale("log")
ax_size.set_xticks(THRESHOLDS)
ax_size.set_xlabel("Cluster-forming threshold (t)")
ax_size.set_ylabel("Largest component size (edges)", color=c_size)
ax_size.tick_params(axis="y", labelcolor=c_size)
ax_size.spines["top"].set_visible(False)
for thr, s in zip(THRESHOLDS, obs_sizes):
    ax_size.annotate(f"{s}", (thr, s), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=9, color=c_size)

# right axis: p_FWER (linear)
c_p = "#c0392b"
ax_p = ax_size.twinx()
ax_p.plot(THRESHOLDS, obs_pvals, marker="s", color=c_p, lw=1.8, ls="--",
          markersize=8, zorder=3)
ax_p.axhline(0.05, color=c_p, lw=1.0, ls=":", alpha=0.7)
ax_p.set_ylabel(r"$p_{\mathrm{FWER}}$ (largest component)", color=c_p)
ax_p.tick_params(axis="y", labelcolor=c_p)
ax_p.set_ylim(0, max(0.35, max(obs_pvals) * 1.15))
ax_p.spines["top"].set_visible(False)
for thr, p in zip(THRESHOLDS, obs_pvals):
    ax_p.annotate(f"{p:.3f}", (thr, p), textcoords="offset points", xytext=(0, -15),
                  ha="center", fontsize=9, color=c_p)
ax_p.text(THRESHOLDS[-1], 0.05, r"$\alpha = 0.05$", fontsize=8, color=c_p,
          ha="right", va="bottom")

ax_size.set_title(r"$p_{\mathrm{FWER}}$ remains stable (no significant component)",
                  fontsize=11)
fig.tight_layout()
f2 = os.path.join(FIG_DIR, "fig_nbs_component_size_and_pfwer_vs_threshold.png")
fig.savefig(f2, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"    -> {f2}")

# ============================================================
# Yeo endpoint normalization — DESCRIPTIVE ONLY (t=3.1)
# Non-significant null cluster: console + CSV, NO thesis figure.
# ============================================================
print(f"\n[Descriptive] Yeo endpoint normalization — largest t={PRIMARY} component")
roi_df  = pd.read_csv(ROI_INFO)
yeo_col = "yeo_network"
edge_path = NBS_DIR / f"nbs_edges_{MODEL_TAG}_{tag_of(PRIMARY)}.csv"
if not edge_path.exists():
    print(f"    [skip] edge CSV not found: {edge_path}")
else:
    edge_df = pd.read_csv(edge_path)
    obs_size, obs_p = obs[PRIMARY]
    if len(edge_df) == 0:
        print("    [skip] no edges in any component at this threshold")
    else:
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
        print(f"\n    NOTE: this component is NOT FWER-significant (p_fwer={obs_p:.3f}). "
              f"The Yeo normalization is purely descriptive; no network-level "
              f"interpretation of a null cluster is warranted.")
        norm_csv = NBS_DIR / "nbs_top_component_yeo_normalized.csv"
        norm.to_csv(norm_csv)
        print(f"    -> {norm_csv}")

print("\n" + "=" * 60)
print("DONE (step5b). Figures in:", FIG_DIR)
print("=" * 60)