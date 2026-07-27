"""
Step 4f: ROI-level localization of within-network FC trends (exploratory).

Localizes the within-network FC trends from step4d to individual ROIs, for the
networks declared in config.TARGET_NETWORKS_BY_ARM[FC_METHOD] (Pearson: Cont,
Limbic, Default; partial: none — script exits by design). Fixed illustrative
selection, not a data-driven threshold; no confirmatory claim.

In:  config.atlas_dir("schaefer400", "step2_pipeline")/comet_matrices,
     step4a_labels/schaefer400_yeo7_roi_info.csv.
Out: step4f_roi_localization/<net>/{roi_d_values.csv, bar/glassbrain/
     distribution/volcano PNGs}, all_networks_roi_d_values.csv.

Per ROI: within-network FC = mean Fisher-z FC to all other ROIs of the same
network (z-then-mean). R2 ②: Freedman-Lane residualised permutation (OLS
group-coefficient t, age+sex adjusted), FDR-BH within each network separately.
Reuses freedman_lane_vectorized/cohens_d from step3d_auc_pipeline (same
covariate model as all outcomes). R1 (superseded): naive Welch-t, no covariates.
"""

import sys
from pathlib import Path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
import config

def _find_step3d_dir(root):
    for cand in sorted(root.iterdir()):
        if cand.is_dir() and (cand / "step3d_auc_pipeline.py").exists():
            return cand
    raise FileNotFoundError(f"step3d_auc_pipeline.py not found under {root}")
sys.path.insert(0, str(_find_step3d_dir(_HERE.parents[1])))
from step3d_auc_pipeline import freedman_lane_vectorized, cohens_d, load_covariates

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from statsmodels.stats.multitest import multipletests
from nilearn import datasets, plotting, image

# ============================================================
# SETTINGS
# ============================================================
MATRIX_DIR    = config.atlas_dir("schaefer400", "step2_pipeline") / "comet_matrices"
ROI_INFO_PATH = config.atlas_dir("schaefer400", "step4a_labels") / "schaefer400_yeo7_roi_info.csv"
OUT_ROOT      = config.ensure(config.atlas_dir("schaefer400", "step4f_roi_localization"))

TARGET_NETWORKS = config.TARGET_NETWORKS_BY_ARM[config.FC_METHOD]
if not TARGET_NETWORKS:
    print(f"step4f: no network-level trend to localize in the {config.FC_METHOD} arm; skipped by design (MD).")
    sys.exit(0)
FISHER_CLIP   = config.FISHER_CLIP
N_PERM        = config.N_PERMUTATIONS
N_BOOT        = 5000
SEED          = config.SEED
FDR_Q         = config.FDR_ALPHA
LABEL_THRESH  = 0.05
GROUP_A, GROUP_B = config.GROUP_ORDER

# ============================================================
# LOAD ROI INFO + COHORT
# ============================================================
roi_info = pd.read_csv(ROI_INFO_PATH)
df_csv   = pd.read_csv(config.GROUP_CSV)
subjects = config.select_included_subjects(
    [p.name for p in config.NII_ROOT.iterdir() if p.is_dir()],
    df_csv, id_col="ID", group_col="Grupo", verbose=False)
missing = [s for s in subjects
           if not os.path.exists(os.path.join(MATRIX_DIR, f"{s}_connectivity_comet.npy"))]
assert not missing, f"Missing FC matrices: {missing}"
group_map = {str(i).strip(): str(g).strip() for i, g in zip(df_csv["ID"], df_csv["Grupo"])}
print(f"Cohort via config: {len(subjects)} subjects")

groups_all = np.array([group_map[s] for s in subjects])
assert len(subjects) == 162 and (groups_all == "COVID").sum() == 123 \
    and (groups_all == "CONTROL").sum() == 39, "cohort deviates from frozen 162 (123/39)"

# --- R2 ②: covariate model (age, sex) aligned to the `subjects` order, so it
#     matches the fc matrix rows built per network below. Same loader/model as A/B/C.
_cov_df, _sex_map = load_covariates(subjects)
_cov_df = _cov_df.set_index("subject")
assert _cov_df.index.is_unique, "duplicate subject IDs in covariate table"
_missing_cov = [s for s in subjects if s not in _cov_df.index]
if _missing_cov:
    raise RuntimeError(f"covariates missing for {len(_missing_cov)} subject(s): {_missing_cov}")
Z_ALL = _cov_df.loc[subjects, ["age", "sex_code"]].values.astype(float)
print(f"Covariates: age + sex, Freedman-Lane (se_type={config.FL_SE_TYPE}), sex coding {_sex_map}")

# Preload all matrices once (reused across the 3 networks)
mats = {s: np.load(os.path.join(MATRIX_DIR, f"{s}_connectivity_comet.npy")) for s in subjects}
for m in mats.values():
    np.fill_diagonal(m, 0.0)

# ============================================================
# HELPERS
# ============================================================
def roi_within_fc(net_idx):
    """(n_subjects, n_roi) matrix: each ROI's mean Fisher-z FC to other same-net ROIs."""
    n_roi = len(net_idx)
    out = np.zeros((len(subjects), n_roi))
    for s_i, subj in enumerate(subjects):
        sub = mats[subj][np.ix_(net_idx, net_idx)]
        for r in range(n_roi):
            others = np.delete(sub[r], r)
            out[s_i, r] = np.mean(np.arctanh(np.clip(others, -FISHER_CLIP, FISHER_CLIP)))
    return out

def bootstrap_ci_d(x_cov, x_ctrl, n_boot, seed):
    """Bootstrap 95% CI of Cohen's d (COVID - CONTROL). cohens_d(a,b) returns b-a,
    so passing (ctrl, cov) yields cov - ctrl = COVID - CONTROL."""
    rng = np.random.default_rng(seed)
    nC, nH = len(x_cov), len(x_ctrl)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        d, _, _ = cohens_d(x_ctrl[rng.integers(0, nH, nH)], x_cov[rng.integers(0, nC, nC)])
        boot[b] = d
    return np.percentile(boot, 2.5), np.percentile(boot, 97.5)

# ============================================================
# PER-NETWORK ANALYSIS
# ============================================================
def analyze_network(net):
    net_rows = roi_info[roi_info["yeo_network"] == net]
    net_idx  = net_rows["roi_idx"].values
    n_roi    = len(net_idx)
    out_dir  = config.ensure(OUT_ROOT / net)
    print(f"\n{'='*64}\nNetwork {net}: {n_roi} ROIs\n{'='*64}")

    fc = roi_within_fc(net_idx)   # (n_subj, n_roi), subjects order
    groups = np.array([group_map[s] for s in subjects])
    g_int = (groups == GROUP_B).astype(int)   # CONTROL=0, COVID=1

    # R2 ②: Freedman-Lane over all ROIs of this network in one vectorised call
    # (OLS group-coefficient t, age+sex adjusted). Z_ALL and g_int follow the
    # same `subjects` order as fc rows. Cohen's d + bootstrap CI stay per ROI
    # (descriptive, unadjusted).
    d_vals, ci_lo, ci_hi = [], [], []
    for r in range(n_roi):
        y = fc[:, r]
        a, b = y[g_int == 0], y[g_int == 1]   # a=CONTROL, b=COVID
        d, _, _ = cohens_d(a, b)
        lo, hi = bootstrap_ci_d(b, a, N_BOOT, SEED + r)
        d_vals.append(d); ci_lo.append(lo); ci_hi.append(hi)

    t_obs_vec, p_perm, p_param = freedman_lane_vectorized(
        fc, g_int.astype(float), Z_ALL, N_PERM, np.random.SeedSequence(SEED))
    t_perm = list(t_obs_vec)
    p_perm = np.asarray(p_perm)
    p_fdr  = multipletests(p_perm, alpha=FDR_Q, method="fdr_bh")[1]

    res = pd.DataFrame({
        "roi_idx_global": net_idx,
        "full_label": net_rows["full_label"].values,
        "hemisphere": net_rows["hemisphere"].values,
        "sub_region": net_rows["sub_region"].values,
        "yeo_network": net,
        "cohens_d": d_vals, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "t_perm": t_perm,
        "p_perm": p_perm, "p_param": p_param, "p_fdr": p_fdr,
    }).sort_values("cohens_d").reset_index(drop=True)
    res.to_csv(out_dir / f"{net}_roi_d_values.csv", index=False)

    n_fdr = int((p_fdr < FDR_Q).sum())
    print(f"  median d={np.median(d_vals):+.3f}, mean d={np.mean(d_vals):+.3f}, "
          f"range [{min(d_vals):+.3f},{max(d_vals):+.3f}]")
    print(f"  d<0: {(np.array(d_vals)<0).sum()}/{n_roi} | "
          f"uncorrected p<0.05: {(p_perm<0.05).sum()}/{n_roi} | "
          f"FDR q<{FDR_Q}: {n_fdr}/{n_roi}")

    _plots(res, net, out_dir, n_roi, n_fdr)
    return res

# ============================================================
# PLOTS (per network)
# ============================================================
def _plots(res, net, out_dir, n_roi, n_fdr):
    d_vals = res["cohens_d"].values
    median_d = float(np.median(d_vals))

    # --- Bar plot (sorted) with bootstrap CI ---
    fig, ax = plt.subplots(figsize=(11, max(6, n_roi * 0.28)))
    y = np.arange(len(res))
    colors = ["#4477aa" if d < 0 else "#cc6677" for d in d_vals]
    ax.barh(y, d_vals, color=colors, alpha=0.85)
    ax.errorbar(d_vals, y, xerr=[d_vals - res["ci_lo"].values, res["ci_hi"].values - d_vals],
                fmt="none", ecolor="0.3", capsize=2, linewidth=0.7)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(median_d, color="black", ls="--", lw=1, alpha=0.7, label=f"Median d={median_d:+.3f}")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{h} {s}" for h, s in zip(res["hemisphere"], res["sub_region"])], fontsize=7)
    ax.set_xlabel("Cohen's d (COVID - CONTROL)")
    ax.set_title(f"Within-{net} FC: per-ROI Cohen's d (descriptive, bootstrap 95% CI)\n"
                 f"n={n_roi} ROIs, sorted; exploratory (no confirmatory claim)", fontsize=10)
    ax.legend(loc="lower right"); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / f"{net}_roi_bar.png", dpi=150, bbox_inches="tight"); plt.close()

    # --- Histogram ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(d_vals, bins=15, color="#4477aa", alpha=0.7, edgecolor="black")
    ax.axvline(0, color="black", lw=1)
    ax.axvline(median_d, color="red", ls="--", lw=1.5, label=f"Median={median_d:+.3f}")
    ax.axvline(np.mean(d_vals), color="orange", ls="--", lw=1.5, label=f"Mean={np.mean(d_vals):+.3f}")
    ax.set_xlabel("Cohen's d (COVID - CONTROL)"); ax.set_ylabel(f"Number of {net} ROIs")
    ax.set_title(f"Distribution of per-ROI Cohen's d within {net} (n={n_roi})", fontsize=10)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / f"{net}_d_distribution.png", dpi=150, bbox_inches="tight"); plt.close()

    # --- Volcano (exploratory; clearly labeled) ---
    p_vals = res["p_perm"].values
    neglogp = -np.log10(np.clip(p_vals, 1e-5, 1.0))
    fdr_sig = res["p_fdr"].values < FDR_Q
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ["#4477aa" if d < 0 else "#cc6677" for d in d_vals]
    ax.scatter(d_vals, neglogp, c=colors, s=85, alpha=0.85,
               edgecolors=["black" if f else "#33333355" for f in fdr_sig],
               linewidths=[1.8 if f else 0.6 for f in fdr_sig], zorder=3)
    ax.axvline(0, color="grey", lw=0.8)
    ax.axhline(-np.log10(0.05), color="black", ls="--", lw=1, label="p=0.05 (uncorrected)")
    ranked = res.sort_values("p_perm").reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    to_label = ranked[ranked["p_perm"] < LABEL_THRESH]
    for k, (_, r) in enumerate(to_label.iterrows()):
        ax.annotate(str(int(r["rank"])), (r["cohens_d"], -np.log10(max(r["p_perm"], 1e-5))),
                    fontsize=8, fontweight="bold",
                    xytext=(6 if k % 2 == 0 else -16, 4), textcoords="offset points", zorder=4)
    ax.text(0.015, 0.97,
            f"EXPLORATORY localization (not a pre-specified family)\n"
            f"Freedman-Lane permutation, age+sex adjusted; FDR-BH within {net}\n"
            f"FDR q<{FDR_Q}: {n_fdr}/{n_roi} survive   "
            f"(min q={res['p_fdr'].min():.3f})",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5, style="italic",
            color=("red" if n_fdr == 0 else "black"),
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.9))
    xmax = np.max(np.abs(d_vals)) * 1.15
    ax.set_xlim(-xmax, xmax)
    ax.set_xlabel("Cohen's d (COVID - CONTROL)"); ax.set_ylabel(r"$-\log_{10}(p_{\mathrm{perm}})$")
    ax.set_title(f"Within-{net} FC: ROI-level volcano (EXPLORATORY)\n"
                 f"Freedman-Lane (age+sex adjusted, {N_PERM} perm, seed={SEED}); n={n_roi}", fontsize=10)
    ax.legend(loc="lower left", fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / f"{net}_volcano.png", dpi=150, bbox_inches="tight"); plt.close()

    # --- Glass brain (markers = d) ---
    try:
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=2)
        atlas_img = image.load_img(atlas.maps)
        atlas_data = atlas_img.get_fdata(); affine = atlas_img.affine
        coords = []
        for gidx in res["roi_idx_global"]:
            mask = atlas_data == (gidx + 1)
            coords.append((affine @ np.array([*ndimage.center_of_mass(mask), 1]))[:3]
                          if mask.sum() else [0, 0, 0])
        coords = np.array(coords)
        amax = np.max(np.abs(d_vals))
        sizes = 120 + 450 * (np.abs(d_vals) / amax)
        disp = plotting.plot_glass_brain(None, display_mode="lzry",
                                         title=f"Within-{net} FC: ROI-level Cohen's d (descriptive)")
        disp.add_markers(marker_coords=coords, marker_color=d_vals, marker_size=sizes,
                         cmap="RdBu_r", vmin=-amax, vmax=amax)
        disp.savefig(out_dir / f"{net}_glassbrain.png", dpi=150); disp.close()
    except Exception as e:
        print(f"  [glassbrain skipped for {net}] {type(e).__name__}: {str(e)[:120]}")

# ============================================================
# RUN
# ============================================================
all_res = []
for net in TARGET_NETWORKS:
    all_res.append(analyze_network(net))

combined = pd.concat(all_res, ignore_index=True)
combined.to_csv(OUT_ROOT / "all_networks_roi_d_values.csv", index=False)
print(f"\n{'='*64}\nSaved combined: {OUT_ROOT / 'all_networks_roi_d_values.csv'}")
print("Reminder: ROI-level is exploratory localization, not a declared inference "
      "family; parent within-network tests (4d) were not FDR-significant.")