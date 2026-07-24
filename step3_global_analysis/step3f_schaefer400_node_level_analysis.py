"""
ROI-level nodal strength analysis — EXPLORATORY (Schaefer-400), not a
pre-specified inference family (only Family A/B/C are). Reported for
transparency and hypothesis generation; the confirmatory localized test is
NBS (Family C).

In: cached matrices via config.atlas_dir("schaefer400", "step2_pipeline"),
    cohort via config.select_included_subjects().
Out: config.step3f_dir("schaefer400", strategy=STRATEGY)/
     roi_strength_subjects.npy/_ids.csv, roi_nodal_strength_results.csv,
     roi_nodal_strength_topd.png, roi_nodal_strength_dmap.png,
     roi_nodal_strength_volcano.png.

Per ROI: AUC of nodal strength over 10-25% (Fisher-z, then
config.apply_sign_strategy(·, config.CONFIRMATORY_SIGN_STRATEGY) +
config.proportional_threshold, np.trapezoid x-arg) — arm-dependent: positive-only
in the Pearson arm, absolute in the partial arm. Cohen's d + bootstrap 95% CI
(descriptive). Naive label-permutation (Welch t, same logic as Family A/B),
FDR-BH over the 400 ROIs. No covariate adjustment; SeedSequence(SEED) drives
the permutation stream.
"""

import sys, os
from pathlib import Path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
import config
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(getattr(config, "JOBLIB_TEMP", config.AO)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

# =====================================================================
# SETTINGS
# =====================================================================
ATLAS      = "schaefer400"
STRATEGY   = config.CONFIRMATORY_SIGN_STRATEGY   # "positive" in the Pearson arm
MATRIX_DIR = config.atlas_dir(ATLAS, "step2_pipeline") / "comet_matrices"
OUT_DIR    = config.ensure(config.step3f_dir(ATLAS, strategy=STRATEGY))

DENSITIES  = np.arange(config.AUC_RANGE_CONFIRMATORY[0],
                       config.AUC_RANGE_CONFIRMATORY[1] + 0.0001, 0.01)
N_BOOT     = 5_000
N_PERM     = config.N_PERMUTATIONS
SEED       = config.SEED
N_JOBS     = config.N_JOBS_DEFAULT
FDR_Q      = config.FDR_ALPHA
USE_CACHE  = True

STRENGTH_CACHE = OUT_DIR / "roi_strength_subjects.npy"
SUBJECTS_CACHE = OUT_DIR / "roi_strength_subjects_ids.csv"
RESULTS_CSV    = OUT_DIR / "roi_nodal_strength_results.csv"

# =====================================================================
# HELPERS
# =====================================================================
def fisher_z(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))

def auc_strength_per_subject(matrix_path, densities):
    R = np.load(matrix_path)
    np.fill_diagonal(R, 0.0)
    Zp = config.apply_sign_strategy(fisher_z(R), STRATEGY)
    strengths = np.zeros((len(densities), Zp.shape[0]))
    for i, d in enumerate(densities):
        strengths[i] = config.proportional_threshold(Zp, d)[0].sum(axis=1)
    return np.trapezoid(strengths, x=densities, axis=0)

def cohens_d_vec(x, y):
    """Vectorized d across ROIs (x=COVID, y=CONTROL). d>0 -> COVID>CONTROL."""
    nx, ny = len(x), len(y)
    s = np.sqrt(((nx-1)*x.var(ddof=1, axis=0) + (ny-1)*y.var(ddof=1, axis=0)) / (nx+ny-2))
    return (x.mean(axis=0) - y.mean(axis=0)) / s

def t_welch_vec(x, y):
    """Vectorized Welch t across ROIs (x=COVID, y=CONTROL). t>0 -> COVID>CONTROL."""
    nx, ny = x.shape[0], y.shape[0]
    vx = x.var(ddof=1, axis=0); vy = y.var(ddof=1, axis=0)
    return (x.mean(axis=0) - y.mean(axis=0)) / np.sqrt(vx/nx + vy/ny)

# =====================================================================
# 1) COHORT + MATCH MATRICES
# =====================================================================
assert STRATEGY is not None, (
    "CONFIRMATORY_SIGN_STRATEGY is None — set the confirmatory strategy before "
    "running nodal-strength analysis (Pearson: 'positive' via FC_METHOD; partial: "
    "after the 3a/3b/3c diagnostic)."
)

group_df = pd.read_csv(config.GROUP_CSV)
on_disk = {p.stem.replace("_connectivity_comet", "")
           for p in MATRIX_DIR.glob("CP*_connectivity_comet.npy")}
included = config.select_included_subjects(sorted(on_disk), group_df, verbose=False)
id_to_path = {p.stem.replace("_connectivity_comet", ""): p
              for p in MATRIX_DIR.glob("CP*_connectivity_comet.npy")}
gmap = {str(i).strip(): str(g).strip() for i, g in zip(group_df["ID"], group_df["Grupo"])}

subjects = [s for s in included if s in id_to_path]
groups_arr = np.array([gmap[s] for s in subjects])
print(f"Strategy: {STRATEGY}")
print(f"Subjects (config): {len(subjects)}  "
      f"(COVID={(groups_arr=='COVID').sum()}, CONTROL={(groups_arr=='CONTROL').sum()})")

# Hard cohort guard: node-level analysis must run on the same frozen cohort as
# the global analysis (no covariate-driven subject drop; Entscheidung B).
assert len(subjects) == 162, f"cohort mismatch: {len(subjects)} != 162"
assert (groups_arr == "COVID").sum() == 123 and (groups_arr == "CONTROL").sum() == 39, \
    "group sizes deviate from frozen cohort 123/39"

# =====================================================================
# 2) AUC NODAL STRENGTH PER SUBJECT (cached)
# =====================================================================
S = None
if USE_CACHE and STRENGTH_CACHE.exists() and SUBJECTS_CACHE.exists():
    cached_ids = pd.read_csv(SUBJECTS_CACHE)["ID"].astype(str).tolist()
    if cached_ids == subjects:
        print(f"Loading cached AUC strengths from {STRENGTH_CACHE}")
        S = np.load(STRENGTH_CACHE)
    else:
        print("Cache subject mismatch -> recomputing")

if S is None:
    print("Computing AUC nodal strength per subject ...")
    S_list = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(auc_strength_per_subject)(id_to_path[s], DENSITIES) for s in subjects)
    S = np.vstack(S_list)
    np.save(STRENGTH_CACHE, S)
    pd.DataFrame({"ID": subjects}).to_csv(SUBJECTS_CACHE, index=False)
    print(f"Saved {STRENGTH_CACHE}, shape={S.shape}")

n_rois = S.shape[1]
covid_mask = (groups_arr == "COVID")
x_covid, x_control = S[covid_mask], S[~covid_mask]

# =====================================================================
# 3) COHEN'S D + BOOTSTRAP CI (descriptive)
# =====================================================================
d_obs = cohens_d_vec(x_covid, x_control)
print(f"Bootstrap CI ({N_BOOT} resamples) ...")
rng = np.random.default_rng(SEED)
n_cov, n_ctrl = x_covid.shape[0], x_control.shape[0]
d_boot = np.zeros((N_BOOT, n_rois))
for b in range(N_BOOT):
    d_boot[b] = cohens_d_vec(x_covid[rng.integers(0, n_cov, n_cov)],
                             x_control[rng.integers(0, n_ctrl, n_ctrl)])
ci_lo = np.percentile(d_boot, 2.5, axis=0)
ci_hi = np.percentile(d_boot, 97.5, axis=0)

# =====================================================================
# 4) NAIVE LABEL-PERMUTATION PER ROI (exploratory inference, unadjusted)
#    Welch t (COVID - CONTROL), group labels permuted; no covariates
#    (Entscheidung B). Same inference logic as Family A/B. Vectorized over ROIs.
# =====================================================================
t_obs = t_welch_vec(x_covid, x_control)
abs_t_obs = np.abs(t_obs)

print(f"Naive label-permutation ({N_PERM} perms, unadjusted, "
      f"vectorized over {n_rois} ROIs) ...")
perm_rng = np.random.default_rng(np.random.SeedSequence(SEED))
n_tot = S.shape[0]
n_cov_perm = int(covid_mask.sum())
count_ge = np.zeros(n_rois, dtype=np.int64)   # |t_perm| >= |t_obs|

for _ in range(N_PERM):
    idx = perm_rng.permutation(n_tot)
    t_perm = t_welch_vec(S[idx[:n_cov_perm]], S[idx[n_cov_perm:]])
    count_ge += (np.abs(t_perm) >= abs_t_obs)

# +1 correction (Phipson & Smyth 2010): permutation p never exactly zero
p_perm = (count_ge + 1) / (N_PERM + 1)
p_fdr = multipletests(p_perm, alpha=FDR_Q, method="fdr_bh")[1]

# =====================================================================
# 5) SAVE
# =====================================================================
out = pd.DataFrame({
    "roi_idx": np.arange(n_rois),
    "cohens_d": d_obs, "ci_lo": ci_lo, "ci_hi": ci_hi,
    "ci_excludes_zero": (ci_lo > 0) | (ci_hi < 0),   # descriptive flag, NOT a test
    "t_obs": t_obs,
    "p_perm": p_perm, "p_fdr": p_fdr,
})
out.to_csv(RESULTS_CSV, index=False)

n_fdr = int((p_fdr < FDR_Q).sum())
n_unc = int((p_perm < 0.05).sum())
expected_chance = 0.05 * n_rois

# =====================================================================
# 6) PLOTS
# =====================================================================
# 6a) Top-|d| bar
order = np.argsort(-np.abs(d_obs))[:25]
fig, ax = plt.subplots(figsize=(9, 6))
yb = np.arange(len(order))
colors = ["#d62728" if d_obs[i] > 0 else "#1f77b4" for i in order]
ax.barh(yb, d_obs[order], color=colors, alpha=0.8)
ax.errorbar(d_obs[order], yb, xerr=[d_obs[order]-ci_lo[order], ci_hi[order]-d_obs[order]],
            fmt="none", ecolor="0.3", capsize=2, linewidth=0.8)
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(yb); ax.set_yticklabels([f"ROI {i}" for i in order], fontsize=7); ax.invert_yaxis()
ax.set_xlabel("Cohen's d (COVID - CONTROL)")
ax.set_title("Schaefer-400 nodal strength — top 25 |d| (descriptive, bootstrap 95% CI)")
plt.tight_layout(); plt.savefig(OUT_DIR / "roi_nodal_strength_topd.png", dpi=150); plt.close()

# 6b) Volcano (exploratory; defensive labeling)
neglogp = -np.log10(np.clip(p_perm, 1e-5, 1.0))
fdr_sig = p_fdr < FDR_Q
fig, ax = plt.subplots(figsize=(11, 7))
colors = ["#1f77b4" if d < 0 else "#d62728" for d in d_obs]
ax.scatter(d_obs, neglogp, c=colors, s=28, alpha=0.6,
           edgecolors=["black" if f else "none" for f in fdr_sig],
           linewidths=[1.6 if f else 0 for f in fdr_sig], zorder=3)
ax.axvline(0, color="grey", lw=0.8)
ax.axhline(-np.log10(0.05), color="black", ls="--", lw=1, label="p=0.05 (uncorrected)")
ax.text(0.015, 0.97,
        f"EXPLORATORY whole-brain localization (NOT a pre-specified family)\n"
        f"Naive label-permutation, unadjusted; FDR-BH over {n_rois} ROIs\n"
        f"FDR q<{FDR_Q}: {n_fdr}/{n_rois} survive   (min q={p_fdr.min():.3f})\n"
        f"uncorrected p<0.05: {n_unc}/{n_rois}  (chance expectation ~{expected_chance:.0f})",
        transform=ax.transAxes, va="top", ha="left", fontsize=8.5, style="italic",
        color=("red" if n_fdr == 0 else "black"),
        bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.92))
xmax = np.max(np.abs(d_obs)) * 1.15
ax.set_xlim(-xmax, xmax)
ax.set_xlabel("Cohen's d (COVID - CONTROL)"); ax.set_ylabel(r"$-\log_{10}(p_{\mathrm{perm}})$")
ax.set_title(f"Schaefer-400 nodal strength: ROI-level volcano (EXPLORATORY)\n"
             f"Naive label-permutation (unadjusted, {N_PERM} perm, seed={SEED}); "
             f"n={n_rois} ROIs", fontsize=10)
ax.legend(loc="lower left", fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT_DIR / "roi_nodal_strength_volcano.png", dpi=150, bbox_inches="tight"); plt.close()

# 6c) Brain map (defensive)
brain_ok = False
try:
    from nilearn import datasets, plotting
    from nilearn.image import load_img
    import nibabel as nib
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=2)
    labels_img = load_img(atlas.maps)
    arr = np.asarray(labels_img.dataobj)
    dmap = np.zeros(arr.shape, dtype=float)
    for i in range(n_rois):
        dmap[arr == (i + 1)] = d_obs[i]
    dmap_img = nib.Nifti1Image(dmap, labels_img.affine, labels_img.header)
    vmax = float(np.nanmax(np.abs(d_obs)))
    disp = plotting.plot_stat_map(dmap_img, display_mode="z", cut_coords=6, cmap="RdBu_r",
                                  vmax=vmax, colorbar=True, threshold=None,
                                  title="Schaefer-400 nodal strength d (COVID-CONTROL), descriptive")
    disp.savefig(OUT_DIR / "roi_nodal_strength_dmap.png", dpi=150); disp.close()
    brain_ok = True
except Exception as e:
    print(f"[brain map skipped] {type(e).__name__}: {str(e)[:140]}")

# =====================================================================
# 7) CONSOLE SUMMARY
# =====================================================================
print("\n=========== ROI Nodal Strength (EXPLORATORY) ===========")
print(f"ROIs analyzed:        {n_rois}")
print(f"d range:              [{d_obs.min():+.3f}, {d_obs.max():+.3f}]")
print(f"Sign balance:         {(d_obs<0).sum()} neg / {(d_obs>0).sum()} pos (COVID</> CONTROL)")
print(f"|d| >= 0.3:           {(np.abs(d_obs)>=0.3).sum()}")
print(f"Bootstrap CI excl. 0: {int(out['ci_excludes_zero'].sum())} ROIs (descriptive flag)")
print(f"uncorrected p<0.05:   {n_unc}/{n_rois}  (chance ~{expected_chance:.0f})")
print(f"FDR q<{FDR_Q}:          {n_fdr}/{n_rois}  (min q={p_fdr.min():.3f})")
print(f"Brain map generated:  {brain_ok}")
print(f"\nResults: {RESULTS_CSV}")
print("Reminder: exploratory localization, not a declared family; Family A was "
      "not FDR-significant; proper localized inference = NBS (Family C).")