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
(descriptive).

Inference (R2 change ②, CONFIRMED): Freedman-Lane residualised permutation
(10,000) under the shared covariate model intercept + mean-centred age + coded
sex + group; statistic = OLS group-coefficient t, vectorised over the 400 ROIs.
The parametric p of that same adjusted coefficient is the sensitivity value.
FDR-BH over the 400 ROIs. The vectorised FL is verified byte-identical to the
per-test freedman_lane_permutation used by Families A/B (same statistic, same
permutation null). SeedSequence(SEED) drives the permutation stream.

R1 (superseded, retained for the audit trail): naive label-permutation, Welch t,
no covariates.
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
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

# step3f lives beside step3d_auc_pipeline; import the covariate loader and the
# vectorised Freedman–Lane so the SAME covariate model / statistic feeds every
# outcome (supervisor requirement) and there is one verified implementation.
sys.path.insert(0, str(_HERE.parent))
from step3d_auc_pipeline import load_covariates, freedman_lane_vectorized

# =====================================================================
# SETTINGS
# =====================================================================
ATLAS      = "schaefer400"
# R2 ⑤: the partial arm treats positive and negative subgraphs SEPARATELY (each
# a full exploratory ROI screen with its own FDR over 400 ROIs), because positive
# and negative conditional associations are distinct objects and share no common
# null — pooling them into one FDR would assert a family they do not form. The
# Pearson arm keeps a single subgraph ("positive"; its negative subgraph is
# degenerate). Nodal strength is NOT the signed S+/S- decomposition (Rubinov &
# Sporns 2011) — it is the positive-only / negative-only subgraph via
# config.apply_sign_strategy, matching the Family-A ⑤ construction.
if config.FC_METHOD == "partial":
    SUBGRAPH_STRATEGIES = ["positive", "negative"]
elif config.FC_METHOD == "pearson":
    SUBGRAPH_STRATEGIES = [config.CONFIRMATORY_SIGN_STRATEGY]   # "positive"
else:
    SUBGRAPH_STRATEGIES = [None]   # will trip the assert below
MATRIX_DIR = config.atlas_dir(ATLAS, "step2_pipeline") / "comet_matrices"

DENSITIES  = np.arange(config.AUC_RANGE_CONFIRMATORY[0],
                       config.AUC_RANGE_CONFIRMATORY[1] + 0.0001, 0.01)
N_BOOT     = 5_000
N_PERM     = config.N_PERMUTATIONS
SEED       = config.SEED
N_JOBS     = config.N_JOBS_DEFAULT
FDR_Q      = config.FDR_ALPHA
FL_SE_TYPE = config.FL_SE_TYPE   # "nonrobust" (OLS) primary
USE_CACHE  = True

# =====================================================================
# HELPERS
# =====================================================================
def fisher_z(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))

def auc_strength_per_subject(matrix_path, densities, strategy):
    R = np.load(matrix_path)
    np.fill_diagonal(R, 0.0)
    Zp = config.apply_sign_strategy(fisher_z(R), strategy)
    strengths = np.zeros((len(densities), Zp.shape[0]))
    for i, d in enumerate(densities):
        strengths[i] = config.proportional_threshold(Zp, d)[0].sum(axis=1)
    return np.trapezoid(strengths, x=densities, axis=0)

def cohens_d_vec(x, y):
    """Vectorized d across ROIs (x=COVID, y=CONTROL). d>0 -> COVID>CONTROL."""
    nx, ny = len(x), len(y)
    s = np.sqrt(((nx-1)*x.var(ddof=1, axis=0) + (ny-1)*y.var(ddof=1, axis=0)) / (nx+ny-2))
    return (x.mean(axis=0) - y.mean(axis=0)) / s

# ---- R2 ②: vectorised Freedman-Lane over ROIs (imported from step3d_auc_pipeline)
# Same statistic as freedman_lane_permutation (OLS group-coefficient t under
# intercept + mean-centred age + coded sex + group), broadcast across the ROI
# axis. Verified identical to the per-test pipeline function.

# =====================================================================
# 1) COHORT + MATCH MATRICES  (strategy-independent — done once)
# =====================================================================
assert config.FC_METHOD in ("pearson", "partial"), (
    f"FC_METHOD must be 'pearson' or 'partial'; got {config.FC_METHOD!r}.")
assert all(s is not None for s in SUBGRAPH_STRATEGIES), (
    "sign strategy unresolved — Pearson uses 'positive'; partial uses the "
    "'positive'/'negative' split (R2 ⑤).")

group_df = pd.read_csv(config.GROUP_CSV)
on_disk = {p.stem.replace("_connectivity_comet", "")
           for p in MATRIX_DIR.glob("CP*_connectivity_comet.npy")}
included = config.select_included_subjects(sorted(on_disk), group_df, verbose=False)
id_to_path = {p.stem.replace("_connectivity_comet", ""): p
              for p in MATRIX_DIR.glob("CP*_connectivity_comet.npy")}
gmap = {str(i).strip(): str(g).strip() for i, g in zip(group_df["ID"], group_df["Grupo"])}

subjects = [s for s in included if s in id_to_path]
groups_arr = np.array([gmap[s] for s in subjects])
print(f"Subgraph strategies: {SUBGRAPH_STRATEGIES} "
      f"({'partial pos/neg split' if config.FC_METHOD == 'partial' else 'single'})")
print(f"Subjects (config): {len(subjects)}  "
      f"(COVID={(groups_arr=='COVID').sum()}, CONTROL={(groups_arr=='CONTROL').sum()})")

# Hard cohort guard: node-level analysis runs on the frozen cohort (covariates
# adjust, they do not drop subjects).
assert len(subjects) == 162, f"cohort mismatch: {len(subjects)} != 162"
assert (groups_arr == "COVID").sum() == 123 and (groups_arr == "CONTROL").sum() == 39, \
    "group sizes deviate from frozen cohort 123/39"

covid_mask = (groups_arr == "COVID")

# --- R2 ②: covariate model (age, sex), aligned to the subject order (once) ---
cov_df, sex_map = load_covariates(subjects)
cov_df = cov_df.set_index("subject")
assert cov_df.index.is_unique, "duplicate subject IDs in covariate table"
missing_cov = [s for s in subjects if s not in cov_df.index]
if missing_cov:
    raise RuntimeError(f"covariates missing for {len(missing_cov)} subject(s): {missing_cov}")
Z_all = cov_df.loc[subjects, ["age", "sex_code"]].values.astype(float)
assert Z_all.shape == (len(subjects), 2), "covariate matrix shape mismatch"
group_bin = covid_mask.astype(float)   # 1 = COVID, 0 = CONTROL (matches d sign)
print(f"Covariates: age + sex, Freedman-Lane (se_type={FL_SE_TYPE}), sex coding {sex_map}")


# =====================================================================
# Per-subgraph analysis (R2 ⑤): one full exploratory ROI screen per strategy,
# each with its own AUC cache, FL inference, FDR over 400 ROIs, and plots.
# =====================================================================
def run_subgraph(strategy):
    OUT_DIR = config.ensure(config.step3f_dir(ATLAS, strategy=strategy))
    STRENGTH_CACHE = OUT_DIR / "roi_strength_subjects.npy"
    SUBJECTS_CACHE = OUT_DIR / "roi_strength_subjects_ids.csv"
    RESULTS_CSV    = OUT_DIR / "roi_nodal_strength_results.csv"
    print(f"\n{'#'*64}\n# SUBGRAPH: {strategy}\n{'#'*64}")

    # ---- AUC nodal strength per subject (cached per strategy) ----
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
            delayed(auc_strength_per_subject)(id_to_path[s], DENSITIES, strategy)
            for s in subjects)
        S = np.vstack(S_list)
        np.save(STRENGTH_CACHE, S)
        pd.DataFrame({"ID": subjects}).to_csv(SUBJECTS_CACHE, index=False)
        print(f"Saved {STRENGTH_CACHE}, shape={S.shape}")

    n_rois = S.shape[1]
    x_covid, x_control = S[covid_mask], S[~covid_mask]
    # ---- Cohen's d + bootstrap CI (descriptive, unadjusted) ----
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
    # 4) FREEDMAN-LANE PER ROI (exploratory inference, age+sex adjusted)
    #    OLS group-coefficient t, vectorised over ROIs; permutation null + the
    #    parametric p of the same adjusted coefficient as sensitivity.
    # =====================================================================
    print(f"Freedman-Lane permutation ({N_PERM} perms, age+sex adjusted, "
          f"vectorized over {n_rois} ROIs) ...")
    t_obs, p_perm, p_param = freedman_lane_vectorized(
        S, group_bin, Z_all, N_PERM, np.random.SeedSequence(SEED))
    p_fdr = multipletests(p_perm, alpha=FDR_Q, method="fdr_bh")[1]

    # =====================================================================
    # 5) SAVE
    # =====================================================================
    out = pd.DataFrame({
        "roi_idx": np.arange(n_rois),
        "cohens_d": d_obs, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "ci_excludes_zero": (ci_lo > 0) | (ci_hi < 0),   # descriptive flag, NOT a test
        "t_obs": t_obs,
        "p_perm": p_perm, "p_param": p_param, "p_fdr": p_fdr,
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
    ax.set_title(f"Schaefer-400 nodal strength [{strategy}] — top 25 |d| (descriptive, bootstrap 95% CI)")
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
            f"Freedman-Lane permutation, age+sex adjusted; FDR-BH over {n_rois} ROIs\n"
            f"FDR q<{FDR_Q}: {n_fdr}/{n_rois} survive   (min q={p_fdr.min():.3f})\n"
            f"uncorrected p<0.05: {n_unc}/{n_rois}  (chance expectation ~{expected_chance:.0f})",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5, style="italic",
            color=("red" if n_fdr == 0 else "black"),
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.92))
    xmax = np.max(np.abs(d_obs)) * 1.15
    ax.set_xlim(-xmax, xmax)
    ax.set_xlabel("Cohen's d (COVID - CONTROL)"); ax.set_ylabel(r"$-\log_{10}(p_{\mathrm{perm}})$")
    ax.set_title(f"Schaefer-400 nodal strength [{strategy}]: ROI-level volcano (EXPLORATORY)\n"
                 f"Freedman-Lane (age+sex adjusted, {N_PERM} perm, seed={SEED}); "
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
                                      title=f"Schaefer-400 nodal strength [{strategy}] d (COVID-CONTROL), descriptive")
        disp.savefig(OUT_DIR / "roi_nodal_strength_dmap.png", dpi=150); disp.close()
        brain_ok = True
    except Exception as e:
        print(f"[brain map skipped] {type(e).__name__}: {str(e)[:140]}")

    # =====================================================================
    # 7) CONSOLE SUMMARY
    # =====================================================================
    print(f"\n=========== ROI Nodal Strength [{strategy}] (EXPLORATORY) ===========")
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

# =====================================================================
# RUN — one exploratory screen per subgraph strategy (R2 ⑤)
# =====================================================================
if __name__ == "__main__":
    for _strat in SUBGRAPH_STRATEGIES:
        run_subgraph(_strat)