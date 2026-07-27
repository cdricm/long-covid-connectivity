"""
Family A inference pipeline (Step 3d): AUC integration + group test.

Family A = 4 global scalars per subject:
  - 3 AUC metrics  : Global Efficiency, Mean Clustering, Assortativity (AUC over
                     the confirmatory 10-25 % range; np.trapezoid with the
                     x-argument; range-width-normalized to a mean metric value
                     over the range, comparable across ranges)
  - 1 single value : signed Modularity Q* (NO AUC; loaded from
                     step3c_modularity.csv)

Inference:
- PRIMARY     : naive permutation (10,000) of the group labels; test statistic =
                Welch's t (unequal-variance) of COVID vs. CONTROL. No covariates.
- SENSITIVITY : Welch's t-test (parametric p on the same statistic) — checks the
                distributional assumption behind the permutation null.
- CORRECTION  : FDR-BH over the 4 PRIMARY permutation p-values (confirmatory
                10-25 % AUC for the 3 metrics + Modularity Q*). The broad
                (5-50 %) range is a declared SENSITIVITY range: descriptive,
                not FDR-corrected, not confirmatory.
- EFFECT SIZE : raw Cohen's d with 95 % CI (Nakagawa & Cuthill 2007), descriptive.
- SEEDING     : SeedSequence(SEED).spawn(n_tests) — one independent reproducible
                substream per test.

Group coding: Cohen's d and the mean difference are COVID - CONTROL.

In: values_long / metrics_df DataFrames supplied by the calling wrapper (built
    from step3c_metrics.csv for the confirmatory strategy + step3c_modularity.csv).
Out: plot_family_a() writes family_a_{range}.png directly; compute_auc(),
     compare_family_a() and format_summary() return DataFrames/strings that the
     calling wrapper writes to family_a_values.csv, family_a_comparison.csv and
     summary.txt.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests


# ===== AUC computation ========================================================
def compute_auc(metrics_df, density_col="density", metric_cols=None,
                subject_col="subject", group_col="group", ranges=None):
    """AUC per subject/metric/range. np.trapezoid with x-argument (unequal spacing),
    normalized by range width -> mean metric value over the range."""
    if ranges is None:
        # Keys are part of the frozen analysis state (see compare_family_a).
        ranges = {"literature": config.AUC_RANGE_CONFIRMATORY,
                 "broad":       config.AUC_RANGE_SENSITIVITY}
    if metric_cols is None:
        raise ValueError("metric_cols required")

    rows = []
    for (subj, grp), sub in metrics_df.groupby([subject_col, group_col]):
        sub = sub.sort_values(density_col)
        for label, (lo, hi) in ranges.items():
            in_range = sub[(sub[density_col] >= lo - 1e-9) & (sub[density_col] <= hi + 1e-9)]
            if len(in_range) < 2:
                continue
            x = in_range[density_col].values
            for m in metric_cols:
                y = in_range[m].values
                auc = np.nan if np.any(np.isnan(y)) else np.trapezoid(y, x) / (x.max() - x.min())
                rows.append({"subject": subj, "group": grp, "range": label,
                             "range_lo": lo, "range_hi": hi, "metric": m,
                             "value": auc, "n_densities": len(in_range)})
    return pd.DataFrame(rows)


def validate_range(metrics_df, density_col="density",
                   lo=config.AUC_RANGE_CONFIRMATORY[0],
                   hi=config.AUC_RANGE_CONFIRMATORY[1]):
    densities = sorted(metrics_df[density_col].unique())
    in_range  = [d for d in densities if lo - 1e-9 <= d <= hi + 1e-9]
    return {"available_densities": densities, "in_range": in_range,
            "n_in_range": len(in_range), "covered": len(in_range) >= 2}


def check_cohort(*dfs, subject_col="subject"):
    """Cross-check the subject IDs of one or more subject-level DataFrames
    (e.g. the step3c sweep metrics, the step3c modularity table) against the
    config-defined analytical sample. Aborts on any mismatch — a stale or
    partial cache must never enter the Family A inference silently."""
    included = set(config.select_included_subjects(
        sorted(p.name for p in config.NII_ROOT.iterdir() if p.is_dir()),
        pd.read_csv(config.GROUP_CSV), verbose=False,
    ))
    for df in dfs:
        loaded = set(df[subject_col].unique())
        missing = sorted(included - loaded)
        extra   = sorted(loaded - included)
        if missing or extra:
            raise AssertionError(
                f"cohort mismatch — missing from data: {missing or 'none'}; "
                f"not in config cohort: {extra or 'none'}"
            )


# ===== Test statistic =========================================================
def _welch_t(a, b):
    """Welch's t (unequal variance) for b vs. a. Returns the t-statistic only.
    Shared by the permutation null and the parametric sensitivity test so both
    rest on exactly the same statistic."""
    t, _ = stats.ttest_ind(b, a, equal_var=False)
    return t


# ===== Inference ==============================================================
def naive_permutation(y, group, n_perm, seed_seq):
    """PRIMARY test: permute group labels, statistic = Welch's t (COVID vs CONTROL).

    group: 0 = CONTROL, 1 = COVID. No covariates. Two-sided p via |t|.
    """
    a_obs = y[group == 0]
    b_obs = y[group == 1]
    t_obs = _welch_t(a_obs, b_obs)

    rng = np.random.default_rng(seed_seq)
    n = len(y)
    n1 = int((group == 0).sum())
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        yp = y[perm]
        t_star = _welch_t(yp[:n1], yp[n1:])
        if abs(t_star) >= abs(t_obs):
            count += 1
    p_perm = (count + 1) / (n_perm + 1)
    return {"t_obs": t_obs, "p_perm": p_perm}


# ============================================================================
# R2 change ② — Freedman–Lane residualised permutation (covariate adjustment).
# ============================================================================
# STATUS: CONFIRMED by supervisor (R2 sign-off). This is the confirmatory primary
# test for Families A and B. naive_permutation above is retained as the R1
# reference path but is no longer the confirmatory analysis.
#
# METHOD (Freedman & Lane, 1983; Winkler et al., 2014, NeuroImage,
# 10.1016/j.neuroimage.2014.01.060 — canonical treatment for neuroimaging):
#   Test the group effect on y while adjusting for nuisance covariates Z (age,
#   sex). Same covariate model for ALL outcomes (supervisor). Full model =
#   1 + mean-centred age + coded sex + group.
#   1. Fit the REDUCED model  y ~ 1 + Z  -> fitted ŷ_red, residuals r = y - ŷ_red.
#   2. For each permutation π: form y* = ŷ_red + π(r)  (permute residuals only).
#   3. Fit the FULL model  y* ~ 1 + Z + group ; record the group coefficient's
#      t-value t*.
#   4. p = (#{|t*| >= |t_obs|} + 1) / (n_perm + 1), where t_obs is the group t
#      from the full model on the unpermuted y.
#
# CONFIRMED SETTINGS (supervisor):
#   [variance model] OLS group-t is the reported statistic (se_type="nonrobust").
#     Canonical Freedman–Lane; in a permutation test the null comes from the
#     permutation, not the SE formula, so robust SEs add little, and at
#     n_CONTROL=39 an HC3 estimator can itself be unstable. HC3 is retained as a
#     one-line SENSITIVITY switch (se_type="HC3"), not the primary analysis.
#   [effect size] raw Cohen's d, unchanged, labelled "unadjusted descriptive".
#     The permutation/parametric p-values are adjusted; d is a raw descriptive
#     companion (METHODS_DECISIONS). An adjusted effect size, if wanted later, is
#     an additional deliverable, not a change here.
#   [sex coding] data-driven 0/1 from the two observed categories (encode_sex);
#     >2 categories or any missing value -> hard error, never a silent miscoding.
#   [age] mean-centred inside the function (center_cols); sex left uncentred.
#   [tested coefficient] the GROUP coefficient only (last design column), never
#     age or sex.
#
# The covariate matrix Z is supplied by the caller (age, coded sex per subject,
# aligned to y). This function does not read any CSV — the caller aligns
# covariates to the same subject order as y, exactly as it aligns the group
# vector. Column order in Z MUST be (age, sex) so center_cols=(0,) centres age.

def _design_group_t(y, Z_with_group, group_col_idx, se_type):
    """OLS fit of y on the full design (intercept + covariates + group); return
    (t-value, two-sided parametric p-value) of the group column.
    se_type: 'nonrobust' or 'HC3'."""
    import statsmodels.api as sm
    model = sm.OLS(y, Z_with_group).fit(
        cov_type=("HC3" if se_type == "HC3" else "nonrobust"))
    return model.tvalues[group_col_idx], model.pvalues[group_col_idx]


def freedman_lane_permutation(y, group, Z, n_perm, seed_seq,
                              se_type="nonrobust", center_cols=(0,)):
    """R2 ② PRIMARY test (CONFIRMED): covariate-adjusted permutation (Freedman–Lane).

    Same covariate model for all outcomes: design = intercept + mean-centred age
    + coded sex + group. The permutation tests the GROUP COEFFICIENT only.

    Parameters
    ----------
    y : (n,) float      response (per-subject measure / AUC value).
    group : (n,) int    0 = CONTROL, 1 = COVID.
    Z : (n, k) float    nuisance covariates, column order (age, coded sex). NO
                        intercept column (added here). Same subject order as y.
    n_perm : int        permutation count.
    seed_seq : SeedSequence   reproducible per-test substream.
    se_type : str       'nonrobust' (OLS, default/primary) or 'HC3' (sensitivity).
    center_cols : tuple  indices of Z columns to mean-centre (default: (0,) = age).
                        Sex (0/1) is left uncentred.

    Returns
    -------
    dict {"t_obs": full-model group t, "p_perm": FL two-sided permutation p,
          "p_param": parametric two-sided p of the group coefficient (same OLS
          fit; adjusted-sensitivity companion to the permutation null)}.

    NOTE: statistic is the FULL-MODEL group t, not Welch t. The permutation null
    and the parametric p rest on the same adjusted statistic; they differ only in
    the null (permutation vs. t-distribution), mirroring the Pearson-arm logic.
    """
    import numpy as np
    y = np.asarray(y, float)
    group = np.asarray(group, float).reshape(-1, 1)
    Z = np.array(Z, float)          # copy: we mean-centre in place below
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    # Mean-centre the requested covariate columns (age). Centring does not change
    # the group coefficient in a purely additive model, but it is explicitly
    # required (supervisor) and improves conditioning of the design.
    for c in center_cols:
        Z[:, c] = Z[:, c] - Z[:, c].mean()
    n = len(y)
    intercept = np.ones((n, 1))

    # Reduced model design: intercept + covariates (NO group).
    X_red = np.hstack([intercept, Z])
    # Full model design: intercept + covariates + group (group is LAST column).
    X_full = np.hstack([intercept, Z, group])
    group_idx = X_full.shape[1] - 1

    # Observed full-model group t + parametric p (adjusted sensitivity).
    t_obs, p_param = _design_group_t(y, X_full, group_idx, se_type)

    # Reduced-model fit -> fitted values + residuals (least squares).
    beta_red, *_ = np.linalg.lstsq(X_red, y, rcond=None)
    y_red = X_red @ beta_red
    resid = y - y_red

    rng = np.random.default_rng(seed_seq)
    count = 0
    for _ in range(n_perm):
        y_star = y_red + rng.permutation(resid)
        t_star, _p = _design_group_t(y_star, X_full, group_idx, se_type)
        if abs(t_star) >= abs(t_obs):
            count += 1
    p_perm = (count + 1) / (n_perm + 1)
    return {"t_obs": t_obs, "p_perm": p_perm, "p_param": p_param}


def _group_t_vec(Y, Xf, gi, XtX_inv):
    """Vectorised OLS group-coefficient t across the columns of Y (one per ROI).
    Y: (n, n_roi); Xf: (n, p) full design with group as column gi. Returns (n_roi,)."""
    import numpy as np
    beta  = XtX_inv @ (Xf.T @ Y)               # (p, n_roi)
    resid = Y - Xf @ beta                      # (n, n_roi)
    n, p  = Xf.shape
    sigma2 = (resid**2).sum(axis=0) / (n - p)  # (n_roi,)
    se_g   = np.sqrt(sigma2 * XtX_inv[gi, gi]) # (n_roi,)
    return beta[gi] / se_g


def freedman_lane_vectorized(Y, group, Z, n_perm, seed, center_cols=(0,)):
    """Vectorised Freedman–Lane over many outcomes (columns of Y), e.g. ROI-wise
    nodal strength or within-network ROI FC. Same statistic and null as
    freedman_lane_permutation (OLS group-coefficient t under intercept +
    mean-centred age + coded sex + group), broadcast across the outcome axis so
    all columns are done in one lstsq per permutation. Verified byte-identical to
    the per-test function (t_obs, p_param, and p_perm under the same permutation
    sequence match to floating-point precision).

    Y : (n_subjects, n_outcomes) float.
    group : (n,) 0/1 (1 = COVID). Z : (n, k) covariates (age, coded sex), no
    intercept. Returns (t_obs, p_perm, p_param), each (n_outcomes,).
    OLS only (nonrobust); the vectorised path does not implement HC3."""
    import numpy as np
    Y = np.asarray(Y, float)
    group = np.asarray(group, float).reshape(-1, 1)
    Z = np.array(Z, float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    for c in center_cols:                       # mean-centre age (supervisor)
        Z[:, c] = Z[:, c] - Z[:, c].mean()
    n = Y.shape[0]
    ic = np.ones((n, 1))
    Xr = np.hstack([ic, Z])                     # reduced: intercept + covariates
    Xf = np.hstack([ic, Z, group])              # full: + group (last col)
    gi = Xf.shape[1] - 1
    XtX_inv = np.linalg.inv(Xf.T @ Xf)

    t_obs = _group_t_vec(Y, Xf, gi, XtX_inv)
    abs_t = np.abs(t_obs)
    dof = n - Xf.shape[1]
    p_param = 2 * stats.t.sf(abs_t, dof)        # parametric p of the same coeff

    beta_r = np.linalg.lstsq(Xr, Y, rcond=None)[0]
    Yr = Xr @ beta_r
    resid = Y - Yr

    rng = np.random.default_rng(seed)
    count = np.zeros(Y.shape[1], dtype=np.int64)
    for _ in range(n_perm):
        perm = rng.permutation(n)
        t_star = _group_t_vec(Yr + resid[perm], Xf, gi, XtX_inv)
        count += (np.abs(t_star) >= abs_t)
    p_perm = (count + 1) / (n_perm + 1)
    return t_obs, p_perm, p_param


def encode_sex(sex_values):
    """Data-driven 0/1 coding of sex from its two observed categories (SWITCH 3).

    No hard-coded label mapping. >2 non-null categories or any missing value ->
    AssertionError, so a miscoding can never enter the model silently. Returns
    (codes float array, mapping dict) for logging/reproducibility.
    """
    import numpy as np
    import pandas as pd
    s = pd.Series(sex_values)
    if s.isna().any():
        raise AssertionError(
            f"sex has {int(s.isna().sum())} missing value(s); Freedman–Lane "
            "cannot proceed without complete covariates")
    cats = sorted(s.unique().tolist())
    if len(cats) != 2:
        raise AssertionError(
            f"sex must have exactly 2 categories for 0/1 coding; found {cats}")
    mapping = {cats[0]: 0.0, cats[1]: 1.0}
    return s.map(mapping).values.astype(float), mapping


def load_covariates(subjects, *, id_col="ID", age_col="Edad", sex_col="Genero"):
    """Load the R2 ② covariate model (age, sex) for the given subjects from
    config.GROUP_CSV and return a DataFrame [subject, age, sex_code] aligned by
    subject id, plus the sex mapping for logging.

    Same covariate model for all outcomes (supervisor): this single loader feeds
    Family A (step3d) and Family B (step4d) so the model is identical everywhere.
    Age is kept raw here (parsed to float); mean-centring happens inside
    freedman_lane_permutation. Any missing/non-numeric age or missing sex among
    the requested subjects -> hard error (no silent covariate gaps).
    """
    import numpy as np
    import pandas as pd
    df = pd.read_csv(config.GROUP_CSV)
    df[id_col] = df[id_col].astype(str).str.strip()
    subjects = [str(s) for s in subjects]
    df = df[df[id_col].isin(subjects)].copy()

    missing_rows = sorted(set(subjects) - set(df[id_col]))
    if missing_rows:
        raise AssertionError(
            f"covariates: {len(missing_rows)} subject(s) not found in GROUP_CSV: "
            f"{missing_rows}")

    age = pd.to_numeric(df[age_col], errors="coerce")
    if age.isna().any():
        bad = df.loc[age.isna(), id_col].tolist()
        raise AssertionError(
            f"covariates: non-numeric/missing age for {len(bad)} subject(s): {bad}")

    sex_raw = pd.Series(df[sex_col]).astype(str).str.strip().values
    sex_code, sex_map = encode_sex(sex_raw)

    out = pd.DataFrame({
        "subject": df[id_col].values,
        "age": age.values.astype(float),
        "sex_code": sex_code,
    })
    # Order-independent: caller merges on 'subject'.
    print(f"  covariates loaded: N={len(out)}, age mean={out['age'].mean():.1f} "
          f"[{out['age'].min():.0f}-{out['age'].max():.0f}], sex coding {sex_map}")
    return out, sex_map


def cohens_d(a, b):
    """Raw Cohen's d (pooled SD) with 95% CI (large-sample approximation,
    Nakagawa & Cuthill 2007). Direction: b - a (COVID - CONTROL)."""
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((a.var(ddof=1)*(n1-1)) + (b.var(ddof=1)*(n2-1))) / (n1+n2-2))
    d = (b.mean() - a.mean()) / pooled if pooled > 0 else np.nan
    se = np.sqrt((n1+n2)/(n1*n2) + d**2/(2*(n1+n2)))
    return d, d - 1.96*se, d + 1.96*se


# ===== Family A comparison ====================================================
def compare_family_a(values_long, *, confirmatory_ranges=("literature", "single"),
                     group_col="group", group_a="CONTROL", group_b="COVID",
                     n_permutations=10000, seed=42,
                     covariate_cols=("age", "sex_code"), se_type="nonrobust"):
    """Run Family A inference — R2: Freedman–Lane covariate-adjusted permutation.

    values_long : long DataFrame with columns [subject, group, range, metric,
                  value] PLUS the covariate columns (age, sex_code) carried per
                  subject so they align to y row-for-row (no external merge).
                  MAY ALSO carry a 'subgraph' column (R2 ⑤): the edge-sign
                  subgraph a metric was computed on ('positive'/'negative'). When
                  absent it defaults to a single constant 'positive' level, so the
                  Pearson arm (one subgraph) is byte-identical to the pre-⑤ code;
                  the partial arm supplies both levels -> 3 metrics x 2 subgraphs
                  + Modularity = 7 tests, one FDR family.
    confirmatory_ranges : ranges forming the confirmatory family. FDR-BH is
                  applied ONLY to the primary p_perm of these tests.
    covariate_cols : column names in values_long holding (age, coded sex), in
                  that order. Age is mean-centred inside freedman_lane_permutation.
    se_type : 'nonrobust' (OLS, primary) or 'HC3' (sensitivity).

    PRIMARY test = Freedman–Lane residualised permutation, statistic = OLS
    group-coefficient t (age + sex adjusted). SENSITIVITY = the parametric p of
    the same adjusted group coefficient. Effect size = raw Cohen's d (unadjusted
    descriptive). Returns one row per (range, metric, subgraph).
    """
    values_long = values_long.copy()
    # R2 ⑤: the edge-sign subgraph is an explicit test dimension. Absent -> a
    # single constant level, which leaves the (range, metric) grouping — and thus
    # the per-test substream assignment — unchanged from the pre-⑤ Pearson arm.
    if "subgraph" not in values_long.columns:
        values_long["subgraph"] = "positive"

    # Enumerate tests; assign one reproducible substream per test. Grouping now
    # includes subgraph; with a constant subgraph the key order is identical to
    # the (range, metric) order, so the Pearson seed mapping is preserved.
    test_keys = list(values_long.groupby(["range", "metric", "subgraph"]).groups.keys())
    substreams = np.random.SeedSequence(seed).spawn(len(test_keys))

    rows = []
    for i, (rng_label, metric, subgraph) in enumerate(test_keys):
        sub = values_long[(values_long["range"] == rng_label) &
                          (values_long["metric"] == metric) &
                          (values_long["subgraph"] == subgraph)].dropna(subset=["value"])
        g = sub[group_col].map({group_a: 0, group_b: 1}).values.astype(float)
        y = sub["value"].values.astype(float)
        Z = sub[list(covariate_cols)].values.astype(float)  # (n, 2): age, sex_code
        # keep rows with a valid group AND complete covariates (FL needs both).
        keep = ~np.isnan(g) & np.isfinite(y) & np.isfinite(Z).all(axis=1)
        y, g, Z = y[keep], g[keep], Z[keep]
        a, b = y[g == 0], y[g == 1]
        if len(a) < 2 or len(b) < 2:
            continue

        # PRIMARY: Freedman–Lane covariate-adjusted permutation (group-coeff t).
        perm = freedman_lane_permutation(y, g, Z, n_permutations, substreams[i],
                                         se_type=se_type)
        # SENSITIVITY: parametric p of the SAME adjusted group coefficient.
        p_param = perm["p_param"]
        # EFFECT SIZE: raw Cohen's d (unadjusted descriptive companion).
        d, d_lo, d_hi = cohens_d(a, b)

        rows.append({
            "range": rng_label, "metric": metric, "subgraph": subgraph,
            "n_a": len(a), "n_b": len(b),
            "mean_a": a.mean(), "mean_b": b.mean(), "diff_b_a": b.mean() - a.mean(),
            "cohen_d": d, "d_ci_lo": d_lo, "d_ci_hi": d_hi,
            "t_perm": perm["t_obs"],   # observed adjusted group-coefficient t
            "p_perm": perm["p_perm"],  # PRIMARY (FL permutation null)
            "t_welch": np.nan,         # R1 Welch statistic no longer computed
            "p_welch": p_param,        # SENSITIVITY: parametric p of adjusted group t
        })

    res = pd.DataFrame(rows)
    if len(res) == 0:
        return res

    # FDR-BH over EXACTLY the confirmatory primary tests (all subgraphs of the
    # confirmatory ranges form ONE family: Pearson 4, partial 7).
    res["family_A_confirmatory"] = res["range"].isin(confirmatory_ranges)
    conf = res[res["family_A_confirmatory"]].copy()
    res["p_perm_fdr"] = np.nan
    if len(conf) > 0:
        _, p_fdr, _, _ = multipletests(conf["p_perm"].values, method="fdr_bh")
        res.loc[conf.index, "p_perm_fdr"] = p_fdr
    return res


# ===== Plotting ===============================================================
def plot_family_a(values_long, comparison_df, out_dir, atlas_label,
                  confirmatory_ranges=("literature", "single"),
                  group_col="group", group_a="CONTROL", group_b="COVID"):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    jit = np.random.default_rng(config.SEED)
    values_long = values_long.copy()
    if "subgraph" not in values_long.columns:
        values_long["subgraph"] = "positive"
    has_sub = values_long["subgraph"].nunique() > 1
    for rng_label in sorted(values_long["range"].unique()):
        rng_df = values_long[values_long["range"] == rng_label]
        # Panel per (metric, subgraph) so positive/negative subgraphs of the same
        # metric are never pooled into one boxplot (R2 ⑤).
        panels = sorted(rng_df.groupby(["metric", "subgraph"]).groups.keys())
        n = len(panels); ncols = 2; nrows = (n + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4*nrows))
        axes = np.atleast_2d(axes).flatten()

        for ax, (metric, subgraph) in zip(axes, panels):
            sm = rng_df[(rng_df["metric"] == metric) &
                        (rng_df["subgraph"] == subgraph)]
            a = sm[sm[group_col] == group_a]["value"].dropna().values
            b = sm[sm[group_col] == group_b]["value"].dropna().values
            bp = ax.boxplot([a, b], positions=[1, 2], widths=0.5, patch_artist=True,
                            showfliers=False, medianprops=dict(color="black", linewidth=1.5))
            for patch, c in zip(bp["boxes"], ["#4878CF", "#EE854A"]):
                patch.set_facecolor(c); patch.set_alpha(0.6)
            ax.scatter(1 + jit.uniform(-0.12, 0.12, len(a)), a, s=12, alpha=0.5, color="#1E4D8C")
            ax.scatter(2 + jit.uniform(-0.12, 0.12, len(b)), b, s=12, alpha=0.5, color="#B45A18")
            ax.set_xticks([1, 2]); ax.set_xticklabels([f"{group_a}\n(n={len(a)})", f"{group_b}\n(n={len(b)})"])
            panel_label = f"{metric} [{subgraph}]" if has_sub else metric
            ax.set_ylabel(metric); ax.set_title(panel_label.replace("_", " ")); ax.grid(alpha=0.3, axis="y")
            r = comparison_df[(comparison_df["range"] == rng_label) &
                              (comparison_df["metric"] == metric) &
                              (comparison_df["subgraph"] == subgraph)]
            if len(r):
                r = r.iloc[0]
                fdr_str = f"{r['p_perm_fdr']:.4f}" if pd.notna(r['p_perm_fdr']) else "n/a (sens.)"
                txt = (f"d = {r['cohen_d']:+.2f} [{r['d_ci_lo']:+.2f},{r['d_ci_hi']:+.2f}]\n"
                       f"p_perm = {r['p_perm']:.4f}\n"
                       f"p_perm_fdr = {fdr_str}\n"
                       f"p_welch = {r['p_welch']:.4f}")
                ax.text(0.98, 0.02, txt, transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                              edgecolor="grey", alpha=0.85))
        for ax in axes[n:]:
            ax.set_visible(False)
        conf_tag = " [CONFIRMATORY]" if rng_label in confirmatory_ranges else " [sensitivity]"
        fig.suptitle(f"{atlas_label} — Family A, range '{rng_label}'{conf_tag}", fontsize=12)
        plt.tight_layout()
        p = os.path.join(out_dir, f"family_a_{rng_label}.png")
        plt.savefig(p, dpi=140); plt.close(); saved.append(p)
    return saved


# ===== Summary ================================================================
def format_summary(comparison_df, atlas_label, confirmatory_ranges=("literature", "single")):
    n_conf = int(comparison_df.get("family_A_confirmatory",
                                   comparison_df["range"].isin(confirmatory_ranges)).sum())
    L = ["=" * 84, f"FAMILY A — GLOBAL GRAPH METRICS — {atlas_label}", "=" * 84,
         "PRIMARY = Freedman–Lane permutation, statistic = OLS group-coefficient t",
         "(COVID vs CONTROL), age + sex adjusted. FDR-BH over the "
         f"{n_conf} confirmatory tests.",
         "Sensitivity: p_welch (parametric p of the same adjusted coefficient).",
         "d = raw COVID-CONTROL (unadjusted descriptive).",
         "Sig (on p_perm_fdr): * <0.05, ** <0.01, *** <0.001", ""]
    has_sub = "subgraph" in comparison_df.columns and \
        comparison_df["subgraph"].nunique() > 1
    for rng_label in sorted(comparison_df["range"].unique()):
        conf = (rng_label in confirmatory_ranges)
        tag = "CONFIRMATORY (FDR-corrected)" if conf else "SENSITIVITY (descriptive, no FDR)"
        L.append(f"--- Range '{rng_label}' — {tag} ---")
        mlabel = "metric [subgraph]" if has_sub else "metric"
        L.append(f"  {mlabel:<28} {'d':>7} {'95% CI':>15} {'t_perm':>8} "
                 f"{'p_perm':>9} {'p_fdr':>9} {'p_welch':>9}")
        L.append("  " + "-" * 96)
        sub = comparison_df[comparison_df["range"] == rng_label]
        for _, r in sub.iterrows():
            def sig(p):
                return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            ci = f"[{r['d_ci_lo']:+.2f},{r['d_ci_hi']:+.2f}]"
            fdr = r["p_perm_fdr"]
            fdr_s = f"{fdr:.4f}{sig(fdr)}" if pd.notna(fdr) else "    n/a"
            name = f"{r['metric']} [{r['subgraph']}]" if has_sub else r["metric"]
            L.append(f"  {name:<28} {r['cohen_d']:+7.3f} {ci:>15} "
                     f"{r['t_perm']:+8.3f} {r['p_perm']:>9.4f} "
                     f"{fdr_s:>9} {r['p_welch']:>9.4f}")
        L.append("")
    return "\n".join(L)