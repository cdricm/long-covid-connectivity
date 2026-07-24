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
                     n_permutations=10000, seed=42):
    """Run Family A inference (naive permutation primary; no covariates).

    values_long : long DataFrame with columns [subject, group, range, metric, value].
                  Modularity is included as range='single' (no AUC). The group label
                  is taken directly from this frame; no external covariate file.
    confirmatory_ranges : ranges forming the 4-test confirmatory family
                  (default: literature-range AUC for the 3 metrics + modularity
                  'single'). FDR-BH is applied ONLY to the primary p_perm of these
                  tests. Other ranges (e.g. 'broad') are descriptive sensitivity,
                  NOT FDR-corrected.
    Returns a DataFrame with one row per (range, metric).
    """
    # Enumerate tests; assign one reproducible substream per test.
    # One permutation test per (range, metric) -> one reproducible substream each.
    test_keys = list(values_long.groupby(["range", "metric"]).groups.keys())
    substreams = np.random.SeedSequence(seed).spawn(len(test_keys))

    rows = []
    for i, (rng_label, metric) in enumerate(test_keys):
        sub = values_long[(values_long["range"] == rng_label) &
                          (values_long["metric"] == metric)].dropna(subset=["value"])
        g = sub[group_col].map({group_a: 0, group_b: 1}).values.astype(float)
        y = sub["value"].values.astype(float)
        keep = ~np.isnan(g)
        y, g = y[keep], g[keep]
        a, b = y[g == 0], y[g == 1]
        if len(a) < 2 or len(b) < 2:
            continue

        # PRIMARY: naive permutation, Welch-t statistic
        perm = naive_permutation(y, g, n_permutations, substreams[i])
        # SENSITIVITY: parametric Welch (same statistic, t-distribution null)
        t_welch, p_welch = stats.ttest_ind(b, a, equal_var=False)
        # EFFECT SIZE: raw Cohen's d (descriptive)
        d, d_lo, d_hi = cohens_d(a, b)

        # t_perm == t_welch (same observed Welch statistic); the two p-values differ
        # only in their null (permutation vs. t-distribution).
        rows.append({
            "range": rng_label, "metric": metric,
            "n_a": len(a), "n_b": len(b),
            "mean_a": a.mean(), "mean_b": b.mean(), "diff_b_a": b.mean() - a.mean(),
            "cohen_d": d, "d_ci_lo": d_lo, "d_ci_hi": d_hi,
            "t_perm": perm["t_obs"],   # observed Welch-t = permutation test statistic
            "p_perm": perm["p_perm"],  # PRIMARY (permutation null)
            "t_welch": t_welch, "p_welch": p_welch,  # sensitivity (parametric null)
        })

    res = pd.DataFrame(rows)
    if len(res) == 0:
        return res

    # FDR-BH over EXACTLY the confirmatory primary tests
    # (literature-range AUC for the 3 metrics + Modularity 'single' = 4 tests).
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
    for rng_label in sorted(values_long["range"].unique()):
        rng_df = values_long[values_long["range"] == rng_label]
        metrics = sorted(rng_df["metric"].unique())
        n = len(metrics); ncols = 2; nrows = (n + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4*nrows))
        axes = np.atleast_2d(axes).flatten()

        for ax, metric in zip(axes, metrics):
            sm = rng_df[rng_df["metric"] == metric]
            a = sm[sm[group_col] == group_a]["value"].dropna().values
            b = sm[sm[group_col] == group_b]["value"].dropna().values
            bp = ax.boxplot([a, b], positions=[1, 2], widths=0.5, patch_artist=True,
                            showfliers=False, medianprops=dict(color="black", linewidth=1.5))
            for patch, c in zip(bp["boxes"], ["#4878CF", "#EE854A"]):
                patch.set_facecolor(c); patch.set_alpha(0.6)
            ax.scatter(1 + jit.uniform(-0.12, 0.12, len(a)), a, s=12, alpha=0.5, color="#1E4D8C")
            ax.scatter(2 + jit.uniform(-0.12, 0.12, len(b)), b, s=12, alpha=0.5, color="#B45A18")
            ax.set_xticks([1, 2]); ax.set_xticklabels([f"{group_a}\n(n={len(a)})", f"{group_b}\n(n={len(b)})"])
            ax.set_ylabel(metric); ax.set_title(metric.replace("_", " ")); ax.grid(alpha=0.3, axis="y")
            r = comparison_df[(comparison_df["range"] == rng_label) & (comparison_df["metric"] == metric)]
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
    L = ["=" * 84, f"FAMILY A — GLOBAL GRAPH METRICS — {atlas_label}", "=" * 84,
         "PRIMARY = naive permutation of group labels, statistic = Welch's t",
         "(COVID vs CONTROL), no covariates. FDR-BH over the 4 confirmatory tests.",
         "Sensitivity: p_welch (parametric, same statistic). d = raw COVID-CONTROL.",
         "Sig (on p_perm_fdr): * <0.05, ** <0.01, *** <0.001", ""]
    for rng_label in sorted(comparison_df["range"].unique()):
        conf = (rng_label in confirmatory_ranges)
        tag = "CONFIRMATORY (FDR-corrected)" if conf else "SENSITIVITY (descriptive, no FDR)"
        L.append(f"--- Range '{rng_label}' — {tag} ---")
        L.append(f"  {'metric':<20} {'d':>7} {'95% CI':>15} {'t_perm':>8} "
                 f"{'p_perm':>9} {'p_fdr':>9} {'p_welch':>9}")
        L.append("  " + "-" * 88)
        sub = comparison_df[comparison_df["range"] == rng_label]
        for _, r in sub.iterrows():
            def sig(p):
                return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            ci = f"[{r['d_ci_lo']:+.2f},{r['d_ci_hi']:+.2f}]"
            fdr = r["p_perm_fdr"]
            fdr_s = f"{fdr:.4f}{sig(fdr)}" if pd.notna(fdr) else "    n/a"
            L.append(f"  {r['metric']:<20} {r['cohen_d']:+7.3f} {ci:>15} "
                     f"{r['t_perm']:+8.3f} {r['p_perm']:>9.4f} "
                     f"{fdr_s:>9} {r['p_welch']:>9.4f}")
        L.append("")
    return "\n".join(L)