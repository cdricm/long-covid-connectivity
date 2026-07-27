"""Family A fragmentation confound check (Schaefer-400) — runs immediately BEFORE
step3d, on the confirmatory sign strategy.

Tests whether graph fragmentation (disconnect severity in the confirmatory
10-25 % range) differs between COVID and CONTROL. A more fragmented graph
mechanically has lower Global Efficiency (and altered clustering / assortativity)
independent of biology, so differential fragmentation would confound the Family A
global AUC metrics. Especially relevant given the differential motion exclusion
(COVID 18.4 % vs CONTROL 9.1 %).

Position in the pipeline: this is a group-AWARE confirmatory validity check, not
part of the group-blind 3a/3b/3c diagnostic. It is bound to the confirmatory sign
strategy (config.CONFIRMATORY_SIGN_STRATEGY) and runs after that strategy is fixed,
right before step3d. Pearson arm: strategy is "positive" a priori, so it can run as
soon as step3b exists. Partial arm: runs only after the 3a/3b/3c diagnostic sets the
strategy (guarded below).

In: step3b_subject_disconnect.csv for the confirmatory strategy (from the
    _cross_strategy tree written by step3b), cohort via
    config.select_included_subjects(). If the confirmatory graphs are fully
    connected (e.g. partial positive-only, 162/162 connected in-range), the
    check is trivially negative — a clean, documented "no fragmentation
    confound" statement, not a reason to skip it.
Out: family_A/{strategy}/step3d_a_fragmentation_confound/
    step3d_a_group_balance.txt (text report)
    step3d_a_group_balance.png (supplement figure: severity + island size by group)

Inference (R2 change ②, CONFIRMED): Freedman-Lane residualised permutation
(10,000) under the shared covariate model intercept + mean-centred age + coded
sex + group; statistic = OLS group-coefficient t. The parametric p of that same
adjusted coefficient is reported as the sensitivity value, and the unadjusted
Welch t is retained alongside it as the R1 reference so the R1 -> R2 change stays
auditable. DESCRIPTIVE only — a validity check, not a hypothesis test: no FDR,
no "significant" claim. No Mann-Whitney (one inference framework throughout).

R1 (superseded, retained for the audit trail): naive permutation of group labels
(10,000), statistic = Welch's t of COVID vs CONTROL, no covariates.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step3d_auc_pipeline import freedman_lane_permutation, load_covariates

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

ATLAS         = "schaefer400"
COVID_LABEL   = "COVID"
CONTROL_LABEL = "CONTROL"
COL_COVID     = "firebrick"
COL_CONTROL   = "steelblue"
N_PERM        = config.N_PERMUTATIONS


# _welch_t is still called for the unadjusted R1 reference value in the report.
# naive_permutation is retained as the R1 reference path but is no longer called.
def _welch_t(a, b):
    """Welch's t (unequal variance) for b vs. a. Returns the t-statistic only."""
    t, _ = stats.ttest_ind(b, a, equal_var=False)
    return t


def naive_permutation(y, group, n_perm, seed_seq):
    """Permute group labels; statistic = Welch's t (COVID vs CONTROL), |t| two-sided.
    group: 0 = CONTROL, 1 = COVID. No covariates. Matches step3d exactly."""
    a_obs = y[group == 0]
    b_obs = y[group == 1]
    t_obs = _welch_t(a_obs, b_obs)
    rng = np.random.default_rng(seed_seq)
    n  = len(y)
    n1 = int((group == 0).sum())
    count = 0
    for _ in range(n_perm):
        yp = y[rng.permutation(n)]
        if abs(_welch_t(yp[:n1], yp[n1:])) >= abs(t_obs):
            count += 1
    return t_obs, (count + 1) / (n_perm + 1)


def cohens_d(a, b):
    """Raw Cohen's d (pooled SD), direction b - a (COVID - CONTROL). Descriptive."""
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((a.var(ddof=1)*(n1-1)) + (b.var(ddof=1)*(n2-1))) / (n1+n2-2))
    return (b.mean() - a.mean()) / pooled if pooled > 0 else np.nan


def main():
    # --- Guard: confirmatory strategy must be fixed before this check runs -----
    strategy = config.CONFIRMATORY_SIGN_STRATEGY
    assert strategy is not None, (
        "CONFIRMATORY_SIGN_STRATEGY is None — run the 3a/3b/3c diagnostic and set "
        "the confirmatory strategy before the Family A fragmentation confound check "
        "(it must run on the same strategy as the confirmatory analysis)."
    )

    # step3b wrote the diagnostic CSV per strategy into the _cross_strategy tree;
    # this confound check is confirmatory, so it READS from there but WRITES into
    # the confirmatory family_A/{strategy} tree.
    in_dir   = config.atlas_dir(ATLAS, f"step3b_diagnose/{strategy}", cross_strategy=True)
    csv_path = in_dir / "step3b_subject_disconnect.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"{csv_path} not found — run step3b for strategy '{strategy}' first."
        )
    out_dir = config.ensure(config.atlas_dir(ATLAS, "step3d_a_fragmentation_confound",
                                             strategy=strategy))

    df = pd.read_csv(csv_path)
    df["subject"] = df["subject"].astype(str).str.strip()
    df["group"]   = df["group"].astype(str).str.strip().str.upper()

    # Cross-check against config single source of truth
    group_df = pd.read_csv(config.GROUP_CSV)
    included = set(config.select_included_subjects(
        df["subject"].tolist(), group_df, verbose=False))
    in_csv  = set(df["subject"])
    extra   = sorted(in_csv - included)
    missing = sorted(included - in_csv)

    L = []
    def out(s=""):
        print(s); L.append(str(s))

    out(f"Atlas      : Schaefer-400 (confirmatory)")
    out(f"Strategy   : {strategy}  (confirmatory sign strategy)")
    if extra:
        out(f"WARNING: {len(extra)} subject(s) in step3b CSV not in config cohort "
            f"(stale step3b run?): {extra[:5]}{' ...' if len(extra) > 5 else ''}")
    if missing:
        out(f"WARNING: {len(missing)} config-cohort subject(s) missing from step3b CSV: "
            f"{missing[:5]}{' ...' if len(missing) > 5 else ''}")
    df = df[df["subject"].isin(included)].reset_index(drop=True)

    n_dens = int(df["n_densities_disconn"].max())

    out(f"N total    : {len(df)}")
    out(f"Group sizes: {df['group'].value_counts().to_dict()}")
    out(f"(Family A confound check; descriptive, no inference claim)\n")

    out("Disconnect severity by group:")
    for group in sorted(df["group"].unique()):
        sub = df[df["group"] == group]; n = len(sub)
        all_disc = int((sub["n_densities_disconn"] == n_dens).sum())
        out(f"  {group:8s} (n={n}): "
            f"all-{n_dens}-disc = {all_disc} ({100*all_disc/n:.1f}%), "
            f"mean disconn-densities = {sub['n_densities_disconn'].mean():.2f}, "
            f"mean max_islands = {sub['max_islands'].mean():.1f}")

    ctrl = df[df["group"] == CONTROL_LABEL]
    cov  = df[df["group"] == COVID_LABEL]

    # One reproducible substream per metric test, like step3d.
    metrics = [("n_densities_disconn", "Disconnect severity (n densities disconnected)"),
               ("max_islands",         "Island size (max ROIs outside main component)")]
    substreams = np.random.SeedSequence(config.SEED).spawn(len(metrics))

    # --- R2 ②: shared covariate model (age + sex), aligned to the y vector ----
    # y is built below as concatenate([CONTROL, COVID]), so the covariate rows
    # must follow exactly that subject order. They are keyed on the subject id,
    # never on row position. load_covariates raises on any missing/non-numeric
    # age or missing sex, so a covariate gap cannot enter silently.
    subj_order = np.concatenate([ctrl["subject"].values, cov["subject"].values])
    cov_df, sex_map = load_covariates(subj_order.tolist())
    cov_df = cov_df.set_index("subject")
    assert cov_df.index.is_unique, (
        "duplicate subject IDs in the covariate table — the .loc reindex below "
        "would silently return more rows than y has")
    missing_cov = [s for s in subj_order if s not in cov_df.index]
    if missing_cov:
        raise RuntimeError(f"covariates missing for {len(missing_cov)} subject(s): "
                           f"{missing_cov}")
    Z_all = cov_df.loc[subj_order, ["age", "sex_code"]].values.astype(float)
    assert Z_all.shape == (len(subj_order), 2), "covariate matrix shape mismatch"
    out(f"Covariates : age + sex, Freedman-Lane (se_type={config.FL_SE_TYPE}), "
        f"sex coding {sex_map}")

    results = {}
    # Fully-connected short-circuit: if nobody is ever disconnected, the confound
    # cannot exist; report it explicitly rather than permuting a constant.
    fully_connected = bool((df["n_densities_disconn"] == 0).all() and
                           (df["max_islands"] == 0).all())
    if fully_connected:
        out("\nAll subjects fully connected across the diagnostic densities — "
            "no fragmentation is present, so it cannot confound Family A. "
            "Confound excluded by construction (no test performed).")

    for i, (metric, label) in enumerate(metrics):
        c, v = ctrl[metric].values.astype(float), cov[metric].values.astype(float)
        d = cohens_d(c, v)
        out(f"\n{label}:")
        if fully_connected or (np.std(np.concatenate([c, v])) == 0):
            out(f"  constant across subjects — no test (no fragmentation to compare)")
            out(f"  Cohen's d (COVID - CONTROL) : {d:+.3f}")
            results[metric] = (d, np.nan)
            continue
        y = np.concatenate([c, v])
        g = np.concatenate([np.zeros(len(c)), np.ones(len(v))]).astype(int)
        # R2 ②: Freedman-Lane residualised permutation; statistic = OLS group
        # coefficient t under intercept + mean-centred age + coded sex + group.
        # Z_all rows are already in the same CONTROL-then-COVID order as y.
        fl = freedman_lane_permutation(y, g, Z_all, N_PERM, substreams[i],
                                       se_type=config.FL_SE_TYPE)
        t_obs, p_perm, p_param = fl["t_obs"], fl["p_perm"], fl["p_param"]
        t_w, p_w = stats.ttest_ind(v, c, equal_var=False)  # unadjusted R1 reference
        out(f"  Freedman-Lane perm (primary): t={t_obs:+.2f}  p={p_perm:.4f}  "
            f"({N_PERM} perms, age+sex adjusted group-coefficient t)")
        out(f"  Adjusted parametric p (sens): t={t_obs:+.2f}  p={p_param:.4f}")
        out(f"  Welch's t (R1, unadjusted)  : t={t_w:+.2f}  p={p_w:.4f}")
        out(f"  Cohen's d (COVID - CONTROL) : {d:+.3f}  (raw, unadjusted)")
        results[metric] = (d, p_perm)

    out("\nInterpretation guide:")
    out("  small |d| / non-significant -> fragmentation is not group-driven;")
    out("  Family A Efficiency/clustering differences are not a fragmentation artifact.")

    with open(out_dir / "step3d_a_group_balance.txt", "w") as f:
        f.write("\n".join(L) + "\n")

    # ===== Supplement figure =================================================
    d_sev, p_sev = results["n_densities_disconn"]
    d_isl, p_isl = results["max_islands"]
    p_sev_s = f"{p_sev:.3f}" if np.isfinite(p_sev) else "n/a"
    p_isl_s = f"{p_isl:.3f}" if np.isfinite(p_isl) else "n/a"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.arange(-0.5, n_dens + 1.5, 1)
    axes[0].hist(cov["n_densities_disconn"],  bins=bins, color=COL_COVID,
                 alpha=0.55, edgecolor="white", label=f"COVID (n={len(cov)})", density=True)
    axes[0].hist(ctrl["n_densities_disconn"], bins=bins, color=COL_CONTROL,
                 alpha=0.55, edgecolor="white", label=f"CONTROL (n={len(ctrl)})", density=True)
    axes[0].axvline(cov["n_densities_disconn"].mean(),  color=COL_COVID,   ls="--", lw=1)
    axes[0].axvline(ctrl["n_densities_disconn"].mean(), color=COL_CONTROL, ls="--", lw=1)
    axes[0].set_xlabel(f"# densities disconnected (0-{n_dens})")
    axes[0].set_ylabel("Density (normalized)")
    axes[0].set_title(f"Disconnect severity\nd={d_sev:+.2f}, FL perm p={p_sev_s}")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    isl_max = int(max(cov["max_islands"].max(), ctrl["max_islands"].max()))
    bins2 = np.linspace(0, max(isl_max, 1), 25)
    axes[1].hist(cov["max_islands"],  bins=bins2, color=COL_COVID,
                 alpha=0.55, edgecolor="white", label=f"COVID (n={len(cov)})", density=True)
    axes[1].hist(ctrl["max_islands"], bins=bins2, color=COL_CONTROL,
                 alpha=0.55, edgecolor="white", label=f"CONTROL (n={len(ctrl)})", density=True)
    axes[1].axvline(cov["max_islands"].mean(),  color=COL_COVID,   ls="--", lw=1)
    axes[1].axvline(ctrl["max_islands"].mean(), color=COL_CONTROL, ls="--", lw=1)
    axes[1].set_xlabel("Max ROIs outside main component")
    axes[1].set_ylabel("Density (normalized)")
    axes[1].set_title(f"Island size\nd={d_isl:+.2f}, FL perm p={p_isl_s}")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    fig.suptitle(f"Schaefer-400 ({strategy}) — graph fragmentation by group "
                 "(Family A confound check, descriptive; age+sex adjusted)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "step3d_a_group_balance.png", dpi=140)
    plt.close()
    out(f"\nFigure saved: {out_dir / 'step3d_a_group_balance.png'}")


if __name__ == "__main__":
    main()