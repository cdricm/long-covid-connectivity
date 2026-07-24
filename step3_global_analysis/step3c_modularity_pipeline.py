"""
Step 3c signed Modularity Q*: one value per subject on the signed, full,
unthresholded matrix (density-independent, no thresholding, no AUC).

In: cached matrices (paths_cfg["mat_dir"]), cohort via
    config.select_included_subjects().
Out: step3c_modularity.csv, step3c_modularity_summary.txt,
     step3c_modularity.png.

BCT call: modularity_louvain_und_sign(C, qtype='sta', seed=s) — signed because
positive/negative weights play intrinsically unequal roles (Rubinov & Sporns,
2011), unlike the positive-only AUC metrics. Louvain is stochastic; Q* = mean
over N_RUNS independent runs via SeedSequence(config.SEED).spawn(N_RUNS)
(per-subject SD/95% CI reported as a stability check). Cohen's d / Welch-t here
are descriptive previews only — the Family A inferential test (naive
permutation + FDR-BH) is step3d.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step3a_threshold_sweep_pipeline import _resolve_cohort

QTYPE = "sta"


def _louvain_seeds(n_runs, base_seed):
    """n_runs reproducible, independent int seeds from SeedSequence(base_seed)."""
    children = np.random.SeedSequence(base_seed).spawn(n_runs)
    return [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]


def run_modularity(paths_cfg, mod_cfg):
    """Compute signed Modularity Q* (multi-run mean) per subject."""
    config.ensure(Path(paths_cfg["out_dir"]))

    import comet
    from comet.graph import bct

    subjects, group_of, meta, cohort_warnings = _resolve_cohort(
        paths_cfg["mat_dir"], paths_cfg["csv_path"],
        mod_cfg["col_id"], mod_cfg["col_group"],
    )
    n_runs = mod_cfg["n_runs"]
    seeds  = _louvain_seeds(n_runs, config.SEED)
    assert tuple(mod_cfg["groups"]) == config.GROUP_ORDER, (
        f"group order {mod_cfg['groups']} breaks the d = COVID - CONTROL "
        f"convention (config.GROUP_ORDER)")

    print(f"Atlas: {paths_cfg['atlas_label']}")
    print(f"COMET {getattr(comet, '__version__', '1.2.4')}")
    print(f"N subjects: {len(subjects)}  Group: {meta['group'].value_counts().to_dict()}")
    print(f"Signed modularity (qtype='{QTYPE}'), full unthresholded matrix, "
          f"multi-run mean over {n_runs} runs (SeedSequence({config.SEED}).spawn)\n")

    def process_subject(subj):
        C = np.load(os.path.join(paths_cfg["mat_dir"], f"{subj}_connectivity_comet.npy"))
        # Signed, full, unthresholded. Ensure clean diagonal; do NOT alter signs.
        C = C.copy()
        np.fill_diagonal(C, 0.0)
        # Enforce exact symmetry (precision-derived matrices carry ~float32 noise);
        # modularity_louvain_und_sign reads the full matrix and assumes symmetry.
        # Signs unchanged.
        C = 0.5 * (C + C.T)
        qs = []
        for s in seeds:
            try:
                _, Q = bct.modularity_louvain_und_sign(C, qtype=QTYPE, seed=s)
                qs.append(float(Q))
            except Exception as e:
                return {"subject": subj, "group": group_of[subj],
                        "modularity_q": np.nan, "q_sd": np.nan,
                        "q_ci_lo": np.nan, "q_ci_hi": np.nan,
                        "n_runs_ok": 0, "error": str(e)[:120]}
        qs = np.array(qs, float)
        ci_lo, ci_hi = np.percentile(qs, [2.5, 97.5])
        return {
            "subject"     : subj, "group": group_of[subj],
            "modularity_q": float(qs.mean()),       # Q* = mean over runs
            "q_sd"        : float(qs.std(ddof=1)),
            "q_ci_lo"     : float(ci_lo), "q_ci_hi": float(ci_hi),
            "n_runs_ok"   : int(len(qs)),
        }

    t0 = time.time()
    rows = Parallel(n_jobs=mod_cfg["n_jobs"], verbose=10)(
        delayed(process_subject)(s) for s in subjects
    )
    runtime = time.time() - t0
    print(f"\nDone in {runtime/60:.1f} min")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(paths_cfg["out_dir"], "step3c_modularity.csv"), index=False)

    _write_summary(df, mod_cfg, paths_cfg, n_runs, runtime, cohort_warnings)
    _make_plots(df, mod_cfg, paths_cfg["out_dir"], paths_cfg["atlas_label"], n_runs)

    print("\nOutputs:")
    for fn in sorted(os.listdir(paths_cfg["out_dir"])):
        print(f"  {fn}")
    return df


def _write_summary(df, mod_cfg, paths_cfg, n_runs, runtime, cohort_warnings):
    groups = mod_cfg["groups"]
    L = ["=" * 72, f"STEP 3c - SIGNED MODULARITY Q* - {paths_cfg['atlas_label']}", "=" * 72,
         f"BCT call   : modularity_louvain_und_sign(C, qtype='{QTYPE}', seed=...)",
         f"Matrix     : signed, full, unthresholded (raw connectivity values)",
         f"Estimator  : multi-run mean over {n_runs} runs "
         f"(SeedSequence({config.SEED}).spawn)",
         f"N subjects : {len(df)} (via config.select_included_subjects)",
         f"Runtime    : {runtime/60:.1f} min", ""]

    if cohort_warnings:
        L += ["--- Cohort / cache warnings ---"] + [f"  {w}" for w in cohort_warnings] + [""]
    else:
        L += ["Cohort check: matrices on disk match the config-defined sample exactly.", ""]

    n_nan = int(df["modularity_q"].isna().sum())
    if n_nan:
        L.append(f"WARNING: {n_nan} subject(s) with NaN Q* (see 'error' column in CSV)")
        for _, r in df[df["modularity_q"].isna()].iterrows():
            L.append(f"  {r['subject']}: {r.get('error', '')}")
        L.append("")

    ok = df.dropna(subset=["modularity_q"])
    # Stability check: how tight is the multi-run mean per subject?
    L += ["--- Multi-run stability (per-subject SD over runs) ---",
          f"  Q* SD across subjects: mean={ok['q_sd'].mean():.5f}, "
          f"max={ok['q_sd'].max():.5f}",
          f"  (small SD relative to Q* range -> {n_runs} runs give a stable mean)",
          ""]

    L += ["--- Q* by group (DESCRIPTIVE) ---"]
    for g in groups:
        sub = ok[ok["group"] == g]
        L.append(f"  {g:8s} (n={len(sub)}): "
                 f"Q* mean={sub['modularity_q'].mean():+.4f}, "
                 f"SD={sub['modularity_q'].std():.4f}, "
                 f"median={sub['modularity_q'].median():+.4f}")

    ctrl = ok[ok["group"] == groups[0]]["modularity_q"]
    cov  = ok[ok["group"] == groups[1]]["modularity_q"]
    if len(ctrl) > 1 and len(cov) > 1:
        pooled = np.sqrt(((ctrl.var(ddof=1)*(len(ctrl)-1)) + (cov.var(ddof=1)*(len(cov)-1)))
                         / (len(ctrl)+len(cov)-2))
        d = (cov.mean() - ctrl.mean()) / pooled if pooled > 0 else np.nan
        # Welch-t shown as a descriptive preview only (it is the sensitivity test in
        # the confirmatory step). The Family A primary is naive permutation (step3d).
        t, p_t = stats.ttest_ind(cov, ctrl, equal_var=False)
        L += ["",
              f"  Cohen's d ({groups[1]} - {groups[0]}): {d:+.3f}  (DESCRIPTIVE)",
              f"  Welch t (descriptive preview): t={t:+.2f}, p={p_t:.4f}",
              "  (descriptive only; Family A inference = naive permutation + Welch"
              " sensitivity + FDR-BH, no covariates, step3d)"]

    with open(os.path.join(paths_cfg["out_dir"], "step3c_modularity_summary.txt"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))


def _make_plots(df, mod_cfg, out_dir, atlas_label, n_runs):
    groups = mod_cfg["groups"]
    ok = df.dropna(subset=["modularity_q"])
    cov  = ok[ok["group"] == groups[1]]
    ctrl = ok[ok["group"] == groups[0]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: Q* distribution by group
    bins = np.linspace(ok["modularity_q"].min(), ok["modularity_q"].max(), 25)
    axes[0].hist(cov["modularity_q"],  bins=bins, color="firebrick", alpha=0.55,
                 edgecolor="white", density=True, label=f"{groups[1]} (n={len(cov)})")
    axes[0].hist(ctrl["modularity_q"], bins=bins, color="steelblue", alpha=0.55,
                 edgecolor="white", density=True, label=f"{groups[0]} (n={len(ctrl)})")
    axes[0].axvline(cov["modularity_q"].mean(),  color="firebrick", ls="--", lw=1)
    axes[0].axvline(ctrl["modularity_q"].mean(), color="steelblue", ls="--", lw=1)
    axes[0].set_xlabel("Modularity Q* (signed, multi-run mean)")
    axes[0].set_ylabel("Density (normalized)")
    axes[0].set_title("Q* distribution by group")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    # Right: per-subject multi-run SD (stability of the mean)
    axes[1].hist(ok["q_sd"], bins=30, color="gray", edgecolor="white")
    axes[1].set_xlabel(f"Per-subject SD of Q* over {n_runs} runs")
    axes[1].set_ylabel("N subjects")
    axes[1].set_title("Multi-run stability")
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"{atlas_label} — signed Modularity Q* "
                 f"(descriptive; Family A inference in step3d)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "step3c_modularity.png"), dpi=140)
    plt.close()