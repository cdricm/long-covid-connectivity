"""
Step 3c graph-metrics sweep: per subject x density — Global Efficiency
(bct.efficiency_wei), Mean Clustering (bct.clustering_coef_wu, mean),
Assortativity (bct.assortativity_wei, flag=0). Modularity Q* is handled
separately (density-independent, signed full matrix).

In: cached matrices (paths_cfg["mat_dir"]), sign strategy
    (sweep_cfg["strategy"]), cohort via config.select_included_subjects().
Out: step3c_metrics.csv, step3c_aggregated.csv, 2 diagnostic PNGs,
     step3c_summary.txt (or step3c_SKIPPED_<strategy>.txt if the
     negative-subgraph degeneracy guard triggers).

This is the SWEEP only — AUC integration (step3d) is downstream; the
per-density Cohen's d printed/plotted here is descriptive only, not the
inferential test.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import os, glob, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step3a_threshold_sweep_pipeline import _resolve_cohort

METRIC_NAMES = ["global_efficiency", "mean_clustering", "assortativity"]


def _compute_metrics(W, bct):
    """Three weighted, density-dependent metrics on the sign-strategy graph W.
    assortativity_wei is called with flag=0 (undirected)."""
    out = {}
    try:
        # weight inversion produces Inf for zero-weight pairs in fragmented
        # graphs; treated as infinite distance
        with np.errstate(divide="ignore", invalid="ignore"):
            out["global_efficiency"] = float(bct.efficiency_wei(W))
    except Exception as e:
        out["global_efficiency"] = np.nan; out["err_ge"] = str(e)[:120]
    try:
        cc = bct.clustering_coef_wu(W)
        out["mean_clustering"] = float(np.nanmean(cc))
    except Exception as e:
        out["mean_clustering"] = np.nan; out["err_clust"] = str(e)[:120]
    try:
        out["assortativity"] = float(bct.assortativity_wei(W, 0))
    except Exception as e:
        out["assortativity"] = np.nan; out["err_assort"] = str(e)[:120]
    return out


def run_metrics_sweep(paths_cfg, sweep_cfg):
    """Run graph metrics sweep across densities (three AUC metrics)."""
    config.ensure(Path(paths_cfg["out_dir"]))

    import comet
    from comet.graph import bct

    subjects, group_of, meta, cohort_warnings = _resolve_cohort(
        paths_cfg["mat_dir"], paths_cfg["csv_path"],
        sweep_cfg["col_id"], sweep_cfg["col_group"],
    )
    densities = sweep_cfg["densities"]
    strategy  = sweep_cfg["strategy"]
    assert tuple(sweep_cfg["groups"]) == config.GROUP_ORDER, (
        f"group order {sweep_cfg['groups']} breaks the d = COVID - CONTROL "
        f"convention (config.GROUP_ORDER)")

    print(f"Atlas: {paths_cfg['atlas_label']}")
    print(f"COMET {getattr(comet, '__version__', '1.2.4')}")
    print(f"N subjects: {len(subjects)}  Group: {meta['group'].value_counts().to_dict()}")
    print(f"Densities: {len(densities)} -> {len(subjects)*len(densities)} datapoints")
    print(f"Strategy: {strategy}  Weights: raw connectivity values (no normalization)\n")

    def process_subject(subj):
        C = np.load(os.path.join(paths_cfg["mat_dir"], f"{subj}_connectivity_comet.npy"))
        M = config.apply_sign_strategy(C, strategy)
        rows = []
        for d in densities:
            W, _, _ = config.proportional_threshold(M, d)
            t0 = time.time()
            m = _compute_metrics(W, bct)
            rows.append({
                "subject"  : subj, "group": group_of[subj],
                "density"  : d, "n_edges": int(np.sum(W > 0) // 2),
                "t_compute": round(time.time() - t0, 2),
                **m,
            })
        return rows

    # Sanity check on first subject
    print("Sanity check on first subject...")
    t0 = time.time()
    test = process_subject(subjects[0])
    print(f"  Done in {time.time() - t0:.1f}s")
    ge_vals = [r['global_efficiency'] for r in test]
    ge_range = max(ge_vals) - min(ge_vals)
    if ge_range < 0.001:
        msg = (f"Global Efficiency density-invariant on first subject "
                f"({subjects[0]}): range={ge_range:.6f} < 0.001 over densities "
                f"{[int(d * 100) for d in densities]} %. GE values: "
                f"{[round(v, 6) for v in ge_vals]}. "
                f"Strategy '{strategy}' skipped (subgraph degeneracy, group-blind).")
        print(f"  SKIP: {msg}\n")
        with open(os.path.join(paths_cfg["out_dir"],
                                f"step3c_SKIPPED_{strategy}.txt"), "w") as f:
            f.write("=" * 72 + "\n")
            f.write(f"STEP 3c - STRATEGY SKIPPED - {paths_cfg['atlas_label']}\n")
            f.write("=" * 72 + "\n")
            f.write(msg + "\n")
        return None
    print(f"  GE varies by {ge_range:.4f} - OK\n")

    # Full sweep
    print(f"Running full sweep with n_jobs={sweep_cfg['n_jobs']}...")
    t_start = time.time()
    results = Parallel(n_jobs=sweep_cfg["n_jobs"], verbose=10)(
        delayed(process_subject)(s) for s in subjects
    )
    flat = [r for sub in results for r in sub]
    df = pd.DataFrame(flat)
    runtime = time.time() - t_start
    print(f"\nDone in {runtime/60:.1f} min")

    df.to_csv(os.path.join(paths_cfg["out_dir"], "step3c_metrics.csv"), index=False)

    agg = df.groupby(["density", "group"])[METRIC_NAMES].agg(["mean", "std", "median"]).round(4)
    agg.to_csv(os.path.join(paths_cfg["out_dir"], "step3c_aggregated.csv"))

    _make_plots(df, densities, sweep_cfg, paths_cfg["out_dir"], paths_cfg["atlas_label"])
    _write_summary(df, densities, sweep_cfg, paths_cfg, subjects, group_of, runtime,
                   cohort_warnings)

    print("\nOutputs:")
    for fn in sorted(os.listdir(paths_cfg["out_dir"])):
        print(f"  {fn}")
    return df


def _make_plots(df, densities, sweep_cfg, out_dir, atlas_label):
    groups = sweep_cfg["groups"]

    # Metric curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, metric in zip(axes.flat, METRIC_NAMES):
        for g in groups:
            sub = df[df["group"] == g]
            m   = sub.groupby("density")[metric].agg(["mean", "std"]).reset_index()
            x   = m["density"] * 100
            ax.plot(x, m["mean"], marker="o", label=g)
            ax.fill_between(x, m["mean"] - m["std"], m["mean"] + m["std"], alpha=0.15)
        # Shade the confirmatory AUC range 10-25 %
        ax.axvspan(10, 25, color="grey", alpha=0.10, label="confirmatory 10-25%")
        ax.set_xlabel("Density (%)"); ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(metric.replace("_", " "))
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"{atlas_label} — Graph metrics across densities (mean ± SD)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "step3c_metric_curves.png"), dpi=140); plt.close()

    # Effect sizes per density (descriptive; the inferential AUC test is step3d)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, metric in zip(axes.flat, METRIC_NAMES):
        diff = []
        for d in densities:
            ctrl  = df[(df["group"] == groups[0]) & (df["density"] == d)][metric].dropna()
            covid = df[(df["group"] == groups[1]) & (df["density"] == d)][metric].dropna()
            if len(ctrl) > 1 and len(covid) > 1:
                pooled = np.sqrt(((ctrl.var()*(len(ctrl)-1)) + (covid.var()*(len(covid)-1))) /
                                 (len(ctrl) + len(covid) - 2))
                d_eff = (covid.mean() - ctrl.mean()) / pooled if pooled > 0 else np.nan
            else:
                d_eff = np.nan
            diff.append({"density": d, "cohen_d": d_eff})
        ddf = pd.DataFrame(diff)
        ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
        ax.axhline(0.2, color="orange", linestyle=":", linewidth=0.6)
        ax.axhline(-0.2, color="orange", linestyle=":", linewidth=0.6)
        ax.axvspan(10, 25, color="grey", alpha=0.10)
        ax.plot(ddf["density"] * 100, ddf["cohen_d"], marker="o", color="darkred")
        ax.set_xlabel("Density (%)"); ax.set_ylabel(f"Cohen's d ({groups[1]} − {groups[0]})")
        ax.set_title(metric.replace("_", " ")); ax.grid(alpha=0.3)
    fig.suptitle(f"{atlas_label} — Per-density group effect size (descriptive; "
                 f"AUC inference in step3d)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "step3c_effect_sizes.png"), dpi=140); plt.close()


def _write_summary(df, densities, sweep_cfg, paths_cfg, subjects, group_of, runtime,
                   cohort_warnings):
    groups = sweep_cfg["groups"]
    L = ["=" * 72, f"STEP 3c - METRICS SWEEP - {paths_cfg['atlas_label']}", "=" * 72,
         f"Strategy   : {sweep_cfg['strategy']}",
         f"Weights    : raw connectivity values of kept edges, {sweep_cfg['strategy']} strategy (NO normalization)",
         f"Densities  : {[int(d*100) for d in densities]} (%)  (sweep; 5-100% for visualization)",
         f"Metrics    : {METRIC_NAMES}  (3 density-dependent AUC metrics; Modularity Q* separate)",
         f"N subjects : {len(subjects)} "
         f"({sum(1 for s in subjects if group_of[s]==groups[0])} {groups[0]}, "
         f"{sum(1 for s in subjects if group_of[s]==groups[1])} {groups[1]})  "
         f"(via config.select_included_subjects)",
         f"Runtime    : {runtime/60:.1f} min",
         ""]

    if cohort_warnings:
        L += ["--- Cohort / cache warnings ---"] + [f"  {w}" for w in cohort_warnings] + [""]
    else:
        L += ["Cohort check: matrices on disk match the config-defined sample exactly.", ""]

    L += ["NOTE: this is the SWEEP only. AUC integration (np.trapz, x-arg) over the",
          "      confirmatory 10-25% range and 5-50% sensitivity range is step3d.",
          "      Per-density Cohen's d below is DESCRIPTIVE, not the inferential test.",
          ""]

    for m in METRIC_NAMES:
        n_nan = int(df[m].isna().sum())
        if n_nan > 0:
            L.append(f"WARNING: {n_nan} NaN values in {m}")

    L.append("\n--- Metric ranges (all subjects/densities) ---")
    for m in METRIC_NAMES:
        v = df[m].dropna()
        L.append(f"  {m:<22}: [{v.min():+.4f}, {v.max():+.4f}], "
                 f"mean={v.mean():+.4f}, SD={v.std():.4f}")

    L.append("\n--- Per-density effect size (Cohen's d, DESCRIPTIVE) ---")
    for m in METRIC_NAMES:
        L.append(f"\n  {m}:")
        for d in densities:
            ctrl  = df[(df["group"] == groups[0]) & (df["density"] == d)][m].dropna()
            covid = df[(df["group"] == groups[1]) & (df["density"] == d)][m].dropna()
            if len(ctrl) > 1 and len(covid) > 1:
                pooled = np.sqrt(((ctrl.var()*(len(ctrl)-1)) + (covid.var()*(len(covid)-1))) /
                                 (len(ctrl) + len(covid) - 2))
                d_eff = (covid.mean() - ctrl.mean()) / pooled if pooled > 0 else np.nan
                in_range = "*" if 0.10 <= d <= 0.25 else " "
                marker = " (|d|>=0.2)" if abs(d_eff) >= 0.2 else ""
                L.append(f"   {in_range}{int(d*100):>3}%: d={d_eff:+.3f}{marker}  "
                         f"({groups[0]} μ={ctrl.mean():+.4f}, {groups[1]} μ={covid.mean():+.4f})")
    L.append("\n  (* = within confirmatory 10-25% AUC range)")

    with open(os.path.join(paths_cfg["out_dir"], "step3c_summary.txt"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))