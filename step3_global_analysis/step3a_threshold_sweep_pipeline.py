"""
Threshold-sweep diagnostic (Step 3a): per subject x sign-strategy x density —
target/kept edge counts, connectedness, component structure, mean edge weight.
Broad grid (1-100%), group-blind (no group inference).

In: cached matrices (paths_cfg["mat_dir"]), cohort via
    config.select_included_subjects() (cross-checked against matrices on disk).
Out: step3a_sweep.csv, step3a_sweep_summary.csv, 4 diagnostic PNGs,
     step3a_summary.txt.

Edge weights are raw connectivity values (no Fisher-z) — this step performs no
group inference or aggregated-FC estimation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import os, glob, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from joblib import Parallel, delayed


# ===== Graph helpers ==========================================================
# Sign strategy and proportional thresholding are centralised in config
# (config.apply_sign_strategy / config.proportional_threshold) so every step
# shares one implementation. Only the connectedness diagnostic below is local.
def graph_diagnostics(W):
    adj = (W > 0).astype(np.int8)
    sp  = csr_matrix(adj)
    n_comp, labels = connected_components(sp, directed=False)
    sizes = np.bincount(labels)
    return {
        "n_components"     : int(n_comp),
        "largest_component": int(sizes.max()),
        "connected"        : bool(n_comp == 1),
        "mean_edge_weight" : float(W[W > 0].mean()) if (W > 0).any() else 0.0,
    }


# ===== Cohort resolution ======================================================
def _resolve_cohort(mat_dir, csv_path, col_id, col_group):
    """Resolve the analytical cohort via config (single source of truth) and
    cross-check against matrices on disk. Returns (subjects, group_of, meta_df, warnings).
    Warns on any discrepancy between the config-defined sample and the matrices
    actually present — this guards against stale / mixed-cohort caches.
    """
    group_df = pd.read_csv(csv_path)

    # Subjects with a matrix on disk
    on_disk = {os.path.basename(p).replace("_connectivity_comet.npy", "")
               for p in glob.glob(os.path.join(mat_dir, "CP*_connectivity_comet.npy"))}

    # Config-defined analytical sample (gates a+b+c)
    included = set(config.select_included_subjects(sorted(on_disk), group_df, verbose=False))

    # Cross-check
    missing_matrix = sorted(included - on_disk)  # config says include, no matrix found
    not_in_cohort = sorted(on_disk - included)  # matrix present, not in cohort (stale?)

    warnings = []
    if missing_matrix:
        w = (f"WARNING: {len(missing_matrix)} cohort subject(s) have NO matrix on disk: "
             f"{missing_matrix}")
        warnings.append(w)
        print(f"  {w}")
    if not_in_cohort:
        w = (f"WARNING: {len(not_in_cohort)} matrix file(s) NOT in the config cohort "
             f"(stale/excluded?): {not_in_cohort} — EXCLUDED from the sweep "
             f"(config cohort is authoritative)")
        warnings.append(w)
        print(f"  {w}")

    subjects = sorted(included & on_disk)

    grp = group_df.copy()
    grp[col_id] = grp[col_id].astype(str).str.strip()
    grp[col_group] = grp[col_group].astype(str).str.strip().str.upper()
    group_of = {s: grp.loc[grp[col_id] == s, col_group].iloc[0] for s in subjects}

    meta = pd.DataFrame({"subject": subjects,
                         "group": [group_of[s] for s in subjects]})
    return subjects, group_of, meta, warnings


# ===== Main entry point =======================================================
def run_threshold_sweep(paths_cfg, sweep_cfg):
    """Run threshold sweep across strategies and densities."""
    config.ensure(Path(paths_cfg["out_dir"]))

    subjects, group_of, meta, cohort_warnings = _resolve_cohort(
        paths_cfg["mat_dir"], paths_cfg["csv_path"],
        sweep_cfg["col_id"], sweep_cfg["col_group"],
    )

    print(f"Atlas: {paths_cfg['atlas_label']}")
    print(f"N subjects: {len(subjects)}  Group: {meta['group'].value_counts().to_dict()}\n")

    def sweep_subject(subj):
        C = np.load(os.path.join(paths_cfg["mat_dir"], f"{subj}_connectivity_comet.npy"))
        rows = []
        for strategy in sweep_cfg["strategies"]:
            M = config.apply_sign_strategy(C, strategy)
            for density in sweep_cfg["densities"]:
                W, n_kept, n_target = config.proportional_threshold(M, density)
                diag = graph_diagnostics(W)
                rows.append({
                    "subject"        : subj,
                    "group"          : group_of[subj],
                    "strategy"       : strategy,
                    "density"        : density,
                    "n_edges_target" : n_target,
                    "n_edges_kept"   : n_kept,
                    "target_reached" : (n_kept == n_target),
                    **diag,
                })
        return rows

    t_start = time.time()
    print(f"Running sweep: {len(subjects)} subj x {len(sweep_cfg['strategies'])} strat "
          f"x {len(sweep_cfg['densities'])} dens = "
          f"{len(subjects)*len(sweep_cfg['strategies'])*len(sweep_cfg['densities'])} datapoints")
    results = Parallel(n_jobs=sweep_cfg["n_jobs"], verbose=10)(
        delayed(sweep_subject)(s) for s in subjects
    )
    flat = [r for sub in results for r in sub]
    df = pd.DataFrame(flat)
    runtime = time.time() - t_start
    print(f"Sweep done in {runtime:.1f}s\n")

    df.to_csv(os.path.join(paths_cfg["out_dir"], "step3a_sweep.csv"), index=False)

    agg = df.groupby(["strategy", "density"]).agg(
        n_subjects       = ("subject", "count"),
        n_connected      = ("connected", "sum"),
        pct_connected    = ("connected", lambda x: 100 * x.mean()),
        median_n_comp    = ("n_components", "median"),
        max_n_comp       = ("n_components", "max"),
        median_largest   = ("largest_component", "median"),
        mean_edges_kept  = ("n_edges_kept", "mean"),
        pct_target_reach = ("target_reached", lambda x: 100 * x.mean()),
    ).round(2)
    agg.to_csv(os.path.join(paths_cfg["out_dir"], "step3a_sweep_summary.csv"))

    agg_grp = df.groupby(["strategy", "density", "group"]).agg(
        n             = ("subject", "count"),
        pct_connected = ("connected", lambda x: 100 * x.mean()),
        median_n_comp = ("n_components", "median"),
        mean_weight   = ("mean_edge_weight", "mean"),
    ).reset_index()

    _make_plots(df, agg_grp, sweep_cfg, paths_cfg["out_dir"], paths_cfg["atlas_label"])
    _write_summary(agg, sweep_cfg, paths_cfg, subjects, meta, runtime, cohort_warnings)

    print("\nOutputs:")
    for fn in sorted(os.listdir(paths_cfg["out_dir"])):
        print(f"  {fn}")
    return df


def _make_plots(df, agg_grp, sweep_cfg, out_dir, atlas_label):
    strategies = sweep_cfg["strategies"]
    densities  = sweep_cfg["densities"]
    groups     = sweep_cfg["groups"]

    # Connectedness
    fig, axes = plt.subplots(1, len(strategies), figsize=(5*len(strategies), 4), sharey=True)
    if len(strategies) == 1: axes = [axes]
    for ax, strat in zip(axes, strategies):
        for g in groups:
            sub = agg_grp[(agg_grp["strategy"] == strat) & (agg_grp["group"] == g)]
            ax.plot(sub["density"] * 100, sub["pct_connected"], marker="o", label=g)
        ax.set_title(strat); ax.set_xlabel("Density (%)")
        ax.axhline(100, color="grey", linestyle=":", linewidth=0.8)
        ax.axhline(95,  color="red",  linestyle=":", linewidth=0.8)
        ax.set_ylim(-5, 105); ax.grid(alpha=0.3)
    axes[0].set_ylabel("% subjects with connected graph")
    axes[-1].legend()
    fig.suptitle(f"{atlas_label} — Connectedness across thresholds (diagnostic)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "step3a_connectedness.png"), dpi=140); plt.close()

    # Components boxplot
    fig, axes = plt.subplots(1, len(strategies), figsize=(5*len(strategies), 4), sharey=True)
    if len(strategies) == 1: axes = [axes]
    for ax, strat in zip(axes, strategies):
        data = [df[(df["strategy"] == strat) & (df["density"] == d)]["n_components"].values
                for d in densities]
        ax.boxplot(data, labels=[f"{int(d*100)}" for d in densities])
        ax.set_title(strat); ax.set_xlabel("Density (%)")
        ax.set_yscale("log"); ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("# components (log)")
    fig.suptitle(f"{atlas_label} — Number of components across thresholds", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "step3a_components.png"), dpi=140); plt.close()

    # Edge weights (raw connectivity values)
    fig, axes = plt.subplots(1, len(strategies), figsize=(5*len(strategies), 4), sharey=False)
    if len(strategies) == 1: axes = [axes]
    for ax, strat in zip(axes, strategies):
        for g in groups:
            sub = agg_grp[(agg_grp["strategy"] == strat) & (agg_grp["group"] == g)]
            ax.plot(sub["density"] * 100, sub["mean_weight"], marker="o", label=g)
        ax.set_title(strat); ax.set_xlabel("Density (%)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean kept-edge weight (raw, group avg)")
    axes[-1].legend()
    fig.suptitle(f"{atlas_label} — Mean kept-edge weight across thresholds", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "step3a_edgeweights.png"), dpi=140); plt.close()

    # Target reached
    fig, axes = plt.subplots(1, len(strategies), figsize=(5*len(strategies), 4))
    if len(strategies) == 1: axes = [axes]
    for ax, strat in zip(axes, strategies):
        sub = df[df["strategy"] == strat].pivot(index="subject", columns="density",
                                                values="target_reached").astype(float)
        ax.imshow(sub.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(densities)))
        ax.set_xticklabels([f"{int(d*100)}%" for d in densities], rotation=90, fontsize=7)
        ax.set_yticks([]); ax.set_xlabel("Density")
        ax.set_title(f"{strat}\n(green=reached, red=insufficient)")
    axes[0].set_ylabel("Subjects")
    fig.suptitle(f"{atlas_label} — Was the requested edge count achievable?", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "step3a_target_reached.png"), dpi=140); plt.close()


def _write_summary(agg, sweep_cfg, paths_cfg, subjects, meta, runtime, cohort_warnings):
    L = ["=" * 72, f"STEP 3a - THRESHOLD SWEEP (DIAGNOSTIC) - {paths_cfg['atlas_label']}", "=" * 72,
         f"N subjects : {len(subjects)}  (resolved via config.select_included_subjects)",
         f"Groups     : {dict(meta['group'].value_counts())}",
         f"Densities  : {[int(d*100) for d in sweep_cfg['densities']]} (%)",
         f"Strategies : {sweep_cfg['strategies']}  (group-blind diagnostic; confirmatory choice set afterwards)",
         f"Runtime    : {runtime:.1f}s",
         ""]

    if cohort_warnings:
        L += ["--- Cohort / cache warnings ---"] + [f"  {w}" for w in cohort_warnings] + [""]
    else:
        L += ["Cohort check: matrices on disk match the config-defined sample exactly.", ""]

    L += ["NOTE: broad diagnostic grid. Confirmatory AUC range (10-25 %) is",
          "      defined downstream (steps 3b/3c), not here.",
          "",
          "--- Connectedness across strategies/densities ---"]

    for strat in sweep_cfg["strategies"]:
        L.append(f"\n  Strategy: {strat}")
        for d in sweep_cfg["densities"]:
            row = agg.loc[(strat, d)]
            L.append(f"    {int(d*100):>3}% density : "
                     f"{int(row['n_connected']):>3}/{int(row['n_subjects'])} connected "
                     f"({row['pct_connected']:>5.1f}%), "
                     f"median #comp={row['median_n_comp']:.0f}, "
                     f"target-reach={row['pct_target_reach']:.0f}%")

    L += ["", "--- Key observations ---"]
    for strat in sweep_cfg["strategies"]:
        found = None
        for d in sweep_cfg["densities"]:
            if agg.loc[(strat, d), "pct_connected"] >= 95.0:
                found = d; break
        if found is not None:
            L.append(f"  {strat:<10}: >=95% subjects connected starting at {int(found*100)}% density")
        else:
            L.append(f"  {strat:<10}: never reaches 95% connectedness in tested range")
    for strat in sweep_cfg["strategies"]:
        found = None
        for d in sweep_cfg["densities"][::-1]:
            if agg.loc[(strat, d), "pct_target_reach"] >= 95.0:
                found = d; break
        if found is not None:
            L.append(f"  {strat:<10}: target reached for >=95% subjects up to {int(found*100)}% density")

    with open(os.path.join(paths_cfg["out_dir"], "step3a_summary.txt"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))