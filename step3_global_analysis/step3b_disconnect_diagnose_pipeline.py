"""
Step 3b disconnect diagnostic — follows up on step3a: which ROIs are
systematically isolated (Q1), which subjects are chronically disconnected (Q2),
and whether the fragmentation pattern is systematic or diffuse (Q3).

In: cached matrices (paths_cfg["mat_dir"]), cohort via
    config.select_included_subjects() (shared with step3a, cross-checked
    against matrices on disk).
Out: step3b_subject_disconnect.csv, step3b_roi_isolation.csv (ROI identity by
     index only, label mapping deferred to step4), 3 diagnostic PNGs,
     step3b_summary.txt.

Edge weights are raw connectivity values (no Fisher-z) — this step computes
only graph topology, no group inference.
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

# Sign strategy + thresholding come from config (single source of truth).
# Cohort resolution is shared with step3a (imported from the sibling module).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step3a_threshold_sweep_pipeline import _resolve_cohort


def run_disconnect_diagnose(paths_cfg, diagnose_cfg):
    """Diagnose disconnectedness patterns."""
    config.ensure(Path(paths_cfg["out_dir"]))

    subjects, group_of, meta, cohort_warnings = _resolve_cohort(
        paths_cfg["mat_dir"], paths_cfg["csv_path"],
        diagnose_cfg["col_id"], diagnose_cfg["col_group"],
    )
    densities = diagnose_cfg["densities"]
    strategy  = diagnose_cfg["strategy"]

    print(f"Atlas: {paths_cfg['atlas_label']}")
    print(f"N subjects: {len(subjects)}  Group: {meta['group'].value_counts().to_dict()}")
    print(f"Diagnosing strategy={strategy}, "
          f"densities={[int(d*100) for d in densities]}\n")

    def diagnose_subject(subj):
        C = np.load(os.path.join(paths_cfg["mat_dir"], f"{subj}_connectivity_comet.npy"))
        M = config.apply_sign_strategy(C, strategy)
        per_d = {}
        for d in densities:
            W, _, _ = config.proportional_threshold(M, d)
            deg = (W > 0).sum(axis=1)
            adj = csr_matrix((W > 0).astype(np.int8))
            n_comp, labels = connected_components(adj, directed=False)
            sizes = np.bincount(labels)
            largest_label = int(np.argmax(sizes))
            in_largest    = (labels == largest_label)
            singletons    = (deg == 0)
            not_in_main   = ~in_largest
            per_d[d] = {
                "n_components"  : int(n_comp),
                "largest_size"  : int(sizes.max()),
                "n_singletons"  : int(singletons.sum()),
                "n_not_in_main" : int(not_in_main.sum()),
                "isolated_rois" : np.where(not_in_main)[0].tolist(),
                "singleton_rois": np.where(singletons)[0].tolist(),
            }
        return subj, per_d

    t0 = time.time()
    results = Parallel(n_jobs=diagnose_cfg["n_jobs"], verbose=10)(
        delayed(diagnose_subject)(s) for s in subjects
    )
    print(f"Diagnosed in {time.time()-t0:.1f}s\n")

    n_roi = np.load(os.path.join(paths_cfg["mat_dir"],
                                 f"{subjects[0]}_connectivity_comet.npy")).shape[0]
    roi_isolation = {d: np.zeros(n_roi, dtype=int) for d in densities}
    roi_singleton = {d: np.zeros(n_roi, dtype=int) for d in densities}

    for subj, per_d in results:
        for d in densities:
            for r in per_d[d]["isolated_rois"]:  roi_isolation[d][r] += 1
            for r in per_d[d]["singleton_rois"]: roi_singleton[d][r] += 1

    ref_d = diagnose_cfg["reference_density"]
    top_isolated = np.argsort(-roi_isolation[ref_d])[:20]

    # Subject disconnect severity
    # Report at the densities closest to 20 % and 30 %.
    ref_low = min(densities, key=lambda x: abs(x - 0.20))
    ref_high = min(densities, key=lambda x: abs(x - 0.30))

    subj_disconnect = []
    for subj, per_d in results:
        n_disc = sum(1 for d in densities if per_d[d]["n_components"] > 1)
        max_islands = max(per_d[d]["n_not_in_main"] for d in densities)
        subj_disconnect.append({
            "subject"             : subj,
            "group"               : group_of[subj],
            "n_densities_disconn" : n_disc,
            "max_islands"         : max_islands,
            f"n_comp_at_{int(ref_low*100)}pct" : per_d[ref_low]["n_components"],
            f"n_comp_at_{int(ref_high*100)}pct": per_d[ref_high]["n_components"],
            f"isolated_at_{int(ref_low*100)}pct": per_d[ref_low]["n_not_in_main"],
        })
    subj_df = pd.DataFrame(subj_disconnect).sort_values("n_densities_disconn", ascending=False)
    subj_df.to_csv(os.path.join(paths_cfg["out_dir"], "step3b_subject_disconnect.csv"), index=False)

    # ROI summary
    roi_summary = pd.DataFrame({"roi": np.arange(n_roi)})
    for d in densities:
        roi_summary[f"isolated_pct_{int(d*100)}"]  = roi_isolation[d] / len(subjects) * 100
        roi_summary[f"singleton_pct_{int(d*100)}"] = roi_singleton[d] / len(subjects) * 100
    roi_summary.to_csv(os.path.join(paths_cfg["out_dir"], "step3b_roi_isolation.csv"), index=False)

    _make_plots(roi_isolation, subj_df, densities, ref_d, n_roi,
                paths_cfg["out_dir"], paths_cfg["atlas_label"], len(subjects))
    _write_summary(roi_isolation, top_isolated, subj_df, densities, strategy,
                    ref_d, len(subjects), paths_cfg, cohort_warnings)

    print("\nOutputs:")
    for fn in sorted(os.listdir(paths_cfg["out_dir"])):
        print(f"  {fn}")
    return results


def _make_plots(roi_isolation, subj_df, densities, ref_d, n_roi, out_dir, atlas_label, n_subj):
    # ROI isolation curves
    fig, ax = plt.subplots(figsize=(10, 5))
    for d in densities:
        sorted_pct = np.sort(roi_isolation[d] / n_subj * 100)[::-1]
        ax.plot(sorted_pct, label=f"{int(d*100)}%")
    ax.set_xlabel("ROI rank (sorted by isolation frequency)")
    ax.set_ylabel("% subjects in which ROI is outside main component")
    ax.set_title(f"{atlas_label} — ROI isolation frequency per density")
    ax.legend(title="Density"); ax.grid(alpha=0.3)
    ax.axhline(50, color="red", linestyle=":", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "step3b_roi_isolation_curves.png"), dpi=140); plt.close()

    # Subject severity
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(subj_df["n_densities_disconn"], bins=range(0, len(densities)+2),
                 color="steelblue", edgecolor="white", align="left")
    axes[0].set_xlabel(f"# densities (out of {len(densities)}) disconnected")
    axes[0].set_ylabel("# subjects")
    axes[0].set_title("Per-subject disconnect severity")
    axes[0].grid(alpha=0.3)
    iso_col = [c for c in subj_df.columns if c.startswith("isolated_at_")][0]
    axes[1].hist(subj_df[iso_col], bins=30, color="steelblue", edgecolor="white")
    axes[1].set_xlabel(f"# ROIs outside main component @ {iso_col.split('_')[-1]}")
    axes[1].set_ylabel("# subjects")
    axes[1].set_title("Per-subject island size")
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"{atlas_label}", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "step3b_subject_severity.png"), dpi=140); plt.close()

    # ROI heatmap (top 30)
    n_show = min(30, n_roi)
    top = np.argsort(-roi_isolation[ref_d])[:n_show]
    heat = np.array([roi_isolation[d][top] / n_subj * 100 for d in densities])
    fig, ax = plt.subplots(figsize=(min(12, n_show*0.4), 5))
    im = ax.imshow(heat, aspect="auto", cmap="Reds", vmin=0, vmax=100)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([f"ROI {r}" for r in top], rotation=90, fontsize=8)
    ax.set_yticks(range(len(densities)))
    ax.set_yticklabels([f"{int(d*100)}%" for d in densities])
    ax.set_xlabel("ROI"); ax.set_ylabel("Density")
    ax.set_title(f"{atlas_label} — Top {n_show} most-isolated ROIs")
    plt.colorbar(im, ax=ax, label="% subjects")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "step3b_roi_heatmap.png"), dpi=140); plt.close()


def _write_summary(roi_isolation, top_isolated, subj_df, densities, strategy,
                   ref_d, n_subj, paths_cfg, cohort_warnings):
    L = ["=" * 72, f"STEP 3b - DISCONNECTEDNESS DIAGNOSE - {paths_cfg['atlas_label']}",
         "=" * 72,
         f"Strategy   : {strategy}  (sign strategy for this diagnostic run)",
         f"Densities  : {[int(d*100) for d in densities]} (%)  (diagnostic; spans the 10-25% range)",
         f"N subjects : {n_subj}  (resolved via config.select_included_subjects)",
         ""]

    if cohort_warnings:
        L += ["--- Cohort / cache warnings ---"] + [f"  {w}" for w in cohort_warnings] + [""]
    else:
        L += ["Cohort check: matrices on disk match the config-defined sample exactly.", ""]

    L += [f"--- Q1: Most-isolated ROIs (reference: {int(ref_d * 100)}%) ---"]
    for r in top_isolated:
        pct = roi_isolation[ref_d][r] / n_subj * 100
        if pct > 5:
            L.append(f"  ROI {r:>3}: {pct:>5.1f}% of subjects")
    L.append("")
    n_chronic = int((roi_isolation[ref_d] / n_subj > 0.5).sum())
    L.append(f"ROIs isolated in >50% of subjects at {int(ref_d*100)}%: {n_chronic}")
    L.append("")

    L.append("--- Q2: Subject disconnect severity ---")
    all_disc = int((subj_df["n_densities_disconn"] == len(densities)).sum())
    none_disc = int((subj_df["n_densities_disconn"] == 0).sum())
    L.append(f"Disconnected at ALL {len(densities)} densities: {all_disc}")
    L.append(f"Never disconnected                  : {none_disc}")
    L.append("\nTop 10 worst subjects:")
    for _, r in subj_df.head(10).iterrows():
        L.append(f"  {r['subject']} ({r['group']}): "
                 f"disc at {r['n_densities_disconn']}/{len(densities)} densities, "
                 f"max islands={r['max_islands']}")
    L.append("")

    L.append("--- Q3: Pattern ---")
    top10_share = (roi_isolation[ref_d][top_isolated[:10]] / n_subj * 100)
    if top10_share.mean() > 50:
        L.append("  -> SYSTEMATIC: top 10 ROIs isolated in >50% of subjects on average")
    elif top10_share.mean() > 20:
        L.append("  -> PARTIAL: some ROIs commonly isolated but not majority")
    else:
        L.append("  -> DIFFUSE: isolation spread across many ROIs")
    L.append(f"  Top-10 ROIs avg isolation pct at {int(ref_d*100)}%: {top10_share.mean():.1f}%")

    with open(os.path.join(paths_cfg["out_dir"], "step3b_summary.txt"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))