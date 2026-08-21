"""
Edge-sign composition of the FC matrices (group-blind QC / graph-construction
figure): fraction of positive vs. negative off-diagonal edges pooled over the
frozen analytical sample (N=162), plus mean edge strength under three
aggregations (positive-only, |negative|, absolute).

Pearson arm: reports the negative-subgraph share across all three atlases,
supporting the positive-only construction choice. Partial arm (Schaefer-400
only): reports the edge-sign split under partial correlation, feeding the
group-blind 3a/3b/3c diagnostic that sets CONFIRMATORY_SIGN_STRATEGY — no
positive-only claim is made here.

In: cached matrices under analysis_outputs/<FC_METHOD>/<atlas>/step2_pipeline/
    comet_matrices, subjects via config.select_included_subjects().
Out: config.CROSS_DIRS["step2_fc_diagnostics"]/graph_construction/
     edge_composition_<arm>.csv/.png.

Group-blind by construction: edges are pooled over all included subjects
irrespective of group; the loaded count is hard-asserted to equal N=162.
"""

import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================
# SETTINGS
# =============================================================
# Atlases per arm: Pearson runs all three (atlas-robustness); partial runs the
# confirmatory atlas only (estimator-robustness, atlas-robustness already
# established in the Pearson arm). Driven by config.FC_METHOD — no manual edit.
ATLASES_BY_ARM = {
    "pearson": ["schaefer400", "schaefer100", "aal"],
    "partial": ["schaefer400"],
}
ATLAS_LABELS = {
    "schaefer400": "Schaefer-400",
    "schaefer100": "Schaefer-100",
    "aal":         "AAL",
}
MATRIX_SUBDIR = "comet_matrices"          # under {atlas}/step2_pipeline/
STEP2_SUBDIR  = "step2_pipeline"
FILENAME_ID   = re.compile(r"(CP\d+)")    # CP####_connectivity_comet.npy -> CP####

EXPECTED_N = 162                          # frozen analytical sample

OUT_DIR = config.ensure(
    config.CROSS_DIRS["step2_fc_diagnostics"] / "graph_construction"
)

COL_POS = "#4575b4"
COL_NEG = "#d73027"
COL_ABS = "#6a51a3"


# =============================================================
# COHORT-BOUND LOADING
# =============================================================
def included_ids():
    """Analytical sample (gates a+b+c) as a sorted list of subject IDs."""
    nii_subjects = sorted(p.name for p in config.NII_ROOT.iterdir() if p.is_dir())
    group_df = pd.read_csv(config.GROUP_CSV)
    return config.select_included_subjects(nii_subjects, group_df, verbose=False)


def matrix_files_for_atlas(atlas: str):
    """Map subject ID -> matrix path for the given atlas (unfiltered)."""
    fc_dir = config.atlas_dir(atlas, STEP2_SUBDIR) / MATRIX_SUBDIR
    if not fc_dir.is_dir():
        raise FileNotFoundError(f"Matrix directory not found: {fc_dir}")
    id_to_path = {}
    for f in fc_dir.glob("*.npy"):
        m = FILENAME_ID.search(f.name)
        if m:
            id_to_path[m.group(1)] = f
    return id_to_path, fc_dir


def load_upper_triangles(atlas: str, included: list[str]):
    """Concatenate upper-triangle edge weights over the INCLUDED subjects only.

    Asserts that every included subject has a matrix and that exactly EXPECTED_N
    matrices are loaded.
    """
    id_to_path, fc_dir = matrix_files_for_atlas(atlas)

    missing = [s for s in included if s not in id_to_path]
    if missing:
        raise AssertionError(
            f"[{atlas}] {len(missing)} included subjects have no matrix in {fc_dir}: "
            f"{missing}"
        )

    ids = sorted(included)
    values = []
    for s in tqdm(ids, desc=f"  loading {atlas}", leave=False):
        mat = np.load(id_to_path[s])
        n = mat.shape[0]
        iu = np.triu_indices(n, k=1)
        values.append(mat[iu])

    n_loaded = len(ids)
    assert n_loaded == EXPECTED_N, (
        f"[{atlas}] loaded {n_loaded} matrices, expected {EXPECTED_N}. "
        f"Cohort binding violated — aborting."
    )
    return np.concatenate(values), n_loaded


# =============================================================
# STATISTICS
# =============================================================
def edge_sign_stats(v: np.ndarray, n_subj: int) -> dict:
    n_total = v.size
    pos = v > 0
    neg = v < 0
    n_pos, n_neg, n_zero = int(pos.sum()), int(neg.sum()), int((v == 0).sum())
    return {
        "n_edges_per_subject": n_total // n_subj,
        "n_edges_total": n_total,
        "frac_pos": n_pos / n_total,
        "frac_neg": n_neg / n_total,
        "frac_zero": n_zero / n_total,
        "mean_pos": float(v[pos].mean()) if n_pos else 0.0,
        "abs_neg": float(abs(v[neg].mean())) if n_neg else 0.0,
        "mean_abs": float(np.abs(v).mean()),
    }


# =============================================================
# FIGURE
# =============================================================
def make_figure(stats: dict, arm: str, atlases: list[str], out_path: Path):
    labels = [ATLAS_LABELS[a] for a in atlases]
    x = np.arange(len(atlases))
    frac_pos = [stats[a]["frac_pos"] * 100 for a in atlases]
    frac_neg = [stats[a]["frac_neg"] * 100 for a in atlases]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # -------- LEFT: stacked composition --------
    ax = axes[0]
    ax.bar(x, frac_pos, color=COL_POS, label="positive edges",
           edgecolor="white", linewidth=1.5)
    ax.bar(x, frac_neg, bottom=frac_pos, color=COL_NEG, label="negative edges",
           edgecolor="white", linewidth=1.5)
    for i in range(len(atlases)):
        ax.text(i, frac_pos[i] / 2, f"{frac_pos[i]:.1f}%",
                ha="center", va="center", color="white", fontweight="bold", fontsize=11)
        ax.text(i, frac_pos[i] + frac_neg[i] / 2, f"{frac_neg[i]:.1f}%",
                ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of edges (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Edge composition")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 0.98), framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Single-atlas (partial arm): make the negative share prominent — it is the
    # single quantity that the sign-strategy diagnostic turns on.
    if len(atlases) == 1:
        ax.annotate(
            f"negative edges: {frac_neg[0]:.1f}%",
            xy=(0, frac_pos[0] + frac_neg[0]), xytext=(0, frac_pos[0] + frac_neg[0] + 6),
            ha="center", fontsize=11, fontweight="bold", color=COL_NEG,
        )
        ax.set_ylim(0, 112)

    # -------- RIGHT: mean strength by aggregation --------
    ax = axes[1]
    mean_pos = [stats[a]["mean_pos"] for a in atlases]
    abs_neg = [stats[a]["abs_neg"] for a in atlases]
    mean_abs = [stats[a]["mean_abs"] for a in atlases]
    bw = 0.27
    b1 = ax.bar(x - bw, mean_pos, bw, color=COL_POS, label="positive-only mean", edgecolor="white")
    b2 = ax.bar(x,      abs_neg,  bw, color=COL_NEG, label="negative |mean|",     edgecolor="white")
    b3 = ax.bar(x + bw, mean_abs, bw, color=COL_ABS, label="absolute-value mean", edgecolor="white")
    for bars in (b1, b2, b3):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.005,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean |Pearson r|" if arm == "pearson"
                  else "Mean |partial correlation|")
    ax.set_title("Mean edge strength by aggregation")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if arm == "pearson":
        suptitle = "Positive-only rationale: empirical edge composition (N=162, group-blind)"
    else:
        suptitle = ("Edge-sign composition under partial correlation "
                    "(N=162, group-blind; GSR rationale does not transfer)")
    fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================
# MAIN
# =============================================================
def main():
    arm = config.FC_METHOD
    if arm not in ATLASES_BY_ARM:
        raise ValueError(f"Unsupported FC_METHOD for this diagnostic: {arm!r}")
    atlases = ATLASES_BY_ARM[arm]

    print(f"FC_METHOD = {arm}")
    print(f"Atlases   = {', '.join(atlases)}")

    included = included_ids()
    assert len(included) == EXPECTED_N, (
        f"select_included_subjects() returned {len(included)}, expected {EXPECTED_N}."
    )
    print(f"Analytical sample: N = {len(included)} (cohort-bound)\n")

    stats = {}
    for atlas in atlases:
        print(f"[{ATLAS_LABELS[atlas]}]")
        v, n_subj = load_upper_triangles(atlas, included)
        stats[atlas] = edge_sign_stats(v, n_subj)
        stats[atlas]["n_subj"] = n_subj

    # ---- console summary ----
    print("\n" + "=" * 74)
    print(f"Edge-sign composition  (arm: {arm}, N={EXPECTED_N}, group-blind)")
    print("=" * 74)
    header = (f"{'Atlas':<14}{'edges/subj':>11}{'%pos':>8}{'%neg':>8}"
              f"{'mean+':>9}{'|mean-|':>9}{'mean|r|':>9}")
    print(header)
    print("-" * len(header))
    for atlas in atlases:
        s = stats[atlas]
        print(f"{ATLAS_LABELS[atlas]:<14}"
              f"{s['n_edges_per_subject']:>11}"
              f"{s['frac_pos']*100:>7.1f}%"
              f"{s['frac_neg']*100:>7.1f}%"
              f"{s['mean_pos']:>9.3f}"
              f"{s['abs_neg']:>9.3f}"
              f"{s['mean_abs']:>9.3f}")

    # ---- CSV (thesis-grade, verifiable) ----
    csv_path = OUT_DIR / f"edge_composition_{arm}.csv"
    rows = []
    for atlas in atlases:
        s = stats[atlas]
        rows.append({
            "arm": arm,
            "atlas": ATLAS_LABELS[atlas],
            "n_subjects": s["n_subj"],
            "edges_per_subject": s["n_edges_per_subject"],
            "edges_total": s["n_edges_total"],
            "frac_pos": s["frac_pos"],
            "frac_neg": s["frac_neg"],
            "frac_zero": s["frac_zero"],
            "mean_pos": s["mean_pos"],
            "abs_mean_neg": s["abs_neg"],
            "mean_abs": s["mean_abs"],
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # ---- figure ----
    fig_path = OUT_DIR / f"edge_composition_{arm}.png"
    make_figure(stats, arm, atlases, fig_path)

    print(f"\nCSV saved   : {csv_path}")
    print(f"Figure saved: {fig_path}")


if __name__ == "__main__":
    main()