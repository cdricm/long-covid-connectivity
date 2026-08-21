"""
Pearson r distribution comparison across Schaefer-400, Schaefer-100, AAL, with
density-threshold markers (5/10/25/50%) and AUC bands (confirmatory 10-25%,
sensitivity 5-50%) — 4-panel figure (overlay + per-atlas detail).

In:  config.atlas_dir(<atlas>, "step2_pipeline")/comet_matrices (Pearson arm
     only; cross-atlas comparison over Schaefer-400 / Schaefer-100 / AAL).
Out: config.CROSS_DIRS["step2_fc_diagnostics"]/graph_construction/
     fc_distribution_3atlases.png.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

assert config.FC_METHOD == "pearson", \
    "cross-atlas FC-distribution plot is Pearson-arm only (partial arm is Schaefer-400 only)"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm
import pandas as pd
import re

# =============================================================
# SETTINGS
# =============================================================
FC_DIRS = {
    "Schaefer-400": config.atlas_dir("schaefer400", "step2_pipeline") / "comet_matrices",
    "Schaefer-100": config.atlas_dir("schaefer100", "step2_pipeline") / "comet_matrices",
    "AAL":          config.atlas_dir("aal",         "step2_pipeline") / "comet_matrices",
}

OUT_DIR = config.ensure(config.CROSS_DIRS["step2_fc_diagnostics"] / "graph_construction")

# density thresholds to visualise (proportional thresholding %)
DENSITY_THRESHOLDS = [0.05, 0.10, 0.25, 0.50]

# AUC bands
AUC_BANDS = {
    "confirmatory (10–25%)": config.AUC_RANGE_CONFIRMATORY,
    "sensitivity (5–50%)":   config.AUC_RANGE_SENSITIVITY,
}

# colour scheme — consistent across all plots
ATLAS_COLORS = {
    "Schaefer-400": "#1f77b4",   # primary
    "Schaefer-100": "#ff7f0e",   # robustness
    "AAL":          "#2ca02c",   # robustness
}

# =============================================================
# HELPERS
# =============================================================
def included_ids():
    """Analytical sample (gates a+b+c) as a sorted list of subject IDs."""
    nii_subjects = sorted(p.name for p in config.NII_ROOT.iterdir() if p.is_dir())
    group_df = pd.read_csv(config.GROUP_CSV)
    return config.select_included_subjects(nii_subjects, group_df, verbose=False)

def load_upper_triangle_fc(fc_dir: Path, included: list[str]):
    """Load FC matrices of the INCLUDED subjects and return flattened
    upper-triangle values. Aborts if the directory does not match the sample."""
    npy_files = sorted(fc_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files in {fc_dir}")

    id_to_path = {}
    for f in npy_files:
        m = re.search(r"(CP\d+)", f.name)
        if m:
            id_to_path[m.group(1)] = f

    missing = [s for s in included if s not in id_to_path]
    if missing:
        raise AssertionError(
            f"{len(missing)} included subjects have no matrix in {fc_dir}: {missing}")

    all_values = []
    for s in tqdm(sorted(included), desc=f"  loading {fc_dir.name}", leave=False):
        mat = np.load(id_to_path[s])
        n = mat.shape[0]
        iu = np.triu_indices(n, k=1)
        all_values.append(mat[iu])
    return np.concatenate(all_values), len(included)

def density_to_r_quantile(fc_values_flat, density):
    """r value at which proportional thresholding keeps the requested density.

    Density is the fraction of ALL possible edges (config.proportional_threshold),
    so the quantile is taken over all edge slots after positive-only masking:
    negative edges are set to zero and cannot be selected.
    """
    w = np.where(fc_values_flat > 0, fc_values_flat, 0.0)
    return float(np.quantile(w, 1 - density))


# =============================================================
# PLOTTING HELPER
# =============================================================
def plot_distribution(ax, atlas_data, r_thresholds, atlas_names,
                      show_legend=True, title=None):
    """Plot Pearson r distribution(s) on a given axis."""
    band_styles = {
        "confirmatory (10–25%)": {"color": "#444444", "alpha": 0.15, "hatch": None},
        "sensitivity (5–50%)": {"color": "#888888", "alpha": 0.10, "hatch": "//"},
    }

    for band_name, (d_low, d_high) in AUC_BANDS.items():
        # density 5%  = top 5%  = highest r threshold (upper edge of band)
        # density 50% = top 50% = lowest  r threshold (lower edge of band)
        r_upper = np.nanmean([r_thresholds[a][d_low]  for a in atlas_names])
        r_lower = np.nanmean([r_thresholds[a][d_high] for a in atlas_names])
        st = band_styles[band_name]
        ax.axvspan(r_lower, r_upper,
                   color=st["color"], alpha=st["alpha"],
                   hatch=st["hatch"], label=band_name, zorder=1)

    for atlas in atlas_names:
        vals = atlas_data[atlas]["values"]
        if len(vals) > 500_000:
            rng = np.random.default_rng(config.SEED)
            vals_plot = rng.choice(vals, size=500_000, replace=False)
        else:
            vals_plot = vals
        ax.hist(vals_plot, bins=120, range=(-1, 1),
                histtype="step", linewidth=1.6,
                color=ATLAS_COLORS[atlas],
                density=True, label=atlas, zorder=3)

    if len(atlas_names) == 1:
        atlas = atlas_names[0]
        for dens in DENSITY_THRESHOLDS:
            r_val = r_thresholds[atlas][dens]
            ax.axvline(r_val, color="black", linewidth=0.8,
                       linestyle="--", alpha=0.6, zorder=2)
            ax.text(r_val, ax.get_ylim()[1] * 0.95,
                    f"{int(dens*100)}%",
                    ha="center", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="white", edgecolor="grey", alpha=0.9))
    else:
        for dens in DENSITY_THRESHOLDS:
            r_val = np.nanmean([r_thresholds[a][dens] for a in atlas_names])
            ax.axvline(r_val, color="black", linewidth=0.8,
                       linestyle="--", alpha=0.5, zorder=2)
            ax.text(r_val, ax.get_ylim()[1] * 0.95,
                    f"{int(dens*100)}%",
                    ha="center", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="white", edgecolor="grey", alpha=0.9))

    ax.set_xlabel("Pearson r")
    ax.set_ylabel("Proportion of edges per unit r")
    ax.set_xlim(-1, 1)
    ax.axvline(0, color="grey", linewidth=0.6, alpha=0.5, zorder=2)
    if title:
        ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)


# =============================================================
# MAIN
# =============================================================
def main():
    atlas_data   = {}
    r_thresholds = {}

    included = included_ids()
    print(f"Analytical sample: N = {len(included)} (cohort-bound)\n")

    print("Loading FC matrices ...")
    for name, fc_dir in FC_DIRS.items():
        print(f"\n[{name}]")
        values, n_subj = load_upper_triangle_fc(fc_dir, included)
        atlas_data[name] = {
            "values": values,
            "n_subj": n_subj,
            "n_edges_per_subj": len(values) // n_subj,
        }
        print(f"  {n_subj} subjects, {atlas_data[name]['n_edges_per_subj']} edges/subj")

    print("\nComputing r-thresholds per atlas/density ...")
    for atlas, d in atlas_data.items():
        r_thresholds[atlas] = {dens: density_to_r_quantile(d["values"], dens)
                               for dens in DENSITY_THRESHOLDS}
        print(f"  {atlas}: " + ", ".join(
            f"{int(dens*100)}%→r={r_thresholds[atlas][dens]:.3f}"
            for dens in DENSITY_THRESHOLDS))

    print("\nBuilding figure ...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    plot_distribution(axes[0, 0], atlas_data, r_thresholds,
                      list(atlas_data.keys()),
                      show_legend=True,
                      title="All atlases overlaid")

    def plot_schaefer400_plain(atlas_data, out_dir: Path):
        """Plain Pearson r distribution for Schaefer-400 — no density markers,
        no AUC bands. Uses all edges of all included subjects (no subsampling)."""
        atlas = "Schaefer-400"
        vals = atlas_data[atlas]["values"]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(vals, bins=200, range=(-1, 1),
                histtype="stepfilled", linewidth=1.6,
                color=ATLAS_COLORS[atlas], alpha=0.25,
                density=True, zorder=2)
        ax.hist(vals, bins=200, range=(-1, 1),
                histtype="step", linewidth=1.8,
                color=ATLAS_COLORS[atlas],
                density=True, zorder=3)

        ax.axvline(0, color="grey", linewidth=0.6, alpha=0.5, zorder=1)
        ax.set_xlabel("Pearson r")
        ax.set_ylabel("Proportion of edges per unit r")
        ax.set_xlim(-1, 1)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{atlas} — Pearson r distribution "
                     f"(n_subj = {atlas_data[atlas]['n_subj']}, "
                     f"{atlas_data[atlas]['n_edges_per_subj']} edges/subj)")

        fig.tight_layout()
        out_path = out_dir / "fc_distribution_schaefer400.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved: {out_path}")

    atlas_list = list(atlas_data.keys())
    plot_positions = [(0, 1), (1, 0), (1, 1)]
    for (row, col), atlas in zip(plot_positions, atlas_list):
        n_subj = atlas_data[atlas]["n_subj"]
        plot_distribution(axes[row, col], atlas_data, r_thresholds, [atlas],
                          show_legend=True,
                          title=f"{atlas} (n_subj = {n_subj})")

    fig.suptitle("Pearson r distribution across atlases\n"
                 "with density thresholds (5/10/25/50%) and AUC bands",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = OUT_DIR / "fc_distribution_3atlases.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")
    plot_schaefer400_plain(atlas_data, OUT_DIR)

    print("\n" + "=" * 70)
    print("r-thresholds per atlas/density (mean across all edges/subjects)")
    print("=" * 70)
    print(f"{'Atlas':<15} " + " ".join(f"{int(d*100)}%".rjust(8) for d in DENSITY_THRESHOLDS))
    for atlas in atlas_data:
        row = f"{atlas:<15} " + " ".join(
            f"{r_thresholds[atlas][d]:.3f}".rjust(8) for d in DENSITY_THRESHOLDS
        )
        print(row)


if __name__ == "__main__":
    main()
