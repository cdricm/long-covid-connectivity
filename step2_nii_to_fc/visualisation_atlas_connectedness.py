"""
3-panel figure comparing connectedness behaviour of the atlases across density
thresholds: (1) % subjects with a connected graph, (2) median number of
components, (3) median largest-component size — all per atlas x density,
positive-only thresholding.

In:  config.atlas_dir(<atlas>, "step3a_sweep", cross_strategy=True)/
     step3a_sweep_summary.csv (Pearson arm only; cross-atlas comparison over
     Schaefer-400 / Schaefer-100 / AAL).
Out: config.CROSS_DIRS["step2_fc_diagnostics"]/graph_construction/
     atlas_connectivity_properties.png.

AUC ranges (confirmatory 10-25%, sensitivity 5-50%) marked as background bands.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

assert config.FC_METHOD == "pearson", \
    "cross-atlas connectedness plot is Pearson-arm only (partial arm is Schaefer-400 only)"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================
# SETTINGS
# =============================================================
ATLAS_CSV_PATHS = {
    "Schaefer-400": config.atlas_dir("schaefer400", "step3a_sweep", cross_strategy=True) / "step3a_sweep_summary.csv",
    "Schaefer-100": config.atlas_dir("schaefer100", "step3a_sweep", cross_strategy=True) / "step3a_sweep_summary.csv",
    "AAL":          config.atlas_dir("aal",         "step3a_sweep", cross_strategy=True) / "step3a_sweep_summary.csv",
}

OUT_DIR = config.ensure(config.CROSS_DIRS["step2_fc_diagnostics"] / "graph_construction")

ATLAS_COLORS = {
    "Schaefer-400": "#1f77b4",
    "Schaefer-100": "#ff7f0e",
    "AAL":          "#2ca02c",
}

AUC_BANDS = {
    "literature (10–25%)": config.AUC_RANGE_CONFIRMATORY,
    "broad (5–50%)":       config.AUC_RANGE_SENSITIVITY,
}

STRATEGY = "positive"

ATLAS_ROIS = {
    "Schaefer-400": 400,
    "Schaefer-100": 100,
    "AAL": 116,
}

# =============================================================
# PLOTTING HELPERS
# =============================================================
def draw_auc_bands(ax):
    """Light shaded bands behind the curves to mark the AUC density ranges."""
    band_styles = {
        "literature (10–25%)": {"color": "#444444", "alpha": 0.13, "hatch": None},
        "broad (5–50%)":       {"color": "#888888", "alpha": 0.08, "hatch": "//"},
    }
    for label, (low, high) in AUC_BANDS.items():
        st = band_styles[label]
        ax.axvspan(low * 100, high * 100,
                   color=st["color"], alpha=st["alpha"],
                   hatch=st["hatch"], label=label, zorder=0)


def plot_metric(ax, data, ycol, ylabel, title, ylim=None,
                show_legend_atlases=False, show_legend_bands=False):
    draw_auc_bands(ax)

    for atlas, df_atlas in data.items():
        ax.plot(df_atlas["density"] * 100, df_atlas[ycol],
                marker="o", linewidth=2,
                color=ATLAS_COLORS[atlas], label=atlas, zorder=3)

    ax.set_xlabel("Density (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legend_atlases:
        handles = [plt.Line2D([0], [0], marker="o", color=ATLAS_COLORS[a],
                              label=a, linewidth=2) for a in data]
        leg1 = ax.legend(handles=handles, loc="best", framealpha=0.9, fontsize=9,
                         title="Atlas")
        ax.add_artist(leg1)
    if show_legend_bands:
        band_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor="#444444", alpha=0.13,
                          label="literature (10–25%)"),
            plt.Rectangle((0, 0), 1, 1, facecolor="#888888", alpha=0.08, hatch="//",
                          label="broad (5–50%)"),
        ]
        ax.legend(handles=band_handles, loc="lower right", framealpha=0.9,
                  fontsize=8, title="AUC range")


# =============================================================
# MAIN
# =============================================================
def main():
    data = {}

    print("Loading step3a sweep results ...")
    for atlas, path in ATLAS_CSV_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        df = pd.read_csv(path)
        df_pos = df[df["strategy"] == STRATEGY].sort_values("density").reset_index(drop=True)
        if df_pos.empty:
            raise ValueError(f"No rows with strategy='{STRATEGY}' in {path}")
        df_pos["largest_fraction"] = df_pos["median_largest"] / ATLAS_ROIS[atlas]
        data[atlas] = df_pos
        print(f"  {atlas}: {len(df_pos)} density steps, "
              f"densities = {df_pos['density'].tolist()}")

    print("\nBuilding 3-panel figure ...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    plot_metric(
        axes[0], data, "pct_connected", "% subjects with connected graph",
        "Connectedness across atlases",
        ylim=(-2, 105),
        show_legend_atlases=True,
    )

    plot_metric(
        axes[1], data, "median_n_comp", "Median number of components",
        "Graph fragmentation",
        show_legend_atlases=False,
    )

    plot_metric(
        axes[2], data, "largest_fraction", "Median size of largest component (fraction of ROIs)",
        "Largest connected component",
        show_legend_atlases=False,
        show_legend_bands=True,
        ylim=(0, 1.05)
    )

    fig.suptitle(
        "Atlas connectivity properties across density thresholds  "
        "(positive-only thresholding)",
        fontsize=13, y=1.02
    )
    fig.tight_layout()

    out_path = OUT_DIR / "atlas_connectivity_properties.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")

    print("\n" + "=" * 70)
    print("Connectedness summary at key density levels (positive-only)")
    print("=" * 70)
    header = f"{'Atlas':<14} " + " ".join(f"d={d*100:.0f}%".rjust(10)
                                           for d in [0.05, 0.10, 0.25, 0.50])
    print(header)
    print("-" * len(header))
    for atlas, df_atlas in data.items():
        row_vals = []
        for d in [0.05, 0.10, 0.25, 0.50]:
            matched = df_atlas[np.isclose(df_atlas["density"], d, atol=1e-3)]
            if len(matched):
                row_vals.append(f"{matched['pct_connected'].iloc[0]:>9.1f}%")
            else:
                row_vals.append("   n/a")
        print(f"{atlas:<14} " + " ".join(row_vals))


if __name__ == "__main__":
    main()
