"""
Family A overview: forest plot + volcano plot, across the three atlases
(Pearson arm only).

In: family_a_comparison.csv per atlas, via
    config.atlas_dir(atlas, "step3d_auc", strategy=STRATEGY).
Out: config.CROSS_DIRS["step3e_forest_plot"]/forest_plot_family_a.png,
     volcano_plot_family_a.png.

Significance shown in two tiers: filled star = survives FDR over the 4
primary tests (p_perm_fdr < 0.05); open/grey star = uncorrected primary
only (p_perm < 0.05), not confirmatory.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

assert config.FC_METHOD == "pearson", \
    "cross-atlas Family A forest/volcano plot is Pearson-arm only (partial arm is Schaefer-400 only)"

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# SETTINGS
# ============================================================
STRATEGY = config.CONFIRMATORY_SIGN_STRATEGY   # "positive" in the Pearson arm

ATLASES = {
    "Schaefer-400": config.atlas_dir("schaefer400", "step3d_auc", strategy=STRATEGY) / "family_a_comparison.csv",
    "Schaefer-100": config.atlas_dir("schaefer100", "step3d_auc", strategy=STRATEGY) / "family_a_comparison.csv",
    "AAL"         : config.atlas_dir("aal",         "step3d_auc", strategy=STRATEGY) / "family_a_comparison.csv",
}

AUC_METRICS = ["assortativity", "mean_clustering", "global_efficiency"]
MOD_METRIC  = "modularity_q"
METRIC_LABELS = {
    "assortativity"    : "Assortativity",
    "mean_clustering"  : "Mean Clustering",
    "global_efficiency": "Global Efficiency",
    "modularity_q"     : "Modularity Q*",
}

# Panels: (range_key, metrics_list, title)
# range_key strings ("literature"/"broad") are part of the frozen analysis
# state (see step3d_auc_pipeline.compare_family_a) — not renamed.
PANELS = [
    ("literature", AUC_METRICS,  "AUC 10-25 %\n(primary)"),
    ("broad",      AUC_METRICS,  "AUC 5-50 %\n(sensitivity)"),
    ("single",     [MOD_METRIC], "Modularity Q*\n(single value)"),
]

ATLAS_COLORS  = {"Schaefer-400": "#1f77b4", "Schaefer-100": "#2ca02c", "AAL": "#d62728"}
ATLAS_MARKERS = {"Schaefer-400": "o", "Schaefer-100": "s", "AAL": "^"}

OUT_DIR = config.ensure(config.CROSS_DIRS["step3e_forest_plot"])

# ============================================================
# LOAD
# ============================================================
rows = []
for atlas, path in ATLASES.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path)
    df["atlas"] = atlas
    rows.append(df)
all_df = pd.concat(rows, ignore_index=True)

required = {"metric", "range", "cohen_d", "d_ci_lo", "d_ci_hi",
            "p_perm", "p_perm_fdr"}
missing = required - set(all_df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}\nFound: {list(all_df.columns)}")


def sig_marker(ax, x_text, y, row, color):
    """Two-tier star: filled if FDR-significant, open/grey if only uncorrected."""
    fdr = row["p_perm_fdr"]
    raw = row["p_perm"]
    if pd.notna(fdr) and fdr < 0.05:
        ax.text(x_text, y, "*", fontsize=16, fontweight="bold", va="center",
                ha="left", color=color)
    elif raw < 0.05:
        ax.text(x_text, y, "*", fontsize=16, fontweight="bold", va="center",
                ha="left", color="0.6")  # open/grey = uncorrected only


# ============================================================
# FOREST PLOT (3 panels)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=False,
                         gridspec_kw={"width_ratios": [3, 3, 1.8]})

for ax, (rng, metrics, title) in zip(axes, PANELS):
    sub = all_df[all_df["range"] == rng].copy()
    y_positions, y_labels = [], []
    y = 0
    for metric in metrics:
        for atlas in ATLASES.keys():
            row = sub[(sub["metric"] == metric) & (sub["atlas"] == atlas)]
            if row.empty:
                continue
            r = row.iloc[0]
            ax.errorbar(
                r["cohen_d"], y,
                xerr=[[r["cohen_d"] - r["d_ci_lo"]], [r["d_ci_hi"] - r["cohen_d"]]],
                fmt=ATLAS_MARKERS[atlas], color=ATLAS_COLORS[atlas],
                markersize=9, capsize=4, linewidth=1.6,
                label=atlas if (metric == metrics[0] and rng == "literature") else None,
            )
            # FDR is applied to the confirmatory tests only (MD §6); the
            # sensitivity range carries no significance claim, so it gets no
            # marker at all — neither filled nor open.
            if rng != "broad":
                sig_marker(ax, r["d_ci_hi"] + 0.04, y, r, ATLAS_COLORS[atlas])
            y_positions.append(y)
            y_labels.append(f"{METRIC_LABELS[metric]} [{atlas}]")
            y += 1
        y += 0.6

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axvspan(-0.2, 0.2, color="gray", alpha=0.08)
    ax.set_yticks(y_positions); ax.set_yticklabels(y_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Cohen's d (COVID - CONTROL)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(-0.7, 0.9)
    if rng == "single":
        ax.set_xticks([-0.5, 0.0, 0.5])   # coarse ticks for the narrow panel
    else:
        ax.set_xticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8])

axes[0].legend(loc="lower right", fontsize=9, framealpha=0.95, title="Atlas")

# Significance legend — figure-level, below the panels (out of the narrow axis)
sig_legend = [
    Line2D([0], [0], marker="*", color="black", linestyle="None", markersize=12,
           label="FDR-sig. (p_fdr<0.05)"),
    Line2D([0], [0], marker="*", color="0.6", linestyle="None", markersize=12,
           label="uncorrected only (p_perm<0.05)"),
]
fig.legend(handles=sig_legend, loc="lower center", ncol=2, fontsize=9,
           framealpha=0.95, bbox_to_anchor=(0.5, -0.04))
axes[2].legend(handles=sig_legend, loc="lower right", fontsize=8, framealpha=0.95)

fig.suptitle("Global graph measures — Cohen's d with 95 % CI across atlases  "
             "(positive d → COVID > CONTROL; Freedman–Lane permutation, "
             "age + sex adjusted)", fontsize=12, y=1.01)
plt.tight_layout(rect=[0, 0.03, 1, 1])   # leave room at bottom for fig.legend
out_png = os.path.join(OUT_DIR, "forest_plot_family_a.png")
plt.savefig(out_png, dpi=200, bbox_inches="tight")
plt.close()
print(f"Forest plot saved: {out_png}")

# ============================================================
# VOLCANO PLOT (on primary p_perm)
# ============================================================
METRIC_MARKERS = {
    "assortativity": "o", "mean_clustering": "s",
    "global_efficiency": "D", "modularity_q": "^",
}
VOLCANO_RANGES = [("literature", "AUC 10-25 % (primary)"),
                  ("broad",      "AUC 5-50 % (sensitivity)")]

fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)

for ax, (rng, title) in zip(axes, VOLCANO_RANGES):
    sub = all_df[all_df["range"] == rng].copy()
    # include modularity (single) on each panel as a range-independent reference point
    mod = all_df[all_df["range"] == "single"].copy()
    plot_df = pd.concat([sub, mod], ignore_index=True)
    plot_df["neg_log10_p"] = -np.log10(plot_df["p_perm"].clip(lower=1e-4))

    for atlas in ATLASES.keys():
        a_sub = plot_df[plot_df["atlas"] == atlas]
        for _, r in a_sub.iterrows():
            ax.scatter(r["cohen_d"], r["neg_log10_p"],
                       color=ATLAS_COLORS[atlas],
                       marker=METRIC_MARKERS.get(r["metric"], "x"),
                       s=90, alpha=0.85)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.axhline(-np.log10(0.05), color="gray", linestyle="--", linewidth=1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Cohen's d (COVID - CONTROL)", fontsize=10)
    ax.grid(alpha=0.3)

axes[0].set_ylabel("-log10(p_perm)  [primary, Freedman–Lane permutation]", fontsize=10)

atlas_legend = [Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=ATLAS_COLORS[a], label=a, markersize=9)
                for a in ATLASES.keys()]
metric_legend = [Line2D([0], [0], marker=METRIC_MARKERS[m], color="black",
                        linestyle="None", label=METRIC_LABELS[m], markersize=8)
                 for m in AUC_METRICS + [MOD_METRIC]]
leg1 = axes[1].legend(handles=atlas_legend, title="Atlas", loc="lower right", bbox_to_anchor=(1, 0.20),
               fontsize=9, framealpha=0.95)
axes[1].add_artist(leg1)
axes[1].add_artist(leg1)
axes[1].legend(handles=metric_legend, title="Metric", loc="lower right",
               fontsize=8, framealpha=0.95)

fig.suptitle("Global graph measures — effect size vs primary permutation p\n"
             "(modularity Q* shown on both panels as a range-independent "
             "reference; line = p_perm 0.05, uncorrected)", fontsize=11)
plt.tight_layout()
volcano_png = os.path.join(OUT_DIR, "volcano_plot_family_a.png")
plt.savefig(volcano_png, dpi=200, bbox_inches="tight")
plt.close()
print(f"Volcano plot saved: {volcano_png}")

# ============================================================
# Sanity print
# ============================================================
print("\nValues plotted (d [CI], p_perm, p_fdr):")
for rng, metrics, _ in PANELS:
    print(f"\n--- Range: {rng} ---")
    sub = all_df[all_df["range"] == rng]
    for metric in metrics:
        for atlas in ATLASES.keys():
            row = sub[(sub["metric"] == metric) & (sub["atlas"] == atlas)]
            if row.empty:
                continue
            r = row.iloc[0]
            fdr = r["p_perm_fdr"]
            fdr_s = f"{fdr:.3f}" if pd.notna(fdr) else "n/a"
            tier = "FDR*" if (pd.notna(fdr) and fdr < 0.05) else \
                   ("raw*" if r["p_perm"] < 0.05 else "")
            print(f"  {atlas:13s} {METRIC_LABELS[metric]:18s} "
                  f"d={r['cohen_d']:+.3f} [{r['d_ci_lo']:+.2f},{r['d_ci_hi']:+.2f}]  "
                  f"p_perm={r['p_perm']:.3f}  p_fdr={fdr_s} {tier}")