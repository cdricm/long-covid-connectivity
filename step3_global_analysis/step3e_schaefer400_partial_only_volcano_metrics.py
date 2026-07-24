"""
Family A volcano for the partial-correlation analysis (Schaefer-400 only,
partial arm only).

In: family_a_comparison.csv, via
    config.atlas_dir("schaefer400", "step3d_auc", strategy=STRATEGY).
Out: config.atlas_dir("schaefer400", "step3e_volcano", strategy=STRATEGY)/
     volcano_plot_family_a.png.

Colour encodes FDR tier (red = survives FDR over the 4 confirmatory tests,
blue = does not) instead of atlas, since there is only one atlas here.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

assert config.FC_METHOD == "partial", \
    "this is the partial-arm single-atlas volcano; use the cross-atlas forest/volcano script for the Pearson arm"

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# SETTINGS
# ============================================================
STRATEGY = config.CONFIRMATORY_SIGN_STRATEGY

ATLAS       = "schaefer400"
ATLAS_LABEL = "Schaefer-400"
CSV_PATH    = config.atlas_dir(ATLAS, "step3d_auc", strategy=STRATEGY) / "family_a_comparison.csv"

AUC_METRICS = ["assortativity", "mean_clustering", "global_efficiency"]
MOD_METRIC  = "modularity_q"
METRIC_LABELS = {
    "assortativity"    : "Assortativity",
    "mean_clustering"  : "Mean Clustering",
    "global_efficiency": "Global Efficiency",
    "modularity_q"     : "Modularity Q*",
}
METRIC_MARKERS = {
    "assortativity": "o", "mean_clustering": "s",
    "global_efficiency": "D", "modularity_q": "^",
}

COLOR_NS  = "#1f77b4"   # not FDR-significant
COLOR_SIG = "#d62728"   # survives FDR over the 4 confirmatory tests

VOLCANO_RANGES = [("literature", "AUC 10-25 % (confirmatory)"),
                  ("broad",      "AUC 5-50 % (sensitivity)")]

OUT_DIR = config.ensure(config.atlas_dir(ATLAS, "step3e_volcano", strategy=STRATEGY))

# ============================================================
# LOAD
# ============================================================
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Missing: {CSV_PATH}")
all_df = pd.read_csv(CSV_PATH)

required = {"metric", "range", "cohen_d", "d_ci_lo", "d_ci_hi",
            "p_perm", "p_perm_fdr"}
missing = required - set(all_df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}\nFound: {list(all_df.columns)}")

# ============================================================
# VOLCANO PLOT (on primary p_perm)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)

for ax, (rng, title) in zip(axes, VOLCANO_RANGES):
    sub = all_df[all_df["range"] == rng].copy()
    # include modularity (single) on each panel as a range-independent reference point
    mod = all_df[all_df["range"] == "single"].copy()
    plot_df = pd.concat([sub, mod], ignore_index=True)
    plot_df["neg_log10_p"] = -np.log10(plot_df["p_perm"].clip(lower=1e-4))

    for _, r in plot_df.iterrows():
        fdr = r["p_perm_fdr"]
        survives = pd.notna(fdr) and fdr < 0.05
        ax.scatter(r["cohen_d"], r["neg_log10_p"],
                   color=COLOR_SIG if survives else COLOR_NS,
                   marker=METRIC_MARKERS.get(r["metric"], "x"),
                   s=90, alpha=0.85)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.axhline(-np.log10(0.05), color="gray", linestyle="--", linewidth=1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Cohen's d (COVID - CONTROL)", fontsize=10)
    ax.grid(alpha=0.3)

axes[0].set_ylabel("-log10(p_perm)  [primary, naive permutation]", fontsize=10)

metric_legend = [Line2D([0], [0], marker=METRIC_MARKERS[m], color="black",
                        linestyle="None", label=METRIC_LABELS[m], markersize=8)
                 for m in AUC_METRICS + [MOD_METRIC]]
tier_legend = [
    Line2D([0], [0], marker="o", color=COLOR_SIG, linestyle="None",
           markersize=9, label="FDR-sig. (p_fdr<0.05)"),
    Line2D([0], [0], marker="o", color=COLOR_NS, linestyle="None",
           markersize=9, label="not FDR-sig."),
]
leg1 = axes[1].legend(handles=metric_legend, title="Metric", loc="upper right",
                      fontsize=9, framealpha=0.95)
axes[1].add_artist(leg1)
axes[1].legend(handles=tier_legend, title="FDR tier", loc="lower right",
               fontsize=8, framealpha=0.95)

fig.suptitle(f"Family A volcano ({ATLAS_LABEL}, partial correlations, "
             f"{STRATEGY}) — effect size vs primary permutation p\n"
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
for rng, _ in VOLCANO_RANGES + [("single", "")]:
    print(f"\n--- Range: {rng} ---")
    sub = all_df[all_df["range"] == rng]
    metrics = AUC_METRICS if rng != "single" else [MOD_METRIC]
    for metric in metrics:
        row = sub[sub["metric"] == metric]
        if row.empty:
            continue
        r = row.iloc[0]
        fdr = r["p_perm_fdr"]
        fdr_s = f"{fdr:.3f}" if pd.notna(fdr) else "n/a"
        tier = "FDR*" if (pd.notna(fdr) and fdr < 0.05) else \
               ("raw*" if r["p_perm"] < 0.05 else "")
        print(f"  {METRIC_LABELS[metric]:18s} "
              f"d={r['cohen_d']:+.3f} [{r['d_ci_lo']:+.2f},{r['d_ci_hi']:+.2f}]  "
              f"p_perm={r['p_perm']:.3f}  p_fdr={fdr_s} {tier}")