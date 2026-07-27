"""
Step 4g: Yeo-7 inference visualizations (visualizes step4d_inference).

In: step4d_inference/yeo_inference_fisher.csv, yeo_inference_raw.csv.
Out: fig_yeo7_heatmap.png (7x7 Cohen's d, within=diagonal, between=off),
     fig_yeo7_within_forest.png, fig_yeo7_between_forest.png (sorted by d),
     fig_yeo7_fisher_vs_raw.png (aggregation consistency scatter).

Significance marking: only FDR-significant cells (p_perm_fdr < 0.05) are
starred, no uncorrected-p threshold is used.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from itertools import combinations

# ============================================================
# SETTINGS
# ============================================================
IN_DIR  = config.atlas_dir("schaefer400", "step4d_inference")
OUT_DIR = config.ensure(config.atlas_dir("schaefer400", "step4g_visualizations"))
YEO_NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]

P_PRIMARY = "p_perm"
P_FDR     = "p_perm_fdr"
FDR_ALPHA = config.FDR_ALPHA

# ============================================================
# LOAD + VERIFY
# ============================================================
def _load(name):
    p = os.path.join(IN_DIR, name)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"CSV not found: {p}")
    return pd.read_csv(p)

df_fisher = _load("yeo_inference_fisher.csv")
df_raw    = _load("yeo_inference_raw.csv")

required = {"measure", "cohens_d", P_PRIMARY, P_FDR}
for name, df in [("fisher", df_fisher), ("raw", df_raw)]:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}\nAvailable: {list(df.columns)}")

CI_LO, CI_HI = ("ci_lower", "ci_upper") if "ci_lower" in df_fisher.columns else (None, None)
n_fdr_sig = int((df_fisher[P_FDR] < FDR_ALPHA).sum())
print(f"[INFO] CI columns: {CI_LO}, {CI_HI}")
print(f"[INFO] Fisher tests: {len(df_fisher)} (expected 28); "
      f"FDR-significant: {n_fdr_sig}")

FDR_NOTE = ("No within- or between-network measure survived FDR correction "
            "(separate within-7 / between-21 families).")

# ============================================================
# PLOT 1: 7x7 Cohen's d heatmap (Fisher-z primary; stars only if FDR-significant)
# ============================================================
d_matrix   = np.zeros((7, 7))
fdr_matrix = np.ones((7, 7))

for i, net in enumerate(YEO_NETWORKS):
    rows = df_fisher[df_fisher["measure"] == f"within_{net}"]
    assert len(rows) == 1, f"within_{net} not unique: {len(rows)}"
    d_matrix[i, i]   = rows["cohens_d"].iloc[0]
    fdr_matrix[i, i] = rows[P_FDR].iloc[0]

for a, b in combinations(YEO_NETWORKS, 2):
    cand = [f"between_{a}_{b}", f"between_{b}_{a}"]
    rows = df_fisher[df_fisher["measure"].isin(cand)]
    assert len(rows) == 1, f"between {a}-{b} not unique: {len(rows)} (sought {cand})"
    i, j = YEO_NETWORKS.index(a), YEO_NETWORKS.index(b)
    d_matrix[i, j] = d_matrix[j, i] = rows["cohens_d"].iloc[0]
    fdr_matrix[i, j] = fdr_matrix[j, i] = rows[P_FDR].iloc[0]

vmax = np.max(np.abs(d_matrix))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
fig, ax = plt.subplots(figsize=(8.5, 7.4))
im = ax.imshow(d_matrix, cmap="RdBu_r", norm=norm, aspect="equal")
for i in range(7):
    for j in range(7):
        d = d_matrix[i, j]
        star = "*" if fdr_matrix[i, j] < FDR_ALPHA else ""   # only FDR-significant
        label = f"{d:+.2f}{star}"
        ax.text(j, i, label, ha="center", va="center", fontsize=9,
                color="black" if abs(d) < 0.5 * vmax else "white")
ax.set_xticks(range(7)); ax.set_yticks(range(7))
ax.set_xticklabels(YEO_NETWORKS, rotation=45, ha="right")
ax.set_yticklabels(YEO_NETWORKS)
ax.set_title("Yeo-7 Cohen's d (COVID - CONTROL, Fisher-z)\n"
             "diagonal = within-FC, off-diagonal = between-FC\n"
             f"* = FDR-significant within family.  {FDR_NOTE}", fontsize=9)
cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("Cohen's d", rotation=270, labelpad=15)
plt.tight_layout()
fp = os.path.join(OUT_DIR, "fig_yeo7_heatmap.png")
plt.savefig(fp, dpi=200, bbox_inches="tight"); plt.close()
print(f"[OK] Plot 1: {fp}")

# ============================================================
# PLOT 2 + 3: Forest plots (within + between); FDR-sig star only
# ============================================================
def forest_plot(df, kind, fname, title):
    sub = df[df["measure"].str.startswith(f"{kind}_")].copy().sort_values("cohens_d")
    labels = sub["measure"].str.replace(f"{kind}_", "", regex=False).str.replace("_", " - ")
    d = sub["cohens_d"].values
    fdr = sub[P_FDR].values
    pa = sub[P_PRIMARY].values
    y = np.arange(len(sub))

    fig, ax = plt.subplots(figsize=(8.5, max(3, 0.32 * len(sub) + 1.8)))
    if CI_LO is not None:
        lo, hi = sub[CI_LO].values, sub[CI_HI].values
        ax.errorbar(d, y, xerr=[d - lo, hi - d], fmt="o", color="black",
                    ecolor="gray", capsize=3, markersize=5)
        xref = hi.max()
    else:
        ax.scatter(d, y, color="black", s=30, zorder=3); xref = d.max()
    ax.axvline(0, color="red", ls="--", lw=0.8, alpha=0.7)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Cohen's d (COVID - CONTROL)")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    # annotate primary permutation p; mark FDR-sig with filled star
    for yi, di, pai, fi in zip(y, d, pa, fdr):
        star = " *" if fi < FDR_ALPHA else ""
        ax.text(xref * 1.05 if xref > 0 else 0.05, yi,
                f"p_perm={pai:.3f}{star}", va="center", fontsize=8, color="gray")
    plt.tight_layout()
    fp = os.path.join(OUT_DIR, fname)
    plt.savefig(fp, dpi=200, bbox_inches="tight"); plt.close()
    print(f"[OK] Forest {kind}: {fp}")

forest_plot(df_fisher, "within", "fig_yeo7_within_forest.png",
            "Yeo-7 within-network FC - forest (Fisher-z, primary family)\n"
            "* = FDR-significant within within-7 family")
forest_plot(df_fisher, "between", "fig_yeo7_between_forest.png",
            "Yeo-7 between-network FC - forest (Fisher-z, secondary family)\n"
            "* = FDR-significant within between-21 family")

# ============================================================
# PLOT 4: Fisher vs raw consistency
# ============================================================
merged = df_fisher[["measure", "cohens_d"]].merge(
    df_raw[["measure", "cohens_d"]], on="measure", suffixes=("_fisher", "_raw"))
r = np.corrcoef(merged["cohens_d_fisher"], merged["cohens_d_raw"])[0, 1]
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(merged["cohens_d_fisher"], merged["cohens_d_raw"],
           c="steelblue", s=35, alpha=0.8, edgecolor="black", linewidth=0.5)
lim = max(abs(merged[["cohens_d_fisher", "cohens_d_raw"]].values).max(), 0.5) * 1.1
ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.5, lw=0.8)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("Cohen's d (Fisher-z aggregation)")
ax.set_ylabel("Cohen's d (raw aggregation)")
ax.set_title(f"Aggregation consistency: Fisher-z vs raw\nr = {r:.3f} (28 tests)", fontsize=10)
ax.grid(alpha=0.3); ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
plt.tight_layout()
fp = os.path.join(OUT_DIR, "fig_yeo7_fisher_vs_raw.png")
plt.savefig(fp, dpi=200, bbox_inches="tight"); plt.close()
print(f"[OK] Plot 4: {fp}  (Fisher-raw r={r:.3f})")

print(f"\n[DONE] All plots in: {OUT_DIR}")
print(f"[NOTE] {FDR_NOTE}")