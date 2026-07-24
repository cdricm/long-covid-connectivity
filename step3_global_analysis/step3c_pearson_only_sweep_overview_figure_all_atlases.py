"""
Cross-atlas overview figure for the graph-construction metrics (per group),
Pearson arm only. Layout: n_groups rows x 4 columns — columns 1-3 are the
density-dependent sweep metrics (one line per atlas), column 4 is signed
Modularity Q* (single value per subject, no sweep) shown as a per-atlas
distribution (strip + box).

In: step3c_aggregated.csv + step3c_modularity.csv per atlas (confirmatory
    strategy = config.CONFIRMATORY_SIGN_STRATEGY, i.e. "positive" in the
    Pearson arm).
Out: config.CROSS_DIRS["step3c_sweep_overview"]/
     graph_metrics_overview__control_vs_covid.png.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

assert config.FC_METHOD == "pearson", \
    "cross-atlas graph-metrics overview is Pearson-arm only (partial arm is Schaefer-400 only)"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================
# SETTINGS
# =============================================================
# Confirmatory graph-construction strategy for the sweep metrics. The overview
# shows the confirmatory strategy only (Pearson arm: positive), not the diagnostic
# negative/absolute variants. Read paths mirror the step3c writer exactly:
# sweep metrics live under family_A/_cross_strategy/step3c_metrics/{strategy}/,
# modularity under the sign-neutral step3c_modularity/ tree.
STRATEGY = config.CONFIRMATORY_SIGN_STRATEGY

ATLAS_AGG_PATHS = {
    "Schaefer-400": config.atlas_dir("schaefer400", f"step3c_metrics/{STRATEGY}", cross_strategy=True) / "step3c_aggregated.csv",
    "Schaefer-100": config.atlas_dir("schaefer100", f"step3c_metrics/{STRATEGY}", cross_strategy=True) / "step3c_aggregated.csv",
    "AAL":          config.atlas_dir("aal",         f"step3c_metrics/{STRATEGY}", cross_strategy=True) / "step3c_aggregated.csv",
}
ATLAS_MOD_PATHS = {
    "Schaefer-400": config.atlas_dir("schaefer400", "step3c_modularity") / "step3c_modularity.csv",
    "Schaefer-100": config.atlas_dir("schaefer100", "step3c_modularity") / "step3c_modularity.csv",
    "AAL":          config.atlas_dir("aal",         "step3c_modularity") / "step3c_modularity.csv",
}

OUT_DIR = config.ensure(config.CROSS_DIRS["step3c_sweep_overview"])

# Sweep metrics (density-dependent); modularity handled separately.
SWEEP_METRICS = [
    ("global_efficiency", "Global Efficiency", "Integration"),
    ("mean_clustering",   "Mean Clustering",   "Segregation"),
    ("assortativity",     "Assortativity",     "Topology"),
]
MOD_TITLE = ("Modularity Q*", "Segregation\n(signed, single value)")

ATLAS_COLORS = {
    "Schaefer-400": "#1f77b4",
    "Schaefer-100": "#ff7f0e",
    "AAL":          "#2ca02c",
}
ATLAS_ORDER = ["Schaefer-400", "Schaefer-100", "AAL"]

AUC_BANDS = {
    "confirmatory (10–25%)": config.AUC_RANGE_CONFIRMATORY,
    "sensitivity (5–50%)":   config.AUC_RANGE_SENSITIVITY,
}

GROUP_DISPLAY = {"COVID": "Long COVID", "CONTROL": "Control"}
SHARE_YLIM = True


# =============================================================
# LOAD + CLEAN
# =============================================================
def load_agg_csv(path):
    """Read the two-row-header aggregated CSV (groupby.agg(['mean','std','median']))."""
    df = pd.read_csv(path, header=[0, 1])
    df.columns = [
        f"{a}_{b}" if not str(b).startswith("Unnamed") else a
        for a, b in df.columns
    ]
    df = df.rename(columns={
        "Unnamed: 0_level_0": "density",
        "Unnamed: 1_level_0": "group",
    })
    # Drop any fully-empty leading row that the MultiIndex header can introduce
    df = df[df["density"].notna() | df["group"].notna()].reset_index(drop=True)
    df["density"] = pd.to_numeric(df["density"], errors="coerce")
    df["group"] = df["group"].astype(str).str.strip().str.upper()
    numeric_cols = [c for c in df.columns if c not in ("density", "group")]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df.dropna(subset=["density"]).sort_values(["density", "group"]).reset_index(drop=True)


def load_mod_csv(path):
    """Single-value signed Q* per subject."""
    df = pd.read_csv(path)
    df["group"] = df["group"].astype(str).str.strip().str.upper()
    return df.dropna(subset=["modularity_q"])


# Cohort cross-check for the subject-level modularity CSVs (step3c_metrics.csv
# is density/group-aggregated, not subject-level, so it has nothing to check
# identities against).
included_ids = set(config.select_included_subjects(
    sorted(p.name for p in config.NII_ROOT.iterdir() if p.is_dir()),
    pd.read_csv(config.GROUP_CSV), verbose=False,
))

print(f"Loading step3c aggregated sweep metrics (strategy={STRATEGY}) + modularity ...")
sweep_data, mod_data = {}, {}
for atlas in ATLAS_ORDER:
    p_agg, p_mod = ATLAS_AGG_PATHS[atlas], ATLAS_MOD_PATHS[atlas]
    if not p_agg.exists():
        raise FileNotFoundError(f"Missing sweep file: {p_agg}")
    if not p_mod.exists():
        raise FileNotFoundError(f"Missing modularity file: {p_mod}")
    sweep_data[atlas] = load_agg_csv(p_agg)
    mod_data[atlas]   = load_mod_csv(p_mod)

    loaded_ids = set(mod_data[atlas]["subject"])
    missing = sorted(included_ids - loaded_ids)
    extra   = sorted(loaded_ids - included_ids)
    if missing or extra:
        raise AssertionError(
            f"[{atlas}] step3c_modularity.csv cohort mismatch — "
            f"missing from CSV: {missing or 'none'}; "
            f"not in config cohort: {extra or 'none'}"
        )

    print(f"  {atlas}: sweep {len(sweep_data[atlas])} rows, "
          f"modularity {len(mod_data[atlas])} subjects (cohort-verified), "
          f"groups = {sorted(sweep_data[atlas]['group'].unique())}")

groups = sorted(set().union(*[set(df['group'].unique()) for df in sweep_data.values()]))
print(f"\nGroups detected: {groups}")


# =============================================================
# shared y-limits
# =============================================================
def shared_ylim_sweep():
    ylim = {}
    for metric_col, _, _ in SWEEP_METRICS:
        vals = []
        for df_atlas in sweep_data.values():
            v = df_atlas[f"{metric_col}_mean"].values
            vals.append(v[np.isfinite(v)])
        vals = np.concatenate(vals)
        lo, hi = float(np.min(vals)), float(np.max(vals))
        pad = 0.05 * (hi - lo) if hi > lo else 0.05 * abs(hi) + 1e-6
        ylim[metric_col] = (lo - pad, hi + pad)
    return ylim

def shared_ylim_mod():
    vals = np.concatenate([df["modularity_q"].values for df in mod_data.values()])
    vals = vals[np.isfinite(vals)]
    lo, hi = float(vals.min()), float(vals.max())
    pad = 0.05 * (hi - lo) if hi > lo else 0.05 * abs(hi) + 1e-6
    return (lo - pad, hi + pad)

sweep_ylim = shared_ylim_sweep() if SHARE_YLIM else None
mod_ylim   = shared_ylim_mod()   if SHARE_YLIM else None


# =============================================================
# PLOTTING
# =============================================================
def draw_auc_bands(ax):
    band_styles = {
        "confirmatory (10–25%)": {"color": "#444444", "alpha": 0.13, "hatch": None},
        "sensitivity (5–50%)":   {"color": "#888888", "alpha": 0.08, "hatch": "//"},
    }
    for label, (low, high) in AUC_BANDS.items():
        st = band_styles[label]
        ax.axvspan(low * 100, high * 100, color=st["color"],
                   alpha=st["alpha"], hatch=st["hatch"], zorder=0)


preferred = config.GROUP_ORDER
ordered_groups = [g for g in preferred if g in groups] + \
                 [g for g in groups if g not in preferred]
n_rows = len(ordered_groups)

print(f"\nBuilding {n_rows} x 4 figure (top->bottom: "
      f"{[GROUP_DISPLAY.get(g, g) for g in ordered_groups]}) ...")

fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows), squeeze=False)
rng = np.random.default_rng(config.SEED)

for row_idx, group in enumerate(ordered_groups):
    disp = GROUP_DISPLAY.get(group, group)
    is_bottom = (row_idx == n_rows - 1)

    # --- Columns 1-3: sweep metrics ---
    for col_idx, (metric_col, metric_label, metric_type) in enumerate(SWEEP_METRICS):
        ax = axes[row_idx][col_idx]
        draw_auc_bands(ax)
        for atlas in ATLAS_ORDER:
            df_g = sweep_data[atlas][sweep_data[atlas]["group"] == group].sort_values("density")
            if df_g.empty:
                print(f"  WARNING: group '{group}' absent in {atlas} sweep — skipped.")
                continue
            ax.plot(df_g["density"].values * 100, df_g[f"{metric_col}_mean"].values,
                    marker="o", linewidth=2.0, markersize=5,
                    color=ATLAS_COLORS[atlas], label=atlas, zorder=3)
        if row_idx == 0:
            ax.set_title(f"{metric_label}\n({metric_type})", fontsize=11)
        if is_bottom:
            ax.set_xlabel("Density (%)")
        ax.set_ylabel(metric_label)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if sweep_ylim is not None:
            ax.set_ylim(*sweep_ylim[metric_col])
        if col_idx == 0:
            ax.text(-0.30, 0.5, disp, transform=ax.transAxes, rotation=90,
                    va="center", ha="center", fontsize=13, fontweight="bold")

    # --- Column 4: signed Modularity Q* (single value -> per-atlas distribution) ---
    axm = axes[row_idx][3]
    positions = np.arange(len(ATLAS_ORDER))
    box_data = []
    for atlas in ATLAS_ORDER:
        vals = mod_data[atlas].loc[mod_data[atlas]["group"] == group, "modularity_q"].values
        box_data.append(vals)
    # boxplot (distribution) + jittered strip (individual subjects)
    bp = axm.boxplot(box_data, positions=positions, widths=0.55,
                     patch_artist=True, showfliers=False, zorder=2)
    for patch, atlas in zip(bp["boxes"], ATLAS_ORDER):
        patch.set_facecolor(ATLAS_COLORS[atlas]); patch.set_alpha(0.30)
    for med in bp["medians"]:
        med.set_color("black"); med.set_linewidth(1.2)
    for pos, atlas in zip(positions, ATLAS_ORDER):
        vals = mod_data[atlas].loc[mod_data[atlas]["group"] == group, "modularity_q"].values
        jitter = rng.normal(0, 0.06, size=len(vals))
        axm.scatter(pos + jitter, vals, s=10, color=ATLAS_COLORS[atlas],
                    alpha=0.5, zorder=3, edgecolors="none")
    if row_idx == 0:
        axm.set_title(f"{MOD_TITLE[0]}\n({MOD_TITLE[1]})", fontsize=11)
    if is_bottom:
        axm.set_xlabel("Atlas")
    axm.set_ylabel("Modularity Q*")
    axm.set_xticks(positions)
    axm.set_xticklabels([a.replace("Schaefer-", "Sch-") for a in ATLAS_ORDER], fontsize=9)
    axm.grid(alpha=0.2, axis="y")
    axm.spines["top"].set_visible(False); axm.spines["right"].set_visible(False)
    if mod_ylim is not None:
        axm.set_ylim(*mod_ylim)

axes[0][0].legend(loc="best", fontsize=8, framealpha=0.9, title="Atlas")
band_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor="#444444", alpha=0.13, label="confirmatory (10–25%)"),
    plt.Rectangle((0, 0), 1, 1, facecolor="#888888", alpha=0.08, hatch="//", label="sensitivity (5–50%)"),
]
fig.suptitle("Graph metrics — three atlases (Control vs Long COVID); "
             "columns 1-3 density sweep, column 4 signed Modularity Q*",
             fontsize=14, y=1.005)
fig.tight_layout()
fig.legend(handles=band_handles, loc="upper left", bbox_to_anchor=(0.93, 1),
           fontsize=9, title="AUC range (sweep cols)", framealpha=0.95)

out_path = OUT_DIR / "graph_metrics_overview__control_vs_covid.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Figure saved: {out_path}")
print("\nDone.")