"""
Descriptive statistics + visual QC of the Yeo-7 within/between FC (group
means/SD/median, Fisher-z primary + raw sensitivity, boxplots). No inference
here — Family B inference is step4d.

In: step4b_aggregation/yeo_fc_fisher.csv, yeo_fc_raw.csv.
Out: yeo_fc_descriptive.csv, yeo_fc_boxplots.png.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

# ============================================================
# SETTINGS
# ============================================================
IN_DIR  = config.atlas_dir("schaefer400", "step4b_aggregation")
OUT_DIR = config.ensure(config.atlas_dir("schaefer400", "step4c_descriptive"))

YEO_NETWORKS  = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
within_cols   = [f"within_{net}" for net in YEO_NETWORKS]
between_pairs = list(combinations(YEO_NETWORKS, 2))
between_cols  = [f"between_{a}_{b}" for a, b in between_pairs]
all_fc_cols   = within_cols + between_cols

GROUPS = list(config.GROUP_ORDER)

# ============================================================
# LOAD (group column comes from step4b; no re-merge needed)
# ============================================================
df_fisher = pd.read_csv(os.path.join(IN_DIR, "yeo_fc_fisher.csv"))
df_raw    = pd.read_csv(os.path.join(IN_DIR, "yeo_fc_raw.csv"))
print(f"Fisher: {df_fisher.shape}, Raw: {df_raw.shape}")

for name, d in [("fisher", df_fisher), ("raw", df_raw)]:
    assert "group" in d.columns, f"{name}: missing 'group' column (rerun step4b)"
    n_na = d["group"].isna().sum()
    assert n_na == 0, f"{name}: {n_na} rows with missing group"

# Cohort validation against config (single source of truth, consistent with 4d)
df_csv = pd.read_csv(config.GROUP_CSV)
expected = set(config.select_included_subjects(
    [p.name for p in config.NII_ROOT.iterdir() if p.is_dir()],
    df_csv, id_col="ID", group_col="Grupo", verbose=False))
actual = set(df_fisher["subject_id"])
assert actual == expected, (
    f"Cohort deviates from config: only in data {sorted(actual - expected)}, "
    f"only in config {sorted(expected - actual)}")
print(f"Cohort matches config: {len(actual)} subjects "
      f"{df_fisher['group'].value_counts().to_dict()}")

# ============================================================
# DESCRIPTIVE STATS PER GROUP (Fisher primary + raw sensitivity)
# ============================================================
rows_desc = []
for col in all_fc_cols:
    for grp in GROUPS:
        vz = df_fisher.loc[df_fisher["group"] == grp, col].values
        vr = df_raw.loc[df_raw["group"] == grp, col].values
        rows_desc.append({
            "measure": col, "group": grp, "n": len(vz),
            "mean_fisher": np.mean(vz), "sd_fisher": np.std(vz, ddof=1),
            "median_fisher": np.median(vz),
            "min_fisher": np.min(vz), "max_fisher": np.max(vz),
            "mean_raw": np.mean(vr), "sd_raw": np.std(vr, ddof=1),
        })
df_desc = pd.DataFrame(rows_desc)
df_desc.to_csv(os.path.join(OUT_DIR, "yeo_fc_descriptive.csv"), index=False)
print(f"\nDescriptive table: {df_desc.shape}")

def pivot_block(cols, title):
    piv = df_desc[df_desc["measure"].isin(cols)].pivot(
        index="measure", columns="group", values="mean_fisher")
    piv["diff_COVID_minus_CONTROL"] = piv["COVID"] - piv["CONTROL"]
    print(f"\n--- {title} (Fisher-z group means, DESCRIPTIVE) ---")
    print(piv.round(4).to_string())

pivot_block(within_cols, "Within-network FC")
pivot_block(between_cols, "Between-network FC")

# ============================================================
# BOXPLOTS — visual QC
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

def grouped_box(ax, cols, labels, title, ylabel):
    positions = np.arange(len(cols)) * 3
    for i, col in enumerate(cols):
        data = [df_fisher.loc[df_fisher["group"] == g, col].values for g in GROUPS]
        bp = ax.boxplot(data, positions=[positions[i], positions[i] + 1],
                        widths=0.8, patch_artist=True, showfliers=True)
        bp["boxes"][0].set_facecolor("#7fbf7f")  # CONTROL green
        bp["boxes"][1].set_facecolor("#ff7f7f")  # COVID red
    ax.set_xticks(positions + 0.5)
    ax.set_xticklabels(labels, rotation=90 if len(cols) > 10 else 30, fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="y", alpha=0.3)

grouped_box(axes[0], within_cols, YEO_NETWORKS,
            "Within-network FC by group (CONTROL green, COVID red)",
            "Within-network FC (Fisher-z)")
grouped_box(axes[1], between_cols,
            [c.replace("between_", "").replace("_", "-") for c in between_cols],
            "Between-network FC by group (CONTROL green, COVID red)",
            "Between-network FC (Fisher-z)")

plt.tight_layout()
boxplot_path = os.path.join(OUT_DIR, "yeo_fc_boxplots.png")
plt.savefig(boxplot_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nSaved boxplots: {boxplot_path}")
print(f"Saved descriptive: {os.path.join(OUT_DIR, 'yeo_fc_descriptive.csv')}")