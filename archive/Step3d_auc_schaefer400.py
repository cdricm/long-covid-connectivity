"""Step 3d: AUC integration + group comparison for Schaefer-400.

Calls the reusable auc_pipeline. For Schaefer-100/AAL: copy this file,
change paths and atlas label only.
"""

import os, sys
import pandas as pd
from pathlib import Path

# Make auc_pipeline importable; adjust if you place it elsewhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step3_global_analysis.step3d_auc_pipeline import compute_auc, validate_range, compare_groups, \
                         plot_auc_by_range, format_summary

# ===== Atlas-specific config ==================================================
ATLAS_LABEL    = "Schaefer-400 (7 networks)"
METRICS_CSV    = "/mnt/d87cc26d-5470-443c-81c1-e09b68ee4730/Cedric/analysis_outputs/step3c_metrics_sweep/step3c_metrics.csv"
OUT_DIR        = "/mnt/d87cc26d-5470-443c-81c1-e09b68ee4730/Cedric/analysis_outputs/step3d_auc_schaefer400"

# A priori AUC ranges - keep identical across atlases unless connectedness forces change
AUC_RANGES = {
    "literature": (0.10, 0.25),
    "broad"     : (0.05, 0.50),
}

METRICS = ["global_efficiency", "modularity_q", "mean_clustering", "assortativity"]

N_PERMUTATIONS = 10000
SEED           = 42

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ===== Load metrics ===========================================================
df = pd.read_csv(METRICS_CSV)
print(f"Loaded {len(df)} rows from {METRICS_CSV}")
print(f"Subjects: {df['subject'].nunique()}, "
      f"Densities: {sorted(df['density'].unique())}")
print(f"Groups: {df['group'].value_counts().to_dict()}\n")

# ===== Validate ranges ========================================================
for label, (lo, hi) in AUC_RANGES.items():
    v = validate_range(df, density_col="density", lo=lo, hi=hi)
    print(f"Range '{label}' ({lo}-{hi}): {v['n_in_range']} densities in range "
          f"({v['in_range']}), covered={v['covered']}")
    if not v["covered"]:
        raise RuntimeError(f"Range {label} not adequately covered; aborting")
print()

# ===== Compute AUC ============================================================
auc_df = compute_auc(df, density_col="density", metric_cols=METRICS,
                    subject_col="subject", group_col="group",
                    ranges=AUC_RANGES)
auc_df.to_csv(os.path.join(OUT_DIR, "auc_values.csv"), index=False)
print(f"Computed AUC: {len(auc_df)} rows "
      f"({df['subject'].nunique()} subjects x {len(AUC_RANGES)} ranges x {len(METRICS)} metrics)\n")

# ===== Group comparison =======================================================
print(f"Running group comparison with {N_PERMUTATIONS} permutations...")
comparison_df = compare_groups(auc_df, group_col="group",
                               group_a="CONTROL", group_b="COVID",
                               n_permutations=N_PERMUTATIONS, seed=SEED)
comparison_df.to_csv(os.path.join(OUT_DIR, "group_comparison.csv"), index=False)

# ===== Plots ==================================================================
plot_paths = plot_auc_by_range(auc_df, comparison_df, METRICS, OUT_DIR,
                               atlas_label=ATLAS_LABEL)
print(f"Plots: {[os.path.basename(p) for p in plot_paths]}\n")

# ===== Text summary ===========================================================
summary = format_summary(comparison_df, ATLAS_LABEL, AUC_RANGES)
print(summary)
with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
    f.write(summary)

print(f"\nOutputs in: {OUT_DIR}")
for fn in sorted(os.listdir(OUT_DIR)):
    print(f"  {fn}")