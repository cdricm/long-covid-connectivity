"""
Step 3d (Family A inference) for AAL (SPM12, 116 ROIs) — ROBUSTNESS atlas: no
independent confirmatory testing, no correction across atlases.
Family A = 4 global scalars: 3 AUC metrics (Global Efficiency, Mean Clustering,
Assortativity) over the confirmatory 10-25 % range + signed Modularity Q*
(single value, no AUC).
PRIMARY = naive permutation of group labels, statistic = Welch's t; no covariates.
FDR-BH over the 4 within-atlas p_perm (literature-range AUC x3 + modularity).
The broad 5-50 % range is a declared sensitivity range (descriptive, no FDR).

Reads step3c sweep metrics for the confirmatory strategy from the _cross_strategy
tree (mirrors the step3c writer path); modularity from the sign-neutral tree.
Writes into family_A/{strategy}/ (consistent with the Schaefer-400 wrapper).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step3d_auc_pipeline import (
    compute_auc, validate_range, check_cohort, compare_family_a, plot_family_a,
    format_summary,
)
import os
import pandas as pd

ATLAS = "aal"
ATLAS_LABEL = "AAL (SPM12)"
STRATEGY = config.CONFIRMATORY_SIGN_STRATEGY  # "positive" in the Pearson arm

METRICS_CSV = config.atlas_dir(
    ATLAS, f"step3c_metrics/{STRATEGY}", cross_strategy=True) / "step3c_metrics.csv"
MOD_CSV = config.atlas_dir(ATLAS, "step3c_modularity") / "step3c_modularity.csv"
OUT_DIR = config.ensure(config.atlas_dir(ATLAS, "step3d_auc", strategy=STRATEGY))

AUC_METRICS = ["global_efficiency", "mean_clustering", "assortativity"]
# Range keys are part of the frozen analysis state: they determine the
# alphabetical groupby order in compare_family_a and thus the assignment of
# seed substreams to tests. Renaming them would change the permutation
# p-values relative to METHODS_DECISIONS.md §6.
AUC_RANGES = {"literature": config.AUC_RANGE_CONFIRMATORY,
             "broad":      config.AUC_RANGE_SENSITIVITY}
CONFIRMATORY_RANGES = ("literature", "single")  # literature AUC (3) + modularity single (1) = 4
N_PERMUTATIONS = config.N_PERMUTATIONS
SEED = config.SEED


def main():
    assert STRATEGY is not None, (
        "CONFIRMATORY_SIGN_STRATEGY is None — set the confirmatory strategy before "
        "running Family A inference."
    )

    # --- AUC metrics from step3c sweep ---
    df = pd.read_csv(METRICS_CSV)
    print(f"Strategy: {STRATEGY}")
    print(f"Loaded {len(df)} sweep rows; subjects={df['subject'].nunique()}, "
          f"densities={sorted(df['density'].unique())}")
    print(f"Groups: {df['group'].value_counts().to_dict()}\n")
    for label, (lo, hi) in AUC_RANGES.items():
        v = validate_range(df, "density", lo, hi)
        print(f"Range '{label}' ({lo}-{hi}): {v['n_in_range']} densities, covered={v['covered']}")
        if not v["covered"]:
            raise RuntimeError(f"Range {label} not covered")
    print()
    auc_df = compute_auc(df, density_col="density", metric_cols=AUC_METRICS,
                         subject_col="subject", group_col="group", ranges=AUC_RANGES)
    # --- Modularity Q* single value (NO AUC) -> range='single' ---
    mod = pd.read_csv(MOD_CSV)[["subject", "group", "modularity_q"]].dropna(subset=["modularity_q"])
    mod = mod.rename(columns={"modularity_q": "value"})
    mod["range"] = "single";
    mod["range_lo"] = float("nan");
    mod["range_hi"] = float("nan")
    mod["metric"] = "modularity_q";
    mod["n_densities"] = 1

    check_cohort(df, mod)

    values_long = pd.concat([auc_df, mod[auc_df.columns]], ignore_index=True)
    values_long.to_csv(os.path.join(OUT_DIR, "family_a_values.csv"), index=False)
    print(f"Family A values: {len(values_long)} rows "
          f"(3 AUC metrics x {len(AUC_RANGES)} ranges + Modularity single)\n")
    # --- Inference (naive permutation primary; FDR over the 4 confirmatory tests) ---
    print(f"Family A inference with {N_PERMUTATIONS} naive permutations "
          f"(Welch-t statistic, no covariates)...")
    comp = compare_family_a(
        values_long,
        confirmatory_ranges=CONFIRMATORY_RANGES,
        group_a=config.GROUP_ORDER[0], group_b=config.GROUP_ORDER[1],
        n_permutations=N_PERMUTATIONS, seed=SEED,
    )
    comp.to_csv(os.path.join(OUT_DIR, "family_a_comparison.csv"), index=False)
    plot_family_a(values_long, comp, OUT_DIR, ATLAS_LABEL,
                  confirmatory_ranges=CONFIRMATORY_RANGES)
    summary = format_summary(comp, ATLAS_LABEL, confirmatory_ranges=CONFIRMATORY_RANGES)
    print(summary)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary)
    print(f"\nOutputs in: {OUT_DIR}")
    for fn in sorted(os.listdir(OUT_DIR)):
        print(f"  {fn}")


if __name__ == "__main__":
    main()