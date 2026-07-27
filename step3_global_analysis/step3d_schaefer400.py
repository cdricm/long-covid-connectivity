"""
Step 3d (Family A inference) for Schaefer-400.
Family A = 4 global scalars: 3 AUC metrics (Global Efficiency, Mean Clustering,
Assortativity) over the confirmatory 10-25 % range + signed Modularity Q*
(single value, no AUC).
PRIMARY = naive permutation of group labels, statistic = Welch's t; no covariates.
FDR-BH over the 4 confirmatory primary p_perm (literature-range AUC x3 + modularity).
The broad 5-50 % range is a declared sensitivity range (descriptive, no FDR).

Reads the step3c sweep metrics for the confirmatory strategy from the
_cross_strategy tree (mirrors the step3c writer path exactly); modularity is
strategy-invariant and read from the sign-neutral tree. Writes the confirmatory
Family-A inference into family_A/{strategy}/ (consistent with step3d_a).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step3d_auc_pipeline import (
    compute_auc, validate_range, check_cohort, compare_family_a, plot_family_a,
    format_summary, load_covariates,
)
import os
import pandas as pd

ATLAS       = "schaefer400"
ATLAS_LABEL = "Schaefer-400 (7 networks)"
STRATEGY    = config.CONFIRMATORY_SIGN_STRATEGY   # "positive" (Pearson) / "absolute" (partial R1)

# R2 ⑤: the partial arm treats positive and negative subgraphs SEPARATELY (one
# 7-test family), replacing the single "absolute" strategy. The Pearson arm keeps
# a single subgraph ("positive"; its negative subgraph is degenerate, §5). This
# list drives both the metric CSVs read and the 'subgraph' column in values_long.
if config.FC_METHOD == "partial":
    SUBGRAPH_STRATEGIES = ["positive", "negative"]     # R2 ⑤: 3 AUC x 2 + mod = 7
else:
    SUBGRAPH_STRATEGIES = [STRATEGY]                    # Pearson: 3 AUC x 1 + mod = 4

# Sweep-metric CSV per subgraph strategy, read from the _cross_strategy tree
# (same path construction as the step3c writer). Modularity: sign-neutral tree.
METRICS_CSV = {
    strat: config.atlas_dir(
        ATLAS, f"step3c_metrics/{strat}", cross_strategy=True) / "step3c_metrics.csv"
    for strat in SUBGRAPH_STRATEGIES
}
MOD_CSV     = config.atlas_dir(ATLAS, "step3c_modularity") / "step3c_modularity.csv"

# Confirmatory Family-A output tree. Pearson: strategy subdir (unchanged). Partial:
# the 7-test family spans both subgraphs, so it is written to a combined subdir.
_OUT_STRATEGY = STRATEGY if config.FC_METHOD != "partial" else "pos_neg_split"
OUT_DIR     = config.ensure(config.atlas_dir(ATLAS, "step3d_auc", strategy=_OUT_STRATEGY))

AUC_METRICS         = ["global_efficiency", "mean_clustering", "assortativity"]
# Range keys are part of the frozen analysis state: they determine the
# groupby order in compare_family_a and thus the assignment of
# seed substreams to tests. Renaming them would change the permutation
# p-values relative to METHODS_DECISIONS.md §6.
AUC_RANGES          = {"literature": config.AUC_RANGE_CONFIRMATORY,
                       "broad":      config.AUC_RANGE_SENSITIVITY}
CONFIRMATORY_RANGES = ("literature", "single")   # literature AUC + modularity single
N_PERMUTATIONS      = config.N_PERMUTATIONS
SEED                = config.SEED


def main():
    assert config.FC_METHOD in ("pearson", "partial"), (
        f"FC_METHOD must be 'pearson' or 'partial' to run Family A; "
        f"got {config.FC_METHOD!r}.")
    assert all(s is not None for s in SUBGRAPH_STRATEGIES), (
        "sign strategy unresolved — Pearson uses 'positive'; partial uses the "
        "'positive'/'negative' split (R2 ⑤).")

    # --- AUC metrics from step3c sweep, per subgraph strategy (R2 ⑤) ---
    print(f"Subgraph strategies requested: {SUBGRAPH_STRATEGIES} "
          f"({'partial pos/neg split' if config.FC_METHOD == 'partial' else 'single'})")
    auc_parts = []
    used_subgraphs, skipped_subgraphs = [], []
    for strat in SUBGRAPH_STRATEGIES:
        csv_path = METRICS_CSV[strat]
        # R2 ⑤ + partial degeneracy: step3c writes NO step3c_metrics.csv for a
        # subgraph it skipped as density-degenerate (group-BLIND: the skip is
        # decided on the first subject's GE density-invariance, before any group
        # comparison). Under partial correlations the negative-only subgraph is
        # such a case — GE does not vary across densities, so its AUC is
        # undefined. We therefore drop that subgraph from Family A here, loudly
        # and on a data-driven basis (missing file == step3c degeneracy skip),
        # and record it so the test count and the write-up reflect what actually
        # entered the family. This is NOT a silent except: only a missing metrics
        # CSV is treated this way; any other read error still raises.
        if not csv_path.is_file():
            print(f"\n[subgraph={strat}] SKIPPED — no step3c_metrics.csv at {csv_path}.")
            print(f"    step3c flagged this subgraph as density-degenerate "
                  f"(group-blind); it carries no valid AUC metrics and is excluded "
                  f"from Family A. Signed information is retained via Modularity Q* "
                  f"(computed on the full signed matrix, not sign-split).")
            skipped_subgraphs.append(strat)
            continue
        df = pd.read_csv(csv_path)
        print(f"\n[subgraph={strat}] Loaded {len(df)} sweep rows; "
              f"subjects={df['subject'].nunique()}, "
              f"densities={sorted(df['density'].unique())}")
        print(f"  Groups: {df['group'].value_counts().to_dict()}")
        for label, (lo, hi) in AUC_RANGES.items():
            v = validate_range(df, "density", lo, hi)
            if not v["covered"]:
                raise RuntimeError(f"[{strat}] Range {label} not covered")
        a = compute_auc(df, density_col="density", metric_cols=AUC_METRICS,
                        subject_col="subject", group_col="group", ranges=AUC_RANGES)
        a["subgraph"] = strat
        auc_parts.append(a)
        used_subgraphs.append(strat)
        if len(used_subgraphs) == 1:
            _df_cohort_ref = df   # for check_cohort against modularity

    if not auc_parts:
        raise RuntimeError(
            "No usable subgraph metrics for Family A — every requested subgraph "
            f"({SUBGRAPH_STRATEGIES}) was degenerate/missing. Cannot proceed.")
    if skipped_subgraphs:
        print(f"\n[Family A] Subgraphs entering the family: {used_subgraphs} "
              f"(skipped as degenerate: {skipped_subgraphs}). "
              f"Test count is data-driven, not the nominal 2-subgraph split.")
    auc_df = pd.concat(auc_parts, ignore_index=True)

    # --- Modularity Q* single value (NO AUC, sign-invariant -> ONE test) ---
    mod = pd.read_csv(MOD_CSV)[["subject", "group", "modularity_q"]].dropna(subset=["modularity_q"])
    mod = mod.rename(columns={"modularity_q": "value"})
    mod["range"] = "single"; mod["range_lo"] = float("nan"); mod["range_hi"] = float("nan")
    mod["metric"] = "modularity_q"; mod["n_densities"] = 1
    mod["subgraph"] = "signed"   # signed Louvain: uses both signs by construction

    check_cohort(_df_cohort_ref, mod)

    values_long = pd.concat([auc_df, mod[auc_df.columns]], ignore_index=True)

    # --- R2 ②: attach the covariate model (age, sex) per subject so it aligns
    #     to y row-for-row inside compare_family_a (no order-dependent merge).
    subjects = sorted(values_long["subject"].unique())
    cov, sex_map = load_covariates(subjects)
    values_long = values_long.merge(cov, on="subject", how="left", validate="many_to_one")
    miss = values_long[["age", "sex_code"]].isna().any(axis=1)
    if miss.any():
        bad = sorted(values_long.loc[miss, "subject"].unique())
        raise RuntimeError(f"covariate merge left NaNs for subjects: {bad}")

    n_auc_tests = len(AUC_METRICS) * len(used_subgraphs)
    values_long.to_csv(os.path.join(OUT_DIR, "family_a_values.csv"), index=False)
    print(f"\nFamily A values: {len(values_long)} rows "
          f"({n_auc_tests} AUC tests [{len(AUC_METRICS)} metrics x "
          f"{len(used_subgraphs)} usable subgraph(s): {used_subgraphs}] + "
          f"Modularity single = {n_auc_tests + 1} confirmatory tests, "
          f"FDR-BH over these)\n")
    # --- Inference: R2 ② Freedman–Lane covariate-adjusted permutation (primary;
    #     OLS group-coefficient t), FDR-BH over the usable confirmatory tests
    #     (data-driven: degenerate subgraphs are excluded upstream). ---
    print(f"Family A inference with {N_PERMUTATIONS} Freedman–Lane permutations "
          f"(age + sex adjusted, OLS group-coefficient t)...")
    comp = compare_family_a(
        values_long,
        confirmatory_ranges=CONFIRMATORY_RANGES,
        group_a=config.GROUP_ORDER[0], group_b=config.GROUP_ORDER[1],
        n_permutations=N_PERMUTATIONS, seed=SEED,
        covariate_cols=("age", "sex_code"), se_type=config.FL_SE_TYPE,
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