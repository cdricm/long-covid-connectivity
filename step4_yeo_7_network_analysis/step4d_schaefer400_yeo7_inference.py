"""
Step 4d: Family B inference — Yeo-7 within/between FC (COVID vs CONTROL).

In: step4b_aggregation/yeo_fc_fisher.csv, yeo_fc_raw.csv.
Out: yeo_inference_fisher.csv (primary aggregation),
     yeo_inference_raw.csv (sensitivity aggregation).

Two separately-corrected FDR families: 7 within-network tests, 21 between-
network tests. Reuses naive_permutation/cohens_d directly from
step3d_auc_pipeline (not reimplemented) so Family A and Family B rest on
exactly the same validated inference functions.
"""

import sys
from pathlib import Path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))   # project root (for config)
import config

# step3d_auc_pipeline lives in the sibling step3 analysis folder. Locate it
# robustly so Family B reuses the EXACT validated Family A inference functions
# (naive_permutation, cohens_d).
def _find_step3d_dir(root):
    for cand in sorted(root.iterdir()):
        if cand.is_dir() and (cand / "step3d_auc_pipeline.py").exists():
            return cand
    raise FileNotFoundError(
        "step3d_auc_pipeline.py not found in any sibling folder under "
        f"{root}. Adjust the path so Family B can reuse the Family A functions.")

_STEP3_DIR = _find_step3d_dir(_HERE.parents[1])
sys.path.insert(0, str(_STEP3_DIR))
from step3d_auc_pipeline import (
    freedman_lane_permutation, cohens_d, load_covariates,
)

import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from itertools import combinations

# ============================================================
# SETTINGS
# ============================================================
IN_DIR  = config.atlas_dir("schaefer400", "step4b_aggregation")
OUT_DIR = config.ensure(config.atlas_dir("schaefer400", "step4d_inference"))

YEO_NETWORKS  = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
within_cols   = [f"within_{net}" for net in YEO_NETWORKS]
between_pairs = list(combinations(YEO_NETWORKS, 2))
between_cols  = [f"between_{a}_{b}" for a, b in between_pairs]

N_PERMUTATIONS = config.N_PERMUTATIONS
SEED           = config.SEED
FDR_ALPHA      = config.FDR_ALPHA
GROUP_A, GROUP_B = config.GROUP_ORDER   # b - a = COVID - CONTROL

# ============================================================
# LOAD (group from step4b) + validate cohort against config
# ============================================================
df_fisher = pd.read_csv(os.path.join(IN_DIR, "yeo_fc_fisher.csv"))
df_raw    = pd.read_csv(os.path.join(IN_DIR, "yeo_fc_raw.csv"))
df_csv    = pd.read_csv(config.GROUP_CSV)

for name, d in [("fisher", df_fisher), ("raw", df_raw)]:
    assert "group" in d.columns, f"{name}: missing 'group' (rerun step4b)"
    assert d["group"].isna().sum() == 0, f"{name}: missing group values"

expected = set(config.select_included_subjects(
    [p.name for p in config.NII_ROOT.iterdir() if p.is_dir()],
    df_csv, id_col="ID", group_col="Grupo", verbose=False))
assert set(df_fisher["subject_id"]) == expected, "Cohort deviates from config"

n_covid   = int((df_fisher["group"] == GROUP_B).sum())
n_control = int((df_fisher["group"] == GROUP_A).sum())
print(f"Cohort verified against config: N={len(expected)} "
      f"(COVID={n_covid}, CONTROL={n_control})")

# Hard cohort guard: Family B runs on the full frozen cohort, no covariate-driven
# subject drop (all 162 retained; covariates adjust, they do not exclude).
assert len(expected) == 162 and n_covid == 123 and n_control == 39, \
    "cohort deviates from frozen 162 (123/39)"

# --- R2 ②: attach the covariate model (age, sex) per subject to both
#     aggregations, so it aligns to y row-for-row inside run_family. Same loader /
#     same covariate model as Family A (supervisor: identical model all outcomes).
_cov, _sex_map = load_covariates(sorted(expected))
df_fisher = df_fisher.merge(_cov, left_on="subject_id", right_on="subject",
                            how="left", validate="one_to_one")
df_raw    = df_raw.merge(_cov, left_on="subject_id", right_on="subject",
                         how="left", validate="one_to_one")
for _name, _d in [("fisher", df_fisher), ("raw", df_raw)]:
    if _d[["age", "sex_code"]].isna().any().any():
        _bad = sorted(_d.loc[_d[["age", "sex_code"]].isna().any(axis=1), "subject_id"])
        raise RuntimeError(f"{_name}: covariate merge left NaNs for {_bad}")

# ============================================================
# INFERENCE — R2 ②: Freedman–Lane covariate-adjusted permutation (primary;
# OLS group-coefficient t, age + sex adjusted). Same model as Family A.
# ============================================================
def run_family(df, family_cols, family_name, agg_label, substreams, offset):
    """One test family. PRIMARY = Freedman–Lane covariate-adjusted permutation
    (OLS group-coefficient t, age + sex adjusted); parametric p of the same
    adjusted coefficient as sensitivity; raw Cohen's d (unadjusted descriptive).
    Returns DataFrame; FDR-BH applied within family."""
    print(f"\n=== {agg_label} | family '{family_name}' ({len(family_cols)} tests) ===")
    rows = []
    for i, col in enumerate(family_cols):
        g = df["group"].map({GROUP_A: 0, GROUP_B: 1}).values.astype(float)
        y = df[col].values.astype(float)
        Z = df[["age", "sex_code"]].values.astype(float)   # (n, 2): age, sex_code
        keep = ~np.isnan(g) & np.isfinite(y) & np.isfinite(Z).all(axis=1)
        y, g, Z = y[keep], g[keep], Z[keep]
        a, b = y[g == 0], y[g == 1]   # a=CONTROL, b=COVID

        # PRIMARY: Freedman–Lane covariate-adjusted permutation (group-coeff t),
        # one substream per test (assignment unchanged from R1).
        perm = freedman_lane_permutation(y, g, Z, N_PERMUTATIONS,
                                         substreams[offset + i],
                                         se_type=config.FL_SE_TYPE)
        # SENSITIVITY: parametric p of the SAME adjusted group coefficient.
        t_welch, p_welch = perm["t_obs"], perm["p_param"]
        # EFFECT: raw Cohen's d + CI (unadjusted descriptive)
        d, d_lo, d_hi = cohens_d(a, b)

        rows.append({
            "measure": col, "family": family_name,
            "n_covid": int((g == 1).sum()), "n_control": int((g == 0).sum()),
            "mean_covid": b.mean(), "mean_control": a.mean(),
            "mean_diff": b.mean() - a.mean(),
            "cohens_d": d, "ci_lower": d_lo, "ci_upper": d_hi,
            "t_perm": perm["t_obs"],     # observed adjusted group-coefficient t
            "p_perm": perm["p_perm"],    # PRIMARY (FL permutation null)
            "t_welch": t_welch,          # = adjusted group-coefficient t
            "p_welch": p_welch,          # SENSITIVITY: parametric p of adjusted t
        })
        print(f"  [{i+1}/{len(family_cols)}] {col}: d={d:+.3f} [{d_lo:+.3f},{d_hi:+.3f}] "
              f"p_perm={perm['p_perm']:.4f} p_param={p_welch:.4f}")

    res = pd.DataFrame(rows)
    # FDR-BH WITHIN family — primary on p_perm; sensitivity on welch
    _, res["p_perm_fdr"], _, _  = multipletests(res["p_perm"],  alpha=FDR_ALPHA, method="fdr_bh")
    _, res["p_welch_fdr"], _, _ = multipletests(res["p_welch"], alpha=FDR_ALPHA, method="fdr_bh")
    return res

# Global substreams: 1 per test x (7 within + 21 between) = 28 per aggregation,
# x 2 aggregations (fisher/raw) = 56, so fisher/raw tests have distinct streams.
n_tests = len(within_cols) + len(between_cols)
all_substreams = np.random.SeedSequence(SEED).spawn(n_tests * 2)  # *2 aggregations

results = {}
for agg_idx, (label, df) in enumerate([("FISHER-Z (PRIMARY)", df_fisher),
                                       ("RAW (SENSITIVITY)", df_raw)]):
    base = agg_idx * n_tests
    res_within  = run_family(df, within_cols,  "within",  label, all_substreams, base)
    res_between = run_family(df, between_cols, "between", label, all_substreams,
                             base + len(within_cols))
    res = pd.concat([res_within, res_between], ignore_index=True)
    res["aggregation"] = "fisher" if "FISHER" in label else "raw"
    results["fisher" if "FISHER" in label else "raw"] = res

# ============================================================
# SUMMARY
# ============================================================
def print_summary(res, label):
    print(f"\n{'='*72}\nSUMMARY — {label}\n{'='*72}")
    for fam in ["within", "between"]:
        sub = res[res["family"] == fam].sort_values("p_perm")
        n_sig = int((sub["p_perm_fdr"] < 0.05).sum())
        print(f"\n  Family '{fam}' ({len(sub)} tests): "
              f"FDR-significant (primary p_perm): {n_sig}")
        cols = ["measure", "cohens_d", "p_perm", "p_perm_fdr", "p_welch"]
        print(sub[cols].head(5).to_string(index=False))

print_summary(results["fisher"], "FISHER-Z (PRIMARY)")
print_summary(results["raw"], "RAW (SENSITIVITY)")

# ============================================================
# DIRECTIONAL CONSISTENCY (context for the global FC shift; see step4e)
# ============================================================
print(f"\n{'='*72}\nDIRECTIONAL CONSISTENCY (Fisher-z, primary)\n{'='*72}")
for fam in ["within", "between"]:
    sub = results["fisher"][results["fisher"]["family"] == fam]
    n_neg = int((sub["cohens_d"] < 0).sum()); n_tot = len(sub)
    print(f"  {fam}: {n_neg}/{n_tot} negative (COVID<CONTROL), "
          f"{n_tot-n_neg}/{n_tot} positive (COVID>CONTROL)")

# ============================================================
# SAVE
# ============================================================
out_fisher = os.path.join(OUT_DIR, "yeo_inference_fisher.csv")
out_raw    = os.path.join(OUT_DIR, "yeo_inference_raw.csv")
results["fisher"].to_csv(out_fisher, index=False)
results["raw"].to_csv(out_raw, index=False)
print(f"\nSaved: {out_fisher}")
print(f"Saved: {out_raw}")