"""
Step 4e: Global mean-FC sanity check (Family B validity check, not a test
family — no FDR, not corrected against the within-7/between-21 families).

In: config.atlas_dir("schaefer400", "step2_pipeline")/comet_matrices.
Out: global_mean_fc_per_subject.csv, global_mean_fc_inference.csv,
     global_mean_fc_boxplot.png.

Global mean FC per subject = mean over all 79,800 unique edges (upper
triangle). Reuses freedman_lane_permutation/cohens_d from step3d_auc_pipeline
(R2 ②: age+sex adjusted, same covariate model as the network-level tests it
contextualizes).
"""

import sys
from pathlib import Path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
import config

def _find_step3d_dir(root):
    for cand in sorted(root.iterdir()):
        if cand.is_dir() and (cand / "step3d_auc_pipeline.py").exists():
            return cand
    raise FileNotFoundError(f"step3d_auc_pipeline.py not found under {root}")
sys.path.insert(0, str(_find_step3d_dir(_HERE.parents[1])))
from step3d_auc_pipeline import freedman_lane_permutation, cohens_d, load_covariates

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================
MATRIX_DIR = config.atlas_dir("schaefer400", "step2_pipeline") / "comet_matrices"
OUT_DIR    = config.ensure(config.atlas_dir("schaefer400", "step4e_global_meanfc"))

N_ROIS         = 400
FISHER_CLIP    = config.FISHER_CLIP
N_PERMUTATIONS = config.N_PERMUTATIONS
SEED           = config.SEED
GROUP_A, GROUP_B = config.GROUP_ORDER

# ============================================================
# COHORT + GLOBAL MEAN FC PER SUBJECT
# ============================================================
df_csv = pd.read_csv(config.GROUP_CSV)
subjects = config.select_included_subjects(
    [p.name for p in config.NII_ROOT.iterdir() if p.is_dir()],
    df_csv, id_col="ID", group_col="Grupo", verbose=False)
print(f"Processing {len(subjects)} subjects (cohort via config)")

missing = [s for s in subjects
           if not os.path.exists(os.path.join(MATRIX_DIR, f"{s}_connectivity_comet.npy"))]
assert not missing, f"Missing FC matrices for config cohort: {missing}"

group_map = {str(i).strip(): str(g).strip()
             for i, g in zip(df_csv["ID"], df_csv["Grupo"])}
iu = np.triu_indices(N_ROIS, k=1)   # 79,800 unique edges

rows = []
for subj in subjects:
    mat = np.load(os.path.join(MATRIX_DIR, f"{subj}_connectivity_comet.npy"))
    np.fill_diagonal(mat, 0.0)
    edges = mat[iu]
    mean_raw    = float(np.mean(edges))
    mean_fisher = float(np.mean(np.arctanh(np.clip(edges, -FISHER_CLIP, FISHER_CLIP))))
    rows.append({"subject_id": subj, "group": group_map[subj],
                 "mean_fc_global_raw": mean_raw, "mean_fc_global_fisher": mean_fisher})

df_global = pd.DataFrame(rows)
assert df_global["group"].isna().sum() == 0
n_covid   = int((df_global["group"] == GROUP_B).sum())
n_control = int((df_global["group"] == GROUP_A).sum())
print(f"Groups: {df_global['group'].value_counts().to_dict()}")

# Hard cohort guard (full frozen cohort; covariates adjust, they do not drop).
assert len(subjects) == 162 and n_covid == 123 and n_control == 39, \
    "cohort deviates from frozen 162 (123/39)"

# --- R2 ②: attach the covariate model (age, sex), same loader/model as A/B/C ---
_cov, _sex_map = load_covariates(sorted(df_global["subject_id"]))
df_global = df_global.merge(_cov, left_on="subject_id", right_on="subject",
                            how="left", validate="one_to_one")
if df_global[["age", "sex_code"]].isna().any().any():
    _bad = sorted(df_global.loc[df_global[["age", "sex_code"]].isna().any(axis=1), "subject_id"])
    raise RuntimeError(f"covariate merge left NaNs for {_bad}")

# ============================================================
# DESCRIPTIVE
# ============================================================
print("\n=== Descriptive ===")
for col in ["mean_fc_global_raw", "mean_fc_global_fisher"]:
    print(f"\n{col}:")
    print(df_global.groupby("group")[col].agg(["mean", "std", "median", "min", "max"]).to_string())

# ============================================================
# INFERENCE — R2 ②: Freedman-Lane covariate-adjusted permutation (OLS group-
# coefficient t, age + sex adjusted). Parametric p of the same coefficient as
# sensitivity. NO FDR (validity check, not a test family).
# ============================================================
print("\n\n=== Inference (Freedman-Lane, age+sex adjusted; sanity check, no FDR) ===")
measures = ["mean_fc_global_fisher", "mean_fc_global_raw"]   # fisher primary first
substreams = np.random.SeedSequence(SEED).spawn(len(measures))  # one per measure

results = []
for i, col in enumerate(measures):
    g = df_global["group"].map({GROUP_A: 0, GROUP_B: 1}).values.astype(float)
    y = df_global[col].values.astype(float)
    Z = df_global[["age", "sex_code"]].values.astype(float)
    a, b = y[g == 0], y[g == 1]

    perm = freedman_lane_permutation(y, g, Z, N_PERMUTATIONS, substreams[i],
                                     se_type=config.FL_SE_TYPE)
    t_welch, p_welch = perm["t_obs"], perm["p_param"]   # adjusted group-coeff t + parametric p
    d, d_lo, d_hi = cohens_d(a, b)

    results.append({
        "measure": col, "aggregation": "fisher" if "fisher" in col else "raw",
        "mean_covid": b.mean(), "mean_control": a.mean(), "mean_diff": b.mean() - a.mean(),
        "cohens_d": d, "ci_lower": d_lo, "ci_upper": d_hi,
        "t_perm": perm["t_obs"], "p_perm": perm["p_perm"],
        "t_welch": t_welch, "p_welch": p_welch,
    })
    print(f"\n{col}:")
    print(f"  COVID={b.mean():.4f}  CONTROL={a.mean():.4f}  diff={b.mean()-a.mean():+.4f}")
    print(f"  Cohen's d: {d:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}]")
    print(f"  p_perm (primary): {perm['p_perm']:.4f}   p_param (sensitivity): {p_welch:.4f}")

df_inference = pd.DataFrame(results)

# ============================================================
# SAVE
# ============================================================
out_per_subject = os.path.join(OUT_DIR, "global_mean_fc_per_subject.csv")
out_inference   = os.path.join(OUT_DIR, "global_mean_fc_inference.csv")
df_global.to_csv(out_per_subject, index=False)
df_inference.to_csv(out_inference, index=False)
print(f"\nSaved: {out_per_subject}")
print(f"Saved: {out_inference}")

# ============================================================
# BOXPLOT
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, col, title in zip(
    axes, ["mean_fc_global_raw", "mean_fc_global_fisher"],
    ["Global Mean FC (raw r)", "Global Mean FC (Fisher-z)"]):
    data = [df_global.loc[df_global["group"] == GROUP_A, col].values,
            df_global.loc[df_global["group"] == GROUP_B, col].values]
    bp = ax.boxplot(data, labels=[GROUP_A, GROUP_B], patch_artist=True, showfliers=True)
    bp["boxes"][0].set_facecolor("#7fbf7f"); bp["boxes"][1].set_facecolor("#ff7f7f")
    ax.set_title(title); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "global_mean_fc_boxplot.png")
plt.savefig(plot_path, dpi=120, bbox_inches="tight"); plt.close()
print(f"Saved: {plot_path}")