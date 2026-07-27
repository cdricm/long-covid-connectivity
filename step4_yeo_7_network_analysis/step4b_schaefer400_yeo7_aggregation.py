"""
Step 4b: Aggregate Schaefer-400 FC matrices into Yeo-7 within/between values.

In: FC matrices (400x400) from step2_pipeline/comet_matrices/ (frozen N=162
    cohort), Yeo-network mapping from step4a_labels/schaefer400_yeo7_roi_info.csv,
    group assignment from config.GROUP_CSV.
Out: yeo_fc_fisher.csv (Fisher-z aggregation, z-then-mean, PRIMARY),
     yeo_fc_raw.csv (direct r mean, SENSITIVITY) — one row per subject,
     columns = group + 7 within + 21 between.

Fisher-z (arctanh) is applied PER EDGE BEFORE averaging (z-then-mean), never
mean-then-z. within = mean over the upper-triangular edges of the network
block; between = mean over the full rectangular A x B block (disjoint index
sets, so each edge appears exactly once). Cohort via
config.select_included_subjects(); 4b is the source from which 4c/4d inherit
the subject set.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import os
import numpy as np
import pandas as pd
from itertools import combinations

# ============================================================
# SETTINGS
# ============================================================
CSV_PATH      = config.GROUP_CSV
MATRIX_DIR    = config.atlas_dir("schaefer400", "step2_pipeline") / "comet_matrices"
ROI_INFO_PATH = config.atlas_dir("schaefer400", "step4a_labels") / "schaefer400_yeo7_roi_info.csv"
OUT_DIR       = config.ensure(config.atlas_dir("schaefer400", "step4b_aggregation"))

YEO_NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
FISHER_CLIP  = config.FISHER_CLIP   # avoids inf at r=1.0 (r=0.9999 -> z=4.95)

ID_COL    = "ID"
GROUP_COL = "Grupo"

# ============================================================
# LOAD ROI INFO
# ============================================================
roi_info = pd.read_csv(ROI_INFO_PATH)
print(f"ROI info loaded: {len(roi_info)} ROIs")
assert len(roi_info) == 400, f"Expected 400 ROIs, got {len(roi_info)}"
assert list(roi_info["roi_idx"]) == list(range(400)), "roi_idx must be 0..399 in order"

network_indices = {
    net: roi_info.loc[roi_info["yeo_network"] == net, "roi_idx"].values
    for net in YEO_NETWORKS
}
for net, ids in network_indices.items():
    print(f"  {net}: {len(ids)} ROIs")

# ============================================================
# COHORT (single source of truth: config) + group map + verify matrices
# ============================================================
df_csv = pd.read_csv(CSV_PATH)
subjects_used = config.select_included_subjects(
    [p.name for p in config.NII_ROOT.iterdir() if p.is_dir()],
    df_csv, id_col=ID_COL, group_col=GROUP_COL, verbose=True)
print(f"\nCohort via config: {len(subjects_used)} subjects")

group_map = {str(i).strip(): str(g).strip()
             for i, g in zip(df_csv[ID_COL], df_csv[GROUP_COL])}

missing = [s for s in subjects_used
           if not os.path.exists(os.path.join(MATRIX_DIR, f"{s}_connectivity_comet.npy"))]
assert not missing, f"Missing FC matrices for config cohort: {missing}"

# ============================================================
# AGGREGATION HELPERS (z-then-mean)
# ============================================================
def aggregate_within(mat, indices, use_fisher):
    """Mean FC within a network (upper triangle, k=1). z-then-mean if use_fisher."""
    sub = mat[np.ix_(indices, indices)]
    iu = np.triu_indices(len(indices), k=1)
    edges = sub[iu]
    if use_fisher:
        edges = np.arctanh(np.clip(edges, -FISHER_CLIP, FISHER_CLIP))
    return np.mean(edges)

def aggregate_between(mat, indices_a, indices_b, use_fisher):
    """Mean FC between two networks (full rectangular block; disjoint sets ->
    each edge once). z-then-mean if use_fisher."""
    edges = mat[np.ix_(indices_a, indices_b)].flatten()
    if use_fisher:
        edges = np.arctanh(np.clip(edges, -FISHER_CLIP, FISHER_CLIP))
    return np.mean(edges)

# ============================================================
# COLUMN NAMES
# ============================================================
within_cols   = [f"within_{net}" for net in YEO_NETWORKS]
between_pairs = list(combinations(YEO_NETWORKS, 2))   # 21 pairs
between_cols  = [f"between_{a}_{b}" for a, b in between_pairs]
all_fc_cols   = within_cols + between_cols
print(f"\nTotal FC columns: {len(all_fc_cols)} (7 within + 21 between)")

# ============================================================
# PROCESS SUBJECTS
# ============================================================
rows_fisher, rows_raw = [], []

for subj in subjects_used:
    mat = np.load(os.path.join(MATRIX_DIR, f"{subj}_connectivity_comet.npy"))
    assert mat.shape == (400, 400), f"{subj}: shape {mat.shape}"
    np.fill_diagonal(mat, 0.0)
    if not np.all(np.isfinite(mat)):
        raise ValueError(f"{subj}: matrix contains NaN/Inf")

    for use_fisher, out_list in [(True, rows_fisher), (False, rows_raw)]:
        row = {"subject_id": subj, "group": group_map[subj]}
        for net in YEO_NETWORKS:
            row[f"within_{net}"] = aggregate_within(mat, network_indices[net], use_fisher)
        for net_a, net_b in between_pairs:
            row[f"between_{net_a}_{net_b}"] = aggregate_between(
                mat, network_indices[net_a], network_indices[net_b], use_fisher)
        out_list.append(row)

df_fisher = pd.DataFrame(rows_fisher)
df_raw    = pd.DataFrame(rows_raw)
print(f"\nFisher-z dataframe: {df_fisher.shape}")
print(f"Raw dataframe: {df_raw.shape}")
print(f"Groups: {df_fisher['group'].value_counts().to_dict()}")

# ============================================================
# VERIFY OUTPUT STRUCTURE
# ============================================================
print("\n--- Fisher-z: first 3 rows, first 5 FC columns ---")
print(df_fisher[["subject_id", "group"] + all_fc_cols[:4]].head(3).to_string(index=False))

print("\n--- Fisher-z: FC value range ---")
print(f"min: {df_fisher[all_fc_cols].min().min():.4f}  max: {df_fisher[all_fc_cols].max().max():.4f}")
print("--- Raw: FC value range ---")
print(f"min: {df_raw[all_fc_cols].min().min():.4f}  max: {df_raw[all_fc_cols].max().max():.4f}")

print("\n--- Fisher-z: column means ---")
print(df_fisher[all_fc_cols].mean().to_string())

# ============================================================
# SAVE
# ============================================================
out_fisher = os.path.join(OUT_DIR, "yeo_fc_fisher.csv")
out_raw    = os.path.join(OUT_DIR, "yeo_fc_raw.csv")
df_fisher.to_csv(out_fisher, index=False)
df_raw.to_csv(out_raw, index=False)
print(f"\nSaved: {out_fisher}")
print(f"Saved: {out_raw}")