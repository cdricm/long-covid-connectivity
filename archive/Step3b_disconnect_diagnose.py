"""Step 3b: Diagnose why graphs are disconnected.

Three questions:
  1. Are specific ROIs isolated in many subjects?
  2. Are specific subjects always disconnected (across all densities)?
  3. Is there a structural pattern (always same ROIs, or random)?
"""

import os, glob, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from joblib import Parallel, delayed

# ===== Config =================================================================
MAT_DIR  = "/mnt/d87cc26d-5470-443c-81c1-e09b68ee4730/Cedric/analysis_outputs/step2_nii_pipeline/comet_matrices"
CSV_PATH = "/mnt/d87cc26d-5470-443c-81c1-e09b68ee4730/Cedric/ResumenRespuestasBasico.csv"
OUT_BASE = "/mnt/d87cc26d-5470-443c-81c1-e09b68ee4730/Cedric/analysis_outputs"
OUT_DIR  = os.path.join(OUT_BASE, "step3b_disconnect_diagnose")

EXCLUDED_SUBJECTS = {"CP0004", "CP0011", "CP0015", "CP0087", "CP0106", "CP0140", "CP0144", "CP0193"}
COL_ID, COL_GROUP = "ID", "Grupo"

# Focus on the strategy we'll use for the main analysis (positive)
STRATEGY  = "positive"
# Densities to inspect (skip 50% - too noisy as reference; skip 1-2% - trivially disconnected)
DENSITIES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

N_JOBS = 4

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ===== Load group assignments =================================================
meta = pd.read_csv(CSV_PATH)[[COL_ID, COL_GROUP]].rename(columns={COL_ID:"subject", COL_GROUP:"group"})
meta = meta[meta["subject"].notna() & meta["group"].notna()]
meta["subject"] = meta["subject"].astype(str).str.strip()
meta["group"]   = meta["group"].astype(str).str.strip().str.upper()
meta = meta[~meta["subject"].isin(EXCLUDED_SUBJECTS)]

available = {os.path.basename(p).replace("_connectivity_comet.npy", "")
             for p in glob.glob(os.path.join(MAT_DIR, "CP*_connectivity_comet.npy"))}
meta = meta[meta["subject"].isin(available)].reset_index(drop=True)

subjects = meta["subject"].tolist()
group_of = dict(zip(meta["subject"], meta["group"]))
print(f"Diagnosing {len(subjects)} subjects, strategy={STRATEGY}, "
      f"densities={[int(d*100) for d in DENSITIES]}\n")

# ===== Threshold helpers (reuse from 3a) ======================================
def apply_strategy(C):
    out = np.where(C > 0, C, 0.0)
    np.fill_diagonal(out, 0.0)
    return out

def proportional_threshold(M, density):
    n  = M.shape[0]
    iu = np.triu_indices(n, k=1)
    w  = M[iu]
    n_target = int(round(density * w.size))
    n_keep   = min(n_target, int(np.sum(w > 0)))
    if n_keep == 0:
        return np.zeros_like(M)
    idx = np.argpartition(w, -n_keep)[-n_keep:]
    mask = np.zeros_like(w, dtype=bool)
    mask[idx] = True
    out = np.zeros_like(M)
    out[iu[0][mask], iu[1][mask]] = w[mask]
    return out + out.T

# ===== Per-subject diagnostics ================================================
def diagnose_subject(subj):
    C = np.load(os.path.join(MAT_DIR, f"{subj}_connectivity_comet.npy"))
    M = apply_strategy(C)
    n_roi = M.shape[0]
    rows  = []
    # Per density: which ROIs are singletons (degree=0)? Which are isolated outside main component?
    per_density = {}
    for d in DENSITIES:
        W   = proportional_threshold(M, d)
        deg = (W > 0).sum(axis=1)
        adj = csr_matrix((W > 0).astype(np.int8))
        n_comp, labels = connected_components(adj, directed=False)
        sizes = np.bincount(labels)
        largest_label = int(np.argmax(sizes))
        in_largest    = (labels == largest_label)
        # ROIs with no edges at all = degree-0 singletons
        singletons    = (deg == 0)
        # ROIs not in main component (singletons + small islands)
        not_in_main   = ~in_largest
        per_density[d] = {
            "n_components"  : int(n_comp),
            "largest_size"  : int(sizes.max()),
            "n_singletons"  : int(singletons.sum()),
            "n_not_in_main" : int(not_in_main.sum()),
            "isolated_rois" : np.where(not_in_main)[0].tolist(),
            "singleton_rois": np.where(singletons)[0].tolist(),
        }
    return subj, per_density

# ===== Run ====================================================================
t0 = time.time()
results = Parallel(n_jobs=N_JOBS, verbose=10)(
    delayed(diagnose_subject)(s) for s in subjects
)
print(f"\nDiagnosed in {time.time()-t0:.1f}s\n")

# ===== Q1: which ROIs are most often isolated? ===============================
n_roi = 400
roi_isolation_count = {d: np.zeros(n_roi, dtype=int) for d in DENSITIES}
roi_singleton_count = {d: np.zeros(n_roi, dtype=int) for d in DENSITIES}

for subj, per_d in results:
    for d in DENSITIES:
        for r in per_d[d]["isolated_rois"]:
            roi_isolation_count[d][r] += 1
        for r in per_d[d]["singleton_rois"]:
            roi_singleton_count[d][r] += 1

# Top problematic ROIs (most often isolated) at a reference density
ref_d = 0.20
top_isolated = np.argsort(-roi_isolation_count[ref_d])[:20]
top_singleton = np.argsort(-roi_singleton_count[ref_d])[:20]

# ===== Q2: which subjects are always disconnected? ===========================
subj_disconnect = []
for subj, per_d in results:
    n_disc = sum(1 for d in DENSITIES if per_d[d]["n_components"] > 1)
    max_islands = max(per_d[d]["n_not_in_main"] for d in DENSITIES)
    subj_disconnect.append({
        "subject"             : subj,
        "group"               : group_of[subj],
        "n_densities_disconn" : n_disc,
        "max_islands"         : max_islands,
        "n_comp_at_20pct"     : per_d[0.20]["n_components"],
        "n_comp_at_30pct"     : per_d[0.30]["n_components"],
        "isolated_at_20pct"   : per_d[0.20]["n_not_in_main"],
    })
subj_df = pd.DataFrame(subj_disconnect).sort_values("n_densities_disconn", ascending=False)
subj_df.to_csv(os.path.join(OUT_DIR, "step3b_subject_disconnect.csv"), index=False)

# ===== Q3: pattern - is the SAME ROI always the problem? =====================
# Compute fraction of subjects in which each ROI is isolated, across densities
roi_summary = pd.DataFrame({"roi": np.arange(n_roi)})
for d in DENSITIES:
    roi_summary[f"isolated_pct_{int(d*100)}"]  = roi_isolation_count[d] / len(subjects) * 100
    roi_summary[f"singleton_pct_{int(d*100)}"] = roi_singleton_count[d] / len(subjects) * 100
roi_summary.to_csv(os.path.join(OUT_DIR, "step3b_roi_isolation.csv"), index=False)

# ===== Plots ==================================================================
# Plot 1: ROI isolation frequency, sorted, per density
fig, ax = plt.subplots(figsize=(10, 5))
for d in DENSITIES:
    sorted_pct = np.sort(roi_isolation_count[d] / len(subjects) * 100)[::-1]
    ax.plot(sorted_pct, label=f"{int(d*100)}%")
ax.set_xlabel("ROI rank (sorted by isolation frequency, descending)")
ax.set_ylabel("% subjects in which this ROI is outside main component")
ax.set_title("ROI isolation frequency across subjects, per density")
ax.legend(title="Density")
ax.grid(alpha=0.3)
ax.axhline(50, color="red", linestyle=":", linewidth=0.8, label="50% threshold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "step3b_roi_isolation_curves.png"), dpi=140)
plt.close()

# Plot 2: per-subject disconnect severity histogram
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(subj_df["n_densities_disconn"], bins=range(0, len(DENSITIES)+2),
             color="steelblue", edgecolor="white", align="left")
axes[0].set_xlabel(f"# densities (out of {len(DENSITIES)}) where graph is disconnected")
axes[0].set_ylabel("# subjects")
axes[0].set_title("Per-subject disconnect severity")
axes[0].grid(alpha=0.3)

axes[1].hist(subj_df["isolated_at_20pct"], bins=30, color="steelblue", edgecolor="white")
axes[1].set_xlabel("# ROIs outside main component @ 20% density")
axes[1].set_ylabel("# subjects")
axes[1].set_title("Per-subject island size @ 20% density")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "step3b_subject_severity.png"), dpi=140)
plt.close()

# Plot 3: ROI isolation heatmap (top 30 most-isolated ROIs at 20%)
top30 = np.argsort(-roi_isolation_count[0.20])[:30]
heat  = np.array([roi_isolation_count[d][top30] / len(subjects) * 100 for d in DENSITIES])
fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(heat, aspect="auto", cmap="Reds", vmin=0, vmax=100)
ax.set_xticks(range(len(top30)))
ax.set_xticklabels([f"ROI {r}" for r in top30], rotation=90, fontsize=8)
ax.set_yticks(range(len(DENSITIES)))
ax.set_yticklabels([f"{int(d*100)}%" for d in DENSITIES])
ax.set_xlabel("ROI index")
ax.set_ylabel("Density")
ax.set_title("Top 30 most-isolated ROIs (% of subjects where ROI is outside main component)")
plt.colorbar(im, ax=ax, label="% subjects")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "step3b_roi_heatmap.png"), dpi=140)
plt.close()

# ===== Text summary ===========================================================
L = ["=" * 72, "STEP 3b - DISCONNECTEDNESS DIAGNOSE", "=" * 72,
     f"Strategy   : {STRATEGY}",
     f"Densities  : {[int(d*100) for d in DENSITIES]} (%)",
     f"N subjects : {len(subjects)}",
     ""]

# Q1: most-isolated ROIs
L.append(f"--- Q1: Are specific ROIs systematically isolated? ---")
L.append(f"Reference density: {int(ref_d*100)}%")
L.append(f"Top 20 most-often-isolated ROIs (sorted, % of subjects):")
for r in top_isolated:
    pct = roi_isolation_count[ref_d][r] / len(subjects) * 100
    if pct > 5:  # only show non-trivial ones
        L.append(f"  ROI {r:>3}: {pct:>5.1f}% of subjects (singleton in "
                 f"{roi_singleton_count[ref_d][r]} subjects)")
L.append("")
n_chronic = int((roi_isolation_count[ref_d] / len(subjects) > 0.5).sum())
L.append(f"ROIs isolated in >50% of subjects at {int(ref_d*100)}%: {n_chronic}")
n_chronic_30 = int((roi_isolation_count[0.30] / len(subjects) > 0.5).sum())
L.append(f"ROIs isolated in >50% of subjects at 30%: {n_chronic_30}")
L.append("")

# Q2: chronic-disconnect subjects
L.append(f"--- Q2: Are specific subjects always disconnected? ---")
all_disc = int((subj_df["n_densities_disconn"] == len(DENSITIES)).sum())
none_disc = int((subj_df["n_densities_disconn"] == 0).sum())
L.append(f"Subjects disconnected at ALL {len(DENSITIES)} densities: {all_disc}")
L.append(f"Subjects never disconnected                : {none_disc}")
L.append(f"Subjects disconnected at 30% but not lower : "
         f"{int((subj_df['n_comp_at_30pct'] > 1).sum())}")
L.append("")
L.append("Top 10 worst subjects (most disconnected):")
for _, r in subj_df.head(10).iterrows():
    L.append(f"  {r['subject']} ({r['group']}): "
             f"disc at {r['n_densities_disconn']}/{len(DENSITIES)} densities, "
             f"max islands={r['max_islands']}, "
             f"comp@20%={r['n_comp_at_20pct']}, comp@30%={r['n_comp_at_30pct']}")
L.append("")

# Q3: pattern interpretation
L.append(f"--- Q3: Is it a systematic ROI problem or subject-level noise? ---")
top10_share_20 = (roi_isolation_count[0.20][top_isolated[:10]] / len(subjects) * 100)
if top10_share_20.mean() > 50:
    pattern = "SYSTEMATIC: Top 10 ROIs are isolated in >50% of subjects on average"
    L.append(f"  -> {pattern}")
    L.append(f"     The problem is concentrated in specific ROIs.")
    L.append(f"     Likely fix: exclude these ROIs from analysis or use a")
    L.append(f"     fragmentation-tolerant metric (e.g. Global Efficiency).")
elif top10_share_20.mean() > 20:
    pattern = "PARTIAL: Some ROIs are commonly isolated, but not in a majority of subjects"
    L.append(f"  -> {pattern}")
    L.append(f"     Mixed pattern. Some ROIs and some subjects contribute.")
else:
    pattern = "DIFFUSE: Isolation is spread across many ROIs, not concentrated"
    L.append(f"  -> {pattern}")
    L.append(f"     The disconnectedness is not driven by specific 'bad' ROIs.")
    L.append(f"     Likely cause: heterogeneity of FC structure across subjects.")
L.append(f"  Average isolation pct of top-10 ROIs at 20%: {top10_share_20.mean():.1f}%")

# Schaefer-7 network mapping note
L.append("")
L.append("--- Note ---")
L.append("ROI indices refer to Schaefer-2018 400-parcel 7-networks order.")
L.append("Top-isolated ROIs can be mapped to network labels via atlas.labels")
L.append("(e.g., to check whether they cluster in cerebellum/subcortex/sensory)")

summary = "\n".join(L)
print(summary)
with open(os.path.join(OUT_DIR, "step3b_summary.txt"), "w") as f:
    f.write(summary)

print("\nOutputs:")
for fn in sorted(os.listdir(OUT_DIR)):
    print(f"  {fn}")