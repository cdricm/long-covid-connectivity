"""Step 3c: Graph metrics sweep across densities (5-50%) - FIXED VERSION.

Bug found in previous run: cg.threshold() ignored the threshold argument and always
produced ~10% density. Replaced with explicit numpy thresholding from step 3a.

Metrics (all from cg.bct, verified to work and vary with density):
  - Global Efficiency       : cg.bct.efficiency_wei
  - Modularity Q (Louvain)  : cg.bct.modularity_louvain_und
  - Mean Clustering         : cg.bct.clustering_coef_wu  (mean of per-node)
  - Assortativity (weighted): cg.bct.assortativity_wei
"""

import os, glob, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import Parallel, delayed

# Suppress harmless RuntimeWarnings from BCT on edge cases (NaN handling we already do)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ===== Config =================================================================
MAT_DIR  = "/mnt/d87cc26d-5470-443c-81c1-e09b68ee4730/Cedric/analysis_outputs/step2_nii_pipeline/comet_matrices"
CSV_PATH = "/mnt/d87cc26d-5470-443c-81c1-e09b68ee4730/Cedric/ResumenRespuestasBasico.csv"
OUT_BASE = "/mnt/d87cc26d-5470-443c-81c1-e09b68ee4730/Cedric/analysis_outputs"
OUT_DIR  = os.path.join(OUT_BASE, "step3c_metrics_sweep")

EXCLUDED_SUBJECTS = {"CP0004", "CP0011", "CP0015", "CP0087", "CP0106", "CP0140", "CP0144", "CP0193"}
COL_ID, COL_GROUP = "ID", "Grupo"

DENSITIES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
STRATEGY  = "positive"

# Modularity Louvain is stochastic; set seed for reproducibility
LOUVAIN_SEED = 42

N_JOBS = 12

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ===== Imports ================================================================
import comet
from comet.graph import bct as bct

print(f"COMET {getattr(comet, '__version__', '1.2.4')}\n")

# ===== Load groups ============================================================
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
print(f"Processing {len(subjects)} subjects x {len(DENSITIES)} densities = "
      f"{len(subjects)*len(DENSITIES)} datapoints\n")

# ===== Own thresholding (numpy, verified in step 3a) ==========================
def prep_and_threshold(C, density):
    """Positive strategy + proportional threshold by edge count. Symmetric."""
    M = np.where(C > 0, C, 0.0).astype(np.float64)
    np.fill_diagonal(M, 0.0)
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

# ===== Metrics ================================================================
def compute_metrics(W):
    out = {}
    # Global Efficiency (weighted)
    try:
        out["global_efficiency"] = float(bct.efficiency_wei(W))
    except Exception as e:
        out["global_efficiency"] = np.nan
        out["err_ge"] = str(e)[:120]
    # Modularity Q via Louvain
    try:
        _, Q = bct.modularity_louvain_und(W, seed=LOUVAIN_SEED)
        out["modularity_q"] = float(Q)
    except Exception as e:
        out["modularity_q"] = np.nan
        out["err_mod"] = str(e)[:120]
    # Mean clustering coefficient (weighted, undirected)
    try:
        cc = bct.clustering_coef_wu(W)
        out["mean_clustering"] = float(np.nanmean(cc))
    except Exception as e:
        out["mean_clustering"] = np.nan
        out["err_clust"] = str(e)[:120]
    # Assortativity (weighted)
    try:
        out["assortativity"] = float(bct.assortativity_wei(W))
    except Exception as e:
        out["assortativity"] = np.nan
        out["err_assort"] = str(e)[:120]
    return out

def process_subject(subj):
    C = np.load(os.path.join(MAT_DIR, f"{subj}_connectivity_comet.npy"))
    rows = []
    for d in DENSITIES:
        W = prep_and_threshold(C, d)
        n_edges = int(np.sum(W > 0) // 2)
        t0 = time.time()
        m = compute_metrics(W)
        rows.append({
            "subject": subj, "group": group_of[subj],
            "density": d, "n_edges": n_edges,
            "t_compute": round(time.time() - t0, 2),
            **m,
        })
    return rows

# ===== Sanity check ===========================================================
print("Sanity check on first subject across densities...")
t0 = time.time()
test = process_subject(subjects[0])
print(f"  Done in {time.time()-t0:.1f}s for {len(DENSITIES)} densities")
print(f"  Subject: {subjects[0]}")
print(f"  {'density':>8}  {'n_edges':>8}  {'GE':>8}  {'Q':>8}  {'clust':>8}  {'assort':>8}")
for r in test:
    print(f"  {int(r['density']*100):>7}%  {r['n_edges']:>8}  "
          f"{r['global_efficiency']:>8.4f}  {r['modularity_q']:>8.4f}  "
          f"{r['mean_clustering']:>8.4f}  {r['assortativity']:>8.4f}")
# Verify metrics actually vary
ge_range = max(r['global_efficiency'] for r in test) - min(r['global_efficiency'] for r in test)
if ge_range < 0.001:
    print("\n  WARNING: Global Efficiency is constant across densities!")
    print("  Stop here and investigate.")
    raise SystemExit(1)
print(f"\n  Global Efficiency varies by {ge_range:.4f} across densities -> looks correct.\n")

# ===== Run full sweep =========================================================
print(f"Running full sweep with n_jobs={N_JOBS}...")
t_start = time.time()
results = Parallel(n_jobs=N_JOBS, verbose=10)(
    delayed(process_subject)(s) for s in subjects
)
flat = [r for sub in results for r in sub]
df = pd.DataFrame(flat)
runtime = time.time() - t_start
print(f"\nDone in {runtime/60:.1f} min")

df.to_csv(os.path.join(OUT_DIR, "step3c_metrics.csv"), index=False)

# ===== Aggregation ============================================================
METRICS = ["global_efficiency", "modularity_q", "mean_clustering", "assortativity"]
agg = df.groupby(["density", "group"])[METRICS].agg(["mean", "std", "median"]).round(4)
agg.to_csv(os.path.join(OUT_DIR, "step3c_aggregated.csv"))

# ===== Plot 1: metric curves per group ========================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, metric in zip(axes.flat, METRICS):
    for g in ("CONTROL", "COVID"):
        sub = df[df["group"] == g]
        m   = sub.groupby("density")[metric].agg(["mean", "std"]).reset_index()
        x   = m["density"] * 100
        ax.plot(x, m["mean"], marker="o", label=g)
        ax.fill_between(x, m["mean"] - m["std"], m["mean"] + m["std"], alpha=0.15)
    ax.set_xlabel("Density (%)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(metric.replace("_", " "))
    ax.grid(alpha=0.3)
    ax.legend()
fig.suptitle("Graph metrics across densities (mean ± SD per group)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "step3c_metric_curves.png"), dpi=140)
plt.close()

# ===== Plot 2: Cohen's d per density ==========================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
effect_summary = []
for ax, metric in zip(axes.flat, METRICS):
    diff = []
    for d in DENSITIES:
        ctrl  = df[(df["group"] == "CONTROL") & (df["density"] == d)][metric].dropna()
        covid = df[(df["group"] == "COVID")   & (df["density"] == d)][metric].dropna()
        if len(ctrl) > 1 and len(covid) > 1:
            pooled = np.sqrt(((ctrl.var()*(len(ctrl)-1)) + (covid.var()*(len(covid)-1))) /
                             (len(ctrl) + len(covid) - 2))
            d_eff = (covid.mean() - ctrl.mean()) / pooled if pooled > 0 else np.nan
        else:
            d_eff = np.nan
        diff.append({"density": d, "cohen_d": d_eff})
        effect_summary.append({"metric": metric, "density": d, "cohen_d": d_eff,
                               "ctrl_mean": ctrl.mean(), "covid_mean": covid.mean()})
    diff_df = pd.DataFrame(diff)
    ax.axhline(0,  color="grey",   linestyle=":",  linewidth=0.8)
    ax.axhline(0.2,color="orange", linestyle=":",  linewidth=0.6)
    ax.axhline(-0.2,color="orange",linestyle=":",  linewidth=0.6)
    ax.plot(diff_df["density"] * 100, diff_df["cohen_d"], marker="o", color="darkred")
    ax.set_xlabel("Density (%)")
    ax.set_ylabel("Cohen's d (COVID − CONTROL)")
    ax.set_title(metric.replace("_", " "))
    ax.grid(alpha=0.3)
fig.suptitle("Group effect size across densities", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "step3c_effect_sizes.png"), dpi=140)
plt.close()

effect_df = pd.DataFrame(effect_summary)

# ===== Text summary ===========================================================
L = ["=" * 72, "STEP 3c - GRAPH METRICS SWEEP (FIXED)", "=" * 72,
     f"Strategy   : {STRATEGY} (own numpy threshold, COMET cg.threshold had bug)",
     f"Densities  : {[int(d*100) for d in DENSITIES]} (%)",
     f"Metrics    : {METRICS}",
     f"N subjects : {len(subjects)} ({sum(1 for s in subjects if group_of[s]=='CONTROL')} CONTROL, "
     f"{sum(1 for s in subjects if group_of[s]=='COVID')} COVID)",
     f"Runtime    : {runtime/60:.1f} min",
     ""]

for m in METRICS:
    n_nan = int(df[m].isna().sum())
    if n_nan > 0:
        L.append(f"WARNING: {n_nan} NaN values in {m}")

L.append("\n--- Metric ranges (across all subjects and densities) ---")
for m in METRICS:
    v = df[m].dropna()
    L.append(f"  {m:<22}: [{v.min():+.4f}, {v.max():+.4f}], "
             f"mean={v.mean():+.4f}, SD={v.std():.4f}")

L.append("\n--- Effect size per density (Cohen's d) ---")
for m in METRICS:
    L.append(f"\n  {m}:")
    for d in DENSITIES:
        row = effect_df[(effect_df.metric==m) & (effect_df.density==d)].iloc[0]
        marker = " *" if abs(row["cohen_d"]) >= 0.2 else "  "
        L.append(f"   {int(d*100):>3}%: d={row['cohen_d']:+.3f}{marker}  "
                 f"(CTRL μ={row['ctrl_mean']:+.4f}, COVID μ={row['covid_mean']:+.4f})")

L.append("\n--- Suggested AUC range (from effect-size stability) ---")
for m in METRICS:
    sub = effect_df[effect_df.metric == m].dropna(subset=["cohen_d"])
    if len(sub) == 0: continue
    peak_d   = sub.loc[sub["cohen_d"].abs().idxmax(), "density"]
    peak_val = sub.loc[sub["cohen_d"].abs().idxmax(), "cohen_d"]
    L.append(f"  {m}: |d| peaks at {int(peak_d*100)}% (d={peak_val:+.3f})")

L.append("\n--- Next step ---")
L.append("Inspect step3c_effect_sizes.png to choose AUC range.")
L.append("Typical rule: include densities where effect size is stable (not in saturation).")
L.append("Densities where Cohen's |d| >= 0.2 are marked with * above.")

summary = "\n".join(L)
print(summary)
with open(os.path.join(OUT_DIR, "step3c_summary.txt"), "w") as f:
    f.write(summary)

print("\nOutputs:")
for fn in sorted(os.listdir(OUT_DIR)):
    print(f"  {fn}")