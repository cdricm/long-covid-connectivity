"""Step 3a: Threshold sweep diagnostics across negative-weight strategies."""

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
OUT_DIR  = os.path.join(OUT_BASE, "step3a_threshold_sweep")

# Exclusions (CP0140 added: 117 TPs deviates from expected 140/200)
EXCLUDED_SUBJECTS = {"CP0004", "CP0011", "CP0015", "CP0087", "CP0106", "CP0140", "CP0144", "CP0193"}

# CSV column names
COL_ID    = "ID"
COL_GROUP = "Grupo"
GROUPS    = ("CONTROL", "COVID")

# Sweep grid
DENSITIES  = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
STRATEGIES = ("positive", "negative", "absolute")

N_JOBS = 12

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ===== Load group assignments =================================================
print("Loading group CSV...")
meta = pd.read_csv(CSV_PATH)
meta = meta[[COL_ID, COL_GROUP]].rename(columns={COL_ID: "subject", COL_GROUP: "group"})
meta = meta[meta["subject"].notna() & meta["group"].notna()]
meta["subject"] = meta["subject"].astype(str).str.strip()
meta["group"]   = meta["group"].astype(str).str.strip().str.upper()

# Apply exclusions
meta = meta[~meta["subject"].isin(EXCLUDED_SUBJECTS)]
print(f"  CSV entries (after exclusion) : {len(meta)}")
print(f"  Group distribution: \n{meta['group'].value_counts().to_string()}")

# Match against available matrix files
available = {os.path.basename(p).replace("_connectivity_comet.npy", "")
             for p in glob.glob(os.path.join(MAT_DIR, "CP*_connectivity_comet.npy"))}
meta = meta[meta["subject"].isin(available)].reset_index(drop=True)
n_missing_mat = (~meta["subject"].isin(available)).sum()
print(f"  Subjects with matrix available: {len(meta)}")
if n_missing_mat > 0:
    print(f"  Warning: {n_missing_mat} subjects in CSV but no matrix file")

# Sanity: also check matrices without CSV entry
matrices_without_meta = available - set(meta["subject"]) - EXCLUDED_SUBJECTS
if matrices_without_meta:
    print(f"  Warning: {len(matrices_without_meta)} matrices without CSV row: "
          f"{sorted(matrices_without_meta)[:5]}...")

subjects = meta["subject"].tolist()
group_of = dict(zip(meta["subject"], meta["group"]))
print()

# ===== Threshold helpers ======================================================
def apply_negative_strategy(C, strategy):
    """Return matrix prepared for thresholding by strategy. Diagonal kept at 0."""
    if strategy == "positive":
        out = np.where(C > 0, C, 0.0)
    elif strategy == "negative":
        # Keep negative edges, flip sign so they become positive weights for thresholding
        out = np.where(C < 0, -C, 0.0)
    elif strategy == "absolute":
        out = np.abs(C)
    else:
        raise ValueError(strategy)
    np.fill_diagonal(out, 0.0)
    return out

def proportional_threshold(M, density):
    """Keep top-`density` fraction of off-diagonal edges by weight. Symmetric."""
    n = M.shape[0]
    iu = np.triu_indices(n, k=1)
    weights = M[iu]
    n_edges_total   = weights.size
    n_edges_target  = int(round(density * n_edges_total))
    n_nonzero       = int(np.sum(weights > 0))
    # cap target at available non-zero edges
    n_edges_keep = min(n_edges_target, n_nonzero)
    if n_edges_keep == 0:
        return np.zeros_like(M), 0, n_edges_target

    # Find threshold = (n_edges_total - n_edges_keep)-th smallest value
    idx_keep   = np.argpartition(weights, -n_edges_keep)[-n_edges_keep:]
    keep_mask  = np.zeros_like(weights, dtype=bool)
    keep_mask[idx_keep] = True

    out = np.zeros_like(M)
    out[iu[0][keep_mask], iu[1][keep_mask]] = weights[keep_mask]
    out = out + out.T
    return out, n_edges_keep, n_edges_target

def graph_diagnostics(W):
    """Compute connectedness, n_components, largest component size, mean weight."""
    n = W.shape[0]
    adj = (W > 0).astype(np.int8)
    sp  = csr_matrix(adj)
    n_comp, labels = connected_components(sp, directed=False)
    sizes = np.bincount(labels)
    largest = int(sizes.max())
    mean_w  = float(W[W > 0].mean()) if (W > 0).any() else 0.0
    return {
        "n_components"      : int(n_comp),
        "largest_component" : largest,
        "connected"         : bool(n_comp == 1),
        "mean_edge_weight"  : mean_w,
    }

# ===== Per-subject sweep worker ===============================================
def sweep_subject(subj):
    C = np.load(os.path.join(MAT_DIR, f"{subj}_connectivity_comet.npy"))
    rows = []
    for strategy in STRATEGIES:
        M = apply_negative_strategy(C, strategy)
        for density in DENSITIES:
            W, n_kept, n_target = proportional_threshold(M, density)
            diag = graph_diagnostics(W)
            rows.append({
                "subject"          : subj,
                "group"            : group_of[subj],
                "strategy"         : strategy,
                "density"          : density,
                "n_edges_target"   : n_target,
                "n_edges_kept"     : n_kept,
                "target_reached"   : (n_kept == n_target),
                **diag,
            })
    return rows

# ===== Run sweep ==============================================================
t_start = time.time()
print(f"Running sweep: {len(subjects)} subjects x {len(STRATEGIES)} strategies "
      f"x {len(DENSITIES)} densities = {len(subjects)*len(STRATEGIES)*len(DENSITIES)} datapoints")
print(f"Using n_jobs={N_JOBS}\n")

results = Parallel(n_jobs=N_JOBS, verbose=10)(
    delayed(sweep_subject)(s) for s in subjects
)
flat = [r for sublist in results for r in sublist]
df = pd.DataFrame(flat)

runtime = time.time() - t_start
print(f"\nSweep done in {runtime:.1f}s")
df.to_csv(os.path.join(OUT_DIR, "step3a_sweep.csv"), index=False)

# ===== Summary table ==========================================================
agg = df.groupby(["strategy", "density"]).agg(
    n_subjects        = ("subject", "count"),
    n_connected       = ("connected", "sum"),
    pct_connected     = ("connected", lambda x: 100 * x.mean()),
    median_n_comp     = ("n_components", "median"),
    max_n_comp        = ("n_components", "max"),
    median_largest    = ("largest_component", "median"),
    mean_edges_kept   = ("n_edges_kept", "mean"),
    pct_target_reach  = ("target_reached", lambda x: 100 * x.mean()),
).round(2)
agg.to_csv(os.path.join(OUT_DIR, "step3a_sweep_summary.csv"))

# Per-group aggregation (for plots)
agg_grp = df.groupby(["strategy", "density", "group"]).agg(
    n              = ("subject", "count"),
    pct_connected  = ("connected", lambda x: 100 * x.mean()),
    median_n_comp  = ("n_components", "median"),
    mean_weight    = ("mean_edge_weight", "mean"),
).reset_index()

# ===== Plot 1: connectedness ==================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, strat in zip(axes, STRATEGIES):
    for g in GROUPS:
        sub = agg_grp[(agg_grp["strategy"] == strat) & (agg_grp["group"] == g)]
        ax.plot(sub["density"] * 100, sub["pct_connected"], marker="o", label=g)
    ax.set_title(f"{strat}")
    ax.set_xlabel("Density (%)")
    ax.axhline(100, color="grey", linestyle=":", linewidth=0.8)
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("% subjects with connected graph")
axes[-1].legend()
fig.suptitle("Connectedness across thresholds", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "step3a_connectedness.png"), dpi=140)
plt.close()

# ===== Plot 2: component count distribution (boxplot) =========================
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, strat in zip(axes, STRATEGIES):
    data_by_density = [df[(df["strategy"] == strat) & (df["density"] == d)]["n_components"].values
                       for d in DENSITIES]
    ax.boxplot(data_by_density, labels=[f"{int(d*100)}" for d in DENSITIES])
    ax.set_title(f"{strat}")
    ax.set_xlabel("Density (%)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("# components (log)")
fig.suptitle("Number of components across thresholds", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "step3a_components.png"), dpi=140)
plt.close()

# ===== Plot 3: mean edge weight per group =====================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
for ax, strat in zip(axes, STRATEGIES):
    for g in GROUPS:
        sub = agg_grp[(agg_grp["strategy"] == strat) & (agg_grp["group"] == g)]
        ax.plot(sub["density"] * 100, sub["mean_weight"], marker="o", label=g)
    ax.set_title(f"{strat}")
    ax.set_xlabel("Density (%)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Mean edge weight (group average)")
axes[-1].legend()
fig.suptitle("Mean kept-edge weight across thresholds", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "step3a_edgeweights.png"), dpi=140)
plt.close()

# ===== Plot 4: target-reached heatmap (negative strategy is the failure mode) ==
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, strat in zip(axes, STRATEGIES):
    sub = df[df["strategy"] == strat].pivot(index="subject", columns="density",
                                            values="target_reached").astype(float)
    im = ax.imshow(sub.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(DENSITIES)))
    ax.set_xticklabels([f"{int(d*100)}%" for d in DENSITIES])
    ax.set_yticks([])
    ax.set_xlabel("Density")
    ax.set_title(f"{strat}\n(green=target reached, red=insufficient edges)")
axes[0].set_ylabel("Subjects")
fig.suptitle("Was the requested edge count achievable?", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "step3a_target_reached.png"), dpi=140)
plt.close()

# ===== Text summary ===========================================================
L = ["=" * 72, "STEP 3a - THRESHOLD SWEEP SUMMARY", "=" * 72,
     f"N subjects     : {len(subjects)}",
     f"Groups         : {dict(meta['group'].value_counts())}",
     f"Densities      : {[int(d*100) for d in DENSITIES]} (%)",
     f"Strategies     : {STRATEGIES}",
     f"Runtime        : {runtime:.1f}s",
     "",
     "--- Connectedness across strategies/densities ---"]
for strat in STRATEGIES:
    L.append(f"\n  Strategy: {strat}")
    for d in DENSITIES:
        row = agg.loc[(strat, d)]
        L.append(f"    {int(d*100):>3}% density : "
                 f"{int(row['n_connected']):>3}/{int(row['n_subjects'])} connected "
                 f"({row['pct_connected']:>5.1f}%), "
                 f"median #comp={row['median_n_comp']:.0f}, "
                 f"target-reach={row['pct_target_reach']:.0f}%")

L += ["", "--- Key observations ---"]
# Find lowest density where ≥95% connected per strategy
for strat in STRATEGIES:
    found = None
    for d in DENSITIES:
        if agg.loc[(strat, d), "pct_connected"] >= 95.0:
            found = d
            break
    if found is not None:
        L.append(f"  {strat:<10}: >=95% subjects connected starting at {int(found*100)}% density")
    else:
        L.append(f"  {strat:<10}: never reaches 95% connectedness in tested range")

# Find highest density where target was reached for ≥95% subjects
for strat in STRATEGIES:
    found = None
    for d in DENSITIES[::-1]:
        if agg.loc[(strat, d), "pct_target_reach"] >= 95.0:
            found = d
            break
    if found is not None:
        L.append(f"  {strat:<10}: target edge count reached for >=95% subjects up to "
                 f"{int(found*100)}% density")

summary = "\n".join(L)
print("\n" + summary)
with open(os.path.join(OUT_DIR, "step3a_summary.txt"), "w") as f:
    f.write(summary)

print("\nOutputs:")
for fn in sorted(os.listdir(OUT_DIR)):
    print(f"  {fn}")