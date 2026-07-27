"""
Step 5: Network-Based Statistic (NBS) — Family C (edge-level inference).

In:  config.atlas_dir("schaefer400", "step2_pipeline")/comet_matrices,
     step4a_labels/schaefer400_yeo7_roi_info.csv, config.GROUP_CSV.
Out: nbs_summary.csv,
     nbs_components_<model>_<contrast>_<thr>.csv,
     nbs_edges_<model>_<contrast>_<thr>.csv,
     nbs_null_<model>_<contrast>_<thr>.npz, nbs_console_log.txt.

NBS (Zalesky et al. 2010): cluster-level permutation FWE on edge space via
bct.nbs_bct, cluster statistic = component extent (edge count). Group order
x=CONTROL, y=COVID (validated in step5_nbs_validation). Two one-sided directional
contrasts (supervisor requirement) replace the former two-sided test:
COVID>CONTROL (tail='left') and COVID<CONTROL (tail='right'), each at every
cluster-forming threshold. The same seed (42) is used for every run, so the
permutation structure is identical across runs — only the cluster-forming t and
the one-sided tail differ. The 2 contrasts x 3 thresholds = 6 runs run in parallel.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import os
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# ---------- Settings ----------
USE_CACHE  = True
SEED       = config.SEED
N_PERM     = config.N_PERMUTATIONS
N_NODES    = 400
THRESHOLDS = config.NBS_THRESHOLDS
PRIMARY_THRESHOLD = config.NBS_PRIMARY_THRESHOLD
CONTRASTS  = config.NBS_CONTRASTS   # [(label, bct_tail), ...] one-sided, directional
NBS_ALPHA  = config.NBS_ALPHA       # R2 ①: FWE 0.025 per directional contrast
FISHER_CLIP = 0.999999   # MD §8 exception: step3f/step5a use 0.999999, not re-run
# 2 contrasts x 3 thresholds = 6 independent runs; parallelize over all of them.
N_JOBS_NBS = min(len(CONTRASTS) * len(THRESHOLDS), config.N_JOBS_DEFAULT)

CSV_PATH = config.GROUP_CSV
FC_DIR   = config.atlas_dir("schaefer400", "step2_pipeline") / "comet_matrices"
ROI_INFO = config.atlas_dir("schaefer400", "step4a_labels") / "schaefer400_yeo7_roi_info.csv"
OUT_DIR  = config.ensure(config.atlas_dir("schaefer400", "step5_nbs"))
ID_COL, GROUP_COL = "ID", "Grupo"

# ---------- Logging (console + file) ----------
class TeeLogger:
    def __init__(self, path):
        self.f = open(path, "w"); self.stdout = sys.stdout
    def write(self, msg): self.f.write(msg); self.f.flush(); self.stdout.write(msg)
    def flush(self): self.f.flush(); self.stdout.flush()

sys.stdout = TeeLogger(os.path.join(OUT_DIR, "nbs_console_log.txt"))

print("=" * 70)
print("Step 5 — NBS Schaefer-400 (Family C)")
print(f"thresholds={THRESHOLDS} (primary={PRIMARY_THRESHOLD}), k={N_PERM}, "
      f"seed={SEED}, parallel jobs={N_JOBS_NBS}")
print(f"directional contrasts: {[c[0] for c in CONTRASTS]} "
      f"(tails {[c[1] for c in CONTRASTS]}); x=CONTROL, y=COVID")
print("label permutation on age + sex residualised edges (R2 ②)")
print("=" * 70)

from comet.graph import bct

# ============================================================
# 1) COHORT via config
# ============================================================
print("\n[1] Cohort (config single source of truth)")
df_csv = pd.read_csv(CSV_PATH)
subjects = config.select_included_subjects(
    [p.name for p in config.NII_ROOT.iterdir() if p.is_dir()],
    df_csv, id_col=ID_COL, group_col=GROUP_COL, verbose=True)
missing = [s for s in subjects if not (FC_DIR / f"{s}_connectivity_comet.npy").exists()]
assert not missing, f"Missing FC matrices: {missing}"

meta = df_csv.copy(); meta[ID_COL] = meta[ID_COL].astype(str).str.strip()
meta = meta.set_index(ID_COL)
groups = np.array([str(meta.loc[s, GROUP_COL]).strip() for s in subjects])

n_control = int((groups == "CONTROL").sum())
n_covid   = int((groups == "COVID").sum())
print(f"    N={len(subjects)} (CONTROL={n_control}, COVID={n_covid})")

# Hard cohort guard: full frozen cohort, no covariate-driven drop.
assert len(subjects) == 162 and n_covid == 123 and n_control == 39, \
    "cohort deviates from frozen 162 (123/39)"

# ============================================================
# 2) Fisher-z tensor (N, N, P)
# ============================================================
print("\n[2] Build Fisher-z tensor + symmetry check")

def load_z(sid):
    m = np.load(FC_DIR / f"{sid}_connectivity_comet.npy")
    np.fill_diagonal(m, 0.0)
    asym = np.abs(m - m.T).max()          # record numerical asymmetry
    m = 0.5 * (m + m.T)                    # enforce exact symmetry
    z = np.arctanh(np.clip(m, -FISHER_CLIP, FISHER_CLIP))
    np.fill_diagonal(z, 0.0)
    return z, asym

loaded = [load_z(s) for s in subjects]
Z = np.stack([z for z, _ in loaded], axis=-1)
max_raw_asym = max(a for _, a in loaded)
max_asym = max(np.abs(Z[..., s] - Z[..., s].T).max() for s in range(Z.shape[-1]))

print(
    f"    tensor {Z.shape}, range [{Z.min():.3f},{Z.max():.3f}], "
    f"max raw asymmetry={max_raw_asym:.2e}, "
    f"max|M-M.T| after symmetrization={max_asym:.2e}"
)
assert max_asym < 1e-12, "Matrices not symmetric after symmetrization"

# ============================================================
# 2b) R2 ②: edge-wise age + sex residualisation (covariate adjustment)
# ============================================================
# NBS takes no design matrix, so age + sex are regressed out per edge BEFORE the
# directional permutation. Same covariate model as Families A/B (mean-centred age
# + 0/1-coded sex); reduced model only (NO group), so the group difference is
# retained in the residuals while age/sex are removed. Vectorised: one design
# matrix, one lstsq over all edges. Symmetry and zero diagonal are re-enforced
# because they are relied on downstream (upper_edges, nbs_bct).
print("\n[2b] Edge-wise age + sex residualisation (R2 ② covariate adjustment)")
age = pd.to_numeric(meta.loc[subjects, "Edad"], errors="coerce").values.astype(float)
if np.isnan(age).any():
    bad = [s for s, a in zip(subjects, age) if np.isnan(a)]
    raise AssertionError(f"non-numeric/missing age for: {bad}")
sex_raw = meta.loc[subjects, "Genero"].astype(str).str.strip().values
sex_cats = sorted(set(sex_raw))
assert len(sex_cats) == 2, f"sex must have exactly 2 categories; found {sex_cats}"
sex_map = {sex_cats[0]: 0.0, sex_cats[1]: 1.0}
sex = np.array([sex_map[s] for s in sex_raw], float)
age_c = age - age.mean()                                   # mean-centred (supervisor)
X_nuis = np.column_stack([np.ones(len(subjects)), age_c, sex])   # 1 + age_c + sex
print(f"    covariates: age mean={age.mean():.1f} [{age.min():.0f}-{age.max():.0f}] "
      f"(centred), sex coding {sex_map}")

# Residualise the (N, N, P) tensor edge-wise via a single least-squares solve.
P = Z.shape[-1]
Z_flat = Z.reshape(N_NODES * N_NODES, P).T                  # (P, N*N)
beta_nuis, *_ = np.linalg.lstsq(X_nuis, Z_flat, rcond=None)
Z_resid_flat = Z_flat - X_nuis @ beta_nuis                  # group NOT in model
Z = Z_resid_flat.T.reshape(N_NODES, N_NODES, P)
# Re-enforce exact symmetry + zero diagonal (identical design across (i,j)/(j,i)
# keeps symmetry to float precision; make it exact for the downstream assert).
Z = 0.5 * (Z + Z.transpose(1, 0, 2))
for _s in range(P):
    np.fill_diagonal(Z[..., _s], 0.0)
resid_asym = max(np.abs(Z[..., s] - Z[..., s].T).max() for s in range(P))
print(f"    residualised tensor {Z.shape}, range [{Z.min():.3f},{Z.max():.3f}], "
      f"max|R-R.T|={resid_asym:.2e}")
assert resid_asym < 1e-12, "Residualised matrices not symmetric"

# ============================================================
# 3) ROI info + per-edge descriptive Welch t (COVID - CONTROL)
# ============================================================
roi_df = pd.read_csv(ROI_INFO)
roi_label_col = "full_label" if "full_label" in roi_df.columns else \
    next((c for c in roi_df.columns if "label" in c.lower()), None)
roi_yeo_col = "yeo_network" if "yeo_network" in roi_df.columns else \
    next((c for c in roi_df.columns if "yeo" in c.lower()), None)

def edge_welch_t(tensor):
    c = tensor[..., groups == "COVID"]; h = tensor[..., groups == "CONTROL"]
    mc, mh = c.mean(-1), h.mean(-1)
    vc, vh = c.var(-1, ddof=1), h.var(-1, ddof=1)
    se = np.sqrt(vc/c.shape[-1] + vh/h.shape[-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(se > 0, (mc - mh)/se, 0.0), mc, mh

def upper_edges(adj, cid):
    iu, ju = np.triu_indices_from(adj, k=1)
    mask = adj[iu, ju] == cid
    return list(zip(iu[mask].tolist(), ju[mask].tolist()))

# ============================================================
# 4) Per-threshold NBS (parallelizable)
# ============================================================
def _run_one_threshold(Xc, Xv, thr, contrast_label, bct_tail, model_tag,
                       edge_t, mfc_cov, mfc_ctl):
    """One NBS run at one (contrast, threshold). Independent across runs. Same
    seed everywhere so the permutation structure is identical; only the
    cluster-forming t and the one-sided tail differ. Group order is fixed
    x=CONTROL, y=COVID, so bct_tail encodes the directional contrast
    ('left' -> COVID>CONTROL, 'right' -> COVID<CONTROL)."""
    tag = f"t{str(thr).replace('.', '')}"
    stem = f"{model_tag}_{contrast_label}_{tag}"
    comp_csv = OUT_DIR / f"nbs_components_{stem}.csv"
    edge_csv = OUT_DIR / f"nbs_edges_{stem}.csv"
    null_npz = OUT_DIR / f"nbs_null_{stem}.npz"

    if USE_CACHE and comp_csv.exists() and null_npz.exists():
        return contrast_label, thr, pd.read_csv(comp_csv), "CACHE-HIT"

    t0 = time.time()
    pval, adj, null = bct.nbs_bct(Xc, Xv, thr, k=N_PERM, tail=bct_tail,
                                  paired=False, verbose=False, seed=SEED)
    pval = np.atleast_1d(np.asarray(pval))
    runtime = (time.time() - t0) / 60
    comp_rows, edge_rows = [], []
    n_comp = int(adj.max())
    for cid in range(1, n_comp + 1):
        edges = upper_edges(adj, cid)
        if not edges:
            continue
        ts = np.array([edge_t[i, j] for i, j in edges])
        p_fwer = float(pval[cid-1]) if cid-1 < len(pval) else np.nan
        comp_rows.append({
            "contrast": contrast_label,
            "comp_id": cid, "n_edges": len(edges), "p_fwer": p_fwer,
            "mean_t": float(ts.mean()), "min_t": float(ts.min()),
            "max_t": float(ts.max()),
            "pct_positive_edges": float((ts > 0).mean()*100),
            "direction": "COVID>CONTROL" if ts.mean() > 0 else "COVID<CONTROL",
            "is_significant": p_fwer < NBS_ALPHA,
        })
        for (i, j), t_ij in zip(edges, ts):
            edge_rows.append({
                "contrast": contrast_label,
                "threshold": thr, "comp_id": cid, "p_fwer": p_fwer,
                "roi_i": i, "roi_j": j,
                "label_i": roi_df.iloc[i][roi_label_col] if roi_label_col else "",
                "label_j": roi_df.iloc[j][roi_label_col] if roi_label_col else "",
                "yeo_i": roi_df.iloc[i][roi_yeo_col] if roi_yeo_col else "",
                "yeo_j": roi_df.iloc[j][roi_yeo_col] if roi_yeo_col else "",
                "t": float(t_ij),
                "mean_fc_covid": float(mfc_cov[i, j]),
                "mean_fc_control": float(mfc_ctl[i, j]),
                "delta_fc": float(mfc_cov[i, j] - mfc_ctl[i, j]),
            })
    comp_df = pd.DataFrame(comp_rows).sort_values("p_fwer").reset_index(drop=True)
    comp_df.to_csv(comp_csv, index=False)
    pd.DataFrame(edge_rows).to_csv(edge_csv, index=False)
    np.savez_compressed(null_npz, null=null, pval=pval, edge_t=edge_t)
    msg = (f"runtime {runtime:.1f} min, n_components={len(pval)}, "
           f"null max [{null.min():.0f},{np.median(null):.0f},{null.max():.0f}]")
    return contrast_label, thr, comp_df, msg


def run_model(tensor, model_tag):
    Xc = tensor[..., groups == "CONTROL"]
    Xv = tensor[..., groups == "COVID"]
    edge_t, mfc_cov, mfc_ctl = edge_welch_t(tensor)
    print(f"\n{'='*64}\n[NBS] model={model_tag}  x=CONTROL(n={Xc.shape[-1]}), "
          f"y=COVID(n={Xv.shape[-1]})")
    n_runs = len(CONTRASTS) * len(THRESHOLDS)
    print(f"  {len(CONTRASTS)} contrasts x {len(THRESHOLDS)} thresholds "
          f"= {n_runs} runs over {N_JOBS_NBS} jobs "
          f"(per-perm progress suppressed)\n{'='*64}")

    results = Parallel(n_jobs=N_JOBS_NBS)(
        delayed(_run_one_threshold)(
            Xc, Xv, thr, contrast_label, bct_tail,
            model_tag, edge_t, mfc_cov, mfc_ctl)
        for contrast_label, bct_tail in CONTRASTS
        for thr in THRESHOLDS)

    summary = []
    for contrast_label, thr, comp_df, msg in sorted(
            results, key=lambda x: (x[0], x[1])):
        print(f"\n  [{contrast_label}] t={thr}: {msg}")
        if len(comp_df):
            print(comp_df.head(5).to_string(index=False))
        for _, r in comp_df.iterrows():
            summary.append({
                "model": model_tag, "contrast": contrast_label,
                "threshold": thr,
                "is_primary": thr == PRIMARY_THRESHOLD,
                "comp_id": int(r["comp_id"]), "n_edges": int(r["n_edges"]),
                "p_fwer": float(r["p_fwer"]), "mean_t": float(r["mean_t"]),
                "direction": r.get("direction", ""),
                "is_significant": bool(r["p_fwer"] < NBS_ALPHA),
            })
    return summary

# ============================================================
# 5) RUN (single model: age + sex residualised edges, R2 ②)
# ============================================================
all_summary = run_model(Z, "age_sex_residualised")

summary_df = pd.DataFrame(all_summary).sort_values(
    ["contrast", "threshold", "p_fwer"]).reset_index(drop=True)
summary_df.to_csv(OUT_DIR / "nbs_summary.csv", index=False)

# ============================================================
# 6) Overview
# ============================================================
print(f"\n{'='*64}\nNBS SUMMARY (Family C)\n{'='*64}")
sub = summary_df
for contrast_label, _ in CONTRASTS:
    print(f"\n  contrast: {contrast_label}")
    csub = sub[sub["contrast"] == contrast_label]
    for thr in THRESHOLDS:
        s2 = csub[csub["threshold"] == thr]
        n_sig = int(s2["is_significant"].sum())
        min_p = s2["p_fwer"].min() if len(s2) else float("nan")
        flag = " (PRIMARY)" if thr == PRIMARY_THRESHOLD else ""
        print(f"    t={thr}{flag}: {len(s2)} components, {n_sig} significant "
              f"(p_fwer<{NBS_ALPHA}), min p={min_p:.4f}")
        for _, r in s2[s2["is_significant"]].iterrows():
            print(f"        * comp {int(r['comp_id'])}: p={r['p_fwer']:.4f}, "
                  f"{int(r['n_edges'])} edges, {r['direction']}")

print(f"\nSaved: {OUT_DIR / 'nbs_summary.csv'}")
print(f"Age + sex residualised edges (R2 ②). Primary = t=3.1; sensitivity = t=2.5/3.5. "
      f"FWE {NBS_ALPHA} per directional contrast.")
print("One-sided directional contrasts: COVID>CONTROL (tail=left), "
      "COVID<CONTROL (tail=right).")
print("=" * 70)