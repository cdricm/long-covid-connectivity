"""
Step 5 NBS validation: verify the bct.nbs_bct API before the productive NBS run
(step5a). Positive control — without it, a Family-C null cannot be distinguished
from a silent pipeline failure.

Checks three things the productive run depends on: (1) tensor axis convention
(nodes, nodes, subjects); (2) argument order + group direction (x vs y, tail)
to interpret component direction; (3) return structure (pval, adj, null). Plus a
synthetic sanity check with a planted 10-node component, verifying NBS recovers it.

The 3v3 real-data call is an API-shape check only, statistically meaningless.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import inspect
import numpy as np
import pandas as pd

SEED        = config.SEED
CSV_PATH    = config.GROUP_CSV
FC_DIR      = config.atlas_dir("schaefer400", "step2_pipeline") / "comet_matrices"
N_NODES     = 400
N_PERM_TEST = 100   # validation only, not productive

def fc_path(sid):
    return FC_DIR / f"{sid}_connectivity_comet.npy"

def fisher_z(mat):
    return np.arctanh(np.clip(mat, -0.999999, 0.999999))

# ====================================================================
# 1) Import + inspect API
# ====================================================================
print("=" * 70); print("1) bct.nbs_bct API inspection"); print("=" * 70)
from comet.graph import bct

print(f"Has 'nbs_bct': {hasattr(bct, 'nbs_bct')}")
if not hasattr(bct, "nbs_bct"):
    print("ERROR: bct.nbs_bct missing. Names containing 'nbs':",
          [n for n in dir(bct) if "nbs" in n.lower()])
    sys.exit(1)

nbs_fn = bct.nbs_bct
sig = inspect.signature(nbs_fn)
param_names = list(sig.parameters.keys())
print(f"\nSignature: nbs_bct{sig}")
print(f"Parameter names: {param_names}")
doc = (nbs_fn.__doc__ or "").strip()
print(f"\nDocstring (first 1500 chars):\n{doc[:1500]}{'...' if len(doc) > 1500 else ''}")

# Helper: call nbs_bct adaptively based on the actual signature
def call_nbs(x, y, thresh, k, tail="both", paired=False, verbose=False):
    """Build kwargs from the real signature; only pass params that exist."""
    avail = set(param_names)
    kwargs = {}
    if "k" in avail: kwargs["k"] = k
    elif "n_perm" in avail: kwargs["n_perm"] = k
    elif "nperm" in avail: kwargs["nperm"] = k
    if "tail" in avail: kwargs["tail"] = tail
    if "paired" in avail: kwargs["paired"] = paired
    if "verbose" in avail: kwargs["verbose"] = verbose
    # positional: x, y, thresh (the near-universal BCT NBS order)
    return nbs_fn(x, y, thresh, **kwargs)

def summarize_return(result, label):
    print(f"\n{label}: return type={type(result).__name__}, "
          f"len={len(result) if isinstance(result, tuple) else 1}")
    if not isinstance(result, tuple):
        print(f"  single value: {str(result)[:120]}")
        return None
    parsed = {}
    for i, r in enumerate(result):
        shp = getattr(r, "shape", "n/a")
        print(f"  [{i}] type={type(r).__name__}, shape={shp}")
        arr = np.asarray(r) if not np.isscalar(r) else r
        if np.isscalar(r) or (hasattr(arr, "ndim") and arr.ndim == 1 and arr.size < 30):
            print(f"      values: {arr}")
        elif hasattr(arr, "ndim") and arr.ndim == 2:
            n_marked = int((arr > 0).sum() // 2)
            involved = np.where(np.asarray(arr).any(axis=0))[0]
            print(f"      adj-like: marked edges (upper tri)={n_marked}, "
                  f"n involved nodes={len(involved)}")
            parsed["adj"] = arr
            parsed["involved"] = involved
        parsed.setdefault("by_index", {})[i] = arr
    return parsed

# ====================================================================
# 2) Real subjects via config cohort (3 CONTROL + 3 COVID)
# ====================================================================
print("\n" + "=" * 70); print("2) Load real subjects (config cohort)"); print("=" * 70)
df = pd.read_csv(CSV_PATH)
included = config.select_included_subjects(
    [p.name for p in config.NII_ROOT.iterdir() if p.is_dir()],
    df, id_col="ID", group_col="Grupo", verbose=False)
gmap = {str(i).strip(): str(g).strip() for i, g in zip(df["ID"], df["Grupo"])}

def take_n(group, n):
    out = []
    for sid in included:
        if gmap.get(sid) == group and fc_path(sid).exists():
            m = np.load(fc_path(sid))
            if m.shape == (N_NODES, N_NODES):
                out.append((sid, m))
        if len(out) == n:
            break
    return out

controls = take_n("CONTROL", 3)
covids   = take_n("COVID", 3)
print(f"CONTROL: {[s for s, _ in controls]}")
print(f"COVID  : {[s for s, _ in covids]}")
assert len(controls) == 3 and len(covids) == 3, "could not load 3+3 subjects"

def stack(fc_list):  # BCT convention: (nodes, nodes, subjects)
    return np.stack([fc for _, fc in fc_list], axis=-1)

Xc = fisher_z(stack(controls))
Xv = fisher_z(stack(covids))
for T in (Xc, Xv):
    for s in range(T.shape[-1]):
        np.fill_diagonal(T[..., s], 0)
print(f"X_control.shape={Xc.shape}, X_covid.shape={Xv.shape}")
print(f"Fisher-z ranges: control[{Xc.min():.3f},{Xc.max():.3f}] "
      f"covid[{Xv.min():.3f},{Xv.max():.3f}]")

# ====================================================================
# 3) API test on real 3v3 (meaningless statistically — API only)
# ====================================================================
print("\n" + "=" * 70)
print("3) bct.nbs_bct on real 3v3, thresh=2.5, k=100 (API ONLY, not valid stats)")
print("=" * 70)
print("Group order passed: x=CONTROL, y=COVID  -> note direction in the docstring!")
try:
    res = call_nbs(Xc, Xv, 2.5, k=N_PERM_TEST, tail="both")
    summarize_return(res, "real 3v3")
except TypeError as e:
    print(f"TypeError (signature mismatch): {e}")
    print("-> inspect the printed signature above and adjust call_nbs.")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# ====================================================================
# 4) SYNTHETIC sanity check — planted 10-node component, VERIFY recovery
# ====================================================================
print("\n" + "=" * 70)
print("4) Synthetic sanity check: planted component in nodes 0-9, VERIFY recovery")
print("=" * 70)
rng = np.random.default_rng(SEED)
n_syn, n_grp = 50, 20
A = rng.standard_normal((n_syn, n_syn, n_grp)) * 0.3
B = rng.standard_normal((n_syn, n_syn, n_grp)) * 0.3
for T in (A, B):
    for s in range(n_grp):
        T[..., s] = (T[..., s] + T[..., s].T) / 2
        np.fill_diagonal(T[..., s], 0)
planted = list(range(10))
for i in planted:
    for j in planted:
        if i != j:
            B[i, j, :] += 0.5   # stronger connectivity within nodes 0-9 in group B

print(f"A.shape={A.shape}, B.shape={B.shape}; planted: nodes 0-9 (+0.5 in B)")
try:
    res_syn = call_nbs(A, B, 2.5, k=500, tail="both")
    parsed = summarize_return(res_syn, "synthetic")
    # Explicit recovery check: is there a component whose nodes overlap planted?
    if parsed and "involved" in parsed:
        involved = set(parsed["involved"].tolist())
        overlap = involved & set(planted)
        print(f"\n  RECOVERY CHECK:")
        print(f"    involved nodes: {sorted(involved)[:20]}")
        print(f"    planted nodes : {planted}")
        print(f"    overlap: {len(overlap)}/10 planted nodes recovered")
        print(f"    -> {'PASS' if len(overlap) >= 8 else 'CHECK'}: "
              f"NBS {'recovers' if len(overlap) >= 8 else 'may not cleanly recover'} "
              f"the planted component")
    else:
        print("\n  Could not auto-locate an adjacency matrix in the return; "
              "inspect the per-index summary above to identify the component output.")
except Exception as e:
    print(f"Error in synthetic test: {type(e).__name__}: {e}")

print("\n" + "=" * 70)
print("=" * 70)