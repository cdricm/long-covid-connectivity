"""
Step 3c QC — density saturation check (partial arm, cross-strategy diagnostic).

Global Efficiency freezes to a constant at high density in the 3c sweep. This
check separates the two possible causes by reading the existing step3c sweep
CSVs (no recomputation):

  (1) TOPOLOGICAL saturation: n_edges keeps rising while GE stays flat -> added
      edges are too weak to shorten any shortest path (efficiency_wei maps
      weight w to distance 1/w, so near-zero partial correlations act as
      effectively infinite distances).
  (2) COUNT saturation: n_edges itself plateaus -> proportional_threshold cannot
      reach the target density because the sign-strategy matrix has too few
      non-zero entries.

Scope note: the confirmatory AUC range is 10-25 %; the 5-50 % broad range is a
declared sensitivity. This check identifies the density above which the sweep
carries no density-resolved information, which bounds the interpretation of the
broad-range AUC.

Out: config.atlas_dir("schaefer400", "step3c_metrics", cross_strategy=True)/
     step3c_saturation_check.txt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

assert config.FC_METHOD == "partial", \
    "density saturation check is partial-arm only (checks GE freeze under partial correlation)"

import numpy as np
import pandas as pd

# ===== Config =================================================================
ATLAS = "schaefer400"
STRATEGIES = ["positive", "absolute"]     # 'negative' was skipped (degenerate)
METRICS = ["global_efficiency", "mean_clustering", "assortativity"]
FLAT_TOL = 1e-4        # per-metric: |mean(d) - mean(d_prev)| below this = frozen
EDGE_TOL = 0.001       # relative change in n_edges below this = count plateau


def load(strategy):
    p = config.atlas_dir(ATLAS, f"step3c_metrics/{strategy}",
                         cross_strategy=True) / "step3c_metrics.csv"
    if not p.exists():
        return None, p
    return pd.read_csv(p), p


def main():
    L = ["Step 3c — density saturation check (partial arm)", "=" * 68,
         "Reads existing step3c sweep CSVs; no recomputation.",
         "Question: above which density does the sweep stop carrying",
         "          density-resolved information, and why?", ""]

    for strategy in STRATEGIES:
        df, path = load(strategy)
        L.append(f"--- Strategy: {strategy} ---")
        if df is None:
            L.append(f"  MISSING: {path}")
            L.append("")
            continue

        # group-blind: pool all subjects, no group split (this is a structural
        # check on the sweep itself, not a group contrast)
        agg = df.groupby("density").agg(
            n_edges=("n_edges", "mean"),
            **{m: (m, "mean") for m in METRICS}
        ).reset_index().sort_values("density")

        L.append(f"  {'dens%':>6} {'n_edges':>10} {'d_edges%':>9} " +
                 " ".join(f"{m[:12]:>13}" for m in METRICS))
        prev = None
        for _, r in agg.iterrows():
            d_edges = (np.nan if prev is None
                       else (r["n_edges"] - prev["n_edges"]) / max(prev["n_edges"], 1) * 100)
            L.append(f"  {r['density']*100:6.0f} {r['n_edges']:10.0f} "
                     f"{d_edges:9.2f} " +
                     " ".join(f"{r[m]:13.6f}" for m in METRICS))
            prev = r

        # Diagnose per metric: first density above which the metric is frozen
        L.append("")
        for m in METRICS:
            v = agg[m].values
            dens = agg["density"].values
            frozen_from = None
            for i in range(1, len(v)):
                if np.all(np.abs(np.diff(v[i - 1:])) < FLAT_TOL):
                    frozen_from = dens[i - 1]
                    break
            if frozen_from is None:
                L.append(f"  {m:<20}: varies across the full sweep (no freeze)")
            else:
                L.append(f"  {m:<20}: frozen from {frozen_from*100:.0f} % density "
                         f"(|delta| < {FLAT_TOL})")

        # Edge-count behaviour in the frozen zone tells us WHICH saturation
        e = agg["n_edges"].values
        rel = np.abs(np.diff(e)) / np.maximum(e[:-1], 1)
        plateau = np.all(rel[-5:] < EDGE_TOL) if len(rel) >= 5 else False
        L.append("")
        if plateau:
            L.append("  -> n_edges plateaus at high density: COUNT saturation")
            L.append("     (proportional_threshold cannot reach target density)")
        else:
            L.append("  -> n_edges keeps rising while metrics freeze:")
            L.append("     TOPOLOGICAL saturation (added edges too weak to alter")
            L.append("     shortest paths; efficiency_wei distance = 1/weight)")
        L.append("")

    out_dir = config.ensure(config.atlas_dir(ATLAS, "step3c_metrics",
                                             cross_strategy=True))
    txt = "\n".join(L)
    print(txt)
    with open(out_dir / "step3c_saturation_check.txt", "w") as f:
        f.write(txt)


if __name__ == "__main__":
    main()