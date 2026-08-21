"""Step 3c (graph metrics sweep, diagnostic) for Schaefer-400 — all three sign strategies.

Three density-dependent weighted metrics (Global Efficiency, Mean Clustering,
Assortativity) swept across densities, run as a group-blind DIAGNOSTIC over all sign
strategies (like ste<p3a/3b) and written to the _cross_strategy tree, one subfolder per
strategy. The sweep spans 5-100 % for visualization; AUC inference (step3d) uses the
10-25 % confirmatory and 5-50 % sensitivity subsets.

Modularity Q* is NOT part of this strategy comparison: it is strategy-invariant
(signed, full unthresholded matrix) and is computed once in step3c_modularity.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step3c_metrics_sweep_pipeline import run_metrics_sweep

ATLAS       = "schaefer400"
ATLAS_LABEL = "Schaefer-400 (7 networks)"
STRATEGIES  = list(config.DIAGNOSTIC_SIGN_STRATEGIES)   # group-blind diagnostic comparison
MAT_DIR     = config.atlas_dir(ATLAS, "step2_pipeline") / "comet_matrices"

for strategy in STRATEGIES:
    out_dir = config.atlas_dir(ATLAS, f"step3c_metrics/{strategy}", cross_strategy=True)
    print("\n" + "=" * 72)
    print(f"STEP 3c METRICS — strategy: {strategy}  ->  {out_dir}")
    print("=" * 72)
    run_metrics_sweep(
        paths_cfg = {
            "mat_dir"     : MAT_DIR,
            "csv_path"    : config.GROUP_CSV,
            "out_dir"     : out_dir,
            "atlas_label" : f"{ATLAS_LABEL} — {strategy}",
        },
        sweep_cfg = {
            "col_id"      : "ID",
            "col_group"   : "Grupo",
            "groups"      : config.GROUP_ORDER,
            "densities"   : [i / 100 for i in range(5, 101, 5)],
            "strategy"    : strategy,
            "n_jobs"      : config.N_JOBS_DEFAULT,
        },
    )