"""
Step 3b disconnect diagnostic for Schaefer-400, all three sign strategies
(group-blind), one subfolder per strategy under the _cross_strategy tree.

In: cached matrices via config.atlas_dir("schaefer400", "step2_pipeline"),
    config.GROUP_CSV.
Out: config.atlas_dir("schaefer400", "step3b_diagnose/<strategy>",
     cross_strategy=True)/step3b_subject_disconnect.csv,
     step3b_roi_isolation.csv, diagnostic PNGs, summary.txt.

Densities = confirmatory range (10-25%) plus 5%/30% context, one strategy per
call to the shared pipeline module.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step3b_disconnect_diagnose_pipeline import run_disconnect_diagnose

# ---- SETTINGS ----
ATLAS       = "schaefer400"
ATLAS_LABEL = "Schaefer-400 (7 networks)"
STRATEGIES  = list(config.DIAGNOSTIC_SIGN_STRATEGIES)   # group-blind diagnostic comparison
DENSITIES   = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]   # confirmatory range + 5/30 % context
REF_DENSITY = 0.20
MAT_DIR     = config.atlas_dir(ATLAS, "step2_pipeline") / "comet_matrices"

for strategy in STRATEGIES:
    out_dir = config.atlas_dir(ATLAS, f"step3b_diagnose/{strategy}", cross_strategy=True)
    print("\n" + "=" * 72)
    print(f"STEP 3b — strategy: {strategy}  ->  {out_dir}")
    print("=" * 72)
    run_disconnect_diagnose(
        paths_cfg = {
            "mat_dir"     : MAT_DIR,                       # sign-neutral (same matrices)
            "csv_path"    : config.GROUP_CSV,
            "out_dir"     : out_dir,
            "atlas_label" : f"{ATLAS_LABEL} — {strategy}",
        },
        diagnose_cfg = {
            "col_id"            : "ID",
            "col_group"         : "Grupo",
            "strategy"          : strategy,               # single strategy per call
            "densities"         : DENSITIES,
            "reference_density" : REF_DENSITY,
            "n_jobs"            : config.N_JOBS_DEFAULT,
        },
    )