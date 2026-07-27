"""
Step 3a threshold-sweep diagnostic for Schaefer-100, all three sign strategies,
group-blind.

In: cached matrices via config.atlas_dir("schaefer100", "step2_pipeline"),
    config.GROUP_CSV.
Out: config.atlas_dir("schaefer100", "step3a_sweep", cross_strategy=True)/
     step3a_sweep.csv, step3a_sweep_summary.csv, diagnostic PNGs, summary.txt.

Density grid = the sweep support points plus a deliberate 35-100% diagnostic
extension, to characterize where graphs become connected — the confirmatory AUC
range itself (10-25%) is applied downstream, never here.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step3a_threshold_sweep_pipeline import run_threshold_sweep

# Confirmatory support points + deliberate diagnostic extension (35-100 %).
DIAGNOSTIC_DENSITIES = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50,
                        0.35, 0.40, 0.45, 0.60, 0.70, 0.80, 0.90, 1.00]

run_threshold_sweep(
    paths_cfg = {
        "mat_dir"     : config.atlas_dir("schaefer100", "step2_pipeline") / "comet_matrices",
        "csv_path"    : config.GROUP_CSV,
        "out_dir"     : config.atlas_dir("schaefer100", "step3a_sweep", cross_strategy=True),
        "atlas_label" : "Schaefer-100 (7 networks)",
    },
    sweep_cfg = {
        "col_id"     : "ID",
        "col_group"  : "Grupo",
        "groups"     : ("CONTROL", "COVID"),
        "densities"  : sorted(DIAGNOSTIC_DENSITIES),
        "strategies" : tuple(config.DIAGNOSTIC_SIGN_STRATEGIES),
        "n_jobs"     : config.N_JOBS_DEFAULT,
    },
)