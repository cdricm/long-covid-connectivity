"""Step 3c (signed Modularity Q*) for Schaefer-400 (7 networks).

Single Q* per subject on the signed, full, unthresholded matrix
(modularity_louvain_und_sign, qtype='sta'), multi-run mean over 100 runs with
reproducible SeedSequence substreams. No thresholding, no AUC, no
normalization. Descriptive group comparison only; Family A inference is step3d.

STRATEGY-INVARIANT: unlike the three swept metrics (step3c metrics), Modularity does
NOT depend on the sign strategy — it reads the full signed matrix directly, so the
positive/negative/absolute choice never enters. It is therefore computed ONCE here,
in the sign-neutral output tree, and is deliberately excluded from the _cross_strategy
comparison (running it per strategy would recompute the identical value three times).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step3c_modularity_pipeline import run_modularity

run_modularity(
    paths_cfg = {
        "mat_dir"     : config.atlas_dir("schaefer400", "step2_pipeline") / "comet_matrices",
        "csv_path"    : config.GROUP_CSV,
        "out_dir"     : config.atlas_dir("schaefer400", "step3c_modularity"),
        "atlas_label" : "Schaefer-400 (7 networks)",
    },
    mod_cfg = {
        "col_id"    : "ID",
        "col_group" : "Grupo",
        "groups"    : config.GROUP_ORDER,
        "n_runs"    : config.MODULARITY_N_RUNS,
        "n_jobs"    : config.N_JOBS_DEFAULT,
    },
)