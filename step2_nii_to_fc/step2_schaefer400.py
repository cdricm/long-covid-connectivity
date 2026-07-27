"""Step 2 for Schaefer-400 (7 networks, 2mm).

Stores raw Pearson r matrices (fisher_z=False). The Fisher-z transform is applied
downstream (steps 3/4/5) in z-then-mean order where group inference or aggregated
FC measures require it. Output tree is namespaced by config.FC_METHOD.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
import os
os.environ["JOBLIB_TEMP_FOLDER"] = str(config.ensure(config.JOBLIB_TEMP))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nii_to_fc_pipeline import run_pipeline


def main():
    run_pipeline(
        atlas_cfg = {
            "family"        : "schaefer",
            "n_rois"        : 400,
            "yeo_networks"  : 7,
            "resolution_mm" : 2,
            "label"         : "Schaefer-400 (7 networks)",
        },
        paths_cfg = {
            "nii_root"    : config.NII_ROOT,
            "out_mat_dir" : config.atlas_dir("schaefer400", "step2_pipeline") / "comet_matrices",
            "out_ts_dir"  : config.atlas_dir("schaefer400", "step2_pipeline") / "comet_timeseries",
            "out_log_dir" : config.atlas_dir("schaefer400", "step2_pipeline"),
        },
        processing_cfg = {
            "n_jobs"            : 2,
            "use_cache"         : True,
            "recompute_fc"      : False,
            "masker_standardize": "zscore_sample",
            "masker_t_r"        : 3.0,
            "fisher_z"          : False,
            "diagonal_val"      : 0,
        },
    )


if __name__ == "__main__":
    main()