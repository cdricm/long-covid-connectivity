"""
Shared NIfTI -> ROI-timeseries -> COMET-connectivity pipeline, called from the
atlas-specific wrappers (Schaefer-400, Schaefer-100, AAL).

In: atlas_cfg/paths_cfg/processing_cfg (wrapper-provided), subjects via
    config.select_included_subjects().
Out: *_timeseries_comet.npy, *_connectivity_comet.npy, step2_log.csv,
     step2_summary.txt.

Stored matrices are raw Pearson r (fisher_z=False); Fisher-z is applied
downstream, in z-then-mean order, wherever inference or aggregated FC measures
require it.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import os, glob, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def first_nii(subj_dir):
    """Return alphabetically first .nii or .nii.gz in folder."""
    cands = sorted(glob.glob(os.path.join(subj_dir, "*.nii"))) \
          + sorted(glob.glob(os.path.join(subj_dir, "*.nii.gz")))
    return cands[0] if cands else None


def run_pipeline(atlas_cfg, paths_cfg, processing_cfg):
    """Run NIfTI -> FC pipeline for one atlas.

    Parameters
    ----------
    atlas_cfg : dict
        {"family": "schaefer", "n_rois", "yeo_networks", "resolution_mm", "label"}
        or {"family": "aal", "version", "label"}
    paths_cfg : dict
        "nii_root", "out_mat_dir", "out_ts_dir", "out_log_dir"
    processing_cfg : dict
        "n_jobs", "use_cache", "recompute_fc", "masker_standardize", "masker_t_r",
        "diagonal_val", and "fisher_z" (False in both arms; z applied downstream)

    Returns
    -------
    DataFrame with one row per processed subject.
    """

    for d in (paths_cfg["out_mat_dir"], paths_cfg["out_ts_dir"], paths_cfg["out_log_dir"]):
        config.ensure(Path(d))

    from nilearn import datasets as nl_datasets
    from nilearn.maskers import NiftiLabelsMasker
    from nilearn.image import resample_to_img, load_img
    import nilearn, comet

    comet_version = getattr(comet, "__version__", "1.2.4")
    print(f"nilearn {nilearn.__version__}, COMET {comet_version}")
    print(f"FC_METHOD     : {config.FC_METHOD}")
    print(f"Atlas         : {atlas_cfg['label']}")
    print(f"N_JOBS={processing_cfg['n_jobs']}, "
          f"USE_CACHE={processing_cfg['use_cache']}, "
          f"RECOMPUTE_FC={processing_cfg['recompute_fc']}, "
          f"FISHER_Z={processing_cfg['fisher_z']}\n")

    # ===== Fetch atlas =========================================================
    if atlas_cfg["family"] == "schaefer":
        atlas = nl_datasets.fetch_atlas_schaefer_2018(
            n_rois=atlas_cfg["n_rois"],
            yeo_networks=atlas_cfg["yeo_networks"],
            resolution_mm=atlas_cfg["resolution_mm"],
        )
    elif atlas_cfg["family"] == "aal":
        atlas = nl_datasets.fetch_atlas_aal(version=atlas_cfg.get("version", "SPM12"))
        # nilearn 0.13.1 ships a 'Background' label at index 0 of atlas.labels and
        # atlas.indices (image value '0'). NiftiLabelsMasker must receive only the
        # real ROI labels; otherwise all ROI names are shifted by one (off-by-one).
        # Background is identified via atlas.indices == '0' (more robust than
        # labels[1:]), with a fallback for older nilearn versions without 'indices'.
        n_before = len(atlas.labels)
        if hasattr(atlas, "indices"):
            keep = [i for i, idx in enumerate(atlas.indices) if str(idx) != "0"]
            atlas.labels = [atlas.labels[i] for i in keep]
            atlas.indices = [atlas.indices[i] for i in keep]
        elif atlas.labels and atlas.labels[0] == "Background":
            atlas.labels = atlas.labels[1:]
        n_after = len(atlas.labels)
        if n_after != n_before:
            print(f"AAL: removed background label ({n_before} -> {n_after} ROIs)")
    else:
        raise ValueError(f"Unsupported atlas family: {atlas_cfg['family']}")
    print(f"Atlas: {atlas.maps}, n_labels={len(atlas.labels)}\n")

    # ===== Collect subjects ====================================================
    all_subjects = sorted(os.path.basename(d) for d in glob.glob(os.path.join(paths_cfg["nii_root"], "CP*")))
    group_df = pd.read_csv(config.GROUP_CSV)
    subjects = config.select_included_subjects(all_subjects, group_df)
    n_total, n_excluded = len(all_subjects), len(all_subjects) - len(subjects)
    print(f"Subjects: {n_total} total, {n_excluded} excluded, {len(subjects)} to process\n")

    # ===== Resample atlas to first subject's grid (once) =======================
    first = first_nii(os.path.join(paths_cfg["nii_root"], subjects[0]))
    atlas_to_use = resample_to_img(
        load_img(atlas.maps), load_img(first),
        interpolation="nearest", force_resample=True, copy_header=True,
    ) if first else atlas.maps

    # ===== Per-subject worker ==================================================
    def process_subject(subj):
        row = {"subject": subj}
        nii_path = first_nii(os.path.join(paths_cfg["nii_root"], subj))
        if not nii_path:
            row["error"] = "no_nifti_found"; return row
        row["nii_file"] = os.path.basename(nii_path)

        out_ts  = os.path.join(paths_cfg["out_ts_dir"],  f"{subj}_timeseries_comet.npy")
        out_mat = os.path.join(paths_cfg["out_mat_dir"], f"{subj}_connectivity_comet.npy")

        # Cache: load timeseries if present
        ts = None
        if processing_cfg["use_cache"] and os.path.exists(out_ts):
            try:
                ts = np.load(out_ts)
                row["cached_ts"]    = True
                row["t_extract_s"]  = 0.0
                row["n_timepoints"] = int(ts.shape[0])
                row["n_parcels"]    = int(ts.shape[1])
            except Exception:
                ts = None

        if ts is None:
            masker = NiftiLabelsMasker(
                labels_img  = atlas_to_use, labels = atlas.labels,
                standardize = processing_cfg["masker_standardize"],
                detrend     = False,  # DPABI preprocessing already applied this
                low_pass = None, high_pass = None,
                t_r         = processing_cfg["masker_t_r"], verbose = 0,
            )
            try:
                t0 = time.time()
                # Partial-arm only: truncate the 4D image to a common length
                # BEFORE the masker, so zscore_sample is computed on the truncated
                # series and every subject enters Ledoit-Wolf at the same p/T
                # regime. Pearson arm keeps the full series unchanged.
                img_in = nii_path
                if config.FC_METHOD == "partial":
                    from nilearn.image import load_img, index_img
                    img4d = load_img(nii_path)
                    n_tp_orig = int(img4d.shape[-1])
                    row["n_tp_original"] = n_tp_orig
                    if n_tp_orig > config.PARTIAL_TRUNCATE_TP:
                        img_in = index_img(
                            img4d, slice(0, config.PARTIAL_TRUNCATE_TP))
                        row["truncated"] = True
                    else:
                        row["truncated"] = False
                ts = masker.fit_transform(img_in)
                row["t_extract_s"]   = round(time.time() - t0, 2)
                row["n_timepoints"]  = int(ts.shape[0])
                row["n_parcels"]     = int(ts.shape[1])
                row["cached_ts"]     = False
                np.save(out_ts, ts)
            except Exception as e:
                row["error"] = f"masker_failed: {e}"; return row

        # FC: skip if cached and recompute disabled
        C = None
        if processing_cfg["use_cache"] and not processing_cfg["recompute_fc"] \
                and os.path.exists(out_mat):
            try:
                C = np.load(out_mat)
                row["cached_fc"]   = True
                row["t_fc_s"]      = 0.0
            except Exception:
                C = None

        if C is None:
            try:
                t0 = time.time()
                sp = config.make_connectivity(
                    time_series = ts,
                    diagonal    = processing_cfg["diagonal_val"],
                    fisher_z    = processing_cfg["fisher_z"],
                    tril        = False,
                )
                C = sp.estimate()
                if hasattr(sp, "postproc"): C = sp.postproc()
                row["t_fc_s"]    = round(time.time() - t0, 2)
                row["cached_fc"] = False
                np.save(out_mat, C)
            except Exception as e:
                row["error"] = f"fc_failed: {e}"; return row

        row["success"] = True
        return row

    # ===== Run in parallel =====================================================
    t_start = time.time()
    print(f"Running {len(subjects)} subjects with n_jobs={processing_cfg['n_jobs']}...\n")
    log_rows = Parallel(n_jobs=processing_cfg["n_jobs"], verbose=10)(
        delayed(process_subject)(subj) for subj in subjects
    )
    total_runtime = time.time() - t_start

    # ===== Save log ============================================================
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(os.path.join(paths_cfg["out_log_dir"], "step2_log.csv"), index=False)

    # ===== Summary =============================================================
    L = ["=" * 70, f"STEP 2 - FULL RUN ({atlas_cfg['label']}, FC_METHOD={config.FC_METHOD})", "=" * 70,
         f"nilearn {nilearn.__version__}, COMET {comet_version}",
         f"N_JOBS={processing_cfg['n_jobs']}, USE_CACHE={processing_cfg['use_cache']}, "
         f"RECOMPUTE_FC={processing_cfg['recompute_fc']}, FISHER_Z={processing_cfg['fisher_z']}",
         f"Total subjects     : {n_total}",
         f"Excluded           : {n_excluded}",
         f"Processed          : {len(log_df)}",
         f"Successful         : {int(log_df.get('success', pd.Series([], dtype=bool)).sum())}",
         f"Errors             : {int(log_df['error'].notna().sum()) if 'error' in log_df.columns else 0}",
         f"Total runtime      : {total_runtime/60:.1f} min", ""]

    if "cached_ts" in log_df.columns:
        n_cached_ts = int(log_df["cached_ts"].fillna(False).sum())
        L.append(f"Timeseries cached/fresh : {n_cached_ts} / {len(log_df) - n_cached_ts}")
    if "cached_fc" in log_df.columns:
        n_cached_fc = int(log_df["cached_fc"].fillna(False).sum())
        L.append(f"FC matrices cached/fresh: {n_cached_fc} / {len(log_df) - n_cached_fc}")
    L.append("")

    if "t_extract_s" in log_df.columns:
        t_ex = log_df.loc[log_df.get("cached_ts", False) == False, "t_extract_s"].dropna() \
            if "cached_ts" in log_df.columns else log_df["t_extract_s"].dropna()
        if len(t_ex):
            L += [f"Atlas extraction (fresh only): mean {t_ex.mean():.1f}s "
                  f"(min {t_ex.min():.1f}s, max {t_ex.max():.1f}s)", ""]

    if "n_timepoints" in log_df.columns:
        L.append("Timeseries lengths:")
        for v, c in log_df["n_timepoints"].dropna().astype(int).value_counts().sort_index().items():
            L.append(f"  {v} TPs : {c} subjects")
        L.append("")

    if "error" in log_df.columns and log_df["error"].notna().any():
        L.append("Errors:")
        for _, r in log_df[log_df["error"].notna()][["subject","error"]].iterrows():
            L.append(f"  {r['subject']}: {r['error']}")
        L.append("")

    summary_text = "\n".join(L)
    print("\n" + summary_text)
    with open(os.path.join(paths_cfg["out_log_dir"], "step2_summary.txt"), "w") as f:
        f.write(summary_text)

    print(f"\nDone in {total_runtime/60:.1f} min.")
    return log_df