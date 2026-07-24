"""
QC on DPABI-preprocessed NIfTI data (FunImgARWSDCFN): header/duration/NaN/coverage
checks plus descriptive tSNR, written to step0a_qc_summary.csv.

In: config.NII_ROOT (subject NIfTI directories).
Out: step0a_qc_summary.csv — one row per subject, QC evidence only, no exclusion applied here.

tSNR is descriptive only (inflated by smoothing), not used for any exclusion verdict.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.datasets import load_mni152_brain_mask
from nilearn.image import resample_to_img
from joblib import Parallel, delayed
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

DATA_DIR = config.NII_ROOT
OUT_DIR = config.ensure(config.PRE_ANALYSIS_DIR)
OUT_CSV  = OUT_DIR / "step0a_qc_summary.csv"

PROTOCOL_DURATIONS_SEC = (7 * 60, 10 * 60)
DURATION_TOLERANCE_SEC = 15
N_JOBS = config.N_JOBS_DEFAULT


def qc_subject(sid, nii_path, mask):
    row = {"subject_id": sid, "file_path": str(nii_path)}
    try:
        img = nib.load(str(nii_path))
    except Exception as e:
        row["status"] = "load_failed"; row["error"] = str(e); return row
    if len(img.shape) != 4:
        row.update({"shape": str(img.shape), "status": "invalid_header"}); return row

    tr, n_vol = float(img.header.get_zooms()[3]), img.shape[3]
    duration = tr * n_vol
    row.update({"shape": str(img.shape), "TR_sec": tr, "n_volumes": n_vol,
                "duration_sec": duration})
    row["duration_check"] = next(
        (f"ok_{int(t/60)}min" for t in PROTOCOL_DURATIONS_SEC
         if abs(duration - t) <= DURATION_TOLERANCE_SEC), "non_protocol")

    try:
        data4d = img.get_fdata(dtype=np.float32, caching="unchanged")
    except Exception as e:
        row["status"] = "data_load_failed"; row["error"] = str(e); return row

    row["n_invalid_voxels"] = int(np.isnan(data4d).sum() + np.isinf(data4d).sum())
    if mask.shape != data4d.shape[:3]:
        row["status"] = "grid_mismatch"; return row
    n_mask = int(mask.sum()); row["n_mask_voxels"] = n_mask
    if n_mask < 1000:
        row["status"] = "insufficient_mask_voxels"; return row

    mean_vol = data4d.mean(axis=3, dtype=np.float32)
    row["coverage_fraction"] = float((np.abs(mean_vol[mask]) > 1e-6).sum() / n_mask)
    std_t = data4d.std(axis=3, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        tsnr = np.where(std_t > 0, mean_vol / std_t, 0).astype(np.float32)
    row["tSNR_median"] = float(np.median(tsnr[mask]))
    row["status"] = "complete"
    return row


def main():
    dirs = sorted(d for d in DATA_DIR.iterdir() if d.is_dir())
    print(f"Found {len(dirs)} subject directories")
    tasks = []
    for d in dirs:
        nii = sorted(d.glob("*.nii*"))
        tasks.append((d.name, nii[0]) if nii else (d.name, None))

    first = next((p for _, p in tasks if p is not None), None)
    if first is None:
        print("No .nii files found — aborting."); return
    print("Resampling MNI mask once on reference grid ...")
    mni = load_mni152_brain_mask()
    mask = np.asarray(resample_to_img(
        mni, nib.load(str(first)), interpolation="nearest",
        force_resample=True, copy_header=True).get_fdata() > 0)
    print(f"Mask voxels: {int(mask.sum())}, shape: {mask.shape}")

    def dispatch(sid, path):
        return {"subject_id": sid, "status": "no_nii_found"} if path is None \
            else qc_subject(sid, path, mask)

    records = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(
        delayed(dispatch)(sid, path) for sid, path in tasks)
    df = pd.DataFrame(records)

    # descriptive tSNR flag (NOT used for any verdict)
    if "tSNR_median" in df:
        med = df["tSNR_median"].median()
        m = np.nanmedian(np.abs(df["tSNR_median"] - med))
        df["tSNR_flag"] = np.where(
            (m > 0) & (df["tSNR_median"] < med - 3.0 * m), "flag_low_desc", "ok")

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print("\nStatus distribution:"); print(df["status"].value_counts())
    if "duration_check" in df:
        print("\nDuration check:"); print(df["duration_check"].value_counts(dropna=False))


if __name__ == "__main__":
    main()