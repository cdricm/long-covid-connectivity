"""
Step 4h: Signal-dropout / signal-quality QC for the within-network FC trends.

In: config.NII_ROOT (raw 4D per subject), step4a_labels/schaefer400_yeo7_roi_info.csv.
Out: mean_<group>.nii.gz, tsnr_<group>.nii.gz, mean/tsnr_diff_COVID_minus_CONTROL.nii.gz,
     network_signal_quality_per_subject.csv, network_signal_quality_summary.csv.

Prerequisite: all 4D volumes in MNI152 on a common grid; the Schaefer-400 atlas
is resampled to the data grid (nearest-neighbor) for ROI extraction. Affine/
shape mismatch aborts. tSNR is computed on fully preprocessed data (post
nuisance regression + bandpass), so it reflects residual variance structure,
not raw signal-to-noise — descriptive only, no exclusion criterion.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================ SETTINGS ============================
NII_ROOT  = config.NII_ROOT
GROUP_CSV = config.GROUP_CSV
ID_COL    = "ID"
GROUP_COL = "Grupo"
GROUPS    = config.GROUP_ORDER

ROI_INFO_PATH = config.atlas_dir("schaefer400", "step4a_labels") / "schaefer400_yeo7_roi_info.csv"
OUT_DIR       = config.ensure(config.atlas_dir("schaefer400", "step4h_signal_dropout_qc"))
NII_GLOB      = "*.nii*"
N_WORKERS     = 2

TARGET_NETWORKS = config.TARGET_NETWORKS_BY_ARM[config.FC_METHOD]
if not TARGET_NETWORKS:
    print(f"step4h: no network-level trend to check dropout for in the {config.FC_METHOD} arm; skipped by design (MD).")
    sys.exit(0)
# ==================================================================

# ---------- atlas (resampled to data grid, built once, shared to workers) ----------
from nilearn import datasets, image as nimage

def build_network_voxel_masks(ref_img):
    """Resample Schaefer-400 to the data grid (NN) and build a {network: bool mask}
    dict over voxels, using the step4a Yeo mapping (ROI label i -> network)."""
    roi_info = pd.read_csv(ROI_INFO_PATH)
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=2)
    atlas_img = nimage.load_img(atlas.maps)
    atlas_res = nimage.resample_to_img(atlas_img, ref_img, interpolation="nearest")
    atlas_data = np.asarray(atlas_res.dataobj).astype(int)   # labels 1..400 (0=bg)

    masks = {}
    for net in TARGET_NETWORKS:
        roi_idx = roi_info.loc[roi_info["yeo_network"] == net, "roi_idx"].values  # 0-based
        labels = roi_idx + 1                                                       # image labels
        masks[net] = np.isin(atlas_data, labels)
    return masks


def find_nii(subject_id):
    folder = os.path.join(NII_ROOT, subject_id)
    if not os.path.isdir(folder):
        return None, f"folder missing: {folder}"
    hits = sorted(glob.glob(os.path.join(folder, NII_GLOB)))
    if len(hits) == 0:
        return None, f"no 4D file in {folder}"
    if len(hits) > 1:
        return None, f"{len(hits)} matches -> refine NII_GLOB: {hits}"
    return hits[0], None


def process_subject(args):
    """Worker: load 4D -> 3D mean + tSNR + grid info."""
    sid, grp = args
    path, err = find_nii(sid)
    if err:
        return sid, grp, None, None, None, None, err
    img = nib.load(path)
    if img.ndim != 4:
        return sid, grp, None, None, None, None, f"not 4D (ndim={img.ndim})"
    data = img.get_fdata(dtype=np.float32)
    m  = data.mean(axis=3)
    sd = data.std(axis=3)
    with np.errstate(divide="ignore", invalid="ignore"):
        tsnr = np.where(sd > 0, m / sd, 0.0)
    return (sid, grp, m.astype(np.float32), tsnr.astype(np.float32),
            img.shape[:3], img.affine.copy(), None)


# ---------- cohort via config (single source of truth) ----------
df = pd.read_csv(GROUP_CSV)
on_disk = [p.name for p in NII_ROOT.iterdir() if p.is_dir()]
included = config.select_included_subjects(on_disk, df, id_col=ID_COL,
                                           group_col=GROUP_COL, verbose=True)
gmap = {str(i).strip(): str(g).strip().upper() for i, g in zip(df[ID_COL], df[GROUP_COL])}
tasks = [(s, gmap[s]) for s in included if gmap.get(s) in GROUPS]
n_total = len(tasks)
print(f"\nDropout QC on config cohort: {n_total} subjects, N_WORKERS={N_WORKERS}\n")

ref_affine = ref_shape = None
net_masks = None
group_sums = {g: {"mean": None, "tsnr": None, "n": 0} for g in GROUPS}
roi_rows = []   # per-subject per-network mean intensity + tSNR
skipped = []
done = 0


def integrate(sid, grp, m, tsnr, shape, affine):
    global ref_affine, ref_shape, net_masks
    if ref_affine is None:
        ref_affine, ref_shape = affine, shape
        net_masks = build_network_voxel_masks(nib.Nifti1Image(m, affine))
    else:
        if shape != ref_shape:
            return f"shape mismatch {shape} != {ref_shape}"
        if not np.allclose(affine, ref_affine, atol=1e-3):
            return "affine mismatch vs reference"
    gs = group_sums[grp]
    if gs["mean"] is None:
        gs["mean"] = np.zeros(ref_shape, dtype=np.float64)
        gs["tsnr"] = np.zeros(ref_shape, dtype=np.float64)
    gs["mean"] += m; gs["tsnr"] += tsnr; gs["n"] += 1
    # per-network ROI extraction
    rec = {"subject_id": sid, "group": grp}
    for net, mask in net_masks.items():
        rec[f"mean_{net}"] = float(m[mask].mean())
        rec[f"tsnr_{net}"] = float(tsnr[mask].mean())
    roi_rows.append(rec)
    return None


def handle(res):
    global done
    sid, grp, m, tsnr, shape, affine, err = res
    done += 1
    if err:
        skipped.append((sid, err)); print(f"[{done}/{n_total}] {sid:8s} SKIP: {err}"); return
    ierr = integrate(sid, grp, m, tsnr, shape, affine)
    if ierr:
        skipped.append((sid, ierr)); print(f"[{done}/{n_total}] {sid:8s} SKIP: {ierr}")
    else:
        print(f"[{done}/{n_total}] {sid:8s} ok ({grp})")


if N_WORKERS <= 1:
    for t in tasks:
        handle(process_subject(t))
else:
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(process_subject, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            handle(fut.result())

# ---------- whole-brain group maps ----------
def save_map(arr, fname):
    nib.save(nib.Nifti1Image(arr.astype(np.float32), ref_affine), os.path.join(OUT_DIR, fname))

group_mean = {}
for g in GROUPS:
    gs = group_sums[g]
    if gs["n"] == 0:
        print(f"WARN: group {g} has 0 usable subjects."); continue
    group_mean[g] = {"mean": gs["mean"]/gs["n"], "tsnr": gs["tsnr"]/gs["n"]}
    save_map(group_mean[g]["mean"], f"mean_{g}.nii.gz")
    save_map(group_mean[g]["tsnr"], f"tsnr_{g}.nii.gz")
if all(g in group_mean for g in GROUPS):
    save_map(group_mean["COVID"]["mean"] - group_mean["CONTROL"]["mean"], "mean_diff_COVID_minus_CONTROL.nii.gz")
    save_map(group_mean["COVID"]["tsnr"] - group_mean["CONTROL"]["tsnr"], "tsnr_diff_COVID_minus_CONTROL.nii.gz")

# ---------- per-network ROI table + descriptive group comparison ----------
df_roi = pd.DataFrame(roi_rows)
df_roi.to_csv(os.path.join(OUT_DIR, "network_signal_quality_per_subject.csv"), index=False)

def cohens_d(a, b):  # b - a = COVID - CONTROL
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1-1)*a.var(ddof=1) + (n2-1)*b.var(ddof=1)) / (n1+n2-2))
    return (b.mean() - a.mean()) / sp if sp > 0 else np.nan

summary = []
for net in TARGET_NETWORKS:
    for metric in ["mean", "tsnr"]:
        col = f"{metric}_{net}"
        a = df_roi.loc[df_roi["group"] == "CONTROL", col].values
        b = df_roi.loc[df_roi["group"] == "COVID", col].values
        summary.append({
            "network": net, "metric": metric,
            "control_mean": a.mean(), "covid_mean": b.mean(),
            "diff_covid_minus_control": b.mean() - a.mean(),
            "cohens_d": cohens_d(a, b),
        })
df_summary = pd.DataFrame(summary)
df_summary.to_csv(os.path.join(OUT_DIR, "network_signal_quality_summary.csv"), index=False)

# ---------- console summary ----------
print("\n=== QC dropout maps ===")
print(f"Reference grid: shape={ref_shape}, voxelwise group mean, no resampling of data")
for g in GROUPS:
    print(f"  {g}: n_used = {group_sums[g]['n']}")
print(f"  skipped: {len(skipped)}")
for sid, why in skipped:
    print(f"    - {sid}: {why}")

print("\n=== Per-network signal quality (DESCRIPTIVE; QC, not an inference family) ===")
print(df_summary.round(4).to_string(index=False))
print("\nInterpretation: similar group means / small |d| -> signal quality is "
      "comparable between groups in these networks, making a dropout-driven "
      "explanation of the FC trend unlikely. tSNR on preprocessed data is "
      "descriptive only (no exclusion criterion).")
print(f"\nOutput in: {os.path.abspath(OUT_DIR)}")