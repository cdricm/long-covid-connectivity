"""
Extract and plot the BOLD time series of a single Schaefer-400 ROI
for a given example subject. Intended for quick visual inspection.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker

# =========================
# SETTINGS
# =========================
SUBJECT_ID = "CP0189"
ROI_INDEX  = 0        # Python index: 0 = first ROI
N_ROIS     = 400
ATLAS_NAME = "schaefer400"

fmri_file  = config.NII_ROOT / SUBJECT_ID / "Filtered_4DVolume.nii.gz"
output_dir = config.ensure(config.CROSS_DIRS["step0_subject_data"] / "visualizations" / "one_roi_signal")


def main():
    # FETCH SCHAEFER 400 ATLAS
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=N_ROIS,
        yeo_networks=7,
        resolution_mm=2
    )

    atlas_img = atlas.maps
    labels = atlas.labels

    # EXTRACT ROI TIME SERIES
    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=True,
        detrend=True,
        low_pass=None,
        high_pass=None,
        t_r=None
    )

    time_series = masker.fit_transform(str(fmri_file))

    roi_signal = time_series[:, ROI_INDEX]
    roi_name = (
        labels[ROI_INDEX].decode("utf-8")
        if isinstance(labels[ROI_INDEX], bytes)
        else labels[ROI_INDEX]
    )

    # make filename safe for Linux
    safe_roi_name = roi_name.replace("/", "_").replace(" ", "_")

    # OUTPUT
    print(f"Subject: {SUBJECT_ID}")
    print(f"ROI index: {ROI_INDEX}")
    print(f"ROI name: {roi_name}")
    print(f"Time points: {len(roi_signal)}")
    print(roi_signal)

    # PLOT
    fig, ax = plt.subplots(figsize=(8 / 2.54, 9 / 2.54))

    ax.plot(roi_signal, linewidth=1)
    ax.set_xlabel("Time point")
    ax.set_ylabel("Standardized BOLD signal")

    plt.tight_layout()

    output_file = output_dir / f"{SUBJECT_ID}_ROI_{ROI_INDEX + 1}_{safe_roi_name}.png"
    plt.savefig(output_file, dpi=600)

    print(f"\nPlot saved to:\n{output_file}")

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
