"""
Visual QC plots based on existing QC output files.

Input:
- qc_summary.csv
- qc_exclusion_summary.csv
- qc_report.md

These files are expected in:
.../analysis_outputs/step0_check_subject_data/qc_outputs/

Purpose:
- Do NOT rerun full QC.
- Load already computed QC table.
- Plot PASS example + flagged/review/excluded subjects.
- Use issue-specific visualization:
    - DVARS issue  -> DVARS time series
    - tSNR issue   -> tSNR map
    - duration     -> text panel
    - coverage     -> mean BOLD / coverage panel
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from nilearn.plotting import plot_carpet, plot_epi, plot_stat_map
from nilearn.image import mean_img, new_img_like
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


# =============================================================
# SETTINGS
# =============================================================
DATA_DIR = config.NII_ROOT

QC_OUT_DIR = config.CROSS_DIRS["step0_subject_data"] / "qc_outputs"

VIS_DIR = config.ensure(config.CROSS_DIRS["step0_subject_data"] / "visualizations" / "qc_checks")

QC_SUMMARY_CSV = QC_OUT_DIR / "qc_summary.csv"
QC_EXCLUSION_CSV = QC_OUT_DIR / "qc_exclusion_summary.csv"
QC_REPORT_MD = QC_OUT_DIR / "qc_report.md"

PASS_EXAMPLE = "CP0189"

# Limit number of flagged subjects in the big combined plot
MAX_FLAGGED_SUBJECTS_TO_PLOT = 12

# Set to None if you want all flagged subjects
# MAX_FLAGGED_SUBJECTS_TO_PLOT = None

FIXED_CUT_COORDS = [40]

TSNR_HARD_MIN = 30

OUT_COMBINED_FIG = VIS_DIR / "qc_visual_review_pass_vs_flagged_subjects.png"
OUT_PLOT_SUBJECTS_CSV = VIS_DIR / "qc_subjects_in_visual_review_plot.csv"


# =============================================================
# HELPERS
# =============================================================
def find_bold_file(subject_id: str) -> Path:
    subject_dir = DATA_DIR / subject_id
    candidates = sorted(list(subject_dir.glob("*.nii*")))

    if not candidates:
        raise FileNotFoundError(f"No .nii or .nii.gz file found for {subject_id} in {subject_dir}")

    return candidates[0]


def load_img(subject_id: str):
    bold_path = find_bold_file(subject_id)
    return nib.load(str(bold_path))


def make_brain_mask(img):
    data = img.get_fdata(dtype=np.float32, caching="unchanged")
    mean_vol = data.mean(axis=3)

    nonzero = mean_vol[mean_vol > 0]

    if len(nonzero) == 0:
        raise ValueError("No nonzero voxels found.")

    thresh = np.percentile(nonzero, 10)
    mask_arr = (mean_vol > thresh).astype(np.int8)

    return new_img_like(img, mask_arr)


def compute_tsnr_image(img):
    data = img.get_fdata(dtype=np.float32, caching="unchanged")

    mean_t = data.mean(axis=3)
    std_t = data.std(axis=3)

    with np.errstate(divide="ignore", invalid="ignore"):
        tsnr = np.where(std_t > 0, mean_t / std_t, 0).astype(np.float32)

    tsnr = np.clip(tsnr, 0, 200)

    return new_img_like(img, tsnr)



def assign_primary_plot_type(row):
    """
    Decide which issue-specific panel should be shown in column 3.
    Priority:
    1. PASS example -> tSNR map
    2. tSNR issue -> tSNR map
    3. duration issue -> text panel
    4. coverage issue -> mean BOLD again / coverage note
    5. technical / NaN -> text panel
    """

    if row["label_for_plot"] == "PASS example":
        return "pass"

    tsnr_flag = str(row.get("tSNR_flag", "ok"))
    coverage_flag = str(row.get("coverage_flag", "ok"))
    nan_flag = str(row.get("nan_flag", "ok"))
    duration_check = str(row.get("duration_check", "ok"))

    if tsnr_flag.startswith(("flag", "fail")):
        return "tsnr"

    if duration_check == "non_protocol":
        return "duration"

    if coverage_flag == "flag_low":
        return "coverage"

    if nan_flag == "flag_nan":
        return "text"

    return "text"


def get_sort_key(row):
    """
    Sort plot rows:
    1. PASS example
    2. REVIEW
    3. EXCLUDE_QC
    4. EXCLUDE_DURATION
    5. EXCLUDE_TECHNICAL
    """

    if row["label_for_plot"] == "PASS example":
        return 0

    verdict = row.get("verdict", "")

    order = {
        "REVIEW": 1,
        "EXCLUDE_QC": 2,
        "EXCLUDE_DURATION": 3,
        "EXCLUDE_TECHNICAL": 4,
    }

    return order.get(verdict, 99)


def build_plot_dataframe(df):
    """
    Select PASS example plus subjects that need review/exclusion.
    """

    if PASS_EXAMPLE not in df["subject_id"].values:
        raise ValueError(f"PASS_EXAMPLE {PASS_EXAMPLE} not found in qc_summary.csv")

    pass_row = df[df["subject_id"] == PASS_EXAMPLE].copy()
    pass_row["label_for_plot"] = "PASS example"

    flagged_df = df[
        df["verdict"].isin([
            "REVIEW",
            "EXCLUDE_QC",
            "EXCLUDE_DURATION",
            "EXCLUDE_TECHNICAL",
        ])
    ].copy()

    flagged_df["label_for_plot"] = flagged_df["verdict"]

    # Sort strongest QC issues first inside flagged subjects
    sort_cols = []
    ascending = []

    if "tSNR_median" in flagged_df.columns:
        sort_cols.append("tSNR_median")
        ascending.append(True)

    if sort_cols:
        flagged_df = flagged_df.sort_values(sort_cols, ascending=ascending)

    if MAX_FLAGGED_SUBJECTS_TO_PLOT is not None:
        flagged_df = flagged_df.head(MAX_FLAGGED_SUBJECTS_TO_PLOT)

    plot_df = pd.concat([pass_row, flagged_df], axis=0).copy()

    plot_df["sort_order"] = plot_df.apply(get_sort_key, axis=1)
    plot_df["primary_plot_type"] = plot_df.apply(assign_primary_plot_type, axis=1)

    plot_df = plot_df.sort_values(["sort_order", "subject_id"])

    return plot_df


# =============================================================
# SINGLE SUBJECT PLOTS
# =============================================================
def plot_single_subject(row):
    """
    Saves per-subject visual QC panels.
    """

    sid = row["subject_id"]
    label = row["label_for_plot"]
    plot_type = row["primary_plot_type"]

    subject_out_dir = VIS_DIR / f"{label}_{sid}"
    subject_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[single] plotting {sid} ({label}, {plot_type})")

    img = load_img(sid)
    mean_subj = mean_img(img, copy_header=True)

    # mean BOLD
    fig = plt.figure(figsize=(10, 3))
    plot_epi(
        mean_subj,
        display_mode="ortho",
        figure=fig,
        title=f"{sid} – mean BOLD volume",
        cut_coords=(0, 0, 0),
    )
    fig.savefig(subject_out_dir / "mean_bold.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # carpet
    try:
        brain_mask = make_brain_mask(img)

        fig, ax = plt.subplots(figsize=(10, 4))
        plot_carpet(
            img,
            mask_img=brain_mask,
            axes=ax,
            title=f"{sid} – carpet plot",
            detrend=True,
            standardize="zscore_sample",
        )
        fig.savefig(subject_out_dir / "carpet.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    except Exception as e:
        print(f"  carpet plot failed for {sid}: {e}")

    # issue-specific third panel
    if plot_type in ["tsnr", "pass"]:
        tsnr_img = compute_tsnr_image(img)

        fig = plt.figure(figsize=(10, 3))
        plot_stat_map(
            tsnr_img,
            display_mode="ortho",
            figure=fig,
            title=f"{sid} – tSNR map",
            cut_coords=(0, 0, 0),
            cmap="viridis",
            colorbar=True,
            vmax=120,
            threshold=10,
        )
        fig.savefig(subject_out_dir / "tsnr.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    elif plot_type == "duration":
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")

        duration_min = row.get("duration_sec", np.nan) / 60
        n_vol = row.get("n_volumes", np.nan)
        tr = row.get("TR_sec", np.nan)
        reason = row.get("reason", "duration non-protocol")

        ax.text(
            0.5,
            0.5,
            f"{sid} – duration issue\n\n"
            f"Duration: {duration_min:.2f} min\n"
            f"Volumes: {n_vol:.0f}\n"
            f"TR: {tr:.2f} s\n\n"
            f"{reason}",
            ha="center",
            va="center",
            fontsize=12,
        )

        fig.tight_layout()
        fig.savefig(subject_out_dir / "duration_issue.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"  saved in {subject_out_dir}")


# =============================================================
# COMBINED FIGURE
# =============================================================
def make_combined_figure(plot_df):
    """
    Combined figure:
    rows = PASS example + flagged subjects
    columns:
    1. mean BOLD
    2. carpet plot
    3. issue-specific panel
    """

    n_rows = len(plot_df)
    fig_height = max(5, 3.8 * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(18, fig_height),
        squeeze=False,
    )

    for row_idx, (_, row) in enumerate(plot_df.iterrows()):
        sid = row["subject_id"]
        verdict = row.get("verdict", "")
        reason = row.get("reason", "—")
        label = row["label_for_plot"]
        plot_type = row["primary_plot_type"]

        print(f"[combined] plotting {sid} ({label}, {plot_type})")

        img = load_img(sid)
        mean_subj = mean_img(img, copy_header=True)

        row_title = f"{label}: {sid}"

        if verdict == "REVIEW":
            row_title += "  [REVIEW]"
        elif str(verdict).startswith("EXCLUDE"):
            row_title += f"  [{verdict}]"

        # -------------------------
        # Column 1: mean BOLD
        # -------------------------
        plot_epi(
            mean_subj,
            axes=axes[row_idx, 0],
            display_mode="z",
            cut_coords=FIXED_CUT_COORDS,
            title=f"{row_title}\nmean BOLD",
        )

        # -------------------------
        # Column 2: carpet plot
        # -------------------------
        try:
            brain_mask = make_brain_mask(img)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                plot_carpet(
                    img,
                    mask_img=brain_mask,
                    axes=axes[row_idx, 1],
                    title="carpet plot",
                    detrend=True,
                    standardize="zscore_sample",
                )

        except Exception as e:
            axes[row_idx, 1].axis("off")
            axes[row_idx, 1].text(
                0.5,
                0.5,
                f"Carpet plot failed:\n{e}",
                ha="center",
                va="center",
                fontsize=10,
            )

        # -------------------------
        # Column 3: issue-specific panel
        # -------------------------
        ax = axes[row_idx, 2]

        if plot_type in ["tsnr", "pass"]:
            tsnr_img = compute_tsnr_image(img)

            plot_stat_map(
                tsnr_img,
                axes=ax,
                display_mode="z",
                cut_coords=FIXED_CUT_COORDS,
                title=f"tSNR map\nmedian={row.get('tSNR_median', np.nan):.1f}",
                cmap="viridis",
                colorbar=True,
                vmax=120,
                threshold=10,
            )

        elif plot_type == "duration":
            ax.axis("off")

            duration_min = row.get("duration_sec", np.nan) / 60
            n_vol = row.get("n_volumes", np.nan)
            tr = row.get("TR_sec", np.nan)

            ax.text(
                0.5,
                0.62,
                "DURATION ISSUE",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )

            ax.text(
                0.5,
                0.38,
                f"Duration: {duration_min:.2f} min\n"
                f"Volumes: {n_vol:.0f}\n"
                f"TR: {tr:.2f} s\n\n"
                f"{reason}",
                ha="center",
                va="center",
                fontsize=10,
            )

        elif plot_type == "coverage":
            plot_epi(
                mean_subj,
                axes=ax,
                display_mode="z",
                cut_coords=FIXED_CUT_COORDS,
                title=f"Coverage issue\ncoverage={row.get('coverage_fraction', np.nan) * 100:.1f}%",
            )

        else:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                f"{verdict}\n\n{reason}",
                ha="center",
                va="center",
                fontsize=10,
            )

    fig.suptitle(
        "Visual QC review: PASS example vs flagged subjects",
        fontsize=18,
        y=0.995,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(OUT_COMBINED_FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nCombined figure written:\n{OUT_COMBINED_FIG}")


# =============================================================
# MAIN
# =============================================================
def main():
    if not QC_SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing file: {QC_SUMMARY_CSV}")

    if not QC_EXCLUSION_CSV.exists():
        print(f"Warning: {QC_EXCLUSION_CSV} not found. Using qc_summary.csv only.")

    if not QC_REPORT_MD.exists():
        print(f"Warning: {QC_REPORT_MD} not found. Continuing without markdown report.")

    df = pd.read_csv(QC_SUMMARY_CSV)

    required_cols = ["subject_id", "verdict"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column missing from qc_summary.csv: {col}")

    print(f"Loaded QC summary: {QC_SUMMARY_CSV}")
    print(f"Subjects in QC table: {len(df)}")

    print("\nVerdict distribution:")
    print(df["verdict"].value_counts(dropna=False))

    plot_df = build_plot_dataframe(df)

    plot_df.to_csv(OUT_PLOT_SUBJECTS_CSV, index=False)

    print("\nSubjects selected for visual review:")
    print(plot_df[["subject_id", "verdict", "reason", "primary_plot_type"]])

    # Single-subject plots
    for _, row in plot_df.iterrows():
        plot_single_subject(row)

    # Combined overview plot
    make_combined_figure(plot_df)

    print("\nAll visual QC plots written.")
    print(f"Visual review subject list:\n{OUT_PLOT_SUBJECTS_CSV}")
    print(f"Combined plot:\n{OUT_COMBINED_FIG}")


if __name__ == "__main__":
    main()