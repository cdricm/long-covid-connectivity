"""
Descriptive cohort characterisation (age, sex, severity, symptom availability)
on the final analytical sample. Purely descriptive — no inference, no
confounder test (that is step1_confound_check_demographics.py).

In: config.GROUP_CSV, step0a_qc_summary.csv (step0a_qc output).
Out: step0c_cohort_description.txt.
"""
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

OUT_DIR = config.ensure(config.PRE_ANALYSIS_DIR)
OUT_TXT = OUT_DIR / "step0c_cohort_description.txt"

# ============================ SETTINGS ===================================
ID_COL, GROUP_COL = "ID", "Grupo"
AGE_COL, SEX_COL  = "Edad", "Genero"
SEVERITY_COL      = "CategoríaCOVID"
CLINICAL_COLS     = ["FAS", "EQ-VAS", "MOCA"]          # existence only
SYMPTOM_COLS      = ["DolorDeCabeza", "Fatiga", "Olfato", "Gusto", "Disnea",
                     "DebilidadMuscular", "DolorMuscular", "Confusion",
                     "Comunicacion", "Dormir", "Memoria", "Atencion"]  # existence only
# =========================================================================


def get_final_ids():
    """Final analytical sample IDs via the single source of truth."""
    g = pd.read_csv(config.GROUP_CSV)
    # NIfTI subjects are taken from the step0a QC summary, which defines gate (a).
    qc = config.PRE_ANALYSIS_DIR / "step0a_qc_summary.csv"
    if not qc.exists():
        sys.exit(f"[ABORT] {qc} not found — run step0a first.")
    nii = pd.read_csv(qc)
    nii_ids = sorted(nii.loc[nii["status"] != "no_nii_found", "subject_id"].astype(str))
    included = config.select_included_subjects(
        nii_ids, g, id_col=ID_COL, group_col=GROUP_COL, verbose=False)
    return set(map(str, included)), g

def main():
    final_ids, g = get_final_ids()
    g[ID_COL] = g[ID_COL].astype(str).str.strip()

    df = g[g[ID_COL].isin(final_ids)].copy()
    lines = []

    def log(text=""):
        print(text)
        lines.append(str(text))

    # --- join sanity: do all final IDs have a CSV row? ---
    matched = set(df[ID_COL])
    missing = final_ids - matched
    log("=" * 66)
    log("COHORT DESCRIPTION — FINAL ANALYTICAL SAMPLE")
    log("=" * 66)
    log(f"final IDs (config)      : {len(final_ids)}")
    log(f"matched to CSV rows     : {len(matched)}")
    if missing:
        log(f"[!!] {len(missing)} final IDs have NO CSV row — string-format "
            f"mismatch likely. Fix before trusting numbers:")
        log(f"     {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
    log()

    # --- group sizes ---
    log("--- Group sizes ---")
    log(df[GROUP_COL].value_counts().to_string())
    log()

    # --- AGE: absolute values per group ---
    df[AGE_COL] = pd.to_numeric(df[AGE_COL], errors="coerce")
    log("--- Age (years) per group ---")
    for grp, sub in df.groupby(GROUP_COL):
        a = sub[AGE_COL].dropna()
        log(f"  {grp:<8} n={len(a):3d}  mean={a.mean():5.1f}  sd={a.std():4.1f}  "
              f"min={a.min():.0f}  max={a.max():.0f}  (missing={sub[AGE_COL].isna().sum()})")
    log()

    # --- SEX: n and % per group ---
    log("--- Sex (n, %) per group ---")
    for grp, sub in df.groupby(GROUP_COL):
        vc = sub[SEX_COL].value_counts(dropna=False)
        tot = len(sub)
        parts = [f"{k}={v} ({100*v/tot:.1f}%)" for k, v in vc.items()]
        log(f"  {grp:<8} " + "  ".join(parts))
    log()

    # --- SEVERITY: frequencies (COVID group primarily) ---
    if SEVERITY_COL in df.columns:
        log(f"--- COVID severity ({SEVERITY_COL}) per group ---")
        ct = pd.crosstab(df[GROUP_COL], df[SEVERITY_COL], dropna=False)
        log(ct.to_string())
        log()

    # --- CLINICAL SCORES + SYMPTOMS: availability only ---
    log("--- Clinical scores & symptoms: availability (non-missing / n) ---")
    for col in CLINICAL_COLS + SYMPTOM_COLS:
        if col in df.columns:
            nonmiss = df[col].notna().sum()
            log(f"  {col:<18} {nonmiss:3d}/{len(df)}")
        else:
            log(f"  {col:<18} [column not found]")
    log()
    log("Note: clinical/symptom columns reported as availability only; "
          "absolute statistics intentionally omitted.")
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nSaved: {OUT_TXT}")


if __name__ == "__main__":
    main()