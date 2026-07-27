"""
Descriptive demographic pre-test (Age: Mann-Whitney U; Sex: chi-square) on the
final analytical sample. Descriptive only — not a gate for covariate inclusion.

In: config.GROUP_CSV, config.NII_ROOT, config.select_included_subjects().
Out: analysis_outputs/pre_analysis/step1_confound_check_demographics.txt.

scipy applies Yates' correction only for 2x2 tables; expected-cell counts are
reported so the chi-square approximation can be judged.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import pandas as pd
import numpy as np
from scipy import stats

# =================== SETTINGS ===================
ID_COL_GROUP  = "ID"
GROUP_COL     = "Grupo"
AGE_COL       = "Edad"
SEX_COL       = "Genero"

COVID_LABEL   = "COVID"
CONTROL_LABEL = "CONTROL"
# ================================================


def main():
    OUT_DIR = config.ensure(config.PRE_ANALYSIS_DIR)
    REPORT_PATH = OUT_DIR / "step1_confound_check_demographics.txt"
    _lines = []
    def out(s=""):
        print(s)
        _lines.append(str(s))

    # --- Analytical sample via single gate ---
    group_df = pd.read_csv(config.GROUP_CSV)
    nii = [p.name for p in config.NII_ROOT.iterdir() if p.is_dir()]
    included = config.select_included_subjects(nii, group_df)

    grp = group_df.copy()
    grp[ID_COL_GROUP] = grp[ID_COL_GROUP].astype(str).str.strip()
    retained = [str(s).strip() for s in included]
    df = grp[grp[ID_COL_GROUP].isin(retained)].copy()

    out(f"Analytical sample IDs: {len(retained)} (via config.select_included_subjects)")

    # --- Merge check ---
    out(f"Group-CSV rows total: {len(grp)}")
    out(f"Matched after filtering to retained: {len(df)}")
    missing = set(retained) - set(grp[ID_COL_GROUP])
    if missing:
        out(f"WARNING: {len(missing)} retained IDs NOT found in group CSV (ID-format mismatch?):")
        out(f"  examples: {list(missing)[:5]}")
        out(f"  group-CSV ID examples: {grp[ID_COL_GROUP].head(3).tolist()}")
        out(f"  retained ID examples: {list(retained)[:3]}")

    out("\nGroup label counts (check exact spelling):")
    out(df[GROUP_COL].value_counts(dropna=False).to_string())
    out("Sex label counts (check exact spelling):")
    out(df[SEX_COL].value_counts(dropna=False).to_string())

    # --- Label guards: fail early and clearly if expected labels are absent ---
    present_groups = set(df[GROUP_COL].dropna().unique())
    for lbl in (COVID_LABEL, CONTROL_LABEL):
        if lbl not in present_groups:
            out(f"\n[FAIL] Expected group label '{lbl}' not found in column "
                f"'{GROUP_COL}'. Present labels: {sorted(present_groups)}")
            with open(REPORT_PATH, "w") as f:
                f.write("\n".join(_lines) + "\n")
            raise SystemExit(1)

    df[AGE_COL] = pd.to_numeric(df[AGE_COL], errors="coerce")
    cov  = df[df[GROUP_COL] == COVID_LABEL]
    ctrl = df[df[GROUP_COL] == CONTROL_LABEL]

    def med_iqr(x):
        x = x.dropna()
        q1, q3 = np.percentile(x, [25, 75])
        return np.median(x), q1, q3, len(x)

    # --- AGE: Mann-Whitney U ---
    a_cov, a_ctrl = cov[AGE_COL].dropna(), ctrl[AGE_COL].dropna()
    U, p_age = stats.mannwhitneyu(a_cov, a_ctrl, alternative="two-sided")
    m_c, q1c, q3c, nc = med_iqr(a_cov)
    m_k, q1k, q3k, nk = med_iqr(a_ctrl)

    # --- SEX: Chi2 (with expected-count diagnostics) ---
    sex_tab = pd.crosstab(df[GROUP_COL], df[SEX_COL])
    chi2, p_sex, dof, expected = stats.chi2_contingency(sex_tab)
    min_expected = float(np.min(expected))
    yates_applied = sex_tab.shape == (2, 2)

    # =================== OUTPUT ===================
    out("\n" + "=" * 60)
    out("CONFOUND CHECK — retained sample (DESCRIPTIVE; no covariate adjustment, a priori)")
    out("=" * 60)
    out(f"N COVID   = {len(cov)}")
    out(f"N CONTROL = {len(ctrl)}")
    out(f"N total   = {len(df)}")

    out("\n--- AGE (Mann-Whitney U) ---")
    out(f"COVID   : median={m_c:.1f}  IQR=[{q1c:.1f}, {q3c:.1f}]  n={nc}")
    out(f"CONTROL : median={m_k:.1f}  IQR=[{q1k:.1f}, {q3k:.1f}]  n={nk}")
    out(f"U={U:.1f}   p={p_age:.4f}")
    if nc + nk < len(df):
        out(f"NOTE: {len(df) - nc - nk} subject(s) excluded from age test (non-numeric/missing age).")

    out("\n--- SEX (chi-square) ---")
    out(sex_tab.to_string())
    out(f"chi2={chi2:.3f}   df={dof}   p={p_sex:.4f}")
    out(f"table shape={sex_tab.shape}   Yates correction applied={yates_applied} "
        f"(scipy applies Yates only for 2x2)")
    out(f"min expected cell count={min_expected:.2f}")
    if min_expected < 5:
        out("NOTE: min expected cell count < 5 -> chi-square approximation is questionable")
    out("=" * 60)

    # --- Write report ---
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(_lines) + "\n")
    out(f"\nReport written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()