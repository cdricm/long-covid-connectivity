"""
FC-matrix sanity/integrity checks across all atlases (shape, symmetry, diagonal,
range, NaN/Inf), plus a descriptive global mean-FC group comparison as a sanity
anchor.

In: cached matrices under analysis_outputs/<FC_METHOD>/<atlas>/step2_pipeline/
    comet_matrices, config.GROUP_CSV.
Out: config.atlas_dir(<atlas>, "step2_diagnose")/{fc_diagnostics.csv/.txt,
     fc_distributions.png, fc_value_histogram.png};
     config.CROSS_DIRS["step2_fc_diagnostics"]/{cross_atlas_summary.txt,
     cross_atlas_table.csv}.

The global mean-FC comparison here is descriptive only (Fisher-z, z-then-mean;
no permutation/FDR/covariates) — the confirmatory test is step4e.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== Config =================================================================
ATLASES_ALL = [
    {"label": "Schaefer-400", "dir": "schaefer400", "expected_n": 400},
    {"label": "Schaefer-100", "dir": "schaefer100", "expected_n": 100},
    {"label": "AAL (SPM12)",  "dir": "aal",         "expected_n": 116},
]

# Tolerances
SYM_TOL       = 1e-6    # float32-realistic sanity bound on raw asymmetry
                        # (float32 eps ~1.2e-7; mathematically symmetric matrices show ~1e-8 noise)
DIAG_EXPECTED = 0.0
VALUE_MIN     = -1.0
VALUE_MAX     = 1.0
VALUE_TOL     = 1e-9    # float slack for [-1, 1] check
ARCTANH_CLIP  = 0.9999  # clip r before arctanh to keep Fisher-z finite at |r|->1

GROUP_CSV_ID    = "ID"
GROUP_CSV_GROUP = "Grupo"
COVID_LABEL     = "COVID"
CONTROL_LABEL   = "CONTROL"


# ===== Helpers ================================================================
def fisher_z_mean(offdiag_finite):
    """z-then-mean: arctanh(r) per edge (clipped), then mean. Returns NaN if empty."""
    if offdiag_finite.size == 0:
        return np.nan
    r = np.clip(offdiag_finite, -ARCTANH_CLIP, ARCTANH_CLIP)
    return float(np.mean(np.arctanh(r)))


def cohens_d(a, b):
    """Cohen's d (pooled SD) with 95% CI. a, b are 1-D arrays. Group a minus group b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan
    s_pooled = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if s_pooled == 0:
        return np.nan, np.nan, np.nan
    d = (a.mean() - b.mean()) / s_pooled
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    return float(d), float(d - 1.96 * se), float(d + 1.96 * se)


def load_group_map():
    """subject_id -> group label, restricted to valid groups."""
    g = pd.read_csv(config.GROUP_CSV)
    g[GROUP_CSV_ID] = g[GROUP_CSV_ID].astype(str).str.strip()
    return {row[GROUP_CSV_ID]: str(row[GROUP_CSV_GROUP]).strip()
            for _, row in g.iterrows()
            if str(row[GROUP_CSV_GROUP]).strip() in (COVID_LABEL, CONTROL_LABEL)}


# ===== Per-atlas check ========================================================
def check_atlas(atlas_cfg, group_map):
    label      = atlas_cfg["label"]
    expected_n = atlas_cfg["expected_n"]

    mat_dir = config.atlas_dir(atlas_cfg["dir"], "step2_pipeline") / "comet_matrices"

    files = sorted(glob.glob(str(mat_dir / "*_connectivity_comet.npy")))
    if not files:
        print(f"  WARN: no matrices found for {label}, skipping")
        return None

    out_dir = config.ensure(config.atlas_dir(atlas_cfg["dir"], "step2_diagnose"))

    print(f"\n{'='*70}\n{label}\n{'='*70}")
    print(f"Matrices : {mat_dir}")
    print(f"Output   : {out_dir}\n")

    group_df = pd.read_csv(config.GROUP_CSV)
    nii = [p.name for p in config.NII_ROOT.iterdir() if p.is_dir()]
    included = set(config.select_included_subjects(nii, group_df, verbose=False))
    found = {Path(f).name.replace("_connectivity_comet.npy", "") for f in files}
    if found != included:
        sys.exit(f"[ABORT] {label}: matrices on disk do not match the analytical "
                 f"sample (on disk {len(found)}, expected {len(included)}; "
                 f"extra {sorted(found - included)}, missing {sorted(included - found)})")

    rows = []
    for f in files:
        subj = Path(f).name.replace("_connectivity_comet.npy", "")
        row  = {"subject": subj, "group": group_map.get(subj, "UNKNOWN")}

        try:
            M = np.load(f)
        except Exception as e:
            row["error"] = f"load_failed: {e}"
            rows.append(row)
            continue

        # Raw asymmetry is float32 rounding noise for these matrices
        # (Pearson r / precision-matrix-derived partials -> mathematically
        # symmetric). Record it descriptively, then enforce exact symmetry for
        # all downstream statistics.
        if M.ndim == 2 and M.shape[0] == M.shape[1]:
            row["max_asymmetry"] = float(np.nanmax(np.abs(M - M.T)))
            M = 0.5 * (M + M.T)
        else:
            row["max_asymmetry"] = np.nan

        row["shape_ok"]    = (M.shape == (expected_n, expected_n))
        row["shape"]       = f"{M.shape[0]}x{M.shape[1]}" if M.ndim == 2 else str(M.shape)
        row["n_nan"]       = int(np.isnan(M).sum())
        row["n_inf"]       = int(np.isinf(M).sum())

        # Finite-only stats for remaining checks
        finite = M[np.isfinite(M)]
        row["finite_min"]  = float(finite.min()) if finite.size else np.nan
        row["finite_max"]  = float(finite.max()) if finite.size else np.nan

        # Symmetry: raw asymmetry already recorded above; M is now exactly
        # symmetric by construction. This flags only genuinely large raw
        # asymmetry (a real problem), not float32 noise.
        if M.ndim == 2 and M.shape[0] == M.shape[1]:
            row["symmetric"] = bool(row["max_asymmetry"] < SYM_TOL)
        else:
            row["symmetric"] = False

        # Diagonal
        if M.ndim == 2 and M.shape[0] == M.shape[1]:
            diag = np.diag(M)
            row["max_diag_abs"] = float(np.nanmax(np.abs(diag - DIAG_EXPECTED)))
            row["diag_ok"]      = bool(row["max_diag_abs"] < 1e-10)
        else:
            row["max_diag_abs"] = np.nan
            row["diag_ok"]      = False

        # Value range
        row["range_ok"] = bool(
            (np.isnan(row["finite_min"]) or row["finite_min"] >= VALUE_MIN - VALUE_TOL) and
            (np.isnan(row["finite_max"]) or row["finite_max"] <= VALUE_MAX + VALUE_TOL)
        )

        # Off-diagonal statistics (upper triangle)
        if M.ndim == 2 and M.shape[0] == M.shape[1]:
            iu          = np.triu_indices_from(M, k=1)
            offdiag     = M[iu]
            offdiag_fin = offdiag[np.isfinite(offdiag)]
            if offdiag_fin.size:
                row["fc_mean"]        = float(np.mean(offdiag_fin))          # raw-r mean (descriptive)
                row["fc_mean_z"]      = fisher_z_mean(offdiag_fin)           # z-then-mean (for group d)
                row["fc_std"]         = float(np.std(offdiag_fin))
                row["fc_median"]      = float(np.median(offdiag_fin))
                row["fc_p95"]         = float(np.percentile(np.abs(offdiag_fin), 95))
                row["pct_strong_pos"] = float(np.mean(offdiag_fin > 0.3) * 100)
                row["pct_strong_neg"] = float(np.mean(offdiag_fin < -0.3) * 100)
            else:
                for k in ("fc_mean", "fc_mean_z", "fc_std", "fc_median", "fc_p95",
                          "pct_strong_pos", "pct_strong_neg"):
                    row[k] = np.nan

        row["all_checks_pass"] = bool(
            row["shape_ok"] and row["symmetric"] and row["diag_ok"] and
            row["range_ok"] and row["n_nan"] == 0 and row["n_inf"] == 0
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "fc_diagnostics.csv", index=False)

    # ===== Summary text =======================================================
    n = len(df)
    L = [f"FC matrix diagnostics — {label}", "=" * 60,
         f"N matrices            : {n}",
         f"Expected shape        : ({expected_n}, {expected_n})", ""]

    # Group composition (sanity: should match frozen cohort)
    if "group" in df.columns:
        gc = df["group"].value_counts(dropna=False).to_dict()
        L.append(f"Group composition     : {gc}")
        n_unknown = int((df["group"] == "UNKNOWN").sum())
        if n_unknown:
            L.append(f"  WARNING: {n_unknown} subject(s) with no group label in GROUP_CSV "
                     f"(ID-format mismatch?) — excluded from group comparison")
        L.append("")

    if "error" in df.columns:
        n_err = df["error"].notna().sum()
        L.append(f"Load errors           : {n_err}")
        if n_err:
            for _, r in df[df["error"].notna()][["subject", "error"]].iterrows():
                L.append(f"  {r['subject']}: {r['error']}")

    L += [
        f"Shape OK              : {int(df['shape_ok'].sum())}/{n}",
        f"Symmetric             : {int(df['symmetric'].sum())}/{n}  "
            f"(max raw asymmetry: {df['max_asymmetry'].max():.2e}; "
            f"symmetrised before stats)",
        f"Diagonal == 0         : {int(df['diag_ok'].sum())}/{n}  "
            f"(max |diag|: {df['max_diag_abs'].max():.2e})",
        f"Values in [-1, 1]     : {int(df['range_ok'].sum())}/{n}  "
            f"(global min: {df['finite_min'].min():.4f}, "
            f"global max: {df['finite_max'].max():.4f})",
        f"NaN-free              : {int((df['n_nan'] == 0).sum())}/{n}  "
            f"(total NaN: {df['n_nan'].sum()})",
        f"Inf-free              : {int((df['n_inf'] == 0).sum())}/{n}  "
            f"(total Inf: {df['n_inf'].sum()})",
        f"ALL CHECKS PASS       : {int(df['all_checks_pass'].sum())}/{n}",
        ""
    ]

    # FC distribution stats
    if "fc_mean" in df.columns:
        L += [
            "Off-diagonal FC distribution (per subject):",
            f"  fc_mean (raw r) : mean {df['fc_mean'].mean():.3f}, "
                f"sd {df['fc_mean'].std():.3f}, "
                f"range [{df['fc_mean'].min():.3f}, {df['fc_mean'].max():.3f}]",
            f"  fc_std          : mean {df['fc_std'].mean():.3f}, "
                f"sd {df['fc_std'].std():.3f}",
            f"  fc_median       : mean {df['fc_median'].mean():.3f}",
            f"  % edges > +0.3  : mean {df['pct_strong_pos'].mean():.1f}%, "
                f"sd {df['pct_strong_pos'].std():.1f}",
            f"  % edges < -0.3  : mean {df['pct_strong_neg'].mean():.1f}%, "
                f"sd {df['pct_strong_neg'].std():.1f}",
            ""
        ]

    # ===== Global mean-FC group comparison (DESCRIPTIVE sanity anchor) ========
    if "fc_mean_z" in df.columns and "group" in df.columns:
        cov  = df[df["group"] == COVID_LABEL]
        ctrl = df[df["group"] == CONTROL_LABEL]
        L += ["Global mean-FC group comparison (DESCRIPTIVE sanity check;",
              "Fisher-z, z-then-mean; no test/FDR/covariates — see step4 for inference):"]
        if len(cov) >= 2 and len(ctrl) >= 2:
            mc, mk = cov["fc_mean_z"].mean(), ctrl["fc_mean_z"].mean()
            d, lo, hi = cohens_d(cov["fc_mean_z"], ctrl["fc_mean_z"])
            L += [
                f"  COVID   : mean z-FC {mc:+.4f}  (n={len(cov)})",
                f"  CONTROL : mean z-FC {mk:+.4f}  (n={len(ctrl)})",
                f"  Cohen's d (COVID - CONTROL): {d:+.3f}  [95% CI {lo:+.3f}, {hi:+.3f}]",
                "  -> small |d| means later network effects are not a trivial global shift.",
                ""
            ]
        else:
            L += ["  insufficient group sizes for comparison", ""]

    # Outlier subjects (>3 SD on fc_mean), reported per group (descriptive only)
    if "fc_mean" in df.columns and df["fc_mean"].notna().any():
        z = (df["fc_mean"] - df["fc_mean"].mean()) / df["fc_mean"].std()
        out_mask = np.abs(z) > 3
        outliers = df[out_mask]
        L.append(f"fc_mean outliers (|z| > 3): {len(outliers)} "
                 f"(descriptive only; cohort is frozen, no exclusion derived)")
        for _, r in outliers.iterrows():
            sub_z = float(z[df["subject"] == r["subject"]].values[0])
            L.append(f"  {r['subject']} [{r['group']}] (z = {sub_z:+.2f})")
        L.append("")

    # Subjects failing any check
    failed = df[~df["all_checks_pass"]][["subject", "shape_ok", "symmetric",
                                          "diag_ok", "range_ok", "n_nan", "n_inf"]]
    if len(failed):
        L.append(f"Subjects failing checks: {len(failed)}")
        for _, r in failed.iterrows():
            issues = []
            if not r["shape_ok"]:  issues.append("shape")
            if not r["symmetric"]: issues.append("asymmetric")
            if not r["diag_ok"]:   issues.append("diag")
            if not r["range_ok"]:  issues.append("range")
            if r["n_nan"] > 0:     issues.append(f"NaN={r['n_nan']}")
            if r["n_inf"] > 0:     issues.append(f"Inf={r['n_inf']}")
            L.append(f"  {r['subject']}: {', '.join(issues)}")

    txt = "\n".join(L)
    print(txt)
    with open(out_dir / "fc_diagnostics.txt", "w") as f:
        f.write(txt)

    # ===== Plots ==============================================================
    if "fc_mean" in df.columns and df["fc_mean"].notna().any():
        cov  = df[df["group"] == COVID_LABEL]
        ctrl = df[df["group"] == CONTROL_LABEL]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

        # FC mean, by group
        bins = np.linspace(df["fc_mean"].min(), df["fc_mean"].max(), 30)
        axes[0].hist(cov["fc_mean"].dropna(),  bins=bins, color="firebrick",
                     alpha=0.6, edgecolor="white", label=f"COVID (n={len(cov)})")
        axes[0].hist(ctrl["fc_mean"].dropna(), bins=bins, color="steelblue",
                     alpha=0.6, edgecolor="white", label=f"CONTROL (n={len(ctrl)})")
        axes[0].set_xlabel("FC mean (off-diagonal, raw r)"); axes[0].set_ylabel("N subjects")
        axes[0].legend(fontsize=8); axes[0].set_title(f"{label} — FC mean by group")

        # FC std
        axes[1].hist(df["fc_std"].dropna(), bins=30, color="gray", edgecolor="white")
        axes[1].set_xlabel("FC std (off-diagonal)"); axes[1].set_ylabel("N subjects")
        axes[1].set_title(f"{label} — FC std")

        # strong edges
        axes[2].hist(df["pct_strong_pos"].dropna(), bins=30, color="steelblue",
                     edgecolor="white", alpha=0.7, label="r > +0.3")
        axes[2].hist(df["pct_strong_neg"].dropna(), bins=30, color="firebrick",
                     edgecolor="white", alpha=0.5, label="r < -0.3")
        axes[2].set_xlabel("% of edges"); axes[2].set_ylabel("N subjects")
        axes[2].legend(fontsize=8); axes[2].set_title(f"{label} — strong edges")

        plt.tight_layout()
        plt.savefig(out_dir / "fc_distributions.png", dpi=140)
        plt.close()

    # Pooled value histogram (sample of subjects to limit memory)
    rng = np.random.default_rng(config.SEED)
    sample = rng.choice(files, size=min(20, len(files)), replace=False)
    pooled = []
    for f in sample:
        M = np.load(f)
        if M.ndim == 2 and M.shape[0] == M.shape[1]:
            M = 0.5 * (M + M.T)   # match the symmetrisation applied above
            iu = np.triu_indices_from(M, k=1)
            pooled.append(M[iu])
    if pooled:
        pooled = np.concatenate(pooled)
        pooled = pooled[np.isfinite(pooled)]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(pooled, bins=100, color="steelblue", edgecolor="none")
        ax.axvline(0, color="k", linestyle="-",  linewidth=0.8)
        ax.axvline(VALUE_MIN, color="r", linestyle="--", linewidth=1, label="±1 bound")
        ax.axvline(VALUE_MAX, color="r", linestyle="--", linewidth=1)
        ax.set_xlabel("FC value"); ax.set_ylabel("Edge count")
        ax.set_title(f"{label} — pooled FC value distribution "
                     f"({len(sample)} subjects, seed={config.SEED})")
        ax.legend(); plt.tight_layout()
        plt.savefig(out_dir / "fc_value_histogram.png", dpi=140)
        plt.close()

    return df


# ===== Main ===================================================================
def main():
    group_map = load_group_map()
    cross = []
    for cfg in ATLASES:
        df = check_atlas(cfg, group_map)
        if df is None:
            continue

        # Global mean-FC d for the cross-atlas table
        cov  = df[df["group"] == COVID_LABEL]["fc_mean_z"] if "fc_mean_z" in df.columns else pd.Series(dtype=float)
        ctrl = df[df["group"] == CONTROL_LABEL]["fc_mean_z"] if "fc_mean_z" in df.columns else pd.Series(dtype=float)
        d_glob, _, _ = cohens_d(cov, ctrl) if len(cov) and len(ctrl) else (np.nan, np.nan, np.nan)

        cross.append({
            "Atlas"             : cfg["label"],
            "N"                 : len(df),
            "Expected ROIs"     : cfg["expected_n"],
            "Shape OK"          : f"{int(df['shape_ok'].sum())}/{len(df)}",
            "Symmetric"         : f"{int(df['symmetric'].sum())}/{len(df)}",
            "Diag == 0"         : f"{int(df['diag_ok'].sum())}/{len(df)}",
            "Values in [-1,1]"  : f"{int(df['range_ok'].sum())}/{len(df)}",
            "NaN-free"          : f"{int((df['n_nan']==0).sum())}/{len(df)}",
            "Inf-free"          : f"{int((df['n_inf']==0).sum())}/{len(df)}",
            "ALL PASS"          : f"{int(df['all_checks_pass'].sum())}/{len(df)}",
            "fc_mean (mean±sd)" : f"{df['fc_mean'].mean():.3f}±{df['fc_mean'].std():.3f}",
            "global mean-FC d"  : f"{d_glob:+.3f}" if np.isfinite(d_glob) else "n/a",
        })

    if cross:
        summary_dir = config.ensure(config.CROSS_DIRS["step2_fc_diagnostics"])
        df_cross = pd.DataFrame(cross)
        df_cross.to_csv(summary_dir / "cross_atlas_table.csv", index=False)

        L = ["FC matrix diagnostics — cross-atlas summary", "=" * 70, "",
             "global mean-FC d = COVID - CONTROL, Fisher-z (z-then-mean), "
             "DESCRIPTIVE sanity anchor only (no test/FDR/covariates).", ""]
        L.append(df_cross.to_string(index=False))
        txt = "\n".join(L)
        print("\n\n" + txt)
        with open(summary_dir / "cross_atlas_summary.txt", "w") as f:
            f.write(txt)


if __name__ == "__main__":
    main()