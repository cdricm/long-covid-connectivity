"""
Ledoit-Wolf shrinkage intensity (lambda) x group check (partial arm, Schaefer-400
only). Discharges the TP-length x group confound (Fisher exact p=.020) by testing
whether lambda — the actual channel through which TP length could bias partial-
correlation estimates — differs between groups.

In: cached timeseries under analysis_outputs/<FC_METHOD>/schaefer400/
    step2_pipeline/comet_timeseries, config.GROUP_CSV.
Out: config.CROSS_DIRS["step2_fc_diagnostics"]/{shrinkage_lambda.csv/.txt/.png}.

Lambda is not stored in the cached matrices (COMET saves only the final partial
matrix), so it is independently recomputed here via sklearn's LedoitWolf
(deterministic, no seed/CV).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import glob
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt

# ===== Config =================================================================
ATLAS = {"label": "Schaefer-400", "dir": "schaefer400", "n_roi": 400}
COVID_LABEL, CONTROL_LABEL = "COVID", "CONTROL"
GROUP_CSV_ID, GROUP_CSV_GROUP = "ID", "Grupo"


def load_group_map():
    g = pd.read_csv(config.GROUP_CSV)
    g[GROUP_CSV_ID] = g[GROUP_CSV_ID].astype(str).str.strip()
    return {row[GROUP_CSV_ID]: str(row[GROUP_CSV_GROUP]).strip()
            for _, row in g.iterrows()
            if str(row[GROUP_CSV_GROUP]).strip() in (COVID_LABEL, CONTROL_LABEL)}


def main():
    if config.FC_METHOD != "partial":
        sys.exit(f"[SKIP] Lambda check applies to the partial arm only "
                 f"(FC_METHOD={config.FC_METHOD}).")

    group_map = load_group_map()
    rows = []

    ts_dir = config.atlas_dir(ATLAS["dir"], "step2_pipeline") / "comet_timeseries"
    files = sorted(glob.glob(str(ts_dir / "*_timeseries_comet.npy")))

    group_df = pd.read_csv(config.GROUP_CSV)
    nii = [p.name for p in config.NII_ROOT.iterdir() if p.is_dir()]
    included = set(config.select_included_subjects(nii, group_df, verbose=False))
    found = {Path(f).name.replace("_timeseries_comet.npy", "") for f in files}
    if found != included:
        sys.exit(f"[ABORT] timeseries on disk do not match the analytical sample "
                 f"(on disk {len(found)}, expected {len(included)}; "
                 f"extra {sorted(found - included)}, missing {sorted(included - found)})")

    for f in files:
        subj = Path(f).name.replace("_timeseries_comet.npy", "")
        grp = group_map.get(subj, "UNKNOWN")
        X = np.load(f).astype(float)
        if X.ndim != 2:
            rows.append({"subject": subj, "group": grp, "n_tp": np.nan,
                         "lambda": np.nan, "note": "not_2d"})
            continue

        # Orient to (T, P): for Schaefer-400 the ROI axis is unambiguously 400.
        n0, n1 = X.shape
        if n1 == ATLAS["n_roi"] and n0 != ATLAS["n_roi"]:
            pass                      # already (T, P)
        elif n0 == ATLAS["n_roi"] and n1 != ATLAS["n_roi"]:
            X = X.T                   # was (P, T)
        else:
            rows.append({"subject": subj, "group": grp, "n_tp": np.nan,
                         "lambda": np.nan, "note": "no_400_axis"})
            continue

        T, P = X.shape
        Xz = (X - X.mean(0)) / X.std(0, ddof=1)   # z-score per ROI (correlation scale)
        lw = LedoitWolf(assume_centered=True).fit(Xz)
        rows.append({"subject": subj, "group": grp, "n_tp": int(T),
                     "lambda": float(lw.shrinkage_), "note": ""})

    df = pd.DataFrame(rows)
    out_dir = config.ensure(config.CROSS_DIRS["step2_fc_diagnostics"])
    df.to_csv(out_dir / "shrinkage_lambda.csv", index=False)

    L = ["Ledoit-Wolf shrinkage intensity (lambda) x group check", "=" * 60,
         "Atlas: Schaefer-400 (partial arm; estimator-robustness scope).",
         "lambda recomputed from timeseries (sklearn LedoitWolf, Ledoit & Wolf 2004).",
         "H0: lambda does not differ by group -> TP imbalance does not bias",
         "    partial-correlation estimation systematically between groups.", ""]

    n_skip = int((df["note"].isin(["not_2d", "no_400_axis"])).sum())
    if n_skip:
        L.append(f"WARNING: {n_skip} subject(s) skipped (shape issue); see CSV 'note'.")
        L.append("")

    sub = df[df["lambda"].notna()]
    cov  = sub[sub["group"] == COVID_LABEL]["lambda"]
    ctrl = sub[sub["group"] == CONTROL_LABEL]["lambda"]

    # TP composition per group (the actual imbalance under test)
    L.append("TP composition (subjects per run length):")
    for grp in (COVID_LABEL, CONTROL_LABEL):
        g = sub[sub["group"] == grp]
        comp = ", ".join(f"T={int(t)}: {int((g['n_tp'] == t).sum())}"
                         for t in sorted(g["n_tp"].unique()))
        L.append(f"  {grp:8s}: n={len(g):3d}  ({comp})")
    L.append("")

    L.append(f"lambda range : [{sub['lambda'].min():.3f}, {sub['lambda'].max():.3f}]")
    if len(cov) >= 2 and len(ctrl) >= 2:
        t_stat, p = stats.ttest_ind(cov, ctrl, equal_var=False)
        sp = np.sqrt(((len(cov)-1)*cov.var(ddof=1) + (len(ctrl)-1)*ctrl.var(ddof=1))
                     / (len(cov)+len(ctrl)-2))
        d = (cov.mean() - ctrl.mean()) / sp if sp > 0 else np.nan
        L.append(f"COVID   : mean {cov.mean():.4f} (n={len(cov)})")
        L.append(f"CONTROL : mean {ctrl.mean():.4f} (n={len(ctrl)})")
        L.append(f"Welch t = {t_stat:+.2f}, p = {p:.3f}, Cohen's d = {d:+.3f}")
    L.append("")

    rng = np.random.default_rng(config.SEED)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for grp, col in ((COVID_LABEL, "firebrick"), (CONTROL_LABEL, "steelblue")):
        g = sub[sub["group"] == grp]
        jit = rng.normal(0, 1.5, len(g))
        ax.scatter(g["n_tp"] + jit, g["lambda"], s=18, alpha=0.6, color=col,
                   label=f"{grp} (n={len(g)})")
    ax.set_xlabel("Timepoints (T)"); ax.set_ylabel("Ledoit-Wolf lambda")
    ax.set_title(ATLAS["label"]); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "shrinkage_lambda.png", dpi=140)
    plt.close()

    txt = "\n".join(L)
    print(txt)
    with open(out_dir / "shrinkage_lambda.txt", "w") as f:
        f.write(txt)


if __name__ == "__main__":
    main()