"""Central configuration: single source of truth for paths, reproducibility
constants, graph-construction primitives (sign strategy, thresholding), and
the subject-exclusion gate. All scripts import from this module.

In: none (defines paths/constants only).
Out: none (imported by every step).

CONFIRMATORY_SIGN_STRATEGY defaults to None outside pearson/partial so
downstream steps fail loudly (assert) instead of silently picking a strategy.
"""

from pathlib import Path
import numpy as np

# --- Base locations ------------------------------------------------------------
BASE      = Path("/mnt/d87cc26d-5470-443c-81c1-e09b68ee4730/Cedric")
NII_ROOT  = BASE / "FunImgARWSDCFN"
GROUP_CSV = BASE / "ResumenRespuestasBasico.csv"
AO        = BASE / "analysis_outputs"

# Method-agnostic pre-analysis QC/cohort tree (step0, step1b): NOT namespaced under
# FC_METHOD, since QC and cohort definition are computed once, before either arm runs.
PRE_ANALYSIS_DIR = AO / "pre_analysis"

# --- Connec1tivity method: single switch for the whole pipeline -----------------
# Change FC_METHOD here to re-run with a different connectivity estimator. ALL
# outputs are namespaced under analysis_outputs/<FC_METHOD>/ (ATLAS_DIRS, CROSS,
# THESIS_FIGURES below), so a method change writes a fresh tree and never serves
# cached results from another method. Both branches in make_connectivity() stay
# active; only this constant selects which one runs.
#
FC_METHOD = "pearson"   # primary arm
#FC_METHOD = "partial"     # partial-correlation sensitivity arm (Ledoit-Wolf shrinkage)

# Covariance estimator for partial correlation. LedoitWolf shrinkage is required:
# for Schaefer-400 (p=400 > T≈150) the unregularised precision matrix is singular
# / unstable. Ledoit-Wolf is analytic, parameter-free (no lambda, no CV) and is
# estimated per subject from the data (Varoquaux & Craddock, 2013).
# Ignored when FC_METHOD != "partial".
PARTIAL_COV_ESTIMATOR = "LedoitWolf"

# Networks localized in step4f (ROI-level within-network localization).
# Fixed illustrative selection, NOT a data-driven threshold: Pearson = the three
# largest descriptive Family-B within-network trends (step4d); partial = none
# (cells ~50/50, no network-level trend to localize).
TARGET_NETWORKS_BY_ARM = {
    "pearson": ["Cont", "Limbic", "Default"],
    "partial": [],
}

def make_connectivity(time_series, *, diagonal, fisher_z, tril=False):
    """Return a COMET connectivity estimator for the configured FC_METHOD.

    The pipeline calls .estimate() (and .postproc() if present) on the returned
    object; keep that interface when adding methods. Both branches are kept
    active so a single change to FC_METHOD switches the whole pipeline.
    """
    if FC_METHOD == "pearson":
        from comet.connectivity import Static_Pearson
        return Static_Pearson(time_series=time_series, diagonal=diagonal,
                              fisher_z=fisher_z, tril=tril)
    if FC_METHOD == "partial":
        from comet.connectivity import Static_Partial
        return Static_Partial(time_series=time_series,
                              cov_estimator=PARTIAL_COV_ESTIMATOR,
                              diagonal=diagonal, fisher_z=fisher_z, tril=tril)
    raise ValueError(f"Unsupported FC_METHOD: {FC_METHOD!r}")


# --- Sign strategy for Family-A graph construction -----------------------------
# Applies ONLY to the path-/clustering-based Family-A metrics (Global Efficiency,
# Mean Clustering, Assortativity) and the exploratory nodal-strength decomposition
# (step3f). Does NOT apply to:
#   - Modularity : ALWAYS signed (modularity_louvain_und_sign; Rubinov & Sporns
#                  2011), strategy-invariant.
#   - Family B   : ALWAYS signed (mean conditional connectivity needs signs).
#   - Family C   : ALWAYS signed-all-edges (NBS tests per-edge group differences).
#
# WORKFLOW (both arms run the same code; the diagnostic CONCLUSION differs):
#   1. steps 3a/3b/3c run over ALL of SIGN_STRATEGIES as a DIAGNOSTIC comparison
#      (connectedness, fragmentation, descriptive metric behaviour; NO group
#      inference -> group-blind).
#   2. ONE strategy is chosen for the confirmatory AUC analysis (3d/3e), based on
#      group-blind construction properties, and recorded in CONFIRMATORY_SIGN_STRATEGY.
#        - Pearson arm: the negative subgraph is DEGENERATE (group-blind 3a/3b/3c
#          diagnostic on N=162: cannot reach target densities in-range, degenerate
#          metrics), so positive-only is the established choice; no real decision is
#          needed -> "positive".
#        - Partial arm: the negative subgraph is SUBSTANTIVE (~47% of edges), so the
#          diagnostic is genuinely informative and the choice is made afterwards.
#   3. the confirmatory AUC analysis (3d/3e) runs on that single strategy only.
# The choice must NOT depend on group differences (that would be circular); it is
# documented in METHODS_DECISIONS.
SIGN_STRATEGIES = ["positive", "negative", "absolute"]   # diagnostic comparison set

# --- Diagnostic strategy set + graph-construction primitives -------------------
# DIAGNOSTIC_SIGN_STRATEGIES is the LIST that steps 3a/3b/3c iterate over in the
# group-blind cross-strategy diagnostic; the confirmatory choice (single strategy,
# CONFIRMATORY_SIGN_STRATEGY below) is derived from inspecting that diagnostic.
# Kept identical to SIGN_STRATEGIES but named explicitly so the diagnostic scope is
# one named constant. apply_sign_strategy() and proportional_threshold() are the two
# shared graph-construction primitives; they live here (single source of truth) so
# all steps call config.* rather than importing from a step module. Both are
# correlation-method-agnostic (Pearson r or partial correlation): which arm runs is
# decided by FC_METHOD (matrix namespacing) upstream, not by these functions.
DIAGNOSTIC_SIGN_STRATEGIES = list(SIGN_STRATEGIES)  # ["positive", "negative", "absolute"]


def apply_sign_strategy(C, strategy):
    """Map a (possibly signed) connectivity matrix to the non-negative weight matrix
    on which density-based thresholding is defined, per sign strategy. Diagonal is
    zeroed; input signs are otherwise untouched.

    Correlation-method-agnostic. Under Pearson the negative subgraph is degenerate;
    under partial it is substantive (~47% of edges) — which is why 3a/3b/3c sweep
    all three strategies as a group-blind diagnostic before the confirmatory choice.

      positive : keep positive edges          -> W_ij = C_ij  if C_ij > 0 else 0
      negative : keep |negative| edges         -> W_ij = -C_ij if C_ij < 0 else 0
      absolute : keep magnitude of all edges   -> W_ij = |C_ij|
    """
    if strategy == "positive":
        out = np.where(C > 0, C, 0.0)
    elif strategy == "negative":
        out = np.where(C < 0, -C, 0.0)
    elif strategy == "absolute":
        out = np.abs(C)
    else:
        raise ValueError(f"Unknown sign strategy: {strategy!r} "
                         f"(expected one of {SIGN_STRATEGIES})")
    np.fill_diagonal(out, 0.0)
    return out


def proportional_threshold(M, density):
    """Keep the top-`density` fraction of off-diagonal edges by weight; return a
    symmetric matrix.

    Own NumPy implementation rather than the toolbox routine, for three reasons:
    sign handling is explicit (positive-only selection happens upstream in
    apply_sign_strategy, before thresholding); it returns n_keep/n_target, which the
    target-reach diagnostic needs; and the thresholding definition is visible here in
    the code.

    `density` is the fraction of ALL POSSIBLE undirected edges (n*(n-1)/2), the
    standard proportional-thresholding definition (van Wijk et al., 2010), NOT a
    fraction of the currently non-zero edges. If the requested count exceeds the
    available non-zero edges (sparse subgraph at high density), it is capped and the
    caller detects this via n_keep < n_target.

    Input M is assumed non-negative (output of apply_sign_strategy). Returns
    (W, n_keep, n_target).
    """
    n = M.shape[0]
    iu = np.triu_indices(n, k=1)
    w = M[iu]
    n_target = int(round(density * w.size))
    n_nonzero = int(np.sum(w > 0))
    n_keep = min(n_target, n_nonzero)
    if n_keep == 0:
        return np.zeros_like(M), 0, n_target
    idx = np.argpartition(w, -n_keep)[-n_keep:]
    mask = np.zeros_like(w, dtype=bool)
    mask[idx] = True
    out = np.zeros_like(M)
    out[iu[0][mask], iu[1][mask]] = w[mask]
    out = out + out.T
    return out, n_keep, n_target


# Confirmatory strategy for 3d/3e, coupled to the arm:
#   - Pearson: "positive" automatically (established; negative subgraph degenerate).
#   - Partial: None until the 3a/3b/3c diagnostic has been inspected, then set to
#     one of SIGN_STRATEGIES. Left None so 3d/3e FAIL LOUDLY (assert) if run before
#     the choice is made, rather than silently defaulting to a strategy.
if FC_METHOD == "pearson":
    CONFIRMATORY_SIGN_STRATEGY = "positive"
elif FC_METHOD == "partial":
    CONFIRMATORY_SIGN_STRATEGY = "absolute"
else:
    CONFIRMATORY_SIGN_STRATEGY = None

# --- Per-atlas output trees (namespaced by FC_METHOD) --------------------------
# Which atlases actually run is decided by which per-atlas wrappers are launched,
# NOT here. In practice: the Pearson arm runs all three (atlas-robustness check);
# the partial sensitivity arm runs Schaefer-400 only (it tests ESTIMATOR robustness,
# not atlas robustness — atlas robustness is already established in the Pearson arm).
# All three entries stay defined so either arm can address any atlas.
ATLAS_DIRS = {
    "schaefer400": AO / FC_METHOD / "schaefer400",   # primary atlas (both arms)
    "schaefer100": AO / FC_METHOD / "schaefer100",   # Pearson robustness check
    "aal":         AO / FC_METHOD / "aal",           # Pearson robustness check
}


def atlas_dir(atlas: str, subdir: str | None = None, *,
              strategy: str | None = None, cross_strategy: bool = False) -> Path:
    """Return a path inside an atlas output tree.

    Sign-neutral (default, strategy=None, cross_strategy=False):
        {FC_METHOD}/{atlas}/{subdir}
        For step2, Family B, Family C, QC, cohort steps. Existing calls unchanged.

    Per-strategy (strategy="positive"|"negative"|"absolute"):
        {FC_METHOD}/{atlas}/family_A/{strategy}/{subdir}
        Sign-separated Family-A tree. Used by the diagnostic metric step (3c) once
        per strategy, and by the confirmatory step (3d/3e) with the single
        CONFIRMATORY_SIGN_STRATEGY.

    Cross-strategy (cross_strategy=True):
        {FC_METHOD}/{atlas}/family_A/_cross_strategy/{subdir}
        Strategy-spanning diagnostics/comparisons (step3a sweep, step3b diagnose,
        cross-strategy overview plots). Strategy-invariant.

    Examples:
        atlas_dir("schaefer400", "step2_pipeline")                       # sign-neutral
        atlas_dir("schaefer400", "step3c_metrics", strategy="negative")  # per-strategy
        atlas_dir("schaefer400", "step3a_sweep", cross_strategy=True)    # diagnostic
    """
    base = ATLAS_DIRS[atlas]
    if cross_strategy and strategy is not None:
        raise ValueError("pass either strategy=... or cross_strategy=True, not both")
    if cross_strategy:
        root = base / "family_A" / "_cross_strategy"
    elif strategy is not None:
        if strategy not in SIGN_STRATEGIES:
            raise ValueError(f"bad strategy: {strategy!r}, expected one of {SIGN_STRATEGIES}")
        root = base / "family_A" / strategy
    else:
        root = base
    return root / subdir if subdir else root


def step3f_dir(atlas: str, subdir: str | None = None, *, strategy: str) -> Path:
    """Nodal-strength (step3f) output tree — sign-dependent but kept SEPARATE from
    the confirmatory family_A tree (exploratory != confirmatory in the layout):
        {FC_METHOD}/{atlas}/step3f_nodal_strength/{strategy}/{subdir}
    """
    if strategy not in SIGN_STRATEGIES:
        raise ValueError(f"bad strategy: {strategy!r}, expected one of {SIGN_STRATEGIES}")
    root = ATLAS_DIRS[atlas] / "step3f_nodal_strength" / strategy
    return root / subdir if subdir else root


# --- Cross-atlas / atlas-independent outputs -----------------------------------
CROSS = AO / FC_METHOD / "cross_atlas"
CROSS_DIRS = {
    "step0_subject_data":   CROSS / "step0_subject_data",
    "step1_inventory":      CROSS / "step1_inventory",
    "step2_fc_diagnostics":          CROSS / "step2_fc_diagnostics",
    "step3c_sweep_overview":        CROSS / "step3c_sweep_overview",
    "step3c_clustering_diagnostic": CROSS / "step3c_clustering_diagnostic",
    "step3e_forest_plot":           CROSS / "step3e_forest_plot",
}

THESIS_FIGURES = AO / FC_METHOD / "thesis_figures"

# --- Reproducibility -----------------------------------------------------------
SEED           = 42
N_JOBS_DEFAULT = 6   # default joblib parallelism for embarrassingly parallel steps
JOBLIB_TEMP    = BASE / "tmp_joblib"   # loky worker temp dir — avoids overflowing OS partition

# --- Analysis constants (single source of truth; METHODS_DECISIONS §3-§6) ------
# Centralised so every step references config.* rather than a local literal. Values
# match METHODS_DECISIONS; changing one here changes it pipeline-wide.
#
# Fisher-z clip: bounds arctanh so r near ±1 cannot produce an unbounded z. 0.9999
# (-> |z| <= ~4.95) is the canonical value (§3). Used for all Fisher-z transforms
# (aggregated FC and inference input).
#   Note: step3f (exploratory nodal, null) and step5a (NBS, FWER-null) were run with
#   0.999999 before centralisation; the difference is result-neutral (both are clear
#   nulls and the clip only affects the few edges with r extremely close to 1), so
#   they are NOT re-run. New/edited code uses config.FISHER_CLIP.
FISHER_CLIP = 0.9999

# Inference (§6): naive permutation count; NBS is FWER, Families A/B are FDR-BH.
N_PERMUTATIONS = 10000
FDR_ALPHA      = 0.05

# Thresholding sweep (§4): proportional-threshold support points and AUC ranges.
DENSITY_SUPPORT_POINTS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
AUC_RANGE_CONFIRMATORY = (0.10, 0.25)   # primary; FDR family A
AUC_RANGE_SENSITIVITY  = (0.05, 0.50)   # broad; declared sensitivity, no FDR

# Modularity Q* (§5): mean over N Louvain runs on the signed full matrix.
MODULARITY_N_RUNS = 100

# NBS (§6 Family C): cluster-forming thresholds; FWER via component-extent null.
NBS_THRESHOLDS        = [2.5, 3.1, 3.5]
NBS_PRIMARY_THRESHOLD = 3.1
NBS_TAIL              = "both"

# Direction convention for all group contrasts: d = COVID - CONTROL
# (MD §6). Index 0 is the reference group, index 1 the contrast group.
GROUP_ORDER = ("CONTROL", "COVID")

# --- Subject exclusion / analytical-sample definition --------------------------
# The analytical sample is defined HERE and applied ONCE, at the connectivity-
# matrix construction step (step2), via select_included_subjects(). Downstream
# steps never re-apply exclusion; they operate on whatever matrices exist on disk.
#
# A subject is INCLUDED iff all three hold:
#   (a) it has NIfTI data (present in the NIfTI root),
#   (b) it has a valid group label in GROUP_CSV (one of VALID_GROUPS), and
#   (c) it is not listed in EXCLUDED_SUBJECTS below.
#
# IMPORTANT: exclusion is produced by BOTH gate (b) and gate (c). The 6 "no
# metadata row" entries below are gate-b subjects (no valid group label) that
# are ALSO listed here for documentation completeness; they are already removed
# by the `labeled` check in select_included_subjects() regardless of their
# presence in this dict, so listing them is redundant but harmless — it does
# not change which subjects are excluded or the resulting N. The 34 gate-c
# entries (2 scan-duration + 32 motion) are the genuine quality exclusions.
#
# Frozen analytical sample: N = 162 (123 COVID + 39 CONTROL).
# Attrition from 202 NIfTI subjects: −2 scan-duration QC, −6 no-CSV-metadata,
# −32 motion (28 COVID / 4 CONTROL; supervisor-curated motion-exclusion list).
# Differential motion exclusion (COVID vs. CONTROL) is reported as a limitation.
VALID_GROUPS = {"COVID", "CONTROL"}
EXCLUDED_SUBJECTS = {
    "CP0004": "scan duration 307 s — outside acquisition protocol",
    "CP0011": "no metadata row in ResumenRespuestasBasico.csv",
    "CP0015": "no metadata row in ResumenRespuestasBasico.csv",
    "CP0038": "motion > threshold detected in preprocessing",
    "CP0061": "motion > threshold detected in preprocessing",
    "CP0062": "motion > threshold detected in preprocessing",
    "CP0067": "motion > threshold detected in preprocessing",
    "CP0072": "motion > threshold detected in preprocessing",
    "CP0084": "motion > threshold detected in preprocessing",
    "CP0087": "no metadata row in ResumenRespuestasBasico.csv",
    "CP0096": "motion > threshold detected in preprocessing",
    "CP0100": "motion > threshold detected in preprocessing",
    "CP0105": "motion > threshold detected in preprocessing",
    "CP0106": "no metadata row in ResumenRespuestasBasico.csv",
    "CP0108": "motion > threshold detected in preprocessing",
    "CP0110": "motion > threshold detected in preprocessing",
    "CP0114": "motion > threshold detected in preprocessing",
    "CP0115": "motion > threshold detected in preprocessing",
    "CP0117": "motion > threshold detected in preprocessing",
    "CP0128": "motion > threshold detected in preprocessing",
    "CP0131": "motion > threshold detected in preprocessing",
    "CP0135": "motion > threshold detected in preprocessing",
    "CP0136": "motion > threshold detected in preprocessing",
    "CP0140": "scan duration 351 s — outside acquisition protocol",
    "CP0144": "no metadata row in ResumenRespuestasBasico.csv",
    "CP0153": "motion > threshold detected in preprocessing",
    "CP0159": "motion > threshold detected in preprocessing",
    "CP0162": "motion > threshold detected in preprocessing",
    "CP0170": "motion > threshold detected in preprocessing",
    "CP0180": "motion > threshold detected in preprocessing",
    "CP0185": "motion > threshold detected in preprocessing",
    "CP0188": "motion > threshold detected in preprocessing",
    "CP0193": "no metadata row in ResumenRespuestasBasico.csv",
    "CP0202": "motion > threshold detected in preprocessing",
    "CP0203": "motion > threshold detected in preprocessing",
    "CP0214": "motion > threshold detected in preprocessing",
    "CP0225": "motion > threshold detected in preprocessing",
    "CP0233": "motion > threshold detected in preprocessing",
    "CP0234": "motion > threshold detected in preprocessing",
    "CP0238": "motion > threshold detected in preprocessing",
}


def select_included_subjects(nii_subjects, group_df, *, id_col="ID",
                             group_col="Grupo", verbose=True):
    """Return the analytical sample as a sorted list of included subject IDs.

    Single source of truth for who enters the analysis (gates a+b+c, see section
    header). Both the group-label gate (b) and the explicit EXCLUDED_SUBJECTS
    gate (c) remove subjects — the label gate alone already drops subjects that
    have no group assignment.
    """
    nii = set(map(str, nii_subjects))
    group_of = {str(i): str(g) for i, g in zip(group_df[id_col], group_df[group_col])}
    labeled = {s for s, g in group_of.items() if g in VALID_GROUPS}
    included = sorted(s for s in nii if s in labeled and s not in EXCLUDED_SUBJECTS)
    if verbose:
        dropped_excluded = sorted(s for s in nii if s in labeled and s in EXCLUDED_SUBJECTS)
        n_nolabel = len([s for s in nii if s not in labeled])
        by_group = {}
        for s in included:
            g = group_of[s]
            by_group[g] = by_group.get(g, 0) + 1
        print("=== Analytical sample selection (single gate, applied in step2) ===")
        print(f"  (a) subjects with NIfTI data        : {len(nii)}")
        print(f"  (b) dropped (no valid group label)  : {n_nolabel}")
        print(f"  (c) dropped (EXCLUDED_SUBJECTS)      : {len(dropped_excluded)}  {dropped_excluded}")
        print(f"  --> included analytical sample      : {len(included)}")
        for g in sorted(by_group):
            print(f"        {g:<8} : {by_group[g]}")
    return included


# --- Helper --------------------------------------------------------------------
def ensure(path: Path) -> Path:
    """Create a directory (and parents) if missing; return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path