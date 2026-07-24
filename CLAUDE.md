# CLAUDE.md — Repository conventions and review protocol

Bachelor's thesis project: graph-theoretic analysis of resting-state fMRI functional
connectivity in Long COVID (San Martín cohort). The analysis is **complete**. This
repository is in the **final review phase before submission**: no new analyses, no
new methodological decisions, only verification and hygiene.

---

## 1. Single source of truth

`METHODS_DECISIONS.md` (repo root) is the binding reference for every methodological
choice. If this file and METHODS_DECISIONS.md ever conflict, **METHODS_DECISIONS.md
wins**.

Read the relevant section of METHODS_DECISIONS.md **before** assessing any script.
Never reconstruct a methodological decision from memory or infer it from the code.
If the answer is not in METHODS_DECISIONS.md, say so and ask — do not guess.

`config.py` is the single source of truth for paths, constants, the `FC_METHOD`
switch, the sign-strategy primitives and the cohort gate. Scripts must bind to
config rather than repeat literals.

---

## 2. Frozen state (do not violate)

- **Cohort freeze:** N = 162 (123 COVID / 39 CONTROL), defined by
  `config.EXCLUDED_SUBJECTS` and `config.select_included_subjects()` (below).
- Attrition: 202 NIfTI subjects − 6 no valid group label (gate b) − 34
  `EXCLUDED_SUBJECTS` (gate c: 2 scan-duration + 32 motion) = 162.
- `config.EXCLUDED_SUBJECTS` holds 40 entries: the 34 gate-c quality exclusions plus
  the 6 gate-b subjects, listed redundantly for documentation completeness. The
  latter are already removed by the group-label gate, so N is unaffected.
- **All N = 194 era results are void.** Never use them as verification anchors
  (e.g. d = −0.32 within-Limbic, NBS 142-edge component, k = 2 clustering).
- Both analytical arms are COMPLETE. Changing code that alters numbers would
  require re-running a frozen arm — flag it, never do it silently.

---

## 3. Two arms, one switch

`config.FC_METHOD` is either `"pearson"` (primary) or `"partial"` (sensitivity).
All outputs are namespaced under `analysis_outputs/<FC_METHOD>/`.

- **Pearson arm:** all three atlases (Schaefer-400, Schaefer-100, AAL);
  `CONFIRMATORY_SIGN_STRATEGY = "positive"`.
- **Partial arm:** Schaefer-400 only (tests estimator robustness, not atlas
  robustness); `CONFIRMATORY_SIGN_STRATEGY = "absolute"`; Ledoit-Wolf shrinkage.

Which atlases actually run is decided by which per-atlas wrapper is launched, not
hardcoded in analysis scripts. Diagnostic/QC scripts should work on whatever
matrices exist on disk rather than encoding the arm-to-atlas mapping — except where
a script is genuinely arm-specific (e.g. the Ledoit-Wolf lambda check), which then
carries an explicit guard.

---

## 4. Review protocol

When asked to review a script, produce exactly this structure:

### `<filename>` — check against METHODS_DECISIONS.md

**1. Purpose per MD**
One sentence: which step or decision in METHODS_DECISIONS.md this script implements,
with the section reference.

**2. Conformity check**
A table: `#` | MD requirement (section) | in the code | status (✅ / ⚠️ / ❌).
Legend: ✅ conformant · ⚠️ conformant but unclear/undocumented · ❌ deviation.

**3. Deviations and open points**
Only for ⚠️/❌: the exact location, what deviates, what METHODS_DECISIONS.md
requires. Propose a fix; do not apply methodological changes unilaterally.

**4. Reproducibility**
Seeds set · paths from config (no hardcoded literals) · constants not duplicated ·
deterministic output · cohort bound to `config.select_included_subjects()` ·
outputs written to the expected tree.

**5. Submission hygiene**
Comments describe the current state only · no dead code paths or commented-out
alternatives · no `_retired_*` remnants in active scripts · no N = 194 artefacts ·
docstring with purpose and In/Out present · no debug prints or hardcoded overrides.

**6. Verdict**
`READY` / `READY AFTER CLARIFICATION (n points)` / `NOT READY (n points)` — plus one
sentence of justification.

---

## 5. Hard rules

### Comments
- Describe the **current state only**. No references to previous versions, changes
  made, or decisions that were considered and rejected.
  Forbidden: "the cohort binding the old script lacked", "(Weg 2)", "lowered for
  memory safety" where nothing was lowered.
- No instructions to a future editor. Forbidden: "Adjust QC_CSV path if needed",
  "use X for high-res".
- No content-free markers. A comment that only restates the function name below it
  should be deleted.
- Comments that carry real information stay: why a value was chosen, why an API
  argument is set that way, what a non-obvious construct guards against.

### Cohort binding
Every script that operates on subjects binds to `config.select_included_subjects()`.
**No glob-based subject selection.** Where a script reads cached matrices, it
cross-checks the files on disk against the config-defined sample and either aborts
or warns explicitly — a stale cache must never enter silently.

### Fail loudly
Prefer `sys.exit` / `assert` with a clear message over silent fallbacks. A script
that cannot verify its preconditions should stop, not proceed on a proxy.

### Constants
Analysis constants live in config: `FISHER_CLIP`, `N_PERMUTATIONS`, `FDR_ALPHA`,
`DENSITY_SUPPORT_POINTS`, `AUC_RANGE_CONFIRMATORY`, `AUC_RANGE_SENSITIVITY`,
`MODULARITY_N_RUNS`, `NBS_THRESHOLDS`, `NBS_PRIMARY_THRESHOLD`, `NBS_TAIL`,
`TARGET_NETWORKS_BY_ARM`, `SEED`, `N_JOBS_DEFAULT`.
Local literals that duplicate a config constant are a ⚠️ even when the value
matches. METHODS_DECISIONS.md §8 explicitly exempts already-run Pearson scripts
from retrofitting where this is result-neutral — check there before flagging.

Exception: the frozen cohort size (162, 123/39) appears as a hardcoded
literal in assert statements across step3f, step4d/e/f and step5a. These
are deliberate guards against silent cohort drift and must NOT be replaced
by a config value — an assertion that derives its expectation from the same
source it checks is void.

### Seeds
Base seed 42 throughout. Permutations use `SeedSequence(42).spawn(n_tests)`.
A `default_rng` created **inside** a loop repeats the same sequence each iteration —
check for this, it has produced a real artefact in this repo before.

### Group-blind vs group-aware
Steps 3a/3b/3c are a **group-blind diagnostic**: they compare sign strategies on
construction properties, with no group inference. The confirmatory sign strategy
must not depend on group differences — that would be circular. Any group contrast
appearing in these steps is a finding worth flagging.

---

## 6. Naming conventions

- Output files carry a step prefix (`step0a_qc_summary.csv`) when written into a
  **flat** output directory such as `pre_analysis/`.
- No prefix is needed when the step is already in the path
  (`step2_pipeline/comet_matrices/CP0001_connectivity_comet.npy`).
- Documentation figures (`doc_vis_*`) follow their own scheme deliberately.
- When a file is renamed, all readers, docstring `In:`/`Out:` lines and print
  statements must follow. This has been missed repeatedly — check for it.

---

## 7. Wrapper plus pipeline-module pattern

Per-atlas wrappers are thin: they supply atlas configuration and paths and call a
shared pipeline module. Wrappers for the same step must be structurally identical
apart from the atlas configuration. Analysis logic lives in the pipeline module,
never duplicated across wrappers.

When a shared pipeline function's signature changes, **every caller must be
checked** — this repo has broken twice on exactly that.

---

## 8. Language and style

- Code, comments, docstrings and all console/file output: **English**.
- Docstring format: purpose, then `In:` and `Out:` lines naming the actual paths.
- Conversation with the user: **German**.

---

## 9. What not to do

- Do not make new methodological decisions. Flag gaps, contradictions or
  unsupported claims and ask.
- Do not change anything that would alter published numbers without saying so
  explicitly and waiting for confirmation.
- Do not mass-apply fixes across files. One script at a time, verified before
  moving on.
- Do not delete: this repo archives via the `_retired_*/` pattern.
- Do not reformat or restructure code that was not part of the review request.
- Archived code under `archive/` or `_retired_*/` is out of review scope.
  It documents a previous state and is neither checked nor updated.

### Never write to the frozen output tree

Review and verification must never write to `analysis_outputs/`. The
analysis is frozen and its numbers are cited in the thesis; a review must
not be able to alter them, even when a re-run would reproduce them exactly.

- Verification runs go to a temporary directory outside the tree, which is
  removed afterwards.
- If a change cannot be verified without writing to the production tree,
  stop and ask instead of running it.
- Before any change, state explicitly whether it is result-neutral. If that
  cannot be established from the code alone, it is not result-neutral until
  proven otherwise — ask before proceeding.
- Renaming a key, label or column is not automatically cosmetic. Check
  first whether it feeds a sort order, a groupby, a seed assignment or a
  filename.

---

## 10. Environment

Python 3.11 (`comet` venv), COMET 1.2.4 (version-locked), Nilearn 0.13.1, numpy,
pandas, scipy, matplotlib, networkx, statsmodels, joblib; BCT via COMET; scikit-learn
for the partial-arm lambda recomputation. Linux, PyCharm.
Data root: `/mnt/d87cc26d-5470-443c-81c1-e09b68ee4730/Cedric/`.

Nilearn 0.13.1 API notes: no `darkness` argument in `plot_surf_roi`; use
`interpolation="nearest_most_frequent"` (not `"nearest"`) for surface projection of
discrete label maps.
