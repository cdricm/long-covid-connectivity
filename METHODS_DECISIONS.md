# METHODS DECISIONS — Long COVID rs-fMRI Graph-Theoretic Analysis

**Status:** Binding methodological specification for the full re-run on the frozen N=162 cohort.
**Purpose:** Single authoritative reference. Every decision below is fixed *a priori*, before computation. Deviations require explicit re-decision and documentation.
**Cohort freeze:** defined by `config.EXCLUDED_SUBJECTS` (40 entries: 34 gate-c quality exclusions + 6 gate-b subjects listed redundantly for documentation completeness; the latter are already removed by the group-label gate, so N is unaffected) together with `config.select_included_subjects()`.
---

## 0. Reference Hierarchy (applies throughout)

The *Handbook of Functional MRI Data Analysis* (Poldrack, Mumford & Nichols, 2011) is the default methodological framework **only where still current**. Where the book has a gap or is outdated, the current literature standard takes precedence. Three cases:

1. Book covers it **and still current** → book is primary reference; modern literature supplementary.
2. Book does **not** cover it (esp. post-2011) → current literature is primary; book not retrofitted.
3. Book covers it but is **superseded** → modern literature is primary; book cited only as historical context, with the newer standard named explicitly.

Every deviation from the book is named and justified in the Methods section. Primary literature is preferred throughout; Poldrack (2011/2019) is the sole admitted textbook. Non-Poldrack textbooks (incl. Fornito et al., 2016) are replaced by primary literature. Every citable claim must be actually supported by the named source; DOI/URL is provided for every source.

---

## 1. Analysis Cohort & Inclusion/Exclusion

- **Final cohort:** N = 162 (COVID n = 123 / CONTROL n = 39), defined by `config.EXCLUDED_SUBJECTS` and `config.select_included_subjects()` (§1).
- **Attrition (gate logic a+b+c):** 202 NIfTI subjects → **−6 no valid group label (gate b:** CP0011/0015/0087/0106/0144/0193) → **−34 EXCLUDED_SUBJECTS (gate c:** 2 scan-duration QC [CP0004, CP0140] + 32 motion [28 COVID / 4 CONTROL]) = **162**.
- **Cohort single source of truth:** `config.EXCLUDED_SUBJECTS` (version-controlled) + `config.select_included_subjects()` implement gates a+b+c. `step0` is evidence-provider only, never edits config. Every downstream script binds to `config.select_included_subjects()` (no glob-based selection). Exclusion is applied **once** (step2, matrix construction), never re-applied downstream.
- **Motion exclusion:** subjects were excluded post-hoc on the basis of the motion output of the DPABI preprocessing, after conspicuous QC outputs were flagged during inspection. The criterion — **mean framewise displacement (FD) > 0.5 mm across the run** — was specified by the supervisor and coincides with an established literature threshold (Power et al., 2012, `10.1016/j.neuroimage.2011.10.018`), so the *threshold value* is a priori and literature-anchored rather than adapted to this sample's distribution. 32 subjects (28 COVID / 4 CONTROL) exceeded it.
- **Differential exclusion** (28 COVID / 4 CONTROL; COVID 18.4 % vs CONTROL 9.1 %) → stated as a **limitation**: possible selection bias toward more functional COVID patients. Because higher-motion COVID subjects are preferentially removed, any residual group motion difference in the retained sample is *reduced*, not inflated; the limitation concerns generalisability, not a motion-driven false positive.
- **Per-subject mean FD is available** (DPABI output) and underlies the exclusion. It is reported descriptively per group as a QC characterisation. It is *not* entered as a covariate — consistent with the a-priori no-covariate decision (§7), which is design-based and applies to all nuisance variables including motion.

> ⚠️ **All earlier N=194 results are void** (d=−0.32 within-Limbic, 23/28 direction counts, NBS 142 edges, k=2 clustering, etc.). Do **not** use old values as verification anchors. Interpret new numbers fresh.

---

## 2. Atlas & Parcellation

- **Primary (confirmatory):** Schaefer-400 (Schaefer et al., 2018), with Yeo-7 network mapping. Modern functional parcellation, not covered by the book → book mentioned only as historical context (recommends ≤ a few hundred ROIs, §8.4.2, superseded).
- **Robustness only:** Schaefer-100 and AAL (116 ROIs after explicit Background-label removal; nilearn 0.13.1 returns 117 incl. Background → off-by-one if uncorrected). **No correction across atlases; no independent confirmatory testing on 100/AAL.**
- **Network resolution (Family B):** Yeo-7. Yeo-17 only a conceptual mention — **no inference computed on Yeo-17.**
- **Exact Schaefer variant:** nilearn `fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=2)`, **7Networks order**. Label format `7Networks_<hemi>_<network>_<subregion>_<n>`; this variant carries **no Yeo subdivision suffix** in the network field (no DefaultA/B etc.), so the Yeo-7 main network is the second underscore token. Background label at index 0 removed via `atlas.indices=='0'` (with fallback), identically in step2 (FC construction) and step4a (Yeo labels) to guarantee positional ROI alignment. Yeo-7 distribution (LH/RH 200/200): Vis 61, SomMot 77, DorsAttn 46, SalVentAttn 47, Limbic 26, Cont 52, Default 91. Limbic is the smallest network (26 ROIs) with the lowest within-network FC baseline → noisiest within-Limbic estimate, interpreted with care.

---

## 3. Functional Connectivity Construction

- **Edge weights:** Pearson r for graph construction; **Fisher-z (arctanh, clip ±0.9999, `config.FISHER_CLIP`)** for group inference and all aggregated FC measures. Fisher-z is **not** applied to the raw matrix. (step3f and step5a ran with ±0.999999 before clip centralisation; result-neutral — both are clear nulls and the clip affects only edges with r extremely close to 1 — so they were not re-run.)
- **Aggregation order:** **z-then-mean** (Fisher-z per edge, then average). Never mean-then-z. Fixed centrally in §3.2.1.2. Verified at each downstream step (Family B within = upper-triangle edges, between = full rectangular A×B block over disjoint sets so each edge once; global mean-FC = all 79,800 upper-triangle edges). Retained unchanged in the partial arm: arctanh is applied to partial correlations identically, and at the near-zero partial magnitudes arctanh(r) ≈ r, so Fisher-z and raw aggregates coincide numerically (verified: Family B column means agree to ~4 decimals; global mean-FC d = −0.029 Fisher vs −0.034 raw).
- **Preprocessing:** external (DPABI), **including GSR**. GSR is declared and briefly defended (motion/physiology cleanup; especially defensible given the differential motion profile) with reference to the open debate (Power et al., 2014; Murphy & Fox, 2017; Ciric et al., 2017). Book treats GSR only briefly (§8.4.3, superseded on the debate).
- **GSR is one of two justifications for the positive-only graph strategy** (§5, Pearson arm): under GSR, negative edges are partly mathematically induced and interpretively ambiguous. The second, independent leg is the group-blind construction diagnostic (degenerate negative subgraph; §5). This GSR rationale does **not** transfer to partial correlations (§5, partial arm).
- **Caching:** method-namespaced (see Reproducibility notes, §8).
- **Parcel time-series standardisation:** `NiftiLabelsMasker(standardize="zscore_sample")`
  — each parcel time series is z-standardised before connectivity estimation, identically
  in both arms. Result-neutral for Pearson (scale-invariant); for the partial arm it means
  Ledoit-Wolf shrinkage is estimated on the correlation rather than the covariance scale.
---

## 4. Thresholding

- **Implementation:** own NumPy `proportional_threshold()`, used consistently across all steps.
  - **Correction (empirically checked against the COMET 1.2.4 source):** the earlier claim that `cg.threshold` contains a bug ("ignores its argument, always ~10 % density") is **not supported**. Called correctly (`density=` as a keyword), the function returns exactly the requested density; the "~10 %" narrative originated from a call error (the second positional argument is `type`, not `density`), and a naive call raises `NotImplementedError` rather than silently returning a wrong density. The bug claim must not appear in the thesis, this document, or the code.
  - **Actual reasons for the own implementation:** (i) sign handling is explicit — positive-only selection happens before thresholding, whereas `cg.threshold` ranks the signed values by absolute rank; (ii) it returns `n_keep`/`n_target`, which the target-reach diagnostic (§5) requires and `cg.threshold` does not provide; (iii) the thresholding definition is visible in the code: `density` is the fraction of **all possible** undirected edges (n·(n−1)/2; van Wijk et al., 2010), not of the currently non-zero edges.
  - **Thesis text:** the toolbox function is not mentioned at all; the own routine is simply stated.
- **Proportional, not absolute:** decouples density from topology under group FC differences (van Wijk et al., 2010; van den Heuvel et al., 2017). This is also why higher segregation alongside lower absolute FC is not contradictory.
- **Weighted graphs throughout** (Rubinov & Sporns, 2010); more robust to fragmentation than binary.
- **Book §8.4.2** (single threshold r>0.1, binary, ~34 ROIs) = historical, superseded by sweep + AUC (Achard & Bullmore, 2007; Garrison et al., 2015).
- **Sweep support points:** 1, 2, 5, 10, 15, 20, 25, 30, 50 %.
- **AUC integration:** `np.trapezoid` **with the `x` argument** (correct weighting of unequal spacing — documented explicitly). AUC additionally range-width-normalized so it reads as a mean metric value over the range and is comparable across ranges.
- **Primary (confirmatory) AUC range:** **10–25 %** (support points 10, 15, 20, 25). Formulation for the thesis text: *"The primary AUC range (10–25 %) was chosen as a conservative subset of the small-world regime (5–34 %) identified by Achard and Bullmore (2007) and falls within the range of commonly reported proportional thresholds (5–40 %) evaluated for stability by Garrison et al. (2015)."* AUC framed as a robustness measure against threshold dependence, not a high-resolution curve characterization. (Fornito et al. 2016 — the range citation in the old spec — is removed per the textbook-exclusion policy, §0.)
- **Sensitivity analysis:** broader range (5–50 %) over all support points — **declared, no independent confirmatory inference**; serves only to demonstrate stable effect direction.
- **Sweep-range validity (partial arm, both surviving strategies):** a saturation check on the step3c sweep CSVs confirms that neither AUC range sits in a degenerate zone. Edge count rises monotonically across the full sweep (no count saturation; `proportional_threshold` reaches every target density). Global Efficiency varies continuously to 80 % density and only freezes above it (|Δ| < 1e−4), i.e. **outside both the confirmatory 10–25 % and the sensitivity 5–50 % range**; Mean Clustering and Assortativity show no freeze anywhere. The freeze is topological saturation — added edges are too weak to shorten any shortest path, since `efficiency_wei` maps weight w to distance 1/w and near-zero partials act as effectively infinite distances — and does not constrain either range.
- **Connectedness (re-checked on N=162):** Schaefer-400 reaches ≥95 % connectedness only at ≥60 % density; within the confirmatory 10–25 % range most graphs are partially fragmented (≈4.9 % connected at 10 % → ≈56.8 % at 25 %), which motivates the fragmentation-tolerant weighted measures (§5). Disconnect pattern is diffuse (step3b, N=162: top isolated ROI ≈20 %, no chronic >50 % isolates), not a coverage artifact. Whether fragmentation is group-driven is checked by the Family A fragmentation confound step (step3d_a) on the confirmatory strategy of each arm (N=162):
  - *Pearson arm (positive):* disconnect-severity d=−0.203, naive-permutation p=.29 (Welch p=.29); island-size d=+0.15, p=.41. Fragmentation is not group-driven and, where weakly group-associated, runs opposite to a motion confound (higher in CONTROL).
  - *Partial arm (absolute):* disconnect-severity d=+0.036, p_perm=.810; island-size d=+0.262, p_perm=.0617 (Welch p=.0294). Severity is balanced; island size is larger in COVID at a sub-threshold level. Permutation is primary here by design (§6) and is the trustworthy figure: `max_islands` is a right-skewed count with many zeros and few extremes, so the parametric Welch p is anti-conservative. **The direction discharges the confound rather than raising it:** larger islands mechanically *lower* Global Efficiency, yet partial-arm GE is *higher* in COVID (d=+0.357) — the fragmentation difference works against the observed effect, not with it. The argument is strongest for Global Efficiency; for Mean Clustering the mechanical link is weaker and the discharge rests on the small effect (d=0.26) and p_perm=.062, and for Assortativity the effect is negligible in any case.

  Range chosen so the total-fragmentation region (≤5 %) does not enter primary inference. AAL strongly fragmented (≥95 % only at ≥90 %; 2 ROIs chronically >50 % isolated) → descriptive only.

---

## 5. Graph-Theoretic Measures

Four weighted, fragmentation-tolerant measures. BCT via `from comet.graph import bct` (Rubinov & Sporns, 2010; bundled in COMET 1.2.4 → version-locked).

| Measure | BCT call | Graph | Statistic |
|---|---|---|---|
| Global Efficiency | `efficiency_wei(W_pos)` | positive-only, per density | AUC over 10–25 % |
| Mean Clustering | `clustering_coef_wu(W_pos)` → mean | positive-only, per density | AUC over 10–25 % |
| Assortativity | `assortativity_wei(W_pos, 0)` | positive-only, per density | AUC over 10–25 % |
| Modularity Q* | `modularity_louvain_und_sign(W_full, qtype='sta', seed=s)` | **signed, full unthresholded matrix** | **single value per subject, NO AUC** |

The `W_pos` column reflects the **Pearson arm**. In the partial arm the same three metrics are computed on the `absolute` graph (`W = |partial r|`, per density); Modularity is unchanged. See the partial-arm section for the strategy decision.

- **Path Length and Small-Worldness are excluded:** both rely on finite shortest paths between all node pairs; under fragmentation these become infinite/undefined. The integrative dimension is already carried by Global Efficiency (sum over inverse shortest paths, finite under fragmentation; Rubinov & Sporns, 2010).
- **Per-measure negative-weight handling:** Efficiency (path lengths ill-defined for negative weights); Clustering/Assortativity (convention given few negative edges; Rubinov & Sporns, 2010); Modularity (positive/negative weights play intrinsically unequal roles → signed Q*; Rubinov & Sporns, 2011).
  - **Pearson arm:** the `W_pos` column above is positive-only. This is a two-part justification, not a bare assertion:
    1. *Construction (group-blind 3a/3b/3c diagnostic, Schaefer-400, N=162):* the negative subgraph is degenerate. In the confirmatory 10–25 % range the negative strategy cannot reach the target edge count for most subjects (target-reach 48 % → 31 % → 17 % → 11 % at 10/15/20/25 %), never reaches ≥95 % connectedness (plateaus at 34.6 %), and yields degenerate metrics (mean clustering ≈ 0.004 vs ≈ 0.34 positive; global efficiency frozen constant from ≈30 % density upward). Positive and absolute are near-identical in connectedness and target-reach (both 100 % in-range), because the few negative edges add almost nothing — so absolute buys no connectivity over positive.
    2. *Interpretability (GSR, §3):* under GSR the sparse negative edges are partly mathematically induced and ambiguous, so absolute would fold in ambiguous edges for no construction gain. Positive-only is therefore the cleaner choice.

    The diagnostic uses group-blind construction properties only (connectedness, target-reach, metric value ranges, fragmentation structure); the per-density Cohen's d printed in the 3c summaries is NOT used for this choice and is not shown in the Methods sign-strategy justification (it belongs to Results, Family A) — this separation must be explicit in the thesis text to avoid any appearance of circularity.
  - **Partial arm:** the graph-construction strategy is **`absolute`**, chosen by the group-blind 3a/3b/3c diagnostic (see partial-arm section). The Pearson reasoning does not carry over: ~47 % of partial edges are negative, so neither the degeneracy argument nor the GSR argument applies. Modularity stays signed in both arms.
- **Modularity asymmetry, stated explicitly:** computed on the full unthresholded signed matrix → no thresholding → no density-range sensitivity check by construction (not an omitted check). The three AUC measures carry the range robustness (§4).
- **Assortativity flag:** `assortativity_wei(W, 0)`, flag = 0 (undirected), passed explicitly.
- **Modularity Louvain runs:** Q* = **mean over n = 100 Louvain runs**, seeds from `SeedSequence(42).spawn(100)` (mean over the Q scalars, not over partitions). Per-subject SD and 95 % CI reported as a stability proof. Empirically the multi-run SD is ≤ 0.007 (mean ≈ 0.0012) → n=100 is stable and the specific seed is immaterial; this shows the Q estimate is reproducible regardless of seed.

---

## 6. Statistical Inference — Three Confirmatory Families (+ exploratory ROI level)

No correction across families (distinct biological questions, each with its own multiple-comparison control; Goeman & Solari, 2014). Book Ch. 7 = canonical framework.

**Unified inference logic (all group tests — Families A & B, the Family-B global mean-FC sanity check, and the exploratory ROI level):**

- **Naive permutation (10,000 perms) = primary.** Group labels are permuted; the test statistic is the t-value of the group difference (COVID − CONTROL). **No covariate adjustment** (§7).
- **Sensitivity:** Welch's t-test (unequal variances/sizes; groups 123 vs 39), reported in parallel. Permutation is the distribution-free reference (book §7.3.1.4).
- **Correction:** **FDR-BH** at q = 0.05 within the relevant family.
- **Effect size:** raw Cohen's d (COVID − CONTROL) with 95 % CI (Nakagawa & Cuthill, 2007), explicitly **descriptive**. Sign convention: d > 0 = COVID > CONTROL.
- **Seeding:** `SeedSequence(42).spawn(n_tests)` — one independent reproducible substream per test.

> **No Freedman–Lane, no GLM covariate model.** The residualized-permutation machinery is not used anywhere in the confirmatory pipeline (follows from the a-priori no-covariate decision, §7).

> **Family C (NBS) is deliberately *not* covered by the unified A/B logic above.** It uses its own permutation scheme (component-extent statistic, FWER not FDR), specified in the Family-C block. The shared elements are the naive label permutation, the seed convention, and the Fisher-z input; the correction and test statistic differ by design.

**Family A comprises exactly 4 tests in both arms** (3 AUC metrics over 10–25 % + Modularity Q*), FDR-BH over those 4. The partial-arm amplitude diagnostic (see partial-arm section) does **not** reduce the metric set: all four are retained and the amplitude finding is carried as an interpretive caveat in Results/Discussion, not as a selection rule. Rationale: dropping metrics post-diagnostic would make the family composition data-contingent and break parity with the Pearson arm.

### Results — verified N=162 (Pearson arm)

**Family A (Pearson, Schaefer-400 confirmatory, N=162) — COMPLETE, FDR-null.** All 4 tests non-significant after FDR-BH. Confirmatory (AUC 10–25%): Assortativity d=+0.37 (p_perm=.043, p_fdr=.114), Mean Clustering d=+0.18 (p_perm=.338, p_fdr=.451), Global Efficiency d=−0.06 (p_perm=.748, p_fdr=.748); Modularity Q* d=+0.29 (p_perm=.057, p_fdr=.114). Assortativity and Modularity show a consistent COVID>CONTROL segregation trend, uncorrected p≈.04–.06, but sub-significant after FDR. Broad-range (5–50%) sensitivity stable (Assortativity d=+0.33). Trend replicates across robustness atlases (S100: Assortativity d=+0.29, Modularity d=+0.29; AAL: Assortativity d=+0.37, Modularity d=+0.30), all FDR-null within atlas, no cross-atlas correction (§2). Fragmentation confound (step3d_a) discharged: disconnect-severity d=−0.20, p=.29 (higher in CONTROL, opposite a motion confound).

**step3f exploratory nodal strength (Pearson, Schaefer-400, positive, N=162) — COMPLETE, null.** Whole-brain ROI localization: 20/400 ROIs uncorrected p<.05 (= chance expectation of 20; no signal above chance), 0/400 FDR-BH (min q=.78). Sign-balance 209 pos / 191 neg (no directional node-strength shift). d range [−0.59, +0.48]; 48 ROIs |d|≥0.3 but none inference-supported (small-CONTROL-n variance). The global Family-A segregation trend is not attributable to individual nodes → consistent with a diffuse/architectural rather than focal effect. Exploratory, not a declared family; confirmatory localization = NBS (Family C).

**Family B (Pearson, Schaefer-400, N=162) — COMPLETE, null.** within-7: 0 FDR-sig (best within_Limbic d=−0.28, p=.16); between-21: 0 FDR-sig. Directional: 27/28 measures COVID<CONTROL. Global mean-FC (4e): d=−0.11, p=.56 (Fisher; raw d=−0.13) — the diffuse source; replaces void N=194 (d≈−0.03/p≈.91). ROI localization (4f, exploratory): diffuse not focal, Limbic 26/26 ROIs negative, 0 FDR. Dropout QC (4h): The tSNR dropout check (Family B, Limbic) is not re-run in the partial arm. The check operates on the preprocessed BOLD time series, not on the connectivity matrices or graphs, and both arms share the identical 4D input (only the estimator differs). tSNR is therefore estimator-invariant: the partial-arm result would be numerically identical to the Pearson-arm result. The Pearson-arm check (Limbic tSNR group-comparable, d≈0.02) already establishes that the signal quality is group-balanced, and this conclusion transfers unchanged. Re-running it would be redundant, not informative. This is distinct from the step4f skip above: step4f is skipped because the partial arm has no network-level pattern to localize, whereas the tSNR check is skipped because it does not depend on the estimator at all.

**Family C (NBS, Pearson, Schaefer-400, N=162) — COMPLETE, FWER-null.** All thresholds: t=3.1 (primary) min p_fwer=0.399 (10 components), t=2.5 min 0.414 (3 comp), t=3.5 min 0.333 (2 comp). No FWER-significant component at any threshold. Larger components consistently COVID<CONTROL (t=3.1 comp 9: 60 edges, 88% negative; the focal trace of the diffuse global reduction); COVID>CONTROL components are tiny (2–5 edges), non-coherent → the flagged sign-reversal phenomenon resolves as non-significant noise. t=2.5 forms a 656-edge component (permissive-threshold merging of the global shift), correctly non-significant (null max to 23310 edges). NBS runtime ~300 min/threshold (parallelized). Naive label permutation, no covariates, tail='both', seed=42.

### Results — verified N=162 (partial arm, Schaefer-400, `absolute`)

**Family A (partial, Schaefer-400 confirmatory, N=162) — COMPLETE, FDR-null.** All 4 tests non-significant after FDR-BH; min p_fdr = .253. Confirmatory (AUC 10–25 %): Global Efficiency d=+0.357 [−0.01,+0.72] (p_perm=.134, p_fdr=.253), Mean Clustering d=+0.300 [−0.06,+0.66] (p_perm=.190, p_fdr=.253), Assortativity d=−0.120 [−0.48,+0.24] (p_perm=.487, p_fdr=.487); Modularity Q* d=−0.321 [−0.68,+0.04] (p_perm=.136, p_fdr=.253). Broad-range (5–50 %) sensitivity consistent in direction and magnitude (GE d=+0.355, Clustering d=+0.340, Assortativity d=−0.121). **Effect directions differ from the Pearson arm:** the Pearson segregation trend (Assortativity +0.37, Modularity +0.29, both COVID>CONTROL) does not replicate — both metrics turn negative under conditioning (−0.120 / −0.321), while GE and Clustering, null under Pearson, turn positive. Both arms are FDR-null, so these are two null results with different point estimates, not conflicting findings. Modularity carries the cleanest form of this contrast: it is strategy-invariant (signed, full unthresholded matrix, no thresholding, no sign strategy), so the sign reversal is attributable to the FC estimator alone and cannot be an artifact of the `absolute` choice. Fragmentation confound (step3d_a) discharged directionally (§4).

**Family B (partial, Schaefer-400, N=162) — COMPLETE, FDR-null.** within-7: 0 FDR-sig (min p_fdr=.965; largest |d| within_Vis d=+0.167, p_perm=.399); between-21: 0 FDR-sig (min p_fdr=.792). The single sub-threshold uncorrected result, between_Vis_Default (d=−0.392, p_perm=.038, p_fdr=.792), is the expected chance yield at 21 tests (21 × 0.05 ≈ 1) and is not read as a trend. **Directional balance: within 4/7 negative, between 10/21 negative (~50/50)** — the Pearson 27/28 COVID<CONTROL shift does not replicate. Global mean-FC (4e): d=−0.029 [−0.389,+0.331], p_perm=.871 (Fisher; raw d=−0.034, p_perm=.847) — replaces the void N=194 value (d≈−0.03 / p≈.91), which it coincidentally resembles and must not be confused with. Fisher-raw agreement to ~4 decimals (arctanh(r) ≈ r at these magnitudes).

**Family C (NBS, partial, Schaefer-400, N=162) — COMPLETE, FWER-null.** No FWER-significant component at any threshold, by a wide margin: t=3.1 (primary) min p_fwer=.424 (65 components, largest 36 edges; null max [6, 32, 178]), t=2.5 min p_fwer=.968 (1 component, 965 edges; null max [875, 1069, 1292]), t=3.5 min p_fwer=.883 (37 components, largest 3 edges; null max [1, 4, 15]). The component progression across thresholds (one giant component → many small → tiny) is the expected behaviour of NBS on data without a coherent subnetwork effect. **Edge-sign balance confirms the absent directional shift:** the t=2.5 giant component is 50.16 % positive; t=3.1 components scatter between 36 % and 67 % without systematic direction. The Pearson sign-reversal phenomenon has no partial-arm counterpart. Naive label permutation, no covariates, tail='both', seed=42, k=10,000; runtime ~300 min/threshold (3 thresholds parallelized over 3 jobs).

**Positive control (step5 validation, prerequisite for reading the Family-C null).** `bct.nbs_bct` recovers a planted component (nodes 0–9, +0.5 offset, 50 ROIs, 20 vs 20 subjects, 500 perms) at 10/10 planted nodes, p=0.000. Without this, "NBS finds nothing" would be indistinguishable from silent pipeline failure. The validation run also fixed the API facts the confirmatory run depends on: return tuple `(pval, adj, null)` with `pval` a per-component vector (not a scalar); tail convention `'left'` = mean(X) < mean(Y), `'right'` = mean(Y) < mean(X); group order passed as x=CONTROL, y=COVID. The real-data 3v3 probe in the same script is an API shape check only, explicitly not valid statistics.

**step3f exploratory nodal strength (partial, Schaefer-400, `absolute`, N=162) — COMPLETE, null.** 31/400 ROIs uncorrected p<.05 (chance expectation ≈20), 0/400 FDR-BH (min q=.293). d range [−0.436, +0.608]; ROIs with |d|≥0.3 flagged descriptively via bootstrap CI excluding 0 (descriptive flag only, no multiplicity control). **Sign balance 265 pos / 135 neg** — unlike the Pearson arm's near-symmetric 209/191, the partial nodal strengths lean COVID>CONTROL, consistent with the positive GE/Clustering point estimates in Family A but likewise without inferential support. Nodal strength is computed on the `absolute` graph via `config.apply_sign_strategy`, consistent with the arm's confirmatory strategy. Exploratory, not a declared family; confirmatory localization = NBS (Family C).

**Cross-arm synthesis — the central finding.** The diffuse negative shift of the Pearson arm (27/28 Yeo cells COVID<CONTROL, global mean-FC d=−0.11) collapses completely under conditioning, on four independent lines of evidence: (i) FC-matrix QC, global mean-FC d=−0.029; (ii) Family B directional balance, within 4/7 and between 10/21 negative; (iii) step4e confound check, global mean-FC d=−0.029, p_perm=.871; (iv) NBS edge-sign balance, 50.16 % positive in the t=2.5 giant component. Since partial correlation removes shared and indirect variance by construction, the Pearson shift was carried by those components rather than by direct conditional associations. **The evidence does not distinguish whether that shared component reflects physiological noise or a genuinely global effect** — partial correlation is blind to globally shared signal by design. Both readings remain open; deciding between them would require non-GSR preprocessing or physiological regressors, neither available here (→ Future Work). Note also that the Pearson shift was itself FDR-null: what fails to replicate is a descriptive pattern, not an established finding.

### ROI level — exploratory localization (NOT a confirmatory family)

- Two ROI-level analyses, added at supervisor request: **whole-brain nodal strength** (400 ROIs, AUC of nodal strength over 10–25 % on the arm's confirmatory sign strategy; step3f) and **within-network ROI localization** for the networks with the largest descriptive Family B trends (step4f, Pearson arm only — the partial arm has no network-level pattern to localize, see below).
- Same naive-permutation machinery (no covariates), FDR-BH within each ROI set, bootstrap 95 % CI as descriptive uncertainty. Seeding differs by test situation: step4f draws one SeedSequence(42) substream per ROI within each network family (52 / 26 / 91 tests), whereas step3f uses a single vectorised permutation stream shared across all 400 ROIs. The shared stream applies the same subject-label permutation to every ROI in a given iteration and therefore preserves the spatial correlation structure between parcels in the null distribution — the appropriate choice for a whole-brain family of that size.
- **Explicitly exploratory / hypothesis-generating, not confirmatory.** Not one of the three pre-specified families; reported for transparency and spatial interpretation. The confirmatory localized test is NBS (Family C). Treated consistently across step3f and step4f (no asymmetry). No result is labeled "significant" in the thesis text.
- **"Exploratory" does not license omitting multiple-testing correction** — FDR-BH is still applied within each ROI set.
- **step4f is not run in the partial arm.** Two independent reasons: (i) scope — within-network ROI localization belongs to the nodal level; (ii) substantively — step4f localizes the networks carrying the largest descriptive Family-B trends, and the partial arm has none (cell differences −0.0002…+0.0001, ~50/50 direction). Localizing an absent network-level pattern would be a fishing expedition. The asymmetry between arms is therefore by data and design, not an omission. **Implementation:** the target networks are config-bound and arm-specific, `config.TARGET_NETWORKS_BY_ARM = {"pearson": ["Cont","Limbic","Default"], "partial": []}`; step4f begins with a `sys.exit` skip-guard that terminates cleanly when the arm's target list is empty, so the partial-arm skip is structurally enforced rather than left to convention. The config-bound target list (rather than a data-driven threshold) was chosen deliberately to avoid the circularity a data-driven selection would introduce.

---

## 7. Covariates (Decision B — final, a-priori)

- **Age and Sex are NOT entered as covariates** in any group test. No confound adjustment.
- **The justification is a-priori, not data-dependent:** group membership is not confounded by demographics *by design* — both groups originate from the same source population, so demographic adjustment is not substantively warranted. This is a design-based decision, **not** the result of a pre-test.
- **Demographic pre-test** (Age: Mann–Whitney U; Sex: χ²) runs before the main analyses, **descriptive only** (sample characterization), **not** a gatekeeper for covariate inclusion. Result on N=162: Age p ≈ .81, Sex p ≈ .33 → balanced. Methods wording preserves the distinction: *not* "test n.s. → no covariate", but "no design-based confounding; pre-test confirms balance descriptively."
- **Motion is not entered as a covariate.** Per-subject mean FD *is* available (§1) but, like Age and Sex, is not adjusted for — the no-covariate decision is design-based and blanket, not variable-specific. The differential motion *exclusion* (§1) remains a generalisability limitation; it is handled by exclusion, not adjustment.
- **Run length is not entered as a covariate** either (partial arm; see the TP-confound block), consistent with the same blanket decision. It is checked and discharged, not adjusted.
- **Signal-quality QC (step4h):** mean intensity + tSNR per network, group-compared descriptively, confirms the within-network FC trends are not attributable to differential signal dropout (tSNR group |d| ≤ 0.11). tSNR on preprocessed data is descriptive only, no exclusion criterion (DVARS inappropriate post-preprocessing).

> The covariate topic is documented here for the record but is **not** introduced into the thesis Background or Methods as a covariate-adjustment subsection (that subsection was rejected). Addressed verbally at defense only if raised.

---

## 8. Reproducibility & Setup

**In the thesis, reproducibility is documented distributed** — no standalone chapter section: the seed in the inference section (§3.2.4.2), software/versions in setup (§3.1.3), cohort definition in the sample section (§3.1.2). **In this decisions document it is bundled below as a working reference**; the bundling is an artefact of this file, not a thesis-structure decision.

- **Seeds:** base seed = 42 throughout. Permutations: `SeedSequence(42).spawn(n_tests)` → one reproducible, independent substream per test (order-independent; avoids sequential RNG-state inheritance). Louvain: `SeedSequence(42).spawn(100)`. NBS: internal seed = 42, passed explicitly (`bct.nbs_bct` defaults to `seed=None` → global RNG state → non-reproducible if omitted).
- **Cohort single source of truth:** see §1 (`config.EXCLUDED_SUBJECTS` + `config.select_included_subjects()`).
- **Central constants:** `FISHER_CLIP=0.9999`, `N_PERMUTATIONS=10000`, `FDR_ALPHA=0.05`, `DENSITY_SUPPORT_POINTS`, `AUC_RANGE_CONFIRMATORY=(0.10,0.25)`, `AUC_RANGE_SENSITIVITY=(0.05,0.50)`, `MODULARITY_N_RUNS=100`, `NBS_THRESHOLDS=[2.5,3.1,3.5]`, `NBS_PRIMARY_THRESHOLD=3.1`, `NBS_TAIL='both'`, `TARGET_NETWORKS_BY_ARM` are centralised in config as single source of truth. Already-run Pearson scripts carry matching hardcoded values and were not retrofitted (result-neutral); step3f/step5a used FISHER_CLIP=0.999999 (result-neutral, see §3).
- **Caching — re-run protocol:**
  1. Archive old N=194 cache to `_cache_N194_archive/`.
  2. Cold re-run with `USE_CACHE=True` (fresh build; do not accidentally overwrite `USE_CACHE`).
  3. **Phase-6 cold verification run** (clear cache, second from-scratch run) → identical results as the canonical determinism / stale-freeness proof, **before deleting legacy output trees**.
  - Cache keys are method-namespaced (protects against cross-*method* staleness, not cross-*cohort* staleness → full clear required on cohort change).
- **Pipeline switch:** `config.FC_METHOD` ("pearson" | "partial"); outputs namespaced under `analysis_outputs/<FC_METHOD>/`. Wrapper + pipeline-module pattern retained. `n_jobs = 6` default (`config.N_JOBS_DEFAULT`).
- **Git:** one logical chunk per commit; documentation commits separate from code commits; `_retired_*/` archive pattern, no hard deletes.
- **Software:** Python 3.11 (`comet` venv), COMET 1.2.4 (version-locked), Nilearn 0.13.1, numpy / pandas / scipy / matplotlib / networkx / statsmodels / joblib; BCT via COMET. Partial-arm QC additionally uses scikit-learn (`sklearn.covariance.LedoitWolf`) to recompute λ independently of COMET's internals.
- **Data:** `/mnt/d87cc26d-5470-443c-81c1-e09b68ee4730/Cedric/`; group CSV `ResumenRespuestasBasico.csv`.

---

## 8a. AI-Tool Declaration & Code-Step Cross-References (thesis text)

Per supervisor feedback (recorded here for the record; implemented in the Methods text, not a computational decision):

- **General AI declaration:** a single general remark in the Methods states that AI-based assistants (Claude, Anthropic) were used. For language editing only — phrasing, grammar, translation of German drafts into English — the general remark is **sufficient**; no inline marking is required at each edited passage.
- **Code marking:** where AI **substantively generated content** — i.e. the analysis code — this is marked **explicitly**. The author defined the analytical design, the pipeline and its output structure and all methodological decisions; the Python implementation was generated largely under these specifications and executed and verified locally by the author. All methodological choices, their justification and the interpretation of the results are the author's own.
- **Code-step cross-references:** each Methods subsection carries a reference binding it to the corresponding pipeline step(s), so that the text and the repository can be followed together (e.g. "3.2.1 Functional Connectivity Estimation" → the step1/step2 scripts). Exact wording of the reference marker is a thesis-text choice; the binding itself is fixed. The complete script-per-step listing is in Appendix 7.2.

---

## 9. (Reserved) Clustering / Subtype Analysis

Clustering / subtype analysis is **not** part of the final thesis at this time (pending supervisor discussion). Not specified here intentionally.

---

## Open Items / TODOs

1. **[Open — thesis text]** The step3f partial run computes nodal strength on the `absolute` graph, not the signed S⁺/S⁻ decomposition (Rubinov & Sporns, 2011). The S⁺/S⁻ decomposition is **not performed** in either arm. Any thesis sentence claiming signed strength decomposition as the handling of negative partials must be corrected or removed.
2. **[Open — Discussion `[CITE?]`]** Two citation markers to resolve in the Discussion: proportional-thresholding reference; negative-edge interpretation reference.

---

## Re-run Order (binding sequence)

**Pearson arm (primary, all three atlases):** one per-atlas wrapper each; `CONFIRMATORY_SIGN_STRATEGY = "positive"` set automatically. The choice rests on the group-blind 3a/3b/3c diagnostic on Schaefer-400 (negative subgraph degenerate: target-reach collapses in-range, metrics degenerate; §5) together with the GSR interpretability argument (§3) — it is group-blind and data-driven, **not** a-priori (unlike the covariate decision, §7). **STATUS: COMPLETE (N=162).** Families A/B/C all null; verified results recorded in §6.

**Partial arm (sensitivity, Schaefer-400 only): STATUS: COMPLETE (N=162).** Families A/B/C all null; verified results recorded in §6.

1. Clear / archive old partial Family-A cache (`_retired_pre_signstrategy/`); Family B/C partial results stay. **[COMPLETE]**
2. step2 — ROI time series + partial (Ledoit-Wolf) FC matrices (already built; sign-neutral) + FC diagnostics. **[COMPLETE]** — all 162 matrices pass shape/symmetry/zero-diagonal/range/NaN/Inf checks; fc_mean 0.002 ± 0.000 (near-zero as expected under conditioning); global mean-FC d = −0.029.
3. **step3a / 3b / 3c — group-blind DIAGNOSTIC over all three sign strategies** (positive / negative / absolute), written to `family_A/_cross_strategy/`. 3a connectedness, 3b fragmentation, 3c descriptive metric behaviour. No group inference. **[COMPLETE]** — `negative` skipped at 3c by the degeneracy guard (see partial-arm section).
4. **Strategy + metric choice** (group-blind; recorded in `config.CONFIRMATORY_SIGN_STRATEGY`). ← decision gate. **[COMPLETE]** — `"absolute"`; all four metrics retained (§6).
5. step3d_a — Family A fragmentation confound check on the confirmatory strategy, group-aware, descriptive. **[COMPLETE]** — §4.
6. step3d — confirmatory naive-permutation AUC inference (Family A) on the chosen strategy, `family_A/{strategy}/`. Guarded by `assert config.CONFIRMATORY_SIGN_STRATEGY is not None`. **[COMPLETE]** — §6.
   - step3e is **not applicable** to the partial arm: it is the cross-atlas comparison plot (Schaefer-400 / Schaefer-100 / AAL), and the partial arm is Schaefer-400 only.
7. step3f — exploratory nodal strength on the confirmatory strategy, `step3f_nodal_strength/{strategy}/`. **[COMPLETE]** — §6.
8. step4 — Yeo-7 (Family B, sign-neutral) **[COMPLETE]** — §6. step4f not applicable (see ROI-level block).
9. step5 — NBS (Family C, sign-neutral), preceded by the positive-control validation run. **[COMPLETE]** — §6.
10. Phase-6 cold verification run. **[Open]**

---

## Partial-correlation arm — sign-strategy diagnostic & confirmatory choice (binding)

**Scope: Schaefer-400 only** (the partial arm tests estimator robustness, not atlas robustness — atlas robustness is established in the Pearson primary arm). Applies to the path-/clustering-based Family-A metrics (Global Efficiency, Mean Clustering, Assortativity) and the exploratory nodal strength (step3f). Does **not** apply to: Modularity (always signed, `modularity_louvain_und_sign`, strategy-invariant), Family B (always signed — mean conditional connectivity needs signs), or Family C (always signed-all-edges — NBS tests per-edge group differences).

**Motivation.** The supervisor requested that positive/negative/absolute be examined for the partial arm. Key asymmetry vs. Pearson: under Pearson the negative subgraph is **degenerate** (empirically: cannot reach target densities in-range, see §5), so positive-only was forced and the GSR rationale applies. Under **partial correlation ~47 % of edges are negative**, so the negative subgraph is substantive, the GSR rationale does **not** transfer, and the diagnostic is genuinely informative. The balanced sign distribution is the expected behaviour of partial correlation, not a peculiarity of this dataset: full correlations typically yield predominantly positive FC with a minority of negative edges, whereas partial correlations often produce a relative balance between positive and negative estimates (Hallquist & Hillary, 2018, `10.1162/netn_a_00054`). The pooled FC value distribution confirms it here: a narrow, symmetric distribution centred on zero, essentially within ±0.1.

**Design (diagnostic → single confirmatory choice; NOT three parallel inference arms):**

1. **steps 3a / 3b / 3c** run over all three strategies (positive / negative / absolute) as a **group-blind DIAGNOSTIC** — connectedness (3a), fragmentation (3b), descriptive metric behaviour over the sweep (3c). **No group inference** at this stage → the comparison cannot be circular. Written to the strategy-invariant `family_A/_cross_strategy/` tree.
2. **One** strategy is chosen for the confirmatory AUC analysis, based on **group-blind construction properties only**. Recorded in `config.CONFIRMATORY_SIGN_STRATEGY`; **must not** depend on group differences.
3. **step 3d** runs the confirmatory naive-permutation AUC inference on that single strategy, `family_A/{strategy}/`. Guard `assert config.CONFIRMATORY_SIGN_STRATEGY is not None` prevents premature runs.

**Diagnostic outcome (group-blind, N=162, Schaefer-400).**

*`negative` — excluded.* Global Efficiency degenerates to a density-invariant constant. Detected by the 3c sanity-check guard on the first subject (CP0001: GE range 0.000941 < 0.001 across densities 5–100 %; values plateau at 0.023874 from 20 % upward), i.e. **before any group contrast is computed** — the exclusion is group-blind by construction. Documented as an output artifact, `family_A/_cross_strategy/step3c_metrics/negative/step3c_SKIPPED_negative.txt`; the sweep records the skip and continues to the next strategy rather than aborting, so the exclusion is a reproducible diagnostic result, not a crash. The finding is consistent across the diagnostic: 3a — at 5 % density only 38.9 % of subjects connected (vs 86.4 % positive, 61.7 % absolute), target-reach lost from 50 % density upward; 3b — only 63/162 subjects never disconnected (vs 140/162 positive, 100/162 absolute). Mechanistically the degeneracy is **topological, not amplitude-related**: the pooled FC distribution is symmetric, so negative edges are not systematically weaker than positive ones — the isolated negative subgraph simply exhausts its useful path structure early, after which added edges open no new shortest paths. This mechanism is a reasoned account, not a verified one; the exclusion rests on the observed density-invariant behaviour, which is sufficient.

*`positive` vs. `absolute` — both viable, no topological disqualification.* Both reach ≥95 % connectedness from 10 % density and hold target-reach across the whole confirmatory range; in 10–25 % they retain identical edge counts (3,990 / 7,980 / 11,970 / 15,960 / 19,950 at 5/10/15/20/25 %). Their diagnostics diverge only outside that range and in opposite directions: `absolute` holds target-reach to 100 % density (`positive` to 50 %) and reaches the complete upper triangle (79,800 edges at 100 %, vs 70,767 for `positive`), while `positive` is more robust to subject-level fragmentation (140/162 never disconnected vs 100/162). Neither is decisive.

**Decision: `CONFIRMATORY_SIGN_STRATEGY = "absolute"`.** The topological diagnostics do not separate the two candidates within the confirmatory range, so the choice rests on a **conceptual, a-priori argument**: under partial correlation, negative edges are valid conditional associations rather than the interpretively ambiguous artefacts they are under GSR-processed Pearson FC. The GSR rationale that justifies positive-only in the Pearson arm explicitly does not transfer (§3), and with ~47 % of edges negative, positive-only would discard roughly half the conditional structure with no substantive justification for doing so. `absolute` retains the full conditional structure. Both strategies are established in the literature — of the studies that report their handling of negative edges at all, ~21 % delete them and ~9 % include them as positive weights via the absolute value; 57 % report insufficient or no information (Hallquist & Hillary, 2018 — so this is a defensible choice among accepted options, not a uniquely correct one. Its transparency and group-blindness exceed the reporting standard of the majority of the published literature.

**Caveats carried into Limitations / Discussion (declared, not resolved):**
- `absolute` equates strong negative with strong positive conditional associations. Whether that equation is biologically warranted is unresolved; negative edges in brain dynamics are not straightforwardly interpretable.
- The arms differ in both estimator and sign strategy, so the comparison is not a clean estimator-only contrast for the three AUC metrics. This is unavoidable — the sign strategies rest on different, non-transferable justifications — and is mitigated by Modularity Q*, which is strategy-invariant and therefore *is* a clean estimator-only contrast. The concern is further weakened by the fact that partial and full correlations yield fundamentally different topologies regardless of sign handling (Cassidy et al., 2018, via Hallquist & Hillary, 2018), so matching the sign strategy would not have bought genuine comparability.
- Positive-only and negative-only subgraphs may carry distinct topological information; a systematic comparison was out of scope (→ Future Work). In these data the negative-only arm is not viable in any case (see above).

**Two-arm code parity.** `config.FC_METHOD` switches arms with a single change: Pearson sets `CONFIRMATORY_SIGN_STRATEGY = "positive"` automatically and runs all three atlases; partial sets it to `"absolute"` (decided post-diagnostic, now fixed) and runs Schaefer-400 only. Sign-neutral steps (step2, Family B, Family C, QC) are unchanged across arms.

---

## Partial-correlation arm — estimator, symmetry, and TP confound (step2 QC)

**Estimator:** `Static_Partial(cov_estimator="LedoitWolf")` (COMET 1.2.4), identical call signature to `Static_Pearson`. Ledoit-Wolf shrinkage is required because for Schaefer-400 p = 400 > T ≈ 150–200 → the unregularised precision matrix is singular/unstable. Shrinkage is analytic, parameter-free (no lambda grid, no CV) and estimated per subject (Ledoit & Wolf, 2004; Varoquaux & Craddock, 2013). Chosen over GraphicalLasso to avoid a sparse output and thus double sparsification with the proportional-threshold sweep; step3–5 logic unchanged. Only code change: a `"partial"` branch in `config.make_connectivity()`; output namespacing produces a parallel `analysis_outputs/partial/` tree.

**Atlas restriction, restated:** the partial arm is Schaefer-400 only *by design* (estimator robustness). Schaefer-100 (p=100 < T) would give a better-conditioned covariance and less shrinkage, but switching atlas for the partial arm would confound estimator and parcellation and destroy the contrast the arm exists to provide. The p > T regime is precisely what Ledoit-Wolf is built for, and the realised shrinkage is mild (λ ≈ 0.05, i.e. ~95 % empirical information retained), so there is no estimation problem for an atlas change to solve. The T > P question is noted in Limitations only.

**Symmetry:** partial matrices are mathematically symmetric (precision-matrix derived); residual |M − Mᵀ| ≤ 6 × 10⁻⁸ is float32 rounding noise. Recorded descriptively in QC, then exact symmetry is enforced (M = ½(M + Mᵀ)) before any statistic that reads the full matrix (e.g. signed Modularity). Threshold-based metrics read the upper triangle only and are unaffected. SYM_TOL = 1e-6 (float32-realistic), not 1e-10. Confirmed in step5: max raw asymmetry 2.98e−08, max|M − Mᵀ| = 0 after symmetrization.

**Edge-weight regime (per atlas, off-diagonal):** strongly shrinkage-dependent. Schaefer-400 (p ≫ T): narrowest range [−0.22, +0.39], fc_std ≈ 0.026. Schaefer-100 / AAL (p < T): broader [−0.36, +0.74] / [−0.43, +0.66], fc_std ≈ 0.076 / 0.073. The confirmatory atlas thus carries the most conservative (most shrunk) estimate. ~47 % of off-diagonal partials are negative across atlases (vs. predominantly positive Pearson FC).

**Run-length × group confound (examined, discharged):** two acquisition lengths (140 vs 200 volumes). Run length is associated with group (Fisher exact, p = .020; controls over-represented among shorter runs, 23.1 % vs 8.1 %). Because Ledoit-Wolf shrinkage depends on p/T, this is examined as a potential confound: λ is the quantity through which run length would bias partial-correlation estimates, since fewer timepoints → worse-conditioned covariance → stronger shrinkage → edges pulled harder toward zero. λ is not stored in the cached matrices (COMET saves only the final partial matrix) and is therefore recomputed from the timeseries with sklearn's analytic Ledoit-Wolf estimator (deterministic, no seed/CV), which characterises each subject's shrinkage regime independently of COMET's internals. Result (Schaefer-400, N=162): λ range [0.018, 0.126]; COVID mean 0.0511 (n=123; 10 × T=140, 113 × T=200), CONTROL mean 0.0545 (n=39; 9 × T=140, 30 × T=200); Welch t=−0.89, p=.377, d=−0.184. Neither per-subject edge-weight dispersion (fc_std × group: all p ≥ .12) nor λ differs systematically between groups. The direction is additionally reassuring: CONTROL holds proportionally more short runs and shows the marginally *higher* λ — mechanistically consistent, and if anything it would damp a COVID<CONTROL difference rather than manufacture one. The imbalance does not translate into a group-systematic estimation bias; run length is NOT entered as a covariate (§7) and is stated as a checked-and-discharged limitation. Both run lengths are pooled without adjustment (supervisor decision); this check is the descriptive evidence that the pooling is unbiased, standing to it exactly as the demographic pre-test stands to the no-covariate decision (§7). (Consistent-outlier CP0192 across atlases is a full-length subject [200 TP] → not a shrinkage artifact; frozen cohort, no exclusion.)

---

## Partial-correlation arm — Family A amplitude diagnostic (Schaefer-400)

Scope: partial arm is Schaefer-400 only. Before inference (step3d), the three weighted AUC metrics are tested for whether their group effects reflect topology or edge-weight amplitude, by residualising the per-subject AUC value (10–25 %) on the mean weight of retained edges (amplitude proxy). Verdicts are collinearity-aware: at |r| > .95 the metric and the proxy are quasi the same variable, so the residual d is noise and the metric is amplitude-driven regardless. Descriptive sensitivity check, NOT a pre-specified inference family.

> **Numbers below predate the `absolute` decision** and are not directly comparable to the step3d results in §6 (which are computed on the `absolute` graph). They are retained as the amplitude verdict; the raw d values quoted here should not be read as the Family-A effect sizes.

- **Global Efficiency:** quasi-collinear with retained-edge weight (r = .99); reflects amplitude, not topology. Raw d = +0.36 arises because the amplitude signal differs, not from a topological effect. Not interpreted as a topological marker.
- **Mean Clustering:** amplitude-dominated (r = .89; residual sign flips +0.27 → −0.12). Not a robust topology marker.
- **Assortativity:** null (|d| ≤ 0.09), amplitude-independent. The Pearson-arm assortativity trend (d ≈ +0.29–0.37, COVID > CONTROL) does NOT replicate under partial correlation.

**Conclusion:** no robust topological group difference in Family A under partial correlation; replicates the Pearson Family-A null and adds the mechanistic finding that weighted global metrics on strongly regularised (Ledoit-Wolf) partials are largely amplitude-driven at p ≫ T.

**Implication — superseded on metric selection (re-decision, recorded).** The earlier version of this block concluded that only Assortativity is a clean topology test and implied a reduced metric set for step3d. That implication is **withdrawn**: Family A retains all four metrics in both arms (§6). Reducing the family post-diagnostic would make its composition data-contingent and break parity with the Pearson arm, at the cost of the fixed 4-test FDR structure. The amplitude finding is instead carried as an **interpretive caveat**: in the partial arm, Efficiency and Clustering are read as amplitude-sensitive rather than as pure topology markers, and this qualification belongs in Results/Discussion. Modularity Q* (signed, full matrix) is unaffected and evaluated separately.
