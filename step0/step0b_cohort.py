"""
Cohort attrition & config.EXCLUDED_SUBJECTS proposal from step0a_qc_summary.csv + GROUP_CSV.
Evidence only — never edits config; re-run after every config.EXCLUDED_SUBJECTS change.

In: step0a_qc_summary.csv (step0a_qc output), config.GROUP_CSV.
Out: step0b_csv_data_match.csv, step0b_cohort_flow.csv, step0b_excluded_subjects_proposal.txt,
     step0b_cohort_attrition_table.txt.

CASE A (ID-format mismatch between NIfTI and CSV) must be repaired, not excluded —
flagged separately from genuine QC/motion/no-metadata exclusions.
"""
from pathlib import Path
import re
import sys
import subprocess
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

OUT_DIR = config.ensure(config.PRE_ANALYSIS_DIR)
QC_CSV     = OUT_DIR / "step0a_qc_summary.csv"
MATCH_CSV  = OUT_DIR / "step0b_csv_data_match.csv"
FLOW_CSV   = OUT_DIR / "step0b_cohort_flow.csv"

ID_COL, GROUP_COL = "ID", "Grupo"
VALID = config.VALID_GROUPS


def norm_id(s):
    return (re.sub(r"\D", "", str(s)).lstrip("0") or "0")


def verdict(r):
    if r.get("status") != "complete":            return "EXCLUDE_TECHNICAL"
    if r.get("duration_check") == "non_protocol": return "EXCLUDE_DURATION"
    if int(r.get("n_invalid_voxels") or 0) > 0:   return "EXCLUDE_QC"
    return "PASS"


def reason(r):
    if r.get("status") != "complete":
        return f"technical: {r.get('status')}"
    if r.get("duration_check") == "non_protocol":
        return f"scan duration {float(r['duration_sec']):.0f} s — outside acquisition protocol"
    if int(r.get("n_invalid_voxels") or 0) > 0:
        return f"NaN/Inf voxels ({int(r['n_invalid_voxels'])})"
    return "—"


def main():
    df = pd.read_csv(QC_CSV)
    df["verdict"] = df.apply(verdict, axis=1)
    df["reason"]  = df.apply(reason, axis=1)

    g = pd.read_csv(config.GROUP_CSV)
    g[ID_COL] = g[ID_COL].astype(str).str.strip()
    group_of = {i: str(gr) for i, gr in zip(g[ID_COL], g[GROUP_COL])}

    data_ids = set(df.loc[df["status"] != "no_nii_found", "subject_id"].astype(str))
    csv_ids  = set(group_of)
    csv_norm = {}
    for c in csv_ids:
        csv_norm.setdefault(norm_id(c), []).append(c)

    rows, case_a, case_b, invalid = [], [], [], []
    for sid in sorted(data_ids):
        label = group_of.get(sid, ""); valid = label in VALID
        if sid in csv_ids and valid:
            cat = "matched_valid"
        elif sid in csv_ids and not valid:
            cat = f"in_csv_invalid_label({label or 'NaN'})"; invalid.append(sid)
        else:
            cand = [c for c in csv_norm.get(norm_id(sid), []) if c != sid]
            if cand:
                cat = f"CASE_A_format_mismatch->{','.join(cand)}"; case_a.append((sid, cand))
            else:
                cat = "CASE_B_no_csv_row"; case_b.append(sid)
        rows.append({"subject_id": sid, "group_label": label,
                     "valid_group": valid, "category": cat})
    for cid in sorted(csv_ids - data_ids):
        if not any(cid in cand for _, cand in case_a):
            rows.append({"subject_id": cid, "group_label": group_of[cid],
                         "valid_group": group_of[cid] in VALID,
                         "category": "csv_row_without_data"})
    df_match = pd.DataFrame(rows).sort_values(["category", "subject_id"])
    df_match.to_csv(MATCH_CSV, index=False)
    print("--- CSV/data match categories ---")
    print(df_match["category"].value_counts().to_string())

    # ---- copy-paste proposal block ----
    qc_excl = df[df["verdict"].str.startswith("EXCLUDE")][["subject_id", "reason"]]
    print("\n" + "=" * 66)
    print("PROPOSAL for config.EXCLUDED_SUBJECTS  (CHECK MANUALLY — CASE A NOT)")
    print("=" * 66)
    print("    # --- QC (technical / duration / NaN-Inf) ---")
    for _, r in qc_excl.sort_values("subject_id").iterrows():
        print(f'    "{r["subject_id"]}": "{r["reason"]}",')
    print("    # --- case B: no CSV row ---")
    for sid in sorted(case_b):
        print(f'    "{sid}": "no metadata row in ResumenRespuestasBasico.csv",')
    if invalid:
        print("    # --- in CSV, invalid label ---")
        for sid in sorted(invalid):
            print(f'    "{sid}": "group label {group_of[sid]!r} not in VALID_GROUPS",')
    if case_a:
        print("\n!! CASE A (ID-format mismatch) — REPAIR, do NOT exclude:")
        for sid, cand in case_a:
            print(f"    NIfTI '{sid}' ~ CSV {cand}")

    # ---- attrition table (group-split) + FINAL N via config ----
    final = set(config.select_included_subjects(
        sorted(data_ids), g, id_col=ID_COL, group_col=GROUP_COL, verbose=False))

    def split(ids):
        ids = list(ids)
        c = sum(group_of.get(s) == "COVID" for s in ids)
        k = sum(group_of.get(s) == "CONTROL" for s in ids)
        return len(ids), c, k, len(ids) - c - k

    def stage_of(rsn):
        r = rsn.lower()
        if any(k in r for k in ["duration", "technical", "nan", "inf", "header", "load"]):
            return "1_excl_qc"
        if "no metadata row" in r:               return "2_excl_no_metadata"
        if "motion" in r or "fd" in r:           return "3_excl_motion"
        return "X_UNCLASSIFIED"

    # Classify the excluded IDs into attrition stages WITHOUT relying on
    # per-subject reasons in config (EXCLUDED_SUBJECTS is now a reason-free set).
    # step0b reconstructs the stage from its own evidence: QC/duration verdicts
    # come from step0a_qc_summary (df['verdict']/df['reason']); no-metadata cases
    # are the CASE_B set; everything else excluded is motion (the supervisor-
    # curated list, not otherwise encoded here).
    qc_stage_of = {}
    for _, r in df[df["verdict"].str.startswith("EXCLUDE")].iterrows():
        qc_stage_of[str(r["subject_id"])] = stage_of(r["reason"])
    case_b_set = set(case_b)

    by_stage, unclass = {}, []
    for sid in config.EXCLUDED_SUBJECTS:
        if sid not in data_ids:
            continue
        if sid in qc_stage_of:
            st = qc_stage_of[sid]
        elif sid in case_b_set:
            st = "2_excl_no_metadata"
        else:
            st = "3_excl_motion"   # by elimination: supervisor-curated motion list
        by_stage.setdefault(st, []).append(sid)

    flow = [["0_screened_nifti", *split(data_ids), "all NIfTI with data"]]
    for st in ["1_excl_qc", "2_excl_no_metadata", "3_excl_motion", "X_UNCLASSIFIED"]:
        ids = by_stage.get(st, [])
        if ids or st != "X_UNCLASSIFIED":
            flow.append([st, *split(ids), "; ".join(sorted(ids)) or "—"])
    flow.append(["9_final_analytical", *split(final), "included"])
    df_flow = pd.DataFrame(flow, columns=["stage", "n", "n_covid",
                                          "n_control", "n_unlabeled", "detail"])
    df_flow.to_csv(FLOW_CSV, index=False)

    try:
        gh = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent),
                             "rev-parse", "HEAD"], capture_output=True,
                            text=True, timeout=5).stdout.strip()[:10] or "UNKNOWN"
    except Exception:
        gh = "UNKNOWN"

    proposal_file = OUT_DIR / "step0b_excluded_subjects_proposal.txt"

    with open(proposal_file, "w", encoding="utf-8") as f:
        f.write("PROPOSAL for config.EXCLUDED_SUBJECTS (CHECK MANUALLY — CASE A NOT)\n")
        f.write("=" * 66 + "\n")
        f.write("# --- QC (technical / duration / NaN-Inf) ---\n")
        for _, r in qc_excl.sort_values("subject_id").iterrows():
            f.write(f'"{r["subject_id"]}": "{r["reason"]}",\n')

        f.write("\n# --- case B: no CSV row ---\n")
        for sid in sorted(case_b):
            f.write(f'"{sid}": "no metadata row in ResumenRespuestasBasico.csv",\n')

        if invalid:
            f.write("\n# --- in CSV, invalid label ---\n")
            for sid in sorted(invalid):
                f.write(f'"{sid}": "group label {group_of[sid]!r} not in VALID_GROUPS",\n')

        if case_a:
            f.write("\n!! CASE A (ID-format mismatch) — REPAIR, do NOT exclude:\n")
            for sid, cand in case_a:
                f.write(f"NIfTI '{sid}' ~ CSV {cand}\n")

    print("\n" + "=" * 66)
    print("COHORT ATTRITION (group-split via GROUP_CSV)")
    print("=" * 66)
    print(df_flow.drop(columns="detail").to_string(index=False))
    if unclass:
        print("\n  !! UNCLASSIFIED reasons (fix config wording):")
        for sid, rsn in unclass:
            print(f"     {sid}: {rsn!r}")
    nf, cf, kf, _ = split(final)
    txt_file = OUT_DIR / "step0b_cohort_attrition_table.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("COHORT ATTRITION (group-split via GROUP_CSV)\n")
        f.write("=" * 66 + "\n")
        f.write(df_flow.drop(columns="detail").to_string(index=False))
        f.write(f"\n\nFINAL N = {nf} (COVID={cf}, CONTROL={kf})\n")
        f.write(f"\n\nFINAL N = {nf} (COVID={cf}, CONTROL={kf})\n")
        f.write(f"EXCLUDED_SUBJECTS={len(config.EXCLUDED_SUBJECTS)} | "
                f"FC_METHOD={config.FC_METHOD} | SEED={config.SEED} | git={gh}\n")
    print(f"\n  step0b_cohort_flow.csv -> {FLOW_CSV}")
    print(f"  step0b_excluded_subjects_proposal.txt -> {proposal_file}")
    print(f"  EXCLUDED_SUBJECTS={len(config.EXCLUDED_SUBJECTS)} | "
          f"FC_METHOD={config.FC_METHOD} | SEED={config.SEED} | git={gh}")
    print(f"\n  >>> FINAL N = {nf}  (COVID={cf}, CONTROL={kf})")


if __name__ == "__main__":
    main()