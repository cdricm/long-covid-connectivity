"""
Step 4a: Extract Schaefer-400 Yeo-7 network labels for Family B (network level).

In: none (atlas fetched from nilearn).
Out: config.atlas_dir("schaefer400", "step4a_labels")/schaefer400_yeo7_roi_info.csv
     (roi_idx 0-399, full_label, hemisphere, yeo_network, yeo_subdivision,
     sub_region), schaefer400_raw_labels.txt.

Same background label removed as step2 (nii_to_fc_pipeline), verified identical
ROI order: this consistency is essential, since the Family B network assignment
is positionally aligned with the FC matrices built in step2 — any divergence
would shift every ROI index by one and corrupt the entire network-level
analysis.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import os
import pandas as pd
from nilearn import datasets

# ============================================================
# SETTINGS
# ============================================================
OUT_DIR = config.ensure(config.atlas_dir("schaefer400", "step4a_labels"))
N_ROIS  = 400
YEO_NETWORKS_OFFICIAL = ["Vis", "SomMot", "DorsAttn", "SalVentAttn",
                         "Limbic", "Cont", "Default"]

# ============================================================
# FETCH ATLAS + REMOVE BACKGROUND (same method as step2)
# ============================================================
atlas = datasets.fetch_atlas_schaefer_2018(
    n_rois=N_ROIS, yeo_networks=7, resolution_mm=2,
)
raw_labels = [lab.decode("utf-8") if isinstance(lab, bytes) else lab
              for lab in atlas.labels]
print(f"Raw labels returned: {len(raw_labels)}")
print(f"First raw label: {raw_labels[0]}")

# Background removal identical to step2: prefer atlas.indices == '0', else fallback.
if hasattr(atlas, "indices"):
    keep = [i for i, idx in enumerate(atlas.indices) if str(idx) != "0"]
    labels = [raw_labels[i] for i in keep]
    print(f"Removed background via atlas.indices=='0'. Labels now: {len(labels)}")
elif raw_labels and raw_labels[0] == "Background":
    labels = raw_labels[1:]
    print(f"Removed Background label (fallback). Labels now: {len(labels)}")
else:
    labels = raw_labels
    print(f"No background label found. Labels: {len(labels)}")

assert len(labels) == N_ROIS, f"Expected {N_ROIS}, got {len(labels)}"

# ============================================================
# PARSE LABELS
# Format: "7Networks_LH_Vis_1" or "7Networks_RH_DefaultB_PFCd_1"
# ============================================================
def resolve_yeo(yeo_raw):
    """Return (main_network, subdivision) by matching yeo_raw against the 7
    official networks as a prefix. Deterministic, no ambiguous regex.
    e.g. 'DefaultB' -> ('Default', 'DefaultB'); 'Vis' -> ('Vis', 'Vis')."""
    for net in sorted(YEO_NETWORKS_OFFICIAL, key=len, reverse=True):
        if yeo_raw == net or yeo_raw.startswith(net):
            return net, yeo_raw
    return None, yeo_raw

parsed = []
for idx, lab in enumerate(labels):
    parts = lab.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected label format at idx {idx}: {lab}")
    hemi       = parts[1]              # "LH" or "RH"
    yeo_raw    = parts[2]              # "Vis", "DefaultB", "ContA", ...
    sub_region = "_".join(parts[3:])

    yeo_network, yeo_subdivision = resolve_yeo(yeo_raw)
    if yeo_network is None:
        raise ValueError(
            f"Unexpected Yeo network at idx {idx} (raw '{yeo_raw}', label '{lab}')")

    parsed.append({
        "roi_idx"        : idx,
        "full_label"     : lab,
        "hemisphere"     : hemi,
        "yeo_network"    : yeo_network,
        "yeo_subdivision": yeo_subdivision,
        "sub_region"     : sub_region,
    })

df = pd.DataFrame(parsed)

# ============================================================
# VERIFY
# ============================================================
# ROI indices must be contiguous 0..N-1 (alignment with FC matrices)
assert list(df["roi_idx"]) == list(range(N_ROIS)), "ROI indices not contiguous 0..399"

print("\n--- Yeo-7 network distribution ---")
counts = df["yeo_network"].value_counts().reindex(YEO_NETWORKS_OFFICIAL)
print(counts)
print(f"\nTotal: {counts.sum()} (expected {N_ROIS})")
assert counts.sum() == N_ROIS, "Yeo counts do not sum to N_ROIS"
assert counts.notna().all(), "Some Yeo network has zero ROIs (parse error?)"

print("\n--- Hemisphere distribution ---")
print(df["hemisphere"].value_counts())

print("\n--- Yeo subdivisions present ---")
print(sorted(df["yeo_subdivision"].unique()))

print("\n--- First 10 rows ---")
print(df.head(10).to_string(index=False))
print("\n--- Last 5 rows ---")
print(df.tail(5).to_string(index=False))

# ============================================================
# SAVE
# ============================================================
out_csv = os.path.join(OUT_DIR, "schaefer400_yeo7_roi_info.csv")
df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")

out_labels = os.path.join(OUT_DIR, "schaefer400_raw_labels.txt")
with open(out_labels, "w") as f:
    for lab in raw_labels:
        f.write(lab + "\n")
print(f"Saved: {out_labels} (includes Background as line 1 if present)")