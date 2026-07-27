"""
Renders Schaefer-400 parcels coloured by Yeo-7 network on the inflated
fsaverage surface (lateral + medial, both hemispheres). Parcel boundaries
are baked into the ROI map as a dedicated black colour index.

In: none (atlas + fsaverage surface fetched from nilearn 0.13.1).
Out: figure_yeo7_surface.png
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from nilearn import datasets, image, surface, plotting

# ============================== SETTINGS ==============================
N_ROIS = 400
YEO_NETWORKS = 7
RESOLUTION_MM = 1         # deliberately 1mm (vs. the 2mm analysis atlas): smoother
                          # surface rendering only, no matrix computed here
MESH = "fsaverage6"       # denser than fsaverage5: one-vertex-wide borders stay thin
OUT_DIR = "."
OUT_BASENAME = "figure_yeo7_surface"
DPI = 300
BOUNDARY_ID = 0           # colour index reserved for parcel borders
# =====================================================================

YEO7_NAMES = [
    "Visual",
    "Somatomotor",
    "Dorsal Attention",
    "Ventral Attention",
    "Limbic",
    "Frontoparietal",
    "Default",
]

YEO7_RGB = {
    "Visual":            (120,  18, 134),
    "Somatomotor":       ( 70, 130, 180),
    "Dorsal Attention":  (  0, 118,  14),
    "Ventral Attention": (196,  58, 250),
    "Limbic":            (220, 248, 164),
    "Frontoparietal":    (230, 148,  34),
    "Default":           (205,  62,  78),
}


def parse_network(label):
    """Yeo-7 network token from a Schaefer parcel label. None for background."""
    if isinstance(label, bytes):
        label = label.decode("utf-8")
    if label == "Background":
        return None
    token_map = {
        "Vis": "Visual",
        "SomMot": "Somatomotor",
        "DorsAttn": "Dorsal Attention",
        "SalVentAttn": "Ventral Attention",
        "Limbic": "Limbic",
        "Cont": "Frontoparietal",
        "Default": "Default",
    }
    for tok, name in token_map.items():
        if re.search(rf"_{tok}_", label) or re.search(rf"_{tok}$", label):
            return name
    raise ValueError(f"Could not parse network from label: {label!r}")


def border_vertices(parcel_tex, faces):
    """Vertices on a parcel border.

    A vertex is a border vertex if it shares a face with a vertex carrying a
    different non-zero parcel id. Only one side of each border is marked, so
    the resulting line stays one vertex wide.
    """
    ids = np.rint(parcel_tex).astype(int)
    mask = np.zeros(ids.shape[0], dtype=bool)

    for a, b in ((0, 1), (1, 2), (2, 0)):
        va, vb = faces[:, a], faces[:, b]
        mixed = (ids[va] != ids[vb]) & (ids[va] > 0) & (ids[vb] > 0)
        # mark the endpoint with the lower id only -> single-sided border
        lower_is_a = ids[va] < ids[vb]
        mask[va[mixed & lower_is_a]] = True
        mask[vb[mixed & ~lower_is_a]] = True

    return mask


def main():
    print("Fetching Schaefer-400 atlas (Yeo-7) ...")
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=N_ROIS, yeo_networks=YEO_NETWORKS, resolution_mm=RESOLUTION_MM,
    )
    atlas_img = image.load_img(atlas.maps)
    labels = list(atlas.labels)

    # Parcel value v -> Yeo network id (1..7); background/0 -> 0
    name_to_id = {name: i + 1 for i, name in enumerate(YEO7_NAMES)}
    parcel_to_net = np.zeros(len(labels), dtype=int)
    counts = {name: 0 for name in YEO7_NAMES}
    for value, lab in enumerate(labels):
        net = parse_network(lab)
        if net is None:
            continue
        parcel_to_net[value] = name_to_id[net]
        counts[net] += 1
    print(f"  parcels assigned: {int((parcel_to_net > 0).sum())}")
    for name in YEO7_NAMES:
        print(f"    {name:<18} {counts[name]}")

    # Recolour volume: parcel id -> network id
    atlas_data = np.asarray(atlas_img.dataobj).astype(int)
    assert atlas_data.max() < len(parcel_to_net), "parcel/label misalignment"
    net_img = image.new_img_like(atlas_img, parcel_to_net[atlas_data])

    print(f"Fetching {MESH} surface ...")
    fsavg = datasets.fetch_surf_fsaverage(mesh=MESH)

    # Project the network image and the original parcel image onto the surface.
    # 'nearest_most_frequent' keeps ids discrete (no blending) and assigns each
    # vertex the most frequent id in its neighbourhood. The parcel-id texture is
    # required for the borders: within a network, neighbouring parcels share the
    # network id and would show no border.
    print("Projecting volume to surface ...")
    roi_map = {}
    for hemi, pial, infl in (("left", fsavg.pial_left, fsavg.infl_left),
                             ("right", fsavg.pial_right, fsavg.infl_right)):
        net_tex = surface.vol_to_surf(
            net_img, pial, interpolation="nearest_most_frequent")
        parcel_tex = surface.vol_to_surf(
            atlas_img, pial, interpolation="nearest_most_frequent")
        faces = surface.load_surf_mesh(infl).faces

        m = np.rint(net_tex).astype(float)
        m[m < 0.5] = np.nan                       # background -> transparent
        m[border_vertices(parcel_tex, faces)] = BOUNDARY_ID
        roi_map[hemi] = m

    colors = [tuple(c / 255.0 for c in YEO7_RGB[n]) for n in YEO7_NAMES]
    cmap = ListedColormap([(0.0, 0.0, 0.0)] + colors)   # index 0 = border

    views = [
        ("left",  "lateral", fsavg.infl_left,  fsavg.sulc_left),
        ("left",  "medial",  fsavg.infl_left,  fsavg.sulc_left),
        ("right", "lateral", fsavg.infl_right, fsavg.sulc_right),
        ("right", "medial",  fsavg.infl_right, fsavg.sulc_right),
    ]

    fig, axes = plt.subplots(
        1, 4, figsize=(16, 4), subplot_kw={"projection": "3d"})

    for ax, (hemi, view, mesh, sulc) in zip(axes, views):
        plotting.plot_surf_roi(
            mesh,
            roi_map=roi_map[hemi],
            hemi=hemi,
            view=view,
            bg_map=sulc,
            bg_on_data=False,
            cmap=cmap,
            vmin=BOUNDARY_ID,
            vmax=YEO_NETWORKS,
            axes=ax,
            figure=fig,
            colorbar=False,
        )
        ax.set_title(f"{hemi} {view}", fontsize=12)

    handles = [Patch(facecolor=colors[i], label=YEO7_NAMES[i])
               for i in range(YEO_NETWORKS)]
    fig.legend(handles=handles, loc="lower center", ncol=YEO_NETWORKS,
               frameon=False, fontsize=13, bbox_to_anchor=(0.5, 0.02))

    fig.subplots_adjust(bottom=0.14, wspace=0.0)

    png = os.path.join(OUT_DIR, OUT_BASENAME + ".png")
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    print(f"Saved:\n  {png}")


if __name__ == "__main__":
    main()