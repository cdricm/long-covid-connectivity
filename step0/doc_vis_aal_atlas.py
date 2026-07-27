"""
Renders the AAL atlas as a 3D volume rendering, one surface per region
(lateral and medial view, both hemispheres). Colours are arbitrary and
cycle through a qualitative palette; AAL carries no functional network
assignment.

In: none (atlas fetched from nilearn 0.13.1).
Out: figure_aal_render.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from nilearn import datasets, image
from scipy import ndimage
from skimage import measure

# ============================== SETTINGS ==============================
OUT_DIR = "."
OUT_BASENAME = "figure_aal_render"
DPI = 300
N_COLORS = 10
PALETTE = "tab10"
SMOOTH_SIGMA = 0.6        # gaussian smoothing of each region mask before
                          # marching cubes; higher = rounder, less blocky
SMOOTH_ITER = 30          # mesh smoothing iterations after extraction
PANEL_PX = (900, 900)     # offscreen render size per panel

# Camera position per panel, as (hemisphere, azimuth label, view vector)
VIEWS = [
    ("left",  "left lateral",  (-1, 0, 0)),
    ("left",  "left medial",   ( 1, 0, 0)),
    ("right", "right lateral", ( 1, 0, 0)),
    ("right", "right medial",  (-1, 0, 0)),
]
# =====================================================================


def region_mesh(mask, affine, sigma, iterations):
    """Smoothed surface mesh of a binary region mask, in world coordinates."""
    vol = ndimage.gaussian_filter(mask.astype(float), sigma=sigma)
    if vol.max() < 0.5:
        return None
    try:
        verts, faces, _, _ = measure.marching_cubes(vol, level=0.5)
    except (ValueError, RuntimeError):
        return None
    if len(verts) == 0:
        return None

    # voxel indices -> MNI world coordinates
    verts = nib_apply_affine(affine, verts)

    faces_pv = np.hstack(
        [np.full((len(faces), 1), 3, dtype=np.int64), faces]).ravel()
    mesh = pv.PolyData(verts, faces_pv)
    return mesh.smooth(n_iter=iterations, relaxation_factor=0.1)


def nib_apply_affine(affine, coords):
    """Apply a 4x4 affine to an (n, 3) array of voxel coordinates."""
    return coords @ affine[:3, :3].T + affine[:3, 3]


def main():
    print("Fetching AAL atlas ...")
    atlas = datasets.fetch_atlas_aal(version="SPM12")
    atlas_img = image.load_img(atlas.maps)
    atlas_data = np.asarray(atlas_img.dataobj).astype(int)
    affine = atlas_img.affine

    region_values = np.array(sorted(v for v in np.unique(atlas_data) if v != 0))
    print(f"  regions in volume: {len(region_values)}")

    # Label code -> region name, from the atlas' own index list
    code_to_name = {int(code): name
                    for code, name in zip(atlas.indices, atlas.labels)}

    # Colour is keyed on the base name without the _L / _R suffix, so that
    # homologous regions share a colour across hemispheres.
    def base_name(name):
        return name[:-2] if name.endswith(("_L", "_R")) else name

    base_names = sorted({base_name(code_to_name[int(v)])
                         for v in region_values})
    name_to_colour_idx = {n: i % N_COLORS for i, n in enumerate(base_names)}
    print(f"  region pairs: {len(base_names)}")

    base = plt.get_cmap(PALETTE)
    colors = [base(i / (N_COLORS - 1))[:3] for i in range(N_COLORS)]

    print("Extracting region surfaces ...")
    meshes = []
    for value in region_values:
        name = code_to_name[int(value)]
        hemi = ("left" if name.endswith("_L")
                else "right" if name.endswith("_R")
        else None)
        if hemi is None:
            # midline structures (vermis) carry no hemisphere suffix;
            # assigned by the sign of their centre of mass in x
            com_vox = np.array(ndimage.center_of_mass(atlas_data == value))
            com_world = nib_apply_affine(affine, com_vox[None, :])[0]
            hemi = "left" if com_world[0] < 0 else "right"

        mesh = region_mesh(atlas_data == value, affine,
                           SMOOTH_SIGMA, SMOOTH_ITER)
        if mesh is None:
            print(f"  skipped region {value} ({name}, no surface)")
            continue
        colour = colors[name_to_colour_idx[base_name(name)]]
        meshes.append((hemi, mesh, colour))
    print(f"  surfaces extracted: {len(meshes)}")

    print("Rendering panels ...")
    panels = []
    for hemi, title, direction in VIEWS:
        pl = pv.Plotter(off_screen=True, window_size=PANEL_PX)
        pl.set_background("white")
        for mesh_hemi, mesh, colour in meshes:
            if mesh_hemi != hemi:
                continue
            pl.add_mesh(mesh, color=colour, smooth_shading=True,
                        specular=0.15, ambient=0.35)
        pl.view_vector(direction, viewup=(0, 0, 1))
        pl.camera.zoom(1.3)
        panels.append((title, pl.screenshot(return_img=True)))
        pl.close()

    fig, axes = plt.subplots(1, len(panels), figsize=(16, 4.5))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.subplots_adjust(wspace=0.01)

    png = os.path.join(OUT_DIR, OUT_BASENAME + ".png")
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    print(f"Saved:\n  {png}")


if __name__ == "__main__":
    main()