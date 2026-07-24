"""
Threshold-Stabilitaet CP0189 - 3 Atlanten x 4 Densities, sagittale Ansicht.
signed |r|-Thresholding (nur Visualisierung; Pipeline nutzt positive-only).
"""

import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets, plotting, image as nimg

# ============================ SETTINGS ============================
BASE = "/mnt/d87cc26d-5470-443c-81c1-e09b68ee4730/Cedric/analysis_outputs"
SUBJECT = "CP0189"

ATLASES = [
    ("Schaefer-400", "step2_schaefer400", 400),
    ("Schaefer-100", "schaefer100", 100),
    ("AAL",          "aal",         116),
]
DENSITIES = [0.05, 0.15, 0.25, 0.50]

OUT_PATH  = f"{SUBJECT}_threshold_stability_grid.png"
EDGE_LW   = 0.25
NODE_SIZE = 5
SAVE_DPI  = 220
# ==================================================================


def matrix_path(folder):
    return f"{BASE}/{folder}/step2_pipeline/comet_matrices/{SUBJECT}_connectivity_comet.npy"


def proportional_threshold(W, density):
    W = W.copy()
    np.fill_diagonal(W, 0.0)
    n = W.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = W[iu]
    k = int(round(density * vals.size))
    out = np.zeros_like(W)
    if k <= 0:
        return out
    keep = np.argsort(np.abs(vals))[::-1][:k]
    r, c = iu[0][keep], iu[1][keep]
    out[r, c] = vals[keep]
    out[c, r] = vals[keep]
    return out


def schaefer_coords(n_rois):
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, yeo_networks=7, resolution_mm=2)
    return plotting.find_parcellation_cut_coords(labels_img=atlas["maps"])


def aal_coords():
    atlas = datasets.fetch_atlas_aal(version="SPM12")
    img = nimg.load_img(atlas["maps"])
    data = img.get_fdata()
    vals = np.sort(np.unique(data)[np.unique(data) != 0])
    coords = np.zeros((vals.size, 3))
    for i, v in enumerate(vals):
        vox = np.argwhere(data == v).mean(axis=0)
        coords[i] = nimg.coord_transform(vox[0], vox[1], vox[2], img.affine)
    return coords


def load_atlas_data():
    """Laedt Matrix + Koordinaten + gemeinsame vmax pro Atlas."""
    out = []
    for name, folder, n in ATLASES:
        W = np.load(matrix_path(folder)).astype(float)
        np.fill_diagonal(W, 0.0)
        if W.shape[0] != n:
            raise ValueError(f"{name}: Matrix n={W.shape[0]} != erwartet {n}")
        coords = aal_coords() if folder == "aal" else schaefer_coords(n)
        if coords.shape[0] != n:
            raise ValueError(f"{name}: coords n={coords.shape[0]} != Matrix n={n}")
        vmax = max(np.abs(proportional_threshold(W, d)).max() for d in DENSITIES)
        out.append((name, W, coords, vmax))
        print(f"{name}: n={n}, vmax={vmax:.3f}")
    return out

def signed_counts(W):
    iu = np.triu_indices(W.shape[0], k=1)
    nz = W[iu][W[iu] != 0]
    return int(np.sum(nz > 0)), int(np.sum(nz < 0))

def first_negative_density(W):
    for d in np.round(np.arange(0.01, 1.01, 0.01), 2):
        _, nneg = signed_counts(proportional_threshold(W, d))
        if nneg > 0:
            return d
    return None

def main():
    data = load_atlas_data()
    nrows, ncols = len(ATLASES), len(DENSITIES)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.8 * nrows))
    axes = np.atleast_2d(axes)

    for i, (name, W, coords, vmax) in enumerate(data):
        for j, d in enumerate(DENSITIES):
            ax = axes[i, j]
            Wt = proportional_threshold(W, d)
            plotting.plot_connectome(
                Wt, coords,
                edge_threshold=None,
                edge_cmap="RdBu_r",
                edge_vmin=-vmax, edge_vmax=vmax,
                node_size=NODE_SIZE, node_color="black",
                display_mode="x",          # sagittal (seitlich)
                axes=ax, colorbar=False, annotate=False,
                edge_kwargs={"linewidth": EDGE_LW},
            )
            # Spaltenkopf (Density) nur in oberster Zeile
            if i == 0:
                ax.set_title(f"{d:.0%}", fontsize=15, fontweight="bold", pad=6)
            # Atlas-Label nur in erster Spalte, links neben dem Plot
            if j == 0:
                ax.text(
                    -0.04, 0.5, name,
                    transform=ax.transAxes,
                    fontsize=15, fontweight="bold",
                    ha="right", va="center", rotation=90,
                )

    fig.suptitle(
        f"{SUBJECT} | threshold stability across densities | sagittal view | "
        f"signed |r|-threshold (red = positive, blue = negative)",
        fontsize=13, y=0.99
    )
    # gemeinsame Density-Achsenbeschriftung oben
    fig.text(0.5, 0.945, "density", ha="center", fontsize=13, style="italic")

    fig.subplots_adjust(
        left=0.07, right=0.995, top=0.91, bottom=0.01,
        wspace=0.02, hspace=0.05
    )
    fig.savefig(OUT_PATH, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"-> gespeichert: {OUT_PATH}")
    plt.close(fig)

    print("\nerste negative Kante ab density = ?")
    for name, W, _, _ in data:
        d = first_negative_density(W)
        label = f"{d:.0%}" if d is not None else "keine bis 100%"
        print(f"    {name}: density = {label}")

if __name__ == "__main__":
    main()
