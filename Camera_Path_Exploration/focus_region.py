"""
focus_region.py
===============
Finds the densest cluster of TC/AR activity per timestep
and renders a focused PNG map of that region.

Steps:
  1. Load TC and AR masks
  2. Find all blob centroids
  3. Cluster centroids — pick the single densest cluster
  4. Compute bounding box around that cluster
  5. Render focused PNG
"""

import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import ndimage
from sklearn.cluster import DBSCAN


# ── Config ────────────────────────────────────────────────────────────────────

NC_FILES_DIR  = "../Feature_detection/Results/TC-AR-Met3d"
OUTPUT_DIR    = "./Outputs/focus"
TIMESTEPS     = range(0, 12)

MIN_TC_PIXELS = 50
MIN_AR_PIXELS = 200

BOX_PADDING    = 20   # degrees of padding around the cluster
MIN_BOX_WIDTH  = 150  # minimum box width in degrees
MIN_BOX_HEIGHT = 60   # minimum box height in degrees

# DBSCAN: features within this many degrees are considered neighbours
CLUSTER_EPS = 40

LONS = np.linspace(-180, 180, 1152)
LATS = np.linspace(-90,   90,  768)

STYLE = {
    "bg":    "#050a14",
    "ocean": "#0a1628",
    "land":  "#1a2a1a",
    "TC":    "#ff4444",
    "AR":    "#00aaff",
}


# ── Blob Detection ────────────────────────────────────────────────────────────

def pixel_to_lonlat(px, py):
    lon = float(LONS[int(np.clip(px, 0, 1151))])
    lat = float(LATS[int(np.clip(py, 0,  767))])
    return lon, lat


def find_centroids(mask, min_size):
    """Find centroids of all blobs passing the size filter."""
    binary         = (mask > 0.5).astype(int)
    labeled, count = ndimage.label(binary)
    centroids      = []

    for blob_id in range(1, count + 1):
        blob = (labeled == blob_id)
        if blob.sum() < min_size:
            continue
        cy, cx   = ndimage.center_of_mass(blob)
        lon, lat = pixel_to_lonlat(cx, cy)
        centroids.append((lon, lat))

    return centroids


# ── Clustering ────────────────────────────────────────────────────────────────

def find_densest_cluster(centroids):
    """
    Use DBSCAN to find clusters of nearby features.
    Returns only the centroids of the single largest cluster.
    Outliers and smaller clusters are ignored.
    """
    if len(centroids) < 2:
        return centroids

    coords = np.array(centroids)
    labels = DBSCAN(eps=CLUSTER_EPS, min_samples=1).fit(coords).labels_

    # find label of the largest cluster (ignoring noise label -1)
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique) == 0:
        return centroids

    best_label    = unique[np.argmax(counts)]
    best_centoids = coords[labels == best_label]

    return [tuple(p) for p in best_centoids]


# ── Bounding Box ──────────────────────────────────────────────────────────────

def compute_bounding_box(centroids):
    """
    Compute a padded bounding box around the given centroids.
    Enforces minimum size and clamps to valid globe coordinates.
    """
    # focus on densest cluster only
    centroids = find_densest_cluster(centroids)

    lons = [c[0] for c in centroids]
    lats = [c[1] for c in centroids]

    lon_min = min(lons) - BOX_PADDING
    lon_max = max(lons) + BOX_PADDING
    lat_min = min(lats) - BOX_PADDING
    lat_max = max(lats) + BOX_PADDING

    # enforce minimum box size
    if (lon_max - lon_min) < MIN_BOX_WIDTH:
        center  = (lon_min + lon_max) / 2
        lon_min = center - MIN_BOX_WIDTH / 2
        lon_max = center + MIN_BOX_WIDTH / 2

    if (lat_max - lat_min) < MIN_BOX_HEIGHT:
        center  = (lat_min + lat_max) / 2
        lat_min = center - MIN_BOX_HEIGHT / 2
        lat_max = center + MIN_BOX_HEIGHT / 2

    # clamp to valid globe range
    lon_min = max(lon_min, -180)
    lon_max = min(lon_max,  180)
    lat_min = max(lat_min,  -90)
    lat_max = min(lat_max,   90)

    return lon_min, lon_max, lat_min, lat_max


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_focused_map(tc_mask, ar_mask, extent, timestep, output_dir):
    """Render a focused PNG of the given extent with TC/AR masks."""
    lon_min, lon_max, lat_min, lat_max = extent

    fig = plt.figure(figsize=(14, 8), facecolor=STYLE["bg"])
    ax  = plt.axes([0.05, 0.05, 0.9, 0.9],
                   projection=ccrs.PlateCarree())

    ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                  crs=ccrs.PlateCarree())
    ax.set_facecolor(STYLE["ocean"])

    ax.add_feature(cfeature.LAND,
                   facecolor=STYLE["land"], zorder=1)
    ax.add_feature(cfeature.COASTLINE,
                   edgecolor="#3a5a3a", linewidth=0.6, zorder=2)
    ax.add_feature(cfeature.BORDERS,
                   edgecolor="#2a4a2a", linewidth=0.3, zorder=2)
    ax.gridlines(color="white", alpha=0.1, linewidth=0.4,
                 draw_labels=True, x_inline=False, y_inline=False)

    if tc_mask is not None:
        ax.contourf(LONS, LATS,
                    np.where(tc_mask > 0.5, 1.0, np.nan),
                    levels=[0.5, 1.5], colors=[STYLE["TC"]],
                    alpha=0.55, transform=ccrs.PlateCarree(), zorder=3)

    if ar_mask is not None:
        ax.contourf(LONS, LATS,
                    np.where(ar_mask > 0.5, 1.0, np.nan),
                    levels=[0.5, 1.5], colors=[STYLE["AR"]],
                    alpha=0.4, transform=ccrs.PlateCarree(), zorder=3)

    ax.set_title(
        f"Timestep {timestep}  |  "
        f"lon [{lon_min:.0f}°, {lon_max:.0f}°]  "
        f"lat [{lat_min:.0f}°, {lat_max:.0f}°]",
        color="white", fontsize=11, pad=8
    )

    out_path = os.path.join(output_dir, f"focus_{timestep:02d}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for t in TIMESTEPS:
        nc_path = f"{NC_FILES_DIR}/{t}.nc"

        if not os.path.exists(nc_path):
            print(f"[t={t}] File not found, skipping.")
            continue

        ds      = xr.open_dataset(nc_path)
        tc_mask = np.array(ds["TC"][0])
        ar_mask = np.array(ds["AR"][0])
        ds.close()

        tc_centroids  = find_centroids(tc_mask, MIN_TC_PIXELS)
        ar_centroids  = find_centroids(ar_mask, MIN_AR_PIXELS)
        all_centroids = tc_centroids + ar_centroids

        if not all_centroids:
            print(f"[t={t}] No features detected, skipping.")
            continue

        extent   = compute_bounding_box(all_centroids)
        out_path = render_focused_map(
            tc_mask, ar_mask, extent, t, OUTPUT_DIR
        )

        print(f"[t={t}] TCs={len(tc_centroids)} "
              f"ARs={len(ar_centroids)} → {out_path}")


if __name__ == "__main__":
    main()