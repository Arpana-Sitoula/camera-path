"""
Meteorological Feature Evolution & Camera Path Visualization
============================================================
This module renders animated GIF visualizations for prioritized weather tracks,
showing the 2D segmentation probability mask evolution, the centroid path history,
and key spatial metrics (timestep, pixel area, physical extent).
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================
# Geographic coordinate buffer (in degrees) around the track to frame the visualization:
MAP_BUFFER_DEG = 30.0

# Figure size (width, height) in inches:
FIGURE_SIZE = (10, 8)

# Animation frames per second and frame interval (ms):
ANIMATION_FPS = 5
FRAME_INTERVAL_MS = 200

# Color map for feature segmentation masks:
FEATURE_CMAP = "Blues"


# ==============================================================================
# ANIMATION GENERATOR
# ==============================================================================

def export_animations(dataset_path: str, top_features: list[dict], output_dir: str):
    """
    Generate and save GIF animations for each prioritized feature track.

    Parameters:
        dataset_path (str): Path to input NetCDF dataset (.nc file or directory).
        top_features (list[dict]): Ranked track metadata and point sequences.
        output_dir (str): Destination directory for saved GIF files.
    """
    os.makedirs(output_dir, exist_ok=True)
    is_directory = os.path.isdir(dataset_path)

    # 1. Determine global coordinate grid extents
    if not is_directory:
        print(f"            -> Loading coordinate grid from {dataset_path}...")
        ds = xr.open_dataset(dataset_path)
        lons = ds["lon"].values
        lats = ds["lat"].values
        global_extent = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]
    else:
        first_file = os.path.join(dataset_path, "0.nc")
        print(f"            -> Loading coordinate grid from {first_file}...")
        ds = xr.open_dataset(first_file)
        lons = ds["lon"].values
        lats = ds["lat"].values
        global_extent = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]
        ds.close()

    # 2. Render animation for each top-ranked feature
    for feature_data in top_features:
        rank = feature_data["rank"]
        feature_type = feature_data["feature"]
        feature_id = feature_data["id"]
        seq = feature_data["seq"]
        camera_motion = feature_data["camera_motion"]

        out_filename = os.path.join(
            output_dir, f"animated_Rank{rank}_{feature_type}_{feature_id}.gif"
        )
        print(f"            -> Rendering Rank #{rank} ({feature_type}-{feature_id} | {camera_motion} Motion)...")

        # Compute bounding map boundaries centered on the entire track with a buffer
        all_lons = [p["lon"] for p in seq]
        all_lats = [p["lat"] for p in seq]
        min_lon = max(float(lons.min()), min(all_lons) - MAP_BUFFER_DEG)
        max_lon = min(float(lons.max()), max(all_lons) + MAP_BUFFER_DEG)
        min_lat = max(float(lats.min()), min(all_lats) - MAP_BUFFER_DEG)
        max_lat = min(float(lats.max()), max(all_lats) + MAP_BUFFER_DEG)

        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

        # Base 2D segmentation mask image
        im = ax.imshow(
            np.zeros((len(lats), len(lons))),
            extent=global_extent,
            origin="lower",
            cmap=FEATURE_CMAP,
            alpha=0.9,
            vmin=0,
            vmax=1,
        )

        # Centroid markers and trajectory trail
        track_line, = ax.plot([], [], "ro-", markersize=4, alpha=0.6, label="Centroid Path History")
        current_centroid, = ax.plot([], [], "yo", markersize=8, markeredgecolor="black", label="Current Centroid")

        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.5)

        title_template = (
            f"Rank #{rank}: {feature_type}-{feature_id} ({camera_motion} Motion)\n"
            f"Timestep: {{t}} | Area: {{area:,}} px | Extent: {{extent:.0f}} km"
        )

        history_lons = []
        history_lats = []

        def update(frame_idx):
            point = seq[frame_idx]
            t = point["timestep"]

            # Extract 2D segmentation slice for current timestep
            if is_directory:
                cur_ds = xr.open_dataset(os.path.join(dataset_path, f"{t}.nc"))
                mask_slice = cur_ds[feature_type]
            else:
                mask_slice = ds[feature_type].isel(time=t)

            while mask_slice.ndim > 2:
                mask_slice = mask_slice.squeeze()
            mask = np.asarray(mask_slice)

            if is_directory:
                cur_ds.close()

            # Update raster mask
            im.set_data(mask)

            # Update trajectory path
            c_lon, c_lat = point["lon"], point["lat"]
            history_lons.append(c_lon)
            history_lats.append(c_lat)

            track_line.set_data(history_lons, history_lats)
            current_centroid.set_data([c_lon], [c_lat])

            area_val = point.get("area", 0)
            extent_val = point.get("extent_km", 0.0)
            ax.set_title(title_template.format(t=t, area=area_val, extent=extent_val))

            return im, track_line, current_centroid

        ani = animation.FuncAnimation(
            fig, update, frames=len(seq), interval=FRAME_INTERVAL_MS, blit=False
        )

        ani.save(out_filename, writer=animation.PillowWriter(fps=ANIMATION_FPS))
        plt.close(fig)
        print(f"               Saved: {out_filename}")

    if not is_directory:
        ds.close()
