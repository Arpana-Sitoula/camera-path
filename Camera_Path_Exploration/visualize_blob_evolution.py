import os
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Configuration
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "./Outputs"))

def export_animations(dataset_path, top_features, output_dir):
    is_directory = os.path.isdir(dataset_path)
    
    if not is_directory:
        print(f"      -> Loading NetCDF dataset from {dataset_path}...")
        ds = xr.open_dataset(dataset_path)
        lons = ds["lon"].values
        lats = ds["lat"].values
        extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    else:
        print(f"      -> Loading first NetCDF file from directory {dataset_path} to get grid bounds...")
        first_file = os.path.join(dataset_path, "0.nc")
        ds = xr.open_dataset(first_file)
        lons = ds["lon"].values
        lats = ds["lat"].values
        extent = [lons.min(), lons.max(), lats.min(), lats.max()]
        ds.close()

    for feature_data in top_features:
        feature_type = feature_data["feature"]
        feature_id = feature_data["id"]
        seq = feature_data["seq"]
        camera_motion = feature_data["camera_motion"]
        
        out_filename = os.path.join(output_dir, f"animated_Rank{feature_data['rank']}_{feature_type}_{feature_id}.gif")
        print(f"      -> Generating animation for Rank #{feature_data['rank']} ({feature_type}-{feature_id})...")

        # 1. Determine fixed map boundaries so we can see actual movement/distance
        all_lons = [p["lon"] for p in seq]
        all_lats = [p["lat"] for p in seq]
        
        # Add a 30-degree buffer around the entire track's extreme points
        buffer = 30.0
        min_lon, max_lon = min(all_lons) - buffer, max(all_lons) + buffer
        min_lat, max_lat = min(all_lats) - buffer, max(all_lats) + buffer

        # Set up the figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # We will update this image in the animation loop
        im = ax.imshow(np.zeros((len(lats), len(lons))), extent=extent, origin='lower', cmap='Blues', alpha=0.9, vmin=0, vmax=1)
        
        # Centroid marker and full track line
        track_line, = ax.plot([], [], 'ro-', markersize=4, alpha=0.5, label="Track History")
        current_centroid, = ax.plot([], [], 'yo', markersize=8, markeredgecolor='black', label="Current Centroid")
        
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle='--', alpha=0.6)

        title_template = f"Rank #{feature_data['rank']}: {feature_type}-{feature_id} ({camera_motion} Motion)\nTimestep: {{t}} | Area: {{area}} px"
        
        history_lons = []
        history_lats = []

        def update(frame_idx):
            point = seq[frame_idx]
            t = point["timestep"]
            
            # Extract 2D mask
            if is_directory:
                current_ds = xr.open_dataset(os.path.join(dataset_path, f"{t}.nc"))
                mask_slice = current_ds[feature_type]
            else:
                mask_slice = ds[feature_type].isel(time=t)
                
            while mask_slice.ndim > 2:
                mask_slice = mask_slice.squeeze()
            mask = np.asarray(mask_slice)
            
            # Update image data
            im.set_data(mask)
            
            # Update track history and current point
            c_lon, c_lat = point["lon"], point["lat"]
            history_lons.append(c_lon)
            history_lats.append(c_lat)
            
            track_line.set_data(history_lons, history_lats)
            current_centroid.set_data([c_lon], [c_lat])
            
            ax.set_title(title_template.format(t=t, area=point['area']))
            return im, track_line, current_centroid

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(seq), interval=200, blit=False)
        
        # Save as GIF using Pillow (built into matplotlib)
        ani.save(out_filename, writer=animation.PillowWriter(fps=5))
        plt.close()
        print(f"         Saved animation: {out_filename}")

    if not is_directory:
        ds.close()

