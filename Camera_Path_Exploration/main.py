"""
Meteorological Feature Tracking & Camera Path Planning Pipeline
"""

import os
import sys
import json
import numpy as np

from extractor import track_single_file, track_directory
from ranking import rank_tracks, DEFAULT_MIN_LIFETIME, DEFAULT_TOP_N
from visualize_blob_evolution import export_animations


# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================
# Path to input NetCDF dataset (single multi-timestep file or directory of per-timestep .nc files):
DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Limerick/AR_TC_result.nc"))
# DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Feature_Detection/Results/TC-AR-Met3d"))

# Output directory for generated JSON camera paths and animated GIF visualizations:
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "./Outputs"))

# Number of top-ranked features to prioritize and export:
TOP_N = DEFAULT_TOP_N

# Minimum number of timesteps a track must persist to qualify for ranking:
MIN_LIFETIME = DEFAULT_MIN_LIFETIME


# ==============================================================================
# PIPELINE ORCHESTRATION
# ==============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy scalar and array types cleanly."""
    def default(self, obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.datetime64):
            return str(obj)
        return super().default(obj)


def run_pipeline():
    print("=" * 110)
    print("      METEOROLOGICAL FEATURE TRACKING & CAMERA PATH PLANNING PIPELINE")
    print("=" * 110)

    # --------------------------------------------------------------------------
    # Stage 1: Feature Extraction & Temporal Tracking
    # --------------------------------------------------------------------------
    if not os.path.exists(DATASET_PATH):
        print(f"\n[ERROR] Dataset path does not exist: {DATASET_PATH}")
        sys.exit(1)

    print(f"Feature Extraction & Temporal Tracking...")
    print(f"            Input Dataset: {DATASET_PATH}")

    if os.path.isfile(DATASET_PATH):
        results = track_single_file(DATASET_PATH, features=("TC", "AR"))
        tc_tracks, tc_blobs = results["TC"]
        ar_tracks, ar_blobs = results["AR"]
    elif os.path.isdir(DATASET_PATH):
        nc_paths = [os.path.join(DATASET_PATH, f"{t}.nc") for t in range(12)]
        tc_tracks, tc_blobs = track_directory(nc_paths, feature="TC")
        ar_tracks, ar_blobs = track_directory(nc_paths, feature="AR")
    else:
        print("\n[ERROR] Invalid dataset path format.")
        sys.exit(1)

    print(f"            -> Extracted {len(tc_tracks)} Tropical Cyclone (TC) candidate tracks")
    print(f"            -> Extracted {len(ar_tracks)} Atmospheric River (AR) candidate tracks")

    # --------------------------------------------------------------------------
    # Stage 2: Multi-Criteria Prioritization & Ranking
    # --------------------------------------------------------------------------
    print(f"Multi-Criteria Prioritization & Z-Score Ranking...")
    all_ranked, top_features = rank_tracks(
        tc_tracks, ar_tracks, tc_blobs, ar_blobs,
        min_lifetime=MIN_LIFETIME, top_n=TOP_N
    )

    print(f"            -> Filtered tracks with lifetime >= {MIN_LIFETIME} steps")
    print(f"            -> Evaluated {len(all_ranked)} valid tracks across 3 motion domains")

    # Display clean formatted summary table
    print("\n" + "=" * 115)
    print(f"{'Rank':<5} {'Feature':<8} {'Motion':<8} {'Lifetime':<10} {'Dist (km)':<14} {'Peak Area (px)':<16} {'Max Ext (km)':<14} {'Volatility':<14} {'Z-Score':<8}")
    print("-" * 115)
    for r in top_features:
        name = f"{r['feature']}-{r['id']}"
        print(f"#{r['rank']:<4} {name:<8} {r['camera_motion']:<8} "
              f"{r['lifetime']:<10} {r['total_distance_traveled']:<14.1f} "
              f"{r['max_area_px']:<16} {r['max_extent_km']:<14.1f} "
              f"{r['shape_volatility']:<14.1f} {r['score']:<8.3f}")


    # --------------------------------------------------------------------------
    # Stage 3: Export Camera Trajectory Data to JSON
    # --------------------------------------------------------------------------
    print(f"Exporting Camera Trajectory Data to JSON...")
    json_dir = os.path.join(OUTPUT_DIR, "json")
    os.makedirs(json_dir, exist_ok=True)
    json_out_file = os.path.join(json_dir, "top_features_camera_data.json")

    with open(json_out_file, "w", encoding="utf-8") as f:
        json.dump(top_features, f, indent=4, cls=NumpyEncoder)

    print(f"            -> Successfully exported {len(top_features)} top feature paths to:")
    print(f"               {json_out_file}")

    # --------------------------------------------------------------------------
    # Stage 4: Generate Animated Visualizations
    # --------------------------------------------------------------------------
    print(f"Generating Animated Visualizations for Top Features...")
    export_animations(DATASET_PATH, top_features, OUTPUT_DIR)

if __name__ == "__main__":
    run_pipeline()