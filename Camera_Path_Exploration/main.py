import os
import sys
import json
import numpy as np
from extractor import track_single_file, track_directory
from ranking import rank_tracks
from visualize_blob_evolution import export_animations

# ── Configuration ─────────────────────────────────────────────────────────────

# You can switch this to either the single .nc file or the directory of .nc files
DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Limerick/AR_TC_result.nc"))
#DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Feature_Detection/Results/TC-AR-Met3d"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "./Outputs"))

Z_TC  = 30.0   # Met3D zoom level for Tropical Cyclones
Z_AR  = 60.0   # Met3D zoom level for Atmospheric Rivers
TOP_N = 3      # Number of top-ranked tracks to prioritize and export


def run_pipeline():
    print("=" * 70)
    print("  METEOROLOGICAL FEATURE TRACKING & PRIORITIZATION PIPELINE")
    print("=" * 70)

    # 1. Extraction and Tracking
    if not os.path.exists(DATASET_PATH):
        print(f"\n[ERROR] Dataset path does not exist: {DATASET_PATH}")
        sys.exit(1)

    if os.path.isfile(DATASET_PATH):
        print(f"\n[1/4] Extracting & Tracking from dataset file:\n      {DATASET_PATH}")
        results = track_single_file(DATASET_PATH, features=("TC", "AR"))
        tc_tracks, tc_blobs = results["TC"]
        ar_tracks, ar_blobs = results["AR"]
    elif os.path.isdir(DATASET_PATH):
        print(f"\n[1/4] Extracting & Tracking from directory:\n      {DATASET_PATH}")
        nc_paths = [os.path.join(DATASET_PATH, f"{t}.nc") for t in range(12)]
        tc_tracks, tc_blobs = track_directory(nc_paths, feature="TC")
        ar_tracks, ar_blobs = track_directory(nc_paths, feature="AR")
    else:
        print("\n[ERROR] Invalid dataset path format.")
        sys.exit(1)

    print(f"      -> Detected {len(tc_tracks)} TC tracks")
    print(f"      -> Detected {len(ar_tracks)} AR tracks")

    # 2. Prioritization / Ranking
    print("\n[2/4] Prioritizing & Ranking Features...")
    all_ranked, top_features = rank_tracks(
        tc_tracks, ar_tracks, tc_blobs, ar_blobs, top_n=TOP_N
    )

    print("\n" + "-" * 105)
    print(f"{'Rank':<5} {'Feature':<8} {'Motion':<8} {'Lifetime':<10} {'Distance(km)':<14} {'Max Extent(km)':<16} {'Volatility':<14} {'Z-Score':<8}")
    print("-" * 105)
    for r in top_features:
        name = f"{r['feature']}-{r['id']}"
        print(f"#{r['rank']:<4} {name:<8} {r['camera_motion']:<8} {r['lifetime']:<10} {r['total_distance_traveled']:<14.1f} {r['max_extent_km']:<16.1f} {r['shape_volatility']:<14.1f} {r['score']:<8.3f}")
    print("-" * 105)

    # 3. Export Data for Camera Animation
    print("\n[3/4] Exporting Track Data to JSON...")
    json_dir = os.path.join(OUTPUT_DIR, "json")
    os.makedirs(json_dir, exist_ok=True)
    out_file = os.path.join(json_dir, "top_features_camera_data.json")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, np.datetime64): return str(obj)
            return super(NumpyEncoder, self).default(obj)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(top_features, f, indent=4, cls=NumpyEncoder)
    
    print(f"      -> Exported all top features data to {out_file}")

    # 4. Generate Animated Visualizations
    print("\n[4/4] Generating Animated Visualizations for Top Features...")
    export_animations(DATASET_PATH, top_features, OUTPUT_DIR)

    print("\nPipeline execution complete successfully.\n")


if __name__ == "__main__":
    run_pipeline()