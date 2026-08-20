import os
import sys
from extractor import track_single_file, track_directory
from ranking import rank_tracks
from exporter import export_single_track_xml


# ── Configuration ─────────────────────────────────────────────────────────────

NC_SINGLE_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Limerick/AR_TC_result.nc"))
NC_FILES_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Feature_Detection/Results/TC-AR-Met3d"))
OUTPUT_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), "./Outputs"))

Z_TC  = 30.0   # Met3D zoom level for Tropical Cyclones
Z_AR  = 60.0   # Met3D zoom level for Atmospheric Rivers
TOP_N = 3      # Number of top-ranked tracks to prioritize and export


def run_pipeline():
    print("=" * 70)
    print("  METEOROLOGICAL FEATURE TRACKING & PRIORITIZATION PIPELINE")
    print("=" * 70)

    # 1. Extraction and Tracking
    if os.path.exists(NC_SINGLE_FILE):
        print(f"\n[1/3] Extracting & Tracking from dataset:\n      {NC_SINGLE_FILE}")
        results = track_single_file(NC_SINGLE_FILE, features=("TC", "AR"))
        tc_tracks, tc_blobs = results["TC"]
        ar_tracks, ar_blobs = results["AR"]
    elif os.path.exists(NC_FILES_DIR):
        print(f"\n[1/3] Extracting & Tracking from directory:\n      {NC_FILES_DIR}")
        nc_paths = [os.path.join(NC_FILES_DIR, f"{t}.nc") for t in range(12)]
        tc_tracks, tc_blobs = track_directory(nc_paths, feature="TC")
        ar_tracks, ar_blobs = track_directory(nc_paths, feature="AR")
    else:
        print("\n[ERROR] No valid dataset found in ../Limerick/ or ../Feature_Detection/Results/")
        sys.exit(1)

    print(f"      -> Detected {len(tc_tracks)} TC tracks")
    print(f"      -> Detected {len(ar_tracks)} AR tracks")

    # 2. Prioritization / Ranking
    print("\n[2/3] Prioritizing & Ranking Features...")
    all_ranked, top_features = rank_tracks(
        tc_tracks, ar_tracks, tc_blobs, ar_blobs, top_n=TOP_N
    )

    print("\n" + "-" * 70)
    print(f"{'Rank':<5} {'Feature':<8} {'Lifetime':<10} {'Speed(km/h)':<14} {'Split/Merge':<14} {'Growth Rate':<14} {'Score':<8}")
    print("-" * 70)
    for r in top_features:
        name = f"{r['feature']}-{r['id']}"
        print(f"#{r['rank']:<4} {name:<8} {r['lifetime']:<10} {r['avg_speed_kmh']:<14.2f} {r['split_merge']:<14.2f} {r['area_growth_rate']:<14.2f} {r['score']:<8.3f}")
    print("-" * 70)

    # 3. Export Met3D Camera Sequences
    print("\n[3/3] Exporting Camera Sequences for Top Features...")
    xml_dir = os.path.join(OUTPUT_DIR, "xml")
    os.makedirs(xml_dir, exist_ok=True)

    for r in top_features:
        z = Z_TC if r["feature"] == "TC" else Z_AR
        out_file = os.path.join(xml_dir, f"camera_sequence_{r['feature']}_{r['id']}.xml")
        export_single_track_xml(r["seq"], z_value=z, output_path=out_file)
        print(f"      -> Exported Rank #{r['rank']} ({r['feature']}-{r['id']}) to {out_file}")

    print("\nPipeline execution complete successfully.\n")


if __name__ == "__main__":
    run_pipeline()