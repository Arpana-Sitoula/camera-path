from extractor import track_features_across_time
from visualizer import save_camera_map
from exporter import export_tracks_to_xml
import json
import os


# ── Config ────────────────────────────────────────────────────────────────────

NC_FILES_DIR = "../Feature_detection/Results/TC-AR-Met3d"
OUTPUT_DIR   = "./Outputs"
TIMESTEPS    = range(0, 12)


# ── Steps ─────────────────────────────────────────────────────────────────────

def get_nc_paths():
    """Build list of all .nc file paths for every timestep."""
    return [f"{NC_FILES_DIR}/{t}.nc" for t in TIMESTEPS]


def track_all_features(nc_paths):
    """
    Track TC and AR blobs across all timesteps.
    Returns two track dicts — one for TC, one for AR.
    """
    print("\nTracking Tropical Cyclones...")
    tc_tracks = track_features_across_time(nc_paths, feature="TC")
    print(f"  → {len(tc_tracks)} TC tracks found")

    print("\nTracking Atmospheric Rivers...")
    ar_tracks = track_features_across_time(nc_paths, feature="AR")
    print(f"  → {len(ar_tracks)} AR tracks found")

    return tc_tracks, ar_tracks


def export_xml(tc_tracks, ar_tracks):
    """Export tracked camera path to Met3D XML."""
    print("\nExporting to Met3D XML...")
    export_tracks_to_xml(tc_tracks, ar_tracks, f"{OUTPUT_DIR}/xml")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    nc_paths          = get_nc_paths()
    tc_tracks, ar_tracks = track_all_features(nc_paths)
    export_xml(tc_tracks, ar_tracks)

    print("\nDone.")


if __name__ == "__main__":
    main()