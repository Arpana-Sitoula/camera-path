from extractor import track_features_across_time
from ranking import rank_tracks
from visualization import plot_ranking, animate_top_tracks, animate_camera_view
from exporter import export_single_track_xml
import os


# ── Config ────────────────────────────────────────────────────────────────────

NC_FILES_DIR = "../Feature_detection/Results/TC-AR-Met3d"
OUTPUT_DIR   = "./Outputs"
TIMESTEPS    = range(0, 12)
Z_TC         = 30     # zoom level for tropical cyclones (closer view)
Z_AR         = 60     # zoom level for atmospheric rivers (wider view)
TOP_N        = 2      # how many top-ranked features to actually export/animate


# ── Steps ─────────────────────────────────────────────────────────────────────

def get_nc_paths():
    """Build list of all .nc file paths for every timestep."""
    return [f"{NC_FILES_DIR}/{t}.nc" for t in TIMESTEPS]


def track_all_features(nc_paths):
    """
    Track TC and AR blobs across all timesteps.
    Returns four things: tc_tracks, ar_tracks, tc_blobs, ar_blobs
    (the *_blobs dicts hold every detected blob per timestep, needed by
    ranking.py to detect nearby splits/merges — see extractor.py's
    return_all_blobs flag).
    """
    print("\nTracking Tropical Cyclones...")
    tc_tracks, tc_blobs = track_features_across_time(nc_paths, feature="TC", return_all_blobs=True)
    print(f"  → {len(tc_tracks)} TC tracks found")

    print("\nTracking Atmospheric Rivers...")
    ar_tracks, ar_blobs = track_features_across_time(nc_paths, feature="AR", return_all_blobs=True)
    print(f"  → {len(ar_tracks)} AR tracks found")

    return tc_tracks, ar_tracks, tc_blobs, ar_blobs


def rank_all_tracks(tc_tracks, ar_tracks, tc_blobs, ar_blobs):
    """
    Score every TC/AR track on speed, lifetime, area growth, and
    split/merge activity. Returns (all_rows_sorted, top_rows).
    """
    print("\nRanking tracks...")
    rows, top = rank_tracks(tc_tracks, ar_tracks, tc_blobs, ar_blobs, top_n=TOP_N)
    for r in top:
        print(f"  #{r['rank']} {r['feature']}-{r['id']}  score={r['score']:.3f}")
    return rows, top


from visualization import plot_ranking, animate_top_tracks_with_shapes, animate_camera_view_with_shape

def generate_visuals(rows, top, nc_paths):
    print("\nGenerating ranking overview PNG...")
    plot_ranking(rows, top_n=5, output_path=f"{OUTPUT_DIR}/ranking_overview.png")

    print("Generating movement GIF (real shapes) for top features...")
    animate_top_tracks_with_shapes(top, nc_paths, output_path=f"{OUTPUT_DIR}/top_tracks.gif")

    for r in top:
        z = Z_TC if r["feature"] == "TC" else Z_AR
        print(f"Generating camera-view GIF (real shapes) for #{r['rank']} {r['feature']}-{r['id']}...")
        animate_camera_view_with_shape(
            r["seq"], r["feature"], z, nc_paths,
            output_path=f"{OUTPUT_DIR}/camera_view_{r['feature']}_{r['id']}.gif"
        )


def export_top_xml(top):
    """Export one clean, single-feature Met3D XML per top-ranked track."""
    print("\nExporting top-ranked tracks to Met3D XML...")
    for r in top:
        z = Z_TC if r["feature"] == "TC" else Z_AR
        out_path = f"{OUTPUT_DIR}/xml/camera_sequence_{r['feature']}_{r['id']}.xml"
        os.makedirs(f"{OUTPUT_DIR}/xml", exist_ok=True)
        export_single_track_xml(r["seq"], z, out_path)
        print(f"  → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    nc_paths = get_nc_paths()
    tc_tracks, ar_tracks, tc_blobs, ar_blobs = track_all_features(nc_paths)
    rows, top = rank_all_tracks(tc_tracks, ar_tracks, tc_blobs, ar_blobs)
    generate_visuals(rows, top, nc_paths)
    export_top_xml(top)

    print("\nDone.")


if __name__ == "__main__":
    main()