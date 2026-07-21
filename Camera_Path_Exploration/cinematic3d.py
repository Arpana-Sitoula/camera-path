"""
prototype_animation.py
======================
Generates a cinematic camera path animation prototype from TC/AR tracks.

Inspired by:
- Conlen et al. (2023): Cinematic Techniques in Narrative Visualization
  → establishing shot → feature focus → anthropocentric perspective
- Optimal Camera Path Planning (IEEE 2015)
  → smooth interpolated path between keyframes

Pipeline:
  1. Load TC/AR tracks
  2. Score each feature by size (proxy for visual importance)
  3. Build cinematic shot sequence: overview → TC focus → AR travel
  4. Interpolate smooth camera path between keyframes
  5. Render PNG overview + MP4/GIF animation

Usage:
    python prototype_animation.py
"""

import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, FFMpegWriter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import CubicSpline

from extractor import track_features_across_time


# ── Config ────────────────────────────────────────────────────────────────────

NC_FILES_DIR        = "../Feature_detection/Results/TC-AR-Met3d"
OUTPUT_DIR          = "./Outputs/animation"
TIMESTEPS           = range(0, 12)
FPS                 = 10    # frames per second
FRAMES_PER_KEYFRAME = 20    # interpolation frames between each keyframe
TRAIL_LENGTH        = 40    # how many frames the camera trail stays visible

LONS = np.linspace(-180, 180, 1152)
LATS = np.linspace(-90,   90,  768)

STYLE = {
    "bg":         "#050a14",
    "ocean":      "#0a1628",
    "land":       "#1a2a1a",
    "TC_fill":    "#ff4444",
    "AR_fill":    "#00aaff",
    "path_color": "#ffd700",
    "cam_color":  "#ffffff",
    "text_color": "#e0e0e0",
}


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_masks(timestep):
    """Load TC and AR masks for a given timestep."""
    nc_path = f"{NC_FILES_DIR}/{timestep}.nc"
    if not os.path.exists(nc_path):
        return None, None
    ds      = xr.open_dataset(nc_path)
    tc_mask = np.array(ds["TC"][0])
    ar_mask = np.array(ds["AR"][0])
    ds.close()
    return tc_mask, ar_mask


def get_nc_paths():
    return [f"{NC_FILES_DIR}/{t}.nc" for t in TIMESTEPS]


# ── Feature Importance Scoring ────────────────────────────────────────────────

def score_track(track_seq):
    """
    Score a track by its maximum blob size across all timesteps.
    Larger area = more visually significant = higher camera priority.
    (Simplified proxy for viewpoint entropy — Sakamoto 2025)
    """
    return max(p["area"] for p in track_seq)


def rank_tracks(tracks):
    """Return track IDs sorted by importance score, largest first."""
    scored = [(tid, score_track(seq)) for tid, seq in tracks.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [tid for tid, _ in scored]


# ── Cinematic Shot Sequence ───────────────────────────────────────────────────

def build_shot_sequence(tc_tracks, ar_tracks):
    """
    Build ordered camera keyframes following cinematic narrative structure
    (Conlen et al. 2023):

        establishing → feature focus → travel → release

    Returns list of keyframe dicts: {lon, lat, label, shot_type}
    """
    keyframes = []

    # Shot 1 — Establishing: overview of North Atlantic
    keyframes.append({
        "lon": -30.0, "lat": 40.0,
        "label": "North Atlantic Overview",
        "shot_type": "overview"
    })

    # Shot 2 — Focus: visit top 3 most significant cyclones
    for tid in rank_tracks(tc_tracks)[:3]:
        seq = tc_tracks[tid]
        mid = seq[len(seq) // 2]   # midpoint = most developed stage
        keyframes.append({
            "lon":   mid["lon"],
            "lat":   mid["lat"],
            "label": f"TC {tid}",
            "shot_type": "TC"
        })

    # Shot 3 — Travel: fly along top 2 atmospheric rivers
    for tid in rank_tracks(ar_tracks)[:2]:
        seq = ar_tracks[tid]
        for idx in [0, len(seq) // 2, -1]:   # start, center, end
            p = seq[idx]
            keyframes.append({
                "lon":   p["lon"],
                "lat":   p["lat"],
                "label": f"AR {tid}",
                "shot_type": "AR"
            })

    # Shot 4 — Release: pull back to overview
    keyframes.append({
        "lon": 0.0, "lat": 20.0,
        "label": "Overview", "shot_type": "overview"
    })

    return keyframes


# ── Path Interpolation ────────────────────────────────────────────────────────

def interpolate_path(keyframes, frames_per_segment=FRAMES_PER_KEYFRAME):
    """
    Cubic spline interpolation between keyframes.
    Handles longitude wraparound at ±180 using np.unwrap.
    Returns list of (lon, lat) smooth positions.
    """
    if len(keyframes) < 2:
        return [(k["lon"], k["lat"]) for k in keyframes]

    lons = np.array([k["lon"] for k in keyframes])
    lats = np.array([k["lat"] for k in keyframes])
    t    = np.arange(len(keyframes), dtype=float)

    lons_unwrapped = np.degrees(np.unwrap(np.radians(lons)))

    cs_lon = CubicSpline(t, lons_unwrapped, bc_type="natural")
    cs_lat = CubicSpline(t, lats,           bc_type="natural")

    t_fine  = np.linspace(0, len(keyframes) - 1,
                          len(keyframes) * frames_per_segment)
    lons_sm = ((cs_lon(t_fine) + 180) % 360) - 180
    lats_sm = np.clip(cs_lat(t_fine), -90, 90)

    return list(zip(lons_sm, lats_sm))


def get_active_label(frame_idx, keyframes, frames_per_segment):
    """Return the label of the nearest keyframe for the current frame."""
    kf_idx = min(frame_idx // frames_per_segment, len(keyframes) - 1)
    return keyframes[kf_idx]["label"]


# ── Drawing Helpers ───────────────────────────────────────────────────────────
def setup_axes(fig):
    ax = plt.axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
    # zoom into North Atlantic instead of global view
    ax.set_extent([-80, 20, 0, 70], crs=ccrs.PlateCarree())
    ax.set_facecolor(STYLE["ocean"])
    ax.add_feature(cfeature.LAND,      facecolor=STYLE["land"], zorder=1)
    ax.add_feature(cfeature.COASTLINE, edgecolor="#2a4a2a", linewidth=0.4, zorder=2)
    ax.add_feature(cfeature.BORDERS,   edgecolor="#1a3a1a", linewidth=0.2, zorder=2)
    return ax

def draw_masks(ax, tc_mask, ar_mask):
    if tc_mask is not None:
        ax.contourf(LONS, LATS, np.where(tc_mask > 0.5, 1.0, np.nan),
                    levels=[0.5, 1.5], colors=[STYLE["TC_fill"]],
                    alpha=0.5, transform=ccrs.PlateCarree(), zorder=3)
    if ar_mask is not None:
        ax.contourf(LONS, LATS, np.where(ar_mask > 0.5, 1.0, np.nan),
                    levels=[0.5, 1.5], colors=[STYLE["AR_fill"]],
                    alpha=0.35, transform=ccrs.PlateCarree(), zorder=3)


def draw_planned_path(ax, smooth_path):
    """Full planned path as a faint dashed line."""
    lons = [p[0] for p in smooth_path]
    lats = [p[1] for p in smooth_path]
    ax.plot(lons, lats, color=STYLE["path_color"], linewidth=0.8,
            linestyle="--", alpha=0.2, transform=ccrs.PlateCarree(), zorder=4)


def draw_keypoints(ax, keyframes, numbered=False):
    colors = {"TC": STYLE["TC_fill"], "AR": STYLE["AR_fill"],
              "overview": "#ffffff"}
    for i, kf in enumerate(keyframes):
        c = colors.get(kf["shot_type"], "#ffffff")
        ax.scatter(kf["lon"], kf["lat"], c=c, s=60, zorder=10,
                   transform=ccrs.PlateCarree(),
                   edgecolors="white", linewidths=0.5, alpha=0.7)
        if numbered:
            ax.text(kf["lon"] + 2, kf["lat"] + 2, str(i),
                    color="white", fontsize=8,
                    transform=ccrs.PlateCarree(), zorder=20,
                    path_effects=[pe.withStroke(linewidth=1.5,
                                                foreground="black")])


def draw_camera(ax, lon, lat, trail):
    """Draw camera position and its recent trail."""
    if len(trail) > 1:
        ax.plot([p[0] for p in trail], [p[1] for p in trail],
                color=STYLE["path_color"], linewidth=1.5,
                alpha=0.6, transform=ccrs.PlateCarree(), zorder=8)
    ax.scatter(lon, lat, c=STYLE["cam_color"], s=120, zorder=15,
               transform=ccrs.PlateCarree(),
               edgecolors=STYLE["path_color"], linewidths=1.5)


def draw_hud(ax, frame_idx, total_frames, label):
    """Minimal heads-up display."""
    progress = frame_idx / max(total_frames - 1, 1)

    stroke = [pe.withStroke(linewidth=2, foreground="black")]
    ax.text(0.02, 0.97, f"Camera → {label}",
            transform=ax.transAxes, color=STYLE["text_color"],
            fontsize=11, va="top", fontfamily="monospace",
            path_effects=stroke)
    ax.text(0.02, 0.92, f"Frame {frame_idx + 1}/{total_frames}",
            transform=ax.transAxes, color=STYLE["text_color"],
            fontsize=8, va="top", fontfamily="monospace", alpha=0.6,
            path_effects=stroke)

    ax.plot([0.02, 0.42], [0.03, 0.03],
            transform=ax.transAxes, color="white",
            linewidth=1.5, alpha=0.2)
    ax.plot([0.02, 0.02 + 0.4 * progress], [0.03, 0.03],
            transform=ax.transAxes, color=STYLE["path_color"],
            linewidth=1.5, alpha=0.8)

    ax.legend(handles=[
        mpatches.Patch(color=STYLE["TC_fill"], alpha=0.6,
                       label="Tropical Cyclone"),
        mpatches.Patch(color=STYLE["AR_fill"], alpha=0.5,
                       label="Atmospheric River"),
        plt.Line2D([0], [0], color=STYLE["path_color"],
                   linestyle="--", label="Camera Path"),
    ], loc="lower right", facecolor="#0a0a1a",
       labelcolor=STYLE["text_color"], fontsize=8, framealpha=0.7)


# ── Output ────────────────────────────────────────────────────────────────────

def save_overview_image(keyframes, smooth_path, tc_mask, ar_mask, output_dir):
    """Save a static PNG showing the full planned camera path."""
    fig = plt.figure(figsize=(16, 9), facecolor=STYLE["bg"])
    ax  = setup_axes(fig)
    draw_masks(ax, tc_mask, ar_mask)
    draw_planned_path(ax, smooth_path)
    draw_keypoints(ax, keyframes, numbered=True)
    plt.title("Cinematic Camera Path — Keyframe Overview",
              color=STYLE["text_color"], fontsize=12, pad=8)
    path = os.path.join(output_dir, "camera_path_overview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Overview image → {path}")


def save_animation(keyframes, smooth_path, tc_mask, ar_mask, output_dir):
    """Render and save the animated camera movement."""
    total_frames = len(smooth_path)
    fig = plt.figure(figsize=(16, 9), facecolor=STYLE["bg"])

    def animate(frame_idx):
        fig.clf()
        ax = setup_axes(fig)
        draw_masks(ax, tc_mask, ar_mask)
        draw_planned_path(ax, smooth_path)
        draw_keypoints(ax, keyframes)

        trail_start = max(0, frame_idx - TRAIL_LENGTH)
        trail       = smooth_path[trail_start:frame_idx + 1]
        lon, lat    = smooth_path[frame_idx]
        draw_camera(ax, lon, lat, trail)

        label = get_active_label(frame_idx, keyframes, FRAMES_PER_KEYFRAME)
        draw_hud(ax, frame_idx, total_frames, label)

        if frame_idx % 20 == 0:
            print(f"  frame {frame_idx}/{total_frames}")

    anim = FuncAnimation(fig, animate, frames=total_frames,
                         interval=1000 // FPS)

    try:
        writer = FFMpegWriter(fps=FPS, bitrate=1800)
        path   = os.path.join(output_dir, "camera_animation.mp4")
        anim.save(path, writer=writer, dpi=120,
                  savefig_kwargs={"facecolor": STYLE["bg"]})
        print(f"  Video → {path}")
    except Exception as e:
        print(f"  ffmpeg not found ({e}), saving as GIF...")
        path = os.path.join(output_dir, "camera_animation.gif")
        anim.save(path, writer="pillow", fps=FPS,
                  savefig_kwargs={"facecolor": STYLE["bg"]})
        print(f"  GIF → {path}")

    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading tracks...")
    nc_paths  = get_nc_paths()
    tc_tracks = track_features_across_time(nc_paths, feature="TC")
    ar_tracks = track_features_across_time(nc_paths, feature="AR")
    print(f"  TC tracks: {len(tc_tracks)}  AR tracks: {len(ar_tracks)}")

    if not tc_tracks and not ar_tracks:
        print("No tracks found — check NC_FILES_DIR path.")
        return

    print("\nBuilding cinematic shot sequence...")
    keyframes   = build_shot_sequence(tc_tracks, ar_tracks)
    smooth_path = interpolate_path(keyframes)
    print(f"  {len(keyframes)} keyframes → {len(smooth_path)} smooth frames")
    for kf in keyframes:
        print(f"  [{kf['shot_type']:8}] {kf['label']:10}"
              f"  lon={kf['lon']:7.1f}  lat={kf['lat']:6.1f}")

    print("\nLoading background masks (timestep 0)...")
    tc_mask, ar_mask = load_masks(0)

    print("\nSaving overview image...")
    save_overview_image(keyframes, smooth_path, tc_mask, ar_mask, OUTPUT_DIR)

    print("\nRendering animation...")
    save_animation(keyframes, smooth_path, tc_mask, ar_mask, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()