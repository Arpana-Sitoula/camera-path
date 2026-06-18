import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import os

LONS = np.linspace(-180, 180, 1152)
LATS = np.linspace(-90,   90,  768)

# ── Styless ─────────────────────────────────────────────────────────────────────

STYLE = {
    "background":  "#0a0a1a",
    "TC_color":    "#ff6b6b",
    "AR_color":    "#4ecdc4",
    "path_color":  "yellow",
    "face_color": "#1a1a2e",
    "label_color": "white",
    "TC_marker":   {"color": "#ff4444", "marker": "*", "s": 200},
    "AR_marker":   {"color": "#00ffcc", "marker": "o", "s":  80},
    "OV_marker":   {"color": "#ffffff", "marker": "D", "s": 100},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _draw_masks(ax, tc_mask, ar_mask):
    ax.contourf(LONS, LATS, np.where(tc_mask > 0.5, 1.0, np.nan),
                levels=[0.5, 1.5], colors=[STYLE["TC_color"]],
                alpha=0.6, transform=ccrs.PlateCarree())

    ax.contourf(LONS, LATS, np.where(ar_mask > 0.5, 1.0, np.nan),
                levels=[0.5, 1.5], colors=[STYLE["AR_color"]],
                alpha=0.4, transform=ccrs.PlateCarree())


def _draw_path(ax, camera_path):
    lons = [p["lon"] for p in camera_path]
    lats = [p["lat"] for p in camera_path]
    ax.plot(lons, lats, color=STYLE["path_color"], linewidth=1.5,
            linestyle="--", alpha=0.8, transform=ccrs.PlateCarree(), zorder=10)


def _draw_keypoints(ax, camera_path):
    # AR_start / AR_center / AR_end all use the same AR marker style
    def get_style(ptype):
        if ptype == "TC":
            return STYLE["TC_marker"]
        elif ptype in ("AR_start", "AR_center", "AR_end"):
            return STYLE["AR_marker"]
        else:
            return STYLE["OV_marker"]

    for i, point in enumerate(camera_path):
        s = get_style(point["type"])
        ax.scatter(point["lon"], point["lat"], zorder=20,
                   transform=ccrs.PlateCarree(), **s)
        ax.text(point["lon"] + 2, point["lat"] + 2, str(i),
                color="white", fontsize=7,
                transform=ccrs.PlateCarree(), zorder=25)


def _draw_legend(ax):
    elements = [
        mpatches.Patch(color=STYLE["TC_color"], alpha=0.6, label="Tropical Cyclone"),
        mpatches.Patch(color=STYLE["AR_color"], alpha=0.4, label="Atmospheric River"),
        plt.Line2D([0], [0], color=STYLE["path_color"],
                   linestyle="--", label="Camera Path"),
    ]
    ax.legend(handles=elements, loc="lower left",
              facecolor=STYLE["face_color"], labelcolor=STYLE["label_color"], fontsize=8)

#--Pipeline---------------------------------------------------------------------------------
def save_camera_map(nc_path, camera_path, timestep, output_dir):
    """
    Renders the TC/AR masks with the extracted camera path
    and saves it as a PNG.
    """
    ds      = xr.open_dataset(nc_path)
    tc_mask = np.array(ds["TC"][0])
    ar_mask = np.array(ds["AR"][0])
    ds.close()

    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor(STYLE["background"])

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.set_facecolor(STYLE["background"])
    ax.coastlines(resolution="110m", color="white", linewidth=0.5)

    _draw_masks(ax, tc_mask, ar_mask)
    _draw_path(ax, camera_path)
    _draw_keypoints(ax, camera_path)
    _draw_legend(ax)

    plt.title(f"Camera Path — Timestep {timestep}",
              color="white", fontsize=13, pad=10)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"camera_path_{timestep:02d}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    print(f"Map saved : {out_path}")