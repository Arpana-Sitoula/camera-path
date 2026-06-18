import numpy as np
import xarray as xr
from scipy import ndimage
from skimage.morphology import skeletonize


# ── Config ────────────────────────────────────────────────────────────────────
# smaller than this are treated as noise
MIN_TC_PIXELS = 50    
MIN_AR_PIXELS = 200   

# ── Grid ──────────────────────────────────────────────────────────────────────

LONS = np.linspace(-180, 180, 1152)
LATS = np.linspace(-90,   90,  768)

def pixel_to_lonlat(px, py):
    lon = float(LONS[int(np.clip(px, 0, 1151))])
    lat = float(LATS[int(np.clip(py, 0,  767))])
    return lon, lat


# ── TC Extraction ─────────────────────────────────────────────────────────────

def extract_tc_points(tc_mask):
    """
    Finds one keypoint per TC blob — the centroid.
    Sorted largest → smallest 
    """
    binary         = (tc_mask > 0.5).astype(int)
    labeled, count = ndimage.label(binary)

    points = []
    for blob_id in range(1, count + 1):
        blob = (labeled == blob_id)
        size = int(blob.sum())

        if size < MIN_TC_PIXELS:
            continue

        cy, cx   = ndimage.center_of_mass(blob)
        lon, lat = pixel_to_lonlat(cx, cy)
        points.append({"type": "TC", "lon": lon, "lat": lat, "size": size})

    points.sort(key=lambda p: p["size"], reverse=True)
    return points


# ── AR Extraction ─────────────────────────────────────────────────────────────

def _spine_points_for_blob(blob_mask):
    """
    Extract exactly 3 points: start, center, end (west → east).
    """
    spine  = skeletonize(blob_mask)
    pixels = np.argwhere(spine)   # (row=y, col=x)

    if len(pixels) < 3:
        return []

    # order west → east along longitude axis
    pixels = pixels[np.argsort(pixels[:, 1])]

    indices = [0, len(pixels) // 2, len(pixels) - 1]
    labels  = ["AR_start", "AR_center", "AR_end"]

    points = []
    for label, idx in zip(labels, indices):
        py, px   = pixels[idx]
        lon, lat = pixel_to_lonlat(px, py)
        points.append({
            "type":  label,
            "lon":   lon,
            "lat":   lat,
            "size":  int(blob_mask.sum())
        })

    return points


def extract_ar_points(ar_mask):
    """
    Finds each separate AR blob, extracts 3 spine points per blob
    (start, center, end), and returns all of them as a flat list.
    """
    binary         = (ar_mask > 0.5).astype(bool)
    labeled, count = ndimage.label(binary)

    all_points = []
    for blob_id in range(1, count + 1):
        blob = (labeled == blob_id)
        size = int(blob.sum())

        if size < MIN_AR_PIXELS:
            continue

        points = _spine_points_for_blob(blob)
        all_points.extend(points)

    return all_points


# ── Nearest Neighbour Ordering ────────────────────────────────────────────────

def _geo_distance(a, b):
    """Simple flat distance in lon/lat degrees"""
    return ((a["lon"] - b["lon"]) ** 2 + (a["lat"] - b["lat"]) ** 2) ** 0.5


def order_by_proximity(points):
    """
    Greedy nearest neighbour ordering.
    """
    if len(points) <= 1:
        return points

    # start from the westernmost point (leftmost on map)
    remaining = sorted(points, key=lambda p: p["lon"])
    ordered   = [remaining.pop(0)]

    while remaining:
        current   = ordered[-1]
        nearest   = min(remaining, key=lambda p: _geo_distance(current, p))
        ordered.append(nearest)
        remaining.remove(nearest)

    return ordered

# --Pipeline-----------------------------------------------------------------------------
def extract_keyframes(nc_path):
    """
    Reads a .nc file and returns an ordered camera path as a list of
    keyframe dicts, each with type, lon, and lat.
    """
    ds      = xr.open_dataset(nc_path)
    tc_mask = np.array(ds["TC"][0])
    ar_mask = np.array(ds["AR"][0])
    ds.close()

    tc_points = extract_tc_points(tc_mask)
    ar_points = extract_ar_points(ar_mask)
    all_points = tc_points + ar_points

    if not all_points:
        print("         No features detected.")
        return []

    ordered = order_by_proximity(all_points)

    # print summary
    tc_count = len(tc_points)
    ar_count = len([p for p in ar_points if p["type"] == "AR_start"])
    print(f"         TCs found : {tc_count}")
    for p in tc_points:
        print(f"           → lon={p['lon']:7.1f}  lat={p['lat']:6.1f}  size={p['size']} px")
    print(f"         ARs found : {ar_count}  ({len(ar_points)} spine points total)")

    return ordered