import os
import math

import numpy as np
import xarray as xr
from scipy import ndimage


# ── Config ────────────────────────────────────────────────────────────────────

MIN_TC_PIXELS = 50      # blobs smaller than this ignored as noise
MIN_AR_PIXELS = 200     # same for AR
MAX_SEARCH_KM = 4000    # max distance to match same feature across timesteps

LONS = np.linspace(-180, 180, 1152)
LATS = np.linspace(-90,   90,  768)



# ── Coordinate Conversion ─────────────────────────────────────────────────────

def pixel_to_lonlat(px, py):
    """Convert pixel column/row to real-world lon/lat."""
    lon = float(LONS[int(np.clip(px, 0, 1151))])
    lat = float(LATS[int(np.clip(py, 0,  767))])
    return lon, lat


# ── Distance ──────────────────────────────────────────────────────────────────

def haversine_km(lon1, lat1, lon2, lat2):
    """Real geographic distance in km between two lon/lat points."""
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a    = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return 6371 * 2 * math.asin(math.sqrt(a))


# ── Blob Detection ────────────────────────────────────────────────────────────

def find_blobs(mask, min_size):
    binary         = (mask > 0.5).astype(int)
    labeled, count = ndimage.label(binary)
    blobs          = []

    for blob_id in range(1, count + 1):
        blob = (labeled == blob_id)
        size = int(blob.sum())

        if size < min_size:
            continue

        cy, cx   = ndimage.center_of_mass(blob)
        lon, lat = pixel_to_lonlat(cx, cy)

        blobs.append({"lon": lon, "lat": lat, "area": size})

    return blobs


def load_mask(nc_path, feature):
    """Load a single TC or AR mask from a .nc file."""
    ds   = xr.open_dataset(nc_path)
    mask = np.array(ds[feature][0])
    ds.close()
    return mask


# ── Tracking ──────────────────────────────────────────────────────────────────

def match_blobs_to_tracks(blobs, prev_blobs, tracks):
    """
    Match current timestep blobs to existing tracks using nearest neighbour.
    Returns set of matched blob indices and updates tracks in place.
    """
    matched = set()

    for (tid, plon, plat) in prev_blobs:
        best_idx  = None
        best_dist = float("inf")

        for i, blob in enumerate(blobs):
            if i in matched:
                continue
            d = haversine_km(plon, plat, blob["lon"], blob["lat"])
            if d < best_dist:
                best_dist = d
                best_idx  = i

        if best_idx is not None and best_dist <= MAX_SEARCH_KM:
            matched.add(best_idx)
            tracks[tid].append(blobs[best_idx])

    return matched


def start_new_tracks(blobs, matched, tracks, next_id, timestep):
    """Start a new track for every unmatched blob."""
    for i, blob in enumerate(blobs):
        if i not in matched:
            tracks[next_id] = [{**blob, "timestep": timestep}]
            next_id += 1
    return next_id


def get_active_blobs(tracks, timestep):
    """Return (track_id, lon, lat) for all tracks active at this timestep."""
    return [
        (tid, seq[-1]["lon"], seq[-1]["lat"])
        for tid, seq in tracks.items()
        if seq and seq[-1]["timestep"] == timestep
    ]


# ── Public API ────────────────────────────────────────────────────────────────

def track_features_across_time(nc_paths, feature="TC", return_all_blobs=False):
    """
    Track one feature type (TC or AR) across all timesteps.

    For each timestep:
      1. Find all blobs in the mask
      2. Match blobs to existing tracks by proximity
      3. Unmatched blobs start new tracks

    Returns:
        dict: track_id -> [{lon, lat, area, timestep}, ...]
        (and, if return_all_blobs=True, also a second dict:
         timestep -> [every blob detected at that timestep], needed later
         for split/merge detection in ranking.py)
    """
    min_size   = MIN_TC_PIXELS if feature == "TC" else MIN_AR_PIXELS
    tracks     = {}
    next_id    = 1
    prev_blobs = []
    all_blobs_by_timestep = {}

    for t, nc_path in enumerate(nc_paths):

        if not os.path.exists(nc_path):
            print(f"  [timestep {t}] File not found, skipping.")
            prev_blobs = []
            all_blobs_by_timestep[t] = []
            continue

        mask  = load_mask(nc_path, feature)
        blobs = find_blobs(mask, min_size)

        for blob in blobs:
            blob["timestep"] = t

        all_blobs_by_timestep[t] = blobs

        matched = match_blobs_to_tracks(blobs, prev_blobs, tracks)
        next_id = start_new_tracks(blobs, matched, tracks, next_id, t)
        prev_blobs = get_active_blobs(tracks, t)

        print(f"  [timestep {t}] {feature} blobs: {len(blobs)}  active tracks: {len(prev_blobs)}")

    if return_all_blobs:
        return tracks, all_blobs_by_timestep
    return tracks