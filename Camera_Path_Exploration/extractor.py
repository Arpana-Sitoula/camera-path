import os
import math
import numpy as np
import xarray as xr
from scipy import ndimage


# ── Configuration Constants ───────────────────────────────────────────────────

MIN_TC_PIXELS = 50       # Minimum connected pixels to qualify as a Tropical Cyclone
MIN_AR_PIXELS = 200      # Minimum connected pixels to qualify as an Atmospheric River
MAX_SEARCH_KM = 4000     # Maximum Haversine distance to link a blob to an existing track
DEFAULT_THRESHOLD = 0.5  # Segmentation mask probability threshold


# ── Geographic Distance & Coordinates ─────────────────────────────────────────

def haversine_km(lon1, lat1, lon2, lat2):
    """
    Compute great-circle distance in kilometers between two lon/lat coordinates.
    """
    r_lon1, r_lat1, r_lon2, r_lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = r_lon2 - r_lon1
    dlat = r_lat2 - r_lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(r_lat1) * math.cos(r_lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2.0 * math.asin(math.sqrt(a))


# ── Blob Extraction ───────────────────────────────────────────────────────────

def find_blobs(mask, min_size, lons=None, lats=None, threshold=DEFAULT_THRESHOLD):
    """
    Extract connected component blobs from a 2D segmentation probability mask.

    Parameters:
        mask (np.ndarray): 2D array of probabilities (lat x lon).
        min_size (int): Minimum pixel area required to keep a blob.
        lons (np.ndarray, optional): 1D array of longitude coordinates.
        lats (np.ndarray, optional): 1D array of latitude coordinates.
        threshold (float): Binarization threshold.

    Returns:
        list[dict]: List of detected blobs with 'lon', 'lat', 'area', and 'centroid_px'.
    """
    binary = (mask > threshold).astype(int)
    labeled, count = ndimage.label(binary)
    blobs = []

    h, w = mask.shape
    if lons is None:
        lons = np.linspace(-180, 180, w)
    if lats is None:
        lats = np.linspace(-90, 90, h)

    for blob_id in range(1, count + 1):
        blob = (labeled == blob_id)
        size = int(blob.sum())
        if size < min_size:
            continue

        cy, cx = ndimage.center_of_mass(blob)
        col_idx = int(np.clip(round(cx), 0, len(lons) - 1))
        row_idx = int(np.clip(round(cy), 0, len(lats) - 1))

        lon = float(lons[col_idx])
        lat = float(lats[row_idx])

        # Calculate bounding box extent in km
        row_indices, col_indices = np.where(blob)
        min_row, max_row = row_indices.min(), row_indices.max()
        min_col, max_col = col_indices.min(), col_indices.max()
        
        lon_min, lon_max = float(lons[min_col]), float(lons[max_col])
        lat_min, lat_max = float(lats[min_row]), float(lats[max_row])
        
        diag1 = haversine_km(lon_min, lat_min, lon_max, lat_max)
        diag2 = haversine_km(lon_min, lat_max, lon_max, lat_min)
        extent_km = max(diag1, diag2)

        blobs.append({
            "lon": lon,
            "lat": lat,
            "area": size,
            "extent_km": extent_km,
            "centroid_px": (float(cx), float(cy)),
        })

    return blobs


# ── Tracking Core ─────────────────────────────────────────────────────────────

def match_blobs_to_tracks(blobs, prev_blobs, tracks, max_distance_km=MAX_SEARCH_KM):
    """
    Match current timestep blobs to active tracks using greedy nearest-neighbor matching.

    Parameters:
        blobs (list[dict]): Blobs detected at the current timestep.
        prev_blobs (list[tuple]): List of (track_id, lon, lat) active at previous timestep.
        tracks (dict): Mapping of track_id -> list of track point dicts.
        max_distance_km (float): Search radius in km.

    Returns:
        set[int]: Set of blob indices matched to an existing track.
    """
    matched = set()

    for (tid, plon, plat) in prev_blobs:
        best_idx = None
        best_dist = float("inf")

        for i, blob in enumerate(blobs):
            if i in matched:
                continue
            d = haversine_km(plon, plat, blob["lon"], blob["lat"])
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx is not None and best_dist <= max_distance_km:
            matched.add(best_idx)
            tracks[tid].append(blobs[best_idx])

    return matched


def start_new_tracks(blobs, matched, tracks, next_id, timestep, timestamp=None):
    """
    Initialize a new track for each unmatched blob.
    """
    for i, blob in enumerate(blobs):
        if i not in matched:
            point = {**blob, "timestep": timestep}
            if timestamp is not None:
                point["time"] = timestamp
            tracks[next_id] = [point]
            next_id += 1
    return next_id


def get_active_blobs(tracks, timestep):
    """
    Return (track_id, lon, lat) for all tracks alive at the given timestep.
    """
    return [
        (tid, seq[-1]["lon"], seq[-1]["lat"])
        for tid, seq in tracks.items()
        if seq and seq[-1]["timestep"] == timestep
    ]


# ── High-Level Loaders & Tracking Pipelines ───────────────────────────────────

def track_single_file(nc_path, features=("TC", "AR")):
    """
    Track features across all timesteps from a single multi-timestep NetCDF file
    (e.g., AR_TC_result.nc).

    Returns:
        dict: feature_name -> (tracks_dict, all_blobs_by_timestep_dict)
    """
    ds = xr.open_dataset(nc_path)
    lons = ds["lon"].values
    lats = ds["lat"].values
    n_timesteps = ds.sizes["time"]
    timestamps = ds["time"].values if "time" in ds.coords else [None] * n_timesteps

    results = {}
    for feature in features:
        min_size = MIN_TC_PIXELS if feature == "TC" else MIN_AR_PIXELS
        tracks = {}
        next_id = 1
        prev_blobs = []
        all_blobs_by_timestep = {}

        for t in range(n_timesteps):
            # Extract 2D slice, handling potential extra singleton dimensions (e.g. plev)
            mask_slice = ds[feature].isel(time=t)
            while mask_slice.ndim > 2:
                mask_slice = mask_slice.squeeze()
            mask = np.asarray(mask_slice)

            blobs = find_blobs(mask, min_size=min_size, lons=lons, lats=lats)
            for b in blobs:
                b["timestep"] = t
                if timestamps[t] is not None:
                    b["time"] = timestamps[t]

            all_blobs_by_timestep[t] = blobs

            matched = match_blobs_to_tracks(blobs, prev_blobs, tracks)
            next_id = start_new_tracks(blobs, matched, tracks, next_id, t, timestamp=timestamps[t])
            prev_blobs = get_active_blobs(tracks, t)

        results[feature] = (tracks, all_blobs_by_timestep)

    ds.close()
    return results


def track_directory(nc_paths, feature="TC"):
    """
    Track features across multiple per-timestep NetCDF files.

    Returns:
        tuple: (tracks_dict, all_blobs_by_timestep_dict)
    """
    min_size = MIN_TC_PIXELS if feature == "TC" else MIN_AR_PIXELS
    tracks = {}
    next_id = 1
    prev_blobs = []
    all_blobs_by_timestep = {}

    for t, nc_path in enumerate(nc_paths):
        if not os.path.exists(nc_path):
            all_blobs_by_timestep[t] = []
            prev_blobs = []
            continue

        ds = xr.open_dataset(nc_path)
        lons = ds["lon"].values if "lon" in ds.coords else None
        lats = ds["lat"].values if "lat" in ds.coords else None
        mask_slice = ds[feature].isel(time=0) if "time" in ds[feature].dims else ds[feature]
        while mask_slice.ndim > 2:
            mask_slice = mask_slice.squeeze()
        mask = np.asarray(mask_slice)

        blobs = find_blobs(mask, min_size=min_size, lons=lons, lats=lats)
        for b in blobs:
            b["timestep"] = t
        ds.close()

        all_blobs_by_timestep[t] = blobs
        matched = match_blobs_to_tracks(blobs, prev_blobs, tracks)
        next_id = start_new_tracks(blobs, matched, tracks, next_id, t)
        prev_blobs = get_active_blobs(tracks, t)

    return tracks, all_blobs_by_timestep