"""
Feature Extraction and Temporal Tracking Engine
================================================
This module handles:
1. Blob Detection: Identifying connected component features (Tropical Cyclones and Atmospheric Rivers)
   from 2D segmentation probability masks.
2. Spatial Measurements: Computing centroid coordinates (lon/lat), pixel area, and physical bounding extent (km).
3. Temporal Tracking: Associating blobs across consecutive timesteps using nearest-neighbor Haversine distance
   to construct continuous trajectories over time.
"""

import os
import math
import numpy as np
import xarray as xr
from scipy import ndimage


# ==============================================================================
# 1. CONFIGURATION PARAMETERS
# ==============================================================================
# Minimum connected pixel count to qualify as a valid feature (filters out small noise):
MIN_TC_PIXELS = 50       # Tropical Cyclones: smaller, concentrated vortex features
MIN_AR_PIXELS = 200      # Atmospheric Rivers: large, elongated plume structures

# Maximum distance (in km) to link a blob in timestep (t) to a track from timestep (t-1):
# Weather systems can travel at high speeds; 4000 km covers rapid movement across a 3h-6h window.
MAX_SEARCH_KM = 4000.0

# Threshold for binarizing neural network probability masks (0.0 to 1.0):
DEFAULT_THRESHOLD = 0.5

# Mean Earth radius in kilometers for Haversine calculations:
EARTH_RADIUS_KM = 6371.0


# ==============================================================================
# 2. GEOGRAPHIC & SPATIAL CALCULATIONS
# ==============================================================================

def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the Great-Circle distance between two points on Earth in kilometers.

    Formula:
        a = sin²(Δlat / 2) + cos(lat1) * cos(lat2) * sin²(Δlon / 2)
        c = 2 * asin(√a)
        distance = R * c
    """
    r_lon1, r_lat1, r_lon2, r_lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = r_lon2 - r_lon1
    dlat = r_lat2 - r_lat1

    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(r_lat1) * math.cos(r_lat2) * math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, a))))
    return EARTH_RADIUS_KM * c


def calculate_blob_extent_km(lons: np.ndarray, lats: np.ndarray,
                             min_col: int, max_col: int,
                             min_row: int, max_row: int) -> float:
    """
    Compute the diagonal diameter in km of a blob.
    """
    lon_min, lon_max = float(lons[min_col]), float(lons[max_col])
    lat_min, lat_max = float(lats[min_row]), float(lats[max_row])

    # Compute both diagonals
    diag1 = haversine_km(lon_min, lat_min, lon_max, lat_max)
    diag2 = haversine_km(lon_min, lat_max, lon_max, lat_min)
    return max(diag1, diag2)


# ==============================================================================
# 3. BLOB DETECTION & SPATIAL MEASUREMENT
# ==============================================================================

def find_blobs(mask: np.ndarray, min_size: int,
               lons: np.ndarray = None, lats: np.ndarray = None,
               threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    """
    Extract individual connected component blobs from a 2D segmentation probability mask.

    Steps:
    1. Binarize mask using probability threshold (> threshold -> 1, else 0).
    2. Label connected 8-neighborhood regions using scipy.ndimage.label.
    3. Filter out regions with pixel count < min_size.
    4. Compute center of mass (centroid in px and lon/lat) and bounding box extent (km).

    Parameters:
        mask (np.ndarray): 2D array of feature probabilities [lat x lon].
        min_size (int): Minimum pixel area required to keep a blob.
        lons (np.ndarray, optional): 1D array of longitude coordinates.
        lats (np.ndarray, optional): 1D array of latitude coordinates.
        threshold (float): Binarization probability cutoff.
    """
    binary = (mask > threshold).astype(np.int32)
    labeled, count = ndimage.label(binary)
    blobs = []

    h, w = mask.shape
    if lons is None:
        lons = np.linspace(-180.0, 180.0, w)
    if lats is None:
        lats = np.linspace(-90.0, 90.0, h)

    for blob_id in range(1, count + 1):
        blob_mask = (labeled == blob_id)
        pixel_count = int(blob_mask.sum())

        if pixel_count < min_size:
            continue

        # Compute centroid using center of mass
        cy, cx = ndimage.center_of_mass(blob_mask)
        col_idx = int(np.clip(round(cx), 0, len(lons) - 1))
        row_idx = int(np.clip(round(cy), 0, len(lats) - 1))

        lon = float(lons[col_idx])
        lat = float(lats[row_idx])

        # Compute bounding box and physical extent
        row_indices, col_indices = np.where(blob_mask)
        min_row, max_row = int(row_indices.min()), int(row_indices.max())
        min_col, max_col = int(col_indices.min()), int(col_indices.max())

        extent_km = calculate_blob_extent_km(lons, lats, min_col, max_col, min_row, max_row)
        bbox = [float(lons[min_col]), float(lons[max_col]), float(lats[min_row]), float(lats[max_row])]

        blobs.append({
            "lon": lon,
            "lat": lat,
            "area": pixel_count,
            "extent_km": extent_km,
            "centroid_px": (float(cx), float(cy)),
            "bbox": bbox,
        })

    return blobs


# ==============================================================================
# 4. TEMPORAL TRACKING ENGINE
# ==============================================================================

def match_blobs_to_tracks(current_blobs: list[dict],
                          prev_active_blobs: list[tuple],
                          tracks: dict,
                          max_distance_km: float = MAX_SEARCH_KM) -> set[int]:
    """
    Match blobs at current timestep (t) to active tracks from timestep (t-1)
    using greedy nearest-neighbor matching based on Haversine distance.

    Parameters:
        current_blobs (list[dict]): Blobs detected at timestep t.
        prev_active_blobs (list[tuple]): List of (track_id, prev_lon, prev_lat) active at t-1.
        tracks (dict): Mapping of track_id -> list of track points.
        max_distance_km (float): Search radius threshold.
    """
    matched_indices = set()

    for (track_id, prev_lon, prev_lat) in prev_active_blobs:
        best_idx = None
        best_distance = float("inf")

        for idx, blob in enumerate(current_blobs):
            if idx in matched_indices:
                continue

            dist = haversine_km(prev_lon, prev_lat, blob["lon"], blob["lat"])
            if dist < best_distance:
                best_distance = dist
                best_idx = idx

        if best_idx is not None and best_distance <= max_distance_km:
            matched_indices.add(best_idx)
            tracks[track_id].append(current_blobs[best_idx])

    return matched_indices


def start_new_tracks(current_blobs: list[dict],
                     matched_indices: set[int],
                     tracks: dict,
                     next_id: int,
                     timestep: int,
                     timestamp=None) -> int:
    """
    Initialize a new track sequence for every unmatched blob detected at current timestep.
    """
    for idx, blob in enumerate(current_blobs):
        if idx not in matched_indices:
            point = {**blob, "timestep": timestep}
            if timestamp is not None:
                point["time"] = timestamp
            tracks[next_id] = [point]
            next_id += 1
    return next_id


def get_active_blobs(tracks: dict, current_timestep: int) -> list[tuple]:
    """
    Return active tracks at the current timestep as (track_id, lon, lat).
    """
    active = []
    for tid, seq in tracks.items():
        if seq and seq[-1]["timestep"] == current_timestep:
            active.append((tid, seq[-1]["lon"], seq[-1]["lat"]))
    return active


# ==============================================================================
# 5. HIGH-LEVEL DATASET LOADERS & PIPELINE
# ==============================================================================

def track_single_file(nc_path: str, features: tuple = ("TC", "AR")) -> dict:
    """
    Track meteorological features across all timesteps in a single NetCDF file (e.g. AR_TC_result.nc).

    Returns:
        dict: {feature_name: (tracks_dict, all_blobs_by_timestep_dict)}
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

            matched = match_blobs_to_tracks(blobs, prev_blobs, tracks, max_distance_km=MAX_SEARCH_KM)
            next_id = start_new_tracks(blobs, matched, tracks, next_id, t, timestamp=timestamps[t])
            prev_blobs = get_active_blobs(tracks, t)

        results[feature] = (tracks, all_blobs_by_timestep)

    ds.close()
    return results


def track_directory(nc_paths: list[str], feature: str = "TC") -> tuple:
    """
    Track features across multiple per-timestep NetCDF files in a folder.

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
        matched = match_blobs_to_tracks(blobs, prev_blobs, tracks, max_distance_km=MAX_SEARCH_KM)
        next_id = start_new_tracks(blobs, matched, tracks, next_id, t)
        prev_blobs = get_active_blobs(tracks, t)

    return tracks, all_blobs_by_timestep