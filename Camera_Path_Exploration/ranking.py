"""
Meteorological Feature Ranking & Multi-Criteria Prioritization Engine
=====================================================================
This module calculates multi-dimensional metrics for weather tracks, normalizes them
using statistical Z-scores, and ranks them to select optimal camera motion paths for Met3D.

Core Concepts:
--------------
1. "Longest" Track Metrics:
   - Temporal Lifetime: Number of timesteps the system persisted (frames).
   - Trajectory Distance: Cumulative Great-Circle Haversine distance (km) traveled by centroid.
     -> High distance = Ideal for "Follow" camera shots.

2. "Big Area" & Spatial Extent Metrics:
   - Peak Pixel Area (max_area_px): Maximum number of connected pixels across all timesteps.
   - Mean Pixel Area (mean_area_px): Average pixel area over the feature's lifetime.
   - Maximum Extent (max_extent_km): Largest bounding box diagonal diameter (km).
     -> High extent/area = Ideal for wide "Spline" overview camera shots.

3. Shape Volatility & Interactions:
   - Average Area Delta: Mean frame-to-frame change in pixel area (|area_{t+1} - area_t|).
   - Split/Merge Events: Count of proximity interactions with neighboring blobs.
     -> High volatility = Ideal for dynamic "Spin" orbit camera shots.

4. Statistical Z-Score Normalization:
   - Converts raw metrics of different units (km, pixels, event counts) into a unified
     dimensionless scale: Z = (x - mean) / std_dev.
   - Identifies how exceptionally a track performs in each specific camera motion domain.
"""

import numpy as np
from extractor import haversine_km, MAX_SEARCH_KM


# ==============================================================================
# 1. CONFIGURATION PARAMETERS
# ==============================================================================

# Minimum number of consecutive timesteps required to consider a track valid (filters transient noise):
DEFAULT_MIN_LIFETIME = 5

# Default number of top-priority features to select for camera animation export:
DEFAULT_TOP_N = 3

# Spatial search radius (in km) to detect neighboring blobs for split/merge interactions:
SPLIT_MERGE_RADIUS_KM = MAX_SEARCH_KM

# Scaling weight added to shape volatility per split/merge event (balances event count with pixel deltas):
SPLIT_MERGE_WEIGHT = 500.0


# ==============================================================================
# 2. MODULAR METRIC CALCULATORS
# ==============================================================================

def calculate_temporal_metrics(seq: list[dict]) -> dict:
    """
    Calculate time-based persistence metrics for a track.
    """
    if not seq:
        return {"lifetime": 0, "start_step": 0, "end_step": 0}

    return {
        "lifetime": len(seq),
        "start_step": seq[0].get("timestep", 0),
        "end_step": seq[-1].get("timestep", 0),
    }


def calculate_trajectory_distance(seq: list[dict]) -> dict:
    """
    Calculate trajectory travel distance ("Longest Path") along the storm's path.

    Cumulative sum of Great-Circle Haversine distances between consecutive centroids:
        total_distance = sum(haversine(point[t], point[t+1])) for t in range(len(seq)-1)
    """
    if len(seq) < 2:
        return {"total_distance_km": 0.0, "avg_speed_km_per_step": 0.0}

    total_dist = 0.0
    for p_prev, p_curr in zip(seq[:-1], seq[1:]):
        dist = haversine_km(p_prev["lon"], p_prev["lat"], p_curr["lon"], p_curr["lat"])
        total_dist += dist

    num_steps = len(seq) - 1
    avg_speed = total_dist / max(1, num_steps)

    return {
        "total_distance_km": total_dist,
        "avg_speed_km_per_step": avg_speed,
    }


def calculate_area_and_extent(seq: list[dict]) -> dict:
    """
    Calculate spatial size ("Big Area") and physical footprint extent for a track.

    Calculations:
    - max_area_px: Peak pixel area at the storm's largest moment.
    - mean_area_px: Average pixel area across all active timesteps.
    - max_extent_km: Peak bounding box diagonal span (km) across all timesteps.
    - mean_extent_km: Average bounding box diagonal span (km).
    """
    if not seq:
        return {
            "max_area_px": 0,
            "mean_area_px": 0.0,
            "max_extent_km": 0.0,
            "mean_extent_km": 0.0,
        }

    areas = [p.get("area", 0) for p in seq]
    extents = [p.get("extent_km", 0.0) for p in seq]

    return {
        "max_area_px": int(max(areas)) if areas else 0,
        "mean_area_px": float(np.mean(areas)) if areas else 0.0,
        "max_extent_km": float(max(extents)) if extents else 0.0,
        "mean_extent_km": float(np.mean(extents)) if extents else 0.0,
    }


def calculate_shape_volatility(seq: list[dict],
                                       all_blobs_by_timestep: dict = None,
                                       radius_km: float = SPLIT_MERGE_RADIUS_KM,
                                       sm_weight: float = SPLIT_MERGE_WEIGHT) -> dict:
    """
    Calculate shape volatility: rate of area change + split/merge interactions.

    1. Area Volatility: Average frame-to-frame absolute change in pixel area:
           avg_area_change = mean(|area_{t+1} - area_t|)
    2. Split/Merge Events: Count of timesteps where another distinct blob was within radius_km.
    3. Composite Volatility: avg_area_change + (split_merge_count * sm_weight).
    """
    if len(seq) < 2:
        return {
            "avg_area_change": 0.0,
            "split_merge_events": 0,
            "composite_volatility": 0.0,
        }

    # Frame-to-frame pixel area change
    area_changes = [abs(curr["area"] - prev["area"]) for prev, curr in zip(seq[:-1], seq[1:])]
    avg_area_change = float(np.mean(area_changes)) if area_changes else 0.0

    # Split / Merge interaction detection
    split_merge_count = 0
    if all_blobs_by_timestep is not None:
        for p in seq:
            timestep = p.get("timestep", 0)
            other_blobs = all_blobs_by_timestep.get(timestep, [])
            nearby = sum(
                1 for b in other_blobs
                if b is not p and 0 < haversine_km(p["lon"], p["lat"], b["lon"], b["lat"]) <= radius_km
            )
            if nearby > 0:
                split_merge_count += 1

    composite_volatility = avg_area_change + (split_merge_count * sm_weight)

    return {
        "avg_area_change": avg_area_change,
        "split_merge_events": split_merge_count,
        "composite_volatility": composite_volatility,
    }


def compute_track_metrics(seq: list[dict], all_blobs_by_timestep: dict = None) -> dict:
    """
    Compute a consolidated dictionary of all metrics for a single track.
    Combines Temporal, Trajectory Distance, Area/Extent, and Volatility measurements.
    """
    temporal = calculate_temporal_metrics(seq)
    dist = calculate_trajectory_distance(seq)
    area_ext = calculate_area_and_extent(seq)
    volatility = calculate_shape_volatility(seq, all_blobs_by_timestep)

    return {
        # Temporal
        "lifetime": temporal["lifetime"],
        "start_step": temporal["start_step"],
        "end_step": temporal["end_step"],

        # Trajectory Distance (Longest Path)
        "total_distance_traveled": dist["total_distance_km"],
        "avg_speed_km_per_step": dist["avg_speed_km_per_step"],

        # Area & Physical Extent (Big Area)
        "max_area_px": area_ext["max_area_px"],
        "mean_area_px": area_ext["mean_area_px"],
        "max_extent_km": area_ext["max_extent_km"],
        "mean_extent_km": area_ext["mean_extent_km"],

        # Morphological Volatility
        "avg_area_change": volatility["avg_area_change"],
        "split_merge": volatility["split_merge_events"],
        "shape_volatility": volatility["composite_volatility"],
    }


# ==============================================================================
# 3. STANDARDIZED STATISTICAL NORMALIZATION (Z-SCORES)
# ==============================================================================

def compute_z_scores(values: list[float]) -> list[float]:
    """
    Calculate standard scores (Z-scores) for an array of metric values:
        Z = (x - mean) / std_dev
    """
    if not values:
        return []

    mean_val = float(np.mean(values))
    std_val = float(np.std(values))

    if std_val == 0.0 or np.isnan(std_val):
        return [0.0] * len(values)

    return [float((v - mean_val) / std_val) for v in values]


# ==============================================================================
# 4. CAMERA MOTION ASSIGNMENT & MULTI-CRITERIA PRIORITIZATION
# ==============================================================================

def assign_camera_motion(z_distance: float, z_extent: float, z_volatility: float) -> tuple[str, float]:
    """
    Determine the optimal camera motion type and overall priority score.

    """
    dominant_score = max(z_distance, z_extent, z_volatility)

    if dominant_score == z_distance:
        motion = "Follow"
    elif dominant_score == z_extent:
        motion = "Spline"
    else:
        motion = "Spin"

    return motion, dominant_score


def rank_tracks(tc_tracks: dict,
                ar_tracks: dict,
                tc_blobs: dict = None,
                ar_blobs: dict = None,
                min_lifetime: int = DEFAULT_MIN_LIFETIME,
                top_n: int = DEFAULT_TOP_N) -> tuple[list[dict], list[dict]]:
    candidate_rows = []

    # 1. Evaluate Tropical Cyclone (TC) tracks
    for tid, seq in tc_tracks.items():
        if len(seq) >= min_lifetime:
            metrics = compute_track_metrics(seq, tc_blobs)
            candidate_rows.append({
                "feature": "TC",
                "id": tid,
                "seq": seq,
                **metrics,
            })

    # 2. Evaluate Atmospheric River (AR) tracks
    for tid, seq in ar_tracks.items():
        if len(seq) >= min_lifetime:
            metrics = compute_track_metrics(seq, ar_blobs)
            candidate_rows.append({
                "feature": "AR",
                "id": tid,
                "seq": seq,
                **metrics,
            })

    if not candidate_rows:
        return [], []

    # 3. Compute Z-Scores across all candidate tracks for each dimension
    distances = [r["total_distance_traveled"] for r in candidate_rows]
    extents = [r["max_extent_km"] for r in candidate_rows]
    volatilities = [r["shape_volatility"] for r in candidate_rows]

    z_distances = compute_z_scores(distances)
    z_extents = compute_z_scores(extents)
    z_volatilities = compute_z_scores(volatilities)

    # 4. Assign normalized Z-scores, overall score, and camera motion type
    for idx, row in enumerate(candidate_rows):
        z_d = z_distances[idx]
        z_e = z_extents[idx]
        z_v = z_volatilities[idx]

        row["z_distance"] = round(z_d, 4)
        row["z_extent"] = round(z_e, 4)
        row["z_volatility"] = round(z_v, 4)

        motion_type, dominant_score = assign_camera_motion(z_d, z_e, z_v)
        row["camera_motion"] = motion_type
        row["score"] = round(dominant_score, 4)

    # 5. Global sorting by highest Z-score
    candidate_rows.sort(key=lambda r: r["score"], reverse=True)

    for rank_idx, row in enumerate(candidate_rows, start=1):
        row["rank"] = rank_idx

    return candidate_rows, candidate_rows[:top_n]