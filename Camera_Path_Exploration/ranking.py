import numpy as np
from extractor import haversine_km, MAX_SEARCH_KM


# ── Metric Computation ────────────────────────────────────────────────────────

def compute_track_metrics(seq):
    """
    Calculate metrics for camera motion mapping.

    Parameters:
        seq (list[dict]): Time-ordered points for one track.

    Returns:
        dict: lifetime, total_distance_traveled, max_extent_km, shape_volatility
    """
    lifetime = len(seq)
    if lifetime == 0:
        return {
            "lifetime": 0,
            "total_distance_traveled": 0.0,
            "max_extent_km": 0.0,
            "shape_volatility": 0.0,
        }

    total_distance = 0.0
    area_changes = []
    
    for a, b in zip(seq[:-1], seq[1:]):
        dist_km = haversine_km(a["lon"], a["lat"], b["lon"], b["lat"])
        total_distance += dist_km
        
        # Absolute change in area (as a proxy for shape changes / grow & vanish)
        area_changes.append(abs(b["area"] - a["area"]))

    # Max extent is based on the extent_km we now extract
    extents = [p.get("extent_km", 0.0) for p in seq]
    max_extent_km = max(extents) if extents else 0.0

    # Shape volatility combines area changes and will later add split/merge counts
    # Using the mean absolute area change over the lifetime
    avg_area_change = float(np.mean(area_changes)) if area_changes else 0.0

    return {
        "lifetime": lifetime,
        "total_distance_traveled": total_distance,
        "max_extent_km": max_extent_km,
        "shape_volatility": avg_area_change,  
    }


def count_split_merge_events(seq, all_blobs_by_timestep, radius_km=MAX_SEARCH_KM):
    """
    Count candidate split/merge ambiguous moments where other detected blobs
    reside within proximity radius of the current feature.
    """
    events = 0
    for p in seq:
        others = all_blobs_by_timestep.get(p["timestep"], [])
        nearby = sum(
            1 for b in others
            if b is not p and 0 < haversine_km(p["lon"], p["lat"], b["lon"], b["lat"]) <= radius_km
        )
        if nearby > 0:
            events += 1
    return events


# ── Multi-Criteria Prioritization ─────────────────────────────────────────────

def rank_tracks(tc_tracks, ar_tracks, tc_blobs, ar_blobs, min_lifetime=3, top_n=5):
    """
    Rank all extracted tracks based on distinct camera motion criteria using Z-scores.
    
    Criteria:
        - total_distance_traveled -> Best for Follow Shots
        - max_extent_km -> Best for Spline Shots
        - shape_volatility -> Best for Spin Shots
    
    Tracks are assigned a primary camera motion based on which Z-score is highest.

    Returns:
        tuple[list[dict], list[dict]]: (all_ranked_tracks, top_n_tracks)
    """
    rows = []
    for tid, seq in tc_tracks.items():
        m = compute_track_metrics(seq)
        if m["lifetime"] >= min_lifetime:
            # Combine avg area change with split_merge events for total volatility
            sm_events = count_split_merge_events(seq, tc_blobs)
            m["split_merge"] = sm_events
            # Arbitrary weighting: add 500 "area change equivalent" per split/merge event
            m["shape_volatility"] += sm_events * 500.0
            
            rows.append({"feature": "TC", "id": tid, "seq": seq, **m})

    for tid, seq in ar_tracks.items():
        m = compute_track_metrics(seq)
        if m["lifetime"] >= min_lifetime:
            sm_events = count_split_merge_events(seq, ar_blobs)
            m["split_merge"] = sm_events
            m["shape_volatility"] += sm_events * 500.0
            
            rows.append({"feature": "AR", "id": tid, "seq": seq, **m})

    if not rows:
        return [], []

    def get_z_scores(key):
        vals = [r[key] for r in rows]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        if std_val == 0:
            return {id(r): 0.0 for r in rows}
        return {id(r): float((r[key] - mean_val) / std_val) for r in rows}

    z_distance = get_z_scores("total_distance_traveled")
    z_extent = get_z_scores("max_extent_km")
    z_volatility = get_z_scores("shape_volatility")

    for r in rows:
        rid = id(r)
        z_d = z_distance[rid]
        z_e = z_extent[rid]
        z_v = z_volatility[rid]
        
        r["z_distance"] = z_d
        r["z_extent"] = z_e
        r["z_volatility"] = z_v
        
        # Overall score is the maximum of its Z-scores (how exceptional is it in its best domain?)
        best_z = max(z_d, z_e, z_v)
        r["score"] = best_z
        
        if best_z == z_d:
            r["camera_motion"] = "Follow"
        elif best_z == z_e:
            r["camera_motion"] = "Spline"
        else:
            r["camera_motion"] = "Spin"

    # Sort globally by best Z-score
    rows.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1

    return rows, rows[:top_n]