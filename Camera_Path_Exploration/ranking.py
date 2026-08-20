import numpy as np
from extractor import haversine_km, MAX_SEARCH_KM


# ── Metric Computation ────────────────────────────────────────────────────────

def compute_track_metrics(seq):
    """
    Calculate headline kinematic and morphological metrics for a single track.

    Parameters:
        seq (list[dict]): Time-ordered points for one track, each containing
                          'lon', 'lat', 'area', 'timestep', and optionally 'time'.

    Returns:
        dict: lifetime, avg_speed_kmh, area_growth_rate, max_area, min_area
    """
    lifetime = len(seq)
    if lifetime == 0:
        return {
            "lifetime": 0,
            "avg_speed_kmh": 0.0,
            "area_growth_rate": 0.0,
            "max_area": 0,
            "min_area": 0,
        }

    speeds = []
    for a, b in zip(seq[:-1], seq[1:]):
        dist_km = haversine_km(a["lon"], a["lat"], b["lon"], b["lat"])

        # Calculate time delta in hours
        if "time" in a and "time" in b and a["time"] is not None and b["time"] is not None:
            # numpy datetime64 delta to hours
            dt_ns = (b["time"] - a["time"]).astype("timedelta64[ns]").astype(float)
            dt_hours = max(dt_ns / (1e9 * 3600.0), 0.1)
        else:
            dt_hours = max(float(b["timestep"] - a["timestep"]), 1.0)

        speeds.append(dist_km / dt_hours)

    areas = [p["area"] for p in seq]
    max_area = max(areas)
    min_area = min(areas)

    return {
        "lifetime": lifetime,
        "avg_speed_kmh": float(np.mean(speeds)) if speeds else 0.0,
        "area_growth_rate": float((max_area - min_area) / lifetime) if lifetime else 0.0,
        "max_area": max_area,
        "min_area": min_area,
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

def rank_tracks(tc_tracks, ar_tracks, tc_blobs, ar_blobs, weights=None, top_n=5):
    """
    Rank all extracted tracks based on literature-grounded life-cycle criteria.

    Default Weights:
        - speed: 0.35 (propagation speed)
        - lifetime: 0.25 (durability of the phenomenon)
        - split_merge: 0.25 (interaction ambiguity / topological complexity)
        - area_growth: 0.15 (intensification / expansion rate)

    Returns:
        tuple[list[dict], list[dict]]: (all_ranked_tracks, top_n_tracks)
    """
    if weights is None:
        weights = {
            "speed": 0.35,
            "lifetime": 0.25,
            "split_merge": 0.25,
            "area_growth": 0.15,
        }

    rows = []
    for tid, seq in tc_tracks.items():
        m = compute_track_metrics(seq)
        m["split_merge"] = count_split_merge_events(seq, tc_blobs)
        rows.append({"feature": "TC", "id": tid, "seq": seq, **m})

    for tid, seq in ar_tracks.items():
        m = compute_track_metrics(seq)
        m["split_merge"] = count_split_merge_events(seq, ar_blobs)
        rows.append({"feature": "AR", "id": tid, "seq": seq, **m})

    if not rows:
        return [], []

    def normalize(key):
        vals = [r[key] for r in rows]
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return {id(r): 0.0 for r in rows}
        return {id(r): (r[key] - lo) / (hi - lo) for r in rows}

    n_speed = normalize("avg_speed_kmh")
    n_life = normalize("lifetime")
    n_split = normalize("split_merge")
    n_area = normalize("area_growth_rate")

    for r in rows:
        rid = id(r)
        r["score"] = float(
            weights["speed"] * n_speed[rid] +
            weights["lifetime"] * n_life[rid] +
            weights["split_merge"] * n_split[rid] +
            weights["area_growth"] * n_area[rid]
        )

    rows.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1

    return rows, rows[:top_n]