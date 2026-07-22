import numpy as np
from extractor import haversine_km, MAX_SEARCH_KM


def track_metrics(seq):
    """seq: one track's list of {lon, lat, area, timestep}, already time-sorted."""
    lifetime = len(seq)
    speeds = []
    for a, b in zip(seq[:-1], seq[1:]):
        dt = max(b["timestep"] - a["timestep"], 1)
        speeds.append(haversine_km(a["lon"], a["lat"], b["lon"], b["lat"]) / dt)
    areas = [p["area"] for p in seq]
    return {
        "lifetime": lifetime,
        "avg_speed_kmh": float(np.mean(speeds)) if speeds else 0.0,
        "area_growth_rate": (max(areas) - min(areas)) / lifetime if lifetime and areas else 0.0,
    }


def count_split_merge_events(seq, all_blobs_by_timestep, radius_km=MAX_SEARCH_KM):
    """
    At each timestep this track was active, count how many OTHER detected
    blobs sit within radius_km of it. >0 means a sibling blob was nearby —
    a candidate split (about to divide) or merge (about to combine) moment,
    the same ambiguous situation the SCAFET paper's Klaus/Australia case
    studies describe.
    """
    events = 0
    for p in seq:
        others = all_blobs_by_timestep.get(p["timestep"], [])
        nearby = sum(
            1 for b in others
            if b is not p and 0 < haversine_km(p["lon"], p["lat"], b["lon"], b["lat"]) <= radius_km
        )
        events += 1 if nearby > 0 else 0
    return events


def rank_tracks(tc_tracks, ar_tracks, tc_blobs, ar_blobs, weights=None, top_n=2):
    weights = weights or {"speed": 0.35, "lifetime": 0.25, "split_merge": 0.25, "area_growth": 0.15}

    rows = []
    for tid, seq in tc_tracks.items():
        m = track_metrics(seq)
        m["split_merge"] = count_split_merge_events(seq, tc_blobs)
        rows.append({"feature": "TC", "id": tid, "seq": seq, **m})
    for tid, seq in ar_tracks.items():
        m = track_metrics(seq)
        m["split_merge"] = count_split_merge_events(seq, ar_blobs)
        rows.append({"feature": "AR", "id": tid, "seq": seq, **m})

    def normed(key):
        vals = [r[key] for r in rows]
        lo, hi = min(vals), max(vals)
        return {id(r): (0.0 if hi == lo else (r[key] - lo) / (hi - lo)) for r in rows}

    n_speed, n_life = normed("avg_speed_kmh"), normed("lifetime")
    n_split, n_area = normed("split_merge"), normed("area_growth_rate")

    for r in rows:
        r["score"] = (
            weights["speed"] * n_speed[id(r)] + weights["lifetime"] * n_life[id(r)] +
            weights["split_merge"] * n_split[id(r)] + weights["area_growth"] * n_area[id(r)]
        )

    rows.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1

    return rows, rows[:top_n]