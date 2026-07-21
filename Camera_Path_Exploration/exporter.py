import os
import json
from collections import defaultdict


# ── Config ────────────────────────────────────────────────────────────────────

Z_TC       = 30     # zoom level for tropical cyclones (closer view)
Z_AR       = 60     # zoom level for atmospheric rivers (wider view)
FRAME_TIME = 10     # time in seconds between keyframes in Met3D
RUNTIME    = 10     # total sequence runtime
NAME       = "AutoSequence"


# ── XML Builders ──────────────────────────────────────────────────────────────

def build_header():
    """Return the opening lines of the CameraSequence XML."""
    return [
        '<!DOCTYPE CameraSequence>',
        f'<CameraSequence frameTime="{FRAME_TIME}" loop="0" '
        f'name="{NAME}" runtime="{RUNTIME}" tension="0">'
    ]


def build_sequence_key(lat, lon, z, advance=0):
    """
    Return one <SequenceKey> line.
    advance=1 tells Met3D to move to the next timestep after this key.
    """
    return (
        f'  <SequenceKey advanceTimestep="{advance}" isOrthographic="1" label="" '
        f'lat="{lat:.6f}" lon="{lon:.6f}" pitch="0" roll="0" '
        f'transition="1" yaw="0" z="{z}"/>'
    )


# ── Track Grouping ────────────────────────────────────────────────────────────

def group_by_timestep(tc_tracks, ar_tracks):
    """
    Flatten all track points into a dict grouped by timestep.
    Structure: {timestep: [{lon, lat, z}, ...]}

    TC points get Z_TC zoom, AR points get Z_AR zoom.
    """
    by_timestep = defaultdict(list)

    for tid, seq in tc_tracks.items():
        for point in seq:
            by_timestep[point["timestep"]].append({
                "lon": point["lon"],
                "lat": point["lat"],
                "z":   Z_TC
            })

    for tid, seq in ar_tracks.items():
        for point in seq:
            by_timestep[point["timestep"]].append({
                "lon": point["lon"],
                "lat": point["lat"],
                "z":   Z_AR
            })

    return by_timestep


def build_keyframe_lines(by_timestep):
    """
    Build all <SequenceKey> lines ordered by timestep.
    The last point of each timestep group gets advanceTimestep=1
    so Met3D advances the clock after visiting all features at that time.
    """
    lines = []

    for t in sorted(by_timestep.keys()):
        points = by_timestep[t]

        for i, p in enumerate(points):
            # advance timestep only on the last point of this group
            advance = 1 if i == len(points) - 1 else 0
            lines.append(build_sequence_key(p["lat"], p["lon"], p["z"], advance))

    return lines


# ── Public API ────────────────────────────────────────────────────────────────

def export_tracks_to_xml(tc_tracks, ar_tracks, output_dir):
    """
    Main function — writes one Met3D CameraSequence XML file
    from TC and AR tracks across all timesteps.
    """
    os.makedirs(output_dir, exist_ok=True)

    by_timestep = group_by_timestep(tc_tracks, ar_tracks)

    lines  = build_header()
    lines += build_keyframe_lines(by_timestep)
    lines.append('</CameraSequence>')

    out_path = os.path.join(output_dir, "camera_sequence_tracked.xml")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"XML saved → {out_path}")