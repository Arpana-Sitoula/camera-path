import os

# ── Met3D Camera Sequence Export ──────────────────────────────────────────────

def export_single_track_xml(track_seq, z_value, output_path, frame_time=10, name="TopFeatureSequence"):
    """
    Export a single feature's track into a Met3D CameraSequence XML file.
    Each keyframe represents one timestep with advanceTimestep="1".

    Parameters:
        track_seq (list[dict]): Time-ordered track points containing 'lat', 'lon', 'timestep'.
        z_value (float): Camera zoom/height level (e.g., 30 for TC, 60 for AR).
        output_path (str): Destination XML file path.
        frame_time (int): Transition time in seconds between keyframes.
        name (str): Sequence identifier for Met3D.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    runtime = frame_time * len(track_seq)

    lines = [
        '<!DOCTYPE CameraSequence>',
        f'<CameraSequence frameTime="{frame_time}" loop="0" name="{name}" runtime="{runtime}" tension="0">'
    ]

    for p in track_seq:
        lines.append(
            f'  <SequenceKey advanceTimestep="1" isOrthographic="1" label="" '
            f'lat="{p["lat"]:.6f}" lon="{p["lon"]:.6f}" pitch="0" roll="0" '
            f'transition="1" yaw="0" z="{z_value}"/>'
        )

    lines.append('</CameraSequence>')

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")