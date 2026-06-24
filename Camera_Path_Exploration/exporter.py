import json
import os


Z_TC       = 30    # zoom for tropical cyclones (closer)
Z_AR       = 60    # zoom for atmospheric rivers (further out)
FRAME_TIME = 10    # time between keyframes in Met3D
RUNTIME    = 10    # total runtime
NAME       = "AutoSequence"


def _z_for_type(point_type):
    if point_type == "TC":
        return Z_TC
    return Z_AR  # AR_start, AR_center, AR_end all get wider zoom


def _make_sequence_key(lat, lon, z):
    return (
        f'  <SequenceKey advanceTimestep="0" isOrthographic="1" label="" '
        f'lat="{lat:.6f}" lon="{lon:.6f}" pitch="0" roll="0" '
        f'transition="1" yaw="0" z="{z}"/>'
    )


def export_to_xml(keyframes_json_path, output_dir):
    """
    Reads keyframes_2d.json and writes one XML file per timestep.
    """
    with open(keyframes_json_path) as f:
        all_keyframes = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for entry in all_keyframes:
        timestep     = entry["timestep"]
        camera_path  = entry["camera_path"]

        lines = []
        lines.append('<!DOCTYPE CameraSequence>')
        lines.append(
            f'<CameraSequence frameTime="{FRAME_TIME}" loop="0" '
            f'name="{NAME}_t{timestep}" runtime="{RUNTIME}" tension="0">'
        )

        for point in camera_path:
            z    = _z_for_type(point["type"])
            line = _make_sequence_key(point["lat"], point["lon"], z)
            lines.append(line)

        lines.append('</CameraSequence>')

        out_path = os.path.join(output_dir, f"camera_sequence_{timestep:02d}.xml")
        with open(out_path, "w") as f:
            f.write("\n".join(lines))

        print(f"[timestep {timestep}] XML saved → {out_path}")