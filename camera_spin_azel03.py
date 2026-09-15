"""
SPIN SEQUENCE FOR A TRACKED ATMOSPHERIC RIVER

One orbit per feature: the camera flies in, circles the AR once, settles,
then holds while the forecast runs through the feature's lifetime.

Conventions taken from the Met.3D source (mcamerasequence.h, camera.cpp,
mcameraanimationcontroller.cpp):

  position   = lon / lat / z, raw values
  rotation   = pitch / yaw / roll
  transition = 0 TELEPORT, 1 SPLINE; independent of advanceTimestep
  pitch 0 = straight down, 90 = horizontal north

  A camera at height z with pitch P aims z*tan(P) ahead of itself, so it
  is placed that far from the target to centre the feature.

  yaw = -atan2(dlon, dlat), verified against the project's working
  sequences.

  The feature is the subject of the animation, so the camera aims at it
  at EVERY key -- including during approaches and settles, where yaw is
  recomputed from the interpolated position rather than blended between
  endpoint orientations.

  Freeze segments run: first frame with advanceTimestep and TELEPORT both
  set, middle frames the same, and a final frame with advanceTimestep set
  and transition SPLINE so the camera can move on.

Run:
  python camera_spin_azel.py --list
  python camera_spin_azel.py --rank 1
  python camera_spin_azel.py --id 4 --pitch 30 --z 30
"""
import json
import math
import argparse
import xml.etree.ElementTree as ET
import xml.dom.minidom

JSON_FILE  = "top_features_camera_data.json"
OUTPUT_XML = "camera_sequence_spin_azel.xml"

TELEPORT = 0
SPLINE   = 1

# --- orbit configuration ---
PITCH_ORBIT  = 30.0     # 0 = straight down, 90 = horizontal
Z_ORBIT      = 90.0     # camera height
DEG_PER_UNIT = 1.0      # world units per degree of latitude

MAX_RING_LAT = 80.0     # ring must stay inside this latitude
MIN_OFFSET   = 4.0

AZ_SWEEP  = 360.0
YAW_DIR   = 1           # flip to -1 to orbit the other way

# --- establishing / freeze framing ---
Z_GLOBAL     = 250.0
PITCH_GLOBAL = 0.0
Z_FREEZE     = 60.0
PITCH_FREEZE = 25.0

ORBIT_KEYS    = 36      # worst mid-segment aim error ~0.8 deg
APPROACH_KEYS = 6       # more steps = smoother fly-in
TENSION       = 1.0

# which point along the feature's life to orbit (0 = first, 0.5 = middle)
ORBIT_AT = 0.0


def aim_distance(z, pitch):
    """Ground distance from the camera to the point it aims at, in degrees."""
    return z * math.tan(math.radians(pitch)) / DEG_PER_UNIT


def safe_offset(target_lat, desired):
    """Shrink the orbit radius so the ring does not run into a pole."""
    room = MAX_RING_LAT - abs(target_lat)
    return max(MIN_OFFSET, min(desired, room))


def great_circle_offset(lat0, lon0, bearing_deg, dist_deg):
    lat1 = math.radians(lat0); lon1 = math.radians(lon0)
    br = math.radians(bearing_deg); d = math.radians(dist_deg)
    lat2 = math.asin(math.sin(lat1) * math.cos(d) +
                     math.cos(lat1) * math.sin(d) * math.cos(br))
    lon2 = lon1 + math.atan2(math.sin(br) * math.sin(d) * math.cos(lat1),
                             math.cos(d) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), (math.degrees(lon2) + 540) % 360 - 180


def yaw_to(cam_lat, cam_lon, tgt_lat, tgt_lon):
    """Negated flat atan2 -- reproduces the project's working sequences."""
    return -math.degrees(math.atan2(tgt_lon - cam_lon, tgt_lat - cam_lat))


def camera_at(tgt_lat, tgt_lon, azimuth, offset):
    """Camera on a ring around the target; azimuth 0 places it due south."""
    cam_lat, cam_lon = great_circle_offset(tgt_lat, tgt_lon,
                                           180 + azimuth, offset)
    return cam_lat, cam_lon, yaw_to(cam_lat, cam_lon, tgt_lat, tgt_lon)


def key(parent, lon, lat, z, pitch, yaw, advance_time, transition=SPLINE):
    k = ET.SubElement(parent, "SequenceKey")
    k.set("advanceTimestep", str(int(advance_time)))
    k.set("isOrthographic", "0")
    k.set("label", "")
    k.set("lat", str(round(lat, 6)))
    k.set("lon", str(round(lon, 6)))
    k.set("pitch", str(round(pitch, 3)))
    k.set("roll", "0")
    k.set("transition", str(int(transition)))
    k.set("yaw", str(round(yaw, 3)))
    k.set("z", str(round(z, 3)))
    return k


def unwrap(prev, new):
    """Continue an angle sequence without 360-degree jumps."""
    return prev + ((new - prev + 180) % 360 - 180)


def interpolate_aimed(root, start, end, n_steps, tgt_lat, tgt_lon,
                      advance_time=0, skip_last=False):
    """Move between two camera states while staying aimed at the target.

    Position, height and pitch are interpolated linearly, but yaw is
    recomputed from each interpolated position -- blending the endpoint
    yaws instead would swing the camera off the feature mid-transition.

    With skip_last, the final key is not written -- use this when the
    next segment already begins at that exact position."""
    yaw_prev = start[4]
    state = start
    last = n_steps - 1 if skip_last else n_steps
    for i in range(1, last + 1):
        g = i / n_steps
        lon   = start[0] + (end[0] - start[0]) * g
        lat   = start[1] + (end[1] - start[1]) * g
        z     = start[2] + (end[2] - start[2]) * g
        pitch = start[3] + (end[3] - start[3]) * g
        yaw_prev = unwrap(yaw_prev, yaw_to(lat, lon, tgt_lat, tgt_lon))
        key(root, lon, lat, z, pitch, yaw_prev, advance_time, SPLINE)
        state = (lon, lat, z, pitch, yaw_prev)
    return state


def build(feature, pitch, z_orbit):
    seq = feature["seq"]
    n_steps = len(seq)
    desired = aim_distance(z_orbit, pitch)

    root = ET.Element("CameraSequence")
    root.set("name", f"SpinAR{feature['id']}")
    root.set("tension", str(TENSION))

    # a single orbit, taken at one point along the feature's life
    i0 = min(int(ORBIT_AT * (n_steps - 1)), n_steps - 1)
    pt = seq[i0]
    tgt_lat, tgt_lon = pt["lat"], pt["lon"]

    offset = safe_offset(tgt_lat, desired)
    f_off  = safe_offset(tgt_lat, aim_distance(Z_FREEZE, PITCH_FREEZE))

    start = feature.get('start_step', 0)
    end   = feature.get('end_step', n_steps - 1)
    note = "" if offset >= desired - 0.01 else f"  (clamped from {desired:.1f})"
    print(f"AR{feature['id']} ({feature['camera_motion']}, rank {feature['rank']}): "
          f"{n_steps} steps, forecast {start}-{end}")
    print(f"pitch {pitch}, z {z_orbit} -> offset {offset:.1f} deg{note}")
    print(f"one orbit at step {pt['timestep']} "
          f"({tgt_lon:.1f}, {tgt_lat:.1f}), then hold for {n_steps} frames")

    # opening establishing view, already aimed at the target
    open_yaw = yaw_to(0.0, 0.0, tgt_lat, tgt_lon)
    key(root, 0, 0, Z_GLOBAL, PITCH_GLOBAL, open_yaw, 0, SPLINE)
    state = (0.0, 0.0, Z_GLOBAL, PITCH_GLOBAL, open_yaw)

    # --- approach, aimed at the target the whole way ---
    s_lat, s_lon, _ = camera_at(tgt_lat, tgt_lon, 0.0, offset)
    state = interpolate_aimed(
        root, state, (s_lon, s_lat, z_orbit, pitch, 0.0),
        APPROACH_KEYS, tgt_lat, tgt_lon, advance_time=0)

    # --- one full orbit ---
    yaw_prev = state[4]
    az_step = AZ_SWEEP / ORBIT_KEYS
    for j in range(1, ORBIT_KEYS + 1):
        az = YAW_DIR * az_step * j
        o_lat, o_lon, o_yaw = camera_at(tgt_lat, tgt_lon, az, offset)
        yaw_prev = unwrap(yaw_prev, o_yaw)
        key(root, o_lon, o_lat, z_orbit, pitch, yaw_prev, 0, SPLINE)
        state = (o_lon, o_lat, z_orbit, pitch, yaw_prev)

    # --- settle toward the freeze position, still aimed at the target ---
    # the final position is the freeze's own first frame, so it is not
    # written twice
    h_lat, h_lon, _ = camera_at(tgt_lat, tgt_lon, 0.0, f_off)
    state = interpolate_aimed(
        root, state, (h_lon, h_lat, Z_FREEZE, PITCH_FREEZE, 0.0),
        3, tgt_lat, tgt_lon, advance_time=0, skip_last=True)
    state = (h_lon, h_lat, Z_FREEZE, PITCH_FREEZE,
             unwrap(state[4], yaw_to(h_lat, h_lon, tgt_lat, tgt_lon)))

    # --- hold still while the forecast runs through the feature's life ---
    #   all frames advance time; the last one uses SPLINE so the camera
    #   is free to move on, the rest TELEPORT so the hold is exact
    for j in range(n_steps):
        trans = SPLINE if j == n_steps - 1 else TELEPORT
        key(root, state[0], state[1], state[2], state[3], state[4],
            1, trans)

    xml_str = xml.dom.minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
    with open(OUTPUT_XML, "w") as f:
        f.write(xml_str)
    print(f"\nwrote {OUTPUT_XML}  ({len(root.findall('SequenceKey'))} keys)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=None)
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--pitch", type=float, default=PITCH_ORBIT)
    ap.add_argument("--z", type=float, default=Z_ORBIT)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    features = json.load(open(JSON_FILE))

    if args.list:
        print(f"{'rank':>4} {'id':>4} {'motion':<9} {'life':>5} {'steps':>9}")
        for f in sorted(features, key=lambda f: f["rank"]):
            s = f.get('start_step', 0)
            e = f.get('end_step', len(f['seq']) - 1)
            print(f"{f['rank']:>4} {f['id']:>4} {f['camera_motion']:<9} "
                  f"{f['lifetime']:>5} {s:>3}-{e:<4}")
        return

    if args.id is not None:
        feature = next((f for f in features if f["id"] == args.id), None)
    else:
        feature = next((f for f in features if f["rank"] == args.rank), None)
    if feature is None:
        raise SystemExit("feature not found")

    build(feature, args.pitch, args.z)


if __name__ == "__main__":
    main()