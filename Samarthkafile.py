"""
camera_final.py
---------------
Generates a Met.3D camera sequence that orbits a single tracked
atmospheric river once, then holds still while the forecast advances.

Input : top_features_camera_data.json   (feature tracking output)
Output: camera_sequence_spin_azel.xml   (load in Met.3D)

Sequence structure (106 keys for a 61-step feature):
    establish   1 key   global view at z=250, pitch=0, already aimed
    approach    6 keys  interpolated descent onto the orbit ring
    orbit      36 keys  one full 360 deg revolution, 10 deg per key
    settle      2 keys  drop to the closer framing for the forecast
    hold       61 keys  camera frozen, advanceTimestep=1 on every key
"""

import json
import math
import xml.etree.ElementTree as ET
from xml.dom import minidom

# ---------------------------------------------------------------- config
JSON_FILE  = "top_features_camera_data.json"
OUTPUT_XML = "camera_sequence_spin_azel.xml"

# Met.3D transition enum (mcamerasequence.h)
TELEPORT = 0
SPLINE   = 1

# --- orbit -------------------------------------------------------------
PITCH_ORBIT  = 30.0    # deg from nadir; 0 = straight down, 90 = horizontal
Z_ORBIT      = 90.0    # Met.3D zoom parameter during the revolution
DEG_PER_UNIT = 1.0     # ground degrees per unit of z
MAX_RING_LAT = 80.0    # ring is clamped so it never crosses the pole
MIN_OFFSET   = 4.0     # deg; floor on the ring radius
AZ_SWEEP     = 360.0   # deg swept in one revolution
YAW_DIR      = 1       # +1 = counter-clockwise, -1 = clockwise

# --- framing before and after the orbit --------------------------------
Z_GLOBAL     = 250.0   # establishing shot
PITCH_GLOBAL = 0.0
Z_FREEZE     = 60.0    # framing held while the forecast runs
PITCH_FREEZE = 25.0

# --- keyframe counts ---------------------------------------------------
ORBIT_KEYS    = 36     # 10 deg of yaw per key
APPROACH_KEYS = 6
TENSION       = 1.0

ORBIT_AT = 0.0         # where in the feature's life to orbit: 0.0 = start,
                       # 0.5 = midpoint, 1.0 = end


# ------------------------------------------------------------- geometry
def aim_distance(z, pitch):
    """Ground distance from the camera to the point it is aimed at.

    Derived from the Met.3D rotation matrix: with pitch measured from
    nadir the view direction is (0, sin P, -cos P), so the ray meets the
    ground z * tan(P) ahead of the camera.
    """
    return z * math.tan(math.radians(pitch)) / DEG_PER_UNIT


def safe_offset(target_lat, desired):
    """Clamp the ring radius so it stays clear of the pole."""
    room = MAX_RING_LAT - abs(target_lat)
    return max(MIN_OFFSET, min(desired, room))


def great_circle_offset(lat0, lon0, bearing_deg, dist_deg):
    """Point dist_deg away from (lat0, lon0) along the given bearing."""
    lat1 = math.radians(lat0)
    lon1 = math.radians(lon0)
    br   = math.radians(bearing_deg)
    d    = math.radians(dist_deg)
    lat2 = math.asin(math.sin(lat1) * math.cos(d) +
                     math.cos(lat1) * math.sin(d) * math.cos(br))
    lon2 = lon1 + math.atan2(math.sin(br) * math.sin(d) * math.cos(lat1),
                             math.cos(d) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), (math.degrees(lon2) + 540) % 360 - 180


def yaw_to(cam_lat, cam_lon, tgt_lat, tgt_lon):
    """Met.3D yaw that points the camera at the target.

    The flat negated atan2 form, not the great-circle bearing. Verified
    against hand-authored sequences: reproduces their yaw values exactly.
    """
    return -math.degrees(math.atan2(tgt_lon - cam_lon, tgt_lat - cam_lat))


def camera_at(tgt_lat, tgt_lon, azimuth, offset):
    """Camera position on the ring at the given azimuth, plus its yaw."""
    cam_lat, cam_lon = great_circle_offset(tgt_lat, tgt_lon,
                                           180 + azimuth, offset)
    return cam_lat, cam_lon, yaw_to(cam_lat, cam_lon, tgt_lat, tgt_lon)


def unwrap(prev, new):
    """Continue `new` from `prev` without a +/-180 jump.

    Met.3D interpolates rotation as a quaternion spline, which always
    takes the shorter arc. Without unwrapping, the orbit reverses
    direction halfway round.
    """
    return prev + ((new - prev + 180) % 360 - 180)


# ------------------------------------------------------------ xml output
def num(v, nd):
    """Format a value the way Met.3D's own sequence files do."""
    return str(round(v, nd))


def add_key(root, lon, lat, z, pitch, yaw,
            transition=SPLINE, advance_time=0):
    ET.SubElement(root, "SequenceKey", {
        "label":           "",
        "lon":             num(lon, 6),
        "lat":             num(lat, 6),
        "z":               num(z, 3),
        "pitch":           num(pitch, 1),
        "yaw":             num(yaw, 3),
        "roll":            "0",
        "transition":      str(transition),
        "advanceTimestep": str(advance_time),
        "isOrthographic":  "0",
    })


def last_key(root):
    k = root[-1]
    return {
        "lon":   float(k.get("lon")),
        "lat":   float(k.get("lat")),
        "z":     float(k.get("z")),
        "pitch": float(k.get("pitch")),
        "yaw":   float(k.get("yaw")),
    }


def interpolate_aimed(root, start, end, n_steps, tgt_lat, tgt_lon,
                      advance_time=0, skip_last=False):
    """Interpolate position, then recompute yaw at each step.

    Blending the endpoint yaws instead let the aim drift by up to 57 deg
    in the middle of a segment. Recomputing from the interpolated
    position keeps the error at zero on every key.
    """
    last = n_steps if skip_last else n_steps + 1
    for i in range(1, last):
        a     = i / n_steps
        lon   = start["lon"]   + (end["lon"]   - start["lon"])   * a
        lat   = start["lat"]   + (end["lat"]   - start["lat"])   * a
        z     = start["z"]     + (end["z"]     - start["z"])     * a
        pitch = start["pitch"] + (end["pitch"] - start["pitch"]) * a
        yaw   = unwrap(last_key(root)["yaw"],
                       yaw_to(lat, lon, tgt_lat, tgt_lon))
        add_key(root, lon, lat, z, pitch, yaw,
                transition=SPLINE, advance_time=advance_time)


# ----------------------------------------------------------------- build
def load_feature():
    with open(JSON_FILE) as fh:
        data = json.load(fh)
    feats = data if isinstance(data, list) else data.get("features", data)
    if isinstance(feats, dict):
        feats = list(feats.values())
    feats.sort(key=lambda f: f.get("rank", 999))
    return feats[0]


def build():
    feat = load_feature()
    seq  = feat.get("seq") or feat.get("sequence")
    n_steps = len(seq)

    idx = int(ORBIT_AT * (n_steps - 1))
    tla = seq[idx]["lat"]
    tlo = seq[idx]["lon"]

    name = "{}{}".format(feat.get("feature", "AR"), feat.get("id", ""))
    print("{} ({}, rank {}): {} steps, forecast 0-{}".format(
        name, feat.get("camera_motion", "Spin"), feat.get("rank", "?"),
        n_steps, n_steps - 1))

    wanted = aim_distance(Z_ORBIT, PITCH_ORBIT)
    off    = safe_offset(tla, wanted)
    note   = "  (clamped from {:.1f})".format(wanted) if off < wanted else ""
    print("pitch {}, z {} -> offset {:.1f} deg{}".format(
        PITCH_ORBIT, Z_ORBIT, off, note))
    print("one orbit at step {} ({:.1f}, {:.1f}), then hold for {} frames"
          .format(idx, tlo, tla, n_steps))

    root = ET.Element("CameraSequence", {
        "name":    "Spin" + name,
        "tension": str(TENSION),
    })

    # -- 1. establishing key -------------------------------------------
    add_key(root, 0, 0, Z_GLOBAL, PITCH_GLOBAL,
            yaw_to(0, 0, tla, tlo), transition=SPLINE)

    # -- 2. approach onto the ring -------------------------------------
    clat, clon, _ = camera_at(tla, tlo, 0.0, off)
    interpolate_aimed(root, last_key(root),
                      {"lon": clon, "lat": clat,
                       "z": Z_ORBIT, "pitch": PITCH_ORBIT},
                      APPROACH_KEYS, tla, tlo)

    # -- 3. one full revolution ----------------------------------------
    run_yaw = last_key(root)["yaw"]
    for k in range(1, ORBIT_KEYS + 1):
        az = YAW_DIR * AZ_SWEEP * k / ORBIT_KEYS
        clat, clon, raw = camera_at(tla, tlo, az, off)
        run_yaw = unwrap(run_yaw, raw)
        add_key(root, clon, clat, Z_ORBIT, PITCH_ORBIT, run_yaw,
                transition=SPLINE)

    # -- 4. settle into the framing the forecast is shown in -----------
    soff = safe_offset(tla, aim_distance(Z_FREEZE, PITCH_FREEZE))
    slat, slon, _ = camera_at(tla, tlo, 0.0, soff)
    interpolate_aimed(root, last_key(root),
                      {"lon": slon, "lat": slat,
                       "z": Z_FREEZE, "pitch": PITCH_FREEZE},
                      3, tla, tlo, skip_last=True)

    # -- 5. hold still while the forecast runs -------------------------
    hold_yaw = unwrap(last_key(root)["yaw"], yaw_to(slat, slon, tla, tlo))
    for j in range(n_steps):
        trans = SPLINE if j == n_steps - 1 else TELEPORT
        add_key(root, slon, slat, Z_FREEZE, PITCH_FREEZE, hold_yaw,
                transition=trans, advance_time=1)

    xml = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
    with open(OUTPUT_XML, "w") as fh:
        fh.write(xml)

    print("\nwrote {}  ({} keys)".format(OUTPUT_XML, len(root)))


if __name__ == "__main__":
    build()