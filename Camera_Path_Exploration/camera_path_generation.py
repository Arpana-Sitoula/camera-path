import xarray as xr
import numpy as np
import xml.etree.ElementTree as ET
import xml.dom.minidom
import math
import json
from scipy.ndimage import label
from skimage.morphology import skeletonize
import os
# ==========================================
# FILE CONFIGURATION
# ==========================================
# NC_FILE_PATH = r"2026_01_20_Dublin\AR_TC_result.nc"
# JSON_FILE_PATH = "top_features_camera_data.json"
VARIABLE_NAME = "AR"
MAX_TIMELINE_FRAMES = 61

# ==========================================
# CAMERA & ANIMATION CONFIGURATION
# ==========================================
CAMERA_CONFIG = {
    # Global view shot
    "global_z": 250.0,
    
    # Follow sequence offsets
    "follow_z": 100.0, # set z = 80 and pitch = 40 for angled view.
    "follow_pitch": 3.0,
    "follow_lat_lon_offset": 35.0,
    
    # Skeleton sequence offsets
    "skel_regional_z": 130.0,
    "skel_regional_pitch": 5.0,
    "skel_regional_lat_offset": 10.0,
    "skel_dive_z": 25.0,
    "skel_dive_pitch_start": 60.0,
    "skel_dive_pitch_end": 20.0,
    "skel_waypoint_lon_offset": 5.0,
}

# --- Orbit / Spin Configuration ---
PITCH_ORBIT  = 30.0     
Z_ORBIT      = 90.0     
DEG_PER_UNIT = 1.0      
MAX_RING_LAT = 80.0     
MIN_OFFSET   = 4.0
AZ_SWEEP     = 360.0
YAW_DIR      = 1           
Z_FREEZE     = 60.0
PITCH_FREEZE = 25.0
ORBIT_KEYS   = 36      
APPROACH_KEYS= 6       
ORBIT_AT     = 0.0

# ==========================================
# HELPER FUNCTIONS: MATH & CAMERA
# ==========================================
def calculate_yaw(lat_cam, lon_cam, lat_target, lon_target):
    """Calculates West-to-East camera yaw."""
    dy = lat_target - lat_cam
    dx = lon_target - lon_cam
    return -math.degrees(math.atan2(dx, dy))

def aim_distance(z, pitch):
    return z * math.tan(math.radians(pitch)) / DEG_PER_UNIT

def safe_offset(target_lat, desired):
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

def camera_at(tgt_lat, tgt_lon, azimuth, offset):
    cam_lat, cam_lon = great_circle_offset(tgt_lat, tgt_lon, 180 + azimuth, offset)
    return cam_lat, cam_lon, calculate_yaw(cam_lat, cam_lon, tgt_lat, tgt_lon)

def unwrap(prev, new):
    return prev + ((new - prev + 180) % 360 - 180)

def create_sequence_key(parent, lon, lat, z, pitch, yaw, advance_time, transition="1"):
    key = ET.SubElement(parent, "SequenceKey")
    key.set("advanceTimestep", str(int(advance_time)))
    key.set("isOrthographic", "0")
    key.set("label", "")
    key.set("lat", str(round(lat, 6)))
    key.set("lon", str(round(lon, 6)))
    key.set("pitch", str(round(pitch, 3)))
    key.set("yaw", str(round(yaw, 3)))
    key.set("roll", "0")
    key.set("transition", str(transition))
    key.set("z", str(round(z, 3)))
    return key

def interpolate_aimed(root, start, end, n_steps, tgt_lat, tgt_lon, advance_time=0, skip_last=False):
    yaw_prev = start[4]
    state = start
    last = n_steps - 1 if skip_last else n_steps
    for i in range(1, last + 1):
        g = i / n_steps
        lon   = start[0] + (end[0] - start[0]) * g
        lat   = start[1] + (end[1] - start[1]) * g
        z     = start[2] + (end[2] - start[2]) * g
        pitch = start[3] + (end[3] - start[3]) * g
        yaw_prev = unwrap(yaw_prev, calculate_yaw(lat, lon, tgt_lat, tgt_lon))
        create_sequence_key(root, lon, lat, z, pitch, yaw_prev, advance_time, "1")
        state = (lon, lat, z, pitch, yaw_prev)
    return state

# ==========================================
# DRY HELPERS (Don't Repeat Yourself)
# ==========================================
def insert_global_view(root, advance_time=0, transition="1"):
    """Snaps the camera to the global wide view at z=250."""
    create_sequence_key(root, lon=0, lat=0, z=CAMERA_CONFIG["global_z"], 
                        pitch=0, yaw=0, advance_time=advance_time, transition=transition)

def smooth_zoom(root, lat, lon, z_start, z_end, pitch, yaw, steps, advance_time=0):
    """Interpolates a smooth z-axis zoom across multiple keyframes."""
    for z_level in np.linspace(z_start, z_end, steps):
        create_sequence_key(root, lon, lat, z_level, pitch, yaw, advance_time, "1")

# ==========================================
# HELPER FUNCTIONS: SKELETON
# ==========================================
def extract_skeleton_path(ds, region_bounds, target_t, num_waypoints=5):
    lat_min, lat_max, lon_min, lon_max = region_bounds
    lat_min, lat_max = max(-90, lat_min), min(90, lat_max)
    lon_min, lon_max = max(-180, lon_min), min(180, lon_max)
    
    ds_region = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    current_frame = ds_region[VARIABLE_NAME].isel(time=target_t).values
    
    labeled_frame, num_features = label(current_frame)
    if num_features == 0: raise ValueError(f"No ARs found at timestep {target_t}")
        
    # OPTIMIZATION: Instantaneously count pixels in all blobs using bincount (C-level efficiency)
    sizes = np.bincount(labeled_frame.ravel())
    sizes[0] = 0  # Ignore the background (label 0)
    largest_label = sizes.argmax()
    
    clean_frame = (labeled_frame == largest_label).astype(int)
    skeleton = skeletonize(clean_frame)
    
    lat_idx, lon_idx = np.where(skeleton)
    if len(lat_idx) < num_waypoints: raise ValueError(f"Skeleton too small.")
        
    actual_lats, actual_lons = ds_region.lat.values[lat_idx], ds_region.lon.values[lon_idx]
    sort_idx = np.argsort(actual_lons)
    sorted_lats, sorted_lons = actual_lats[sort_idx], actual_lons[sort_idx]
    
    indices = np.linspace(0, len(sorted_lats) - 1, num_waypoints).astype(int)
    return sorted_lats[indices], sorted_lons[indices]

# ==========================================
# MODULARIZED CAMERA SEQUENCES
# ==========================================
def build_skeleton_sequence(root, ds, ar):
    """Generates the cinematic intro, zoom, time-freeze, and flythrough for Skeleton motions."""
    seq = ar['seq']
    best_frame = max(seq, key=lambda x: x['extent_km'])
    target_t = best_frame['timestep']
    
    # Establish dynamic bounds to isolate the AR
    dyn_bounds = (best_frame['lat'] - 60, best_frame['lat'] + 60, 
                  best_frame['lon'] - 75, best_frame['lon'] + 75)
    
    try: 
        skel_lats, skel_lons = extract_skeleton_path(ds, dyn_bounds, target_t, num_waypoints=5)
    except Exception as e: 
        print(f"Skipping Skeleton: {e}")
        return
        
    total_wp = len(skel_lats)
    mid_idx = total_wp // 2   
    start_lat, start_lon = skel_lats[0], skel_lons[0]
    mid_lat, mid_lon = skel_lats[mid_idx], skel_lons[mid_idx]
    
    # Calculate the regional hold view metrics
    reg_lat = mid_lat - CAMERA_CONFIG["skel_regional_lat_offset"]
    reg_lon = mid_lon
    reg_yaw = calculate_yaw(reg_lat, reg_lon, mid_lat, mid_lon)

    # 1. Global Intro
    insert_global_view(root)

    # 2. Smooth zoom in and Part 1 of Time Evolution (Time advancing to peak)
    smooth_zoom(root, reg_lat, reg_lon, 200, 150, CAMERA_CONFIG["skel_regional_pitch"], reg_yaw, steps=2)
    for j in range(target_t):
        trans = "1" if j == target_t - 1 else "0"
        create_sequence_key(root, reg_lon, reg_lat, CAMERA_CONFIG["skel_regional_z"], 
                            CAMERA_CONFIG["skel_regional_pitch"], reg_yaw, advance_time=1, transition=trans)

    # 3. Time Frozen: Skeleton Fly-Through Smooth Entry
    d_lat = skel_lats[1] - skel_lats[0]
    d_lon = skel_lons[1] - skel_lons[0]
    cam_lat = start_lat - d_lat
    cam_lon = start_lon - d_lon
    yaw = calculate_yaw(cam_lat, cam_lon, start_lat, start_lon)
    smooth_zoom(root, cam_lat, cam_lon, 130, 75, 15, yaw, steps=2)
    
    # Dynamic pitch shift during the skeleton dive
    pitch_values = np.linspace(CAMERA_CONFIG["skel_dive_pitch_start"], CAMERA_CONFIG["skel_dive_pitch_end"], total_wp)
    
    for i in range(total_wp):
        cam_lat, cam_lon = skel_lats[i], skel_lons[i] - CAMERA_CONFIG["skel_waypoint_lon_offset"]
        if i < (total_wp - 1):
            yaw = calculate_yaw(cam_lat, cam_lon, skel_lats[i+1], skel_lons[i+1])
        else:
            yaw = calculate_yaw(cam_lat, cam_lon, skel_lats[i], skel_lons[i])
            
        create_sequence_key(root, cam_lon, cam_lat, CAMERA_CONFIG["skel_dive_z"], pitch_values[i], yaw, 0, "1")

    # 4. Return to Regional View & Resume Time Evolution
    smooth_zoom(root, reg_lat, reg_lon, 75, 100, CAMERA_CONFIG["skel_regional_pitch"], reg_yaw, steps=2)
    remaining_time = ar['lifetime'] - target_t
    for j in range(remaining_time):
        trans = "1" if j == remaining_time - 1 else "0"
        create_sequence_key(root, reg_lon, reg_lat, CAMERA_CONFIG["skel_regional_z"], 
                            CAMERA_CONFIG["skel_regional_pitch"], reg_yaw, 1, trans)

    # 5. Global Outro
    insert_global_view(root)


def build_follow_sequence(root, ar):
    """Generates the third-person tracking sequence matching the AR's speed."""
    seq = ar['seq']
    start_lat, start_lon = seq[0]['lat'], seq[0]['lon']
    offset = CAMERA_CONFIG["follow_lat_lon_offset"]
    
    insert_global_view(root)
    
    cam_lat, cam_lon = start_lat - offset, start_lon - offset
    yaw = calculate_yaw(cam_lat, cam_lon, start_lat, start_lon)
    create_sequence_key(root, cam_lon, cam_lat, CAMERA_CONFIG["follow_z"], CAMERA_CONFIG["follow_pitch"], yaw, 0, "0")
    
    # Iterate dynamically with advancing time[cite: 3]
    for i in range(len(seq)):
        current_lat, current_lon = seq[i]['lat'], seq[i]['lon']
        cam_lat, cam_lon = current_lat - offset, current_lon - offset
    
        if i < len(seq) - 1:
            next_lat, next_lon = seq[i+1]['lat'], seq[i+1]['lon']
            yaw = calculate_yaw(cam_lat, cam_lon, next_lat, next_lon)
    
        create_sequence_key(root, cam_lon, cam_lat, CAMERA_CONFIG["follow_z"], CAMERA_CONFIG["follow_pitch"], yaw, 1, "1")
    
    insert_global_view(root)


def build_spin_sequence(root, ar):
    """Generates a 360-degree orbital view of the AR[cite: 3]."""
    seq = ar['seq']
    n_steps = len(seq)
    desired = aim_distance(Z_ORBIT, PITCH_ORBIT)
    i0 = min(int(ORBIT_AT * (n_steps - 1)), n_steps - 1)
    tgt_lat, tgt_lon = seq[i0]["lat"], seq[i0]["lon"]

    offset = safe_offset(tgt_lat, desired)
    f_off  = safe_offset(tgt_lat, aim_distance(Z_FREEZE, PITCH_FREEZE))

    open_yaw = calculate_yaw(0.0, 0.0, tgt_lat, tgt_lon)
    create_sequence_key(root, 0, 0, CAMERA_CONFIG["global_z"], 0, open_yaw, 0, "1")
    state = (0.0, 0.0, CAMERA_CONFIG["global_z"], 0.0, open_yaw)

    s_lat, s_lon, _ = camera_at(tgt_lat, tgt_lon, 0.0, offset)
    state = interpolate_aimed(root, state, (s_lon, s_lat, Z_ORBIT, PITCH_ORBIT, 0.0), APPROACH_KEYS, tgt_lat, tgt_lon, advance_time=0)

    yaw_prev = state[4]
    az_step = AZ_SWEEP / ORBIT_KEYS
    
    for j in range(1, ORBIT_KEYS + 1):
        az = YAW_DIR * az_step * j
        o_lat, o_lon, o_yaw = camera_at(tgt_lat, tgt_lon, az, offset)
        yaw_prev = unwrap(yaw_prev, o_yaw)
        create_sequence_key(root, o_lon, o_lat, Z_ORBIT, PITCH_ORBIT, yaw_prev, 0, "1")
        state = (o_lon, o_lat, Z_ORBIT, PITCH_ORBIT, yaw_prev)

    h_lat, h_lon, _ = camera_at(tgt_lat, tgt_lon, 0.0, f_off)
    state = interpolate_aimed(root, state, (h_lon, h_lat, Z_FREEZE, PITCH_FREEZE, 0.0), 3, tgt_lat, tgt_lon, advance_time=0, skip_last=True)
    state = (h_lon, h_lat, Z_FREEZE, PITCH_FREEZE, unwrap(state[4], calculate_yaw(h_lat, h_lon, tgt_lat, tgt_lon)))

    for j in range(n_steps):
        trans = "1" if j == n_steps - 1 else "0"
        create_sequence_key(root, state[0], state[1], state[2], state[3], state[4], 1, trans)
        
    insert_global_view(root)

# ==========================================
# MAIN MASTER GENERATOR (The Controller)
# ==========================================
def generate_master_xml(nc_path, json_path, output_dir):
    """Parses data and routes each AR to the appropriate modular sequence builder."""
    ds = xr.open_dataset(nc_path)
    with open(json_path, 'r') as f: 
        ar_data = json.load(f)
        
    ar_data = sorted(ar_data, key=lambda x: x['rank'])
    
    total_lifetime = sum(ar['lifetime'] for ar in ar_data)
    separate_files = total_lifetime > MAX_TIMELINE_FRAMES
    
    if not separate_files:
        root = ET.Element("CameraSequence")
        root.set("name", "Master_AR_Sequence")
        
    for ar in ar_data:
        # Default to 'Follow' if motion attribute is missing[cite: 3]
        rank, motion = ar['rank'], ar.get('camera_motion', 'Follow')
        print(f"\nProcessing AR Rank {rank} | Motion: {motion} | Lifetime: {ar['lifetime']}")
        
        if separate_files:
            root = ET.Element("CameraSequence")
            root.set("name", f"AR_Rank_{rank}_{motion}")
            
        # --- ROUTING SYSTEM ---
        if motion == "Skeleton":
            build_skeleton_sequence(root, ds, ar)
        elif motion == "Follow":
            build_follow_sequence(root, ar)
        elif motion == "Spin":
            build_spin_sequence(root, ar)
        else:
            print(f"Unknown camera_motion '{motion}'. Skipping.")

        # Save individual file logic
        if separate_files:
            output_name = f"AR_Rank_{rank}_{motion}.xml"
            output_path = os.path.join(output_dir, output_name)
            xml_str = xml.dom.minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
            with open(output_path, "w") as f: 
                f.write(xml_str)
            print(f"Generated -> {output_path}")

    # Save single unified file logic
    if not separate_files:
        output_name = "AR_Master_Sequences.xml"
        output_path = os.path.join(output_dir, output_name)
        xml_str = xml.dom.minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
        with open(output_path, "w") as f: 
            f.write(xml_str)
        print(f"\nGenerated -> {output_path}")

if __name__ == "__main__":
    generate_master_xml(NC_FILE_PATH, JSON_FILE_PATH, OUTPUT_DIR=".")