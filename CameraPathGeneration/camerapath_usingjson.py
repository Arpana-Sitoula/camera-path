import xarray as xr
import numpy as np
import xml.etree.ElementTree as ET
import xml.dom.minidom
import math
import json
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from scipy.ndimage import label 
# ==========================================
# CONFIGURATION
# ==========================================
NC_FILE_PATH = r"C:\G\MASTERS\sem4\ResearchProjectMet3d\NAWDIC_CNN_Features\2026_01_20_Dublin\AR_TC_result.nc"
JSON_FILE_PATH = "top_features_camera_data.json"
VARIABLE_NAME = "AR"
MAX_TIMELINE_FRAMES = 61

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def calculate_yaw(lat_cam, lon_cam, lat_target, lon_target):
    """Calculates West-to-East camera yaw."""
    dy = lat_target - lat_cam
    dx = lon_target - lon_cam
    yaw = -math.degrees(math.atan2(dx, dy))
    return yaw

def extract_skeleton_path(ds, region_bounds, target_t, num_waypoints=5):
    """Extracts the skeleton path of ONLY the largest connected AR blob."""
    lat_min, lat_max, lon_min, lon_max = region_bounds
    
    # Clip bounds to ensure they don't exceed valid geographical limits
    lat_min, lat_max = max(-90, lat_min), min(90, lat_max)
    lon_min, lon_max = max(-180, lon_min), min(180, lon_max)
    
    ds_region = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    
    current_frame = ds_region[VARIABLE_NAME].isel(time=target_t).values
    
    # --- NEW: ISOLATE THE LARGEST BLOB ---
    # 1. Label all distinct, disconnected AR blobs in the frame
    labeled_frame, num_features = label(current_frame)
    
    if num_features == 0:
        raise ValueError(f"No ARs found in dynamic bounds at timestep {target_t}")
        
    # 2. Find which label belongs to the largest blob
    largest_label = 0
    max_pixels = 0
    for i in range(1, num_features + 1):
        pixel_count = np.sum(labeled_frame == i)
        if pixel_count > max_pixels:
            max_pixels = pixel_count
            largest_label = i
            
    # 3. Create a clean frame keeping ONLY the largest AR
    clean_frame = (labeled_frame == largest_label).astype(int)
    
    # --- Proceed with skeletonizing the clean frame ---
    skeleton = skeletonize(clean_frame)
    
    lat_idx, lon_idx = np.where(skeleton)
    
    if len(lat_idx) < num_waypoints:
        raise ValueError(f"AR skeleton too small in dynamic bounds at timestep {target_t}")
        
    actual_lats = ds_region.lat.values[lat_idx]
    actual_lons = ds_region.lon.values[lon_idx]
    
    # Sort coordinates by longitude (West to East)
    sort_idx = np.argsort(actual_lons)
    sorted_lats = actual_lats[sort_idx]
    sorted_lons = actual_lons[sort_idx]
    
    indices = np.linspace(0, len(sorted_lats) - 1, num_waypoints).astype(int)
    wp_lats = sorted_lats[indices]
    wp_lons = sorted_lons[indices]
    # --- PLOTTING ---
    plt.figure(figsize=(12, 7))
    
    # 1. Plot original AR mask (Blue)
    lons, lats = np.meshgrid(ds_region.lon.values, ds_region.lat.values)
    plt.pcolormesh(lons, lats, current_frame, cmap='Blues', alpha=0.5, shading='auto')
    
    # 2. Plot Skeleton (Red)
    plt.scatter(actual_lons, actual_lats, color='red', s=10, label='1-Pixel Skeleton')
    
    # 3. Plot chosen Waypoints (Yellow with black edge)
    plt.scatter(wp_lons, wp_lats, color='yellow', edgecolor='black', s=150, zorder=5, label='Camera Waypoints')
    
    # Formatting
    plt.title(f"AR Skeletonization & Camera Path\n(Region: {region_bounds} | Timestep: {target_t})", fontsize=14)
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    return sorted_lats[indices], sorted_lons[indices]

def create_sequence_key(parent, lon, lat, z, pitch, yaw, advance_time, transition="1"):
    key = ET.SubElement(parent, "SequenceKey")
    key.set("advanceTimestep", str(advance_time))
    key.set("isOrthographic", "0")
    key.set("label", "")
    key.set("lat", str(lat))
    key.set("lon", str(lon))
    key.set("pitch", str(pitch))
    key.set("yaw", str(yaw))
    key.set("roll", "0")
    key.set("transition", str(transition))
    key.set("z", str(z))
    return key

# ==========================================
# MAIN GENERATOR
# ==========================================
def generate_xml_from_json(nc_path, json_path):
    ds = xr.open_dataset(nc_path)
    
    with open(json_path, 'r') as f:
        ar_data = json.load(f)
        
    # Sort by rank
    ar_data = sorted(ar_data, key=lambda x: x['rank'])
    
    # Determine file generation strategy
    total_lifetime = sum(ar['lifetime'] for ar in ar_data)
    separate_files = total_lifetime > MAX_TIMELINE_FRAMES
    
    if not separate_files:
        root = ET.Element("CameraSequence")
        root.set("name", "Ranked_AR_Sequence")
        
    for ar in ar_data:
        rank = ar['rank']
        motion = ar['camera_motion']
        seq = ar['seq']
        print(f"\nProcessing AR Rank {rank} | Motion: {motion} | Lifetime: {ar['lifetime']}")
        
        if separate_files:
            root = ET.Element("CameraSequence")
            root.set("name", f"AR_Rank_{rank}_{motion}")
            root.set("tension", "0.3")
            
        if motion == "Spline":
            # 1. Find the exact frame where the AR is longest
            best_frame = max(seq, key=lambda x: x['extent_km'])
            target_t = best_frame['timestep']
            
            dyn_bounds = (best_frame['lat'] - 60, best_frame['lat'] + 60, 
                          best_frame['lon'] - 75, best_frame['lon'] + 75)
            
            try:
                skel_lats, skel_lons = extract_skeleton_path(ds, dyn_bounds, target_t, num_waypoints=7)
            except Exception as e:
                print(f"Skipping Spline generation for Rank {rank}: {e}")
                continue

            total_wp = len(skel_lats)
            mid_idx = len(skel_lats) // 2
            
            start_lat, start_lon = skel_lats[0], skel_lons[0]
            mid_lat, mid_lon = skel_lats[mid_idx], skel_lons[mid_idx]
            
            # Regional target for the time evolution hold
            reg_lat, reg_lon = mid_lat - 10, mid_lon
            reg_yaw = calculate_yaw(reg_lat, reg_lon, mid_lat, mid_lon)
            print(reg_yaw)
            # --- STEP 1: Global Intro ---
            create_sequence_key(root, lon=0, lat=0, z=250, pitch=0, yaw=0, advance_time=0, transition="1")
            
            # --- STEP 2: Zoom in and start Part 1 of Time Evolution ---
            # Smoothly zoom down to the regional hold position (Time is frozen)
            for z_level in np.linspace(200, 150, 2):
                create_sequence_key(root, lon=reg_lon, lat=reg_lat, z=z_level, pitch=5, yaw=reg_yaw, advance_time=0, transition="1")
                
            # Run time evolution until it hits the max extent_km timestep
            for j in range(target_t):
                # Only transition="1" on the final frame so the camera can move for the skeleton phase
                trans = "1" if j == target_t - 1 else "0"
                create_sequence_key(root, lon=reg_lon, lat=reg_lat, z=130, pitch=5, yaw=reg_yaw, advance_time=1, transition=trans)

            # --- STEP 3: Time Frozen - Skeleton Fly-Through ---
            # Smooth entry down to the start of the skeleton
            for z_level in np.linspace(130, 75, 2):
                # Find the direction vector between the first two waypoints
                d_lat = skel_lats[1] - skel_lats[0]
                d_lon = skel_lons[1] - skel_lons[0]
                
                cam_lat = start_lat - (d_lat)
                cam_lon = start_lon - (d_lon)
                yaw = calculate_yaw(cam_lat, cam_lon, start_lat, start_lon)
                create_sequence_key(root, lon=cam_lon, lat=cam_lat, z=z_level, pitch=15, yaw=yaw, advance_time=0, transition="1")
            
            pitch_values = np.linspace(60, 20, total_wp)
            
            for i in range(total_wp):
                cam_lat, cam_lon = skel_lats[i], skel_lons[i]-5
                
                # Dynamically check if we are at the last index
                if i < (total_wp - 1):
                    yaw = calculate_yaw(cam_lat, cam_lon, skel_lats[i+1], skel_lons[i+1])
                else:
                    yaw = calculate_yaw(cam_lat, cam_lon, skel_lats[i], skel_lons[i])
                    
                create_sequence_key(root, lon=cam_lon, lat=cam_lat, z=25, pitch=pitch_values[i], yaw=yaw, advance_time=0, transition="1")

            # --- STEP 4: Return to Regional View & Finish Time Evolution ---
            # Smoothly zoom back out to the regional hold position
            for z_level in np.linspace(75, 100, 2):
                create_sequence_key(root, lon=reg_lon, lat=reg_lat, z=z_level, pitch=5, yaw=reg_yaw, advance_time=0, transition="1")
            
            # Resume time for the remainder of the AR's lifetime
            remaining_time = ar['lifetime'] - target_t
            for j in range(remaining_time):
                trans = "1" if j == remaining_time - 1 else "0"
                create_sequence_key(root, lon=reg_lon, lat=reg_lat, z=130, pitch=5, yaw=reg_yaw, advance_time=1, transition=trans)

            # --- STEP 5: Global Outro ---
            create_sequence_key(root, lon=0, lat=0, z=250, pitch=0, yaw=0, advance_time=0, transition="1")

        elif motion == "Follow":
            # with yaw calculated dynamically
            # start_lat, start_lon = seq[0]['lat'], seq[0]['lon']
            
            # create_sequence_key(root, lon=0, lat=0, z=250, pitch=0, yaw=0, advance_time=0, transition="1")
                
            # cam_lat, cam_lon = start_lat - 35, start_lon - 35
            # yaw = calculate_yaw(cam_lat, cam_lon, start_lat, start_lon)
            # create_sequence_key(root, lon=cam_lon, lat=cam_lat, z=80, pitch=40, yaw=yaw, advance_time=0, transition="0")
            
            # for i in range(len(seq)):
            #     current_lat, current_lon = seq[i]['lat'], seq[i]['lon']
            #     cam_lat, cam_lon = current_lat - 35, current_lon - 35
                
            #     if i < len(seq) - 1:
            #         next_lat, next_lon = seq[i+1]['lat'], seq[i+1]['lon']
            #         yaw = calculate_yaw(cam_lat, cam_lon, next_lat, next_lon)
                    
            #     create_sequence_key(root, lon=cam_lon, lat=cam_lat, z=80, pitch=40, yaw=yaw, advance_time=1, transition="1")
                
            # create_sequence_key(root, lon=0, lat=0, z=250, pitch=0, yaw=0, advance_time=0, transition="1")

            # with yaw zero
            start_lat, start_lon = seq[0]['lat'], seq[0]['lon']
                        
            create_sequence_key(root, lon=0, lat=0, z=250, pitch=0, yaw=0, advance_time=0, transition="1")
            
            cam_lat, cam_lon = start_lat, start_lon
            create_sequence_key(root, lon=cam_lon, lat=cam_lat, z=100, pitch=3, yaw=0, advance_time=0, transition="0")
            
            for i in range(len(seq)):
                cam_lat, cam_lon = seq[i]['lat'], seq[i]['lon']
                create_sequence_key(root, lon=cam_lon, lat=cam_lat, z=100, pitch=3, yaw=0, advance_time=1, transition="1")
                
            create_sequence_key(root, lon=0, lat=0, z=250, pitch=0, yaw=0, advance_time=0, transition="1")
        if separate_files:
            output_name = f"AR_Rank_{rank}_{motion}extent.xml"
            xml_str = xml.dom.minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
            with open(output_name, "w") as f:
                f.write(xml_str)
            print(f"Generated -> {output_name}")

    if not separate_files:
        output_name = "AR_All_Ranked_Sequences.xml"
        xml_str = xml.dom.minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
        with open(output_name, "w") as f:
            f.write(xml_str)
        print(f"\nGenerated -> {output_name}")

if __name__ == "__main__":
    generate_xml_from_json(NC_FILE_PATH, JSON_FILE_PATH)