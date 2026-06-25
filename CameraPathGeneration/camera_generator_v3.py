"""Camera revolves around each of the ARs and goes to the nearest detected AR. The 3d component is not working. Need to fix!"""
import math
import numpy as np
from typing import Dict, List, Tuple
import sys
import xml.etree.ElementTree as ET

from netCDF4 import Dataset
from scipy import ndimage

def read_nc_features(nc_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (times, lats, lons, dict of masks) where masks['AR'] and masks['TC'] are boolean arrays
    with shape (time, lat, lon).
    """
    ds = Dataset(nc_path)

    ar = ds['AR'][:]
    tc = ds['TC'][:]

    if 'lat' in ds.variables:
        lats = ds['lat'][:]
    else:
        print("lat variable not found")

    if 'lon' in ds.variables:
        lons = ds['lon'][:]
    else:
        print("lon variable not found")
        
    if 'time' in ds.variables:
        times = ds['time'][:]
    else:
        print("time variable not found")

    masks = {'AR': (ar > 0.5) if ar is not None else None,
             'TC': (tc > 0.5) if tc is not None else None}
    

    return times, lats, lons, masks


def extract_objects(mask: np.ndarray) -> List[List[Dict]]:
    """Given mask shape (time, lat, lon), return list per time of objects with properties.
    Each object: {'centroid': (lon, lat), 'area': pixels, 'bbox': (minr,minc,maxr,maxc)}
    """
    objects_per_time: List[List[Dict]] = []
    tdim = mask.shape[0]
    print(tdim)
    for t in range(tdim):
        frame = mask[t].astype(np.bool_)
        if frame.sum() == 0:
            objects_per_time.append([])
            continue
        labeled, n = ndimage.label(frame)
        objs = []
        slices = ndimage.find_objects(labeled)
        for lab in range(1, n + 1):
            comp = (labeled == lab)
            area = int(comp.sum())
            # centroid in array coords (row, col)
            cy, cx = ndimage.center_of_mass(comp)
            # bounding box via find_objects
            slc = slices[lab - 1]
            minr, maxr = slc[0].start, slc[0].stop
            minc, maxc = slc[1].start, slc[1].stop
            objs.append({'centroid_px': (cx, cy), 'area': area, 'bbox_px': (minr, minc, maxr, maxc)})
        objects_per_time.append(objs)
    return objects_per_time


def px_to_lonlat(px: Tuple[float, float], lons: np.ndarray, lats: np.ndarray) -> Tuple[float, float]:
    """Convert pixel coords (x=col, y=row) to lon/lat by linear indexing."""
    x, y = px
    # cols -> lon, rows -> lat
    nx = lons.size
    ny = lats.size
    col = np.clip(int(round(x)), 0, nx - 1)
    row = np.clip(int(round(y)), 0, ny - 1)
    return float(lons[col]), float(lats[row])

# https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r


def track_objects(objects_per_time: List[List[Dict]], lons: np.ndarray, lats: np.ndarray,
                  max_search_km: float = 4000.0) -> Dict[int, List[Tuple[float, float, int]]]:
    """Link objects across time using nearest-centroid within max_search_km.
    Returns dict track_id -> list of (lon, lat, time_index)
    """
    tracks: Dict[int, List[Tuple[float, float, int]]] = {}
    next_id = 1
    prev_centroids = []  # list of (track_id, lon, lat)

    for t, objs in enumerate(objects_per_time):
        cur_centroids = []
        for obj in objs:
            lon, lat = px_to_lonlat(obj['centroid_px'], lons, lats)
            cur_centroids.append({'lon': lon, 'lat': lat, 'obj': obj, 'assigned': False})

        # Try to match to prev
        for prev in prev_centroids:
            pid, plong, plat = prev
            # find nearest current
            best = None
            best_d = float('inf')
            for ci, cur in enumerate(cur_centroids):
                if cur['assigned']:
                    continue
                d = haversine(plong, plat, cur['lon'], cur['lat'])
                if d < best_d:
                    best_d = d
                    best = ci
            if best is not None and best_d <= max_search_km:
                cur_centroids[best]['assigned'] = True
                if pid not in tracks:
                    tracks[pid] = []
                tracks[pid].append((cur_centroids[best]['lon'], cur_centroids[best]['lat'], t))

        # Unassigned cur -> new tracks
        for cur in cur_centroids:
            if not cur['assigned']:
                tid = next_id
                next_id += 1
                tracks[tid] = [(cur['lon'], cur['lat'], t)]

        # build prev_centroids for next step
        prev_centroids = []
        for tid, seq in tracks.items():
            # take last entry with time < t+1
            if seq and seq[-1][2] == t:
                prev_centroids.append((tid, seq[-1][0], seq[-1][1]))

    return tracks


def calculate_yaw(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculates the heading (yaw) from point 1 to point 2."""
    lat1_r, lon1_r, lat2_r, lon2_r = map(math.radians, [lat1, lon1, lat2, lon2])
    dLon = lon2_r - lon1_r
    y = math.sin(dLon) * math.cos(lat2_r)
    x = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dLon)
    yaw_rad = math.atan2(y, x)
    return (math.degrees(yaw_rad) + 360) % 360

def lerp(start: float, end: float, t: float) -> float:
    """Linear interpolation between two values."""
    return start + (end - start) * t

def lerp_angle(start_angle: float, end_angle: float, t: float) -> float:
    """Interpolates shortest path between two angles (handles 360 wrap)."""
    diff = (end_angle - start_angle + 180) % 360 - 180
    return (start_angle + diff * t + 360) % 360


def sort_hemispheres_top_left(centers: Dict[int, Tuple[float, float]]) -> List[int]:
    """Sorts tracks: North (starting Top-Left) then South."""
    if not centers: return []
    
    # Split hemispheres
    north = {tid: c for tid, c in centers.items() if c[1] >= 0}
    south = {tid: c for tid, c in centers.items() if c[1] < 0}
    
    def nearest_neighbor_sort(points, start_id):
        if not points: return []
        unvisited = list(points.keys())
        current = start_id
        sorted_ids = [current]
        unvisited.remove(current)
        
        while unvisited:
            best_dist, best_id = float('inf'), None
            lon1, lat1 = points[current]
            for candidate in unvisited:
                lon2, lat2 = points[candidate]
                d = haversine(lon1, lat1, lon2, lat2) 
                if d < best_dist:
                    best_dist, best_id = d, candidate
            
            sorted_ids.append(best_id)
            current = best_id
            
            unvisited.remove(best_id) 
            
        return sorted_ids

    tour = []
    
    # 1. Process North
    if north:
        top_left_id = max(north.keys(), key=lambda k: north[k][1] - north[k][0])
        tour.extend(nearest_neighbor_sort(north, top_left_id))
        
    # 2. Process South
    if south:
        if tour:
            last_lon, last_lat = centers[tour[-1]]
            first_south_id = min(south.keys(), key=lambda k: haversine(last_lon, last_lat, south[k][0], south[k][1]))
        else:
            first_south_id = max(south.keys(), key=lambda k: south[k][1] - south[k][0])
            
        tour.extend(nearest_neighbor_sort(south, first_south_id))
        
    return tour

def generate_hemisphere_tour(tracks: Dict[int, List[Tuple[float, float, int]]], feature_type: str = 'AR') -> List[Dict]:
    """Generates the Map View -> Dive -> Half-Orbit -> Fly sequence."""
    
    # 1. Filter out absolute noise (must exist for at least 3 frames)
    valid_tracks = {tid: seq for tid, seq in tracks.items() if len(seq) >= 3}
    
    # --- Cap the total number of features to visit ---
    MAX_STOPS = 10
    
    # Sort the valid tracks by their lifespan (length of sequence) in descending order
    top_track_ids = sorted(valid_tracks.keys(), key=lambda tid: len(valid_tracks[tid]), reverse=True)[:MAX_STOPS]
    
    # Rebuild valid_tracks with ONLY the Top 10 longest-lasting storms
    valid_tracks = {tid: valid_tracks[tid] for tid in top_track_ids}
    # ----------------------------------------------------------

    # 2. Find the geographic center of the Top 10 tracks
    centers = {tid: (np.mean([pt[0] for pt in seq]), np.mean([pt[1] for pt in seq])) for tid, seq in valid_tracks.items()}
    
    # 3. Sort them spatially for the tour
    tour_order = sort_hemispheres_top_left(centers)
    
    sequence = []
    
 
    Z_LOCAL = 50           
    PITCH_LOCAL = 45.0
    

    ORBIT_RADIUS = 15.0
    Z_GLOBAL = 250        
    PITCH_GLOBAL = 0
    
    # ORBIT_FRAMES = 6       
    # FLIGHT_FRAMES = 3
    # DIVE_FRAMES = 5        
    
    # ... the rest of the Dive and Main Loop remains exactly the same ...
    
    ORBIT_FRAMES =  3      # 5 points is enough for Met.3D to render a smooth semi-circle
    FLIGHT_FRAMES = 2      # Just a midpoint and an endpoint for the transition flight
    DIVE_FRAMES = 4        # 4 frames for the initial zoom from the global map
    
    if not tour_order: return sequence

    # --- STATE 0: THE INTRO DIVE ---
    first_target_lon, first_target_lat = centers[tour_order[0]]
    first_start_lon = first_target_lon + ORBIT_RADIUS * math.cos(0)
    first_start_lat = first_target_lat + ORBIT_RADIUS * math.sin(0)
    first_start_yaw = calculate_yaw(first_start_lon, first_start_lat, first_target_lon, first_target_lat)

    # Start at 0,0, high altitude
    start_global = {'lon': 0.0, 'lat': 0.0, 'z': Z_GLOBAL, 'pitch': PITCH_GLOBAL, 'yaw': 0.0}
    end_dive = {'lon': first_start_lon, 'lat': first_start_lat, 'z': Z_LOCAL, 'pitch': PITCH_LOCAL, 'yaw': first_start_yaw}
    
    for f in range(DIVE_FRAMES):
        t = f / (DIVE_FRAMES - 1)
        sequence.append({
            'lon': lerp(start_global['lon'], end_dive['lon'], t),
            'lat': lerp(start_global['lat'], end_dive['lat'], t),
            'z': lerp(start_global['z'], end_dive['z'], t),
            'pitch': lerp(start_global['pitch'], end_dive['pitch'], t),
            'yaw': lerp_angle(start_global['yaw'], end_dive['yaw'], t)
        })

    # --- MAIN LOOP ---
    for i, tid in enumerate(tour_order):
        target_lon, target_lat = centers[tid]
        
        # --- STATE 1: HALF-ORBIT ---
        orbit_frames = []
        for f in range(ORBIT_FRAMES):
            progress = f / (ORBIT_FRAMES - 1)
            # Only go from 0 to Pi (180 degrees)
            angle_rad = progress * math.pi 
            
            cam_lon = target_lon + ORBIT_RADIUS * math.cos(angle_rad)
            cam_lat = target_lat + ORBIT_RADIUS * math.sin(angle_rad)
            yaw = calculate_yaw(cam_lon, cam_lat, target_lon, target_lat)
            
            orbit_frames.append({'lon': cam_lon, 'lat': cam_lat, 'z': Z_LOCAL, 'pitch': PITCH_LOCAL, 'yaw': yaw})
            
        sequence.extend(orbit_frames)
        
        # --- STATE 2: FLIGHT TO NEXT AR ---
        if i < len(tour_order) - 1:
            next_tid = tour_order[i + 1]
            next_lon, next_lat = centers[next_tid]
            
            # Next orbit ALWAYS starts at angle 0 
            next_start_lon = next_lon + ORBIT_RADIUS * math.cos(0)
            next_start_lat = next_lat + ORBIT_RADIUS * math.sin(0)
            next_start_yaw = calculate_yaw(next_start_lon, next_start_lat, next_lon, next_lat)
            
            start_state = orbit_frames[-1]
            
            for f in range(1, FLIGHT_FRAMES + 1):
                t = f / (FLIGHT_FRAMES + 1)
                sequence.append({
                    'lon': lerp(start_state['lon'], next_start_lon, t),
                    'lat': lerp(start_state['lat'], next_start_lat, t),
                    'z': Z_LOCAL,
                    'pitch': PITCH_LOCAL,
                    'yaw': lerp_angle(start_state['yaw'], next_start_yaw, t)
                })

    return sequence

def write_met3d_xml(sequence: List[Dict], out_path: str,
                    frameTime: int = 10, loop: int = 0, name: str = 'CamGenv3.2',
                    runtime: int = 10, tension: int = 0):
    """Writes a Met.3D CameraSequence XML maintaining the exact frame order."""
    root = ET.Element('CameraSequence', attrib={
        'frameTime': str(frameTime),
        'loop': str(loop),
        'name': name,
        'runtime': str(runtime),
        'tension': str(tension)
    })

    for rec in sequence:
        attrib = {
            'advanceTimestep': '0', # Pausing time while the camera tours the space
            'isOrthographic': '0',
            'label': '',
            'lat': f"{rec['lat']:.6f}",
            'lon': f"{rec['lon']:.6f}",
            'pitch': f"{rec['pitch']:.6f}",
            'roll': '0',
            'transition': '1',
            'yaw': f"{rec['yaw']:.6f}",
            'z': str(rec.get('z', 30))
        }
        ET.SubElement(root, 'SequenceKey', attrib=attrib)

    tree = ET.ElementTree(root)
    tree.write(out_path, encoding='utf-8', xml_declaration=True)

def main(nc_path: str, out_xml: str, feature: str = 'AR'):
    times, lats, lons, masks = read_nc_features(nc_path)
    mask = masks.get(feature.upper())
    
    if mask is None:
        raise RuntimeError(f"Feature {feature} not found in {nc_path}")
        
    objs = extract_objects(mask)
    tracks = track_objects(objs, lons, lats)
    
    # Generate the cinematic sequence directly into a flat list
    tour_sequence = generate_hemisphere_tour(tracks)
    
    # Write the strictly ordered XML
    write_met3d_xml(tour_sequence, out_xml)
    print(f"Wrote cinematic camera sequence to {out_xml} ({len(tour_sequence)} total keyframes across {len(tracks)} tracks)")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python -m Feature_Detection.camera_generator /path/to/file.nc out.xml [AR|TC]')
        sys.exit(1)
    nc = sys.argv[1]
    out = sys.argv[2]
    feat = sys.argv[3] if len(sys.argv) > 3 else 'AR'
    main(nc, out, feat)

# To run paste command (python -m Feature_Detection.camera_generator "C:\G\MASTERS\sem4\ResearchProjectMet3d\NAWDIC_CNN_Features\2026_01_20_Dublin\AR_TC_result.nc" out_sequence.xml AR)
