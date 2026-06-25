"""Camera goes from centre to northern hemisphere and then southern hemisphere and just scans the ARs."""

import sys
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple

import numpy as np
from netCDF4 import Dataset
from scipy import ndimage

def read_nc_features(nc_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reads NetCDF and returns coordinates and masks."""
    ds = Dataset(nc_path)
    ar = ds['AR'][:]
    tc = ds['TC'][:]

    lats = ds['lat'][:] if 'lat' in ds.variables else np.array([])
    lons = ds['lon'][:] if 'lon' in ds.variables else np.array([])
    times = ds['time'][:] if 'time' in ds.variables else np.array([])

    masks = {
        'AR': (ar > 0.5) if ar is not None else None,
        'TC': (tc > 0.5) if tc is not None else None
    }
    return times, lats, lons, masks


def extract_objects(mask: np.ndarray) -> List[List[Dict]]:
    """Identifies individual weather features in each time frame."""
    objects_per_time = []
    tdim = mask.shape[0]
    for t in range(tdim):
        frame = mask[t].astype(np.bool_)
        if frame.sum() == 0:
            objects_per_time.append([])
            continue
            
        labeled, n = ndimage.label(frame)
        objs = []
        for lab in range(1, n + 1):
            comp = (labeled == lab)
            cy, cx = ndimage.center_of_mass(comp)
            objs.append({'centroid_px': (cx, cy)})
        objects_per_time.append(objs)
    return objects_per_time


def px_to_lonlat(px: Tuple[float, float], lons: np.ndarray, lats: np.ndarray) -> Tuple[float, float]:
    """Converts array pixel indices to geographic Longitude/Latitude."""
    x, y = px
    col = np.clip(int(round(x)), 0, lons.size - 1)
    row = np.clip(int(round(y)), 0, lats.size - 1)
    return float(lons[col]), float(lats[row])


def track_objects(objects_per_time: List[List[Dict]], lons: np.ndarray, lats: np.ndarray) -> Dict[int, List[Tuple[float, float, int]]]:
    """Links objects across time using simple proximity mapping."""
    tracks = {}
    next_id = 1
    prev_centroids = []

    for t, objs in enumerate(objects_per_time):
        cur_centroids = [{'lon': px_to_lonlat(obj['centroid_px'], lons, lats)[0],
                          'lat': px_to_lonlat(obj['centroid_px'], lons, lats)[1],
                          'assigned': False} for obj in objs]

        for pid, plong, plat in prev_centroids:
            best, best_d = None, float('inf')
            for ci, cur in enumerate(cur_centroids):
                if not cur['assigned']:
                    # Simple Pythagorean distance for tracking (sufficient for basic linkage)
                    d = (plong - cur['lon'])**2 + (plat - cur['lat'])**2 
                    if d < best_d:
                        best_d, best = d, ci
            
            if best is not None and best_d <= 400.0: # Rough bounding box threshold
                cur_centroids[best]['assigned'] = True
                tracks[pid].append((cur_centroids[best]['lon'], cur_centroids[best]['lat'], t))

        for cur in cur_centroids:
            if not cur['assigned']:
                tracks[next_id] = [(cur['lon'], cur['lat'], t)]
                next_id += 1

        prev_centroids = [(tid, seq[-1][0], seq[-1][1]) for tid, seq in tracks.items() if seq and seq[-1][2] == t]

    return tracks



def generate_structured_tour(tracks: Dict[int, List[Tuple[float, float, int]]]) -> List[Dict]:
    """
    Implements the discrete hemispheric scanning logic with corrected focal offsets.
    """
    valid_tracks = {tid: seq for tid, seq in tracks.items() if len(seq) >= 3}
    
    centers = {tid: (np.mean([pt[0] for pt in seq]), np.mean([pt[1] for pt in seq])) 
               for tid, seq in valid_tracks.items()}

    north_ars = [(tid, centers[tid]) for tid in centers if centers[tid][1] >= 0]
    south_ars = [(tid, centers[tid]) for tid in centers if centers[tid][1] < 0]

    north_ars.sort(key=lambda item: item[1][0])
    south_ars.sort(key=lambda item: item[1][0], reverse=True)

    sequence = []
    
    # Constants
    Z_MAP = 250.0
    PITCH_MAP = 0.0
    Z_TRACK = 50.0
    PITCH_TRACK = 44.0
    
    # --- THE FIX: Updated Pullback Offset ---
    # Based on your manual tests, 44.0 degrees perfectly centers the AR 
    # when the camera is at Z=50 and Pitch=44.
    OFFSET = 44.0 

    def clamp_lat(lat: float) -> float:
        """Prevents the camera from exceeding the North/South poles."""
        return max(-89.9, min(89.9, lat))

    # --- START: GLOBAL MAP ---
    sequence.append({'lat': 0.0, 'lon': 0.0, 'z': Z_MAP, 'pitch': PITCH_MAP, 'yaw': 0.0})

    # --- PHASE 1: NORTHERN HEMISPHERE ---
    if north_ars:
        first_id, (first_lon, first_lat) = north_ars[0]
        
        # Keyframe A: Camera at South of AR, looking North
        sequence.append({'lat': clamp_lat(first_lat - OFFSET), 'lon': first_lon, 'z': Z_TRACK, 'pitch': PITCH_TRACK, 'yaw': 0.0})
        # Keyframe B: Camera at West of AR, looking East
        sequence.append({'lat': first_lat, 'lon': first_lon - OFFSET, 'z': Z_TRACK, 'pitch': PITCH_TRACK, 'yaw': 90.0})
        # Keyframe C: Camera at North of AR, looking South
        sequence.append({'lat': clamp_lat(first_lat + OFFSET), 'lon': first_lon, 'z': Z_TRACK, 'pitch': PITCH_TRACK, 'yaw': -180.0})

        # Remaining Northern ARs: Keep going East, camera at North pointing South
        for tid, (lon, lat) in north_ars[1:]:
            sequence.append({'lat': clamp_lat(lat + OFFSET), 'lon': lon, 'z': Z_TRACK, 'pitch': PITCH_TRACK, 'yaw': -180.0})

    # --- PHASE 2: SOUTHERN HEMISPHERE ---
    if south_ars:
        # Scan moving West, camera at South pointing North
        for tid, (lon, lat) in south_ars:
            sequence.append({'lat': clamp_lat(lat - OFFSET), 'lon': lon, 'z': Z_TRACK, 'pitch': PITCH_TRACK, 'yaw': 0.0})

    # --- END: GLOBAL MAP ---
    sequence.append({'lat': 0.0, 'lon': 0.0, 'z': Z_MAP, 'pitch': PITCH_MAP, 'yaw': 0.0})

    return sequence

# --- 3. XML EXPORT ---
def write_met3d_xml(sequence: List[Dict], out_path: str):
    root = ET.Element('CameraSequence', attrib={
        'frameTime': '10',
        'loop': '0',
        'name': 'Sequence_samLatfix',
        'runtime': '35',
        'tension': '1.00'
    })

    for rec in sequence:
        attrib = {
            'advanceTimestep': '1',
            'isOrthographic': '0',
            'label': '',
            'lat': f"{rec['lat']:.6f}",
            'lon': f"{rec['lon']:.6f}",
            'pitch': f"{rec['pitch']:.6f}",
            'roll': '0',
            'transition': '1',
            'yaw': f"{rec['yaw']:.6f}",
            'z': f"{rec['z']:.6f}"
        }
        ET.SubElement(root, 'SequenceKey', attrib=attrib)

    # Use built-in ElementTree indentation for clean formatting
    tree = ET.ElementTree(root)
    if hasattr(ET, 'indent'):  # Available in Python 3.9+
        ET.indent(tree, space="  ", level=0)
        
    tree.write(out_path, encoding='utf-8', xml_declaration=True)


def main(nc_path: str, out_xml: str, feature: str = 'AR'):
    print(f"Reading {feature} tracks from {nc_path}...")
    times, lats, lons, masks = read_nc_features(nc_path)
    mask = masks.get(feature.upper())
    
    if mask is None:
        raise RuntimeError(f"Feature {feature} not found in {nc_path}")
        
    objs = extract_objects(mask)
    tracks = track_objects(objs, lons, lats)
    
    print("Generating discrete scanning sequence...")
    tour_sequence = generate_structured_tour(tracks)
    
    write_met3d_xml(tour_sequence, out_xml)
    print(f"Success! Wrote exactly {len(tour_sequence)} cinematic keyframes to {out_xml}.")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python camera_generator.py /path/to/file.nc out.xml [AR|TC]')
        sys.exit(1)
    
    nc = sys.argv[1]
    out = sys.argv[2]
    feat = sys.argv[3] if len(sys.argv) > 3 else 'AR'
    main(nc, out, feat)