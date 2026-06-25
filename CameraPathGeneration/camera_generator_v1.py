import sys
import math
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple

import numpy as np
from netCDF4 import Dataset
from scipy import ndimage
from scipy.interpolate import CubicSpline


def read_nc_features(nc_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (times, lats, lons, dict of masks) where masks['AR'] and masks['TC'] are boolean arrays
    with shape (time, lat, lon).
    """
    ds = Dataset(nc_path)

    ar = ds['AR'][:]
    tc = ds['TC'][:]

    # Attempt to find lat/lon or fall back to generated grid used elsewhere (1152x768)
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


def smooth_track(track: List[Tuple[float, float, int]], n_samples: int = 20) -> List[Tuple[float, float, float]]:
    """Interpolate and smooth a single track (lon,lat,time) -> list of (t_frac, lon, lat) sampled uniformly.
    Returns times normalized 0..1 and coordinates.
    """
    if len(track) == 0:
        return []
    times = np.array([t for (_, _, t) in track], dtype=float)
    lons = np.array([lon for (lon, _, _) in track], dtype=float)
    lats = np.array([lat for (_, lat, _) in track], dtype=float)
    # unwrap lon to avoid dateline jumps
    lons_un = np.unwrap(np.radians(lons))
    lons_un = np.degrees(lons_un)

    if len(times) == 1:
        return [(0.0, float(lons[0]), float(lats[0]))]

    cs_lon = CubicSpline(times, lons_un, bc_type='natural')
    cs_lat = CubicSpline(times, lats, bc_type='natural')
    ts = np.linspace(times[0], times[-1], max(n_samples, len(times)))
    out = []
    for t in ts:
        lon = float(cs_lon(t))
        # rewrap to -180..180
        lon = ((lon + 180) % 360) - 180
        lat = float(cs_lat(t))
        # normalized time fraction
        tfrac = (t - times[0]) / (times[-1] - times[0]) if times[-1] != times[0] else 0.0
        out.append((tfrac, lon, lat))
    return out


def choose_focus_track(tracks: Dict[int, List[Tuple[float, float, int]]]) -> Tuple[int, List[Tuple[float, float, int]]]:
    """Pick one representative object track to focus the camera on."""
    if not tracks:
        raise RuntimeError("No feature tracks found to build a camera path")

    def track_score(item):
        tid, seq = item
        duration = seq[-1][2] - seq[0][2] if len(seq) > 1 else 0
        return (len(seq), duration, -tid)

    return max(tracks.items(), key=track_score)


def generate_camera_keyframes(tracks: Dict[int, List[Tuple[float, float, int]]], feature_type: str = 'AR') -> Dict[int, List[Dict]]:
    """Produce a sequence for one feature: world view, then rotating close view."""
    tid, seq = choose_focus_track(tracks)
    sm = smooth_track(seq, n_samples=30)

    keyframes: Dict[int, List[Dict]] = {}
    kf = [{
        't': -1.0,
        'lon': 0.0,
        'lat': 0.0,
        'z': 250,
        'pitch': 0.0,
        'yaw': 0.0,
        'roll': 0.0,
    }]

    if len(sm) == 1:
        sm = [(0.0, sm[0][1], sm[0][2]), (1.0, sm[0][1], sm[0][2])]

    for idx, (tfrac, lon, lat) in enumerate(sm):
        # z around 30-50 gives a tighter close view once the camera focuses on the AR.
        z = 50 - int(round(20 * tfrac))
        pitch = 50.0
        yaw = (idx * 18.0) % 360.0
        kf.append({
            't': tfrac,
            'lon': lon,
            'lat': lat,
            'z': z,
            'pitch': pitch,
            'yaw': yaw,
            'roll': 0.0,
        })

    keyframes[tid] = kf
    return keyframes


def write_met3d_xml(keyframes: Dict[int, List[Dict]], out_path: str,
                    frameTime: int = 10, loop: int = 0, name: str = 'AutoSequence',
                    runtime: int = 10, tension: int = 0):
    """Write a simple Met.3D CameraSequence XML using SequenceKey entries.

    Each keyframe becomes a `SequenceKey` element with attributes similar to the
    example `TestSequence.xml` the user provided.
    """
    root = ET.Element('CameraSequence', attrib={
        'frameTime': str(frameTime),
        'loop': str(loop),
        'name': name,
        'runtime': str(runtime),
        'tension': str(tension)
    })

    # Flatten all tracks into a single sequence by ordering by track id then by t
    entries = []
    for tid, kf_list in keyframes.items():
        for rec in kf_list:
            entries.append((tid, rec))
    # sort by track id then time fraction
    entries.sort(key=lambda x: (x[0], x[1]['t']))

    for tid, rec in entries:
        # map our fields to Met.3D attributes
        lon = rec['lon']
        lat = rec['lat']
        zval = int(rec.get('z', 30))
        pitch = rec.get('pitch', rec.get('tilt', 0.0))
        yaw = rec.get('yaw', 0.0)
        roll = rec.get('roll', 0.0)
        attrib = {
            'advanceTimestep': '0',
            'isOrthographic': '1',
            'label': '',
            'lat': f"{lat:.6f}",
            'lon': f"{lon:.6f}",
            'pitch': f"{pitch:.6f}",
            'roll': f"{roll:.6f}",
            'transition': '1',
            'yaw': f"{yaw:.6f}",    
            'z': str(zval)
        }
        ET.SubElement(root, 'SequenceKey', attrib=attrib)
        ET.indent(root, space="  ", level=0)

    tree = ET.ElementTree(root)
    tree.write(out_path, encoding='utf-8', xml_declaration=True)


def main(nc_path: str, out_xml: str, feature: str = 'AR'):
    times, lats, lons, masks = read_nc_features(nc_path)
    mask = masks.get(feature.upper())
    if mask is None:
        raise RuntimeError(f"Feature {feature} not found in {nc_path}")
    objs = extract_objects(mask)
    tracks = track_objects(objs, lons, lats)
    keyframes = generate_camera_keyframes(tracks, feature_type=feature)
    # write Met.3D-style XML
    write_met3d_xml(keyframes, out_xml)
    print(f"Wrote camera sequence to {out_xml} ({len(keyframes)} tracks)")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python -m Feature_Detection.camera_generator /path/to/file.nc out.xml [AR|TC]')
        
        sys.exit(1)
    nc = sys.argv[1]
    out = sys.argv[2]
    feat = sys.argv[3] if len(sys.argv) > 3 else 'AR'
    main(nc, out, feat)

# To run paste command (python -m camera_generator_v1 "C:\G\MASTERS\sem4\ResearchProjectMet3d\NAWDIC_CNN_Features\2026_01_20_Dublin\AR_TC_result.nc" out_sequence.xml AR)
