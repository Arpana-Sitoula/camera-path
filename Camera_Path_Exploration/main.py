from extractor import extract_keyframes
from visualizer import save_camera_map
from exporter import export_to_xml
import json, os


NC_FILES_DIR = "../Feature_detection/Results/TC-AR-Met3d"   # where .nc files are
OUTPUT_DIR    = "./Outputs"                 # where to save results
TIMESTEPS     = range(0, 12)               # which timesteps to process


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_keyframes = []

    for t in TIMESTEPS:
        nc_path = f"{NC_FILES_DIR}/{t}.nc"

        if not os.path.exists(nc_path):
            print(f"[timestep {t}] File not found, skipping.")
            continue

        keyframes = extract_keyframes(nc_path)
        all_keyframes.append({"timestep": t, "camera_path": keyframes})

        save_camera_map(nc_path, keyframes, timestep=t, output_dir=OUTPUT_DIR)

        print(f"[timestep {t}] Done — {len(keyframes)} keyframes extracted.")

    # save all keyframes to JSON for later Met3D XML export
    out_path = f"{OUTPUT_DIR}/keyframes_2d.json"

    with open(out_path, "w") as f:
        json.dump(all_keyframes, f, indent=2)

    print(f"\nAll done. Keyframes saved to {out_path}")

    export_to_xml(
    keyframes_json_path=f"{OUTPUT_DIR}/keyframes_2d.json",
    output_dir=f"{OUTPUT_DIR}/xml"
)

if __name__ == "__main__":
    main()