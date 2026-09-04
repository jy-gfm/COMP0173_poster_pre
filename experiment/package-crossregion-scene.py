"""
Package a small sample of the two cross-region check sites (Shangri-La,
Yunnan; Horqin Sandy Land, Liaoning) for upload to Drive and inference in
Colab against the already-trained Harbin model -- no ground truth, no
retraining, a visual generalization check only (same spirit as the
T52TDT unlabeled-scene check).

Run with: venv/bin/python3 package-crossregion-scene.py
"""
import os
import glob
import shutil

SOURCE_DIR = "CrossRegion_processed"
OUTPUT_DIR = "CrossRegion_scene"
N_SAMPLE = 12

REGIONS = {
    "T47RNL": "Shangri-La (Yunnan)",
    "T51TVH": "Horqin Sandy Land (Liaoning)",
}


def package():
    for tile, label in REGIONS.items():
        date_dirs = sorted(glob.glob(f"{SOURCE_DIR}/{tile}/*"))
        image_paths = []
        for d in date_dirs:
            image_paths.extend(sorted(glob.glob(f"{d}/*_image.npy")))
        print(f"{tile} ({label}): {len(image_paths)} total tiles available")

        step = max(len(image_paths) // N_SAMPLE, 1)
        sampled = image_paths[::step][:N_SAMPLE]
        print(f"  sampling {len(sampled)} tiles")

        out_dir = os.path.join(OUTPUT_DIR, tile)
        os.makedirs(out_dir, exist_ok=True)
        for image_path in sampled:
            base = os.path.basename(image_path).replace("_image.npy", "")
            date = image_path.split("/")[-2]
            ndvi_path = image_path.replace("_image.npy", "_ndvi.npy")
            shutil.copy(image_path, os.path.join(out_dir, f"{date}_{base}_image.npy"))
            shutil.copy(ndvi_path, os.path.join(out_dir, f"{date}_{base}_ndvi.npy"))

    print(f"\nDone. Tiles written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    package()
