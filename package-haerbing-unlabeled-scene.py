"""
Package a small sample of tile T52TDT's raw preprocessed imagery for a
purely visual "apply the model to genuinely unlabeled data" check.

T52TDT is excluded from EVERY ground-truth generation script in this
project (generate-haerbing-ground-truth.py, generate-haerbing-
multitemporal-ground-truth.py) because it only has one acquisition date
(20250618) -- the NDVI-persistence check needs >=2 dates to be meaningful,
so this tile never had a forest_mask computed at all, and none of its 324
positions appear in any training/validation/test split of any dataset
variant. That makes it a genuinely unseen, never-labelled scene from the
same broader Sentinel-2 dataset -- unlike the test splits used elsewhere,
which are unseen by a given model but still have a generated label.

This is NOT a metrics experiment: there is no ground truth to score
against here (that's the point). It's for visually eyeballing whether the
model's predictions look plausible on a completely fresh scene, the way
you'd actually use the model in deployment on new imagery.

Only a small sample of tiles is packaged (not all 324) since this is for
visual inspection, not a statistical comparison -- keeping the tar small
means a fast Drive upload.

Run with: venv/bin/python3 package-haerbing-unlabeled-scene.py
"""
import os
import glob
import shutil

PROCESSED_DIR = "Haerbing_processed/T52TDT/20250618"
OUTPUT_DIR = "Haerbing_unlabeled_scene"
N_SAMPLE = 12


def package():
    image_paths = sorted(glob.glob(f"{PROCESSED_DIR}/*_image.npy"))
    print(f"{len(image_paths)} total tiles available in {PROCESSED_DIR}")

    # Evenly spaced sample across the tile grid (not just the first N),
    # so the sample covers different parts of the scene rather than one corner.
    step = max(len(image_paths) // N_SAMPLE, 1)
    sampled = image_paths[::step][:N_SAMPLE]
    print(f"Sampling {len(sampled)} tiles for visual inspection")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for image_path in sampled:
        base = os.path.basename(image_path).replace("_image.npy", "")
        ndvi_path = image_path.replace("_image.npy", "_ndvi.npy")
        shutil.copy(image_path, os.path.join(OUTPUT_DIR, f"{base}_image.npy"))
        # NDVI is copied for visual reference only (a quick eyeball of
        # "does this look plausibly forest-like"), NOT as a training label
        # or ground truth -- there is no generated mask for this tile.
        shutil.copy(ndvi_path, os.path.join(OUTPUT_DIR, f"{base}_ndvi.npy"))

    print(f"\nDone. {len(sampled)} tiles written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    package()
