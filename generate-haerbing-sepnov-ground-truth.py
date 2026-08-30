"""
Generate ground-truth forest/non-forest masks for the preprocessed
Sept-Nov 2025 Harbin tiles (Haerbing_processed_sepnov/), using the same
NDVI-based multi-temporal persistence method as
generate-haerbing-ground-truth.py (the June-Aug dataset), then split into
training/validation/test.

Kept as its own script rather than a parameterised version of the
June-Aug one, same reasoning as everywhere else in this project: separate
files for separate dataset variants so a run can't be mixed up or
re-generate the wrong one by accident (see poster_notes.md section 8).

Only the FILTERED (shelterbelt-opening) variant is generated here, not an
unfiltered counterpart. The Jun-Aug ablation already showed the filtered
labels produce a significantly better model (Wilcoxon p=5.374e-10, see
code_phase2_harbin_comparison.ipynb) -- given the time available before
the poster deadline, that question doesn't need re-asking for a second
season. This script exists to test a different question: does moving the
acquisition window to autumn (crop senescence/harvest vs. persistently
green natural forest) change model performance under the *same*, already
best-performing labelling method? See code_phase2_harbin_comparison_
sepnov.ipynb for that comparison.

IMPORTANT -- two things to verify before trusting this script's output,
both flagged as open questions rather than settled the way they were for
June-Aug:

  1. EXCLUDED_TILE_IDS below is empty by default. The June-Aug run
     excluded T52TDT because it turned out to have only one acquisition
     date (see that script's docstring). Whether the same is true for
     this Sept-Nov window is unknown until you've run
     preprocess-haerbing-sepnov-sentinel2-data.py and read its printed
     "distinct kept dates per tile" summary -- add any single-date tile's
     ID to EXCLUDED_TILE_IDS below if so, mirroring the June-Aug decision.

  2. FOREST_NDVI_THRESHOLD=0.6 was chosen by inspecting June-Aug
     (peak-growing-season) NDVI. Autumn NDVI is systematically lower for
     deciduous/mixed forest and especially for senescing cropland, so 0.6
     may mislabel real forest as non-forest here. Run
     show_image_mask_pairs() (from code_phase2_harbin_sepnov.ipynb) on a
     sample of this output before training -- if masks look sparse
     relative to visibly green regions in the image, lower the threshold
     and re-run.

Run with: venv/bin/python3 generate-haerbing-sepnov-ground-truth.py
"""
import os
import json
import hashlib
from collections import defaultdict

import numpy as np
from skimage.morphology import binary_opening, disk

PROCESSED_DIR = "Haerbing_processed_sepnov"
OUTPUT_DIR = "Haerbing_ground_truth_sepnov"
FOREST_NDVI_THRESHOLD = 0.6
SPLIT_BOUNDARIES = {"training": 70, "validation": 85, "test": 100}  # cumulative %

# Empty by default -- see docstring point 1. Fill in after checking
# preprocess-haerbing-sepnov-sentinel2-data.py's printed per-tile date
# counts, e.g. EXCLUDED_TILE_IDS = {"T52TDT"} if it's single-date again.
EXCLUDED_TILE_IDS = set()

# Same shelterbelt-removal filter as the June-Aug filtered dataset (see
# generate-haerbing-ground-truth.py for the full rationale) -- always on
# here, there is no unfiltered variant of this script.
APPLY_SHELTERBELT_FILTER = True
SHELTERBELT_FILTER_RADIUS = 2


def load_manifest(processed_dir):
    with open(os.path.join(processed_dir, "manifest.json")) as f:
        return json.load(f)


def assign_split(tile_id, row, col):
    # Identical hash function to generate-haerbing-ground-truth.py's
    # assign_split -- but this dataset's positions are a disjoint set from
    # the June-Aug one (different acquisition dates go into the same
    # PROCESSED_DIR/tile_id/date/... structure, not merged), so this is
    # its own independent 70/15/15 split, not a shared one.
    key = f"{tile_id}_{row}_{col}".encode()
    bucket = int(hashlib.md5(key).hexdigest(), 16) % 100
    for split, upper in SPLIT_BOUNDARIES.items():
        if bucket < upper:
            return split
    return "test"


def group_positions(manifest):
    positions = defaultdict(list)
    for record in manifest:
        if not record["kept"] or record["tile_id"] in EXCLUDED_TILE_IDS:
            continue
        key = (record["tile_id"], record["row"], record["col"])
        positions[key].append(record["date"])
    return positions


def generate():
    manifest = load_manifest(PROCESSED_DIR)
    positions = group_positions(manifest)
    print(f"{len(positions)} distinct tile positions with at least one clear-sky date")

    for split in SPLIT_BOUNDARIES:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "masks"), exist_ok=True)

    gt_manifest = []
    split_counts = defaultdict(int)
    forest_fractions = []

    for (tile_id, row, col), dates in positions.items():
        dates_sorted = sorted(dates)
        ndvi_stack = []
        for date in dates_sorted:
            base = os.path.join(PROCESSED_DIR, tile_id, date, f"row{row:02d}_col{col:02d}")
            ndvi_stack.append(np.load(f"{base}_ndvi.npy"))
        ndvi_stack = np.stack(ndvi_stack, axis=0)  # (n_dates, 512, 512)

        forest_mask = np.all(ndvi_stack > FOREST_NDVI_THRESHOLD, axis=0)

        if APPLY_SHELTERBELT_FILTER:
            forest_mask = binary_opening(forest_mask, disk(SHELTERBELT_FILTER_RADIUS))

        forest_fraction = float(forest_mask.mean())
        forest_fractions.append(forest_fraction)

        latest_date = dates_sorted[-1]
        image_path = os.path.join(PROCESSED_DIR, tile_id, latest_date, f"row{row:02d}_col{col:02d}_image.npy")
        image = np.load(image_path)

        split = assign_split(tile_id, row, col)
        split_counts[split] += 1

        out_name = f"{tile_id}_row{row:02d}_col{col:02d}"
        np.save(os.path.join(OUTPUT_DIR, split, "images", f"{out_name}.npy"), image.astype(np.float32))
        np.save(
            os.path.join(OUTPUT_DIR, split, "masks", f"{out_name}.npy"),
            forest_mask.astype(np.uint8).reshape(512, 512, 1),
        )

        gt_manifest.append({
            "tile_id": tile_id, "row": row, "col": col,
            "n_dates_used": len(dates_sorted), "latest_date": latest_date,
            "split": split, "forest_fraction": forest_fraction,
        })

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump(gt_manifest, f, indent=1)

    print("\nSplit counts:", dict(split_counts))
    print(f"Mean forest fraction across positions: {np.mean(forest_fractions):.3f}")
    print(f"Positions with >=2 dates (persistence check meaningful): "
          f"{sum(1 for r in gt_manifest if r['n_dates_used'] >= 2)}/{len(gt_manifest)}")
    if np.mean(forest_fractions) < 0.05:
        print("\nWARNING: mean forest fraction is very low -- see docstring point 2 "
              "(FOREST_NDVI_THRESHOLD may need lowering for autumn NDVI).")
    print(f"\nDone. Manifest written to {os.path.join(OUTPUT_DIR, 'manifest.json')}")


if __name__ == "__main__":
    generate()
