"""
Generate ground-truth forest/non-forest masks for the preprocessed Harbin
tiles (Haerbing_processed/) using NDVI-based multi-temporal persistence,
then split into training/validation/test.

Method: for each fixed tile position (same tile_id, row, col -- which is
the same geographic patch across all its available acquisition dates,
since Sentinel-2's tiling grid is fixed per tile ID), a pixel is labelled
forest (1) only if its NDVI exceeds FOREST_NDVI_THRESHOLD at EVERY
available date for that position, not just one. This is a temporal
persistence check: single-date NDVI is noisy (cloud edges, phenology,
crop growth spikes can all look "forest-like" in one snapshot), so
requiring high NDVI to hold across the whole June-Sept 2025 window is a
more robust signal than a single-date threshold or the original paper's
single-date k-means clustering.

Known limitation (documented in poster_notes.md): this persistence check
identifies land that is consistently well-vegetated across the window,
which will also include some non-forest perennial vegetation (stable
grassland, unharvested-within-window cropland). This is a disclosed proxy
label, not a verified ground truth -- same category of limitation the
original paper's own k-means masks had.

A second, more specific limitation found by visual QC: Heilongjiang
farmland is criss-crossed with narrow tree windbreaks/shelterbelts
planted along roads and field edges (part of China's Three-North Shelter
Forest Program), which genuinely have high, persistent NDVI and so get
correctly labelled "forest" by the rule above -- but they read as thin
linear streaks tracing roads, not blocky forest cover, and are a
land-cover class the original Amazon dataset never had to deal with. A
morphological opening (erosion + dilation) removes any connected region
narrower than SHELTERBELT_FILTER_RADIUS pixels, keeping larger, blockier
forest cover while dropping the thin linear shelterbelt strands.

Model input for each position is the LATEST available date's image
(freshest observed state); the persistence mask across all dates is the
label. Train/val/test split is done by (tile_id, row, col) position via a
deterministic hash, so all dates of a given position land in the same
split -- this avoids leaking near-identical same-location, different-date
tiles across splits.

Run with: venv/bin/python3 generate-haerbing-ground-truth.py
"""
import os
import json
import hashlib
from collections import defaultdict

import numpy as np
from skimage.morphology import binary_opening, disk

PROCESSED_DIR = "Haerbing_processed"
OUTPUT_DIR = "Haerbing_ground_truth"
FOREST_NDVI_THRESHOLD = 0.6
SPLIT_BOUNDARIES = {"training": 70, "validation": 85, "test": 100}  # cumulative %

# Removes connected forest regions narrower than ~2*radius+1 pixels (10m/px,
# so radius=2 drops anything narrower than ~50m) -- shelterbelt rows are
# typically only 1-3 trees wide (~10-30m), well within that; contiguous
# forest blocks many pixels wide survive largely intact.
APPLY_SHELTERBELT_FILTER = True
SHELTERBELT_FILTER_RADIUS = 2


def load_manifest(processed_dir):
    with open(os.path.join(processed_dir, "manifest.json")) as f:
        return json.load(f)


def assign_split(tile_id, row, col):
    key = f"{tile_id}_{row}_{col}".encode()
    bucket = int(hashlib.md5(key).hexdigest(), 16) % 100
    for split, upper in SPLIT_BOUNDARIES.items():
        if bucket < upper:
            return split
    return "test"


def group_positions(manifest):
    positions = defaultdict(list)
    for record in manifest:
        if not record["kept"]:
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
    print(f"\nDone. Manifest written to {os.path.join(OUTPUT_DIR, 'manifest.json')}")


if __name__ == "__main__":
    generate()
