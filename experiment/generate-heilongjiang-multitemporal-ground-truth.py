"""
Generate ground-truth forest/non-forest masks + multi-temporal (bi-date)
input images for the preprocessed Harbin tiles (Haerbing_processed/),
extending generate-haerbing-ground-truth.py's NDVI persistence method.

Kept as its own script (not a toggle on generate-haerbing-ground-truth.py),
same convention as elsewhere in this project: this produces a genuinely
different model INPUT SHAPE (8 channels, not 4), so it needs its own
output directory and its own downstream training notebook
(code_phase2_harbin_multitemporal.ipynb), not something that should be
silently mixed up with the single-date variants.

Why this exists (lecturer feedback on the architecture-adaptation
requirement): the single-date variants (filtered/unfiltered) label each
position using NDVI persistence across ALL available dates, but only
ever show the MODEL one date's image (the latest). This script instead
stacks TWO dates' 4-band images (earliest + latest available) into a
single 8-channel model input, so the network itself can see change
across time directly -- not just benefit from a temporally-informed label
it never actually observes as input. This is a genuine architectural
adaptation (the first layer's input shape changes: 4 -> 8 channels), and
unlike the code_phase2_harbin_filtered_archfix.ipynb bug fix, it IS
specifically motivated by this project's new-context data (having
multiple 2025 acquisition dates to draw on), rather than being a
universal fix that would apply unchanged to the original single-date
Amazon setup.

Only positions with >=2 available dates can be used (a bi-temporal input
needs two distinct dates) -- this drops any position that would otherwise
fall back to a single-date NDVI threshold, which is a stricter, cleaner
subset of the already-filtered dataset (see generate-haerbing-ground-truth.py's
own docstring for why T52TDT was already excluded for this exact reason).

Uses the SAME NDVI persistence labelling as the filtered variant (all
available dates for a position, not just the two used for input) and the
same shelterbelt morphological filter -- only the model INPUT
construction differs. Run generate-haerbing-ground-truth.py first if
Haerbing_processed/manifest.json isn't already present (this script reads
the same manifest, it does not re-run preprocessing).

Run with: venv/bin/python3 generate-haerbing-multitemporal-ground-truth.py
"""
import os
import json
import hashlib
from collections import defaultdict

import numpy as np
from skimage.morphology import binary_opening, disk

PROCESSED_DIR = "Haerbing_processed"
OUTPUT_DIR = "Haerbing_ground_truth_multitemporal"
FOREST_NDVI_THRESHOLD = 0.6
SPLIT_BOUNDARIES = {"training": 70, "validation": 85, "test": 100}  # cumulative %

# Same exclusion as the filtered/unfiltered variants -- see
# generate-haerbing-ground-truth.py's docstring.
EXCLUDED_TILE_IDS = {"T52TDT"}

# Same shelterbelt-removal filter as the filtered variant -- always on
# here, this script only produces the multi-temporal-input analogue of
# the *filtered* dataset (the better-performing labelling method), not a
# second unfiltered variant.
APPLY_SHELTERBELT_FILTER = True
SHELTERBELT_FILTER_RADIUS = 2

# A bi-temporal input needs exactly two distinct dates per position.
N_DATES_STACKED = 2


def load_manifest(processed_dir):
    with open(os.path.join(processed_dir, "manifest.json")) as f:
        return json.load(f)


def assign_split(tile_id, row, col):
    # Identical to generate-haerbing-ground-truth.py's assign_split, so a
    # given position lands in the same split as it would in the
    # single-date filtered dataset -- keeps splits comparable where
    # positions overlap between the two datasets.
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
    all_positions = group_positions(manifest)
    # Bi-temporal input needs >=2 dates -- drop anything short of that
    # rather than falling back to a weaker single-date input.
    positions = {k: v for k, v in all_positions.items() if len(v) >= N_DATES_STACKED}
    print(f"{len(all_positions)} distinct tile positions with at least one clear-sky date")
    print(f"{len(positions)} positions have >= {N_DATES_STACKED} dates -- "
          f"only these can be used for a multi-temporal input "
          f"({len(all_positions) - len(positions)} dropped)")

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

        # Label uses the persistence check across ALL available dates for
        # this position, same as the single-date filtered dataset -- only
        # the model INPUT (below) is restricted to two dates.
        forest_mask = np.all(ndvi_stack > FOREST_NDVI_THRESHOLD, axis=0)

        if APPLY_SHELTERBELT_FILTER:
            forest_mask = binary_opening(forest_mask, disk(SHELTERBELT_FILTER_RADIUS))

        forest_fraction = float(forest_mask.mean())
        forest_fractions.append(forest_fraction)

        # Bi-temporal input: earliest + latest available date, each a
        # 4-band (R,G,B,NIR) image, channel-stacked -> (512, 512, 8). This
        # is what actually changes the model's input shape -- and with it,
        # what the first conv layer's kernels operate over (a genuine
        # multi-temporal signal, not just a single snapshot).
        stack_dates = [dates_sorted[0], dates_sorted[-1]]
        images = []
        for date in stack_dates:
            image_path = os.path.join(PROCESSED_DIR, tile_id, date, f"row{row:02d}_col{col:02d}_image.npy")
            images.append(np.load(image_path))
        image = np.concatenate(images, axis=-1)  # (512, 512, 8)

        split = assign_split(tile_id, row, col)
        split_counts[split] += 1

        out_name = f"{tile_id}_row{row:02d}_col{col:02d}"
        # float16, not float32 -- halves the tar size (10GB -> ~5GB) for a
        # faster Drive upload. Reflectance values here already came from
        # 16-bit integer DNs divided by 10000, so float16's ~3-decimal-digit
        # precision loses essentially no meaningful signal at this
        # magnitude (~0.03-0.15 typical land reflectance). Cast back to
        # float32 on load in the training notebook -- Keras expects float32
        # input, so this is a storage/transfer optimisation only, not a
        # training-precision change.
        np.save(os.path.join(OUTPUT_DIR, split, "images", f"{out_name}.npy"), image.astype(np.float16))
        np.save(
            os.path.join(OUTPUT_DIR, split, "masks", f"{out_name}.npy"),
            forest_mask.astype(np.uint8).reshape(512, 512, 1),
        )

        gt_manifest.append({
            "tile_id": tile_id, "row": row, "col": col,
            "n_dates_used_for_label": len(dates_sorted),
            "dates_stacked_for_input": stack_dates,
            "split": split, "forest_fraction": forest_fraction,
        })

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump(gt_manifest, f, indent=1)

    print("\nSplit counts:", dict(split_counts))
    print(f"Mean forest fraction across positions: {np.mean(forest_fractions):.3f}")
    print(f"\nDone. Manifest written to {os.path.join(OUTPUT_DIR, 'manifest.json')}")


if __name__ == "__main__":
    generate()
