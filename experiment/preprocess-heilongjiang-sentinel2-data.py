"""
Preprocess the Harbin-area Sentinel-2 L2A dataset (Haerbing_Dataset/):
  - band selection (B04/B03/B02/B08 at 10m) + NDVI
  - reflectance correction for processing baseline N0511 (all 14 scenes)
  - tiling into 512x512 patches matching the Attention U-Net's input size
  - cloud/no-data filtering per tile using the SCL band

This script only prepares the IMAGE inputs, organised by tile ID and
acquisition date -- ground-truth masks are deliberately NOT generated
here, since the labelling method (k-means vs. NDVI change detection) is
still pending confirmation. NDVI is saved per tile alongside each image
either way, since it's needed for the NDVI change-detection route and is
cheap to compute now regardless of which method is used.

Run with: venv/bin/python3 preprocess-haerbing-sentinel2-data.py
"""
import os
import re
import glob
import json
import numpy as np
import rioxarray as rxr

SOURCE_DIR = "Haerbing_Dataset"
OUTPUT_DIR = "Haerbing_processed"
TILE_SIZE = 512

# ESA processing baseline >= 04.00 (all scenes here are N0511) subtracts a
# fixed offset from the raw digital number before the x10000 reflectance
# scaling. Skipping this correction would systematically bias reflectance
# and NDVI values for every scene in this dataset.
BOA_ADD_OFFSET = -1000
REFLECTANCE_SCALE = 10000

# Discard a tile if more than this fraction of its pixels are nodata,
# saturated, cloud, cloud shadow, cirrus, or snow (per the SCL band).
MAX_BAD_FRACTION = 0.10
BAD_SCL_CLASSES = {0, 1, 3, 8, 9, 10, 11}

SAFE_NAME_RE = re.compile(
    r"^(?P<sensor>S2[ABC])_MSIL2A_(?P<date>\d{8})T\d{6}_N\d{4}_R\d{3}_"
    r"(?P<tile>T\d{2}[A-Z]{3})_\d{8}T\d{6}\.SAFE$"
)


def find_scenes(source_dir):
    scenes = []
    for name in sorted(os.listdir(source_dir)):
        m = SAFE_NAME_RE.match(name)
        if not m:
            continue
        scenes.append({
            "path": os.path.join(source_dir, name),
            "tile": m.group("tile"),
            "date": m.group("date"),
        })
    return scenes


def find_granule_dir(safe_path):
    granule_root = os.path.join(safe_path, "GRANULE")
    subdirs = [d for d in os.listdir(granule_root) if not d.startswith('.')]
    assert len(subdirs) == 1, f"expected exactly one GRANULE subfolder in {safe_path}, got {subdirs}"
    return os.path.join(granule_root, subdirs[0])


def band_path(granule_dir, band, resolution):
    pattern = os.path.join(granule_dir, "IMG_DATA", f"R{resolution}m", f"*_{band}_{resolution}m.jp2")
    matches = glob.glob(pattern)
    assert len(matches) == 1, f"expected exactly one match for {pattern}, got {matches}"
    return matches[0]


def read_band(path):
    return rxr.open_rasterio(path).values[0]  # drop the singleton band dim -> (H, W)


def load_scene(safe_path):
    granule_dir = find_granule_dir(safe_path)

    b02 = read_band(band_path(granule_dir, "B02", 10)).astype(np.float32)
    b03 = read_band(band_path(granule_dir, "B03", 10)).astype(np.float32)
    b04 = read_band(band_path(granule_dir, "B04", 10)).astype(np.float32)
    b08 = read_band(band_path(granule_dir, "B08", 10)).astype(np.float32)

    def to_reflectance(band):
        r = (band + BOA_ADD_OFFSET) / REFLECTANCE_SCALE
        return np.clip(r, 0.0, 1.0)

    r, g, b, nir = (to_reflectance(x) for x in (b04, b03, b02, b08))
    image = np.stack([r, g, b, nir], axis=-1)  # (H, W, 4) -- R, G, B, NIR

    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - r) / (nir + r)
    ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)

    # SCL is at 20m -- upsample 2x (nearest-neighbour) to match the 10m grid
    scl_20m = read_band(band_path(granule_dir, "SCL", 20))
    scl_10m = np.repeat(np.repeat(scl_20m, 2, axis=0), 2, axis=1)
    h, w = image.shape[:2]
    scl_10m = scl_10m[:h, :w]
    if scl_10m.shape != (h, w):
        pad_h, pad_w = h - scl_10m.shape[0], w - scl_10m.shape[1]
        scl_10m = np.pad(scl_10m, ((0, max(pad_h, 0)), (0, max(pad_w, 0))), constant_values=0)

    return image, ndvi, scl_10m


def tile_scene(image, ndvi, scl, tile_size=TILE_SIZE):
    h, w = image.shape[:2]
    n_rows, n_cols = h // tile_size, w // tile_size
    for row in range(n_rows):
        for col in range(n_cols):
            r0, c0 = row * tile_size, col * tile_size
            img_tile = image[r0:r0 + tile_size, c0:c0 + tile_size]
            ndvi_tile = ndvi[r0:r0 + tile_size, c0:c0 + tile_size]
            scl_tile = scl[r0:r0 + tile_size, c0:c0 + tile_size]
            bad_fraction = np.isin(scl_tile, list(BAD_SCL_CLASSES)).mean()
            yield row, col, img_tile, ndvi_tile, bad_fraction


def process_all(source_dir=SOURCE_DIR, output_dir=OUTPUT_DIR, max_bad_fraction=MAX_BAD_FRACTION, scene_limit=None):
    scenes = find_scenes(source_dir)
    if scene_limit is not None:
        scenes = scenes[:scene_limit]

    manifest = []
    for i, scene in enumerate(scenes):
        print(f"[{i + 1}/{len(scenes)}] {scene['tile']} {scene['date']} ...", flush=True)
        image, ndvi, scl = load_scene(scene["path"])

        tile_dir = os.path.join(output_dir, scene["tile"], scene["date"])
        os.makedirs(tile_dir, exist_ok=True)

        kept, discarded = 0, 0
        for row, col, img_tile, ndvi_tile, bad_fraction in tile_scene(image, ndvi, scl):
            record = {
                "tile_id": scene["tile"], "date": scene["date"],
                "row": row, "col": col, "bad_fraction": float(bad_fraction),
            }
            if bad_fraction > max_bad_fraction:
                discarded += 1
                record["kept"] = False
                manifest.append(record)
                continue
            kept += 1
            record["kept"] = True
            base = f"row{row:02d}_col{col:02d}"
            np.save(os.path.join(tile_dir, f"{base}_image.npy"), img_tile.astype(np.float32))
            np.save(os.path.join(tile_dir, f"{base}_ndvi.npy"), ndvi_tile.astype(np.float32))
            manifest.append(record)

        print(f"    kept {kept} tiles, discarded {discarded} (cloud/nodata > {max_bad_fraction:.0%})", flush=True)

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\nDone. Manifest written to {manifest_path}")


if __name__ == "__main__":
    process_all()
