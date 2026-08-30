"""
Preprocess the Sept-Nov 2025 Harbin-area Sentinel-2 L2A dataset
(Haerbing_Dataset_sepnov/, from download-haerbing-sepnov-sentinel2-data.py):
  - band selection (B04/B03/B02/B08 at 10m) + NDVI
  - reflectance correction for processing baseline N0511
  - tiling into 512x512 patches matching the Attention U-Net's input size
  - cloud/no-data filtering per tile using the SCL band

Identical method to preprocess-haerbing-sentinel2-data.py (the June-Aug
dataset) -- kept as a separate script, same convention as that one being
kept separate from the original Amazon/Atlantic preprocessing scripts, so
the two date windows can't be mixed up and each has its own manifest. See
poster_notes.md section 8 for why a second (autumn) date window is worth
trying: Heilongjiang cropland and natural forest have overlapping NDVI in
mid-summer, but crops senesce/are harvested by autumn while forest stays
comparatively green, which may make forest vs. non-forest separation
easier for both the NDVI-persistence labelling and the model itself. This
is a hypothesis to check via the ground-truth QC step, not an assumption.

Run with: venv/bin/python3 preprocess-haerbing-sepnov-sentinel2-data.py
"""
import gc
import os
import re
import glob
import json
import numpy as np
import rioxarray as rxr

SOURCE_DIR = "Haerbing_Dataset_sepnov"
OUTPUT_DIR = "Haerbing_processed_sepnov"
TILE_SIZE = 512

# ESA processing baseline >= 04.00 (all scenes here are N0511) subtracts a
# fixed offset from the raw digital number before the x10000 reflectance
# scaling. Skipping this correction would systematically bias reflectance
# and NDVI values for every scene in this dataset.
BOA_ADD_OFFSET = -1000
REFLECTANCE_SCALE = 10000

# Discard a tile if more than this fraction of its pixels are nodata,
# saturated, cloud, cloud shadow, cirrus, or snow (per the SCL band).
# Autumn in Heilongjiang can bring early snow -- SCL class 11 (snow) is
# already in this set, same as the June-Aug run, so early-snow scenes get
# filtered out at the tile level rather than silently mislabelled.
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
    # Close the underlying GDAL dataset handle explicitly rather than
    # relying on garbage collection to get to it eventually -- with 5
    # bands opened per scene across 18 scenes, deferred cleanup was
    # letting file handles and GDAL's block cache pile up.
    with rxr.open_rasterio(path) as da:
        return da.values[0].copy()  # drop the singleton band dim -> (H, W)


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

        # Free this scene's ~3GB of arrays *before* the next iteration's
        # load_scene() call builds the next scene's arrays -- without this,
        # Python only rebinds (and frees the old) image/ndvi/scl *after*
        # the new scene is fully loaded, so both scenes' memory briefly
        # coexist. That overlap was enough to trigger the OOM killer right
        # at the scene 1 -> scene 2 boundary.
        del image, ndvi, scl
        gc.collect()

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\nDone. Manifest written to {manifest_path}")

    # Same diagnostic that originally justified excluding T52TDT from the
    # June-Aug dataset (see generate-haerbing-ground-truth.py) -- check
    # whether it (or any other tile) again ends up single-date-only before
    # deciding on EXCLUDED_TILE_IDS in the ground-truth generation step.
    from collections import defaultdict
    dates_per_tile = defaultdict(set)
    for r in manifest:
        if r["kept"]:
            dates_per_tile[r["tile_id"]].add(r["date"])
    print("\nDistinct kept dates per tile (check before running "
          "generate-haerbing-sepnov-ground-truth.py):")
    for tile_id, dates in sorted(dates_per_tile.items()):
        print(f"  {tile_id}: {len(dates)} date(s) -- {sorted(dates)}")


if __name__ == "__main__":
    process_all()
