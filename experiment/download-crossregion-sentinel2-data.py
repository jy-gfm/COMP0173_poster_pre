"""
Download a handful of Sentinel-2 L2A scenes for the two cross-region
generalization-check sites (applying the already-trained Harbin model to
genuinely new geography, no retraining, no ground truth -- a visual
plausibility check only, same spirit as the T52TDT unlabeled-scene check):

  - Shangri-La (Zhongdian), Yunnan -- MGRS tile T47RNL. High-altitude
    alpine/subalpine coniferous forest, a real cross-biome domain shift
    from Heilongjiang's boreal-temperate forest.
  - Horqin Sandy Land / Zhanggutai, Liaoning -- MGRS tile T51TVH. A
    Three-North Shelterbelt afforestation success site, same broad
    climate zone as the training data but historically low-forest,
    tests whether the model recognizes young/planted forest.

Only 1-2 low-cloud scenes per tile are needed (visual check, not training),
so this keeps the download small.

Run with: venv/bin/python3 download-crossregion-sentinel2-data.py
"""
import os
import time
import zipfile

import requests


def load_env_file(path=".env.cdse"):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


load_env_file()

OUTPUT_DIR = "CrossRegion_Dataset"

DATE_START = "2025-06-01T00:00:00.000Z"
DATE_END = "2025-09-30T00:00:00.000Z"
SCENES_PER_TILE = 2

# Per-tile cloud threshold -- Shangri-La (Yunnan) sits in the summer monsoon
# belt, so June-Sept has almost no <10%-cloud scenes (confirmed: 0 at <10%,
# only 1 at <20%, 4 at <30%). Loosened just for that tile rather than
# lowering the bar everywhere; Horqin already has 15 scenes at <10%.
TILE_CLOUD_LIMITS = {"T47RNL": 30, "T51TVH": 10}
TARGET_TILES = set(TILE_CLOUD_LIMITS)

IDENTITY_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL_TEMPLATE = "https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"


def get_access_token():
    username = os.environ.get("CDSE_USERNAME")
    password = os.environ.get("CDSE_PASSWORD")
    if not username or not password:
        raise RuntimeError("Set CDSE_USERNAME and CDSE_PASSWORD in .env.cdse.")
    response = requests.post(
        IDENTITY_URL,
        data={"client_id": "cdse-public", "username": username, "password": password, "grant_type": "password"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def search_products_for_tile(tile):
    cloud_limit = TILE_CLOUD_LIMITS[tile]
    filter_expr = (
        "Collection/Name eq 'SENTINEL-2' and "
        "contains(Name,'MSIL2A') and "
        f"contains(Name,'{tile}') and "
        f"ContentDate/Start gt {DATE_START} and "
        f"ContentDate/Start lt {DATE_END} and "
        "Attributes/OData.CSC.DoubleAttribute/any("
        f"att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value lt {cloud_limit}.00)"
    )
    params = {"$filter": filter_expr, "$top": 20, "$orderby": "Attributes/OData.CSC.DoubleAttribute/Value asc"}
    response = requests.get(CATALOGUE_URL, params=params, timeout=60)
    response.raise_for_status()
    products = response.json().get("value", [])
    print(f"{tile}: found {len(products)} low-cloud scenes")
    return products[:SCENES_PER_TILE]


def download_product(product, access_token, output_dir):
    safe_name = product["Name"]
    target_dir = os.path.join(output_dir, safe_name)
    if os.path.exists(target_dir):
        print(f"  already have {safe_name}, skipping")
        return

    zip_path = os.path.join(output_dir, safe_name + ".zip")
    url = DOWNLOAD_URL_TEMPLATE.format(product_id=product["Id"])
    headers = {"Authorization": f"Bearer {access_token}"}

    print(f"  downloading {safe_name} ...", flush=True)
    with requests.get(url, headers=headers, stream=True, timeout=300) as response:
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(output_dir)
    os.remove(zip_path)
    print(f"  done: {target_dir}")


def download_all(output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    all_products = []
    for tile in sorted(TARGET_TILES):
        all_products.extend(search_products_for_tile(tile))

    for i, product in enumerate(all_products):
        print(f"[{i + 1}/{len(all_products)}] {product['Name']}")
        access_token = get_access_token()
        try:
            download_product(product, access_token, output_dir)
        except requests.HTTPError as e:
            print(f"  FAILED ({e}) -- skipping")
        time.sleep(1)

    print(f"\nDone. Scenes saved under {output_dir}/")


if __name__ == "__main__":
    download_all()
