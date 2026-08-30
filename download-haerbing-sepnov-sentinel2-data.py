"""
Download Sentinel-2 L2A scenes over the Harbin AOI for the Sept-Nov 2025
window, via the Copernicus Data Space Ecosystem (CDSE) OData API.

This is the automated counterpart to how `Haerbing_Dataset/` (the original
June-Sept 2025 window) was obtained -- that one was pulled manually through
the Copernicus Browser UI, tile by tile. Doing that again for a second date
window is tedious and error-prone (it's exactly how the T30UWV/T30UXB
France mismatch happened -- see poster_notes.md section 3), so this script
automates it: search by AOI tile ID + date range + cloud cover via the
OData API, then download each match. If the API changes or auth is
unavailable, fall back to the manual Copernicus Browser route used the
first time.

Same 4 MGRS tiles as the original download (see preprocess-haerbing-
sentinel2-data.py / generate-haerbing-ground-truth.py): T52TCT, T52UCU,
T52UDU, T52TDT. T52TDT is downloaded here too even though it was excluded
from the Jun-Aug ground truth for having only one usable date -- whether
that's still true for Sept-Nov can only be checked after this run, via the
manifest.json produced by the preprocessing step. Don't hardcode its
exclusion here.

Setup:
  1. Create a free account at https://dataspace.copernicus.eu/
  2. Put your credentials in `.env.cdse` (repo-local, gitignored -- never
     commit this file):
       CDSE_USERNAME=you@example.com
       CDSE_PASSWORD=your-password
     Environment variables of the same name, if already set, take
     priority over the file.
  3. pip install requests

Run with: venv/bin/python3 download-haerbing-sepnov-sentinel2-data.py
"""
import os
import time
import zipfile

import requests


def load_env_file(path=".env.cdse"):
    """Load KEY=VALUE lines from a gitignored local file into os.environ,
    without overwriting anything already set in the real environment."""
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

OUTPUT_DIR = "Haerbing_Dataset_sepnov"

TARGET_TILES = {"T52TCT", "T52UCU", "T52UDU", "T52TDT"}
DATE_START = "2025-09-01T00:00:00.000Z"
DATE_END = "2025-12-01T00:00:00.000Z"

# Same rationale as the original download: keep scenes usable after the
# SCL-based cloud/nodata tile filter in preprocess-haerbing-sentinel2-data.py
# (MAX_BAD_FRACTION=0.10) without discarding so many candidate scenes up
# front that too few dates survive per tile.
MAX_CLOUD_COVER_PCT = 30

IDENTITY_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL_TEMPLATE = "https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"


def get_access_token():
    username = os.environ.get("CDSE_USERNAME")
    password = os.environ.get("CDSE_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "Set CDSE_USERNAME and CDSE_PASSWORD in .env.cdse (or as "
            "environment variables) -- free account at "
            "https://dataspace.copernicus.eu/."
        )
    response = requests.post(
        IDENTITY_URL,
        data={
            "client_id": "cdse-public",
            "username": username,
            "password": password,
            "grant_type": "password",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def search_products():
    """Query the OData catalogue for Sentinel-2 L2A scenes in the date
    window, then keep only results over the target MGRS tiles (products
    are named e.g. S2B_MSIL2A_..._T52TCT_..., so a substring match on the
    Name field is enough -- the API itself doesn't filter by MGRS tile
    directly)."""
    filter_expr = (
        "Collection/Name eq 'SENTINEL-2' and "
        "contains(Name,'MSIL2A') and "
        f"ContentDate/Start gt {DATE_START} and "
        f"ContentDate/Start lt {DATE_END} and "
        "Attributes/OData.CSC.DoubleAttribute/any("
        f"att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value lt {MAX_CLOUD_COVER_PCT}.00)"
    )
    params = {"$filter": filter_expr, "$top": 100, "$orderby": "ContentDate/Start asc"}

    products = []
    url, request_params = CATALOGUE_URL, params
    while url:
        response = requests.get(url, params=request_params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        products.extend(payload.get("value", []))
        url = payload.get("@odata.nextLink")
        request_params = None  # nextLink already has query params baked in

    matched = [p for p in products if any(tile in p["Name"] for tile in TARGET_TILES)]
    print(f"Found {len(products)} candidate scenes in date/cloud window, "
          f"{len(matched)} over the target tiles {sorted(TARGET_TILES)}")
    return matched


def download_product(product, access_token, output_dir):
    safe_name = product["Name"]  # e.g. S2B_MSIL2A_..._T52TCT_....SAFE
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
    products = search_products()

    for i, product in enumerate(products):
        print(f"[{i + 1}/{len(products)}] {product['Name']}")
        # Re-authenticate before each download -- access tokens expire
        # after 10 minutes, and a handful of ~1GB downloads over a slow
        # connection can easily exceed that in total.
        access_token = get_access_token()
        try:
            download_product(product, access_token, output_dir)
        except requests.HTTPError as e:
            print(f"  FAILED ({e}) -- skipping, re-run this script later to retry just the missing scenes")
        time.sleep(1)  # be polite to the API between requests

    print(f"\nDone. Scenes saved under {output_dir}/")
    print("Next step: preprocess-haerbing-sepnov-sentinel2-data.py")


if __name__ == "__main__":
    download_all()
