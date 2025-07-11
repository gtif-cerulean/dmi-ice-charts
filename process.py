import os
import json
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box

# --- Configuration ---

# Load env vars with fallback to defaults
GROUPED_PARQUET_PATH = os.getenv("GROUPED_PARQUET_PATH", "daily_items.parquet")
ZIP_PARQUET_PATH = os.getenv("ZIP_PARQUET_PATH", "zipped_assets.parquet")
FLATGEOBUF_DIR = Path(os.getenv("FLATGEOBUF_DIR", "flatgeobufs"))
ZIPPED_DIR = Path(os.getenv("ZIPPED_DIR", "zips"))
SHAPEFILE_BASE_URL = os.getenv("SHAPEFILE_BASE_URL", "https://download.dmi.dk/public/ICESERVICE/SIGRID3/")
ASSET_BASE_URL_FGB = os.getenv("ASSET_BASE_URL_FGB", "https://your-bucket.example.com/daily")
ASSET_BASE_URL_ZIP = os.getenv("ASSET_BASE_URL_ZIP", "https://your-bucket.example.com/zips")

# Get current year and add it to shapefile base URL
current_year = datetime.now().year
SHAPEFILE_BASE_URL = SHAPEFILE_BASE_URL.rstrip('/') + f"/{current_year}/"

# Confirm loaded config
print(f"Using config:\n"
      f"  GEOPARQUET_PATH: {GROUPED_PARQUET_PATH}\n"
      f"  ZIP_INDEX_PATH: {ZIP_PARQUET_PATH}\n"
      f"  SHAPEFILE_REMOTE_BASE_URL: {SHAPEFILE_BASE_URL}\n"
      f"  DAILY_ITEMS_BASE_URL: {ASSET_BASE_URL_FGB}\n"
      f"  ZIP_ITEMS_BASE_URL: {ASSET_BASE_URL_ZIP}")

FLATGEOBUF_DIR.mkdir(exist_ok=True)
ZIPPED_DIR.mkdir(exist_ok=True)

def extract_date(folder_name):
    try:
        return datetime.strptime(folder_name[:8], "%Y%m%d").date()
    except ValueError:
        return None

def download_shapefile_folder(folder_name: str, destination: Path):
    base_url = f"{SHAPEFILE_BASE_URL}/{folder_name}/"
    expected_exts = [".shp", ".shx", ".dbf", ".prj", ".cpg"]
    downloaded = False

    for ext in expected_exts:
        url = f"{base_url}{folder_name}{ext}"
        local_path = destination / f"{folder_name}{ext}"
        r = requests.get(url)
        if r.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(r.content)
            downloaded = True
        else:
            print(f"⚠️ Missing: {url}")
    return downloaded

def zip_folder(source_folder: Path, zip_path: Path):
    shutil.make_archive(str(zip_path).replace(".zip", ""), 'zip', root_dir=source_folder)

def convert_to_flatgeobuf(shp_folder: Path, folder_name: str, out_path: Path):
    shp_file = shp_folder / f"{folder_name}.shp"
    if not shp_file.exists():
        print(f"❌ No .shp file found for {folder_name}")
        return False
    gdf = gpd.read_file(shp_file)
    gdf.to_file(out_path, driver="FlatGeobuf")
    return True

def load_existing(path):
    if os.path.exists(path):
        return gpd.read_parquet(path)
    return gpd.GeoDataFrame(columns=["filename", "item_id", "geometry", "date", "asset_url"])

def main(json_path):
    with open(json_path) as f:
        folders = json.load(f)["list"]

    existing_zip_items = load_existing(ZIP_PARQUET_PATH)
    existing_filenames = set(existing_zip_items["filename"])

    new_zip_records = []
    grouped_items = defaultdict(list)

    for folder_name in folders:
        if folder_name in existing_filenames:
            continue

        date = extract_date(folder_name)
        if not date:
            print(f"⚠️ Invalid date format in {folder_name}")
            continue

        # Create temporary folder for download
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            success = download_shapefile_folder(folder_name, tmp_path)
            if not success:
                print(f"❌ Skipping {folder_name} (download failed)")
                continue

            # Zip the folder
            zip_path = ZIPPED_DIR / f"{folder_name}.zip"
            zip_folder(tmp_path, zip_path)

            # Convert to flatgeobuf
            fgb_path = FLATGEOBUF_DIR / f"{folder_name}.fgb"
            if not convert_to_flatgeobuf(tmp_path, folder_name, fgb_path):
                continue

            # Read geometry
            try:
                gdf = gpd.read_file(fgb_path)
                geom = gdf.unary_union.envelope  # Bounding box
            except Exception as e:
                print(f"❌ Error reading {fgb_path}: {e}")
                continue

            zip_url = f"{ASSET_BASE_URL_ZIP}/{folder_name}.zip"
            fgb_url = f"{ASSET_BASE_URL_FGB}/{folder_name}.fgb"

            # Register in zip_items (single asset per item)
            new_zip_records.append({
                "filename": folder_name,
                "item_id": folder_name,
                "geometry": geom,
                "date": pd.to_datetime(date),
                "asset_url": zip_url
            })

            grouped_items[date].append({
                "filename": folder_name,
                "geometry": geom,
                "fgb_url": fgb_url
            })

    # Save updated zip_items.parquet
    if new_zip_records:
        zip_gdf = gpd.GeoDataFrame(new_zip_records, crs="EPSG:4326")
        updated_zip = pd.concat([existing_zip_items, zip_gdf], ignore_index=True)
        updated_zip.to_parquet(ZIP_PARQUET_PATH)
        print(f"✅ Updated {ZIP_PARQUET_PATH} with {len(new_zip_records)} items.")
    else:
        print("✅ No new zip items to add.")

    # Generate grouped_items.parquet (many assets per item)
    existing_grouped = load_existing(GROUPED_PARQUET_PATH)
    grouped_records = []

    for date, items in grouped_items.items():
        item_id = f"daily_{date}"
        geometries = [i["geometry"] for i in items]
        envelope = gpd.GeoSeries(geometries).unary_union.envelope

        grouped_records.append({
            "item_id": item_id,
            "geometry": envelope,
            "date": pd.to_datetime(date),
            "assets": [i["fgb_url"] for i in items]
        })

    if grouped_records:
        grouped_gdf = gpd.GeoDataFrame(grouped_records, crs="EPSG:4326")
        updated_grouped = pd.concat([existing_grouped, grouped_gdf], ignore_index=True)
        updated_grouped.to_parquet(GROUPED_PARQUET_PATH)
        print(f"✅ Updated {GROUPED_PARQUET_PATH} with {len(grouped_records)} grouped items.")
    else:
        print("✅ No new grouped items to add.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python process.py <input.json>")
        sys.exit(1)
    main(sys.argv[1])
