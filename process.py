import os
import json
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from bs4 import BeautifulSoup
from shapely import union_all
from shapely.ops import unary_union

import geopandas as gpd
import pandas as pd
import requests

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
SYNC_YEAR = os.getenv("SYNC_YEAR", str(datetime.now().year))
SHAPEFILE_BASE_URL = SHAPEFILE_BASE_URL.rstrip('/') + f"/{SYNC_YEAR}/"

# Confirm loaded config
print(f"Using config:\n"
      f"  GEOPARQUET_PATH: {GROUPED_PARQUET_PATH}\n"
      f"  ZIP_INDEX_PATH: {ZIP_PARQUET_PATH}\n"
      f"  SHAPEFILE_REMOTE_BASE_URL: {SHAPEFILE_BASE_URL}\n"
      f"  DAILY_ITEMS_BASE_URL: {ASSET_BASE_URL_FGB}\n"
      f"  ZIP_ITEMS_BASE_URL: {ASSET_BASE_URL_ZIP}")

FLATGEOBUF_DIR.mkdir(exist_ok=True, parents=True)
ZIPPED_DIR.mkdir(exist_ok=True, parents=True)

def fetch_folder_list_from_remote(base_url):
    print(f"Fetching folder list from remote: {base_url}")
    response = requests.get(base_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    folder_names = []

    for link in soup.find_all('a'):
        href = link.get('href')

        if (
            not href or
            href in ('../', '/') or
            href.startswith('public') or
            href.startswith('/')  # Avoid absolute paths like /public/...
        ):
            continue

        if href.endswith('/'):
            folder = href.rstrip('/')
            folder_names.append(folder)

    folder_names = sorted(set(folder_names))
    return {"list": folder_names}

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
    print(f"Zipping folder {source_folder} to {zip_path}")
    shutil.make_archive(str(zip_path).replace(".zip", ""), 'zip', root_dir=source_folder)

def convert_to_flatgeobuf(shp_folder: Path, folder_name: str, out_path: Path):
    print(f"Converting {folder_name} shapefile to FlatGeobuf at {out_path}")
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
    return gpd.GeoDataFrame(columns=[
        "id", "type", "stac_version", "datetime",
        "geometry", "bbox", "assets", "links"], crs="EPSG:4326"
    )

def create_stac_item(date, id, assets, asset_type):
    datetime_obj = pd.to_datetime(date)

    # Union geometries and get envelope
    geometries = [i["geometry"] for i in assets]
    envelope = union_all(gpd.GeoSeries(geometries)).envelope

    # Construct valid STAC Item dictionary
    stac_item = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": id,
        "datetime": datetime_obj,
        "geometry": envelope,
        "bbox": list(envelope.bounds),
        "assets": {
            f"asset_{idx}": {
                "href": item["url"],
                "type": asset_type,
                "roles": ["data"]
            }
            for idx, item in enumerate(assets)
        },
        "links": []
    }
    return stac_item
    

def add_style_link(row):
    STYLE_URL = os.getenv("STYLE_URL", "")
    # Skip if base URL isn't set
    if not STYLE_URL:
        return row.get("links", [])

    assets = row.get("assets", {})
    asset_keys = list(assets.keys())
    if not asset_keys:
        return row.get("links", [])

    # Remove old style links
    links = [link for link in row.get("links", []) if link.get("rel") != "style"]

    # Append new style link
    links.append({
        "rel": "style",
        "href": f"{STYLE_URL}",
        "type": "text/vector-styles",
        "asset:keys": asset_keys
    })

    return links

def merge_items_per_day(df):
    merged_records = []

    for item_id, group in df.groupby("id"):
        # Merge geometries and using envelope around them
        geoms = group["geometry"].tolist()
        merged_geom = union_all(gpd.GeoSeries(geoms)).envelope

        # Flatten and filter assets
        flat_assets = []
        for assets in group["assets"]:
            if isinstance(assets, dict):
                flat_assets.extend([
                    asset for asset in assets.values() if asset  # filter out nulls
                ])

        # Reindex to asset_0, asset_1, ...
        merged_assets = {
            f"asset_{i}": asset for i, asset in enumerate(flat_assets)
        }

        # Merge links, deduplicating by (rel, href)
        seen_links = set()
        merged_links = []
        for links in group["links"]:
            for link in links:
                key = (link.get("rel"), link.get("href"))
                if key not in seen_links:
                    seen_links.add(key)
                    merged_links.append(link)

        # Use the first datetime (assumed same day)
        date = pd.to_datetime(group["datetime"].iloc[0])

        merged_records.append({
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": item_id,
            "geometry": merged_geom,
            "bbox": list(gpd.GeoSeries([merged_geom]).total_bounds),
            "datetime": date,
            "assets": merged_assets,
            "links": merged_links
        })

    return gpd.GeoDataFrame(merged_records, geometry="geometry", crs="EPSG:4326")

def main(args):
    if len(args) > 1:
        json_path = Path(args[1])
        if not json_path.exists():
            print(f"❌ JSON file {json_path} does not exist.")
            return
    else:
        # Fetch folder list from remote if no JSON file provided
        json_data = fetch_folder_list_from_remote(SHAPEFILE_BASE_URL)
        json_path = Path("folders.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"✅ Fetched folder list and saved to {json_path}")

    with open(json_path) as f:
        folders = json.load(f)["list"]

    existing_zip_items = load_existing(ZIP_PARQUET_PATH)
    existing_ids = set(existing_zip_items["id"])

    new_zip_records = []
    grouped_items = defaultdict(list)

    for folder_name in folders:
        if folder_name in existing_ids:
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
                geom = union_all(gdf.geometry).envelope  # Bounding box
            except Exception as e:
                print(f"❌ Error reading {fgb_path}: {e}")
                continue

            zip_url = f"{ASSET_BASE_URL_ZIP}/{folder_name}.zip"
            fgb_url = f"{ASSET_BASE_URL_FGB}/{folder_name}.fgb"

            # Register in zip_items (single asset per item)
            # we use download folder name as id
            new_zip_records.append(
                create_stac_item(
                    date,
                    folder_name,
                    [{"url": zip_url, "geometry": geom}],
                    "application/zip"
                )
            )
            grouped_items[date].append(
               {"url": fgb_url, "geometry": geom}
            )

    # Save updated zip_items.parquet
    if new_zip_records:
        zip_gdf = gpd.GeoDataFrame(new_zip_records, crs="EPSG:4326")
        updated_zip = pd.concat([existing_zip_items, zip_gdf], ignore_index=True)
        updated_zip.to_parquet(ZIP_PARQUET_PATH)
        print(f"✅ Updated {ZIP_PARQUET_PATH} with {len(new_zip_records)} items.")
    else:
        print("✅ No new zip items to add.")
    # copy updated parquet file to zipped files output directory
    print(f"Copying {ZIP_PARQUET_PATH} to {ZIPPED_DIR}")
    shutil.copy(ZIP_PARQUET_PATH, ZIPPED_DIR)

    # Generate grouped_items.parquet (many assets per item)
    existing_grouped = load_existing(GROUPED_PARQUET_PATH)
    grouped_records = []

    for date, assets in grouped_items.items():
        grouped_records.append(
            create_stac_item(date, date.strftime("%Y-%m-%d") , assets, "application/vnd.flatgeobuf")
        )

    # updating grouped items parquet file
    if grouped_records:
        grouped_gdf = gpd.GeoDataFrame(grouped_records, crs="EPSG:4326")
        updated_grouped = pd.concat([existing_grouped, grouped_gdf], ignore_index=True)
    else:
        updated_grouped = existing_grouped
    # Add style links to grouped items
    updated_grouped["links"] = updated_grouped.apply(add_style_link, axis=1)
    # make sure daily items are merged from previous runs
    deduplicated = merge_items_per_day(updated_grouped)
    print(f"✅ Previous length {len(updated_grouped)}, deduplicated {len(deduplicated)}")
    deduplicated.to_parquet(GROUPED_PARQUET_PATH)
    print(f"✅ Updated {GROUPED_PARQUET_PATH} with {len(deduplicated)} grouped items.")
    
    # copy updated parquet file to flatgeobufs output directory
    print(f"Copying {GROUPED_PARQUET_PATH} to {FLATGEOBUF_DIR}")
    shutil.copy(GROUPED_PARQUET_PATH, FLATGEOBUF_DIR)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        print("Usage: python process.py [jsonfile_path]")
        sys.exit(1)
    main(sys.argv)
