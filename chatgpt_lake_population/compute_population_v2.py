#!/usr/bin/env python3
from __future__ import annotations

import base64
import gzip
import json
import math
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds
from scipy.ndimage import distance_transform_edt
from shapely.geometry import mapping
from shapely.ops import transform as shp_transform
from shapely.validation import make_valid

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
CACHE = ROOT / "cache_v2"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

POP_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_POP_GLOBE_R2023A/GHS_POP_E2020_GLOBE_R2023A_54009_1000/V1-0/"
    "GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0.zip"
)
LAND_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_LAND_GLOBE_R2022A/GHS_LAND_E2018_GLOBE_R2022A_54009_1000/V1-0/"
    "GHS_LAND_E2018_GLOBE_R2022A_54009_1000_V1_0.zip"
)
ARCGIS_URL = (
    "https://services7.arcgis.com/poOcx60xJtGtoR7g/ArcGIS/rest/services/"
    "Lakes/FeatureServer/0/query"
)
RADII_KM = (10, 25, 50)
SIGMAS_KM = (5, 10, 20, 30)
MAX_DISTANCE_KM = 90
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "academic-lake-population-density/2.0"


def read_ids() -> list[int]:
    payload = (ROOT / "hylak_ids_3891.txt.gz.b64").read_text().strip()
    text = gzip.decompress(base64.b64decode(payload)).decode("utf-8")
    ids = [int(x) for x in text.splitlines() if x.strip()]
    if len(ids) != 3891 or len(set(ids)) != 3891:
        raise RuntimeError(f"Invalid HydroLAKES ID payload: {len(ids)} rows")
    return ids


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 10_000_000:
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(6):
        try:
            with SESSION.get(url, stream=True, timeout=(30, 600)) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            handle.write(block)
            partial.replace(destination)
            print("Downloaded", destination.name, destination.stat().st_size, flush=True)
            return
        except Exception as exc:
            print("Download retry", attempt + 1, url, repr(exc), flush=True)
            time.sleep(2**attempt)
    raise RuntimeError(f"Could not download {url}")


def extract_first_tif(archive: Path, folder: Path) -> Path:
    folder.mkdir(exist_ok=True)
    existing = list(folder.rglob("*.tif"))
    if existing:
        return existing[0]
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(folder)
    files = list(folder.rglob("*.tif"))
    if not files:
        raise RuntimeError(f"No TIFF in {archive}")
    return files[0]


def fetch_geometries(ids: list[int]) -> gpd.GeoDataFrame:
    cached = CACHE / "hydrolakes_3891.geojson"
    if cached.exists():
        return gpd.read_file(cached)
    features: list[dict] = []
    for start in range(0, len(ids), 25):
        batch = ids[start : start + 25]
        form = {
            "where": "Hylak_id IN (" + ",".join(map(str, batch)) + ")",
            "outFields": "Hylak_id,Lake_area,Shore_len",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
            "geometryPrecision": "5",
            "maxAllowableOffset": "0.0001",
        }
        for attempt in range(6):
            try:
                response = SESSION.post(ARCGIS_URL, data=form, timeout=(30, 300))
                response.raise_for_status()
                payload = response.json()
                if "error" in payload:
                    raise RuntimeError(payload["error"])
                features.extend(payload.get("features", []))
                break
            except Exception as exc:
                if attempt == 5:
                    raise
                print("Geometry retry", start, attempt + 1, repr(exc), flush=True)
                time.sleep(2**attempt)
        if start % 250 == 0:
            print("Geometry progress", start, len(features), flush=True)
    cached.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    frame = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    print("Geometry rows", len(frame), flush=True)
    return frame


def valid_geometry(geometry):
    if geometry is None or geometry.is_empty:
        return None
    try:
        geometry = make_valid(geometry)
    except Exception:
        geometry = geometry.buffer(0)
    return None if geometry.is_empty else geometry


def clipped_window(raw: Window, width: int, height: int) -> Window | None:
    col0 = max(0, math.floor(raw.col_off))
    row0 = max(0, math.floor(raw.row_off))
    col1 = min(width, math.ceil(raw.col_off + raw.width))
    row1 = min(height, math.ceil(raw.row_off + raw.height))
    if col1 <= col0 or row1 <= row0:
        return None
    return Window(col0, row0, col1 - col0, row1 - row0)


def main() -> None:
    ids = read_ids()
    pop_zip = CACHE / "ghs_pop_2020_1km.zip"
    land_zip = CACHE / "ghs_land_2018_1km.zip"
    download(POP_URL, pop_zip)
    download(LAND_URL, land_zip)
    pop_tif = extract_first_tif(pop_zip, CACHE / "population")
    land_tif = extract_first_tif(land_zip, CACHE / "land")

    lakes = fetch_geometries(ids)
    lakes["Hylak_id"] = pd.to_numeric(lakes["Hylak_id"]).astype(int)
    lakes = lakes.drop_duplicates("Hylak_id").set_index("Hylak_id")
    transformer = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)
    results: list[dict] = []

    with rasterio.open(pop_tif) as pop_src, rasterio.open(land_tif) as land_src:
        if (
            pop_src.crs != land_src.crs
            or pop_src.transform != land_src.transform
            or pop_src.width != land_src.width
            or pop_src.height != land_src.height
        ):
            raise RuntimeError("GHS-POP and GHS-LAND grids are not aligned")
        pixel_x_km = abs(pop_src.transform.a) / 1000
        pixel_y_km = abs(pop_src.transform.e) / 1000
        print("Raster grid", pop_src.width, pop_src.height, pop_src.crs, flush=True)

        for position, lake_id in enumerate(ids, 1):
            row: dict[str, object] = {"hylak_id": lake_id, "spatial_status": "OK"}
            try:
                if lake_id not in lakes.index:
                    raise KeyError("HydroLAKES geometry not returned")
                source_geometry = valid_geometry(lakes.loc[lake_id].geometry)
                projected = valid_geometry(shp_transform(transformer.transform, source_geometry))
                if projected is None:
                    raise ValueError("Invalid geometry")
                raw_window = from_bounds(
                    *projected.buffer(MAX_DISTANCE_KM * 1000).bounds,
                    transform=pop_src.transform,
                )
                window = clipped_window(raw_window, pop_src.width, pop_src.height)
                if window is None:
                    raise ValueError("Raster window outside grid")
                population = pop_src.read(1, window=window, masked=False).astype("float64")
                land_m2 = land_src.read(1, window=window, masked=False).astype("float64")
                if pop_src.nodata is not None:
                    population[population == pop_src.nodata] = 0
                if land_src.nodata is not None:
                    land_m2[land_m2 == land_src.nodata] = 0
                population[~np.isfinite(population) | (population < 0)] = 0
                land_m2[~np.isfinite(land_m2) | (land_m2 < 0)] = 0
                transform = pop_src.window_transform(window)
                lake_mask = rasterize(
                    [(mapping(projected), 1)],
                    out_shape=population.shape,
                    transform=transform,
                    fill=0,
                    all_touched=True,
                    dtype="uint8",
                ).astype(bool)
                if not lake_mask.any():
                    lake_mask = rasterize(
                        [(mapping(projected.buffer(500)), 1)],
                        out_shape=population.shape,
                        transform=transform,
                        fill=0,
                        all_touched=True,
                        dtype="uint8",
                    ).astype(bool)
                distances_km = distance_transform_edt(
                    ~lake_mask, sampling=(pixel_y_km, pixel_x_km)
                )
                land_km2 = land_m2 / 1_000_000.0
                population[lake_mask] = 0
                land_km2[lake_mask] = 0

                for radius in RADII_KM:
                    include = (distances_km > 0) & (distances_km <= radius) & (land_km2 > 0)
                    pop_total = float(population[include].sum())
                    land_total = float(land_km2[include].sum())
                    row[f"population_mean_{radius}km"] = pop_total
                    row[f"land_area_mean_{radius}km_km2"] = land_total
                    row[f"density_mean_{radius}km"] = (
                        pop_total / land_total if land_total > 0 else np.nan
                    )

                for sigma in SIGMAS_KM:
                    include = (
                        (distances_km > 0)
                        & (distances_km <= 3 * sigma)
                        & (land_km2 > 0)
                    )
                    weights = np.exp(
                        -(distances_km[include] ** 2) / (2 * sigma * sigma)
                    )
                    weighted_pop = float((population[include] * weights).sum())
                    weighted_land = float((land_km2[include] * weights).sum())
                    row[f"population_gaussian_sigma{sigma}km"] = weighted_pop
                    row[f"effective_land_area_gaussian_sigma{sigma}km_km2"] = weighted_land
                    row[f"density_gaussian_sigma{sigma}km"] = (
                        weighted_pop / weighted_land if weighted_land > 0 else np.nan
                    )
            except Exception as exc:
                row["spatial_status"] = f"ERROR:{type(exc).__name__}:{exc}"
            results.append(row)
            if position % 100 == 0:
                pd.DataFrame(results).to_csv(OUT / "partial.csv", index=False)
                ok = sum(x["spatial_status"] == "OK" for x in results)
                print("Processed", position, "OK", ok, flush=True)

    frame = pd.DataFrame(results)
    frame.to_csv(OUT / "lake_population_metrics.csv", index=False)
    summary = {
        "n_target": len(ids),
        "n_rows": len(frame),
        "n_ok": int((frame["spatial_status"] == "OK").sum()),
        "n_error": int((frame["spatial_status"] != "OK").sum()),
        "population_source": "GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0",
        "land_source": "GHS_LAND_E2018_GLOBE_R2022A_54009_1000_V1_0",
        "geometry_source": "HydroLAKES ArcGIS FeatureServer",
        "mean_buffers_km": list(RADII_KM),
        "gaussian_sigma_km": list(SIGMAS_KM),
        "density_denominator": "sum of GHS-LAND land surface area within shoreline-distance zone",
    }
    (OUT / "spatial_summary.json").write_text(json.dumps(summary, indent=2))
    print(summary, flush=True)


if __name__ == "__main__":
    main()
