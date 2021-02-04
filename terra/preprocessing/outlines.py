"""Stable-ground / glacier outlines and mask functions."""
from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import shapely
import shapely.ops
from tqdm import tqdm

from terra import base_dem, files, utilities
from terra.constants import CONSTANTS
from terra.preprocessing import image_meta

TEMP_DIRECTORY = os.path.join(files.TEMP_DIRECTORY, "preprocessing/")
CACHE_FILES = {
    "glacier_mask": os.path.join(TEMP_DIRECTORY, "glacier_mask.tif"),
    "stable_ground_mask": os.path.join(TEMP_DIRECTORY, "stable_ground_mask.tif"),
    "lk50_outlines": os.path.join(TEMP_DIRECTORY, "lk50_outlines.shp"),
}


def rasterize_outlines(input_filepath: str, output_filepath: str, overwrite: bool = False,
                       resolution: float = CONSTANTS.dem_resolution) -> None:
    """Generate a boolean glacier mask from the 1935 map."""
    # Skip if it already exists
    if not overwrite and os.path.isfile(output_filepath):
        return
    # Get the bounds from the reference DEM
    reference_bounds = json.loads(
        subprocess.run(
            ["gdalinfo", "-json", base_dem.CACHE_FILES["base_dem"]],
            check=True,
            stdout=subprocess.PIPE
        ).stdout
    )["cornerCoordinates"]

    # Generate the mask
    gdal_commands = [
        "gdal_rasterize",
        "-burn", 1,  # Glaciers get a value of 1
        "-a_nodata", 0,  # Non-glaciers get a value of 0
        "-ot", "Byte",
        "-tr", resolution, resolution,
        "-te", *reference_bounds["lowerLeft"], *reference_bounds["upperRight"],
        input_filepath,
        output_filepath
    ]
    subprocess.run(list(map(str, gdal_commands)), check=True, stdout=subprocess.PIPE)

    # Unset the nodata value to correctly display in e.g. QGIS
    subprocess.run(["gdal_edit.py", "-unsetnodata", output_filepath],
                   check=True, stdout=subprocess.PIPE)


def generate_glacier_mask(overwrite: bool = False, resolution: float = CONSTANTS.dem_resolution):
    rasterize_outlines(
        input_filepath=files.INPUT_FILES["outlines_1935"],
        output_filepath=CACHE_FILES["glacier_mask"],
        overwrite=overwrite,
        resolution=resolution)


def generate_stable_ground_mask(overwrite: bool = False, resolution: float = CONSTANTS.dem_resolution):

    if not overwrite and os.path.isfile(CACHE_FILES["stable_ground_mask"]):
        return
    # Make/read the glacier mask
    generate_glacier_mask(overwrite=overwrite, resolution=resolution)
    glacier_mask = rio.open(CACHE_FILES["glacier_mask"])

    land_use = gpd.read_file(files.INPUT_FILES["lake_outlines"]).to_crs(CONSTANTS.crs_epsg.replace("::", ":"))
    lakes = land_use.copy()  # [land_use["CODE_18"] == "512"]

    temp_dir = tempfile.TemporaryDirectory()
    lakes_temp_path = os.path.join(temp_dir.name, "lakes.shp")
    rasterized_lakes_temp_path = os.path.join(temp_dir.name, "lakes_rasterized.tif")

    lakes.to_file(lakes_temp_path)

    rasterize_outlines(lakes_temp_path, rasterized_lakes_temp_path, resolution=resolution)

    lake_mask = rio.open(rasterized_lakes_temp_path)

    # Get the area where neither lakes nor glaciers exist
    stable_ground_mask = (lake_mask.read(1) != 1) & (glacier_mask.read(1) != 1)

    with rio.open(
            CACHE_FILES["stable_ground_mask"],
            mode="w",
            driver="GTiff",
            width=glacier_mask.width,
            height=glacier_mask.height,
            count=1,
            crs=glacier_mask.crs,
            transform=glacier_mask.transform,
            dtype=np.uint8) as raster:
        raster.write(stable_ground_mask.astype(np.uint8), 1)


def read_mask(filepath: str, bounds: dict[str, float], resolution: float = CONSTANTS.dem_resolution,
              crs: rio.crs.CRS = None) -> np.ndarray:
    """
    Read a mask and crop/resample it to the given bounds.

    Uses nearest neighbour if the target resolution is the same, bilinear if larger, and cubic spline if lower.

    :param filepath: The path to the mask file.
    :param bounds: A dictionary with the keys: west, east, south, north.
    :param resolution: The target resolution of the mask file.
    :param crs: An optional other CRS. Defaults to the mask CRS.
    :returns: A boolean numpy array with a shape corresponding to the given bounds and resolution.
    """
    mask = rio.open(filepath)

    if abs(resolution - mask.res[0]) < 1e-2:
        resampling_method = rio.warp.Resampling.nearest
    elif resolution > mask.res[0]:
        resampling_method = rio.warp.Resampling.bilinear
    elif resolution < mask.res[0]:
        resampling_method = rio.warp.Resampling.cubic_spline

    # Calculate new shape of the dataset
    dst_shape = (int((bounds["north"] - bounds["south"]) // resolution),
                 int((bounds["east"] - bounds["west"]) // resolution))

    # Make an Affine transform from the bounds and the new size
    dst_transform = rio.transform.from_bounds(**bounds, width=dst_shape[1], height=dst_shape[0])
    # Make an empty numpy array which will later be filled with elevation values
    resampled_values = np.empty(dst_shape, mask.dtypes[0])
    # Set all values to nan right now
    resampled_values[:, :] = np.nan

    # Reproject the DEM and put the output in the destination array
    rio.warp.reproject(
        source=mask.read(1),
        destination=resampled_values,
        src_transform=mask.transform,
        dst_transform=dst_transform,
        resampling=resampling_method,
        src_crs=mask.crs,
        dst_crs=mask.crs if crs is None else crs
    )

    # Convert the float(?) array to a boolean.
    return resampled_values == 1


def read_glacier_mask(bounds: dict[str, float], resolution: float = CONSTANTS.dem_resolution) -> np.ndarray:
    """
    Read and crop/resample the glacier mask to the given bounds and resolution.

    :param bounds: A dictionary with the keys: west, east, south, north.
    :param resolution: The target resolution of the mask.
    :returns: A boolean numpy array with a shape corresponding to the given bounds and resolution.
    """
    return read_mask(CACHE_FILES["glacier_mask"], bounds=bounds, resolution=resolution)


def read_stable_ground_mask(bounds: dict[str, float], resolution: float = CONSTANTS.dem_resolution) -> np.ndarray:
    """
    Read and crop/resample the stable ground mask to the given bounds and resolution.

    :param bounds: A dictionary with the keys: west, east, south, north.
    :param resolution: The target resolution of the mask.
    :returns: A boolean numpy array with a shape corresponding to the given bounds and resolution.
    """
    return read_mask(CACHE_FILES["stable_ground_mask"], bounds=bounds, resolution=resolution)


def fix_freudinger_outlines():
    min_overlap_threshold = 0.05
    temp_dir = os.path.join(files.TEMP_DIRECTORY, "outlines")
    freudinger = gpd.read_file(files.INPUT_FILES["outlines_1935"])
    sgi_2016 = gpd.read_file(files.INPUT_FILES["sgi_2016"]).to_crs(CONSTANTS.crs_epsg.replace("::", ":"))

    new_outlines = gpd.GeoDataFrame(columns=["EZGNR", "sgi-ids", "geometry"], crs=sgi_2016.crs)
    for i, row in tqdm(freudinger.iterrows(), total=freudinger.shape[0]):
        overlapping = sgi_2016.geometry.overlaps(row.geometry)
        if np.count_nonzero(overlapping.values) == 0:
            continue
        diff = sgi_2016[overlapping].geometry.intersection(row.geometry)

        fraction = diff.area / row.geometry.area

        mask = overlapping.copy()
        mask[mask] = fraction > min_overlap_threshold

        sorting = np.argsort(fraction[fraction > min_overlap_threshold])[::-1]

        sgi_ids = sgi_2016[mask].iloc[sorting]["sgi-id"].values
        sgi_col = ",".join(sgi_ids)

        merged_geom = shapely.ops.unary_union(np.r_[sgi_2016[mask].geometry.values, row.geometry])

        new_outlines.loc[i] = row["EZGNR"], sgi_col, merged_geom

    new_outlines.to_file(os.path.join(temp_dir, "new_outlines.shp"))
    print(new_outlines)


def fix_lk50_outlines():
    pd.set_option('mode.chained_assignment', "raise")

    lk50_sgi1973 = gpd.read_file(files.INPUT_FILES["lk50_modified_sgi_1973"])
    merged_sgi_lk50 = gpd.GeoDataFrame(columns=lk50_sgi1973.columns, crs=lk50_sgi1973.crs)

    # Merge all shapes that have the same SGI id.
    for sgi, glaciers in tqdm(lk50_sgi1973.groupby("SGI"), total=np.unique(lk50_sgi1973["SGI"]).shape[0],
                              desc="Merging SGI labels"):
        glacier = glaciers.iloc[0].copy()
        glacier["SGI"] = sgi
        if glaciers.shape[0] > 1:
            for _, glacier2 in glaciers.iterrows():
                # Skip if self-comparison
                if glacier["OBJECTID"] == glacier2["OBJECTID"]:
                    continue
                glacier.geometry = glacier.geometry.union(glacier2.geometry)

        merged_sgi_lk50.loc[merged_sgi_lk50.shape[0]] = glacier

    assert np.unique(merged_sgi_lk50["SGI"]).shape[0] == merged_sgi_lk50.shape[0]
    merged_parent_lk50 = gpd.GeoDataFrame(columns=lk50_sgi1973.columns, crs=lk50_sgi1973.crs)

    # Merge all shapes that have the same parent and that touch.
    for parent, glaciers in tqdm(merged_sgi_lk50.groupby("Parent1850"),
                                 desc="Merging touching polygons with same parent"):
        glaciers_copy = glaciers.copy()
        glaciers_copy["Parent1850"] = parent
        if parent is None:
            merged_parent_lk50 = merged_parent_lk50.append(glaciers, ignore_index=True)
            continue

        glaciers_copy["area"] = glaciers.geometry.apply(lambda x: x.area)
        glaciers_copy.sort_values("area", ascending=False, inplace=True)
        glacier = glaciers_copy.iloc[0].copy()

        if glaciers.shape[0] > 1:
            for _, glacier2 in glaciers.iterrows():
                # Skip if self-comparison
                if glacier["SGI"] == glacier2["SGI"]:
                    continue
                # Merge geometries if they touch
                if glacier.geometry.touches(glacier2.geometry):
                    glacier.geometry = glacier.geometry.union(glacier2.geometry)
                # Otherwise, add glacier2 to the output dataframe
                else:
                    merged_parent_lk50.loc[merged_parent_lk50.shape[0]] = glacier2[glaciers.columns]

        # Add the biggest glacier to the dataframe
        merged_parent_lk50.loc[merged_parent_lk50.shape[0]] = glacier[glaciers.columns]

    sgi_1973 = gpd.read_file(files.INPUT_FILES["sgi_1973"]).to_crs(lk50_sgi1973.crs)
    sgi_1850 = gpd.read_file(files.INPUT_FILES["sgi_1850"]).to_crs(lk50_sgi1973.crs)

    image_metadata = image_meta.read_metadata()

    lk50 = gpd.GeoDataFrame(columns=list(lk50_sgi1973.columns) + ["changed_from_1973", "date"], crs=lk50_sgi1973.crs)
    for i, glacier in tqdm(merged_parent_lk50.iterrows(), total=merged_parent_lk50.shape[0],
                           desc="Validating geometries and finding proper date"):
        glacier_1973 = sgi_1973.loc[sgi_1973["OBJECTID"] == glacier["OBJECTID"]].iloc[0]

        # Check if the difference is larger than 10 m2. If so, it is assumed to have been changed from the 1973 outline.
        changed_from_1973 = abs(glacier.geometry.area - glacier_1973.geometry.area) > 10
        glacier["changed_from_1973"] = changed_from_1973

        centroid = glacier.geometry.centroid
        # Calculate the distance to all cameras
        image_metadata["dist"] = np.linalg.norm([image_metadata["easting"] - centroid.x,
                                                 image_metadata["northing"] - centroid.y
                                                 ], axis=0)
        # Sort the values by the distance
        image_metadata.sort_values("dist", inplace=True)
        # Take the date of the closest camera.
        glacier["date"] = str(image_metadata.iloc[0]["date"])

        # If it has been changed since 1973, check that it is indeed larger, and that it is smaller than in 1850.
        if changed_from_1973:
            fixed_geom = glacier.geometry.union(glacier_1973.geometry)

            # First try to find the 1850 parent
            potential_glacier_1850 = sgi_1850.loc[sgi_1850["SGI"] == glacier["Parent1850"]]
            # If that doesn't work, try to find the 1850 glacier by its SGI id
            # if potential_glacier_1850.shape[0] == 0:
            #    potential_glacier_1850 = sgi_1850.loc[sgi_1850["SGI"] == glacier["SGI"]]
            # If that succeeded, fix so the glacier's extent does not extend farther than in 1850.
            if potential_glacier_1850.shape[0] != 0:
                try:
                    # Shapely sends a comparison exception as a warning, so it should be treated appropriately.
                    with warnings.catch_warnings():
                        glacier_1850 = potential_glacier_1850.iloc[0].copy()
                        # Merge geometries if there are more than one SGI entries of the same name
                        if potential_glacier_1850.shape[0] > 1:
                            for j in range(1, potential_glacier_1850.shape[0]):
                                glacier_1850.geometry = glacier_1850.geometry.buffer(0).union(
                                    potential_glacier_1850.iloc[j].geometry.buffer(0))
                        warnings.simplefilter("always")
                        outside_1850 = fixed_geom.buffer(0).difference(glacier_1850.geometry.buffer(0))
                        # Try to loop over the presumed multi-polygon
                        try:
                            for polygon in outside_1850:
                                fixed_geom = fixed_geom.difference(polygon)
                        except TypeError:  # TypeError comes up if it's a single polygon
                            fixed_geom = fixed_geom.difference(outside_1850)
                except shapely.errors.TopologicalError:
                    pass

            # Measure the old (1973) and new (LK50) areas to make sure that the geometry did not turn invalid somehow.
            if fixed_geom.geom_type not in ["MultiPolygon", "Polygon"]:
                continue

            new_area = sum([geom.area for geom in fixed_geom]) \
                if fixed_geom.geom_type == "MultiPolygon" else fixed_geom.area
            old_area = sum([geom.area for geom in glacier_1973.geometry]) \
                if glacier_1973.geometry.geom_type == "MultiPolygon" else glacier_1973.geometry.area
            if new_area > old_area and fixed_geom.is_valid:
                glacier.geometry = fixed_geom

        lk50.loc[i] = glacier

    # There is a new modified column. The length and area columns are now old!
    lk50.drop(columns=["Shape_Leng", "Shape_Area", "modified"], inplace=True)

    lk50.to_file(CACHE_FILES["lk50_outlines"])
    print(lk50)


if __name__ == "__main__":
    fix_lk50_outlines()
