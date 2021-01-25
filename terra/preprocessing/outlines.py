"""Stable-ground / glacier outlines and mask functions."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile

import geopandas as gpd
import numpy as np
import rasterio as rio

from terra import base_dem, files
from terra.constants import CONSTANTS

TEMP_DIRECTORY = os.path.join(files.TEMP_DIRECTORY, "preprocessing/")
CACHE_FILES = {
    "glacier_mask": os.path.join(TEMP_DIRECTORY, "glacier_mask.tif"),
    "stable_ground_mask": os.path.join(TEMP_DIRECTORY, "stable_ground_mask.tif")
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


if __name__ == "__main__":
    generate_stable_ground_mask()
