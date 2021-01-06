from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import subprocess
import tempfile
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import rasterio.warp  # pylint: disable=unused-import
from tqdm import tqdm

from terra import files
from terra.constants import CONSTANTS
from terra.processing import inputs, processing_tools

TEMP_DIRECTORY = os.path.join(files.TEMP_DIRECTORY, "evaluation")

CACHE_FILES = {
    "metashape_dems_dir": os.path.join(inputs.TEMP_DIRECTORY, "output/dems"),
    "ddems_dir": os.path.join(TEMP_DIRECTORY, "ddems/"),
    "dem_coreg_dir": os.path.join(TEMP_DIRECTORY, "coreg/"),
    "dem_coreg_meta_dir": os.path.join(TEMP_DIRECTORY, "coreg_meta/"),
    "glacier_mask": os.path.join(TEMP_DIRECTORY, "glacier_mask.tif"),
}


def find_dems(folder: str) -> list[str]:
    """Find all .tif files in a folder."""
    dem_names = [os.path.join(folder, filename) for filename in os.listdir(folder) if filename.endswith(".tif")]

    return dem_names


def reproject_dem(dem: rio.DatasetReader, bounds: dict[str, float],
                  resolution: float, crs: Optional[rio.crs.CRS] = None) -> np.ndarray:
    """
    Reproject a DEM to the given bounds.

    :param dem: A DEM read through rasterio.
    :param bounds: The target west, east, north, and south bounding coordinates.
    :param resolution: The target resolution in metres.
    :param crs: Optional. The target CRS (defaults to the input DEM crs)
    :returns: The elevation array in the destination bounds, resolution and CRS.
    """
    # Calculate new shape of the dataset
    dst_shape = (int((bounds["north"] - bounds["south"]) // resolution),
                 int((bounds["east"] - bounds["west"]) // resolution))

    # Make an Affine transform from the bounds and the new size
    dst_transform = rio.transform.from_bounds(**bounds, width=dst_shape[1], height=dst_shape[0])
    # Make an empty numpy array which will later be filled with elevation values
    try:
        destination = np.empty(dst_shape, dem.dtypes[0])
    except AttributeError:
        print(dem)
        print(dir(dem))
        raise AttributeError
    # Set all values to nan right now
    destination[:, :] = np.nan

    # Reproject the DEM and put the output in the destination array
    rio.warp.reproject(
        source=dem.read(1),
        destination=destination,
        src_transform=dem.transform,
        dst_transform=dst_transform,
        resampling=rio.warp.Resampling.cubic_spline,
        src_crs=dem.crs,
        dst_crs=dem.crs if crs is None else crs
    )

    return destination


def load_reference_elevation(bounds: dict[str, float], resolution: float = CONSTANTS.dem_resolution) -> np.ndarray:
    """
    Load the reference DEM and reproject it to the given bounds.

    Due to memory constraints, it is first cropped with GDAL, and then resampled with Rasterio.

    :param bounds: east, west, north and south bounding coordinates to crop the raster with.
    :param resolution: The target resolution. Defaults to the generated DEM resolution.
    :returns: An array of elevation values corresponding to the given bounds and resolution.
    """
    # Expand the bounds slightly to crop a larger area with GDAL (to easily resample afterward)
    larger_bounds = bounds.copy()
    for key in ["west", "south"]:
        larger_bounds[key] -= CONSTANTS.dem_resolution
    for key in ["east", "north"]:
        larger_bounds[key] += CONSTANTS.dem_resolution

    # Generate a temporary directory to save the cropped DEM in
    temp_dir = tempfile.TemporaryDirectory()
    temp_dem_path = os.path.join(temp_dir.name, "dem.tif")

    # Crop the very large reference DEM to the expanded bounds.
    gdal_commands = [
        "gdal_translate",
        "-projwin",
        larger_bounds["west"],
        larger_bounds["north"],
        larger_bounds["east"],
        larger_bounds["south"],
        files.INPUT_FILES["base_DEM"],
        temp_dem_path
    ]
    subprocess.run(list(map(str, gdal_commands)), check=True, stdout=subprocess.PIPE)

    # Open the cropped DEM in rasterio and resample it to fit the bounds perfectly.
    reference_dem = rio.open(temp_dem_path)
    reference_elevation = reproject_dem(reference_dem, bounds, resolution=resolution)
    reference_dem.close()

    reference_elevation[reference_elevation > CONSTANTS.max_height] = np.nan
    reference_elevation[reference_elevation < -9999] = np.nan

    return reference_elevation


def compare_dem(filepath: str) -> None:
    """Compare the DEM difference between the reference DEM and the given filepath."""
    dem = rio.open(filepath)
    dem_elevation = dem.read(1)
    dem_elevation[dem_elevation < -999] = np.nan
    dem_elevation[dem_elevation > CONSTANTS.max_height] = np.nan

    # Check if it's only NaNs
    if np.all(~np.isfinite(dem_elevation)):
        return

    bounds = dict(zip(["west", "south", "east", "north"], list(dem.bounds)))
    reference_elevation = load_reference_elevation(bounds)
    ddem = reference_elevation - dem_elevation

    # Get the station name from the filepath, e.g. station_3500
    station_name = filepath[filepath.index("station_"):filepath.index("station_") + 12]

    # Write the DEM difference product.
    with rio.open(
            os.path.join(CACHE_FILES["ddems_dir"], f"{station_name}_ddem.tif"),
            mode="w",
            driver="GTiff",
            width=ddem.shape[1],
            height=ddem.shape[0],
            crs=dem.crs,
            transform=rio.transform.from_bounds(**bounds, width=ddem.shape[1], height=ddem.shape[0]),
            dtype=ddem.dtype,
            count=1) as raster:
        raster.write(ddem, 1)


def generate_glacier_mask(overwrite: bool = False, resolution: float = CONSTANTS.dem_resolution):
    """Generate a boolean glacier mask from the 1935 map."""
    # Skip if it already exists
    if not overwrite and os.path.isfile(files.INPUT_FILES["base_DEM"]):
        return
    # Get the bounds from the reference DEM
    reference_bounds = json.loads(
        subprocess.run(
            ["gdalinfo", "-json", files.INPUT_FILES["base_DEM"]],
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
        files.INPUT_FILES["outlines_1935"],
        CACHE_FILES["glacier_mask"]
    ]
    subprocess.run(list(map(str, gdal_commands)), check=True, stdout=subprocess.PIPE)

    # Unset the nodata value to correctly display in e.g. QGIS
    subprocess.run(["gdal_edit.py", "-unsetnodata", CACHE_FILES["glacier_mask"]],
                   check=True, stdout=subprocess.PIPE)


def coregister_dem(filepath: str, glacier_mask_path: str) -> Optional[dict[str, Any]]:
    """
    Coregister a DEM to the reference using ICP coregistration.

    :param filepath: The filepath of the DEM to be aligned.
    :param glacier_mask_path: The filepath to the glacier mask to exclude glaciers with.
    :returns: A modified version of the PDAL ICP metadata. Returns None if the process failed.
    """
    dem = rio.open(filepath)
    dem_elevation = dem.read(1)
    dem_elevation[dem_elevation < -999] = np.nan

    # Check that valid values exist in the DEM
    if np.all(~np.isfinite(dem_elevation)):
        return None

    bounds = dict(zip(["west", "south", "east", "north"], list(dem.bounds)))

    # Load and crop/resample the glacier mask and reference DEM
    # The glacier mask is converted to a boolean array using the "== 1" comparison.
    glacier_mask_dataset = rio.open(glacier_mask_path)
    cropped_glacier_mask = reproject_dem(glacier_mask_dataset, bounds, CONSTANTS.dem_resolution) == 1
    glacier_mask_dataset.close()
    cropped_reference_dem = load_reference_elevation(bounds)

    # Set all glacier values to nan
    dem_elevation[cropped_glacier_mask] = np.nan
    cropped_reference_dem[cropped_glacier_mask] = np.nan

    # Create a temporary directory and temporary filenames for the analysis.
    temp_dir = tempfile.TemporaryDirectory()
    temp_dem_path = os.path.join(temp_dir.name, "dem.tif")
    temp_ref_path = os.path.join(temp_dir.name, "ref_dem.tif")
    result_path = os.path.join(temp_dir.name, "result_meta.json")

    # Determine the name of the output DEM
    station_name = filepath[filepath.rindex("station_"):filepath.index("_dense_")]
    aligned_dem_path = os.path.join(CACHE_FILES["dem_coreg_dir"], f"{station_name}_coregistered.tif")

    # Save the cropped and glacier-masked DEM and reference DEM
    rio_params = dict(
        mode="w",
        driver="GTiff",
        width=dem_elevation.shape[1],
        height=dem_elevation.shape[0],
        crs=dem.crs,
        transform=rio.transform.from_bounds(**bounds, width=dem_elevation.shape[1], height=dem_elevation.shape[0]),
        dtype=dem_elevation.dtype,
        count=1
    )
    with rio.open(temp_dem_path, **rio_params) as raster:
        raster.write(dem_elevation, 1)
    with rio.open(temp_ref_path, **rio_params) as raster:
        raster.write(cropped_reference_dem, 1)

    # Run the DEM coalignment
    processing_tools.coalign_dems(
        reference_path=temp_ref_path,
        aligned_path=temp_dem_path,
        pixel_buffer=5,
        temp_dir=temp_dir.name
    )

    # Load the resulting statistics
    with open(result_path) as infile:
        stats = json.load(infile)["stages"]["filters.icp"]

    # Use the resultant transformation to transform the original DEM
    processing_tools.run_pdal_pipeline(
        pipeline="""
        [
            {
                "type": "readers.gdal",
                "filename": "INFILE",
                "header": "Z"
            },
            {
                "type": "filters.range",
                "limits": "Z[-999:MAX_HEIGHT]"
            },
            {
                "type": "filters.transformation",
                "matrix": "MATRIX"
            },
            {
                "type": "writers.gdal",
                "resolution": RESOLUTION,
                "bounds": "([WEST,EAST],[SOUTH,NORTH])",
                "output_type": "mean",
                "data_type": "float32",
                "gdalopts": ["COMPRESS=DEFLATE", "PREDICTOR=3"],
                "filename": "OUTFILE"
            }
        ]""",
        parameters={
            "INFILE": filepath,
            "MAX_HEIGHT": str(CONSTANTS.max_height),
            "MATRIX": stats["composed"].replace("\n", " "),
            "RESOLUTION": str(CONSTANTS.dem_resolution),
            "WEST": bounds["west"],
            "EAST": bounds["east"],
            "SOUTH": bounds["south"],
            "NORTH": bounds["north"],
            "OUTFILE": aligned_dem_path
        }
    )
    # Fix the CRS information of the aligned DEM
    subprocess.run(["gdal_edit.py", "-a_srs", f"EPSG:{dem.crs.to_epsg()}", aligned_dem_path], check=True)

    # Count the overlapping pixels.
    # TODO: Maybe threshold the overlap to remove bad alignments?
    stats["overlap"] = np.count_nonzero(np.isfinite(cropped_reference_dem) & np.isfinite(dem_elevation))
    with open(os.path.join(CACHE_FILES["dem_coreg_meta_dir"], f"{station_name}_coregistration.json"), "w") as outfile:
        json.dump(stats, outfile)

    return stats


def coregister_all_dems():
    dem_filepaths = find_dems(CACHE_FILES["metashape_dems_dir"])
    progress_bar = tqdm(total=len(dem_filepaths))

    os.makedirs(CACHE_FILES["dem_coreg_dir"], exist_ok=True)
    os.makedirs(CACHE_FILES["dem_coreg_meta_dir"], exist_ok=True)

    def coregister_filepath(filepath):
        """Coregister a DEM and update the progress bar."""
        coregister_dem(filepath, CACHE_FILES["glacier_mask"])
        progress_bar.update()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        list(executor.map(coregister_filepath, dem_filepaths))

    progress_bar.close()


def compare_all_dems(folder: str = CACHE_FILES["dem_coreg_dir"]):

    dem_filepaths = find_dems(folder)
    progress_bar = tqdm(total=len(dem_filepaths))

    os.makedirs(CACHE_FILES["ddems_dir"], exist_ok=True)

    def compare_filepath(filepath):
        """Compare a DEM and update the progress bar."""
        compare_dem(filepath)
        progress_bar.update()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        list(executor.map(compare_filepath, dem_filepaths))


if __name__ == "__main__":

    generate_glacier_mask()
    coregister_all_dems()
    compare_all_dems()
