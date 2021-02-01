"""Tools for DEM, dDEM and orthomosaic processing."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import Any, Optional

import cv2
import numpy as np
import rasterio as rio
from tqdm import tqdm

from terra import base_dem, evaluation, files
from terra.constants import CONSTANTS
from terra.evaluation import CACHE_FILES
from terra.preprocessing import image_meta, outlines
from terra.processing import inputs, processing_tools


def find_dems(folder: str) -> list[str]:
    """Find all .tif files in a folder."""
    dem_names = [os.path.join(folder, filename) for filename in os.listdir(folder) if filename.endswith(".tif")]

    return dem_names


def extract_station_name(filepath: str) -> str:
    """Find the station name (e.g. station_3500) inside a filepath."""
    return filepath[filepath.index("station_"):filepath.index("station_") + 12]


def reproject_dem(dem: rio.DatasetReader, bounds: dict[str, float], resolution: float, band: int = 1,
                  crs: Optional[rio.crs.CRS] = None, resampling=rio.warp.Resampling.cubic_spline) -> np.ndarray:
    """
    Reproject a DEM to the given bounds.

    :param dem: A DEM read through rasterio.
    :param bounds: The target west, east, north, and south bounding coordinates.
    :param resolution: The target resolution in metres.
    :param band: The band number. Default=1.
    :param crs: Optional. The target CRS (defaults to the input DEM crs)
    :returns: The elevation array in the destination bounds, resolution and CRS.
    """
    # Calculate new shape of the dataset
    dst_shape = (int((bounds["north"] - bounds["south"]) // resolution),
                 int((bounds["east"] - bounds["west"]) // resolution))

    # Make an Affine transform from the bounds and the new size
    dst_transform = rio.transform.from_bounds(**bounds, width=dst_shape[1], height=dst_shape[0])
    # Make an empty numpy array which will later be filled with elevation values
    destination = np.empty(dst_shape, dem.dtypes[0])
    # Set all values to nan right now
    destination[:, :] = np.nan

    # Reproject the DEM and put the output in the destination array
    rio.warp.reproject(
        source=dem.read(band),
        destination=destination,
        src_transform=dem.transform,
        dst_transform=dst_transform,
        resampling=resampling,
        src_crs=dem.crs,
        dst_crs=dem.crs if crs is None else crs
    )

    return destination


def load_reference_elevation(bounds: dict[str, float], resolution: float = CONSTANTS.dem_resolution, base_dem_prefix="base_dem") -> np.ndarray:
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
        base_dem.CACHE_FILES[base_dem_prefix],
        temp_dem_path
    ]
    subprocess.run(list(map(str, gdal_commands)), check=True, stdout=subprocess.PIPE)

    # Open the cropped DEM in rasterio and resample it to fit the bounds perfectly.
    reference_dem = rio.open(temp_dem_path)
    resampling_method = rio.warp.Resampling.cubic_spline if resolution < CONSTANTS.dem_resolution\
        else rio.warp.Resampling.bilinear
    reference_elevation = reproject_dem(reference_dem, bounds, resolution=resolution, resampling=resampling_method)
    reference_dem.close()

    reference_elevation[reference_elevation > CONSTANTS.max_height] = np.nan
    reference_elevation[reference_elevation < -9999] = np.nan

    return reference_elevation


def generate_ddem(filepath: str, save: bool = True, output_dir: str = CACHE_FILES["ddems_dir"]) -> np.ndarray:
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
    station_name = extract_station_name(filepath)

    # Write the DEM difference product.
    if save:
        with rio.open(
                os.path.join(output_dir, f"{station_name}_ddem.tif"),
                mode="w",
                driver="GTiff",
                width=ddem.shape[1],
                height=ddem.shape[0],
                crs=dem.crs,
                transform=rio.transform.from_bounds(**bounds, width=ddem.shape[1], height=ddem.shape[0]),
                dtype=ddem.dtype,
                count=1) as raster:
            raster.write(ddem, 1)

    return ddem


def generate_glacier_mask(overwrite: bool = False, resolution: float = CONSTANTS.dem_resolution) -> None:
    """Generate a boolean glacier mask from the 1935 map."""
    raise DeprecationWarning
    # Skip if it already exists
    if not overwrite and os.path.isfile(outlines.CACHE_FILES["glacier_mask"]):
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
        files.INPUT_FILES["outlines_1935"],
        outlines.CACHE_FILES["glacier_mask"]
    ]
    subprocess.run(list(map(str, gdal_commands)), check=True, stdout=subprocess.PIPE)

    # Unset the nodata value to correctly display in e.g. QGIS
    subprocess.run(["gdal_edit.py", "-unsetnodata", outlines.CACHE_FILES["glacier_mask"]],
                   check=True, stdout=subprocess.PIPE)


def coregister_dem(filepath: str) -> Optional[dict[str, Any]]:
    """
    Coregister a DEM to the reference using ICP coregistration.

    :param filepath: The filepath of the DEM to be aligned.
    :returns: A modified version of the PDAL ICP metadata. Returns None if the process failed.
    """
    dem = rio.open(filepath)
    dem_elevation = dem.read(1)
    dem_elevation[dem_elevation < -999] = np.nan

    # Check that valid values exist in the DEM
    if np.all(~np.isfinite(dem_elevation)):
        return None

    bounds = dict(zip(["west", "south", "east", "north"], list(dem.bounds)))

    # Load and crop/resample the stable ground mask and reference DEM
    stable_ground_mask = outlines.read_stable_ground_mask(bounds)
    cropped_reference_dem = load_reference_elevation(bounds)

    # Set all non-stable values to nan
    dem_elevation[~stable_ground_mask] = np.nan
    cropped_reference_dem[~stable_ground_mask] = np.nan

    if np.all(np.isnan(cropped_reference_dem)):
        return None

    # Create a temporary directory and temporary filenames for the analysis.
    temp_dir = tempfile.TemporaryDirectory()
    temp_dem_path = os.path.join(temp_dir.name, "dem.tif")
    temp_ref_path = os.path.join(temp_dir.name, "ref_dem.tif")
    result_path = os.path.join(temp_dir.name, "result_meta.json")

    # Determine the name of the output DEM
    station_name = extract_station_name(filepath)
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

    for path in [temp_dem_path, temp_ref_path]:
        assert os.path.isfile(path)

    # Run the DEM coalignment
    processing_tools.coalign_dems(
        reference_path=temp_ref_path,
        aligned_path=temp_dem_path,
        pixel_buffer=5,
        temp_dir=temp_dir.name
    )

    # Load the resulting statistics
    try:
        with open(result_path) as infile:
            stats = json.load(infile)["stages"]["filters.icp"]
    except FileNotFoundError:
        print(f"Filepath {filepath} stats not found. Failed alignment?")
        return None

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
    stats["overlap"] = np.count_nonzero(np.isfinite(cropped_reference_dem) & np.isfinite(dem_elevation))
    with open(os.path.join(CACHE_FILES["dem_coreg_meta_dir"], f"{station_name}_coregistration.json"), "w") as outfile:
        json.dump(stats, outfile)

    return stats


def coregister_all_dems(overwrite: bool = False) -> None:
    """
    Run the coregistration over all of the DEMs created in Metashape.

    :param overwrite: Overwrite existing coregistered DEMs?
    """
    dem_filepaths = find_dems(CACHE_FILES["metashape_dems_dir"])

    os.makedirs(CACHE_FILES["dem_coreg_dir"], exist_ok=True)
    os.makedirs(CACHE_FILES["dem_coreg_meta_dir"], exist_ok=True)

    progress_bar = tqdm(total=len(dem_filepaths), desc="Coregistering DEMs")

    for filepath in dem_filepaths:
        progress_bar.desc = filepath
        # Determine the name of the output DEM
        station_name = extract_station_name(filepath)
        aligned_dem_path = os.path.join(CACHE_FILES["dem_coreg_dir"], f"{station_name}_coregistered.tif")
        if not overwrite and os.path.isfile(aligned_dem_path):
            progress_bar.update()
            continue
        coregister_dem(filepath)
        progress_bar.update()


def generate_all_ddems(dem_dir: str = CACHE_FILES["dem_coreg_dir"],
                       out_dir: str = CACHE_FILES["ddems_coreg_dir"], overwrite: bool = False) -> None:
    """
    Generate dDEMs for all DEMs in a directory.

    The DEMs are compared to the reference 'base_DEM' defined in files.py.

    :param dem_dir: The directory of the DEMs to compare.
    :param out_dir: The output directory of the dDEMs.
    """
    dem_filepaths = find_dems(dem_dir)
    if not overwrite:
        for filepath in dem_filepaths.copy():
            output_filepath = os.path.join(out_dir, f"{extract_station_name(filepath)}_ddem.tif")
            if os.path.isfile(output_filepath):
                dem_filepaths.remove(filepath)

    if len(dem_filepaths) == 0:
        return

    os.makedirs(out_dir, exist_ok=True)
    for filepath in tqdm(dem_filepaths, desc="Generating dDEMs"):
        generate_ddem(filepath, output_dir=out_dir)


def read_and_crop_glacier_mask(raster: rio.DatasetReader, resampling=rio.warp.Resampling.cubic_spline) -> np.ndarray:
    """
    Read the glacier mask and crop it to the input raster dataset.

    :param raster: A Rasterio dataset to crop the glacier mask to.
    :returns: A boolean array with the same dimensions and bounds as the input dataset.
    """
    bounds = dict(zip(["west", "south", "east", "north"], list(raster.bounds)))
    glacier_mask_dataset = rio.open(outlines.CACHE_FILES["glacier_mask"])
    cropped_glacier_mask = reproject_dem(glacier_mask_dataset, bounds,
                                         CONSTANTS.dem_resolution, resampling=resampling) == 1
    glacier_mask_dataset.close()

    return cropped_glacier_mask


def make_yearly_ddems(overwrite: bool = False):
    """
    Generate yearly dDEMs from their given date difference.

    :param overwrite: Overwrite already existing dDEMs?
    """
    ddem_filepaths = find_dems(CACHE_FILES["ddems_coreg_dir"])
    image_metadata = image_meta.read_metadata()
    image_metadata["station_number"] = image_metadata["Base number"]\
        .str.replace("_A", "")\
        .str.replace("_B", "")\
            .astype(int)

    os.makedirs(CACHE_FILES["ddems_yearly_coreg_dir"], exist_ok=True)
    for filepath in tqdm(ddem_filepaths, desc="Making yearly dDEMs"):
        station_name = extract_station_name(filepath)
        output_filepath = os.path.join(CACHE_FILES["ddems_yearly_coreg_dir"], f"{station_name}_yearly_ddem.tif")
        if not overwrite and os.path.isfile(output_filepath):
            continue
        station_number = int(station_name.replace("station_", ""))
        date = image_metadata.loc[image_metadata["station_number"] == station_number, "date"].iloc[0]
        age_years = (CONSTANTS.base_dem_date - date).total_seconds() / 31556952

        ddem = rio.open(filepath)

        ddem_values = ddem.read(1)

        with rio.open(
                output_filepath,
                mode="w",
                driver="GTiff",
                width=ddem.width,
                height=ddem.height,
                count=1,
                crs=ddem.crs,
                transform=ddem.transform,
                dtype=ddem_values.dtype,
                nodata=ddem.nodata) as raster:
            raster.write(ddem_values / age_years, 1)

        ddem.close()


def merge_rasters(raster_filepaths: list[str], output_filepath: str = CACHE_FILES["merged_ddem"]) -> None:
    """
    Merge multiple GeoTiffs into one file using mean blending.

    IMPORTANT: The GeoTiffs are assumed to be in the same resolution and grid.

    :param raster_filepaths: A list of GeoTiff filepaths to merge.
    :param output_filepath: The filepath of the output raster.
    """
    # Find the maximum bounding box of the rasters.
    bounds: dict[str, float] = {}
    for filepath in raster_filepaths:
        raster = rio.open(filepath)

        bounds["west"] = min(bounds.get("west") or 1e22, raster.bounds.left)
        bounds["east"] = max(bounds.get("east") or -1e22, raster.bounds.right)
        bounds["south"] = min(bounds.get("south") or 1e22, raster.bounds.bottom)
        bounds["north"] = max(bounds.get("north") or -1e22, raster.bounds.top)

        dtype = raster.dtypes[0]
        raster.close()

    # Find the corresponding output shape of the merged raster
    output_shape = (
        int((bounds["north"] - bounds["south"]) / CONSTANTS.dem_resolution),
        int((bounds["east"] - bounds["west"]) / CONSTANTS.dem_resolution),
    )

    # Generate a new merged raster with only NaN values
    merged_raster = np.zeros(shape=output_shape, dtype=np.float32)
    # Generate a similarly shaped count array. This is used for the mean calculation
    # Values are added to the merged_raster, then divided by the value_count to get the mean
    value_count = np.zeros(merged_raster.shape, dtype=np.uint8)

    for filepath in tqdm(raster_filepaths, desc="Merging rasters"):
        raster = rio.open(filepath)

        # Find the top left pixel index
        top_index = int((bounds["north"] - raster.bounds.top) / CONSTANTS.dem_resolution)
        left_index = int((raster.bounds.left - bounds["west"]) / CONSTANTS.dem_resolution)

        raster_values = raster.read(1).astype(np.float32)
        if raster.dtypes[0] == "uint8":
            raster_values[raster_values == 255] = np.nan
        # Add the raster value to the merged raster
        merged_raster[top_index:top_index + raster.height,
                      left_index:left_index + raster.width] += np.nan_to_num(raster_values, nan=0.0)
        # Increment the value count where the raster values were not NaN
        value_count[top_index:top_index + raster.height, left_index:left_index +
                    raster.width] += np.logical_not(np.isnan(raster_values)).astype(np.uint8)

        raster.close()

    # Set all cells with no valid values to nan
    merged_raster[value_count == 0] = np.nan

    # Divide by the value count to get the mean (instead of the sum)
    merged_raster /= value_count

    nodata_value = -9999 if not dtype == "uint8" else 255
    merged_raster[value_count == 0] = nodata_value

    # Save the merged raster
    transform = rio.transform.from_bounds(**bounds, width=merged_raster.shape[1], height=merged_raster.shape[0])
    with rio.open(
            output_filepath,
            mode="w",
            driver="GTiff",
            width=merged_raster.shape[1],
            height=merged_raster.shape[0],
            count=1,
            crs=rio.open(raster_filepaths[0]).crs,
            transform=transform,
            dtype=dtype,
            nodata=nodata_value
    ) as raster:
        raster.write(merged_raster.astype(dtype), 1)


def coregister_and_merge_ddems(redo: bool = False):
    """Run each step from start to finish."""
    # Generate both the glacier mask and stable ground mask
    outlines.generate_stable_ground_mask(overwrite=redo)
    coregister_all_dems(overwrite=redo)
    generate_all_ddems(overwrite=redo)
    make_yearly_ddems(overwrite=redo)

    good_ddem_filepaths = evaluation.evaluate_ddems(improve=False)
    good_yearly_ddems = [
        os.path.join(CACHE_FILES["ddems_yearly_coreg_dir"], f"{extract_station_name(filepath)}_yearly_ddem.tif")
        for filepath in good_ddem_filepaths
    ]
    good_ddem_fraction = len(good_ddem_filepaths) / len(find_dems(CACHE_FILES["ddems_coreg_dir"]))
    print(f"{len(good_ddem_filepaths)} classified as good ({good_ddem_fraction * 100:.2f}%)")
    print("Merging dDEMs")
    merge_rasters(good_ddem_filepaths)
    merge_rasters(
        good_yearly_ddems,
        output_filepath=CACHE_FILES["merged_yearly_ddem"]
    )
