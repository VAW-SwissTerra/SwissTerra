"""Orthomosaic generation tools."""
import concurrent.futures
import json
import os
import subprocess
import tempfile

import cv2
import jinja2
import numpy as np
import rasterio as rio
import rasterio.warp  # pylint: disable=unused-import
from tqdm import tqdm

from terra import dem_tools, evaluation, files
from terra.constants import CONSTANTS
from terra.processing import inputs, processing_tools

TEMP_DIRECTORY = os.path.join(files.TEMP_DIRECTORY, "orthomosaics/")

CACHE_FILES = {
    "ortho_coreg_dir": os.path.join(TEMP_DIRECTORY, "ortho_coreg"),
    "merged_ortho": os.path.join(TEMP_DIRECTORY, "merged_ortho.tif"),
    "metashape_ortho_dir": os.path.join(inputs.TEMP_DIRECTORY, "output/orthos/"),
    "metashape_dem_dir": os.path.join(inputs.TEMP_DIRECTORY, "output/dems/"),
}


def apply_coreg_to_ortho(station_name: str, overwrite: bool = False):
    """
    Apply a DEM coregistration matrix to an orthomosaic.

    The orthomosaic is merged with its DEM as a point cloud and is then transformed and regridded.

    Needs an equally named DEM to work.

    :param station_name: The station name of the orthomosaic (used to correlate DEMs with orthos).
    :param overwrite: Overwrite the orthomosaic if it already exists?
    """
    # Find the needed filepaths using the station name.
    ortho_path = os.path.join(CACHE_FILES["metashape_ortho_dir"], f"{station_name}_orthomosaic.tif")
    dem_path = os.path.join(CACHE_FILES["metashape_dem_dir"], f"{station_name}_dense_DEM.tif")
    coreg_meta_path = os.path.join(dem_tools.CACHE_FILES["dem_coreg_meta_dir"], f"{station_name}_coregistration.json")
    out_filepath = os.path.join(CACHE_FILES["ortho_coreg_dir"], f"{station_name}_ortho_coreg.tif")

    # Check if the file already exists and shouldn't be redone.
    if not overwrite and os.path.isfile(out_filepath):
        return

    # Check that all needed input paths exist.
    for path in [ortho_path, dem_path, coreg_meta_path]:
        if not os.path.isfile(path):
            return

    # Make the output directory if it doesn't yet exist.
    if not os.path.isdir(os.path.dirname(out_filepath)):
        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

    # Make a temporary directory that will disappear after the analysis is done.
    temp_dir = tempfile.TemporaryDirectory()
    temp_ortho_dem_path = os.path.join(temp_dir.name, "ortho.xyz")

    ortho = rio.open(ortho_path)
    dem = rio.open(dem_path)

    # Generate a grid of easting and northing coordinates, representing the middle coordinate of each cell.
    eastings, northings = np.meshgrid(
        np.arange(dem.bounds.left, dem.bounds.right,
                  step=CONSTANTS.dem_resolution) + CONSTANTS.dem_resolution / 2,
        np.arange(dem.bounds.bottom, dem.bounds.top,
                  step=CONSTANTS.dem_resolution)[::-1] + CONSTANTS.dem_resolution / 2
    )

    # Resample the orthomosaic to the DEM's shape
    # First, generate empty "destination" arrays
    resampled_ortho = np.zeros((dem.height, dem.width), dtype=ortho.dtypes[0])
    resampled_mask = np.zeros(resampled_ortho.shape)

    reprojection_params = dict(
        src_transform=ortho.transform,
        dst_transform=dem.transform,
        src_crs=ortho.crs,
        dst_crs=dem.crs,
        dst_resolution=dem.res
    )

    # Warp the values into the destination array
    rio.warp.reproject(ortho.read(1), destination=resampled_ortho, **
                       reprojection_params, resampling=rio.warp.Resampling.bilinear)
    # Warp the validity mask into the destination array using nearest neighbour resampling.
    rio.warp.reproject(ortho.read(2), destination=resampled_mask, **reprojection_params)
    # Make sure no values have 255 (it will be the nan value)
    resampled_ortho[resampled_ortho == 255] -= 1

    dem_vals = dem.read(1)
    # Find the areas where there is an orthomosaic value and a DEM value.
    mask = (resampled_mask == 255) & (dem_vals != -9999)

    # Stop if this mask is empty.
    if np.count_nonzero(mask) == 0:
        return

    # Merge the eastings, northings, elevatons and orthomosaic lightness (inside the mask) to one array.
    merged_vals = np.dstack((eastings[mask], northings[mask], dem_vals[mask], resampled_ortho[mask]))[0, :, :]
    # Save the array as a temporary xyz comma delimited point cloud.
    np.savetxt(temp_ortho_dem_path, merged_vals, delimiter=",", fmt=["%f"] * 3 + ["%d"])

    # Get the bounds of the current orthomosaic to use as the output bounds.
    # TODO: Maybe read the bounds from the transformed DEM instead.
    bounds = np.array([merged_vals[:, 0].min(), merged_vals[:, 0].max(),
                       merged_vals[:, 1].min(), merged_vals[:, 1].max()])
    # Correct the bounds to be divisible by the DEM resolution
    bounds -= bounds % CONSTANTS.dem_resolution
    bounds[[1, 3]] += CONSTANTS.dem_resolution

    with open(coreg_meta_path) as infile:
        coreg_meta = json.load(infile)

    # Run a PDAL pipeline to transform the "point cloud" and grid it into a new orthomosaic.
    # The lightness is parsed as "Red" since its uint8. "Intensity" is maybe a more apt name, but is uint16.
    processing_tools.run_pdal_pipeline(jinja2.Template("""
    [
        {
            "type": "readers.text",
            "filename": "{{ filename }}",
            "header": "X,Y,Z,Red"
        },
        {
            "type": "filters.transformation",
            "matrix": "{{ matrix }}"
        },
        {
            "type": "writers.gdal",
            "dimension": "Red",
            "resolution": {{ resolution }},
            "bounds": "([{{ xmin }},{{ xmax }}],[{{ ymin }}, {{ ymax }}])",
            "data_type": "uint8",
            "gdalopts": "COMPRESS=LZW",
            "output_type": "mean",
            "filename": "{{ out_filename }}"
        }

    ]""").render(
        filename=temp_ortho_dem_path,
        matrix=coreg_meta["composed"].replace("\n", " "),
        resolution=int(dem.res[0]),
        xmin=bounds[0], xmax=bounds[1], ymin=bounds[2], ymax=bounds[3],
        out_filename=out_filepath))

    # Set the correct CRS and nodata value.
    gdal_commands = ["gdal_edit.py", "-a_srs", CONSTANTS.crs_epsg.replace("::", ":"), "-a_nodata", "255", out_filepath]
    subprocess.run(gdal_commands, check=True)


def merge_orthomosaics(ortho_filepaths: list[str], output_filepath: str):
    """
    Merge a series of orthomosaics using average blending where they overlap.

    Rasters that overlap are first corrected using the mean offset between each other to improve the seams a bit.
    The merged orthomosaic is then sharpened slightly.

    IMPORTANT: The GeoTiffs are assumed to be in the same resolution and grid.

    :param ortho_filepaths: Filepaths to the orthomosaics to merge.
    :param output_filepath: The output filepath of the merged orthomosaic.
    """
    # Find the maximum bounding box of the rasters.
    bounds: dict[str, float] = {}
    for filepath in ortho_filepaths:
        raster = rio.open(filepath)

        # bounds.get(*) may return None, so set it to an abnormal value to prefer the raster.bounds instead
        bounds["west"] = min(bounds.get("west") or 1e22, raster.bounds.left)
        bounds["east"] = max(bounds.get("east") or -1e22, raster.bounds.right)
        bounds["south"] = min(bounds.get("south") or 1e22, raster.bounds.bottom)
        bounds["north"] = max(bounds.get("north") or -1e22, raster.bounds.top)

        raster.close()

    # Find the corresponding output shape of the merged raster
    output_shape = (
        int((bounds["north"] - bounds["south"]) / CONSTANTS.dem_resolution),
        int((bounds["east"] - bounds["west"]) / CONSTANTS.dem_resolution),
    )

    # Generate a new merged raster with only NaN values
    merged_raster = np.empty(shape=output_shape, dtype=np.float32)
    value_mask = np.zeros_like(merged_raster).astype(bool)

    for filepath in tqdm(ortho_filepaths, desc="Merging rasters"):
        raster = rio.open(filepath)

        # Find the top left pixel index
        top_index = int((bounds["north"] - raster.bounds.top) / CONSTANTS.dem_resolution)
        left_index = int((raster.bounds.left - bounds["west"]) / CONSTANTS.dem_resolution)

        # Generate a slice object for the new raster to index the merged raster.
        raster_slice = np.s_[top_index:top_index + raster.height, left_index:left_index + raster.width]

        raster_values = raster.read(1).astype(np.float32)
        raster_values[raster_values == 255] = np.nan
        valid_mask = ~np.isnan(raster_values)

        # Skip the raster if all values are nans
        if np.all(~valid_mask):
            raster.close()
            continue

        # The valid value mask is the union of the existing mask and the new valid mask.
        value_mask[raster_slice] = np.logical_or(value_mask[raster_slice], valid_mask)
        # Find the mean value of the raster values that already exist in the slice.
        mean_existing_val = np.nanmean(merged_raster[raster_slice][valid_mask]) \
            if not np.all(np.isnan(merged_raster[raster_slice][valid_mask])) else np.nan
        # Correct for the mean lightness offset
        raster_values -= (np.nanmean(raster_values) - mean_existing_val) if not np.isnan(mean_existing_val) else 0.0

        # Add the new values to the merged raster.
        merged_raster[raster_slice] = np.nanmean([merged_raster[raster_slice], raster_values], axis=0)

        raster.close()

    # Set the cells without a value to nan
    merged_raster[~value_mask] = np.nan
    # Normalise the raster to 0-244 (exluding the 255 nan value)
    merged_raster -= np.nanmin(merged_raster)
    merged_raster = (merged_raster / (np.nanmax(merged_raster) / 254))
    # Set the nans to 255
    merged_raster[np.isnan(merged_raster)] = 255

    # Sharpen the raster
    # TODO: Try to fix the value-nan edges now being dark.
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_raster = cv2.filter2D(merged_raster.astype("uint8"), -1, kernel)
    merged_raster = np.average([sharpened_raster, merged_raster], axis=0, weights=[1, 4])  # Blend in a 1:5 ratio

    # Save the merged raster
    transform = rio.transform.from_bounds(**bounds, width=merged_raster.shape[1], height=merged_raster.shape[0])
    with rio.open(
            output_filepath,
            mode="w",
            driver="GTiff",
            width=merged_raster.shape[1],
            height=merged_raster.shape[0],
            count=1,
            crs=rio.open(ortho_filepaths[0]).crs,
            transform=transform,
            dtype="uint8",
            nodata=255,
            compress="lzw"
    ) as raster:
        raster.write(merged_raster.astype("uint8"), 1)


def apply_coregistrations(overwrite: bool = False):
    """
    Apply the coregistration matrices to the orthomosaics and mergem into one file.

    :param overwrite: Overwrite already existing orthomosaics?
    """
    stations = np.unique([dem_tools.extract_station_name(filename)
                          for filename in os.listdir(CACHE_FILES["metashape_ortho_dir"])])

    progress_bar = tqdm(total=len(stations), desc="Transforming orthomosaics")

    def apply_coreg(station_name: str):
        """Apply the coregistration in one thread."""
        apply_coreg_to_ortho(station_name, overwrite=overwrite)
        progress_bar.update()

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(apply_coreg, stations))

    progress_bar.close()

    ortho_filepaths = [os.path.join(CACHE_FILES["ortho_coreg_dir"], filename)
                       for filename in os.listdir(CACHE_FILES["ortho_coreg_dir"])]

    merge_orthomosaics(ortho_filepaths, output_filepath=CACHE_FILES["merged_ortho"])
