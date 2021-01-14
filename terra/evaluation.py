from __future__ import annotations

import concurrent.futures
import datetime
import json
import os
import pickle
import subprocess
import tempfile
import warnings
from typing import Any, Optional

import matplotlib.pyplot as plt
import matplotlib.widgets
import numpy as np
import pandas as pd
import pytransform3d
import pytransform3d.rotations
import pytransform3d.transformations
import rasterio as rio
import rasterio.warp  # pylint: disable=unused-import
import sklearn.linear_model
import sklearn.utils
from tqdm import tqdm

from terra import files
from terra.constants import CONSTANTS
from terra.processing import inputs, processing_tools

TEMP_DIRECTORY = os.path.join(files.TEMP_DIRECTORY, "evaluation")

CACHE_FILES = {
    "metashape_dems_dir": os.path.join(inputs.TEMP_DIRECTORY, "output/dems"),
    "ddems_dir": os.path.join(TEMP_DIRECTORY, "ddems/"),
    "ddems_coreg_dir": os.path.join(TEMP_DIRECTORY, "coreg_ddems/"),
    "dem_coreg_dir": os.path.join(TEMP_DIRECTORY, "coreg/"),
    "dem_coreg_meta_dir": os.path.join(TEMP_DIRECTORY, "coreg_meta/"),
    "glacier_mask": os.path.join(TEMP_DIRECTORY, "glacier_mask.tif"),
    "ddem_quality": os.path.join(files.MANUAL_INPUT_DIR, "ddem_quality.csv"),
    "ddem_stats": os.path.join(TEMP_DIRECTORY, "ddem_stats.pkl"),
    "merged_ddem": os.path.join(TEMP_DIRECTORY, "merged_ddem.tif"),
}


def find_dems(folder: str) -> list[str]:
    """Find all .tif files in a folder."""
    dem_names = [os.path.join(folder, filename) for filename in os.listdir(folder) if filename.endswith(".tif")]

    return dem_names


def extract_station_name(filepath: str) -> str:
    """Find the station name (e.g. station_3500) inside a filepath."""
    return filepath[filepath.index("station_"):filepath.index("station_") + 12]


def reproject_dem(dem: rio.DatasetReader, bounds: dict[str, float], resolution: float,
                  crs: Optional[rio.crs.CRS] = None, resampling=rio.warp.Resampling.cubic_spline) -> np.ndarray:
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
    destination = np.empty(dst_shape, dem.dtypes[0])
    # Set all values to nan right now
    destination[:, :] = np.nan

    # Reproject the DEM and put the output in the destination array
    rio.warp.reproject(
        source=dem.read(1),
        destination=destination,
        src_transform=dem.transform,
        dst_transform=dst_transform,
        resampling=resampling,
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
    # Skip if it already exists
    if not overwrite and os.path.isfile(CACHE_FILES["glacier_mask"]):
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

    progress_bar = tqdm(total=len(dem_filepaths))

    for filepath in dem_filepaths:
        progress_bar.desc = filepath
        # Determine the name of the output DEM
        station_name = extract_station_name(filepath)
        aligned_dem_path = os.path.join(CACHE_FILES["dem_coreg_dir"], f"{station_name}_coregistered.tif")
        if not overwrite and os.path.isfile(aligned_dem_path):
            progress_bar.update()
            continue
        coregister_dem(filepath, CACHE_FILES["glacier_mask"])
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


def read_icp_stats(station_name: str) -> dict[str, float]:
    """
    Read the ICP coregistration statistics from a specified station name.

    :param station_name: The name of the station, e.g. station_3500.
    :returns: A dictionary of coregistration statistics.
    """

    with open(os.path.join(CACHE_FILES["dem_coreg_meta_dir"], f"{station_name}_coregistration.json")) as infile:
        coreg_stats = json.load(infile)

    # Calculate the transformation angle from the transformation matrix
    matrix = np.array(coreg_stats["transform"].replace("\n", " ").split(" ")).astype(float).reshape((4, 4))
    quaternion = pytransform3d.transformations.pq_from_transform(matrix)[3:]
    angle = np.rad2deg(pytransform3d.rotations.axis_angle_from_quaternion(quaternion)[3])

    stats = {"fitness": coreg_stats["fitness"], "overlap": coreg_stats["overlap"], "angle": angle}

    return stats


def read_and_crop_glacier_mask(raster: rio.DatasetReader, resampling=rio.warp.Resampling.cubic_spline) -> np.ndarray:
    """
    Read the glacier mask and crop it to the input raster dataset.

    :param raster: A Rasterio dataset to crop the glacier mask to.
    :returns: A boolean array with the same dimensions and bounds as the input dataset.
    """
    bounds = dict(zip(["west", "south", "east", "north"], list(raster.bounds)))
    glacier_mask_dataset = rio.open(CACHE_FILES["glacier_mask"])
    cropped_glacier_mask = reproject_dem(glacier_mask_dataset, bounds,
                                         CONSTANTS.dem_resolution, resampling=resampling) == 1
    glacier_mask_dataset.close()

    return cropped_glacier_mask


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

        raster_values = raster.read(1)
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
            dtype=merged_raster.dtype,
            nodata=-9999
    ) as raster:
        raster.write(merged_raster, 1)


def examine_ddem(filepath: str) -> bool:
    """
    Examine a dDEM and rate its validity (good, bad or unclear).

    :param filepath: The path to the dDEM.
    :returns: A flag of whether or not the user has pressed the stop button.
    """
    def log_quality(quality: str):
        """Log the given quality value."""
        print(f"Registered {filepath} as {quality}")

        station_name = extract_station_name(filepath)
        stats = read_icp_stats(station_name)
        date = datetime.datetime.now()

        with open(CACHE_FILES["ddem_quality"], "a+") as outfile:
            line = f"{filepath},{quality},{stats['fitness']},{stats['overlap']},{stats['angle']},{date}\n"
            outfile.write(line)

    # Read the dDEM
    ddem = rio.open(filepath)
    ddem_values = ddem.read(1)

    # Load and crop/resample the glacier mask and reference DEM
    glacier_mask = read_and_crop_glacier_mask(ddem)

    # Stop button, signalling that the examination is done.
    should_stop = False

    def stop(*_):
        nonlocal should_stop
        should_stop = True

    plt.subplot(111)
    plt.imshow(glacier_mask, cmap="Greys_r")
    plt.imshow(ddem_values, cmap="coolwarm_r", vmin=-100, vmax=100)

    good_button = matplotlib.widgets.Button(plt.axes([0.9, 0.0, 0.1, 0.075]), "Good", )
    good_button.on_clicked(lambda _: log_quality("good"))

    bad_button = matplotlib.widgets.Button(plt.axes([0.8, 0.0, 0.1, 0.075]), "Bad")
    bad_button.on_clicked(lambda _: log_quality("bad"))

    unclear_button = matplotlib.widgets.Button(plt.axes([0.7, 0.0, 0.1, 0.075]), "Unclear")
    unclear_button.on_clicked(lambda _: log_quality("unclear"))

    stop_button = matplotlib.widgets.Button(plt.axes([0.5, 0.0, 0.1, 0.075]), "Stop")
    stop_button.on_clicked(stop)
    plt.show()

    return should_stop


def model_df(data: pd.DataFrame, sample_cols: list[str],
             target_col: str) -> tuple[sklearn.linear_model.LogisticRegression, float]:
    """
    Train a logistic model on a dataframe and return the model + model score.

    The target_col must be boolean.

    :param data: The dataframe to analyse.
    :param sample_cols: The columns of X values (predictors).
    :param target_col: The column of y values (targets).
    :returns: The linear model and its model score.
    """
    # Make sure that the target_col is a boolean field
    data[target_col].astype(bool)

    # Ugly implementation of outlier detection. Kind of works like RANSAC:
    # Try to estimate the model at most ten times and exclude 10% of the data each time.
    # Possible outliers therefore have a chance of being excluded, possibly leading to a better model.
    # Finally, the model with the most valid data is returned.
    models = {}
    for _ in range(10):
        shuffled_data = sklearn.utils.shuffle(data)

        # Exclude the last 10% of the shuffled data.
        exclusion_border = int(shuffled_data.shape[0] * 0.9)
        shuffled_data = shuffled_data.iloc[:exclusion_border]

        # Warnings may arise here which should just be repressed.
        with warnings.catch_warnings(record=True) as warning_filter:
            warnings.simplefilter("always")
            model = sklearn.linear_model.LogisticRegression(
                random_state=0, solver='lbfgs', multi_class='ovr', max_iter=300)
            # Try to fit the model.
            try:
                model.fit(shuffled_data[sample_cols], shuffled_data[target_col])
            # ValueErrors may be raised if the data contain only True's or False's (then try again with other data)
            except ValueError:
                continue
            score = model.score(shuffled_data[sample_cols], shuffled_data[target_col])

            n_good = np.count_nonzero(model.predict(shuffled_data[sample_cols]))
            models[n_good] = model, score

    return models[max(models.keys())]


def get_ddem_stats(directory: str, redo=False) -> pd.DataFrame:
    """
    Calculate statistics from each dDEM in a directory.

    The pixel count, median, 95th percentile, and 5th percentile is calculated for on- and off-glacier surfaces.

    :param directory: The directory of the dDEMs to calculate statistics from.
    :param redo: Whether to redo the analysis if it has already been made (it takes a while to run)
    :returns: A dataframe with each of the dDEM's statistics in one row.
    """
    ddems = find_dems(directory)

    if os.path.isfile(CACHE_FILES["ddem_stats"]) and not redo:
        return pd.read_pickle(CACHE_FILES["ddem_stats"])

    stats = pd.DataFrame(data=ddems, columns=["filepath"])
    stats["station_name"] = stats["filepath"].apply(extract_station_name)

    for i, filepath in tqdm(enumerate(ddems), total=len(ddems), desc="Calculating dDEM statistics"):
        ddem = rio.open(filepath)
        glacier_mask = read_and_crop_glacier_mask(ddem)
        ddem_values = ddem.read(1)

        for on_off, values in zip(["on", "off"], [ddem_values[glacier_mask], ddem_values[~glacier_mask]]):
            col_names = [f"{on_off}_{stat}" for stat in ["count", "median", "upper", "lower"]]
            # Return a zero count and 200 as a "nodata" value if there is no on- / off-glacier pixel
            # The 200 is because these stats are used for regression, and 200 is a bad value
            # TODO: Handle the "bad values" in a better fashion?
            if np.all(np.isnan(values)):
                stats.loc[i, col_names] = [0] + [200] * (len(col_names) - 1)
                continue
            count = np.count_nonzero(~np.isnan(values))
            median = np.nanmedian(values)
            upper_percentile = np.nanpercentile(values, 95)
            lower_percentile = np.nanpercentile(values, 5)
            stats.loc[i, col_names] = count, median, upper_percentile, lower_percentile

    stats.to_pickle(CACHE_FILES["ddem_stats"])
    return stats


def evaluate_ddems(improve: bool = False) -> list[str]:
    """
    Train and or run a model to evaluate the quality of the dDEMs, and return a list of the ones deemed good.

    :param improve: Improve the model?
    :returns: A list of filepaths to the dDEMs that are deemed good.
    """
    if improve:
        coreg_ddems = np.array(find_dems(CACHE_FILES["ddems_coreg_dir"]))

        # Shuffle the data so that dDEMs from different areas come up in succession.
        np.random.shuffle(coreg_ddems)

        # For the improvement part: Generate a list of dDEMs that have already been looked at.
        existing_filepaths: list[str] = []
        if os.path.exists(CACHE_FILES["ddem_quality"]):
            existing_filepaths += list(pd.read_csv(CACHE_FILES["ddem_quality"])["filepath"])

        stopped = False
        for filepath in coreg_ddems:
            if stopped:
                break
            if filepath in existing_filepaths:
                continue
            stopped = examine_ddem(filepath)
            existing_filepaths.append(filepath)

    # Read the log of the manually placed dDEM quality flags.
    log = pd.read_csv(CACHE_FILES["ddem_quality"]).drop_duplicates("filepath", keep="last")
    # Remove all the ones where the quality was deemed unclear.
    log = log[log["quality"] != "unclear"]
    # Extract the station name from the filepath.
    log["station_name"] = log["filepath"].apply(extract_station_name)
    # Make a boolean field of the quality
    log["good"] = log["quality"] == "good"

    # TODO: Why is it shuffled?
    log = sklearn.utils.shuffle(log)

    # Read the ICP coregistration statistics from each respective coregistered DEM
    # for i, row in log.iterrows():
    #    stats = read_icp_stats(row["station_name"])
    #    log.loc[i, list(stats.keys())] = list(stats.values())

    # Train a classifier to classify the dDEM quality based on the ICP coregistration statistics.
    # Try ten times to get a model with a score >= 0.9
    for _ in range(10):
        model, score = model_df(log, ["fitness", "overlap", "angle"], "good")
        if score > 0.9:
            break
    else:
        raise ValueError(f"Score too low: {score:.2f}. Try running again.")

    print(f"Modelled coregistered dDEM usability with an accuracy of: {score:.2f}")

    # Load all of the dDEMs (not just the ones with manuall quality flags)
    all_files = pd.DataFrame(data=find_dems(CACHE_FILES["ddems_coreg_dir"]), columns=["filepath"])
    all_files["station_name"] = all_files["filepath"].apply(extract_station_name)

    # Read the ICP coregistration statistics from each respective coregistered DEM
    for i, row in all_files.iterrows():
        try:
            stats = read_icp_stats(row["station_name"])
        # If coregistration failed, it will not have generated a stats file.
        except FileNotFoundError:
            all_files.drop(index=i, inplace=True)
            continue
        all_files.loc[i, list(stats.keys())] = list(stats.values())

    # Use the classifier to classify all dDEMs based on the coregistration quality.
    all_files["good"] = model.predict(all_files[["fitness", "overlap", "angle"]])

    # Find the filepaths corresponding to the dDEMs that were classified as good
    good_filepaths = list(all_files[all_files["good"]]["filepath"].values)

    return good_filepaths

    stopped = False
    if improve:
        for filepath in all_files[~all_files["good"]]["filepath"].values:
            if filepath in existing_filepaths:
                continue
            stopped = examine_ddem(filepath)
            existing_filepaths.append(filepath)

    non_coreg_log = pd.read_csv(CACHE_FILES["ddem_quality"], header=None, names=["filepath", "quality"])
    # .drop_duplicates("filepath", keep="last")
    # Keep only the filepaths containing /ddems/ (not /coreg_ddems/)
    non_coreg_log = non_coreg_log[non_coreg_log["filepath"].str.contains("/ddems/")]
    # Remove the "unclear" marks
    non_coreg_log = non_coreg_log[non_coreg_log["quality"] != "unclear"]
    # Convert the good/bad quality column into a boolean column
    non_coreg_log["good"] = non_coreg_log["quality"] == "good"
    non_coreg_log["station_name"] = non_coreg_log["filepath"].apply(extract_station_name)

    ddem_stats = get_ddem_stats(CACHE_FILES["ddems_dir"])
    training_cols = ddem_stats.columns[2:]
    for i, row in non_coreg_log.iterrows():
        stat_row = ddem_stats.loc[ddem_stats["filepath"] == row["filepath"]].iloc[0]

        non_coreg_log.loc[i, training_cols] = stat_row.loc[training_cols]

    model, score = model_df(non_coreg_log, training_cols, "good")
    print(f"Modelled normal dDEM usability with an accuracy of {score:.2f}")

    ddem_stats["good"] = model.predict(ddem_stats[training_cols])

    good_log = log[log["good"]]
    output_filepaths: list[str] = list(log[log["good"]]["filepath"].values)

    return output_filepaths

    usable_old_ddems = 0
    for i, row in ddem_stats.iterrows():
        if not row["good"] or row["station_name"] in good_log["station_name"].values:
            continue
        usable_old_ddems += 1
        output_filepaths.append(row["filepath"])

    return output_filepaths


def compute_merged_ddem_stats():
    """
    Compute and plot statistics from the merged dDEM.
    """
    cache_filepath = os.path.join(TEMP_DIRECTORY, "heights_and_glacier_ddem.pkl")

    if os.path.isfile(cache_filepath):
        with open(cache_filepath, "rb") as infile:
            heights, glacier_vals = pickle.load(infile)
    else:
        ddem_dataset = rio.open(CACHE_FILES["merged_ddem"])
        glacier_mask = read_and_crop_glacier_mask(ddem_dataset, resampling=rio.warp.Resampling.nearest)
        print("Read glacier mask")

        ddem_values = ddem_dataset.read(1)
        print("Read dDEM")

        eastings, northings = np.meshgrid(
            np.linspace(ddem_dataset.bounds.left, ddem_dataset.bounds.right, num=ddem_dataset.width),
            np.linspace(ddem_dataset.bounds.top, ddem_dataset.bounds.bottom, num=ddem_dataset.height)
        )
        glacier_vals = ddem_values[glacier_mask]
        inlier_mask = ~np.isnan(glacier_vals)
        glacier_vals = glacier_vals[inlier_mask]

        filtered_eastings = eastings[glacier_mask][inlier_mask]
        filtered_northings = northings[glacier_mask][inlier_mask]

        assert glacier_vals.shape == filtered_eastings.shape

        dem = rio.open(files.INPUT_FILES["base_DEM"])
        print("Sampling DEM values...")
        heights = np.fromiter(dem.sample(zip(filtered_eastings, filtered_northings)), dtype=np.float64)
        heights[heights < 0] = np.nan

        assert heights.shape == glacier_vals.shape

        with open(cache_filepath, "wb") as outfile:
            pickle.dump((heights, glacier_vals), outfile)

    old_heights = heights - glacier_vals
    inlier_mask = (glacier_vals < 100) & (old_heights < CONSTANTS.max_height)

    old_heights = old_heights[inlier_mask]
    glacier_vals = glacier_vals[inlier_mask]

    count, y_bins, x_bins = np.histogram2d(old_heights, glacier_vals, bins=500)

    plt.imshow(count, extent=(x_bins.min(), x_bins.max(), y_bins.max(), y_bins.min()),
               cmap="terrain_r", interpolation="bilinear", aspect="auto")
    plt.ylim(y_bins.min(), y_bins.max())

    plt.ylabel("Elevation (m a.s.l.)")
    plt.xlabel("Elevation change (m)")

    cbar = plt.colorbar()
    cbar.set_label("Frequency")
    plt.show()


def main(redo: bool = False):
    """Run each step from start to finish."""
    generate_glacier_mask(overwrite=redo)
    coregister_all_dems(overwrite=redo)
    generate_all_ddems(overwrite=redo)

    good_ddem_filepaths = evaluate_ddems(improve=False)
    if redo or not os.path.isfile(CACHE_FILES["merged_ddem"]):
        merge_rasters(good_ddem_filepaths)


if __name__ == "__main__":
    main()

    compute_merged_ddem_stats()
