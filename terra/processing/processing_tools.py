"""Functions to process data."""
import json
import os
import subprocess
import tempfile
import warnings
from collections import namedtuple
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import rasterio as rio
import scipy.ndimage

from terra import files
from terra.constants import CONSTANTS
from terra.processing import inputs, main
from terra.utilities import no_stdout

CACHE_FILES = {
    "asift_temp_dir": os.path.join(inputs.TEMP_DIRECTORY, "asift_temp")
}


def generate_dem(point_cloud_path: str, output_dem_path: str,
                 resolution: float = 1.0, interpolate_pixels: int = 10) -> str:
    """
    Generate a DEM from a point cloud using PDAL.

    param: point_cloud_path: The path to the input point cloud.
    param: output_dem_path: The path to the output DEM.
    param: resolution: The resolution in m of the output DEM.
    param: interpolate_pixels: Fill holes with the given size.
    """
    cloud_meta = json.loads(subprocess.run(["pdal", "info", point_cloud_path, "--stats"],
                                           stdout=subprocess.PIPE, encoding="utf-8", check=True).stdout)
    cloud_bounds = cloud_meta["stats"]["bbox"]["native"]["bbox"]

    min_x = cloud_bounds["minx"] - (cloud_bounds["minx"] % resolution)
    max_x = cloud_bounds["maxx"] - (cloud_bounds["maxx"] % resolution) + resolution
    min_y = cloud_bounds["miny"] - (cloud_bounds["miny"] % resolution)
    max_y = cloud_bounds["maxy"] - (cloud_bounds["maxy"] % resolution) + resolution

    params = {
        "INPUT_CLOUD": point_cloud_path,
        "RESOLUTION": resolution,
        "BOUNDS": f"([{min_x}, {max_x}], [{min_y}, {max_y}])",
        "OUTPUT_DEM": output_dem_path,

    }
    # Construct the pipeline for PDAL
    pdal_pipeline = '''
    [
        "INPUT_CLOUD",
        {
            "resolution": RESOLUTION,
            "bounds": "BOUNDS",
            "filename": "OUTPUT_DEM",
            "output_type": "mean",
            "data_type": "float32",
            "gdalopts": ["COMPRESS=DEFLATE", "PREDICTOR=3"]
        }
    ]
    '''
    # Fill the pipeline with variables
    for parameter in params:
        pdal_pipeline = pdal_pipeline.replace(parameter, str(params[parameter]))

    run_pdal_pipeline(pdal_pipeline)

    # Fill small holes using linear spatial interpolation.
    if interpolate_pixels != 0:
        gdal_commands = ["gdal_fillnodata.py", "-md", str(interpolate_pixels), output_dem_path, "-q"]
        subprocess.run(gdal_commands, check=True)

    # Set the correct CRS metadata.
    gdal_commands = ["gdal_edit.py", "-a_srs", str(CONSTANTS["crs_epsg"]).replace("::", ":"), output_dem_path]
    subprocess.run(gdal_commands, check=True)

    return output_dem_path


def run_pdal_pipeline(pipeline: str, output_metadata_file: Optional[str] = None,
        parameters: Optional[Dict[str, str]] = None, show_warnings: bool = False) -> Dict[str, Any]:
    """
    Run a PDAL pipeline.

    param: pipeline: The pipeline to run.
    param: output_metadata_file: Optional. The filepath for the pipeline metadata.
    param: parameters: Optional. Parameters to fill the pipeline with, e.g. {"FILEPATH": "/path/to/file"}.
    :param show_warnings: Show the full stdout of the PDAL process.

    return: output_meta: The metadata produced by the output.
    """
    # Create a temporary directory to save the output metadata in
    temp_dir = tempfile.TemporaryDirectory()
    # Fill the pipeline with
    if parameters is not None:
        for key in parameters:
            # Warn if the key cannot be found in the pipeline
            if key not in pipeline:
                warnings.warn(
                    f"{key}:{parameters[key]} given to the PDAL pipeline but the key was not found", RuntimeWarning)
            # Replace every occurrence of the key inside the pipeline with its corresponding value
            pipeline = pipeline.replace(key, str(parameters[key]))

    try:
        json.loads(pipeline)  # Throws an error if the pipeline is poorly formatted
    except json.decoder.JSONDecodeError as exception:
        raise ValueError("Pipeline was poorly formatted: \n" + pipeline + "\n" + str(exception))

    # Run PDAL with the pipeline as the stdin
    commands = ["pdal", "pipeline", "--stdin", "--metadata", os.path.join(temp_dir.name, "meta.json")]
    stdout = subprocess.run(commands, input=pipeline, check=True, stdout=subprocess.PIPE, encoding="utf-8").stdout

    if show_warnings and len(stdout.strip()) != 0:
        print(stdout)

    # Load the temporary metadata file
    with open(os.path.join(temp_dir.name, "meta.json")) as infile:
        output_meta = json.load(infile)

    # Save it with a different name if one was provided
    if output_metadata_file is not None:
        with open(output_metadata_file, "w") as outfile:
            json.dump(output_meta, outfile)

    return output_meta


def coalign_dems(reference_path: str, aligned_path: str, pixel_buffer=3, nan_value=-9999) -> Optional[Dict[str, str]]:
    """
    Align two DEMs and return the ICP result.

    param: reference_path: The filepath to the DEM acting reference.
    param: aligned_path: The filepath to the DEM acting aligned.
    param: pixel_buffer: The amount of allowed overlap between the clouds (should be the expected offset).
    param: nan_value: The NaN value of both DEMs.

    return: result: The resultant PDAL output. Returns None if there was no alignment.

    """
    # Create a bounds object to more explicitly handle bounds.
    Bounds = namedtuple("Bounds", ["x_min", "x_max", "y_min", "y_max"])

    # temp_dir = tempfile.TemporaryDirectory()
    temp_dir = os.path.join(
        os.path.dirname(reference_path),
        "coalignment",
        "_to_".join([
            os.path.splitext(os.path.basename(reference_path))[0],
            os.path.splitext(os.path.basename(aligned_path))[0]
        ])
    )
    tempfiles = {os.path.splitext(filename)[0]: os.path.join(temp_dir, filename) for filename in [
        "reference_cropped.tif", "aligned_cropped.tif", "aligned_post_icp.tif", "result_meta.json"
    ]}

    def get_bounds(filepath) -> Bounds:
        """
        Use GDAL to load the bounds of a raster.

        param: filepath: The path to the raster.

        return: bounds: The bounding coordinates of the raster.
        """
        meta = json.loads(subprocess.run(["gdalinfo", filepath, "-stats", "-json"],
                                         stdout=subprocess.PIPE, encoding="utf-8", check=True).stdout)
        corners = meta["cornerCoordinates"]

        x_min, y_max = corners["upperLeft"]
        x_max, y_min = corners["lowerRight"]

        return Bounds(x_min, x_max, y_min, y_max)

    # Load the bounds of the two rasters
    reference_bounds = get_bounds(reference_path)
    aligned_bounds = get_bounds(aligned_path)

    # Check if the bounds are not intersecting with each other
    not_overlapping = any([
        reference_bounds.x_max < aligned_bounds.x_min,
        aligned_bounds.x_max < reference_bounds.x_min,
        reference_bounds.y_max < aligned_bounds.y_min,
        aligned_bounds.y_max < reference_bounds.y_min
    ])

    # Stop here if they do no intersect
    if not_overlapping:
        return None

    # Read the DEMs
    dtype = np.float32
    reference_original = cv2.imread(reference_path, cv2.IMREAD_ANYDEPTH).astype(dtype)
    aligned_original = cv2.imread(aligned_path, cv2.IMREAD_ANYDEPTH).astype(dtype)

    # Check that the DEMs were read correctly
    if reference_original is None or aligned_original is None:
        return None

    # Set the nan_value to actual nans
    reference_original[reference_original == nan_value] = np.nan
    aligned_original[aligned_original == nan_value] = np.nan

    # Get the resolution by dividing the real world width with the pixel width
    resolution: float = (reference_bounds.x_max - reference_bounds.x_min) / reference_original.shape[1]

    # Create new bounding coordinates that encompass both datasets
    new_bounds = Bounds(
        min(reference_bounds.x_min, aligned_bounds.x_min),
        max(reference_bounds.x_max, aligned_bounds.x_max),
        min(reference_bounds.y_min, aligned_bounds.y_min),
        max(reference_bounds.y_max, aligned_bounds.y_max))

    # Determine what shape the new_bounds corresponds to
    new_shape = (int((new_bounds.y_max - new_bounds.y_min) // resolution),
                 int((new_bounds.x_max - new_bounds.x_min) // resolution))

    def resize(heights: np.ndarray, old_bounds: Bounds) -> np.ndarray:
        """
        Resize a DEM to fit the new_bounds.

        param: heights: The height values of the DEM.
        param: old_bounds: The initial bounds of the DEM.

        return: resized: The resized DEM.
        """
        # Create an empty array with the target shape and fill it with nans
        resized = np.empty(new_shape, dtype=np.float32)
        resized[:] = np.nan
        # Determine the initial dataset's integer location in the new resized array
        upper_corner = (int((new_bounds.y_max - old_bounds.y_max) // resolution),
                        int((old_bounds.x_min - new_bounds.x_min) // resolution))

        if 0 in heights.shape or 0 in resized.shape:
            raise ValueError("Ruh roh!")

        # Assign the initial dataset's values to the new array
        resized[upper_corner[0]: upper_corner[0] + heights.shape[0],
                upper_corner[1]: upper_corner[1] + heights.shape[1]] = heights

        return resized

    # Resize the DEMs to the same shape
    reference = resize(reference_original, reference_bounds)
    aligned = resize(aligned_original, aligned_bounds)

    # Check where the datasets overlap (where both DEMs don't have nans)
    overlapping = np.logical_and(np.logical_not(np.isnan(reference)), np.logical_not(np.isnan(aligned)))
    # Buffer the mask to increase the likelyhood of including the correct values
    overlapping_buffered = scipy.ndimage.maximum_filter(overlapping, size=pixel_buffer, mode="constant")

    # Filter the DEMs to only where they overlap
    reference[~overlapping_buffered] = np.nan
    aligned[~overlapping_buffered] = np.nan

    if any([
            # Check if all values are nan
            np.all(np.isnan(reference)), np.all(np.isnan(aligned)),
            # Check if it's all extreme values (basically +- inf).
            np.all(np.abs(reference) > 1e20), np.all(np.abs(aligned) > 1e20)
    ]):
        return None

    def write_raster(filepath: str, heights: np.ndarray):
        """
        Write a DEM as a GeoTiff.

        param: filepath: The output filepath of the DEM.
        param: heights: The height values of the DEM.
        """
        # Make a transform that Rasterio understands
        transform = rio.transform.from_bounds(new_bounds.x_min, new_bounds.y_min,
                                              new_bounds.x_max, new_bounds.y_max, new_shape[1], new_shape[0])

        # Write the DEM as a single band
        with rio.open(filepath, "w", driver="GTiff", height=heights.shape[0], width=heights.shape[1],
                      count=1, transform=transform, dtype=heights.dtype) as outfile:
            outfile.write(heights, 1)

    # Write the cropped DEMs
    os.makedirs(temp_dir, exist_ok=True)
    write_raster(tempfiles["reference_cropped"], reference)
    write_raster(tempfiles["aligned_cropped"], aligned)

    # Check that rasters were written
    if len(os.listdir(temp_dir)) == 0:
        return None

    def validate_raster(filepath: str) -> bool:
        with rio.open(filepath) as dataset:
            heights = dataset.read(1)

            n_valid = np.count_nonzero(~np.isnan(heights.flatten()))
            if n_valid > 0:
                return True

            return False

    for filename in ["reference_cropped", "aligned_cropped"]:
        if not validate_raster(tempfiles[filename]):
            return None

    # Read the datasets (A1, B1), filter out nans (A2, B2) and then run ICP
    pdal_pipeline = '''
    [
        {
            "type": "readers.gdal",
            "filename": "REFERENCE_FILEPATH",
            "header": "Z",
            "tag": "A1"
        },
        {
            "type": "readers.gdal",
            "filename": "ALIGNED_FILEPATH",
            "header": "Z",
            "tag": "B1"
        },
        {
            "inputs": "A1",
            "type": "filters.range",
            "limits": "Z[0:100000]",
            "tag": "A2"
        },
        {
            "inputs": "B1",
            "type": "filters.range",
            "limits": "Z[0:100000]",
            "tag": "B2"
        },
        {
            "inputs": ["A2", "B2"],
            "type": "filters.icp"
        },
        {
            "type": "writers.gdal",
            "filename": "OUTPUT_FILEPATH",
            "resolution": RESOLUTION
        }
    ]
    '''
    # Run the pdal pipeline
    try:
        result = run_pdal_pipeline(
            pipeline=pdal_pipeline,
            parameters={
                "REFERENCE_FILEPATH": tempfiles["reference_cropped"],
                "ALIGNED_FILEPATH": tempfiles["aligned_cropped"],
                "OUTPUT_FILEPATH": tempfiles["aligned_post_icp"],
                "RESOLUTION": str(resolution),
            },
            output_metadata_file=tempfiles["result_meta"]
        )["stages"]["filters.icp"]
    except subprocess.CalledProcessError as exception:
        print(f"Exception reached:\n{exception}")
        return None

    result["composed"] = result["composed"].replace("\n", " ")

    return result


def transform_points(points: pd.DataFrame, composed_transform: str, inverse: bool = False) -> pd.DataFrame:
    """
    Transform a set of points using a composed PDAL transform matrix.

    param: points: The points to transform.
    param: composed_transform: The composed PDAL transform matrix.
    param: inverse: Whether to apply the inverse transform.

    return: transformed_points: The transformed points.
    """
    # Check that all expected dimensions exist in the input points
    for dim in ["X", "Y", "Z"]:
        assert dim in points.columns

    # Make a temporary directory and create temporary filepaths
    temp_dir = tempfile.TemporaryDirectory()
    tempfiles = {
        "points": os.path.join(temp_dir.name, "points.csv"),
        "transformed_points": os.path.join(temp_dir.name, "transformed_points.csv")
    }
    # Save the points to a temporary file
    points[["X", "Y", "Z"]].to_csv(tempfiles["points"], index=False)

    # Construct the PDAL transformation pipeline
    pdal_pipeline = '''
    [
        {
            "type": "readers.text",
            "filename": "INPUT_FILEPATH"
        },
        {
            "type": "filters.transformation",
            "matrix": "COMPOSED_TRANSFORM",
            "invert": "INVERT"
        },
        {
            "type": "writers.text",
            "filename": "OUTPUT_FILEPATH"
        }
    ]
    '''
    # Run the pipeline with the correct parameters
    run_pdal_pipeline(pdal_pipeline, parameters={
        "INPUT_FILEPATH": tempfiles["points"],
        "COMPOSED_TRANSFORM": composed_transform,
        "INVERT": str(inverse).lower(),
        "OUTPUT_FILEPATH": tempfiles["transformed_points"]
    })

    # Read the transformed points as a DataFrame again
    transformed_points = pd.read_csv(tempfiles["transformed_points"])

    return transformed_points


def show_processing_log():
    """Show a formatted version of the processing log if one exists."""
    log_path = os.path.join(inputs.TEMP_DIRECTORY, "progress.log")
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"Log not found: {log_path}")

    log = pd.read_csv(log_path, names=["date", "dataset", "event"])
    log.index = pd.to_datetime(log["date"], format="%Z %Y/%m/%d %H:%M:%S")

    # Sort all dataset processing runs into separate numberings
    # A new run is assumed to always start with the event "Processing started"
    processing_nr = 0
    for i, row in log.iterrows():
        if "Processing started" in row["event"]:
            processing_nr += 1

        log.loc[i, "processing_nr"] = processing_nr

    for processing_nr, processing_log in log.groupby("processing_nr"):
        processing_time = processing_log.index.max() - processing_log.index.min()

        print(processing_log.iloc[-1]["dataset"])
        for date, row in processing_log.iterrows():
            print(f"\t{date}\t{row['event']}")
        print(f"Duration: {processing_time}\n")

    n_dems = len([filepath for filepath in os.listdir("export/dems") if ".tif" in filepath])
    print(f"Currently at {n_dems} DEMs")


def is_dataset_finished(dataset: str) -> bool:
    """Check if 'finished' exists on a progress row with the dataset."""
    log_path = os.path.join(inputs.TEMP_DIRECTORY, "progress.log")

    with open(log_path) as infile:
        log = infile.read().splitlines()

    # A bug (05/02/2021) led to some chunks not generating dense clouds but registering as finished
    # This checks for "finished" and "Made 0 dense clouds"
    has_zero_dense_clouds = False
    for line in log:
        if dataset not in line:
            continue
        if "Made 0 dense clouds" in line:
            has_zero_dense_clouds = True
        elif "dense clouds" in line:
            has_zero_dense_clouds = False

        if "finished" in line:
            if has_zero_dense_clouds:
                continue
            return True

    return False
    
