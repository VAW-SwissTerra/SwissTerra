"""Functions to process data."""
import json
import os
import subprocess
from collections import namedtuple
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import rasterio as rio
import scipy.ndimage
import statictypes

from terra import constants, files
from terra.processing import inputs
from terra.utilities import no_stdout

CACHE_FILES = {
    "asift_temp_dir": os.path.join(inputs.TEMP_DIRECTORY, "asift_temp")
}


@statictypes.enforce
def generate_dem(point_cloud_path: str, output_dem_path: str,
                 resolution: float = 1.0, interpolate_pixels: int = 10) -> None:
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
    gdal_commands = ["gdal_edit.py", "-a_srs", constants.CONSTANTS["crs_epsg"].replace("::", ":"), output_dem_path]
    subprocess.run(gdal_commands, check=True)

    return output_dem_path


@ statictypes.enforce
def run_pdal_pipeline(pipeline: str, output_metadata_file: Optional[str] = None) -> None:
    """
    Run a PDAL pipeline.

    param: pipeline: The pipeline to run.
    param: output_metadata_file: Optional. The filepath for the pipeline metadata.
    """
    commands = ["pdal", "pipeline", "--stdin"]

    if output_metadata_file is not None:
        commands += ["--metadata", output_metadata_file]

    subprocess.run(commands, input=pipeline, check=True, encoding="utf-8")


def match_asift(filepath1: str, filepath2: str, verbose: bool = True) -> pd.DataFrame:
    """
    Find matches between two images using the ASIFT (Affine-Scale Invariant Feature transform).

    param: filepath1: The first image filepath to match.
    param: filepath2: The second image filepath to match.
    param: verbose: Whether to print ASIFT outputs.

    return: matches: The computed matches between the images.
    """
    image1 = cv2.imread(filepath1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(filepath2, cv2.IMREAD_GRAYSCALE)

    # Trim the image to remove the frame and reduce it so ASIFT can use it
    side_trim = 500  # To remove the frame
    new_width = 2500  # To downscale the image so ASIFT can work
    os.makedirs(CACHE_FILES["asift_temp_dir"], exist_ok=True)

    # Go through each image and crop, resize, then save it as a PNG (needed for ASIFT)
    for i, image in enumerate([image1, image2]):
        trimmed_image = image[side_trim:-side_trim, side_trim:-side_trim]

        new_height = int(trimmed_image.shape[0] * (new_width / trimmed_image.shape[1]))
        resized_image = cv2.resize(trimmed_image, (new_width, new_height))
        assert resized_image.shape[1] == new_width  # Check that everything went right

        cv2.imwrite(os.path.join(CACHE_FILES["asift_temp_dir"], f"image_{i + 1}.png"), resized_image)

    # Files that ASIFT needs/produces (first/second image input, horizontal/vertical match plot, matches, and keypoints)
    asift_files = ["image_1.png", "image_2.png", "matches_vert.png",
                   "matches_horiz.png", "matches.txt", "keys1.txt", "keys2.txt"]
    # Put the commands in order (asift INPUTS+OUTPUTS 0). The 0 means don't resize to 800x600
    asift_commands = ["asift"] + [os.path.join(CACHE_FILES["asift_temp_dir"], filename)
                                  for filename in asift_files] + ["0"]
    # Run ASIFT
    with no_stdout(disable=verbose):
        subprocess.run(asift_commands, check=True)

    # Load the matches produced by ASIFT
    matches = pd.read_csv(os.path.join(CACHE_FILES["asift_temp_dir"], "matches.txt"), delimiter="  ",
                          skiprows=1, engine="python", names=["img1_x", "img1_y", "img2_x", "img2_y"])

    # Convert the match coordinates from trimmed/scaled to the original size
    for label, image in zip(["img1", "img2"], [image1, image2]):
        matches[[f"{label}_x", f"{label}_y"]] *= (image.shape[1] - side_trim * 2) / new_width
    matches += side_trim

    return matches


def coalign_dems(reference_path: str, aligned_path: str, pixel_buffer=10, nan_value=-9999) -> Optional[Dict[str, str]]:
    """
    Align two DEMs and return the ICP result.

    param: reference_path: The filepath to the DEM acting reference.
    param: aligned_path: The filepath to the DEM acting aligned.
    param: pixel_buffer: The amount of allowed overlap between the clouds (should be the expected offset).
    param: nan_value: The NaN value of both DEMs.

    return: result: The resultant PDAL output. Returns None if there was no alignment.

    """
    # TODO: Replace the temporary files with actual non-persistent tempfiles
    # Create a bounds object to more explicitly handle bounds.
    Bounds = namedtuple("Bounds", ["x_min", "x_max", "y_min", "y_max"])

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
    reference_original = cv2.imread(reference_path, cv2.IMREAD_ANYDEPTH)
    aligned_original = cv2.imread(aligned_path, cv2.IMREAD_ANYDEPTH).astype(reference_original.dtype)

    # Set the nan_value to actual nans
    reference_original[reference_original == nan_value] = np.nan
    aligned_original[aligned_original == nan_value] = np.nan

    # Get the resolution by dividing the real world width with the pixel width
    resolution = (reference_bounds.x_max - reference_bounds.x_min) / reference_original.shape[1]

    # Create new bounding coordinates that encompass both datasets
    new_bounds = Bounds(
        min(reference_bounds.x_min, aligned_bounds.x_min),
        max(reference_bounds.x_max, aligned_bounds.x_max),
        min(reference_bounds.y_min, aligned_bounds.y_min),
        max(reference_bounds.y_max, aligned_bounds.y_min))

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

    # Assign filepaths to the new cropped DEMs
    new_reference_path = os.path.splitext(reference_path)[0] + "_cropped.tif"
    new_aligned_path = os.path.splitext(aligned_path)[0] + "_cropped.tif"

    # Write the cropped DEMs
    write_raster(new_reference_path, reference)
    write_raster(new_aligned_path, aligned)

    # Set the parameters for PDAL to include in the ICP registration.
    pdal_params = {
        "REFERENCE_FILENAME": new_reference_path,
        "ALIGNED_FILENAME": new_aligned_path,
        "OUTPUT_FILENAME": os.path.splitext(new_aligned_path)[0] + "_aligned.tif",
        "RESOLUTION": resolution,
    }

    # Read the datasets (A1, B1), filter out nans (A2, B2) and then run ICP
    # TODO: Maybe remove the output raster
    pdal_pipeline = '''
    [
        {
            "type": "readers.gdal",
            "filename": "REFERENCE_FILENAME",
            "header": "Z",
            "tag": "A1"
        },
        {
            "type": "readers.gdal",
            "filename": "ALIGNED_FILENAME",
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
            "filename": "OUTPUT_FILENAME",
            "resolution": "RESOLUTION",
            "output_type": "mean"
        }
    ]
    '''
    # Fill the pdal parameters to the pipeline
    for key in pdal_params:
        pdal_pipeline = pdal_pipeline.replace(key, str(pdal_params[key]))

    # Assign a filepath to the output metadata
    output_metadata_path = os.path.splitext(new_aligned_path)[0] + "_icp_meta.json"

    # Run the pdal pipeline
    run_pdal_pipeline(pdal_pipeline, output_metadata_file=output_metadata_path)

    # Open the resultant metadata file and extract the ICP results.
    with open(output_metadata_path) as infile:
        result = json.loads(infile.read())["stages"]["filters.icp"]

    return result


if __name__ == "__main__":
    coalign_dems("/home/erik/Projects/ETH/SwissTerra/temp/processing/rhone/temp/station_1666_local_dense_DEM.tif",
                 "/home/erik/Projects/ETH/SwissTerra/temp/processing/rhone/temp/station_1668_local_dense_DEM.tif")
