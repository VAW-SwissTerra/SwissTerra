"""Functions to process data."""
import os
import subprocess
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import statictypes

from terra import constants, files
from terra.processing import inputs

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

    params = {
        "INPUT_CLOUD": point_cloud_path,
        "RESOLUTION": resolution,
        "OUTPUT_DEM": output_dem_path,

    }
    # Construct the pipeline for PDAL
    pdal_pipeline = '''
    [
        "INPUT_CLOUD",
        {
            "type": "filters.range",
            "limits": "confidence[2:]"
        },
        {
            "resolution": RESOLUTION,
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


def match_asift(image1: np.ndarray, image2: np.ndarray) -> pd.DataFrame:
    """
    Find matches between two images using the ASIFT (Affine-Scale Invariant Feature transform).

    param: image1: The first image to match.
    param: image2: The second image to match.

    return: matches: The computed matches between the images.
    """
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
    subprocess.run(asift_commands, check=True)

    # Load the matches produced by ASIFT
    matches = pd.read_csv(os.path.join(CACHE_FILES["asift_temp_dir"], "matches.txt"), delimiter="  ",
                          skiprows=1, engine="python", names=["img1_x", "img1_y", "img2_x", "img2_y"])

    # Convert the match coordinates from trimmed/scaled to the original size
    for label, image in zip(["img1", "img2"], [image1, image2]):
        matches[[f"{label}_x", f"{label}_y"]] *= (image.shape[1] - side_trim * 2) / new_width
    matches += side_trim

    matches.to_csv(os.path.join(CACHE_FILES["asift_temp_dir"], "fixed_matches.csv"))
    return matches
