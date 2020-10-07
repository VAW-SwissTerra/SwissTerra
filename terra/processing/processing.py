"""Functions to process data."""
import os
import subprocess
from typing import Optional

import statictypes

from terra import constants, files
from terra.processing import inputs


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

    subprocess.run(commands, input=pipeline,  check=True, encoding="utf-8")
