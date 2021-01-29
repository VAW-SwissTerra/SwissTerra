import concurrent.futures
import json
import os
import subprocess
import tempfile
from collections import deque

import jinja2
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import rasterio.warp
from tqdm import tqdm

from terra import evaluation, files
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

    ortho_path = os.path.join(CACHE_FILES["metashape_ortho_dir"], f"{station_name}_orthomosaic.tif")
    dem_path = os.path.join(CACHE_FILES["metashape_dem_dir"], f"{station_name}_dense_DEM.tif")
    coreg_meta_path = os.path.join(evaluation.CACHE_FILES["dem_coreg_meta_dir"], f"{station_name}_coregistration.json")
    out_filepath = os.path.join(CACHE_FILES["ortho_coreg_dir"], f"{station_name}_ortho_coreg.tif")

    if not overwrite and os.path.isfile(out_filepath):
        return

    for path in [ortho_path, dem_path, coreg_meta_path]:
        if not os.path.isfile(path):
            return

    if not os.path.isdir(os.path.dirname(out_filepath)):
        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

    temp_dir = tempfile.TemporaryDirectory()
    temp_ortho_dem_path = os.path.join(temp_dir.name, "ortho.xyz")

    ortho = rio.open(ortho_path)

    dem = rio.open(dem_path)
    x_coords, y_coords = np.meshgrid(
        np.arange(dem.bounds.left, dem.bounds.right,
                  step=CONSTANTS.dem_resolution) + CONSTANTS.dem_resolution / 2,
        np.arange(dem.bounds.bottom, dem.bounds.top,
                  step=CONSTANTS.dem_resolution)[::-1] + CONSTANTS.dem_resolution / 2
    )
    resampled_ortho = np.zeros((dem.height, dem.width), dtype=ortho.dtypes[0])
    resampled_mask = np.zeros(resampled_ortho.shape)

    reprojection_params = dict(
        src_transform=ortho.transform,
        dst_transform=dem.transform,
        src_crs=ortho.crs,
        dst_crs=dem.crs,
        dst_resolution=dem.res
    )

    rio.warp.reproject(ortho.read(1), destination=resampled_ortho, **
                       reprojection_params, resampling=rio.warp.Resampling.bilinear)
    rio.warp.reproject(ortho.read(2), destination=resampled_mask, **reprojection_params)
    resampled_ortho[resampled_ortho == 255] -= 1

    dem_vals = dem.read(1)
    mask = (resampled_mask == 255) & (dem_vals != -9999)

    if np.count_nonzero(mask) == 0:
        return

    merged_vals = np.dstack((x_coords[mask], y_coords[mask], dem_vals[mask], resampled_ortho[mask]))[0, :, :]

    bounds = np.array([merged_vals[:, 0].min(), merged_vals[:, 0].max(),
                       merged_vals[:, 1].min(), merged_vals[:, 1].max()])
    bounds -= bounds % CONSTANTS.dem_resolution
    bounds[[1, 3]] += CONSTANTS.dem_resolution

    np.savetxt(temp_ortho_dem_path, merged_vals, delimiter=",", fmt=["%f"] * 3 + ["%d"])

    with open(coreg_meta_path) as infile:
        coreg_meta = json.load(infile)

    meta = processing_tools.run_pdal_pipeline(jinja2.Template("""
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

    gdal_commands = ["gdal_edit.py", "-a_srs", CONSTANTS.crs_epsg.replace("::", ":"), "-a_nodata", "255", out_filepath]
    subprocess.run(gdal_commands, check=True)


def apply_coregistrations(overwrite: bool = False):

    stations = np.unique([evaluation.extract_station_name(filename)
                          for filename in os.listdir(CACHE_FILES["metashape_ortho_dir"])])

    progress_bar = tqdm(total=len(stations), desc="Transforming orthomosaics")

    def apply_coreg(station_name: str):

        apply_coreg_to_ortho(station_name, overwrite=overwrite)
        progress_bar.update()

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(apply_coreg, stations))

    progress_bar.close()

    ortho_filepaths = [os.path.join(CACHE_FILES["ortho_coreg_dir"], filename)
                       for filename in os.listdir(CACHE_FILES["ortho_coreg_dir"])]

    evaluation.merge_rasters(ortho_filepaths, output_filepath=CACHE_FILES["merged_ortho"])
