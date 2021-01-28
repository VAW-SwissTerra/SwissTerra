from __future__ import annotations

import os
import subprocess
import tempfile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import rasterio.features
import rasterio.fill
import scipy.interpolate
import shapely

from terra import files
from terra.constants import CONSTANTS

TEMP_DIRECTORY = os.path.join(files.TEMP_DIRECTORY, "base_dem")


CACHE_FILES = {
    "base_dem": os.path.join(TEMP_DIRECTORY, "base_dem.tif"),
    "base_dem_years": os.path.join(TEMP_DIRECTORY, "base_dem_years.tif"),
    "hillshade": os.path.join(TEMP_DIRECTORY, "base_dem_hillshade.tif"),
    "slope": os.path.join(TEMP_DIRECTORY, "base_dem_slope.tif"),
    "aspect": os.path.join(TEMP_DIRECTORY, "base_dem_aspect.tif")
}


def read_metadata():
    filepath = "input/shapefiles/swissALTI3D_Metatatenlayer_032019/Metadata_SwissALTI3D_preRelease2019_LV95.shp"

    data = gpd.read_file(filepath).to_crs(CONSTANTS.crs_epsg.replace("::", ":"))

    def mean_year(row: gpd.GeoSeries):
        years = np.array(row["yr_vl_his"].replace("NA", "nan").split(",")).astype(float)
        years += (9 / 12)  # Assume that all years are in September
        counts = np.array(row["yr_cnt_his"].split(",")).astype(float)
        average_year = np.average(years[~np.isnan(years)], weights=counts[~np.isnan(years)])
        return average_year
    data["mean_year"] = data.apply(mean_year, axis=1)
    return data


def rasterize_year():
    data = read_metadata()

    temp_dir = tempfile.TemporaryDirectory()
    data_temp_path = os.path.join(temp_dir.name, "data.shp")
    data.to_file(data_temp_path)

    template_raster = rio.open(CACHE_FILES["base_dem"])

    gdal_commands = ["gdal_grid",
                     "-txe", template_raster.bounds.left, template_raster.bounds.right,
                     "-tye", template_raster.bounds.bottom, template_raster.bounds.top,
                     "-outsize", template_raster.width, template_raster.height,
                     "-a_srs", CONSTANTS.crs_epsg.replace("::", ":"),
                     "-zfield", "mean_year",
                     "-a", "nearest:radius1=0.0:radius2=0.0:angle=0.0:nodata=0.0",
                     "-co", "COMPRESS=DEFLATE",
                     "-ot", "Float32",
                     "--config", "GDAL_NUM_THREADS=ALL_CPUS",
                     data_temp_path,
                     CACHE_FILES["base_dem_years"]]
    subprocess.run(list(map(str, gdal_commands)), check=True)


def reproject_base_dem():

    orig_dem = rio.open(files.INPUT_FILES["base_DEM"])

    bounds = gpd.GeoDataFrame(index=[0, 1], geometry=[
        shapely.geometry.Point([orig_dem.bounds.left, orig_dem.bounds.top]),
        shapely.geometry.Point([orig_dem.bounds.right, orig_dem.bounds.bottom])],
        crs=orig_dem.crs).to_crs(CONSTANTS.crs_epsg.replace("::", ":"))

    left, right = bounds.geometry.x - (bounds.geometry.x % CONSTANTS.dem_resolution)
    top, bottom = bounds.geometry.y - (bounds.geometry.y % CONSTANTS.dem_resolution)

    gdal_bounds = [left, bottom, right + CONSTANTS.dem_resolution, top + CONSTANTS.dem_resolution]

    gdal_commands = ["gdalwarp",
                     "-t_srs", CONSTANTS.crs_epsg.replace("::", ":"),
                     "-tr", CONSTANTS.dem_resolution, CONSTANTS.dem_resolution,
                     "-te", *gdal_bounds,
                     "-te_srs", CONSTANTS.crs_epsg.replace("::", ":"),
                     "-co", "COMPRESS=DEFLATE",
                     "-co", "BIGTIFF=YES",
                     "-r", "bilinear",
                     "-wo", "NUM_THREADS=ALL_CPUS",
                     "-multi",
                     files.INPUT_FILES["base_DEM"],
                     CACHE_FILES["base_dem"]]

    subprocess.run(list(map(str, gdal_commands)), check=True)


def make_hillshade(overwrite: bool = False):

    if not overwrite and os.path.isfile(CACHE_FILES["hillshade"]):
        return

    gdal_commands = [
        "gdaldem", "hillshade",
        "-co", "COMPRESS=DEFLATE",
        "-multidirectional",
        CACHE_FILES["base_dem"],
        CACHE_FILES["hillshade"]
    ]

    subprocess.run(list(map(str, gdal_commands)), check=True)


def make_slope_and_aspect(overwrite: bool = False):

    for product in ["slope", "aspect"]:
        if not overwrite and os.path.isfile(CACHE_FILES[product]):
            return

        gdal_commands = [
            "gdaldem", product,
            "-co", "COMPRESS=DEFLATE",
            "-co", "BIGTIFF=YES",
            CACHE_FILES["base_dem"],
            CACHE_FILES[product]
        ]

        print(f"Generating {product}...")
        subprocess.run(list(map(str, gdal_commands)), check=True)


if __name__ == "__main__":
    reproject_base_dem()
