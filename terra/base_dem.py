from __future__ import annotations

import subprocess
import tempfile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import rasterio.features
import rasterio.fill
import scipy.interpolate

from terra.constants import CONSTANTS


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
    data.to_file("/tmp/shapefile.shp")
    return data


def rasterize_year():
    #data = read_metadata()
    data = gpd.read_file("/tmp/shapefile.shp")
    # this is where we create a generator of geom, value pairs to use in rasterizing
    shapes = ((geom, value) for geom, value in zip(data.geometry, data["mean_year"]))

    template_raster = rio.open("temp/evaluation/merged_ddem.tif")

    gdal_commands = ["gdal_grid",
                     "-txe", template_raster.bounds.left, template_raster.bounds.right,
                     "-tye", template_raster.bounds.bottom, template_raster.bounds.top,
                     "-outsize", template_raster.width, template_raster.height,
                     "-a_srs", CONSTANTS.crs_epsg.replace("::", ":"),
                     "-zfield", "mean_year",
                     "-a", "nearest:radius1=0.0:radius2=0.0:angle=0.0:nodata=0.0",
                     "--config", "GDAL_NUM_THREADS=ALL_CPUS",
                     "/tmp/shapefile.shp",
                     "/tmp/rasterized.tif"]
    subprocess.run(list(map(str, gdal_commands)), check=True)

    return

    rasterized_years = rio.features.rasterize(
        shapes,
        out_shape=template_raster.shape,
        transform=template_raster.transform,
        fill=np.nan)

    meta = template_raster.meta

    with rio.open("/tmp/raster.tif", "w", **meta) as raster:
        raster.write(rasterized_years, 1)


if __name__ == "__main__":
    rasterize_year()
