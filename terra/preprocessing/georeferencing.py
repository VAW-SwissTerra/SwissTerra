from __future__ import annotations

import os
import subprocess

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import scipy.interpolate
import statictypes

from terra import files, metadata
from terra.constants import CONSTANTS

TEMP_DIRECTORY = os.path.join(files.TEMP_DIRECTORY, "preprocessing")


CACHE_FILES = {
    "slope_map": os.path.join(TEMP_DIRECTORY, "base_DEM_slope.tif"),
    "aspect_map": os.path.join(TEMP_DIRECTORY, "base_DEM_aspect.tif")
}


def get_slope_and_aspect(redo: bool = False) -> tuple[rio.DatasetReader, rio.DatasetReader]:

    filepaths = []
    for name in ["slope", "aspect"]:
        filepath = CACHE_FILES[f"{name}_map"]
        filepaths.append(filepath)
        if os.path.isfile(filepath) and not redo:
            continue
        gdal_commands = ["gdaldem", name, files.INPUT_FILES["base_DEM"], filepath]
        print(f"Generating {name} map")
        subprocess.run(gdal_commands, check=True)

    # Return the slope and the aspect
    return (rio.open(filepaths[0]), rio.open(filepaths[1]))


def get_metadata_with_sampled_elevation() -> pd.DataFrame:
    image_meta = metadata.image_meta.read_metadata()
    with rio.open(files.INPUT_FILES["base_DEM"]) as base_dem:
        image_meta["dem_elevation"] = np.fromiter(base_dem.sample(
            image_meta[["easting", "northing"]].values, indexes=1), dtype=np.float64)
        # Set values that are approximately nodata to nan
        image_meta.loc[image_meta["dem_elevation"] < base_dem.nodata + 100, "dem_elevation"] = np.nan

    slope, aspect = get_slope_and_aspect()
    # Sample slope values
    image_meta["dem_slope"] = np.fromiter(slope.sample(
        image_meta[["easting", "northing"]].values, indexes=1), dtype=np.float64)
    # Sample aspect values
    image_meta["dem_aspect"] = np.fromiter(aspect.sample(
        image_meta[["easting", "northing"]].values, indexes=1), dtype=np.float64)
    image_meta.loc[image_meta["dem_aspect"] < -9000, "dem_aspect"] = np.nan

    slope.close()
    aspect.close()

    # Convert the image metadata to a geodataframe and drop rows where no DEM values could be extracted.
    camera_locations = gpd.GeoDataFrame(
        image_meta,
        geometry=gpd.points_from_xy(image_meta["easting"], image_meta["northing"]),
        crs=CONSTANTS.crs_epsg.replace("::", ":")
    ).dropna(how="any", subset=["dem_elevation", "dem_slope", "dem_aspect"]).reset_index()

    # Load the 1935 glacier outlines to find which cameras were taken on a glacier.
    outlines_1935 = gpd.read_file(files.INPUT_FILES["outlines_1935"])

    # Remove all the cameras that are taken on a glacier.
    indices_inside_glacier, _ = camera_locations.sindex.query_bulk(outlines_1935.geometry, predicate="intersects")
    camera_locations = camera_locations[~np.isin(camera_locations.index, indices_inside_glacier)]

    camera_locations["z_diff"] = camera_locations["dem_elevation"] - camera_locations["altitude"]
    print(camera_locations["z_diff"].median())
    camera_locations["z_diff"] += 2.4
    camera_locations["xy_diff"] = camera_locations["z_diff"] / np.tan(np.deg2rad(camera_locations["dem_slope"]))
    camera_locations = camera_locations[camera_locations["xy_diff"].abs() < 400]
    camera_locations["y_diff"] = camera_locations["xy_diff"] * np.sin(np.deg2rad(camera_locations["dem_aspect"]))
    camera_locations["x_diff"] = camera_locations["xy_diff"] * np.cos(np.deg2rad(camera_locations["dem_aspect"]))

    xx, yy = np.meshgrid(camera_locations["easting"], camera_locations["northing"])
    zs = camera_locations["z_diff"].values
    num = 200

    grid_x = np.linspace(camera_locations["easting"].min(), camera_locations["easting"].max(), num=num)
    grid_y = np.linspace(camera_locations["northing"].min(), camera_locations["northing"].max(), num=num)

    plt.scatter(camera_locations["easting"], camera_locations["northing"],
                c=camera_locations["y_diff"], vmin=-5, vmax=5)
    plt.colorbar()
    plt.show()
    return
#    interpolated = scipy.interpolate.griddata(
#        camera_locations[["easting", "northing"]].values, camera_locations["z_diff"].values, np.meshgrid(grid_x, grid_y))

#    print(interpolated)

 #   plt.imshow(interpolated, vmin=-20, vmax=20)
 #   plt.show()

    for i, col in enumerate(["x_diff", "y_diff", "z_diff"]):
        lim = 20
        plt.subplot(3, 1, i + 1)
        plt.title(col)
        plt.hist(camera_locations.loc[camera_locations[col].abs() < lim, col], bins=100)
        plt.xlim(-lim, lim)

    plt.xlabel("Offset (m)")
    plt.tight_layout()
    plt.show()

    return
    plt.scatter(camera_locations.index, camera_locations["dem_elevation"] - camera_locations["altitude"])
    plt.show()

    return camera_locations


if __name__ == "__main__":
    get_metadata_with_sampled_elevation()
