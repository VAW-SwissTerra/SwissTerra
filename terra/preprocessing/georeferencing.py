"""
Functions for correcting the georeferencing information given my the metadata.

Corrections are estimated from the spatial distribution of offsets from the reference DEM.
"""
from __future__ import annotations

import os
import subprocess

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import scipy.interpolate

from terra import base_dem, files, preprocessing
from terra.constants import CONSTANTS

TEMP_DIRECTORY = os.path.join(files.TEMP_DIRECTORY, "preprocessing")


CACHE_FILES = {
    "slope_map": os.path.join(TEMP_DIRECTORY, "base_DEM_slope.tif"),
    "aspect_map": os.path.join(TEMP_DIRECTORY, "base_DEM_aspect.tif")
}


def get_slope_and_aspect(redo: bool = False) -> tuple[rio.DatasetReader, rio.DatasetReader]:
    """
    Generate and/or load slope and aspect maps from the base DEM.

    :param redo: If the maps should be redone even though they exist.

    :returns: Rasterio dataset readers for the rasters (slope, aspect).
    """
    # Generate a slope and aspect map using GDAL
    filepaths = []
    for name in ["slope", "aspect"]:
        filepath = CACHE_FILES[f"{name}_map"]
        filepaths.append(filepath)
        # Skip if the file already exists (but not if it should be redone)
        if os.path.isfile(filepath) and not redo:
            continue
        # Use gdal to create the map
        gdal_commands = ["gdaldem", name, base_dem.CACHE_FILES["base_dem"],
                         filepath, "-co", "COMPRESS=DEFLATE", "-co", "BIGTIFF=YES"]
        print(f"Generating {name} map")
        subprocess.run(gdal_commands, check=True)

    # Return the slope and the aspect
    return (rio.open(filepaths[0]), rio.open(filepaths[1]))


def correct_metadata_coordinates(old_camera_locations: pd.DataFrame,
                                 nocorr=False, gridsize=250) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """
    Use a base DEM to correct systematic spatial offsets in the location data.

    :param old_camera_locations: The initial camera_locations DataFrame.
    :param nocorr: Whether to only calculate the differences but not correct for them (for visualisation).
    :param gridsize: The width/height of the offset fields to generate (250 ~= 1 km / px)

    :returns: The correction fields and the corrected camera locations.
    """
    # Load the 1935 glacier outlines to find which cameras were taken on a glacier.
    outlines_1935 = gpd.read_file(files.INPUT_FILES["outlines_1935"])

    # Remove all the cameras that are taken on a glacier.
    indices_inside_glacier, _ = old_camera_locations.sindex.query_bulk(outlines_1935.geometry, predicate="intersects")
    cameras_on_terrain = old_camera_locations[~np.isin(old_camera_locations.index, indices_inside_glacier)]

    cameras_on_terrain["elevation"] = cameras_on_terrain["altitude"] - CONSTANTS.tripod_height

    # Open the DEM and read elevation values from it
    with rio.open(base_dem.CACHE_FILES["base_dem"]) as base_dem_dataset:
        cameras_on_terrain["dem_elevation"] = np.fromiter(base_dem_dataset.sample(
            cameras_on_terrain[["easting", "northing"]].values, indexes=1), dtype=np.float64)
        # Set values that are approximately nodata to nan
        cameras_on_terrain.loc[cameras_on_terrain["dem_elevation"] <
                               base_dem_dataset.nodata + 100, "dem_elevation"] = np.nan

    # Read/create the slope and aspect maps
    slope, aspect = get_slope_and_aspect()
    # Sample slope values
    cameras_on_terrain["dem_slope"] = np.fromiter(slope.sample(
        cameras_on_terrain[["easting", "northing"]].values, indexes=1), dtype=np.float64)
    # Sample aspect values
    cameras_on_terrain["dem_aspect"] = np.fromiter(aspect.sample(
        cameras_on_terrain[["easting", "northing"]].values, indexes=1), dtype=np.float64)
    # Remove nodata values in the DEM aspect samples
    cameras_on_terrain.loc[cameras_on_terrain["dem_aspect"] < -9000, "dem_aspect"] = np.nan

    # Close the datasets
    slope.close()
    aspect.close()

    # Calculate the difference between the base DEM elevation and the measured elevation
    cameras_on_terrain["z_diff"] = cameras_on_terrain["dem_elevation"] - cameras_on_terrain["elevation"]
    # Get the horizontal component of the offset using the tangent of the slope
    cameras_on_terrain["xy_diff"] = cameras_on_terrain["z_diff"] / np.tan(np.deg2rad(cameras_on_terrain["dem_slope"]))
    # Remove outliers that skew the offset fields
    cameras_on_terrain = cameras_on_terrain[cameras_on_terrain["xy_diff"].abs() < 10]
    # Calculate the x (easting) and y (northing) components from the aspect of the terrain.
    cameras_on_terrain["y_diff"] = cameras_on_terrain["xy_diff"] * np.sin(np.deg2rad(cameras_on_terrain["dem_aspect"]))
    cameras_on_terrain["x_diff"] = cameras_on_terrain["xy_diff"] * np.cos(np.deg2rad(cameras_on_terrain["dem_aspect"]))

    # Stop here if no correction should be done.
    if nocorr:
        return cameras_on_terrain

    # Upscale the gridding to this size to get better localised values
    upscale_size = 5000

    # Create an equally spaced grid that covers the entire dataset to interpolate values within.
    grid_x, grid_y = np.meshgrid(
        np.linspace(cameras_on_terrain["easting"].min(), cameras_on_terrain["easting"].max(), num=gridsize),
        np.linspace(cameras_on_terrain["northing"].min(), cameras_on_terrain["northing"].max(), num=gridsize)
    )

    # Make corresponding x and y values in the shape of the upscaled interpolated grid
    upscaled_xs = np.linspace(grid_x.min(), grid_x.max(), num=upscale_size)
    upscaled_ys = np.linspace(grid_y.min(), grid_y.max(), num=upscale_size)

    def closest_index(arr: np.ndarray, value: float) -> int:
        """
        Find the index of a cell that shares the most resemblance to a given value.

        :param arr: The 1D array to query.
        :param value: The value to find the corresponding index to in 'arr'

        :returns: The closest index in 'arr'
        """
        return np.abs(arr - value).argmin()

    def correct_coordinate(x_coord: float, y_coord: float, z_coord: float, x_correction_array: np.ndarray,
                           y_correction_array: np.ndarray, z_correction_array: np.ndarray) -> list[float]:
        """
        Correct a 3D coordinate using coordinate correction fields.

        :param x_coord: The x-coordinate (easting).
        :param y_coord: The y-coordinate (northing).
        :param z_coord: The z-coordinate (altitude).
        :param x_correction_array: The x correction field.
        :param y_correction_array: The y correction field.
        :param z_correction_array: The z correction field.

        return: [new_x_coord, new_y_coord, new_z_coord]
        """
        # Find the closest index in the arrays
        closest_x_index = closest_index(upscaled_xs, x_coord)
        closest_y_index = closest_index(upscaled_ys, y_coord)

        # Find the offsets of said index
        # TODO: The x and y directions are seemingly flipped. Find out why
        x_corr = x_correction_array[closest_x_index, closest_y_index]
        y_corr = y_correction_array[closest_x_index, closest_y_index]
        z_corr = z_correction_array[closest_x_index, closest_y_index]

        # Make new coordinates, assuming the corrections were not nan
        new_x_coord = x_coord + x_corr if not np.isnan(x_corr) else x_coord
        new_y_coord = y_coord + y_corr if not np.isnan(y_corr) else y_coord
        new_z_coord = z_coord + z_corr if not np.isnan(z_corr) else z_coord

        # TODO: Change this to a tuple for better annotation?
        return [new_x_coord, new_y_coord, new_z_coord]

    # Get the corrections for each dimension (x, y, z)
    corrs = {dim: cv2.resize(  # Upscale the grid once it's made
        scipy.interpolate.griddata(  # Grid the data
            points=cameras_on_terrain[["easting", "northing"]],
            values=cameras_on_terrain[dim],
            xi=(grid_x, grid_y),
            method="linear"  # Use linear interpolation
        ),
        dsize=(upscale_size, upscale_size),  # Upscale to this size
        interpolation=cv2.INTER_CUBIC  # Upscale using cubic interpolation
    ) for dim in ["x_diff", "y_diff", "z_diff"]}  # Run this for each dimension

    corrected_camera_locations = old_camera_locations.copy()
    # Apply the correction fields along all rows
    corrected_camera_locations[["easting", "northing", "altitude"]] = np.apply_along_axis(
        lambda row: correct_coordinate(
            x_coord=row[0],
            y_coord=row[1],
            z_coord=row[2],
            x_correction_array=corrs["x_diff"],
            y_correction_array=corrs["y_diff"],
            z_correction_array=corrs["z_diff"]
        ),
        axis=1,  # Row-wise should be 0! Why is this 1??!!!!
        arr=corrected_camera_locations[["easting", "northing", "altitude"]].values
    )

    return corrs, corrected_camera_locations


def generate_corrected_metadata() -> pd.DataFrame:
    """
    Load and correct the image location metadata.

    :returns: The metadata file with corrected camera locations
    """
    print("Correcting georeferencing information.")
    image_meta = preprocessing.image_meta.read_metadata()

    # Set the gridsize to be approximately 1 km / px
    gridsize = 250

    # Convert the metadata to a GeoDataFrame
    camera_locations = gpd.GeoDataFrame(
        image_meta,
        geometry=gpd.points_from_xy(image_meta["easting"], image_meta["northing"]),
        crs=CONSTANTS.crs_epsg.replace("::", ":")
    )

    # Run the correction
    _, corrected_camera_locations = correct_metadata_coordinates(camera_locations, gridsize=gridsize)

    return corrected_camera_locations


def plot_metadata_position_correction():
    """Plot the correction maps and error distributions."""
    image_meta = preprocessing.image_meta.read_metadata()

    # Set the gridsize to be approximately 1 km / px
    gridsize = 250

    # Convert the metadata to a GeoDataFrame
    camera_locations = gpd.GeoDataFrame(
        image_meta,
        geometry=gpd.points_from_xy(image_meta["easting"], image_meta["northing"]),
        crs=CONSTANTS.crs_epsg.replace("::", ":")
    )

    # Run the correction
    corrections, corrected_camera_locations = correct_metadata_coordinates(camera_locations, gridsize=gridsize)

    residuals, corrected_camera_locations = correct_metadata_coordinates(corrected_camera_locations, gridsize=gridsize)

    not_corrected = correct_metadata_coordinates(camera_locations, nocorr=True, gridsize=gridsize)

    resolution = (camera_locations["easting"].max() - camera_locations["easting"].min()) / gridsize
    print(resolution)

    plt.figure(figsize=(8, 5))
    lim = 15
    colours = ["green", "orange", "blue"]
    for i, df in enumerate([not_corrected, corrected_camera_locations]):
        label = "Uncorrected" if i == 0 else "Corrected"
        for j, dim in enumerate(residuals.keys()):
            plt.subplot2grid((3, 2), (j, i))

            plt.xlabel(f"{label} {dim} (m)")
            plt.hist(df[dim][df[dim].abs() < lim], bins=70, color=colours[j])

            plt.text(
                x=0.7,
                y=0.9,
                s="mean: {mean}\nmedian: {median}\nstd: {std}".format(
                    mean=round(df[dim].mean(), 1),
                    median=round(df[dim].median(), 1),
                    std=round(df[dim].std(), 1)
                ),
                transform=plt.gca().transAxes,
                va="top"
            )
            plt.xlim(-15, 15)

    plt.tight_layout()
    plt.savefig("temp/figures/spatial_coordinate_correction.png", dpi=600)
    plt.show()

    for i, dim in enumerate(residuals.keys()):

        plt.subplot(2, 3, i + 1)
        plt.xlabel(f"Corrections {dim} (m)")
        plt.imshow(corrections[dim], vmin=-20, vmax=20)

    plt.colorbar()
    plt.tight_layout()
    plt.show()
