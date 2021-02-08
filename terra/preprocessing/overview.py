#!/bin/env python
"""
Calculate which photographs have viewsheds covering what glaciers.
The output is a csv giving the distance of each photograph to each glacier.
If the photograph's viewshed does not cover a specific glacier, its distance value will be set to NaN.

"""
import datetime
import os

import earthpy.spatial
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from tqdm import tqdm

from terra import base_dem, files
from terra.preprocessing import image_meta

# Temporary files
TEMP_DIRECTORY = os.path.join(files.TEMP_DIRECTORY, "overview")
if not os.path.isdir(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY, exist_ok=True)

CACHE_FILES = {
    "camera_glacier_coverage": os.path.join(TEMP_DIRECTORY, "camera_glacier_coverage.pkl"),
}


def get_camera_glacier_viewshed_and_distances() -> gpd.GeoDataFrame:
    """
    Get the distances between each glacier and photograph, and calculate whether the photograph's viewsheds intersect which glaciers.
    The photographs without an intersecting viewsheds will have a NaN distance to the glacier in question.
    The process takes a long time, so it is saved to a file, which can later be used as a cache.

    return: camera_locations: Locations of each photograph with information of what glaciers they portray.
    """
    # Read SGI glacier polygons and convert its coordinate system to the same as the others
    print("Loading glacier outlines")
    glaciers = gpd.read_file(files.INPUT_FILES["sgi_1973"]).to_crs("EPSG:21781")
    # Read camera viewsheds
    print("Loading camera viewsheds")
    viewsheds = gpd.read_file(files.INPUT_FILES["viewsheds"])
    # Read the location of the images
    print("Loading camera locations")
    camera_locations = gpd.read_file(files.INPUT_FILES["camera_locations"])

    # Loop through each glacier and add its distance to each glacier in separate columns
    # Cameras without a viewshed that intersects the glacier will be set to NaN
    # TODO: Make this multithreaded?
    print("Measuring distances and checking viewsheds to glaciers for all images")
    for i, glacier in tqdm(glaciers.iterrows(), total=len(glaciers)):
        # glacier_code = f"glacier_{glacier['id']"  # This is used as the column name
        glacier_code = glacier["SGI"]
        # Measure the camera distances to the glacier and round to nearest metre (just to keep file size down a bit)
        camera_locations[glacier_code] = np.round(camera_locations.distance(glacier.geometry))
        # Calculate which viewsheds overlaps with the glacier
        # TODO: Check for potential errors (https://gis.stackexchange.com/questions/375407/geopandas-intersects-doesnt-find-any-intersection)
        overlap = viewsheds.intersects(glacier.geometry)
        # Get the INVENTORY_NUMBER values from each viewshed that overlaps with the glacier
        overlapping_viewshed_inventory_numbers = viewsheds.loc[overlap.index]["INVENTORY_"].values
        # Extract the cameras that do NOT have an overlapping viewshed, and set their distance to NaN
        camera_locations.loc[~camera_locations["INVENTORY_"].isin(
            overlapping_viewshed_inventory_numbers), glacier_code] = np.nan

    print("Finished. Saving...")
    # Save the file as a temporary cache
    camera_locations.to_pickle(CACHE_FILES["camera_glacier_coverage"])
    return camera_locations


def threshold_camera_distance(camera_locations: gpd.GeoDataFrame, max_distance: float = 2000.0) -> gpd.GeoDataFrame:
    """
    Set a maximum threshold for photograph-to-glacier distance.

    param: camera_locations: GeoDataFrame with camera locations and glacier distances.
    param: max_distance: The maximum allowed distance a glacier can be.

    return: camera_locations: With the exact same shape as the input, only with more NaNs!
    """
    # Extract all the distance values (all columns from index 24 and forward) to a Numpy Array
    distances = camera_locations.iloc[:, 24:].values
    # Set all the distances exceeding the max to NaN
    distances[distances > max_distance] = np.nan
    # Put the modified values back into the camera_locations GeoDataFrame
    camera_locations.iloc[:, 24:] = distances
    return camera_locations


def extract_largest_glaciers(camera_locations, area_fraction=0.8):
    """
    Extract the largest glaciers that correspond to a set fraction of their total area (e.g. 0.8 => 80% of the area is covered).
    The glaciers that are excluded due to the area threshold have their corresponding camera distances set to NaN.

    param: camera_locations: GeoDataFrame with camera locations and glacier distances.
    param: area_fraction: Fraction of area to keep in the GeoDataFrame.

    return: camera_locations: With the exact same shape as the input, only with more NaNs!
    """
    # Read the SGI outlines and convert the coordinate system to the one used by the other data
    glaciers = gpd.read_file(files.INPUT_FILES["sgi_1973"]).to_crs("EPSG:21781")
    # Sort the values by area (largest first)
    glaciers.sort_values("Shape_Area", ascending=False, inplace=True)

    # Take the cumulative sum of the sorted dataframe
    # This is used to extract the area fraction
    glaciers["area_cumsum"] = glaciers["Shape_Area"].cumsum()
    max_cumsum_area = glaciers["area_cumsum"].iloc[-1] * area_fraction  # This is the max allowed value for a row

    # All small glaciers are near the end, so when the max_cumsum_area is reached, only the small ones are left
    too_small_glaciers = glaciers[glaciers["area_cumsum"] > max_cumsum_area]["SGI"].values

    # Set all the distances for too small glaciers to NaN
    camera_locations.loc[:, too_small_glaciers] = np.nan

    return camera_locations


def main(cache: bool = True, max_distance: float = 2500.0, area_fraction: float = 0.8) -> None:
    # Read from the cache (if wanted and if it exists) or start from scratch
    if os.path.isfile(CACHE_FILES["camera_glacier_coverage"]) and cache:
        print("Reading cached glacier viewshed coverage")
        camera_locations = gpd.GeoDataFrame(pd.read_pickle(CACHE_FILES["camera_glacier_coverage"]))
    else:
        print("Generating new glacier viewshed coverage map")
        camera_locations = get_camera_glacier_viewshed_and_distances()

    # And only the ones that are relatively close
    print(f"Thresholding the camera distances to {max_distance} m")
    camera_locations = threshold_camera_distance(camera_locations, max_distance=max_distance)
    # Remove all photographs that (after filtering) do not portray a single glacier
    camera_locations = camera_locations[camera_locations.iloc[:, 24:].any(axis=1)]
    # camera_locations["V_TERRA_13"].to_csv("temp/swissterra_order_20200907.txt", index=False, header=None)
    return camera_locations


def get_glacier_area_camera_count_relation(cache=True):
    """
    Figure out the relationship between the X% largest glaciers (by cumulative area) and how many images are needed to cover them.

    Prints the relationship and saves it to a temporary file.
    """
    # Read the camera_locations either from the cache or from the function
    camera_locations = gpd.GeoDataFrame(pd.read_pickle(CACHE_FILES["camera_glacier_coverage"]) if cache and
                                        os.path.isfile(CACHE_FILES["camera_glacier_coverage"]) else
                                        get_camera_glacier_viewshed_and_distances())
    # Set the "far away" glacier distances to NaN
    camera_locations = threshold_camera_distance(camera_locations)

    relation = pd.Series(dtype=int)
    # Try different area fractions and record how many cameras that corresponds to.
    for fraction in tqdm(np.linspace(0.1, 1, 100)):
        # Set all camera-to-glacier distances that are NOT within the fraction to NaN
        cameras_within_fraction = extract_largest_glaciers(camera_locations.copy(), area_fraction=fraction)

        # Extract the cameras that has at least one valid glacier distance
        # The first column with distances is column with index 24
        cameras_within_fraction = cameras_within_fraction[cameras_within_fraction.iloc[:, 24:].any(axis=1)]
        # Count how many are left
        number_of_cameras = cameras_within_fraction.shape[0]

        # Put the result in the output file
        relation[fraction] = number_of_cameras

    print(relation)
    relation.to_csv("temp/relation_area_image_count.csv")


def get_select_viewsheds():
    camera_locations = main()
    viewsheds = gpd.read_file(files.INPUT_FILES["viewsheds"])
    relevant_viewsheds = viewsheds[viewsheds["IMAGE_UUID"].isin(camera_locations["IMAGE_UUID"].values)]
    relevant_viewsheds.to_file("temp/relevant_viewsheds.shp")


def get_capture_date_distribution():
    camera_locations = main()
    year_column = "V_TERRA_11"
    years = camera_locations[year_column].astype(int).values

    plt.hist(years, bins=np.max(years) - np.min(years), rwidth=0.9, color="k", zorder=2)
    plt.ylabel("Number of photographs")
    plt.xlabel("Year")
    plt.grid(zorder=3, linewidth=1)
    plt.show()
    # print(camera_locations[year_column])


def plot_camera_count_histogram():
    bins = 75

    camera_locations = image_meta.read_metadata()

    base_dem.make_hillshade(overwrite=False)
    hillshade = rio.open(base_dem.CACHE_FILES["hillshade"])
    extent = (hillshade.bounds.left, hillshade.bounds.right, hillshade.bounds.bottom, hillshade.bounds.top)

    hillshade_vals = hillshade.read(1)
    hillshade_vals = np.ma.MaskedArray(hillshade_vals, mask=hillshade_vals == 0)

    fig = plt.figure(figsize=(8, 6))
    xbins = np.linspace(camera_locations["easting"].min(), camera_locations["easting"].max(), num=bins)
    ybins = np.linspace(camera_locations["northing"].min(),
                        camera_locations["northing"].min() + (xbins.max() - xbins.min()), num=bins)
    bin_extent = (xbins.min(), xbins.max(), ybins.max(), ybins.min())
    hist, _, _ = np.histogram2d(camera_locations["easting"], camera_locations["northing"], bins=(xbins, ybins))
    hist = np.ma.MaskedArray(hist, mask=hist == 0)

    x_indices = np.digitize(camera_locations["easting"], xbins)
    y_indices = np.digitize(camera_locations["northing"], ybins)
    camera_locations["indices"] = y_indices + (x_indices * bins)

    ranges = camera_locations.groupby("indices").agg(lambda df: (df.date.max() - df.date.min()).days / 365).iloc[:, 0]\
        .reindex(np.arange(0, bins ** 2)).fillna(0)\
        .values.reshape((75, 75))
    ranges = np.ma.MaskedArray(ranges, mask=ranges == 0)

    plt.subplot(211)
    plt.imshow(hillshade_vals, cmap="Greys_r", extent=extent)
    plt.imshow(ranges.T, extent=bin_extent, cmap="coolwarm", alpha=0.7)
    plt.ylim(camera_locations["northing"].min(), camera_locations["northing"].max())

    plt.xlim(camera_locations["easting"].min(), camera_locations["easting"].max())
    cbar = plt.colorbar()
    cbar.set_label("Year range")

    plt.subplot(212)
    plt.imshow(hillshade_vals, cmap="Greys_r", extent=extent)
    plt.imshow(hist.T, extent=bin_extent, alpha=0.7)
    cbar = plt.colorbar()
    cbar.set_label("Image count")

    plt.ylim(camera_locations["northing"].min(), camera_locations["northing"].max())
    plt.xlim(camera_locations["easting"].min(), camera_locations["easting"].max())
    plt.xlabel("Easting (m)")
    plt.text(0.012, 0.5, "Northing (m)", transform=fig.transFigure, ha="center", va="center", rotation=90)

    plt.subplots_adjust(left=0.05, bottom=0.075, right=1.06, top=0.99, hspace=0.09)

    plt.savefig(os.path.join(files.FIGURE_DIRECTORY, "camera_count_histogram.jpg"), dpi=300)


if __name__ == "__main__":
    plot_camera_count_histogram()
