"""Stable-ground / glacier outlines and mask functions."""
from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import warnings

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import shapely
import shapely.geometry
import shapely.ops
import skimage.graph
from tqdm import tqdm

from terra import base_dem, dem_tools, files, utilities
from terra.constants import CONSTANTS
from terra.preprocessing import image_meta

TEMP_DIRECTORY = os.path.join(files.TEMP_DIRECTORY, "preprocessing/")
CACHE_FILES = {
    "glacier_mask": os.path.join(TEMP_DIRECTORY, "glacier_mask.tif"),
    "stable_ground_mask": os.path.join(TEMP_DIRECTORY, "stable_ground_mask.tif"),
    "lk50_outlines": os.path.join(TEMP_DIRECTORY, "lk50_outlines.shp"),
    "lk50_centrelines": os.path.join(TEMP_DIRECTORY, "lk50_centrelines.shp"),
}


def rasterize_outlines(input_filepath: str, output_filepath: str, overwrite: bool = False,
                       resolution: float = CONSTANTS.dem_resolution) -> None:
    """Generate a boolean glacier mask from the 1935 map."""
    # Skip if it already exists
    if not overwrite and os.path.isfile(output_filepath):
        return
    # Get the bounds from the reference DEM
    reference_bounds = json.loads(
        subprocess.run(
            ["gdalinfo", "-json", base_dem.CACHE_FILES["base_dem"]],
            check=True,
            stdout=subprocess.PIPE
        ).stdout
    )["cornerCoordinates"]

    # Generate the mask
    gdal_commands = [
        "gdal_rasterize",
        "-burn", 1,  # Glaciers get a value of 1
        "-a_nodata", 0,  # Non-glaciers get a value of 0
        "-ot", "Byte",
        "-tr", resolution, resolution,
        "-te", *reference_bounds["lowerLeft"], *reference_bounds["upperRight"],
        input_filepath,
        output_filepath
    ]
    subprocess.run(list(map(str, gdal_commands)), check=True, stdout=subprocess.PIPE)

    # Unset the nodata value to correctly display in e.g. QGIS
    subprocess.run(["gdal_edit.py", "-unsetnodata", output_filepath],
                   check=True, stdout=subprocess.PIPE)


def generate_glacier_mask(overwrite: bool = False, resolution: float = CONSTANTS.dem_resolution):
    rasterize_outlines(
        input_filepath=files.INPUT_FILES["outlines_1935"],
        output_filepath=CACHE_FILES["glacier_mask"],
        overwrite=overwrite,
        resolution=resolution)


def generate_stable_ground_mask(overwrite: bool = False, resolution: float = CONSTANTS.dem_resolution):

    if not overwrite and os.path.isfile(CACHE_FILES["stable_ground_mask"]):
        return
    # Make/read the glacier mask
    generate_glacier_mask(overwrite=overwrite, resolution=resolution)
    glacier_mask = rio.open(CACHE_FILES["glacier_mask"])

    land_use = gpd.read_file(files.INPUT_FILES["lake_outlines"]).to_crs(CONSTANTS.crs_epsg.replace("::", ":"))
    lakes = land_use.copy()  # [land_use["CODE_18"] == "512"]

    temp_dir = tempfile.TemporaryDirectory()
    lakes_temp_path = os.path.join(temp_dir.name, "lakes.shp")
    rasterized_lakes_temp_path = os.path.join(temp_dir.name, "lakes_rasterized.tif")

    lakes.to_file(lakes_temp_path)

    rasterize_outlines(lakes_temp_path, rasterized_lakes_temp_path, resolution=resolution)

    lake_mask = rio.open(rasterized_lakes_temp_path)

    # Get the area where neither lakes nor glaciers exist
    stable_ground_mask = (lake_mask.read(1) != 1) & (glacier_mask.read(1) != 1)

    with rio.open(
            CACHE_FILES["stable_ground_mask"],
            mode="w",
            driver="GTiff",
            width=glacier_mask.width,
            height=glacier_mask.height,
            count=1,
            crs=glacier_mask.crs,
            transform=glacier_mask.transform,
            dtype=np.uint8) as raster:
        raster.write(stable_ground_mask.astype(np.uint8), 1)


def read_mask(filepath: str, bounds: dict[str, float], resolution: float = CONSTANTS.dem_resolution,
              crs: rio.crs.CRS = None) -> np.ndarray:
    """
    Read a mask and crop/resample it to the given bounds.

    Uses nearest neighbour if the target resolution is the same, bilinear if larger, and cubic spline if lower.

    :param filepath: The path to the mask file.
    :param bounds: A dictionary with the keys: west, east, south, north.
    :param resolution: The target resolution of the mask file.
    :param crs: An optional other CRS. Defaults to the mask CRS.
    :returns: A boolean numpy array with a shape corresponding to the given bounds and resolution.
    """
    mask = rio.open(filepath)

    if abs(resolution - mask.res[0]) < 1e-2:
        resampling_method = rio.warp.Resampling.nearest
    elif resolution > mask.res[0]:
        resampling_method = rio.warp.Resampling.bilinear
    elif resolution < mask.res[0]:
        resampling_method = rio.warp.Resampling.cubic_spline

    # Calculate new shape of the dataset
    dst_shape = (int((bounds["north"] - bounds["south"]) // resolution),
                 int((bounds["east"] - bounds["west"]) // resolution))

    # Make an Affine transform from the bounds and the new size
    dst_transform = rio.transform.from_bounds(**bounds, width=dst_shape[1], height=dst_shape[0])
    # Make an empty numpy array which will later be filled with elevation values
    resampled_values = np.empty(dst_shape, np.float32)
    # Set all values to nan right now
    resampled_values[:, :] = np.nan

    # Reproject the DEM and put the output in the destination array
    rio.warp.reproject(
        source=mask.read(1),
        destination=resampled_values,
        src_transform=mask.transform,
        dst_transform=dst_transform,
        resampling=resampling_method,
        src_crs=mask.crs,
        dst_crs=mask.crs if crs is None else crs
    )

    # Convert the float(?) array to a boolean.
    return resampled_values == 1


def read_glacier_mask(bounds: dict[str, float], resolution: float = CONSTANTS.dem_resolution) -> np.ndarray:
    """
    Read and crop/resample the glacier mask to the given bounds and resolution.

    :param bounds: A dictionary with the keys: west, east, south, north.
    :param resolution: The target resolution of the mask.
    :returns: A boolean numpy array with a shape corresponding to the given bounds and resolution.
    """
    return read_mask(CACHE_FILES["glacier_mask"], bounds=bounds, resolution=resolution)


def read_stable_ground_mask(bounds: dict[str, float], resolution: float = CONSTANTS.dem_resolution) -> np.ndarray:
    """
    Read and crop/resample the stable ground mask to the given bounds and resolution.

    The array is True where there is stable ground, and False where it is unstable.

    :param bounds: A dictionary with the keys: west, east, south, north.
    :param resolution: The target resolution of the mask.
    :returns: A boolean numpy array with a shape corresponding to the given bounds and resolution.
    """
    return read_mask(CACHE_FILES["stable_ground_mask"], bounds=bounds, resolution=resolution)


def fix_freudinger_outlines():
    min_overlap_threshold = 0.05
    temp_dir = os.path.join(files.TEMP_DIRECTORY, "outlines")
    freudinger = gpd.read_file(files.INPUT_FILES["outlines_1935"])
    sgi_2016 = gpd.read_file(files.INPUT_FILES["sgi_2016"]).to_crs(CONSTANTS.crs_epsg.replace("::", ":"))

    new_outlines = gpd.GeoDataFrame(columns=["EZGNR", "sgi-ids", "geometry"], crs=sgi_2016.crs)
    for i, row in tqdm(freudinger.iterrows(), total=freudinger.shape[0]):
        overlapping = sgi_2016.geometry.overlaps(row.geometry)
        if np.count_nonzero(overlapping.values) == 0:
            continue
        diff = sgi_2016[overlapping].geometry.intersection(row.geometry)

        fraction = diff.area / row.geometry.area

        mask = overlapping.copy()
        mask[mask] = fraction > min_overlap_threshold

        sorting = np.argsort(fraction[fraction > min_overlap_threshold])[::-1]

        sgi_ids = sgi_2016[mask].iloc[sorting]["sgi-id"].values
        sgi_col = ",".join(sgi_ids)

        merged_geom = shapely.ops.unary_union(np.r_[sgi_2016[mask].geometry.values, row.geometry])

        new_outlines.loc[i] = row["EZGNR"], sgi_col, merged_geom

    new_outlines.to_file(os.path.join(temp_dir, "new_outlines.shp"))
    print(new_outlines)


def is_polygon(geometry) -> bool:
    return geometry.geom_type in ["MultiPolygon", "Polygon"]


def fix_lk50_outlines():
    pd.set_option('mode.chained_assignment', "raise")

    lk50_sgi1973 = gpd.read_file(files.INPUT_FILES["lk50_modified_sgi_1973"])
    merged_sgi_lk50 = lk50_sgi1973.dissolve(by="SGI", as_index=False)

    # At this point, the SGI ids are the same as SGI1973, but in the parent-merging, these may change.
    merged_sgi_lk50["SGI1973"] = merged_sgi_lk50["SGI"]

    assert np.unique(merged_sgi_lk50["SGI"]).shape[0] == merged_sgi_lk50.shape[0]
    merged_parent_lk50 = gpd.GeoDataFrame(columns=merged_sgi_lk50.columns, crs=lk50_sgi1973.crs)
    # Merge all shapes that have the same parent and that touch.
    for parent, glaciers in tqdm(merged_sgi_lk50.groupby("Parent1850"),
                                 desc="Merging touching polygons with same parent"):
        glaciers_copy = glaciers.copy()
        glaciers_copy["Parent1850"] = parent
        if parent is None:
            merged_parent_lk50 = merged_parent_lk50.append(glaciers, ignore_index=True)
            continue

        glaciers_copy["area"] = glaciers.geometry.apply(lambda x: x.area)
        glaciers_copy.sort_values("area", ascending=False, inplace=True)
        glacier = glaciers_copy.iloc[0].copy()

        if glaciers.shape[0] > 1:
            for _, glacier2 in glaciers.iterrows():
                # Skip if self-comparison
                if glacier["SGI"] == glacier2["SGI"]:
                    continue
                # Merge geometries if they touch
                if glacier.geometry.touches(glacier2.geometry):
                    # Merge the geometry
                    glacier.geometry = glacier.geometry.union(glacier2.geometry)
                    # Append the SGI id of the smaller glacier to the larger's SGI1973 column.
                    glacier["SGI1973"] += "," + glacier2["SGI"]
                # Otherwise, add glacier2 to the output dataframe
                else:
                    merged_parent_lk50.loc[merged_parent_lk50.shape[0]] = glacier2

        # Add the biggest glacier to the dataframe
        merged_parent_lk50.loc[merged_parent_lk50.shape[0]] = glacier[merged_parent_lk50.columns]

    merged_parent_lk50.to_file(os.path.join(os.path.dirname(CACHE_FILES["lk50_outlines"]), "lk50_merged_parents.shp"))

    # Read the 1973 outlines to make sure the new geometry is consistently larger or the same as in 1973
    sgi_1973 = gpd.read_file(files.INPUT_FILES["sgi_1973"]).to_crs(lk50_sgi1973.crs)
    # Read the 1850 ouitlines to make sure the new geometry is consistently smaller or the same as in 1850
    # sgi_1850 = gpd.read_file(files.INPUT_FILES["sgi_1850"]).to_crs(lk50_sgi1973.crs)
    image_metadata = image_meta.read_metadata()

    lk50 = gpd.GeoDataFrame(columns=list(merged_parent_lk50.columns) +
                            ["mod_1973", "date"], crs=lk50_sgi1973.crs)
    for i, glacier in tqdm(merged_parent_lk50.iterrows(), total=merged_parent_lk50.shape[0],
                           desc="Validating geometries and finding proper date"):
        glacier_1973 = sgi_1973.loc[sgi_1973["OBJECTID"] == glacier["OBJECTID"]].iloc[0]

        # Check if the difference is larger than 10 m2. If so, it is assumed to have been changed from the 1973 outline.
        changed_from_1973 = abs(glacier.geometry.area - glacier_1973.geometry.area) > 10
        glacier["mod_1973"] = changed_from_1973

        centroid = glacier.geometry.centroid
        # Calculate the distance to all cameras
        image_metadata["dist"] = np.linalg.norm([image_metadata["easting"] - centroid.x,
                                                 image_metadata["northing"] - centroid.y
                                                 ], axis=0)
        # Sort the values by the distance
        image_metadata.sort_values("dist", inplace=True)
        # Take the date of the closest camera.
        glacier["date"] = str(image_metadata.iloc[0]["date"].date())

        # If it has been changed since 1973, check that it is indeed larger, and that it is smaller than in 1850.
        if changed_from_1973:
            fixed_geom = glacier.geometry.union(glacier_1973.geometry)

            if fixed_geom.geom_type not in ["MultiPolygon", "Polygon"]:
                continue

            # Measure the old (1973) and new (LK50) areas to make sure that the geometry did not turn invalid somehow.
            new_area = sum([geom.area for geom in fixed_geom]) \
                if fixed_geom.geom_type == "MultiPolygon" else fixed_geom.area
            old_area = sum([geom.area for geom in glacier_1973.geometry]) \
                if glacier_1973.geometry.geom_type == "MultiPolygon" else glacier_1973.geometry.area
            if new_area > old_area and fixed_geom.is_valid:
                glacier.geometry = fixed_geom

        lk50.loc[i] = glacier

    # There is a new modified column. The length and area columns are now old!
    lk50.drop(columns=["Shape_Leng", "Shape_Area", "modified", "OBJECTID"], inplace=True)

    lk50.to_file(CACHE_FILES["lk50_outlines"])
    print(lk50)


def generate_lk50_centrelines():
    """
    ADD THIS
    """
    lk50 = gpd.read_file(CACHE_FILES["lk50_outlines"])
    centrelines_2016 = gpd.read_file(files.INPUT_FILES["centrelines_2016"]).to_crs(lk50.crs)
    centrelines_lk50 = gpd.GeoDataFrame(columns=["SGI", "geometry"], crs=lk50.crs)
    ref_dem = rio.open(base_dem.CACHE_FILES["base_dem"])
    # full_elevation = ref_dem.read(1)

    # assert full_elevation.shape == full_glacier_mask.shape

    assert lk50.shape[0] == lk50["SGI"].unique().shape[0]

    for sgi_id in tqdm(lk50["SGI"].unique(), desc="Extending centrelines"):

        # temp_dir = tempfile.TemporaryDirectory()
        # ref_dem_path = os.path.join(temp_dir.name, "ref_dem.tif")
        # if sgi_id != "A54l-19":
        #    continue

        # The SGI2016 SGI ids are sometimes zfilled: B43-03 instead of B43-3
        try:
            first, second = sgi_id.split("-")
            sgi_2016_id = first + "-" + second.zfill(2)
        except ValueError:
            sgi_2016_id = sgi_id

        glacier_1927 = lk50.loc[lk50["SGI"] == sgi_id].iloc[0]
        glacier_centrelines_2016 = centrelines_2016.loc[centrelines_2016["sgi-id"] == sgi_2016_id].copy()
        if glacier_centrelines_2016.shape[0] == 0:
            continue

        glacier_centrelines_2016["within"] = glacier_centrelines_2016.geometry.intersects(glacier_1927.geometry)
        glacier_centrelines_2016["length"] = glacier_centrelines_2016.geometry.length
        try:
            centreline_2016 = glacier_centrelines_2016[glacier_centrelines_2016["within"]].sort_values(
                "length").iloc[-1]
        except IndexError:
            continue

        # bounds = dict(zip(["west", "south", "east", "north"], list(glacier_1927.geometry.bounds)))
        # for key in bounds:
        #    bounds[key] = bounds[key] - bounds[key] % CONSTANTS.dem_resolution

        # bounds["east"] += CONSTANTS.dem_resolution
        # bounds["north"] += CONSTANTS.dem_resolution
        # bounds["west"] += CONSTANTS.dem_resolution
        # bounds["north"] += CONSTANTS.dem_resolution

        glacier_mask, _, window = rio.mask.raster_geometry_mask(ref_dem, (glacier_1927.geometry,), crop=True)
        ref_elevation = ref_dem.read(1, window=window)
        ref_elevation = np.ma.MaskedArray(ref_elevation, mask=ref_elevation < -9999)
        cropped_1927 = ref_elevation.copy()
        cropped_1927[glacier_mask] = np.nan

        assert cropped_1927.shape == ref_elevation.shape

        start_point_coord = centreline_2016.geometry.xy[0][-1], centreline_2016.geometry.xy[1][-1]
        start_point_indices = ref_dem.index(start_point_coord[0], start_point_coord[1], precision=0)
        # Find the start point (image index) that is as close as possible to the start_point_coord
        start_point = (
            np.clip(start_point_indices[0] - window.row_off, 0, cropped_1927.shape[0] - 1),
            np.clip(start_point_indices[1] - window.col_off, 0, cropped_1927.shape[1] - 1)
        )

        end_point = np.argwhere(cropped_1927 == np.nanmin(cropped_1927))[0]
        with np.errstate(over="ignore"):
            # Set the cost to be the slope
            cost = cv2.blur(np.rad2deg(np.arctan(np.linalg.norm(np.gradient(ref_elevation), axis=0))), (5, 5))
            # Scale the cost by the normalized elevation (to penalize but not exclude upward slopes)
            cost *= cv2.blur(ref_elevation, (5, 5)) / np.nanmax(ref_elevation)
        # Set values outside the glacier to something crazy, so that the route will not take it into account.
        cost[glacier_mask] = 1e6
        try:
            path = skimage.graph.route_through_array(cost, start_point, end_point)[0]
        except ValueError as exception:
            if "no minimum-cost path" in str(exception):
                area_km2 = glacier_1927.geometry.area / 1e6
                warnings.warn(f"No path found for {sgi_id}. Area: {area_km2} km²")
                if area_km2 < 5:
                    continue
                plt.subplot(211)
                plt.imshow(cost, vmin=0, vmax=90)
                plt.scatter([start_point[1], end_point[1]], [start_point[0], end_point[1]])
                plt.subplot(212)
                plt.imshow(ref_elevation)
                plt.scatter([start_point[1], end_point[1]], [start_point[0], end_point[1]])
                plt.show()
                continue
            print(ref_elevation.shape, start_point, end_point)
            raise exception

        coordinates: list[shapely.geometry.Point] = []
        for easting, northing in zip(centreline_2016.geometry.xy[0], centreline_2016.geometry.xy[1]):
            coordinates.append(shapely.geometry.Point(easting, northing))
        for row, col in path:
            easting, northing = ref_dem.xy(row + window.row_off, col + window.col_off)
            coordinates.append(shapely.geometry.Point(easting, northing))

        lk50_centreline = shapely.geometry.LineString(coordinates)

        centrelines_lk50.loc[centrelines_lk50.shape[0]] = sgi_id, lk50_centreline

        # print(path)
        # plt.imshow(cropped_1927)
        # plt.plot([p[1] for p in path], [p[0] for p in path])
        # plt.show()

        # ref_dem.close()
        # temp_dir.cleanup()

    print("Simplifying geometry")

    print(centrelines_lk50)
    centrelines_lk50.geometry = centrelines_lk50.geometry.simplify(tolerance=50, preserve_topology=True)

    centrelines_lk50.to_file(CACHE_FILES["lk50_centrelines"])


def extrapolate_point(point_1: tuple[float, float], point_2: tuple[float, float]) -> tuple[float, float]:
    """Creates a point extrapoled in p1->p2 direction"""
    # p1 = [p1.x, p1.y]
    # p2 = [p2.x, p2.y]
    extrap_ratio = 10
    return (point_1[0]+extrap_ratio*(point_2[0]-point_1[0]), point_1[1]+extrap_ratio*(point_2[1]-point_1[1]))


def buffer_centreline():
    centrelines = gpd.read_file(CACHE_FILES["lk50_centrelines"])
    old_outlines = gpd.read_file(CACHE_FILES["lk50_outlines"])

    for sgi_id in centrelines["SGI"].values:

        if sgi_id != "B43-3":
            continue

        centreline = centrelines.loc[centrelines["SGI"] == sgi_id].iloc[-1]
        distance_threshold = centreline.geometry.length * 0.1
        old_outline = old_outlines.loc[old_outlines["Parent1850"] == sgi_id].iloc[-1]
        # ext_centreline_parts = [
        #    getExtrapoledLine(centreline.geometry.interpolate(0), centreline.geometry.interpolate(1)),
        #    centreline.geometry,
        #    getExtrapoledLine(centreline.geometry.interpolate(c_length - 1), centreline.geometry.interpolate(c_length))
        # ]
        # extended_centreline = shapely.ops.linemerge(ext_centreline_parts)
        coords = list(centreline.geometry.coords)
        coords.insert(0, extrapolate_point(coords[1], coords[0]))
        coords.insert(-1, extrapolate_point(coords[-2], coords[-1]))

        extended_centreline = shapely.geometry.LineString(coords)
        buffered_centrelines = []

        plt.subplot(131)

        for buffer in np.linspace(5, 50, num=20):
            buffered = extended_centreline.buffer(buffer).boundary

            lines = shapely.ops.split(buffered, extended_centreline)
            intersection = lines.intersection(old_outline.geometry)

            lines_inside = []
            for line in intersection:
                merged = False
                for i, line2 in enumerate(lines_inside):
                    if line2.touches(line):
                        lines_inside[i] = shapely.ops.linemerge([line, line2])
                        merged = True
                if not merged:
                    lines_inside.append(line)

            for line in lines_inside:
                first_and_last_points = np.array([
                    [line.xy[0][0], line.xy[1][0]],
                    [line.xy[0][-1], line.xy[1][-1]]
                ])
                distances = np.linalg.norm(
                    first_and_last_points - np.array([centreline.geometry.xy[0][0], centreline.geometry.xy[1][0]]),
                    axis=1)
                if np.count_nonzero(distances < distance_threshold) == 0:
                    continue

                if (line.length / centreline.geometry.length) < 0.6:
                    continue
                buffered_centrelines.append(line)

                plt.plot(*line.xy)

        plt.plot(*centreline.geometry.xy, color="black", linestyle="--")
        outline_iter = [
            old_outline.geometry.boundary] if old_outline.geometry.geom_type == "LineString" else old_outline.geometry.boundary
        for line in outline_iter:
            plt.plot(*line.xy, color="blue")
        # plt.plot(*polygon.exterior.xy)
        plt.axis("equal")

        plt.subplot(132)
        new_outlines = gpd.read_file(files.INPUT_FILES["sgi_2016"])

        # The SGI2016 SGI ids are sometimes zfilled: B43-03 instead of B43-3
        try:
            first, second = sgi_id.split("-")
            sgi_2016_id = first + "-" + second.zfill(2)
        except ValueError:
            sgi_2016_id = sgi_id
        new_outline = new_outlines.loc[new_outlines["sgi-id"] == sgi_2016_id].to_crs(old_outlines.crs).iloc[0]

        cropped_centrelines = []
        intersection = shapely.ops.linemerge(buffered_centrelines).intersection(new_outline.geometry)
        for line in intersection:
            first_and_last_points = np.array([
                [line.xy[0][0], line.xy[1][0]],
                [line.xy[0][-1], line.xy[1][-1]]
            ])
            distances = np.linalg.norm(
                first_and_last_points - np.array([centreline.geometry.xy[0][0], centreline.geometry.xy[1][0]]),
                axis=1)
            if np.count_nonzero(distances < distance_threshold) == 0:
                continue

            if (line.length / centreline.geometry.length) < 0.6:
                continue
            plt.plot(*line.xy)
            cropped_centrelines.append(line)

        outline_iter = [
            new_outline.geometry.boundary] if new_outline.geometry.geom_type == "LineString" else new_outline.geometry.boundary
        for line in outline_iter:
            assert line.geom_type == "LineString", line.geom_type
            plt.plot(*line.xy, color="blue")
        plt.plot(*centreline.geometry.xy, color="black", linestyle="--")

        plt.subplot(133)
        old_lengths = np.array([line.length for line in buffered_centrelines])
        new_lengths = np.array([line.length for line in cropped_centrelines])
        xs = [int(old_outline.date.split("-")[0]), 2016]
        plt.plot(xs, [old_lengths.mean(), new_lengths.mean()])
        plt.boxplot([old_lengths, new_lengths], positions=xs, widths=10)
        # plt.gca().set_xticklabels([plt.Text(0, text=), plt.Text(1, text="2016")])
        lengths = [line.length for line in buffered_centrelines]

        #plt.ylim(0, centreline.geometry.length * 1.05)
        plt.ylim(plt.gca().get_ylim()[0] * 0.9, plt.gca().get_ylim()[1])
        plt.xlim(xs[0] - 10, xs[1] + 10)
        print(np.mean(lengths), len(lengths), np.std(lengths))
        plt.show()


if __name__ == "__main__":

    # generate_lk50_centrelines()
    buffer_centreline()
