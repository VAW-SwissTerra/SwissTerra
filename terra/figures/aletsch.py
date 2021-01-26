from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Union

import cv2
import earthpy.spatial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio

from terra import base_dem, evaluation, files
from terra.constants import CONSTANTS


def crop_geotiff(filepath: str, output_filepath: str, bounds: dict[str, int], resolution: float) -> np.ndarray:
    gdal_commands = [
        "gdal_translate",
        "-projwin",
        bounds["west"],
        bounds["north"],
        bounds["east"],
        bounds["south"],
        "-tr",
        resolution,
        resolution,
        "-r",
        "cubicspline",
        filepath,
        output_filepath
    ]
    subprocess.run(list(map(str, gdal_commands)), check=True, stdout=subprocess.PIPE)

    with rio.open(output_filepath) as raster:
        return raster.read(1)


def plot_aletsch():

    bounds = {"west": 643000, "north": 141000}
    bounds["east"] = bounds["west"] + 5000
    bounds["south"] = bounds["north"] - 5000

    temp_dir = tempfile.TemporaryDirectory()
    base_dem_path = os.path.join(temp_dir.name, "base_dem.tif")
    ddem_path = os.path.join(temp_dir.name, "ddem.tif")

    base_dem_values = crop_geotiff(base_dem.CACHE_FILES["base_dem"], base_dem_path,
                                   bounds, resolution=CONSTANTS.dem_resolution)
    base_dem_dataset = rio.open(base_dem_path)
    hillshade = earthpy.spatial.hillshade(base_dem_values) / 255

    ddem = crop_geotiff(evaluation.CACHE_FILES["merged_ddem"], ddem_path,
                        bounds=bounds, resolution=CONSTANTS.dem_resolution)
    ddem_dataset = rio.open(ddem_path)

    min_value = -250
    max_value = -min_value
    ddem_norm = np.clip((ddem - min_value) / (max_value - min_value), 0, 1)
    ddem_norm = np.ma.masked_array(ddem_norm, mask=np.isnan(ddem_norm))

    cmap = plt.get_cmap("coolwarm_r")
    ddem_color = cmap(ddem_norm)

    # plt.imshow(hillshade, cmap="Greys_r")
    # plt.imshow(ddem_color * np.clip(hillshade + 0.2, 0, 1), vmin=0, vmax=1)
    hillshade_brightness = 0.4
    hillshade_mult = np.clip(hillshade_brightness + np.repeat(hillshade,
                                                              3).reshape((ddem_color.shape[0], ddem_color.shape[1], 3)), 0, 1)
    hillshade_mult = np.append(hillshade_mult, np.ones(hillshade_mult.shape[:2] + (1,)), axis=2)

    line = pd.DataFrame(columns=["easting", "northing"], dtype=float)
    line.loc[0] = bounds["west"] + 1800, bounds["north"] - 1600
    line.loc[1] = bounds["east"] - 2400, bounds["south"] + 2100
    line.index = [0, np.linalg.norm(line.loc[1] - line.loc[0])]
    new_indices = np.arange(line.index[0], line.index[-1], step=5)
    line = line.reindex(np.unique(np.r_[line.index, new_indices])).interpolate().reindex(new_indices)

    line["height_2018"] = np.fromiter(base_dem_dataset.sample(line[["easting", "northing"]].values), float)
    line["ddem"] = np.fromiter(ddem_dataset.sample(line[["easting", "northing"]].values), float)
    line["height_old"] = line["height_2018"] - np.clip(line["ddem"], -1e3, 0)

    fig = plt.figure(figsize=(8, 3))
    plt.subplot2grid((1, 3), (0, 0))
    extent = (bounds["west"], bounds["east"], bounds["south"], bounds["north"])
    plt.imshow(hillshade_mult, extent=extent, zorder=1)
    plt.imshow(ddem_color * hillshade_mult, vmin=0, vmax=1, extent=extent, zorder=2)
    plt.plot(line["easting"], line["northing"], zorder=3, color="black", linewidth=2)
    plt.annotate("A", (line.iloc[0][["easting", "northing"]].values), ha="center", fontsize=12)
    plt.annotate("A'", (line.iloc[-1][["easting", "northing"]].values), va="top", fontsize=12)
    plt.ylim(bounds["south"], bounds["north"])
    plt.xlim(bounds["west"], bounds["east"])
    # plt.xticks([])
    # plt.yticks([])
    plt.yticks(np.array(plt.gca().get_yticks())[[0, 4]], rotation=90, va="center")
    plt.xticks(np.array(plt.gca().get_xticks())[[1, 2]])
    plt.text(0.01, 0.5, "Northing (m)", transform=fig.transFigure, ha="center", va="center", rotation=90)
    plt.xlabel("Easting (m)")

    plt.subplot2grid((1, 3), (0, 1), colspan=2)
    plt.fill_between(line.index, line["height_old"], line["height_2018"].min(), color="skyblue", label="Glacier (1927)")
    plt.fill_between(line.index, line["height_2018"], line["height_2018"].min() -
                     50, color="#5F513D", label="Ground (2018)")
    plt.xlim(0, np.max(line.index))
    plt.ylim(line["height_2018"].min() - 50, line["height_2018"].max())
    plt.gca().yaxis.tick_right()
    plt.xlabel("Distance (m)")
    plt.text(0.99, 0.5, "Elevation (m a.s.l.)", transform=fig.transFigure, ha="center", va="center", rotation=-90)
    plt.text(0.005, 0.93, "A", transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.995, 0.93, "A'", transform=plt.gca().transAxes, fontsize=12, ha="right")
    plt.legend(loc="lower right")

    thickest_part = line.index[np.argwhere(line["ddem"].values == line["ddem"].min())[0]][0]
    plt.plot(
        [thickest_part, thickest_part],
        [line.loc[thickest_part, "height_old"], line.loc[thickest_part, "height_2018"]],
        linestyle="--",
        color="darkslategray",
        linewidth=1)
    plt.annotate(f"{line.loc[thickest_part, 'ddem']:.0f} m",
                 xy=(thickest_part + 7, line.loc[thickest_part, "height_old"] - 100), fontsize=8)

    cbar_ax = fig.add_axes([0.125, 0.28, 0.2, 0.2])
    cbar_ax.axis("off")
    cbar = plt.colorbar(
        mappable=plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin=min_value, vmax=max_value),
            cmap="coolwarm_r"
        ),
        ax=cbar_ax,
        location="bottom",
        aspect=10,
        fraction=0.2)
    cbar.set_ticks([min_value, 0, max_value])
    cbar.set_ticklabels([f"{tick} m" for tick in cbar.get_ticks()])

    plt.subplots_adjust(left=0.05, bottom=0.18, right=0.92, top=0.94, wspace=0.01, hspace=0.01)

    plt.savefig(os.path.join(files.FIGURE_DIRECTORY, "aletsch_example.jpg"), dpi=300)


if __name__ == "__main__":
    plot_aletsch()
