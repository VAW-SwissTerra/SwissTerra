"""Analyse and plot the distribution of different instruments in the dataset."""
from __future__ import annotations

import datetime
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from terra import files, preprocessing
from terra.constants import CONSTANTS
from terra.preprocessing import image_meta, manual_picking

CACHE_FILES = {
    "instrument_year_histogram": os.path.join(files.FIGURE_DIRECTORY, "instrument_year_histogram.jpg")
}


def get_instrument_names() -> np.ndarray:
    """Return the names of each instrument in the metadata cache."""
    image_meta = preprocessing.image_meta.read_metadata()

    return np.unique(image_meta["Instrument"])


def plot_instrument_years() -> None:
    """Plot the temporal distribution of used instruments."""
    image_meta = preprocessing.image_meta.read_metadata()

    fig = plt.figure(figsize=(9, 5), dpi=80)
    fig.canvas.set_window_title('Instrument distribution')
    axis = plt.gca()

    instrument_labels = list(np.unique(image_meta["Instrument"]))
    instrument_labels.sort()
    indices = {instrument: index for index, instrument in enumerate(instrument_labels)}
    image_meta["instrument_index"] = image_meta["Instrument"].apply(lambda x: indices[x])
    image_meta["date_s"] = image_meta["date"].astype(int)
    camera_order = image_meta.groupby("instrument_index").median()["date_s"].sort_values()

    for i, instrument in enumerate(instrument_labels[index] for index in camera_order.index):
        cameras = image_meta.loc[image_meta["Instrument"] == instrument]

        axis.text(x=i, y=cameras["date"].max().year, s=cameras.shape[0], ha="center", va="bottom")
        axis.violinplot(dataset=cameras["date"].apply(
            lambda x: x.year), positions=[i], widths=0.8, showmedians=True)

    axis.set_xticks(camera_order.index)
    axis.set_xticklabels([plt.Text(x=position, text=instrument_labels[position])
                          for position in camera_order.index], rotation=45)
    plt.ylabel("Year")
    plt.xlabel("Instrument label")
    plt.grid(alpha=0.5, zorder=0)
    plt.tight_layout()

    dirname = os.path.join(files.TEMP_DIRECTORY, "figures")
    os.makedirs(dirname, exist_ok=True)
    plt.savefig(os.path.join(dirname, "instrument_years.png"), dpi=300)
    plt.show()


def plot_frame_type_distribution():

    marked_fiducials = manual_picking.MarkedFiducials.create_or_load(files.INPUT_FILES["marked_fiducials"])
    unique_frame_types = marked_fiducials.get_used_frame_types()
    image_metadata = image_meta.read_metadata()

    marked_metadata = pd.DataFrame(columns=list(image_metadata.columns) + ["frame_type"])

    for frame_type in unique_frame_types:
        filenames = marked_fiducials.get_filenames_for_frame_type(frame_type)
        meta_with_frame = image_metadata[image_metadata["Image file"].isin(filenames)].copy()
        meta_with_frame["frame_type"] = frame_type
        marked_metadata = marked_metadata.append(meta_with_frame)

    for frame_type in np.unique(marked_metadata["frame_type"]):
        frame_data = marked_metadata[marked_metadata["frame_type"] == frame_type]

        print(frame_type)
        for i, df in frame_data.groupby("Instrument"):
            count = df['Instrument'].count()
            print(f"\t{i}: {count}")
            if count < 10:
                for j, row in df.iterrows():
                    print("\t\t" + row["Image file"])

    return
    plt.figure(figsize=(8, 5))

    for frame_type, data in marked_metadata.groupby("frame_type"):
        plt.scatter(data["easting"], data["northing"], label=frame_type, alpha=0.7)

    plt.legend()

    plt.savefig("temp/figures/wild_frame_distribution.jpg", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    row1 = plt.subplot(2, 1, 1)
    row2 = plt.subplot(2, 1, 2)

    labels: list[plt.Text] = []
    for i, (frame_type, data) in enumerate(marked_metadata.groupby("frame_type")):
        row1.violinplot(dataset=data["date"].apply(lambda row: row.year), positions=[i])
        row2.violinplot(dataset=data["date"].apply(lambda row: row.month), positions=[i])
        labels.append(plt.Text(x=i, text=frame_type))

    for row in [row1, row2]:
        row.set_xticks(range(i + 1))  # pylint: disable=undefined-loop-variable
        row.set_xticklabels(labels)

    row1.set_ylabel("Year")
    row2.set_ylabel("Month")

    plt.savefig("temp/figures/wild_frame_temporal_distribution.jpg", dpi=300)

    plt.show()


def show_different_frame_types():

    marked_fiducials = manual_picking.MarkedFiducials.create_or_load(files.INPUT_FILES["marked_fiducials"])
    unique_frames = marked_fiducials.get_used_frame_types()

    window_radius = 250
    coords = CONSTANTS.wild_fiducial_locations["right"]

    filenames = {
        "blocky": "000-381-215.tif",
        "triangular": "000-387-154.tif",
        "rhone": "000-379-619.tif"

    }

    plt.figure(figsize=(8, 3.5))

    for i, frame_type in enumerate(unique_frames):

        plt.subplot(1, 3, i + 1)

        image = cv2.imread(os.path.join(
            files.INPUT_DIRECTORIES["image_dir"], filenames[frame_type]), cv2.IMREAD_GRAYSCALE)
        fiducial = image[coords[0] - window_radius: coords[0] + window_radius,
                         coords[1] - window_radius: coords[1] + window_radius]

        plt.axis("off")
        plt.title(frame_type)
        plt.imshow(fiducial, cmap="Greys_r")

    plt.tight_layout()
    os.makedirs("temp/figures/", exist_ok=True)
    plt.savefig("temp/figures/wild_frame_types.jpg", dpi=300)
    plt.show()
    print(unique_frames)


def plot_year_distribution():
    image_metadata = image_meta.read_metadata()
    plt.figure(figsize=(8, 5))
    plt.hist(image_metadata["date"], bins=30, color="black")

    median_date = datetime.datetime.utcfromtimestamp(np.median(image_metadata["date"].astype(np.int64)) / 1e9)

    plt.vlines(median_date, *plt.gca().get_ylim(), color="red")
    plt.annotate(f"{median_date.year}", (median_date, plt.gca().get_ylim()[1] * 0.97), ha="center")
    plt.xlabel("Capture date")
    plt.tight_layout()
    plt.savefig(CACHE_FILES["instrument_year_histogram"], dpi=300)


if __name__ == "__main__":
    # plot_instrument_years()
    plot_frame_type_distribution()
    #image_metadata = image_meta.read_metadata()
    # for filename in image_metadata[image_metadata["Instrument"].isin(["Wild4"])]["Image file"]:
    #    print(filename)
