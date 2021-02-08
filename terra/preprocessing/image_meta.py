"""Functions to handle and collect image metadata files."""
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from terra import files

CACHE_FILES = {
    "image_meta": os.path.join(files.TEMP_DIRECTORY, "metadata/image_meta.pkl")
}


def collect_metadata(use_cached=True) -> pd.DataFrame:
    """
    Collect the image metadata into one DataFrame.

    :param use_cached: Use a cached version if available.
    :returns: The merged DataFrame.
    """
    if os.path.isfile(CACHE_FILES["image_meta"]) and use_cached:
        metadata = pd.read_pickle(CACHE_FILES["image_meta"])
        print(f"Loaded cached image metadata with {metadata.shape[0]} entries.")
        return metadata

    metadata = pd.DataFrame()
    print("Reading image metadata from files")
    for filename in tqdm(list(files.list_image_meta_paths())):
        meta = pd.read_csv(filename, engine="python", skiprows=11, sep=":   ",
                           index_col=0, squeeze=True).dropna()

        meta = meta.str.strip()
        # Yaw is defined as the clockwise pointing direction from north (0-360 degrees)
        meta["yaw"] = (float(meta["Viewing direction"].replace(" deg", "")) +
                       float(meta["Panning"].replace(" deg", ""))) % 360
        # Pitch is defined as the direction off nadir (down = 0 degrees, horizontal = 90 degrees)
        meta["pitch"] = 90 + float(meta["Tilting"].replace(" deg", ""))
        # No roll information is given, so this is assumed to be zero
        meta["roll"] = 0

        easting, northing, altitude = [float(string.replace(" m", ""))
                                       for string in meta["Camera position"].split("|")]
        meta["easting"] = easting
        meta["northing"] = northing
        meta["altitude"] = altitude

        meta["date"] = pd.to_datetime(meta["Acquisition date"].strip(), format="%d.%m.%Y").date()

        meta["focal_length"] = float(meta["Focal length"].replace(" mm", ""))

        meta["station_name"] = f"station_{meta['Base number']}_{meta['Position']}"

        metadata.loc[int(meta["Inventory number"]), meta.index] = meta

    # Fix dtypes
    for col in metadata:
        if col == "date":
            metadata["date"] = metadata["date"].astype(np.datetime64)
            continue
        try:
            metadata[col] = pd.to_numeric(metadata[col])
        except (ValueError, TypeError):
            metadata[col] = metadata[col].astype(pd.StringDtype())
            continue

    os.makedirs(os.path.dirname(CACHE_FILES["image_meta"]), exist_ok=True)
    metadata.to_pickle(CACHE_FILES["image_meta"])

    return metadata


def read_metadata() -> pd.DataFrame:
    """
    Read the already processed metadata file.

    return: metadata: The metadata for all images.
    """
    if not os.path.isfile(CACHE_FILES["image_meta"]):
        print("Cached metadata collection could not be found. Generating new collection.")
        return collect_metadata()

    metadata = pd.read_pickle(CACHE_FILES["image_meta"])

    return metadata


def get_cameras_from_bounds(left: float, right: float, top: float, bottom: float) -> np.ndarray:
    """
    Extract every camera (image filename) that can be found in the specified bounds.

    :param left: The left/west bounding coordinate.
    :param right: The right/east bounding coordinate.
    :param top: The top/north bounding coordinate.
    :param bottom: The bottom/south bounding coordinate.
    """
    image_metadata = read_metadata()

    fullfilling_left = image_metadata["easting"] > left
    fullfilling_right = image_metadata["easting"] < right
    fullfilling_top = image_metadata["northing"] < top
    fullfilling_bottom = image_metadata["northing"] > bottom

    meta_within_bounds = image_metadata[fullfilling_left & fullfilling_right & fullfilling_top & fullfilling_bottom]
    cameras_within_bounds = meta_within_bounds["Image file"].values

    return cameras_within_bounds


def get_filenames_for_instrument(instrument: str) -> np.ndarray:
    """Return all camera filenames with a specific instrument."""
    image_metadata = read_metadata()
    filenames = image_metadata[image_metadata["Instrument"] == instrument]["Image file"].values

    return filenames
