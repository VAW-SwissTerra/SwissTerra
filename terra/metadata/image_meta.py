import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from terra import files

CACHE_FILES = {
    "image_meta": os.path.join(files.TEMP_DIRECTORY, "metadata/image_meta.pkl")
}


def collect_metadata(use_cached=True) -> pd.DataFrame:

    if os.path.isfile(CACHE_FILES["image_meta"]) and use_cached:
        metadata = pd.read_pickle(CACHE_FILES["image_meta"])
        print(f"Loaded cached image metadata with {metadata.shape[0]} entries.")
        return metadata

    metadata = pd.DataFrame()
    print("Reading image metadata from files")
    for i, filename in enumerate(tqdm(list(files.list_image_meta_paths()))):
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

    metadata = pd.read_pickle(CACHE_FILES["image_meta"])

    return metadata


def get_matching_candidates(angle_threshold: float = 45.0, distance_threshold: float = 1500):
    """
    Find images that are relatively close and look the same direction, but are not in the same base (stereo-pair).

    This may be the ugliest code I've ever written! But it works, and it seems to be efficient.

    param: angle_threshold: The maximum allowed angle difference in degrees.
    param: distance_threshold: The maximum allowed distance between the two images in metres.

    return: pairs: Which image pairs may get matches between them.
    """

    image_meta = read_metadata()

    # Generate the vectorized functions to filter pairs from
    @np.vectorize
    def angle_diff(angle_a: float, angle_b: float) -> float:
        """
        Calculate an unsigned angle difference.

        param: angle_a: The first angle
        param: angle_b: The second angle

        return: diff: The unsigned angle difference
        """
        diff = angle_a - angle_b
        return abs((diff + 180) % 360 - 180)

    @np.vectorize
    def get_distance(xyz_a: str, xyz_b: str) -> float:
        """
        Get the distance between two position strings ("easting,northing,altitude").

        param: xyz_a: The first position string.
        param: xyz_b: The second position string.

        return: diff: The distance in meters.
        """
        position_a = np.array(xyz_a.split(",")).astype(float)
        position_b = np.array(xyz_b.split(",")).astype(float)
        diff = np.linalg.norm(position_a - position_b)
        return diff

    @np.vectorize
    def has_different_base(base_a: int, base_b: int) -> bool:
        """
        Check whether two base (stereo-pair) numbers are the different.

        param: station_a: The first base
        param: station_b: The second base 

        return: has_different_station: Whether the two bases are different.

        """
        return base_a != base_b

    #
    # Check the angle criterion
    #
    # Create a meshgrid of yaw (viewing direction) angles
    xx_grid, yy_grid = np.meshgrid(image_meta["yaw"], image_meta["yaw"])
    # Get the indices of the pairs that fill the angle threshold
    fulfilling_angle = np.unique(np.sort(np.argwhere(angle_diff(xx_grid, yy_grid) < angle_threshold), axis=1), axis=0)

    #
    # Check the distance criterion
    #
    # Extract the values (as a 2D array)
    xyzs = image_meta[["easting", "northing", "altitude"]].values
    # Convert each image's coordinates into a "easting,northing,altitude" string
    # This is because I can't find out how to get the vectorized function to take an array!
    xyz_strings = np.apply_along_axis(lambda x: ",".join(x.astype(str)), axis=1, arr=xyzs)
    # Create a meshgrid of position strings
    xx_grid, yy_grid = np.meshgrid(xyz_strings, xyz_strings)
    # Get the indices of the pairs that fill the distance threshold
    fulfilling_distance = np.unique(
        np.sort(np.argwhere(get_distance(xx_grid, yy_grid) < distance_threshold), axis=1), axis=0)

    #
    # Check that the images are not in the same base (stereo-pair)
    #
    # Create a meshgrid of base numbers
    xx_grid, yy_grid = np.meshgrid(image_meta["Base number"], image_meta["Base number"])
    # Get the indices of the pairs that fill the not-same-base threshold
    fulfilling_base = np.unique(np.sort(np.argwhere(has_different_base(xx_grid, yy_grid)), axis=1), axis=0)

    #
    # Merge the criterion results
    #
    # Merge angles and distances
    fulfilling_angle_distance = fulfilling_angle[(fulfilling_angle[:, None] == fulfilling_distance).all(-1).any(-1)]
    # Merge angles+distances with base numbers
    fulfilling_all = fulfilling_angle_distance[(
        fulfilling_angle_distance[:, None] == fulfilling_base).all(-1).any(-1)]

    @np.vectorize
    def get_filename(index: int) -> str:
        """
        Get the corresponding filename of a positional index.

        param: index: The locational index in the image_meta dataframe.

        return: filename: The filename of the image.
        """
        return image_meta.iloc[index].loc["Image file"]

    # Extract the filenames of the pairs
    pairs = get_filename(fulfilling_all)

    return pairs


if __name__ == "__main__":
    get_matching_candidates()
