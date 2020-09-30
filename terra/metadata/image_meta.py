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
