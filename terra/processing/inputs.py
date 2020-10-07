import os
from collections import namedtuple

import numpy as np
import pandas as pd

from terra import fiducials, files, metadata, preprocessing

DATASETS = ["rhone", "full"]


CACHE_FILES = {
}


TEMP_DIRECTORY = os.path.join(files.TEMP_DIRECTORY, "processing")

for dataset in DATASETS:
    CACHE_FILES[f"{dataset}_dir"] = os.path.join(TEMP_DIRECTORY, dataset)
    CACHE_FILES[f"{dataset}_input_dir"] = os.path.join(CACHE_FILES[f"{dataset}_dir"], "input")
    CACHE_FILES[f"{dataset}_camera_orientations"] = os.path.join(
        CACHE_FILES[f"{dataset}_input_dir"], "camera_orientations.csv")
    CACHE_FILES[f"{dataset}_temp_dir"] = os.path.join(CACHE_FILES[f"{dataset}_dir"], "temp")


def get_dataset_filenames(dataset: str):
    return open(files.INPUT_FILES[f"{dataset}_image_filenames"]).read().splitlines()


def get_dataset_metadata(dataset: str) -> pd.DataFrame:
    """
    Extract the image metadata for one dataset.

    Returns all metadata if the dataset is "full"

    return: all_data / subset_data: Metadata for each image.
    """
    all_data = metadata.image_meta.read_metadata()

    if dataset == "full":
        return all_data

    dataset_filenames = get_dataset_filenames(dataset)
    subset_data = all_data[np.isin(all_data["Image file"].values, dataset_filenames)]
    return subset_data


def export_camera_orientation_csv(dataset: str):
    image_metadata = get_dataset_metadata(dataset)

    #image_metadata["label"] = image_metadata["Image file"].str.replace(".tif", "")
    image_metadata.rename(columns={"Image file": "label"}, inplace=True)

    if not os.path.isdir(CACHE_FILES[f"{dataset}_input_dir"]):
        os.makedirs(CACHE_FILES[f"{dataset}_input_dir"], exist_ok=True)
    image_metadata[["label", "easting", "northing", "altitude", "yaw", "pitch", "roll"]].to_csv(
        CACHE_FILES[f"{dataset}_camera_orientations"], index=False)


def check_inputs(dataset: str) -> None:
    """
    Check whether all inputs for a dataset exist in the cache.

    param: dataset: The name of the dataset.
    """
    missing_inputs = []
    MissingInput = namedtuple("MissingInput", ["filepath", "filetype"])
    image_filenames = get_dataset_filenames(dataset)
    image_filepaths = [os.path.join(files.INPUT_DIRECTORIES["image_dir"], filename) for filename in image_filenames]

    frame_matcher = fiducials.fiducials.FrameMatcher(verbose=False)
    transforms = frame_matcher.load_transforms("merged_transforms.pkl")

    try:
        image_metadata = get_dataset_metadata(dataset)
        meta_exists = True
    except FileNotFoundError:
        meta_exists = False
        missing_inputs.append(MissingInput(metadata.image_meta.CACHE_FILES["image_meta"], "image_meta_file"))

    for filepath in image_filepaths:
        if not os.path.isfile(filepath):
            missing_inputs.append(MissingInput(filepath, "image"))

        if not os.path.basename(filepath) in transforms.keys():
            missing_inputs.append(MissingInput(os.path.basename(filepath), "transform"))

        mask_name = os.path.join(preprocessing.masks.CACHE_FILES["mask_dir"], os.path.basename(filepath))

        if not os.path.isfile(mask_name):
            missing_inputs.append(MissingInput(mask_name, "frame_mask"))

        if meta_exists and not os.path.basename(filepath) in image_metadata["Image file"].values:
            missing_inputs.append(MissingInput(os.path.basename(filepath), "image_meta"))

    if len(missing_inputs) == 0:
        print("Every file located. Processing pipeline is ready.")
        return

    print("\n============================")
    print(f"Missing {len(missing_inputs)} input file{'s' if len(missing_inputs) > 1 else ''}.")
    print("=============================\n")

    pretty_names = {
        "image": ("image", "Run 'terra files check'.", "multiple"),
        "image_meta": ("image metadata row", "Check the image metadata folder and rerun 'terra metadata collect-files'.", "multiple"),
        "image_meta_file": ("image metadata file", "Run 'terra metadata collect-files'", "one"),
        "transform": ("image frame transform", "Run the fiducial estimation.", "multiple"),
        "frame_mask": ("image frame mask", "Run 'terra preprocessing generate-masks'", "multiple"),
    }

    for key, (pretty_name, help_text, count) in pretty_names.items():
        missing_keys = [missing_input for missing_input in missing_inputs if missing_input.filetype == key]

        if len(missing_keys) == 0:
            continue

        if count == "one":
            print(f"Missing {pretty_name}.")

        elif count == "multiple":
            print(f"Missing {len(missing_keys)} {pretty_name}{'s' if len(missing_keys) > 1 else ''}")
            if len(missing_keys) < 10:
                print(*(f"\tMissing {missing.filepath}" for missing in missing_keys))

        print(f"Suggested fix: {help_text}\n")
