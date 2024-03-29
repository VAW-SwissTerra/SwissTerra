import os
from collections import namedtuple

import numpy as np
import pandas as pd

from terra import files
from terra.preprocessing import fiducials, georeferencing, image_meta, masks

TEMP_DIRECTORY = os.path.join(files.TEMP_DIRECTORY, "processing")
CACHE_FILES = {
    "log_filepath": os.path.join(TEMP_DIRECTORY, "progress.log"),
}


def get_dataset_names() -> list[str]:
    """Return a list of dataset names, sorted by median easting coordinate (left to right)."""
    image_metadata = image_meta.read_metadata()
    image_metadata["year"] = image_metadata["date"].apply(lambda date: date.year)
    unique_instruments = np.unique(image_metadata["Instrument"])

    datasets: list[str] = []
    dataset_sorting: list[float] = []
    for instrument in unique_instruments:
        unique_years = np.unique(image_metadata.loc[image_metadata["Instrument"] == instrument, "year"])
        for year in unique_years:
            dataset_images = image_metadata.loc[
                (image_metadata["Instrument"] == instrument) & (image_metadata["year"] == year)]
            dataset_sorting.append(dataset_images["easting"].median())
            datasets.append(f"{instrument}_{year}")

    sorted_datasets = [dataset for _, dataset in sorted(zip(dataset_sorting, datasets))]

    return sorted_datasets


DATASETS = get_dataset_names() + ["full", "failed_pairs"]

for _dataset in DATASETS:
    CACHE_FILES[f"{_dataset}_dir"] = os.path.join(TEMP_DIRECTORY, _dataset)
    CACHE_FILES[f"{_dataset}_input_dir"] = os.path.join(CACHE_FILES[f"{_dataset}_dir"], "input")
    CACHE_FILES[f"{_dataset}_camera_orientations"] = os.path.join(
        CACHE_FILES[f"{_dataset}_input_dir"], "camera_orientations.csv"
    )
    CACHE_FILES[f"{_dataset}_temp_dir"] = os.path.join(CACHE_FILES[f"{_dataset}_dir"], "temp")


def get_dataset_filenames(dataset: str) -> np.ndarray:
    """
    Get the image filenames of images that correspond to the dataset.

    param: dataset: The name of the dataset.

    return: dataset_filenames: The filenames corresponding to the dataset.
    """
    instrument = dataset.split("_")[0]
    year = int(dataset.split("_")[1])

    all_metadata = image_meta.read_metadata()
    all_metadata["year"] = all_metadata["date"].apply(lambda date: date.year)
    dataset_meta = all_metadata[(all_metadata["Instrument"] == instrument) & (all_metadata["year"] == year)]
    filenames = dataset_meta["Image file"].values

    return filenames
    dataset_meta = files.read_dataset_meta(dataset)

    dataset_filenames = image_meta.get_cameras_from_bounds(
        left=float(dataset_meta["bounds"]["left"]),
        right=float(dataset_meta["bounds"]["right"]),
        top=float(dataset_meta["bounds"]["top"]),
        bottom=float(dataset_meta["bounds"]["bottom"]),
    )

    return dataset_filenames


def get_dataset_metadata(dataset: str) -> pd.DataFrame:
    """
    Extract the image metadata for one dataset.

    Returns all metadata if the dataset is "full"

    return: all_data / subset_data: Metadata for each image.
    """
    all_data = image_meta.read_metadata()

    if dataset == "full":
        return all_data

    dataset_filenames = get_dataset_filenames(dataset)
    subset_data = all_data[np.isin(all_data["Image file"].values, dataset_filenames)]
    return subset_data


def export_camera_orientation_csv(dataset: str):
    #image_metadata = get_dataset_metadata(dataset)
    filenames = get_dataset_filenames(dataset)
    all_image_meta = georeferencing.generate_corrected_metadata()
    image_metadata = all_image_meta[all_image_meta["Image file"].isin(filenames)].copy()

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

    transforms = fiducials.get_all_instrument_transforms()

    try:
        image_metadata = get_dataset_metadata(dataset)
        meta_exists = True
    except FileNotFoundError:
        meta_exists = False
        missing_inputs.append(MissingInput(image_meta.CACHE_FILES["image_meta"], "image_meta_file"))

    for filepath in image_filepaths:
        if not os.path.isfile(filepath):
            missing_inputs.append(MissingInput(filepath, "image"))

        if not os.path.basename(filepath) in transforms.keys():
            missing_inputs.append(MissingInput(os.path.basename(filepath), "transform"))

        mask_name = os.path.join(masks.CACHE_FILES["mask_dir"], os.path.basename(filepath))

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


def generate_inputs(dataset: str):
    """
    Generate all necessary inputs for a dataset.

    param: dataset: The name of the dataset to check.
    """
    files.check_data()

    def big_print(string: str):
        print("\n================================")
        print(string)
        print("================================\n")

    big_print("Collecting metadata")
    image_meta.collect_metadata()

    big_print("Finding fiducial locations")
    raise NotImplementedError("Frame matching works in a different way now")
    frame_matcher = preprocessing.fiducials.FrameMatcher()
    frame_matcher.filenames = [filename for filename in get_dataset_filenames(dataset)
                               if filename != frame_matcher.orb_reference_filename]
    frame_matcher.train()
    frame_matcher.estimate()

    big_print("Generating masks")
    preprocessing.masks.generate_masks()

    big_print("Finished. The processing pipeline is ready")


if __name__ == "__main__":
    print(get_dataset_filenames(""))
