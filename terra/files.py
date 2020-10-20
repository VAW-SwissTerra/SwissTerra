"""File handling helper functions."""
import os
import pathlib
import shutil
from collections import namedtuple
from typing import Generator, List, Union

import magic
import statictypes
import toml

# Set the default input root directory
INPUT_ROOT_DIRECTORY = "input"
# Check if an overriding environment variable exists
if "SWISSTERRA_INPUT_DIR" in os.environ:
    INPUT_ROOT_DIRECTORY = os.environ["SWISSTERRA_INPUT_DIR"]

# Set the name of the temporary files directory
TEMP_DIRECTORY = "temp"

# Set the names of the input directory names (excluding the root)
# Note that more directories exist. These are the directories where all filetypes are assumed to be the same.
INPUT_DIRECTORIES = {
    "image_dir": "images/",
    "image_meta_dir": "image_metadata/",
    "rhone_meta": "dataset_metadata/rhone",
}
# Set the names of the input files (excluding the root)
INPUT_FILES = {
    "manual_fiducials": "fiducials/Rhone_ManualFiducials_200909.csv",
    "sgi_1973": "shapefiles/SGI_1973.shp",
    "outlines_1935": "shapefiles/Glacierarea_1935_split.shp",
    "camera_locations": "shapefiles/V_TERRA_VIEWSHED_PARAMS.shp",
    "viewsheds": "shapefiles/V_TERRA_BGDI.shp",
    "rhone_image_filenames": "dataset_metadata/rhone/image_filenames.txt",
}
# Prepend the directory and file paths with the input root directory path.
INPUT_DIRECTORIES = {key: os.path.join(INPUT_ROOT_DIRECTORY, value) for key, value in INPUT_DIRECTORIES.items()}
INPUT_FILES = {key: os.path.join(INPUT_ROOT_DIRECTORY, value) for key, value in INPUT_FILES.items()}
# Set the expected types of the files (which can then be checked)
INPUT_FILE_TYPES = {
    "manual_fiducials": "CSV text",
    "sgi_1973": "ESRI Shapefile",
    "outlines_1935": "ESRI Shapefile",
    "viewsheds": "ESRI Shapefile",
    "camera_locations": "ESRI Shapefile",
    "image_dir": "TIFF image",
    "image_meta_dir": "text",
    "rhone_meta": ["text", "data"],
    "rhone_image_filenames": "text",
}


# Retrieve the dataset tags
DATASETS: List[str] = []
for _filename in os.listdir(os.path.join(INPUT_ROOT_DIRECTORY, "dataset_metadata")):
    extension, name = os.path.splitext(_filename)
    if extension == ".toml":
        DATASETS.append(name)


def clear_cache() -> None:
    """Clear the cache (temp) directory."""
    if not os.path.isdir(TEMP_DIRECTORY):
        print("Cache not existing")
    shutil.rmtree(TEMP_DIRECTORY)


@ statictypes.enforce
def check_filetype(estimated_filetype: str, allowed_types: Union[str, List[str]]) -> bool:
    """
    Check if a file has an expected filetype.

    param: estimated_filetype: The filetype that was estimated by magic.
    param: allowed_types: A string or a list of strings to check.

    return: correct: Whether the filetype contains the correct keyword.
    """
    if isinstance(allowed_types, str):
        allowed_types = [allowed_types]

    correct = any([file_type in estimated_filetype for file_type in allowed_types])
    return correct


def remove_locks() -> None:
    from terra import processing

    if not os.path.isdir(processing.inputs.TEMP_DIRECTORY):
        print("No lockfiles present")
        return

    datasets = os.listdir(processing.inputs.TEMP_DIRECTORY)
    if len(datasets) == 0:
        print("No lockfiles present")
        return

    for dataset in datasets:
        dataset_root = processing.inputs.CACHE_FILES[f"{dataset}_dir"]
        lockfile_path = os.path.join(dataset_root, f"{dataset}.files", "lock")

        if os.path.isfile(lockfile_path):
            print(f"Found {lockfile_path}. Removing.")
            os.remove(lockfile_path)


# TODO: Make this more usable
def list_cache() -> None:
    """List each file in the cache."""
    if not os.path.isdir(TEMP_DIRECTORY):
        print("No cache present")
        return
    for root_dir, _, filenames in os.walk(TEMP_DIRECTORY):
        for filename in filenames:
            print(os.path.join(root_dir, filename))


def check_data() -> None:
    """Check that all data can be found and that they have the correct file types."""
    max_entries = 50  # The maximum amount of wrong entries to find before stopping
    invalid_files = []
    missing_files = []
    MissingData = namedtuple("MissingData", ["filepath", "entry_type"])
    InvalidData = namedtuple("InvalidData", ["filepath", "expected_filetype", "actual_filetype"])

    # TODO: Maybe make this recursive and not just in one level
    # Loop through each directory and check filetypes
    print("Checking filetypes in directories")
    for key, directory in INPUT_DIRECTORIES.items():
        # Check if the maximum amount of wrong entries has been reached
        if (len(invalid_files) + len(missing_files)) > max_entries:
            break
        # Check if the directory exists
        if not os.path.isdir(directory):
            missing_files.append(MissingData(directory, "directory"))
        # Check filetypes inside the directory
        for filename in os.listdir(directory):
            # Check if the maximum amount of wrong entries has been reached
            if (len(invalid_files) + len(missing_files)) > max_entries:
                break

            full_path = os.path.join(directory, filename)
            estimated_filetype = magic.from_file(full_path)

            valid = check_filetype(estimated_filetype, INPUT_FILE_TYPES[key])
            if not valid:
                invalid_files.append(InvalidData(full_path, INPUT_FILE_TYPES[key], estimated_filetype))

    # Loop through each input file and check filetypes
    print("Checking filetypes for each input file")
    for key in INPUT_FILES:
        # Check if the maximum amount of wrong entries has been reached
        if (len(invalid_files) + len(missing_files)) > max_entries:
            break
        # Check if the file exists
        if not os.path.isfile(INPUT_FILES[key]):
            missing_files.append(MissingData(INPUT_FILES[key], "file"))
            continue

        # Check its filetype
        estimated_filetype = magic.from_file(INPUT_FILES[key])
        valid = check_filetype(estimated_filetype, INPUT_FILE_TYPES[key])
        if not valid:
            invalid_files.append(InvalidData(INPUT_FILES[key], INPUT_FILE_TYPES[key], estimated_filetype))

    if len(invalid_files) == 0 and len(missing_files) == 0:
        print("All files and directories found")
        return

    if (len(invalid_files) + len(missing_files)) == max_entries:
        print(f"Reached the maximum amount of incorrect entries. Printing the first {max_entries}.")

    for missing_data in missing_files:
        print(f"{missing_data.entry_type.capitalize()} {missing_data.filepath} not found")

    for invalid_file in invalid_files:
        print("File {} has an expected type of {} but has {}".format(
            invalid_file.filepath, invalid_file.expected_filetype, invalid_file.actual_filetype))


def list_input_directory(directory):
    """
    List a directory and return the full paths of objects, including the input root dir.

    param: directory: The directory path without the root dir path.
    """
    for filename in pathlib.Path(directory).glob("**/*"):
        yield str(filename)


def list_image_paths() -> Generator[str, None, None]:
    """List each image path in the input directory."""
    return list_input_directory(INPUT_DIRECTORIES["image_dir"])


def list_image_meta_paths() -> Generator[str, None, None]:
    """List each image metadata path in the input directory."""
    return list_input_directory(INPUT_DIRECTORIES["image_meta_dir"])


def read_dataset_meta(dataset: str):
    dataset_meta = toml.load(os.path.join(INPUT_ROOT_DIRECTORY, "dataset_metadata", f"{dataset}.toml"))

    dataset_meta["tag"] = dataset
    print(dataset_meta)

    return dataset_meta


if __name__ == "__main__":
    read_dataset_meta("rhone")
