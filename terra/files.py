"""File handling helper functions."""
import os
import shutil
import magic
from collections import namedtuple


INPUT_ROOT_DIRECTORY = "input"

if "SWISSTERRA_INPUT_DIR" in os.environ:
    INPUT_ROOT_DIRECTORY = os.environ["SWISSTERRA_INPUT_DIR"]

TEMP_DIRECTORY = "temp"


INPUT_DIRECTORIES = {
    "image_dir": "images/",
    "image_meta_dir": "image_metadata/"
}

INPUT_FILES = {
    "manual_fiducials": "fiducials/Rhone_ManualFiducials_200909.csv",
    "sgi_1973": "shapefiles/SGI_1973.shp",
    "outlines_1973": "shapefiles/Glacierarea_1935_split.shp",
    "viewsheds": "shapefiles/V_TERRA_VIEWSHED_PARAMS.shp"
}

# Prepend the directory and file paths with the input root directory path.
INPUT_DIRECTORIES = {key: os.path.join(INPUT_ROOT_DIRECTORY, value) for key, value in INPUT_DIRECTORIES.items()}
INPUT_FILES = {key: os.path.join(INPUT_ROOT_DIRECTORY, value) for key, value in INPUT_FILES.items()}

INPUT_FILE_TYPES = {
    "manual_fiducials": "CSV text",
    "sgi_1973": "ESRI Shapefile",
    "outlines_1973": "ESRI Shapefile",
    "viewsheds": "ESRI Shapefile",
    "image_dir": "TIFF image",
    "image_meta_dir": "text"
}


def clear_cache():
    """Clear the cache (temp) directory."""
    if not os.path.isdir(TEMP_DIRECTORY):
        print("Cache not existing")
    shutil.rmtree(TEMP_DIRECTORY)


# TODO: Make this more usable
def list_cache():
    """List each file in the cache."""
    if not os.path.isdir(TEMP_DIRECTORY):
        print("No cache present")
        return
    for root_dir, _, filenames in os.walk(TEMP_DIRECTORY):
        for filename in filenames:
            print(os.path.join(root_dir, filename))


def check_data():

    max_entries = 50
    invalid_files = []
    missing_files = []
    MissingData = namedtuple("MissingData", ["filepath", "entry_type"])
    InvalidData = namedtuple("InvalidData", ["filepath", "expected_filetype", "actual_filetype"])

    # TODO: Maybe make this recursive and not just in one level
    for key, directory in INPUT_DIRECTORIES.items():
        if (len(invalid_files) + len(missing_files)) > max_entries:
            break
        if not os.path.isdir(directory):
            missing_files.append(MissingData(directory, "directory"))
        for filename in os.listdir(directory):
            full_path = os.path.join(directory, filename)
            estimated_filetype = magic.from_file(full_path)

            valid = INPUT_FILE_TYPES[key] in estimated_filetype
            if not valid:
                invalid_files.append(InvalidData(full_path, INPUT_FILE_TYPES[key], estimated_filetype))

    print("Checking filetypes for each input file")
    for key in INPUT_FILES:
        if (len(invalid_files) + len(missing_files)) > max_entries:
            break
        if not os.path.isfile(INPUT_FILES[key]):
            missing_files.append(MissingData(INPUT_FILES[key], "file"))
        estimated_filetype = magic.from_file(INPUT_FILES[key])
        valid = INPUT_FILE_TYPES[key] in estimated_filetype
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
