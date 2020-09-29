"""File handling helper functions."""
import os
import shutil


INPUT_ROOT_DIRECTORY = "input"

if "SWISSTERRA_INPUT_DIR" in os.environ:
    INPUT_ROOT_DIRECTORY = os.environ["SWISSTERRA_INPUT_DIR"]

TEMP_DIRECTORY = "temp"


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
