import os


INPUT_ROOT_DIRECTORY = "input"

if "SWISSTERRA_INPUT_DIR" in os.environ:
    INPUT_ROOT_DIRECTORY = os.environ["SWISSTERRA_INPUT_DIR"]

TEMP_DIRECTORY = "temp"
