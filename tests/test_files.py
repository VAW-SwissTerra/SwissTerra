import os
import shutil

from terra import files


def test_list_directory():
    """Test the directory tree generator by making a dummy tree and listing it."""
    root = files.INPUT_ROOT_DIRECTORY
    test_dir = os.path.join(root, "test_dir")

    os.makedirs(test_dir, exist_ok=True)

    testfiles = [os.path.join(test_dir, f"testfile_{number}.txt") for number in [1, 2]]

    for testfile in testfiles:
        with open(testfile, "w") as outfile:
            outfile.write("hello there")

    for i, filename in enumerate(files.list_input_directory("test_dir")):
        assert filename == testfiles[i], f"Expected {testfiles[i]}, got {filename}."

    shutil.rmtree(test_dir)
