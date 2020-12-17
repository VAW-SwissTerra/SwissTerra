import concurrent.futures
import os
import pickle
import warnings
from collections import deque
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import skimage.transform
from tqdm import tqdm

from terra import files
from terra.constants import CONSTANTS
from terra.preprocessing import fiducials, image_meta

TEMP_DIRECTORY = os.path.join(files.TEMP_DIRECTORY, "preprocessing")

CACHE_FILES = {
    "mask_dir": os.path.join(TEMP_DIRECTORY, "masks"),
    "reference_mask": os.path.join(TEMP_DIRECTORY, "reference_mask.tif"),
}


# TODO: Consider transforming the input images directly instead of reading the already transformed images
# This reduces the reliance on intermediate files, but might as a consequence be slower.

def prepare_frame(original_frame: np.ndarray, buffer_size: int = 20) -> np.ndarray:
    """
    Preprocess an image frame to produce a good image mask.

    :param buffer_size: The amount of pixels to buffer the calculated reference frame with.
    """
    # Remove imperfections in the mask by flood-filling it, starting from the centre.
    center_xy = (original_frame.shape[1] // 2, original_frame.shape[0] // 2)
    cv2.floodFill(image=original_frame, mask=None, seedPoint=center_xy, newVal=200)
    # Extract the wanted colour (an arbitrary value of 200) to get the binary mask
    filled_frame = (original_frame == 200).astype(np.uint8) * 255

    # Buffer the mask to account for outlying imperfections and save the mask
    buffered_mask = scipy.ndimage.minimum_filter(filled_frame, size=buffer_size, mode="constant")

    return buffered_mask


def generate_masks(overwrite_existing: bool = False) -> None:
    """
    Generate frame masks for all images with a frame transform.
    """
    transforms = fiducials.get_all_instrument_transforms(verbose=False)

    reference_frame_names = os.listdir(fiducials.CACHE_FILES["image_frames_dir"])
    instruments = [frame_filename.replace("frame_", "").replace(".tif", "") for frame_filename in reference_frame_names]
    reference_frames = {instrument: prepare_frame(cv2.imread(
        os.path.join(fiducials.CACHE_FILES["image_frames_dir"], filename),
        cv2.IMREAD_GRAYSCALE
    )) for instrument, filename in zip(instruments, reference_frame_names)}

    instrument_filenames = {instrument: image_meta.get_filenames_for_instrument(
        instrument) for instrument in instruments}


    filenames_to_process = []
    filename_instruments: dict[str, str] = {}
    for instrument in instrument_filenames:
        for filename in instrument_filenames[instrument]:
            filename_instruments[filename] = instrument
            filenames_to_process.append(filename)

    # Reserve the variable for progress bars which jump in and out of existence
    progress_bar: Optional[tqdm] = None

    # Make temporary directories
    os.makedirs(CACHE_FILES["mask_dir"], exist_ok=True)

    def transform_and_write_frame(filename: str) -> None:
        """
        Transform a frame/mask to the original transform of an image.

        param: filename: The filename (excluding the path) of an image.

        return: None
        """
        full_path = os.path.join(CACHE_FILES["mask_dir"], filename)
        if os.path.isfile(full_path) and not overwrite_existing:
            return
        # Read the shape of the original image
        original_shape = cv2.imread(os.path.join(
            files.INPUT_DIRECTORIES["image_dir"], filename), cv2.IMREAD_GRAYSCALE).shape
        reference_frame = reference_frames[filename_instruments[filename]]
        # Transform the mask
        transformed_frame = fiducials.transform_image(
            reference_frame, transforms[filename], output_shape=original_shape)
        # Write it to the temporary mask directory
        cv2.imwrite(full_path, transformed_frame)
        progress_bar.update()

    print("Transforming masks and writing them")
    # Transform the masks to the images' original transform.
    with tqdm(total=len(filenames_to_process)) as progress_bar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONSTANTS.max_threads) as executor:
            # Unwrap the generator using a zero-length deque (empty collection) in order to actually run it.
            # Why this is needed is beyond me!
            deque(executor.map(transform_and_write_frame, filenames_to_process), maxlen=0)


def show_reference_mask():
    """Show the generated reference mask, if there is one."""
    if not os.path.isfile(CACHE_FILES["reference_mask"]):
        print("No reference mask found!")
        return
    frame = cv2.imread(CACHE_FILES["reference_mask"], cv2.IMREAD_GRAYSCALE)

    plt.imshow(frame, cmap="Greys_r")
    plt.title("Reference mask")
    plt.show()


if __name__ == "__main__":
    generate_masks()
