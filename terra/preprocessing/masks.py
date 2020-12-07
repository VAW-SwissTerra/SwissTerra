import concurrent.futures
import os
import pickle
from collections import deque
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import skimage.transform
import statictypes
from tqdm import tqdm

from terra import files
from terra.preprocessing import fiducials

TEMP_DIRECTORY = os.path.join(files.TEMP_DIRECTORY, "preprocessing")

CACHE_FILES = {
    "mask_dir": os.path.join(TEMP_DIRECTORY, "masks"),
    "reference_mask": os.path.join(TEMP_DIRECTORY, "reference_mask.tif"),
}


# TODO: Consider transforming the input images directly instead of reading the already transformed images
# This reduces the reliance on intermediate files, but might as a consequence be slower.


@statictypes.enforce
def generate_masks(buffer_size: int = 20) -> None:
    """
    Generate frame masks for all images with a frame transform.

    First, a reference frame mask is generated from all images.
    Then this reference is transformed back to each image's original transform.

    param: buffer_size: The amount of pixels to buffer the calculated reference frame with.
    """
    # TODO: Fix new loading of transforms
    raise NotImplementedError("Transforms are loaded in a new way")
    # Instantiate a FrameMatcher instance to use static methods and get metadata
    matcher = fiducials.FrameMatcher(verbose=False)

    transforms = matcher.load_transforms("merged_transforms.pkl")

    # Reserve the variable for progress bars which jump in and out of existence
    progress_bar: Optional[tqdm] = None

    def extract_frame(filename_and_transform) -> np.ndarray:
        """
        Read an image and extract its frame.

        param: filename_and_transform: The filename and the transform of the image.

        return: frame: The thresholded image frame.
        """
        filename, transform = filename_and_transform
        image = matcher.read_image(filename)
        transformed_image = matcher.transform_image(image, transform, output_shape=matcher.reference_frame.shape)
        frame = matcher.extract_image_frame(transformed_image)

        progress_bar.update()
        return frame

    print("Extracting image frames")
    # Extract frames from every image
    with tqdm(total=len(transforms)) as progress_bar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=matcher.max_open_files) as executor:
            frames = executor.map(extract_frame, list(transforms.items()))

    print("Merging results")
    # Get the median frame as a reference frame
    median_frame = np.median(list(frames), axis=0).astype(np.uint8)

    # Remove imperfections in the mask by flood-filling it, starting from the centre.
    center_xy = (median_frame.shape[1] // 2, median_frame.shape[0] // 2)
    cv2.floodFill(image=median_frame, mask=None, seedPoint=center_xy, newVal=200)
    # Extract the wanted colour (an arbitrary value of 200) to get the binary mask
    filled_frame = (median_frame == 200).astype(np.uint8) * 255

    # Make temporary directories
    os.makedirs(CACHE_FILES["mask_dir"], exist_ok=True)

    # Buffer the mask to account for outlying imperfections and save the mask
    reference_mask = scipy.ndimage.minimum_filter(filled_frame, size=buffer_size, mode="constant")
    cv2.imwrite(CACHE_FILES["reference_mask"], reference_mask)

    @ statictypes.enforce
    def transform_and_write_frame(filename: str) -> None:
        """
        Transform a frame/mask to the original transform of an image.

        param: filename: The filename (excluding the path) of an image.

        return: None
        """
        full_path = os.path.join(CACHE_FILES["mask_dir"], filename)
        # Read the shape of the original image
        original_shape = cv2.imread(os.path.join(
            files.INPUT_DIRECTORIES["image_dir"], filename), cv2.IMREAD_GRAYSCALE).shape
        # Transform the mask
        transformed_frame = matcher.transform_image(
            reference_mask, matcher.invert_transform(transforms[filename]), output_shape=original_shape)
        # Write it to the temporary mask directory
        cv2.imwrite(full_path, transformed_frame)
        progress_bar.update()

    print("Transforming masks and writing them")
    # Transform the masks to the images' original transform.
    with tqdm(total=len(transforms.keys())) as progress_bar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=matcher.max_open_files) as executor:
            # Unwrap the generator using a zero-length deque (empty collection) in order to actually run it.
            # Why this is needed is beyond me!
            deque(executor.map(transform_and_write_frame, list(transforms.keys())), maxlen=0)


def show_reference_mask():
    """Show the generated reference mask, if there is one."""
    if not os.path.isfile(CACHE_FILES["reference_mask"]):
        print("No reference mask found!")
        return
    frame = cv2.imread(CACHE_FILES["reference_mask"], cv2.IMREAD_GRAYSCALE)

    plt.imshow(frame, cmap="Greys_r")
    plt.title("Reference mask")
    plt.show()
