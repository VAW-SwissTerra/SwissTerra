
import os
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.transform
import statictypes

from terra import files
from terra.preprocessing import manual_picking


class ImageTransforms:

    def __init__(self, frame_type: str, filenames: Optional[list[str]] = None,
                 transforms: Optional[list[skimage.transform.EuclideanTransform]] = None) -> None:

        if not None in [filenames, transforms] and len(filenames) == len(transforms):  # type: ignore
            raise ValueError("The given filenames and transforms are not equally long.")

        self._inner_dict: dict[str, skimage.transform.EuclideanTransform] = dict(
            zip(filenames, transforms)) if not None in [filenames, transforms] else {}  # type: ignore

        self.frame_type = frame_type

        self.residuals: Optional[np.ndarray] = None

    @ statictypes.enforce
    def __setitem__(self, filename: str, transform: skimage.transform.EuclideanTransform) -> None:
        self._inner_dict[filename] = transform

    @ statictypes.enforce
    def __getitem__(self, filename: str) -> skimage.transform.EuclideanTransform:
        return self._inner_dict[filename]

    def __repr__(self) -> str:
        return f"ImageTransforms object of {len(self._inner_dict)} entries of the type {self.frame_type}. Error: {self.get_error():.2f} px"

    def __iter__(self):
        for filename in self._inner_dict:
            yield filename

    def add_residuals(self, residuals: np.ndarray) -> None:

        if len(self._inner_dict) != residuals.shape[0]:
            raise ValueError("Residuals have a different shape than the transform entries!")

        self.residuals = residuals

    def get_error(self) -> float:

        return np.sqrt(np.mean(np.square(self.residuals)))


def make_reference_transforms(marked_fiducials: manual_picking.MarkedFiducials, frame_type: str) -> ImageTransforms:
    """
    Generate reference transforms from manually placed fiducial marks.

    The centre is defined as the mean coordinate of the top and bottom fiducial.
    First, rotations between left-right and bottom-top are estimated and accounted for (they should be near 0).
    Then, they are shifted to a common centre, which is the median of the top-bottom mean centres.

    :param fiducial_marks: An instance of the marked fiducial collection.
    :param frame_type: The name of the frame type to look for.

    :returns: A collection of image transforms made from the marked fiducials.
    """
    filenames = marked_fiducials.get_filenames_for_frame_type(frame_type)

    # Instantiate dictionaries of original and corrected positions. These will be used to estimate transforms.
    original_positions: dict[str, np.ndarray] = {}
    corrected_positions: dict[str, np.ndarray] = {}
    # Define the order in which the corners are represented
    corners = ["left", "right", "top", "bottom"]

    # Estimate the rotational offset of images with valid fiducial marks.
    # The ones that are valid get an original and (not yet finished) corrected position entry in the above dicts.
    for filename in filenames:
        # Find the latest fiducla
        fiducials: dict[str, manual_picking.FiducialMark] = {
            corner: mark for corner, mark in marked_fiducials.get_fiducial_marks(filename).items() if mark is not None
        }

        # Only continue if all the fiducials are marked
        if len(fiducials) < 4:
            continue

        # Get the horizontal angle between the left and right fiducials.
        if (fiducials["left"].y_position - fiducials["right"].y_position) == 0:  # type: ignore
            horizontal_angle = 0.0
        else:
            horizontal_angle = np.arctan(
                float((fiducials["left"].y_position - fiducials["right"].y_position)) /  # type: ignore
                float((fiducials["left"].x_position - fiducials["right"].x_position))  # type: ignore
            )

        # Get the vertical angle (in the same reference as the horizontal) between the top and bottom fiducials.
        if (fiducials["top"].x_position - fiducials["bottom"].x_position) == 0:  # type: ignore
            vertical_angle = 0.0
        else:
            vertical_angle = -np.arctan(
                float(fiducials["top"].x_position - fiducials["bottom"].x_position) /  # type: ignore
                float(fiducials["top"].y_position - fiducials["bottom"].y_position)  # type: ignore
            )

        # Remove outliers whose angles are very different from each other
        if abs(horizontal_angle - vertical_angle) > np.deg2rad(4):
            continue

        # TODO: Save the vertical and horizontal angle difference.

        # Make a rotational transform from the angles (used to correct the points)
        rotation_transform = skimage.transform.EuclideanTransform(rotation=np.mean([horizontal_angle, vertical_angle]))

        # Make empty entires for the original and corrected positions (axis 0 are the corners, axis1 are the y/x coords)
        original_positions[filename] = np.empty(shape=(4, 2))
        corrected_positions[filename] = original_positions[filename].copy()
        # Go through each fiducial and add its original and corrected positions to the filename entry
        for i, corner in enumerate(corners):
            original_positions[filename][i, :] = [fiducials[corner].y_position, fiducials[corner].x_position]
            # The transform object wants x/y coordinates, so ::-1 has to be made to switch the two coordinates.
            new_x, new_y = rotation_transform.inverse(original_positions[filename][i, ::-1])[0]
            corrected_positions[filename][i, :] = [new_y, new_x]

    # Find the centre of the mean top and bottom coordinates between all images.
    top_bottom = [corners.index("top"), corners.index("bottom")]
    centre = np.mean([coordinate[top_bottom, :].mean(axis=0)
                      for coordinate in corrected_positions.values()], axis=0)

    # Correct all the x and y offsets of the fiducial locations
    for filename in corrected_positions:
        corrected_positions[filename] -= corrected_positions[filename][top_bottom, :].mean(axis=0) - centre

    # Find the residuals after the above correction
    values = np.array(list(corrected_positions.values()))
    residuals = np.linalg.norm(values - np.median(values, axis=0), axis=2)

    # Mark all entries as outliers whose residuals are too large
    inliers: list[int] = []
    outliers: list[str] = []
    for i, filename in enumerate(list(corrected_positions.keys())):
        if np.mean(residuals[i]) > 5:
            outliers.append(filename)
        else:
            inliers.append(i)

    # Make a new image transforms collection instance
    image_transforms = ImageTransforms(frame_type=frame_type)
    # Loop over the filenames in the corrected_positions collection and remove outliers.
    for filename in corrected_positions:
        if filename in outliers:
            continue
        # Make x/y source and destination coordinates to estimate the transform with.
        source_coords = original_positions[filename][:, ::-1]
        destination_coords = corrected_positions[filename][:, ::-1]

        # Estimate the transform and add it to the new instance.
        transform = skimage.transform.estimate_transform("euclidean", source_coords, destination_coords)
        image_transforms[filename] = transform

    # Add the residuals of the inliers for later analysis
    image_transforms.add_residuals(residuals[inliers])

    return image_transforms


def load_image(filename: str) -> np.ndarray:
    """
    Load an image from its base filename (exluding the folder path).

    :param filename: The name of the image.
    """
    image = cv2.imread(os.path.join(files.INPUT_DIRECTORIES["image_dir"], filename), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot find image {filename}")
    image -= image.min()
    image = (image * (255 / image.max())).astype(image.dtype)

    return image


def transform_image(image: np.ndarray, transform: skimage.transform.EuclideanTransform, output_shape: tuple[int, int]) -> np.ndarray:
    transformed_image = skimage.transform.warp(image, transform, output_shape=output_shape, preserve_range=True)
    return transformed_image.astype(image.dtype)


def generate_reference_frame(manual_transforms: ImageTransforms) -> np.ndarray:

    transformed_images = []
    for i, filename in enumerate(manual_transforms):
        image = load_image(filename)
        if i == 0:
            output_shape = image.shape

        transformed_image = transform_image(image, manual_transforms[filename], output_shape)
        transformed_images.append(transformed_image)

    variance = np.std(transformed_images, axis=0)

    print(variance.shape)


if __name__ == "__main__":
    marked_fiducials = manual_picking.MarkedFiducials.read_csv(os.path.join(
        files.INPUT_ROOT_DIRECTORY, "../manual_input/marked_fiducials.csv"))

    manual_transforms = make_reference_transforms(marked_fiducials, "blocky")
    generate_reference_frame(manual_transforms)
