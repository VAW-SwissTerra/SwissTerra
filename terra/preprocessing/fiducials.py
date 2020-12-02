
import concurrent.futures
import json
import os
import pickle
from typing import NamedTuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.transform
import statictypes
from tqdm import tqdm

from terra import files
from terra.constants import CONSTANTS
from terra.preprocessing import image_meta, manual_picking

TEMP_DIRECTORY = os.path.join(files.TEMP_DIRECTORY, "fiducials")
CACHE_FILES = {
    "fiducial_template_dir": os.path.join(TEMP_DIRECTORY, "templates"),
}


# TODO: Maybe replace hardcoded fiducial locations with ones estimated from the manual picking?


class ImageTransforms:
    """A collection of image transforms."""

    def __init__(self, frame_type: str, filenames: Optional[list[str]] = None,
                 transforms: Optional[list[skimage.transform.EuclideanTransform]] = None) -> None:
        """
        Generate a new ImageTransforms instance.

        :param: frame_type: The type of the frame, e.g. blocky, triangular, zeiss_normal etc.
        :param: filenames: Optional: Filenames from which to create a transform collection.
        :param: transforms: Optional: Transforms to map to the above filenames (shapes have to be the same).
        """
        # Check that the given filenames and transforms (if any) have the same length
        if not None in [filenames, transforms] and len(filenames) != len(transforms):  # type: ignore
            raise ValueError("The given filenames and transforms are not equally long.")

        # Map the given filenames and transforms (if any) to each other, or create an empty collection
        self._inner_dict: dict[str, skimage.transform.EuclideanTransform] = dict(
            zip(filenames, transforms)) if not None in [filenames, transforms] else {}  # type: ignore

        self.frame_type = frame_type

        # Residuals may be added lateGggr
        self.residuals: Optional[np.ndarray] = None
        self.matching_uncertainties: Optional[np.ndarray] = None
        # self.error: Optional[list[float]] = None

    @statictypes.enforce
    def __setitem__(self, filename: str, transform: skimage.transform.EuclideanTransform) -> None:
        """
        Add or modify an item to the collection.

        :param filename: The key to map.
        """
        self._inner_dict[filename] = transform

    @statictypes.enforce
    def __getitem__(self, filename: str) -> skimage.transform.EuclideanTransform:
        """Retrieve an item from the collection."""
        return self._inner_dict[filename]

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        string = f"ImageTransforms object with {len(self._inner_dict)} entries of the type {self.frame_type}. "

        if self.residuals is not None:
            string += f"Residual error: {self.get_residual_rmse():.2f} px"
        if self.matching_uncertainties is not None:
            string += f"Mean uncertainty: {self.get_mean_uncertainty() * 100: .2f}%,"
            string += f" max: {np.max(self.matching_uncertainties) * 100:.2f}%"
        return string

    def __iter__(self):
        """Return an iterator of the collection's filenames."""
        for filename in self._inner_dict:
            yield filename

    def add_transforms(self, filenames: list[str], transforms: list[skimage.transform.EuclideanTransform]):

        if len(filenames) != len(transforms):
            raise ValueError("Filenames and transforms length differ")

        for filename, transform in zip(filenames, transforms):
            self._inner_dict[filename] = transform

    def keys(self) -> list[str]:
        """Return a list of the filenames."""
        return list(self._inner_dict.keys())

    def add_residuals(self, residuals: np.ndarray) -> None:
        """
        Add residuals (alignment errors) to the collection.

        They have to be the of same shape (shape[0]) as the transform collection.
        :param residuals: The residual errors.
        """
        if len(self._inner_dict) != residuals.shape[0]:
            raise ValueError("Residuals have a different shape than the transform entries!")

        self.residuals = residuals

    def add_uncertainties(self, uncertainties: list[list[float]]) -> None:
        """
        Add feature-matching related uncertainties to the collection.

        :param uncertainties: A list of four uncertainties (for each fiducial) with the same len as the collection.
        """
        if len(self._inner_dict) != len(uncertainties):
            raise ValueError("Uncertainties list does not match the length of the collection")

        self.matching_uncertainties = np.array(uncertainties)

    def get_mean_uncertainty(self) -> float:
        """Return the mean uncertainty of the collection."""

        if self.matching_uncertainties is None:
            return np.NaN

        return np.mean(self.matching_uncertainties)

    def get_residual_rmse(self) -> float:
        """
        Get the root mean square error of the residuals, assuming they exist.

        Returns NaN if no residuals are available.
        """
        if self.residuals is None:
            return np.NaN

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

    if len(filenames) == 0:
        raise ValueError(f"No filenames of frame type: {frame_type}")

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
        horizontal_angle = np.arctan(
            float((fiducials["left"].y_position - fiducials["right"].y_position)) /  # type: ignore
            float((fiducials["left"].x_position - fiducials["right"].x_position))  # type: ignore
        ) if (fiducials["left"].y_position - fiducials["right"].y_position) != 0 else 0.0  # type: ignore

        # Get the vertical angle (in the same reference as the horizontal) between the top and bottom fiducials.
        vertical_angle = -np.arctan(
            float(fiducials["top"].x_position - fiducials["bottom"].x_position) /  # type: ignore
            float(fiducials["top"].y_position - fiducials["bottom"].y_position)  # type: ignore
        ) if (fiducials["top"].x_position - fiducials["bottom"].x_position) != 0 else 0.0  # type:ignore

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

    # Increase the contrast of the image.
    image -= image.min()
    image = (image * (255 / image.max())).astype(image.dtype)

    return image


def transform_image(image: np.ndarray, transform: skimage.transform.EuclideanTransform, output_shape: tuple[int, int]) -> np.ndarray:
    """
    Transform an image using a transform object.

    Note that the transform is made on the inverse of the transform object.

    :param image: The image to transform.
    :param transform: The transform object.
    :param output_shape: The target shape for the image.
    :returns: A transformed image with the same dtype as the input and the same shape as output_shape.
    """
    transformed_image = skimage.transform.warp(image, transform.inverse, output_shape=output_shape, preserve_range=True)
    return transformed_image.astype(image.dtype)


def generate_reference_frame(image_transforms: ImageTransforms) -> np.ndarray:
    """
    Extract the median frame of the images in the transform keys.

    First, the images are transformed, then they are compared to each other and thresholded to extract the frame.

    :param manual_transforms: Transforms of the same frame type.
    :returns: A binary (0/255) mask of the reference frame.
    """
    # Set the shape of all the images to match the first one. TODO: Make this an argument?
    first_image = load_image(image_transforms.keys()[0])
    output_shape = first_image.shape
    output_dtype = first_image.dtype

    # Instantiate a progress bar.
    progress_bar = tqdm(total=len(image_transforms.keys()), desc="Transforming images")

    def get_transformed_image(filename: str) -> np.ndarray:
        """
        Load and transform an image in one thread.

        :param filename: The filename (excluding the directory path) to load.
        :returns: The transformed image.
        """
        image = load_image(filename)

        transformed_image = transform_image(image, image_transforms[filename], output_shape)

        progress_bar.update()
        return transformed_image

    # Load and transform all the available images.
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONSTANTS.max_threads) as executor:
        transformed_images = list(executor.map(get_transformed_image, image_transforms.keys()))

    progress_bar.close()

    # Threshold the median values to get the frame.
    _, frame = cv2.threshold(src=np.median(transformed_images, axis=0), thresh=80, maxval=255, type=cv2.THRESH_BINARY)

    return frame.astype(first_image.dtype)


class Bounds:  # pylint: disable=too-few-public-methods
    """Image bounds object"""

    def __init__(self, top: int, bottom: int, left: int, right: int):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return json.dumps(self.as_dict())

    def as_dict(self) -> dict[str, int]:
        return {"top": self.top, "bottom": self.bottom, "left": self.left, "right": self.right}

    def as_slice(self) -> np.lib.index_tricks.IndexExpression:
        """Convert the bounds object to a numpy slice"""
        return np.s_[self.top:self.bottom, self.left:self.right]


class Fiducial:

    def __init__(self, corner: str, fiducial_image: np.ndarray, bounds: Bounds, center: tuple[int, int]):
        self.corner = corner
        self.fiducial_image = fiducial_image
        self.bounds = bounds
        self.center = center

    def __repr__(self) -> str:
        return f"{self.corner.capitalize()} fiducial with bounds: {self.bounds}"

    def index(self, y_index, x_index) -> tuple[int, int]:
        return self.bounds.top + y_index, self.bounds.left + x_index

    def match(self, other_fiducial) -> tuple[tuple[int, int], float]:
        """
        Match a fiducial to another fiducial.

        The other fiducial needs to have a smaller image shape than 'self'.
        The returned error is the fraction of pixels within a 5% likelihood to also be a match (lower is better).

        :param other_fiducial: Another fiducial instance.
        :returns: The estimated center of the other fiducial after matching, and the uncertianty of the match.
        """
        # Template matching requires that the source image is larger than the template
        if self.fiducial_image.shape < other_fiducial.fiducial_image.shape:
            raise ValueError("The compared image is smaller than the source image.")

        # TODO: Test other matching methods
        # Run the main template matching. Returns a field (an image) of probabilites for where the top left
        # corner of the template (other_fiducial) is. The field has the same shape as self.fiducial_image
        result = cv2.matchTemplate(self.fiducial_image, templ=other_fiducial.fiducial_image, method=cv2.TM_CCOEFF)

        # Find the highest value in the result field and its corresponding index.
        _, max_score, _, max_score_index = cv2.minMaxLoc(result)

        # Calculate the new top left coordinate for other_fiducial
        new_top_coord = self.bounds.top + max_score_index[0]
        new_left_coord = self.bounds.left + max_score_index[1]

        # Calculate the new center of other_fiducial
        new_center = (
            other_fiducial.center[0] + (new_top_coord - other_fiducial.bounds.top),
            other_fiducial.center[1] + (new_left_coord - other_fiducial.bounds.left)
        )

        # The error is the fraction of pixels that are within 95% of the highest (a sharp peak is near 0)
        uncertainty = (np.count_nonzero(result > (max_score * 0.95)) - 1) / (result.shape[0] * result.shape[1])

        return new_center, uncertainty


def extract_fiducials(image: np.ndarray, frame_type: str, window_size: int = 250,
                      equalize: bool = True) -> dict[str, Fiducial]:
    """
    Extract all four fiducials from an image.

    :param image: The image to extract the fiducials from.
    :param window_size: The maximum radius of the window to extract the fiducial from (truncated by the image borders).
    :param equalize: Whether to equalize the luminance of each fiducial to make it clearer.
    :returns: A dictionary of Fiducial objects for each corner.
    """
    # Get the hardcoded approximate fiducial coordinates
    fiducial_coords = CONSTANTS.wild_fiducial_locations if "wild" in frame_type else CONSTANTS.zeiss_fiducial_locations

    # Instantiate a dictionary of 'corner':'fiducial instance' pairs
    fiducials: dict[str, Fiducial] = {}

    # Loop over each fiducial and create a Fiducial object from it.
    for corner in fiducial_coords:

        # Define the bounds of the window (making sure to stay within the bounds of the image)
        # The window may therefore not be entirely quadratic
        fiducial_bounds = Bounds(
            top=max(0, fiducial_coords[corner][0] - window_size),
            bottom=min(image.shape[0], fiducial_coords[corner][0] + window_size),
            left=max(0, fiducial_coords[corner][1] - window_size),
            right=min(image.shape[1], fiducial_coords[corner][1] + window_size)
        )

        # Extract the fiducial part of the image
        fiducial = image[fiducial_bounds.as_slice()]
        # Optionally stretch the lighness to between the 1st and 99th percentile of the lightness
        if equalize:
            max_value = np.percentile(fiducial.flatten(), 99)
            min_value = np.percentile(fiducial.flatten(), 1)
            fiducial = np.clip((fiducial - min_value) * (255 / (max_value - min_value)),
                               a_min=0, a_max=255).astype(fiducial.dtype)

        # Make a new fiducial instance and add it to the output dictionary
        fiducials[corner] = Fiducial(
            corner=corner,
            fiducial_image=fiducial,
            bounds=fiducial_bounds,
            center=(fiducial_coords[corner])
        )

    return fiducials


def get_fiducial_templates(transforms: ImageTransforms, frame_type: str, cache=True) -> dict[str, Fiducial]:
    """
    Construct or load fiducial templates from every image with manually picked fiducial marks.

    :param transforms: Manual transforms to align the images with.
    :param frame_type: The frame type of the images.
    :param cache: Whether to load and save cached results (this may take a while).

    :returns: A dictionary of fiducial templates (Fiducial objects) for every corner.
    """
    # Define the filepath to look for / save / ignore the cache file
    cache_filepath = os.path.join(CACHE_FILES["fiducial_template_dir"], frame_type + ".pkl")

    # Return a cached version if it exists and should be used.
    if cache and os.path.isfile(cache_filepath):
        return pickle.load(open(cache_filepath, mode="rb"))

    print("Generating new fiducial templates")
    # Generate a reference frame from the manually picked fiducial positions.
    reference_frame = generate_reference_frame(transforms)
    # Extract fiducials in predefined places from the frame
    fiducials = extract_fiducials(reference_frame, frame_type=frame_type, window_size=250, equalize=False)

    if cache:
        os.makedirs(CACHE_FILES["fiducial_template_dir"], exist_ok=True)
        # Save the fiducials object
        with open(cache_filepath, mode="wb") as outfile:
            pickle.dump(fiducials, outfile)

        # Save previews of the templates for debugging/visualisation purposes
        for corner in fiducials:
            cv2.imwrite(os.path.join(CACHE_FILES["fiducial_template_dir"],
                                     f"{frame_type}_{corner}_preview.jpg"), fiducials[corner].fiducial_image)
    return fiducials


def match_templates(filenames: list[str], frame_type: str, templates: dict[str, Fiducial],
                    cache: bool = True) -> ImageTransforms:
    """
    Use template matching to match fiducials in images, compared to the provided fiducial templates.

    :param filenames: A list of filenames to analyse.
    :param frame_type: The frame type of the images.
    :param templates: Fiducial templates to try to match.
    :param cache: Whether to load and save cached results (this may take a while).
    :returns: Transforms estimated from feature matching, with corresponding uncertainties.
    """
    # Define the filepath to look for / save / ignore the cache file
    cache_filepath = os.path.join(TEMP_DIRECTORY, "transforms", frame_type + ".pkl")
    # Return a cached version if it exists and should be used
    if cache and os.path.isfile(cache_filepath):
        return pickle.load(open(cache_filepath, "rb"))

    print("Generating new template transforms")

    # Instantiate a progress bar to monitor the progress.
    progress_bar = tqdm(total=len(filenames), desc="Matching templates")

    def template_match(filename: str) -> tuple[skimage.transform.EuclideanTransform, list[float]]:
        """
        Match fiducials to the template fiducials in one thread.
        :param filename: The filename of the image to analyse.
        :returns: The estimated transform and the uncertainties of the four fiducials.
        """
        image = load_image(filename)
        # Extract a larger window than the default (500 vs 250) for the image to match the fiducial templates on.
        fiducials = extract_fiducials(image, frame_type=frame_type, window_size=500, equalize=True)

        original_positions = []
        new_positions = []
        uncertainties: list[float] = []
        # Loop over all fiducials and try to match them
        for corner in fiducials:
            new_position, uncertainty = fiducials[corner].match(templates[corner])
            original_positions.append(fiducials[corner].center)
            new_positions.append(new_position)
            uncertainties.append(uncertainty)

        # Estimate a transform using the x/y-swapped source and destination positions.
        transform = skimage.transform.estimate_transform(
            ttype="euclidean",
            src=np.array(original_positions)[:, ::-1],
            dst=np.array(new_positions)[:, ::-1]
        )
        progress_bar.update()
        return transform, uncertainties

    uncertainties: list[list[float]] = []
    transforms: list[skimage.transform.EuclideanTransform] = []
    # Match templates on all the given filenames
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONSTANTS.max_threads) as executor:
        for transform, error in executor.map(template_match, filenames):
            uncertainties.append(error)
            transforms.append(transform)
    progress_bar.close()

    template_transforms = ImageTransforms(frame_type=frame_type)
    template_transforms.add_transforms(filenames, transforms)
    template_transforms.add_uncertainties(uncertainties)

    # Optionally save the result
    if cache:
        os.makedirs(os.path.dirname(cache_filepath), exist_ok=True)
        with open(cache_filepath, "wb") as outfile:
            pickle.dump(template_transforms, outfile)

    return template_transforms


def get_all_filenames_for_frame_type(frame_type: str, marked_fiducials: manual_picking.MarkedFiducials) -> np.ndarray:
    """
    Retrieve all image filenames corresponding to the given frame type.

    :param frame_type: The type of the frame to look for.
    :param marked_fiducials: The manually marked fiducials to acquire the instrument label vs. frame type correlation.
    :returns: A list of filenames.
    """
    # Read all of the image metadata
    image_metadata = image_meta.read_metadata()
    # Extract the images that have been marked manually of the specified frame type
    marked_metadata = image_metadata[
        image_metadata["Image file"].isin(marked_fiducials.get_filenames_for_frame_type(frame_type))
    ]

    # Find which instruments the marked images belong to
    instruments = np.unique(marked_metadata["Instrument"].values)

    # Return all of the filenames taken by the corresponding instruments.
    return image_metadata[image_metadata["Instrument"].isin(instruments)]["Image file"].values


def estimate_transforms(frame_type: str, marked_fiducials: manual_picking.MarkedFiducials) -> ImageTransforms:
    """
    Estimate transforms on all images with the given frame type
    """
    pass


if __name__ == "__main__":
    marked_fiducials = manual_picking.MarkedFiducials.read_csv(os.path.join(
        files.INPUT_ROOT_DIRECTORY, "../manual_input/marked_fiducials.csv"))
    frame_type = "wild_triangular"

    all_filenames = get_all_filenames_for_frame_type(frame_type, marked_fiducials=marked_fiducials)

    manual_transforms = make_reference_transforms(marked_fiducials, frame_type=frame_type)
    templates = get_fiducial_templates(manual_transforms, frame_type=frame_type)

    transforms = match_templates(filenames=all_filenames, frame_type=frame_type, templates=templates)

    print(transforms)
