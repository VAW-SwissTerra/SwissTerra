"""Functions to handle and estimate fiducial locations for the internal orientation of the images."""
import concurrent.futures
import csv
import json
import os
import pickle
import warnings
from typing import Optional, Union

import cv2
import numpy as np
import skimage.measure
import skimage.transform
from tqdm import tqdm

from terra import files
from terra.constants import CONSTANTS
from terra.preprocessing import image_meta, manual_picking

TEMP_DIRECTORY = os.path.join(files.TEMP_DIRECTORY, "fiducials")
CACHE_FILES = {
    "fiducial_template_dir": os.path.join(TEMP_DIRECTORY, "templates"),
    "image_frames_dir": os.path.join(TEMP_DIRECTORY, "templates/image_frames/"),
    "fiducial_location_dir": os.path.join(TEMP_DIRECTORY, "locations"),
}
# Define the order in which fiducial corners are considered.
CORNERS = ["left", "right", "top", "bottom"]


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
        self.manual_residuals: Optional[np.ndarray] = None
        self.estimated_residuals: Optional[np.ndarray] = None
        # self.error: Optional[list[float]] = None

    def __setitem__(self, filename: str, transform: skimage.transform.EuclideanTransform) -> None:
        """
        Add or modify an item to the collection.

        :param filename: The key to map.
        """
        self._inner_dict[filename] = transform

    def __getitem__(self, filename: str) -> skimage.transform.EuclideanTransform:
        """Retrieve an item from the collection."""
        return self._inner_dict[filename]

    def __delitem__(self, filename: str):
        """Remove an item and its corresponding values from the collection."""
        index = self.keys().index(filename)
        if self.estimated_residuals is not None:
            self.estimated_residuals = np.delete(self.estimated_residuals, index, axis=0)

        if self.manual_residuals is not None:
            self.manual_residuals = np.delete(self.estimated_residuals, index, axis=0)

        del self._inner_dict[filename]

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        string = f"ImageTransforms object with {len(self._inner_dict)} entries of the type {self.frame_type}."

        if self.manual_residuals is not None:
            string += f" Median manual residual error: {np.nanmedian(self.manual_residuals):.2f} px,"
            string += f" max: {np.nanmax(self.manual_residuals):.2f} px."
        if self.estimated_residuals is not None:
            string += f" Median estimated residual error: {np.nanmedian(self.estimated_residuals):.2f} px,"
            string += f" max: {np.nanmax(self.estimated_residuals):.2f} px"
        return string

    def __iter__(self):
        """Return an iterator of the collection's filenames."""
        for filename in self._inner_dict:
            yield filename

    def add_transforms(self, filenames: list[str], transforms: list[skimage.transform.EuclideanTransform]):
        """
        Add a batch of transform objects  with their corresponding filenames.

        The filenames and transforms list need to have the same length.

        :param filenames: The filenames of the images belonging to the transforms.
        :param transforms: The transforms belonging to the images.
        """
        if len(filenames) != len(transforms):
            raise ValueError("Filenames and transforms length differ")

        for filename, transform in zip(filenames, transforms):
            self._inner_dict[filename] = transform

    @staticmethod
    def from_multiple(transform_objects):
        """
        Merge multiple ImageTransforms objects into one new object.

        :param transform_objects: An iterable of transform objects to merge.
        :type transform_objects: Iterable of ImageTransforms objects.

        :returns: A new merged ImageTransforms object.
        """
        # Create new empty attributes which will gradually be filled.
        frame_types: list[str] = []
        manual_residuals = np.empty(shape=(0, 4))
        estimated_residuals = np.empty(shape=(0, 4))
        inner_dict: dict[str, skimage.transform.EuclideanTransform] = {}

        # Fill the above attributes with the transform_objects' values.
        for transform_object in transform_objects:
            frame_types.append(transform_object.frame_type)
            manual_residuals = np.append(
                arr=manual_residuals,
                values=transform_object.manual_residuals if transform_object.manual_residuals is not None else
                np.empty(shape=(len(transform_object.keys()), 4)) + np.nan,
                axis=0
            )

            estimated_residuals = np.append(
                arr=estimated_residuals,
                values=transform_object.estimated_residuals
                if transform_object.estimated_residuals is not None else
                np.empty(shape=(len(transform_object.keys()), 4)) + np.nan,
                axis=0
            )
            # Sorry, pylint
            inner_dict |= transform_object._inner_dict  # pylint: disable=protected-access

        # Set the frame type to be "mixed" if there is more than one unique frame type
        frame_type = frame_types[0] if np.unique(frame_types).shape[0] == 1 else "mixed"

        merged_transforms = ImageTransforms(frame_type)
        merged_transforms._inner_dict = inner_dict  # Why is pylint not complaining here??
        merged_transforms.add_manual_residuals(manual_residuals)
        merged_transforms.add_estimated_residuals(estimated_residuals)

        return merged_transforms

    def keys(self) -> list[str]:
        """Return a list of the filenames."""
        return list(self._inner_dict.keys())

    def add_manual_residuals(self, residuals: np.ndarray) -> None:
        """
        Add residuals (alignment errors) to the collection.

        They have to be the of same shape (shape[0]) as the transform collection.
        :param residuals: The residual errors.
        """
        if len(self._inner_dict) != residuals.shape[0]:
            raise ValueError("Residuals have a different shape than the transform entries!")

        self.manual_residuals = residuals

    def add_estimated_residuals(self, residuals: list[list[float]]) -> None:
        """
        Add residuals from estimate transforms to the collection.

        :param residuals: A list of four residual (for each fiducial) with the same len as the collection.
        """
        residual_array = np.array(residuals)
        if len(self._inner_dict) != residual_array.shape[0]:
            raise ValueError("Residual list does not match the length of the collection")

        self.estimated_residuals = residual_array

    def get_outlier_filenames(self) -> np.ndarray:
        """Find the image filenames whose uncertainties are statistical outliers."""
        if self.estimated_residuals is None:
            raise ValueError("No uncertainties exist")

        if np.all(np.isnan(self.estimated_residuals)):
            return np.array([])

        # Get the maximum uncertainty for each image
        # It will warn if the residuals of all cornerns is NaN, which is completely normal.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            uncertainties = np.nanmax(self.estimated_residuals, axis=1)

        return np.array(self.keys())[uncertainties > 47 * 2]

    def get_residual_rmse(self) -> float:
        """
        Get the root mean square error of the residuals, assuming they exist.

        Returns NaN if no residuals are available.
        """
        if self.manual_residuals is None:
            return np.NaN

        return np.sqrt(np.nanmean(np.square(self.manual_residuals)))


def get_median_fiducial_locations(marked_fiducials: manual_picking.MarkedFiducials) -> np.ndarray:
    """
    Find the median location of the marked fiducials.

    All fiducial marks in marked_fiducials should have the same frame type (through "marked_fiducials.subset()").

    :param marked_fiducials: Marked fiducials of the same frame type.
    :returns: A numpy array of shape (4, 2) of the left, right, top, and bottom fiducials, respectively.
    """
    # Get the filenames for the marked fiducials
    filenames = np.unique([mark.filename for mark in marked_fiducials.fiducial_marks])

    # Instantiate an array of de-rotated fiducial positions.
    de_rotated_positions = np.empty(shape=(filenames.shape[0], 4, 2))

    # Go through each filename and try to remove the rotational component of the offset.
    # This is done by correcting the left-right and top-bottom angles.
    for i, filename in enumerate(filenames):
        # Find the latest fiducla
        fiducials: dict[str, manual_picking.FiducialMark] = {
            corner: mark for corner, mark in marked_fiducials.get_fiducial_marks(filename).items()
            if mark is not None and mark.x_position is not None
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

        # TODO: Save the vertical and horizontal angle difference?

        # Make a rotational transform from the angles (used to correct the points)
        rotation_transform = skimage.transform.EuclideanTransform(rotation=np.mean([horizontal_angle, vertical_angle]))

        # Make empty entires for the original and corrected positions (axis 0 are the corners, axis1 are the y/x coords)
        # Go through each fiducial and add its original and corrected positions to the filename entry
        for j, corner in enumerate(CORNERS):
            new_x, new_y = rotation_transform.inverse([fiducials[corner].x_position, fiducials[corner].y_position])[0]
            de_rotated_positions[i, j, :] = [new_y, new_x]

    # Get the median positions of the de-rotated fiducial locations
    median_fiducial_locations = np.median(de_rotated_positions, axis=0)

    return median_fiducial_locations


def calculate_manual_transforms(marked_fiducials: manual_picking.MarkedFiducials, frame_type: str) -> ImageTransforms:
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

    median_fiducial_locations = get_median_fiducial_locations(marked_fiducials.subset(filenames))

    # Make a new image transforms collection instance
    image_transforms = ImageTransforms(frame_type=frame_type)
    residuals: list[np.ndarray] = []

    # Loop over the filenames and estimate transforms for them.
    for filename in filenames:
        fiducial_marks = marked_fiducials.get_fiducial_marks(filename)

        # Get the source coordinates for the transform estimation.
        all_source_coords = np.array([
            ([fiducial_marks[corner].x_position, fiducial_marks[corner].y_position]
             if (fiducial_marks[corner] is not None and fiducial_marks[corner].x_position is not None)
             else [np.nan, np.nan]) for corner in CORNERS
        ])
        missing_values = np.isnan(np.mean(all_source_coords, axis=1))

        # Skip if two or fewer fiducials were marked.
        if np.count_nonzero(missing_values) > 1:
            continue

        source_coords = all_source_coords[~missing_values]
        destination_coords = median_fiducial_locations[~missing_values, ::-1]

        # If only three fiducials are marked, estimate a transform directly.
        ransac: bool = np.count_nonzero(missing_values) == 0

        # If four exist, estimate using RANSAC (possibly removing ONE outlier)
        # RANSAC may fail (sometimes with warnings, sometimes not). If so, fallback to the rigid method
        if ransac:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                transform, inliers = skimage.measure.ransac(
                    data=(
                        source_coords,
                        destination_coords
                    ),
                    model_class=skimage.transform.EuclideanTransform,
                    min_samples=3,  # At least three points should be used
                    stop_sample_num=1,  # If it finds one good outlier, it stops.
                    residual_threshold=CONSTANTS.transform_outlier_threshold)

                # If the transform params are NaNs (silent RANSAC failure), estimate a transform again without it.
                if np.any(np.isnan(transform.params)):
                    ransac = False

        # Estimate a transform without RANSAC if too few points are available or if RANSAC failed.
        if not ransac:
            transform = skimage.transform.estimate_transform("euclidean", source_coords, destination_coords)
            inliers = ~missing_values

        if np.any(np.isnan(transform.params)):
            raise ValueError(f"Transform was not estimated for {filename}: {transform}")

        # Now, two validation masks exist, so combine these into one.
        valid_mask = np.logical_or(missing_values, ~inliers)  # pylint: disable=invalid-unary-operand-type
        # Estimate the residuals on each valid point.
        residual = np.linalg.norm(median_fiducial_locations[~valid_mask, ::-1] -
                                  transform(all_source_coords)[~valid_mask], axis=1)

        # If there were invalid values, the residual array has to be filled with NaNs to have residual.shape[0] == 4
        for i, potential_nan in enumerate(valid_mask):
            if potential_nan:
                residual = np.insert(arr=residual, obj=i, values=np.nan)

        # Increase the residuals if marked fiducials are missing (if one is missing, multiply by two)
        residual *= np.count_nonzero(missing_values) + 1

        # Toss the results if they weren't good enough  # TODO: Remove the 4 multiplier for errors
        if np.mean(residual) > CONSTANTS.transform_outlier_threshold * 10:
            continue
        # Otherwise, save them
        residuals.append(residual)
        image_transforms[filename] = transform

    # Add the residuals of the inliers for later analysis
    image_transforms.add_manual_residuals(np.array(residuals))

    return image_transforms


def load_image(filename: str) -> np.ndarray:
    """
    Load an image from its base filename(exluding the folder path).

    :param filename: The name of the image.
    """
    image = cv2.imread(os.path.join(files.INPUT_DIRECTORIES["image_dir"], filename), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot find image {filename}")

    # Increase the contrast of the image.
    image -= image.min()
    image = (image * (255 / image.max())).astype(image.dtype)

    return image


def transform_image(image: np.ndarray, transform: skimage.transform.EuclideanTransform,
                    output_shape: tuple[int, int]) -> np.ndarray:
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
    :returns: A binary(0/255) mask of the reference frame.
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

        :param filename: The filename(excluding the directory path) to load.
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
    """Image bounds object for fiducial image extent calculations."""

    def __init__(self, top: int, bottom: int, left: int, right: int):
        """Create a new bounds object from the bounding coordinates."""
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def __str__(self) -> str:
        """Return a json string version of the bounds object."""
        return json.dumps(self.as_dict())

    def as_dict(self) -> dict[str, int]:
        """Convert the bounds object to a dictionary."""
        return {"top": self.top, "bottom": self.bottom, "left": self.left, "right": self.right}

    def as_slice(self) -> np.lib.index_tricks.IndexExpression:
        """Convert the bounds object to a numpy slice."""
        return np.s_[self.top:self.bottom, self.left:self.right]

    def as_plt_extent(self) -> tuple[int, int, int, int]:
        """Convert the bounds object to an extent for matplotlib."""
        return (self.left, self.right, self.bottom, self.top)


class Fiducial:
    """An extracted fiducial object which can be matched to another one."""

    def __init__(self, corner: str, fiducial_image: np.ndarray, bounds: Bounds, center: tuple[int, int]):
        """
        Instantiate a new Fiducial object.

        :param corner: The corner (left, right, top, bottom) of the fiducial.
        :param fiducial_image: The image data (the picture) of the fiducial.
        :param bounds: The bounds of the image data in the non-extracted image coordinates.
        :param center: The center of the fiducial (center of the bounds if it wasn't cropped by an image edge).
        """
        self.corner = corner
        self.fiducial_image = fiducial_image
        self.bounds = bounds
        self.center = center

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return f"{self.corner.capitalize()} fiducial with bounds: {self.bounds}"

    def index(self, y_index, x_index) -> tuple[int, int]:
        """Convert fiducial image coordinates to non-extracted image coordinates."""
        return self.bounds.top + y_index, self.bounds.left + x_index

    def match(self, other_fiducial) -> tuple[tuple[int, int], float]:
        """
        Match a fiducial to another fiducial.

        The other fiducial needs to have a smaller image shape than 'self'.
        The returned error is the fraction of pixels within a 5 % likelihood to also be a match (lower is better).

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
        new_top_coord = self.bounds.top + max_score_index[1]
        new_left_coord = self.bounds.left + max_score_index[0]

        # Calculate the new center of other_fiducial
        offset = (
            (new_top_coord - other_fiducial.bounds.top),
            (new_left_coord - other_fiducial.bounds.left)
        )

        # The error is the fraction of pixels that are within 95% of the highest (a sharp peak is near 0)
        uncertainty = (np.count_nonzero(result > (max_score * 0.95)) - 1) / (result.shape[0] * result.shape[1])

        return offset, uncertainty


def extract_fiducials(image: np.ndarray, frame_type: str, window_size: int = 250,
                      equalize: bool = False) -> dict[str, Fiducial]:
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
            min_value = np.percentile(fiducial.flatten(), 10)
            #max_value = np.percentile(fiducial.flatten(), 90)
            # fiducial = np.clip((fiducial - min_value) * (255 / (max_value - min_value)),
            #                   a_min=0, a_max=255).astype(fiducial.dtype)
            fiducial = cv2.threshold(
                src=np.clip(fiducial - min_value, a_min=0, a_max=255),
                thresh=40,
                maxval=255,
                type=cv2.THRESH_BINARY
            )[1].astype(fiducial.dtype)

        # Make a new fiducial instance and add it to the output dictionary
        fiducials[corner] = Fiducial(
            corner=corner,
            fiducial_image=fiducial,
            bounds=fiducial_bounds,
            center=(fiducial_coords[corner])
        )

    return fiducials


def get_fiducial_templates(transforms: ImageTransforms, frame_type: str,
                           instrument: str, cache=True) -> dict[str, Fiducial]:
    """
    Construct or load fiducial templates from every image with manually picked fiducial marks.

    :param transforms: Manual transforms to align the images with.
    :param frame_type: The frame type of the images.
    :param cache: Whether to load and save cached results(this may take a while).

    :returns: A dictionary of fiducial templates(Fiducial objects) for every corner.
    """
    # Define the filepath to look for / save / ignore the cache file
    cache_filepath = os.path.join(CACHE_FILES["fiducial_template_dir"], instrument + ".pkl")

    # Return a cached version if it exists and should be used.
    if cache and os.path.isfile(cache_filepath):
        with open(cache_filepath, "rb") as infile:
            return pickle.load(infile)

    print(f"Generating new fiducial templates for {instrument}")
    # Generate a reference frame from the manually picked fiducial positions.
    reference_frame = generate_reference_frame(transforms)

    # Extract fiducials in predefined places from the frame
    fiducials = extract_fiducials(reference_frame, frame_type=frame_type, window_size=250)

    if cache:
        os.makedirs(CACHE_FILES["fiducial_template_dir"], exist_ok=True)
        # Save the fiducials object
        with open(cache_filepath, mode="wb") as outfile:
            pickle.dump(fiducials, outfile)

        # Save previews of the templates for debugging/visualisation purposes
        preview_dir = os.path.join(CACHE_FILES["fiducial_template_dir"], "previews")
        os.makedirs(preview_dir, exist_ok=True)
        for corner in fiducials:
            cv2.imwrite(os.path.join(preview_dir,
                                     f"{instrument}_{corner}_preview.jpg"), fiducials[corner].fiducial_image)

        frame_dir = os.path.join(CACHE_FILES["fiducial_template_dir"], "image_frames")
        os.makedirs(frame_dir, exist_ok=True)
        cv2.imwrite(os.path.join(frame_dir, f"frame_{instrument}.tif"), reference_frame)

    return fiducials


def match_templates(filenames: list[str], frame_type: str, instrument: str, templates: dict[str, Fiducial],
                    cache: bool = True) -> ImageTransforms:
    """
    Use template matching to match fiducials in images, compared to the provided fiducial templates.

    :param filenames: A list of filenames to analyse.
    :param frame_type: The frame type of the images.
    :param templates: Fiducial templates to try to match.
    :param cache: Whether to load and save cached results(this may take a while).
    :returns: Transforms estimated from feature matching, with corresponding uncertainties.
    """
    # Define the filepath to look for / save / ignore the cache file
    cache_filepath = os.path.join(TEMP_DIRECTORY, "transforms", instrument + ".pkl")
    # Return a cached version if it exists and should be used
    if cache and os.path.isfile(cache_filepath):
        with open(cache_filepath, "rb") as infile:
            return pickle.load(infile)

    print(f"Generating new template transforms for {instrument}")

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
            offset, uncertainty = fiducials[corner].match(templates[corner])
            new_position = (templates[corner].center[0] + offset[0], templates[corner].center[1] + offset[1])
            original_positions.append(templates[corner].center)
            new_positions.append(new_position)
            uncertainties.append(uncertainty)

        # Estimate a transform using the x/y-swapped source and destination positions.
        # Some warnings show up when RANSAC gets too few inliers, but it will in a different iteration.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transform, inliers = skimage.measure.ransac(
                data=(
                    np.array(new_positions)[:, ::-1],
                    np.array(original_positions)[:, ::-1]
                ),
                model_class=skimage.transform.EuclideanTransform,
                min_samples=3,  # At least three points should be used
                stop_sample_num=1,  # If it finds one good outlier, it stops.
                residual_threshold=CONSTANTS.transform_outlier_threshold  # A threshold of 5 px is the outlier threshold
            )
            # If a transform was not possible to estimate using RANSAC (probably due to more than one outlier),
            # estimate one anyway. This will have large residuals which can later be excluded.
            if np.any(np.isnan(transform.params)):
                transform = skimage.transform.estimate_transform(
                    ttype="euclidean",
                    src=np.array(new_positions)[:, ::-1],
                    dst=np.array(original_positions)[:, ::-1]
                )
                inliers = np.array([True, True, True, True])

        # Calculate residuals by subtracting the actual estimated positions with those made from the transform
        # These should be close to zero if the fiducials were marked perfectly
        residuals = np.linalg.norm(np.array(new_positions) -
                                   transform.inverse(np.array(original_positions)[:, ::-1])[:, ::-1], axis=1)
        residuals[~inliers] = np.nan  # pylint: disable=invalid-unary-operand-type
        assert residuals.shape == (4,), f"Residuals have weird shape: {residuals.shape}"
        assert ~np.any(np.isnan(transform.params)), f"Params for {filename} are NaNs"
        progress_bar.update()
        return transform, residuals

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
    template_transforms.add_estimated_residuals(uncertainties)

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


def merge_manual_and_estimated_transforms(manual_transforms: ImageTransforms,
                                          estimated_transforms: ImageTransforms) -> ImageTransforms:
    """
    Substitute poorly estimated transforms with manual ones (if they exist).

    :param manual_transforms: Transforms from the manually picked fiducials.
    :param estimated_transforms: Transforms from the estimated fiducials.
    :returns: A merged transforms object with substituted poor estimated transforms.
    """
    # Make sure that the manual and estimated transforms have the same frame type
    if manual_transforms.frame_type != estimated_transforms.frame_type:
        raise ValueError("Estimated and manual transforms have differing frame types")

    # Get the filenames that were estimated poorly
    estimated_outliers = estimated_transforms.get_outlier_filenames()

    # Make a new ImageTransforms object which will be filled further down
    merged_transforms = ImageTransforms(frame_type=estimated_transforms.frame_type)

    # Make arrays of manual residuals and estimated uncertainties.
    # They will respecively be NaNs if they are estimated/manual
    merged_transforms.manual_residuals = np.empty(shape=(len(estimated_transforms.keys()), 4))
    merged_transforms.manual_residuals[:] = np.nan
    merged_transforms.estimated_residuals = merged_transforms.manual_residuals.copy()

    # Generate a list of filenames that can replace the estimated if needed
    manually_picked_filenames: list[str] = manual_transforms.keys()

    # Loop over all filenames and add the appropriate version (manual/estimated) to merged_transforms
    for i, filename in enumerate(estimated_transforms.keys()):
        # Check if the manual transform should be added
        if filename in estimated_outliers and filename in manually_picked_filenames:
            merged_transforms[filename] = manual_transforms[filename]
            merged_transforms.manual_residuals[i, :] = manual_transforms.manual_residuals[
                manually_picked_filenames.index(filename), :]  # type: ignore
        else:
            merged_transforms[filename] = estimated_transforms[filename]
            merged_transforms.estimated_residuals[i, :] = estimated_transforms.estimated_residuals[i, :]  # type: ignore

    # If no manual transforms were used, this should be reflected in the dataset
    if np.all(np.isnan(merged_transforms.manual_residuals)):
        merged_transforms.manual_residuals = None

    return merged_transforms


def compare_transforms(reference_transforms: ImageTransforms, compared_transforms: ImageTransforms) -> float:
    """
    Compare the RMS of the residuals between fiducials (in simulated positions) from two transform objects.

    The reference transforms are assumed to not contain outliers.

    :param reference_transforms: Transforms without outliers.
    :param compared_transforms: Transforms with possible outliers (which will be excluded)
    :returns: The RMS of the residuals.
    """
    # Get arbitrary coordinates to compare the reference vs. compared transforms with
    fiducial_positions = CONSTANTS.wild_fiducial_locations if "wild" in compared_transforms.frame_type \
        else CONSTANTS.zeiss_fiducial_locations
    # Convert the coordinates to an array and invert the second axis to have x/y order instead of y/x
    start_coords = np.array(list(fiducial_positions.values()))[:, ::-1]

    compared_filenames = np.array(compared_transforms.keys())
    # Find the filenames that should be compared
    valid_filenames = compared_filenames[~np.isin(compared_filenames, np.unique(np.r_[
        compared_transforms.get_outlier_filenames(),  # They are estimated outliers.
        compared_filenames[~np.isin(compared_filenames, reference_transforms.keys())]  # They exist in both transforms.
    ]))]

    rmses: list[float] = []
    for filename in valid_filenames:
        reference_points = reference_transforms[filename](start_coords)
        compared_points = compared_transforms[filename](start_coords)

        rmse = np.sqrt(np.mean(np.square(np.linalg.norm(compared_points - reference_points, axis=1))))
        rmses.append(rmse)

    return np.mean(rmses)


def get_all_instrument_transforms(complement: bool = False, verbose: bool = True) -> ImageTransforms:
    """
    Loop through each of the instruments and fetch/estimate image transforms.

    :param complement: Look for outliers in the estimation and start the marking GUI to complement them.
    :param verbose: Print statistics about the transforms.
    :returns: A transforms object of each estimated/manually substituted image transform.
    """
    # Load the manually marked fiducials.
    marked_fiducials = manual_picking.MarkedFiducials.read_csv(files.INPUT_FILES["marked_fiducials"])
    # Find which instruments were marked and what frame type they correspond to.
    instruments = marked_fiducials.get_instrument_frame_type_relation()

    # Find the corresponding filenames for each instrument.
    instrument_filenames = {
        instrument: image_meta.get_filenames_for_instrument(
            instrument
        ) for instrument in instruments
    }
    # Calculate manual transforms from the marked fiducials.
    manual_transforms = {
        instrument: calculate_manual_transforms(
            marked_fiducials.subset(instrument_filenames[instrument]),
            frame_type=instruments[instrument]
        ) for instrument in instruments
    }
    # Extract fiducial templates from each instrument to use with the feature matching.
    fiducial_templates = {
        instrument: get_fiducial_templates(
            transforms=manual_transforms[instrument],
            frame_type=instruments[instrument],
            instrument=instrument,
            cache=True
        ) for instrument in instruments
    }

    estimated_transforms = {
        instrument: match_templates(
            filenames=instrument_filenames[instrument],
            frame_type=instruments[instrument],
            templates=fiducial_templates[instrument],
            instrument=instrument,
            cache=True
        ) for instrument in instruments
    }

    # TEMPORARY (11/12/2020)
    # Only needed for the currently cached transforms.
    # for instrument in instruments:
    #    estimated_transforms[instrument].estimated_residuals = estimated_transforms[instrument].estimation_uncertainties

    merged_transforms = {
        instrument: merge_manual_and_estimated_transforms(
            manual_transforms[instrument],
            estimated_transforms[instrument]
        ) for instrument in instruments
    }

    if verbose:
        rmses: list[float] = []
        for instrument in instruments:
            rms = compare_transforms(manual_transforms[instrument], estimated_transforms[instrument])
            rmses.append(rms)
            print(f"{instrument} manual-to-estimated error: {rms:.2f} px")
        print(f"\nMean manual-to-estimated error: {np.mean(rmses):.2f} px\n")

        for key in merged_transforms:
            print(key, "\t\t", merged_transforms[key])

        print("\nLooking for outliers")
        total_outliers = 0
        for frame_type in merged_transforms:
            n_outliers = len(merged_transforms[frame_type].get_outlier_filenames())
            if n_outliers == 0:
                continue
            print(f"\t{frame_type} has {n_outliers} outlier(s)")
            total_outliers += n_outliers

        if total_outliers != 0:
            print(f"There are {total_outliers} outlier(s) in total")

    if complement:
        for instrument in instruments:
            outliers = merged_transforms[instrument].get_outlier_filenames()

            if len(outliers) == 0:
                continue
            manual_picking.gui(instrument_type=instruments[instrument],
                               filenames=outliers)

    return ImageTransforms.from_multiple(list(merged_transforms.values()))


def save_fiducial_locations(image_transforms: ImageTransforms, instrument: str) -> str:
    """
    Save the estimated/marked fiducial locations to a CSV.

    :param image_transforms: An unsorted collection of image transforms.
    :param instrument: The instrument to save the fiducial locations from.
    :returns: The path to the CSV with the name template: "fiducials_{instrument}.csv"
    """
    # Get the filenames of the images that should be processed.
    filenames = image_meta.get_filenames_for_instrument(instrument)
    # Find the median location of the fiducials
    median_fiducial_locations = get_median_fiducial_locations(
        manual_picking.MarkedFiducials.read_csv(files.INPUT_FILES["marked_fiducials"]).subset(filenames)
    )
    # Create a header list of the filename and x/y coordinates in px and mm for all corners.
    # The header should have an item length of 9
    data_header: list[str] = ["filename"] + \
        list(np.array([[f"{corner}_x", f"{corner}_y"] for corner in CORNERS]).flatten()) + \
        list(np.array([[f"{corner}_x_mm", f"{corner}_y_mm"] for corner in CORNERS]).flatten())

    # Find the center of the image using the mean of the top and bottom fiducials
    # Left and right on the Zeiss has the same approximate Y-coordinate as top, hence why only top and bottom are used.
    top_and_bottom = [CORNERS.index("top"), CORNERS.index("bottom")]
    center = np.mean(median_fiducial_locations[top_and_bottom], axis=0)

    # Calculate the locations in mm from the centre of the image using the scanning resolution as a converter
    fiducial_locations_mm = (median_fiducial_locations - center) * CONSTANTS.scanning_resolution * 1e3
    # Switch the axis on the Y-coordinate so the top fiducial has positive millimeters and the bottom has negative.
    fiducial_locations_mm[:, 0] *= -1

    # Create a list to warn about possible missing filenames
    missing_filenames: list[str] = []
    # Create a list of lists to iteratively fill
    data_rows: list[list[Union[str, float]]] = []
    for filename in filenames:
        row: list[Union[str, float]] = [filename]
        if filename not in image_transforms:
            missing_filenames.append(filename)
            continue
        # Loop over all corners and mark their transformed coordinates. TODO: Look up if it should be the inverse.
        for corner_coord in median_fiducial_locations:
            x_coord, y_coord = image_transforms[filename](corner_coord[::-1])[0]
            row += [x_coord, y_coord]

        row += list(fiducial_locations_mm[:, ::-1].flatten())
        data_rows.append(row)

    if len(missing_filenames) > 0:
        warnings.warn(f"{len(missing_filenames)} missing when exporting fiducial locations" +
                      ", ".join(missing_filenames))

    # Write the CSV.
    output_filepath = os.path.join(CACHE_FILES["fiducial_location_dir"], f"fiducials_{instrument}.csv")
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data_header)
        writer.writerows(data_rows)

    return output_filepath


def save_all_fiducial_locations(image_transforms: ImageTransforms) -> dict[str, str]:
    """Write CSVs for all instruments in the image_transforms collection."""
    marked_fiducials = manual_picking.MarkedFiducials.read_csv(files.INPUT_FILES["marked_fiducials"])
    # Find which instruments were marked and what frame type they correspond to.
    instruments = marked_fiducials.get_instrument_frame_type_relation()

    # Save the fiducial locations for each instrument
    filepaths = {instrument: save_fiducial_locations(image_transforms, instrument) for instrument in instruments}

    return filepaths


if __name__ == "__main__":
    transforms = get_all_instrument_transforms(complement=False, verbose=False)

    save_all_fiducial_locations(transforms)
