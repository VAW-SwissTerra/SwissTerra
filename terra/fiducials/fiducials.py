#!/bin/env python
"""Aligns the frames of each input image compared to one reference image, based on feature matching to the reference."""

import concurrent.futures
import os
import pickle
import shutil
import subprocess
import warnings
from collections import namedtuple
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.measure
import skimage.transform
import statictypes
from tqdm import tqdm

from terra import files

TEMP_DIRECTORY = os.path.join(files.TEMP_DIRECTORY, "fiducials")

CACHE_FILES = {
    "fiducial_template_dir":  os.path.join(TEMP_DIRECTORY, "fiducial_templates"),
    "transforms_dir": os.path.join(TEMP_DIRECTORY, "transforms"),
    "transformed_image_dir": os.path.join(TEMP_DIRECTORY, "transformed_images"),
}


# TODO: Fix weird estimate bug

@statictypes.enforce
def get_reference_fiducials(file_path: str = files.INPUT_FILES["manual_fiducials"]) -> pd.DataFrame:
    """
    Read reference fiducial placements from an exported Metashape file.

    param: file_path: Path to the manual fiducial placements.

    return reference_fiducials: A Pandas DataFrame with the read fiducial positions.
    """
    # Read the reference fiducials marked in Metashape
    reference_fiducials = pd.read_csv(file_path)
    # Add extension since Metashape tends to remove them
    reference_fiducials["camera"] += ".tif"

    return reference_fiducials


@statictypes.enforce
def get_reference_transforms(reference_fiducials: pd.DataFrame, verbose: bool = True) -> Dict[str, skimage.transform.EuclideanTransform]:
    """
    Find the average position of each fiducial and return the transform for each image to put it there.

    This function can probably be done MUCH easier. It works for now, however!

    param: reference_fiducials: Fiducial markers exported from Metashape.
    return: matrices: An skimage transform matrix for each separate filename.
    """
    corrected_fiducials = pd.DataFrame()  # Instantiate a dataframe

    # Find the rotation of the image based on the angle between the left-right and top-bottom fiducial pairs
    # Then, rotate the fiducials and put them in the corrected_fiducials dataframe
    for filename, grouped_df in reference_fiducials.groupby("camera"):
        fiducials = grouped_df.set_index("fiducial")  # For easier indexing

        # Find the horizontal (left-right) fiducial angle
        # using the inverse tangent of the y offset divied by the x offset
        ydiff_hor = fiducials.loc[2, "y_px"] - fiducials.loc[4, "y_px"]
        xdiff_hor = fiducials.loc[2, "x_px"] - fiducials.loc[4, "x_px"]
        angle_hor = np.arctan(float(ydiff_hor) / float(xdiff_hor))

        # Do the same for the vertical (top-bottom) fiducials
        ydiff_vert = fiducials.loc[1, "y_px"] - fiducials.loc[3, "y_px"]
        xdiff_vert = fiducials.loc[1, "x_px"] - fiducials.loc[3, "x_px"]
        angle_vert = -np.arctan(float(xdiff_vert) / float(ydiff_vert))

        # Construct a transform object from the mean of these two angles
        matrix = skimage.transform.EuclideanTransform(rotation=np.mean([angle_hor, angle_vert]))
        # Rotate the coordinates in the temporary "grouped_df" dataframe
        grouped_df[["x_px", "y_px"]] = matrix.inverse(grouped_df[["x_px", "y_px"]])

        # Append these to the corrected_fiducials dataframe
        corrected_fiducials = pd.concat([corrected_fiducials, grouped_df], ignore_index=True)

    # Find the average centre of the fiducials
    # Tts exact point is irrelevant. It's just a common point to align them all to.
    mean_x = reference_fiducials["x_px"].mean()
    mean_y = reference_fiducials["y_px"].mean()

    # Go through each of the (rotationally) corrected fiducials and subtract the offset to the mean center point
    # This means that all fiducial pairs are aligned as well as they can be
    for filename, grouped_df in corrected_fiducials.groupby("camera"):
        corrected_fiducials.loc[corrected_fiducials["camera"] == filename, "x_px"] -= grouped_df["x_px"].mean() - mean_x
        corrected_fiducials.loc[corrected_fiducials["camera"] == filename, "y_px"] -= grouped_df["y_px"].mean() - mean_y

    # Calculate errors based on the corrected fiducial offsets to each other
    # Make a nicer looking error estimation
    errors = []
    for fiducial in np.unique(corrected_fiducials["fiducial"]):
        vals = corrected_fiducials[corrected_fiducials["fiducial"] == fiducial]
        x_errors = vals["x_px"] - vals["x_px"].mean()
        y_errors = vals["y_px"] - vals["y_px"].mean()

        error = np.sqrt(np.mean(np.square(np.linalg.norm([x_errors, y_errors], axis=0))))
        errors.append(error)
    if verbose:
        print(f"Fiducial correction RMSE: {round(np.mean(errors), 3)} px")

    matrices = {}
    # Go through each filename and estimate a Euclidean transform matrix between the original and corrected fiducials
    for filename in np.unique(reference_fiducials["camera"]):
        source_fiducials = reference_fiducials[reference_fiducials["camera"] == filename]
        destination_fiducials = corrected_fiducials[corrected_fiducials["camera"] == filename]

        source_coords = source_fiducials[["x_px", "y_px"]].values
        destination_coords = destination_fiducials[["x_px", "y_px"]].values

        # Find the transform without outlier removal
        matrix = skimage.transform.estimate_transform("euclidean", source_coords, destination_coords)
        matrices[filename] = matrix

    return matrices


def find_frame_luminance_threshold(img: np.ndarray) -> int:
    """
    Find the most frequent luminance value of the frame.

    The most frequent values in the image histograms are either the dark frame values or the bright sky.
    By looking only at the darker parts of the histogram, the most common value seems to always be the frame.

    param: img: The input image

    return: frame_luminance: The most common luminance of the frame.
    """
    # It seems like the frame rarely has a luminance above 50. To be safe, a threshold of 80 is set.
    max_likely_luminance = 100

    values_to_consider = img.flatten()[img.flatten() < max_likely_luminance]
    values_to_consider = values_to_consider[values_to_consider != 0]  # Remove the values that are from transformation
    # Extract the values that are below the max_likely_luminance and get the most frequent value
    frame_luminance = np.bincount(values_to_consider).argmax()
    return frame_luminance


class FrameMatcher:
    """Class to train and run image frame matching and correction."""

    def __init__(self, image_folder: str = files.INPUT_DIRECTORIES["image_dir"], orb_reference_filename: str = "000-175-212.tif", frame_luminance_extra: int = 30,
                 max_orb_features: int = 6000, orb_patch_size: int = 100, max_image_offset: float = 300.0,
                 ransac_reprojection_threshold: float = 8.0, ransac_min_samples: int = 100,
                 ransac_max_trials: int = 1000, template_size: int = 500, max_template_diff: int = 10,
                 verbose: bool = True, cache: bool = True):
        """
        Match image frames using a multitude of methods.

        ARGUMENTS
        param: image_folder: Where the source images are.
        param: orb_reference_filename: What image to detect reference features from (ORB).

        KEYWORD ARGUMENTS
        ========== FOR PREPROCESSING ===========
        param: frame_luminance_extra: What luminance value above the most common frame value to still consider a frame.

        ========== FOR ORB MATCHING ============
        param: max_orb_features: The maximum features to detect on a frame using ORB.
        param: orb_patch_size: The patch size of the ORB algorithm.
        param: max_image_offset: The maximum offset to attempt a correction for (to remove erroneous outliers).
        param: ransac_reprojection_threshold: The residual threshold to consider an outlier or not.
        param: ransac_max_trials: The maximum amount of iterations to find a solution.

        ========= FOR TEMPLATE MATCHING =========
        param: template_size: The size in pixels of the templates to generate.
        param: max_template_diff: The maximum difference in pixels to look for change.

        ========== EXTRA ============
        param: verbose: Whether to print progress or not.
        param: cache: Whether to cache the results or not (it takes a long time!)
        """
        self.verbose = verbose
        self.cache = cache

        self.frame_luminance_extra = 10  # TODO: Evaluate if this should be configurable
        # Get all files with valid extensions
        self.filenames: List[str] = []
        for filename in os.listdir(image_folder):
            # Exclude the image if it's the reference image
            if filename == orb_reference_filename:
                continue
            # If file or folder doesn't have at least one . in its filename
            if len(filename.split(".")) < 2:
                continue
            suffix = filename.split(".")[-1]
            if suffix in ["tif", "jpg", "jpeg", "tiff"]:
                self.filenames.append(filename)

        # Instantiate the ORB and BFMatchers
        self.orb = cv2.ORB_create(nfeatures=max_orb_features, patchSize=orb_patch_size)
        self.feature_matcher = cv2.BFMatcher_create(normType=cv2.NORM_HAMMING, crossCheck=True)

        # Get the manual fiducials and calculate their equivalent transforms.
        self.manual_fiducials = get_reference_fiducials()
        if not os.path.isfile(os.path.join(CACHE_FILES["transforms_dir"], "manual_transforms.pkl")) or not self.cache:
            self.manual_transforms = get_reference_transforms(self.manual_fiducials, verbose=self.verbose)
            if self.cache:
                self.save_transforms("manual_transforms.pkl", self.manual_transforms)
        else:
            if self.verbose:
                print("Loading cached manual transforms")
            self.manual_transforms = self.load_transforms("manual_transforms.pkl")

        # Load the reference image and extract its frame
        self.orb_reference_filename = orb_reference_filename
        reference_frame = self.extract_image_frame(self.read_image(orb_reference_filename))
        manual_transform = self.manual_transforms[orb_reference_filename]
        corrected_reference_frame = self.transform_image(image=reference_frame,
                                                         transform=self.invert_transform(manual_transform),
                                                         output_shape=reference_frame.shape)
        self.reference_frame = corrected_reference_frame

        # Variables for reference features from the reference frame
        self.reference_keypoints: Optional[List[cv2.KeyPoint]] = None
        self.reference_descriptors: Optional[np.ndarray] = None

        for folder in [CACHE_FILES["fiducial_template_dir"], CACHE_FILES["transforms_dir"],
                       CACHE_FILES["transformed_image_dir"]]:
            os.makedirs(folder, exist_ok=True)

        # The approximate coordinates of the fiducial centres after homogenisation transform (in y,x format)
        self.fiducial_locations = {
            "top": (250, 3500),
            "left": (2450, 250),
            "right": (2350, 6749),
            "bottom": (4370, 3500)
        }

        # Set other attributes
        self.max_orb_features = max_orb_features
        self.orb_patch_size = orb_patch_size
        self.template_size = template_size
        self.max_image_offset = max_image_offset
        self.ransac_reprojection_threshold = ransac_reprojection_threshold
        self.ransac_min_samples = ransac_min_samples
        self.ransac_max_trials = ransac_max_trials
        self.max_template_diff = max_template_diff
        self.progress_bar: Optional[tqdm] = None
        # Allow only approximately 16 Gb of files to be open at once
        self.max_open_files = 16

        # If multiple transformations are made in order, this is the latest valid one
        self.latest_transforms: Optional[Dict[str, skimage.transform.EuclideanTransform]] = None

    @staticmethod
    @statictypes.enforce
    def invert_transform(transform: skimage.transform.EuclideanTransform) -> skimage.transform.EuclideanTransform:
        """
        Return the inverse of an Skimage transform object.

        This is different from the builtin "transform.inverse" because this returns a proper class, not a method.

        param: transform: The input transform.
        return: inverted_transform: The inverse transform.
        """
        inverse_matrix = transform._inv_matrix

        inverted_transform = skimage.transform.EuclideanTransform(matrix=inverse_matrix)

        return inverted_transform

    @staticmethod
    # @statictypes.enforce  # TODO: Check why this was not allowed to have
    def read_image(filename: str) -> np.ndarray:
        """
        Load an image as grayscale and raise an error if it doesn't exist.

        param: filename: Name of the file (excluding the directory path).

        return: image: The loaded image.
        """
        image_path = os.path.join(files.INPUT_DIRECTORIES["image_dir"], filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:  # OpenCV returns a NoneType if the image doesn't exist.
            raise AttributeError(f"Image {filename} could not be read (filepath: {image_path})")
        return image

    @staticmethod
    @statictypes.enforce
    def transform_image(image: np.ndarray,
                        transform: skimage.transform.EuclideanTransform,
                        output_shape: Tuple[int, int]) -> np.ndarray:
        """
        Transform an image using the specified transform, to the specified output shape (y, x).

        Note that the warp function often wants the inverse transform.

        param: image: The input image to transform.
        param: transform: The Skimage transform object.
        param: output_shape: The shape of the output image (y, x).

        return: transformed_image: A transformed image with the specified shape.
        """
        transformed_image = skimage.transform.warp(image, transform, output_shape=output_shape, preserve_range=True)
        return transformed_image.astype(image.dtype)

    @statictypes.enforce
    def save_transforms(self, filename: str, transforms: Dict[str, skimage.transform.EuclideanTransform]) -> None:
        """
        Save (cache) a transform dictionary as a pickle.

        param: filename: The filename (excluding directory path) of the output file.
        param: transforms: The object to be pickled.

        return: None.
        """
        if not os.path.isdir(CACHE_FILES["transforms_dir"]):
            os.makedirs(CACHE_FILES["transforms_dir"])
        with open(os.path.join(CACHE_FILES["transforms_dir"], filename), "wb") as file:
            pickle.dump(transforms, file)

    @statictypes.enforce
    def load_transforms(self, filename: str) -> Dict[str, skimage.transform.EuclideanTransform]:
        """
        Load a pickled transform object.

        param: filename: The filename (excluding directory path) of the input file.

        return: transforms: The previously pickled transform dictionary.
        """
        with open(os.path.join(CACHE_FILES["transforms_dir"], filename), "rb") as file:
            transforms = pickle.load(file)
        return transforms

    @statictypes.enforce
    def extract_image_frame(self, image: np.ndarray) -> np.ndarray:
        """
        Extract the frame around an image using luminance thresholding.

        param: image: The input grayscale image.

        return: image_frame: The thresholded image.
        """
        # It seems like the frame rarely has a luminance above 50. To be safe, a threshold of 100 is set.
        # If an upper limit is not set, sometimes the most common value is that of the sky
        max_likely_luminance = 100

        values_to_consider = image.flatten()[image.flatten() < max_likely_luminance]
        # Empty values after a transformation is exactly 0, so these should not be considered
        values_to_consider = values_to_consider[values_to_consider != 0]
        # Extract the values that are below the max_likely_luminance and get the most frequent value
        frame_luminance = np.bincount(values_to_consider).argmax()
        _, image_frame = cv2.threshold(image, frame_luminance + self.frame_luminance_extra, 255, cv2.THRESH_BINARY)

        return image_frame

    @statictypes.enforce
    def get_orb_transform(self, filename: str) -> skimage.transform.EuclideanTransform:
        """
        Find the most likely transform of an image using ORB feature matching.

        param: filename: The input filename (excluding its directory).

        return: transform: An Skimage transform object.
        """
        image = self.read_image(filename)
        frame = self.extract_image_frame(image)

        keypoints, descriptors = self.orb.detectAndCompute(frame, None)
        matches = self.feature_matcher.match(self.reference_descriptors, descriptors)

        # Extract the 2D coordinates of the matches
        # src_pts is where they were originally, dst_pts is where they should be according to the reference
        src_pts = np.array([self.reference_keypoints[match.queryIdx].pt for match in matches])
        dst_pts = np.array([keypoints[match.trainIdx].pt for match in matches])

        # Extract only the points whose matches indicate an offset within a reasonable distance.
        point_diff = np.linalg.norm(np.abs(src_pts - dst_pts), axis=1)
        src_pts_filtered = src_pts[point_diff < self.max_image_offset]
        dst_pts_filtered = dst_pts[point_diff < self.max_image_offset]

        # Calculate the most likely Euclidean transform using RANSAC filtering.
        transform, _ = skimage.measure.ransac((src_pts_filtered, dst_pts_filtered),
                                              skimage.transform.EuclideanTransform,
                                              min_samples=self.ransac_min_samples,
                                              residual_threshold=self.ransac_reprojection_threshold,
                                              max_trials=self.ransac_max_trials)
        if self.progress_bar is not None:
            self.progress_bar.update(1)
        return transform

    @statictypes.enforce
    def get_orb_transforms(self) -> Dict[str, skimage.transform.EuclideanTransform]:
        """
        Use ORB to detect features in images and match them to the reference image.

        Using RANSAC, a Euclidean transform matrix is estimated from this.

        return: transforms: Dictionary of transforms for each filename.

        """
        self.reference_keypoints, self.reference_descriptors = self.orb.detectAndCompute(self.reference_frame, None)
        transforms = {}
        if self.verbose:
            print("Extracting ORB features, matching them, and estimating transform")
        # Loop over all images and extract their transforms.
        filenames_without_reference = []
        for filename in self.filenames:
            if filename != self.orb_reference_filename:
                filenames_without_reference.append(filename)

        with tqdm(total=len(filenames_without_reference), disable=(not self.verbose)) as self.progress_bar:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(self.get_orb_transform, filenames_without_reference)

        transforms = dict(zip(filenames_without_reference, results))

        if self.cache:
            self.save_transforms("orb_transforms.pkl", transforms)

        return transforms

    @staticmethod
    @statictypes.enforce
    def extract_fiducial(image: np.ndarray, x_center: int, y_center: int, window_size: int) -> np.ndarray:
        """
        Extract a fiducial at specific coordinates.

        The black-level is also corrected for (frame should be at zero).

        param: image: The input grayscale image.
        param: x_center: The center x coordinate of the fiducial.
        param: y_center: The center y coordinate of the fiducial.
        param: window_size: The output (square) shape of the image.

        return: fiducial: The cropped image showing one fiducial.
        """
        # Get the corner coordinates of the image subset to be extracted
        top = y_center - window_size // 2
        bottom = y_center + window_size // 2
        left = x_center - window_size // 2
        right = x_center + window_size // 2

        # Check that bounds are within the shape of the image
        if bottom + 1 > image.shape[0]:  # If the top coordinate is larger than the image
            excess = image.shape[0] - (bottom + 1)
            top -= excess
            bottom -= excess
        elif top < 0:  # If the bottom coordinate is negative
            bottom -= top
            top = 0
        if right + 1 > image.shape[1]:  # If the right coordinate is larger than the image
            excess = image.shape[1] - (right + 1)
            right -= excess
            left -= excess
        elif left < 0:  # If the left coordinate is negative
            right -= left
            left = 0

        # Extract the fiducial and set its dtype to a signed integer (for luminance correction)
        fiducial = image[top:bottom, left:right].astype(np.int16)
        # Set the luminance of the frame as zero
        fiducial -= find_frame_luminance_threshold(fiducial)
        # Fix negative values
        fiducial[fiducial < 0] = 0

        return fiducial.astype(np.uint8)

    # @statictypes.enforce
    def extract_all_fiducials(self, filename: str) -> Optional[List[np.ndarray]]:
        """
        Extract fiducials from all corners of an image.

        param: filename: The filename of the input image.

        return: fiducials: A list of the extracted fiducials, or None if extraction was unsuccessful.
        """
        image = self.read_image(filename)
        if filename not in self.manual_transforms.keys():
            return None
        transformed_image = self.transform_image(image=image, transform=self.invert_transform(self.manual_transforms[filename]),
                                                 output_shape=self.reference_frame.shape)
        fiducials = []
        # Extract each fiducial from the opened image using predefined fiducial coordinates
        for corner in self.fiducial_locations.keys():
            y_center, x_center = self.fiducial_locations[corner]
            fiducial = self.extract_fiducial(image=transformed_image, x_center=x_center,
                                             y_center=y_center, window_size=self.template_size)
            fiducials.append(fiducial)

        if self.progress_bar is not None:
            self.progress_bar.update(1)
        return fiducials

    @statictypes.enforce
    def generate_fiducial_templates(self) -> Dict[str, np.ndarray]:
        """
        Loop through all images (aligned with reference transforms) and acquire the median fiducial shapes.

        The output images are thresholded (binary values, in 8bit format).

        return: templates: A dictionary (top, right, bottom, left) with the fiducial templates.
        """
        if not os.path.isdir(CACHE_FILES["fiducial_template_dir"]) and self.cache:
            os.makedirs(CACHE_FILES["fiducial_template_dir"])

        # Instantiate an empty list of fiducials for each corner (corner == fiducial right now)
        fiducials = {corner: [] for corner in self.fiducial_locations}
        if self.verbose:
            print("Generating fiducial templates")

        # Extract all fiducials from every image using multithreading
        with tqdm(total=len(self.filenames), disable=(not self.verbose)) as self.progress_bar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_open_files) as executor:
                results = executor.map(self.extract_all_fiducials, self.filenames)

        # Sort the results by corner
        for image_fiducials in results:
            if image_fiducials is None:
                continue
            for i, corner in enumerate(fiducials.keys()):
                fiducials[corner].append(image_fiducials[i])

        templates: Dict[str, np.ndarray] = {}
        for corner, fiducial_list in fiducials.items():
            # The median of all images seems to be the best approach to get the most common shape.
            median_fiducial = np.median(np.dstack(fiducial_list), axis=2)
            # The template is just a thresholded version of this.
            template = cv2.threshold(median_fiducial, 30, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

            templates[corner] = template
            if self.cache:
                cv2.imwrite(os.path.join(CACHE_FILES["fiducial_template_dir"],
                                         f"fiducial_template_{corner}.tif"), template)

        return templates

    @statictypes.enforce
    def load_fiducial_templates(self, generate_if_not_existing: bool = True) -> Dict[str, np.ndarray]:
        """
        Read cached fiducial templates.

        param: generate_if_not_existing: Generate new templates if cached templates cannot be found.

        return: templates: A dictionary (top, right, bottom, left) with the fiducial templates.
        """
        corners = list(self.fiducial_locations.keys())

        templates = {}
        for corner in corners:
            filepath = os.path.join(CACHE_FILES["fiducial_template_dir"], f"fiducial_template_{corner}.tif")
            template = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if template is None:
                # If it should generate new ones, do so
                if generate_if_not_existing:
                    templates = self.generate_fiducial_templates()
                    return templates
                raise ValueError(f"Template {filepath} doesn't exist!")
            templates[corner] = template

        return templates

    def get_template_transform(self, filename: str) -> skimage.transform.EuclideanTransform:
        """
        Use template matching to estimate an image transform to the reference.

        param: filename: The name of the file (excluding its directory) to be processed.

        return: transform: The output transform object.
        """
        original_image = self.read_image(filename)
        templates = self.load_fiducial_templates()

        # Transform the image to the latest estimated transform.
        # Template matching is usually run after ORB, so the ORB transforms will be the latest ones.
        if self.latest_transforms is not None:
            image = self.transform_image(image=original_image,
                                         transform=self.latest_transforms[filename],
                                         output_shape=self.reference_frame.shape)
        else:
            image = original_image

        # Get source coordinates in [[x, y], [x, y] etc.]
        # The coordinates themselves are arbitrary, and are just used to estimate a Euclidean transform later
        source_coords = np.array(list(self.fiducial_locations.values()))[:, [1, 0]]
        # These will be modified soon
        destination_coords = source_coords.copy()

        # Loop over all corners (corner == fiducial right now) individually
        for i, corner in enumerate(self.fiducial_locations.keys()):
            # x_center, y_center = source_coords[i, :].astype(int)
            y_center, x_center = self.fiducial_locations[corner]
            fiducial_template_sized = self.extract_fiducial(image=image,
                                                            x_center=x_center,
                                                            y_center=y_center,
                                                            window_size=self.template_size)
            fiducial = fiducial_template_sized[self.max_template_diff:self.template_size - self.max_template_diff,
                                               self.max_template_diff:self.template_size - self.max_template_diff]

            _, fiducial_thresholded = cv2.threshold(fiducial, 30, 255, cv2.THRESH_BINARY)

            # Run the template matching algorithm. The output is a field of most likely positions.
            # Note that the template and the fiducial is switched,
            # because the fiducial should be aligned to the template, not vice versa.
            # TODO: Maybe try other matching methods?
            # TODO: Add an uncertainty measure and a limit to remove outliers.
            result = cv2.matchTemplate(image=templates[corner], templ=fiducial_thresholded, method=cv2.TM_CCOEFF)

            # Extract the maximum value in this field (the most likely position)
            _, _, _, max_correlation_location = cv2.minMaxLoc(result)
            # Calculate how many pixels of offset the location corresponds to
            # If the location == self.max_template_diff, there is no movement
            px_to_add_in_y = max_correlation_location[1] - self.max_template_diff
            px_to_add_in_x = max_correlation_location[0] - self.max_template_diff
            # Shift the destination coordinates accordingly
            destination_coords[i, 0] -= px_to_add_in_x
            destination_coords[i, 1] -= px_to_add_in_y

        # Estimate a Euclidean transform without outlier detection (without RANSAC) since it's only for four points.
        transform = skimage.transform.estimate_transform("euclidean", source_coords, destination_coords)

        if self.progress_bar is not None:
            self.progress_bar.update(1)

        return transform

    # @statictypes.enforce TODO: Fix recursive generic types in statictypes so this evaluation works
    def get_template_transforms(self, templates: Optional[Dict[str, np.ndarray]] = None,
                                ) -> skimage.transform.EuclideanTransform:
        """
        Use template matching to calculate image transforms.

        param: templates: Optional templates file (will otherwise read from cache or generate a new one)

        return: transforms: Dictionary of transforms for each filename.

        """
        # Load a cached template or generate a new one if not provided as an argument.
        if templates is None:
            if self.cache:
                templates = self.load_fiducial_templates(generate_if_not_existing=True)
            else:
                templates = self.generate_fiducial_templates()

        if self.verbose:
            print("Matching fiducials")

        # Loop over all images and extract transforms based on feature matching
        with tqdm(total=len(self.filenames), disable=(not self.verbose)) as self.progress_bar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_open_files) as executor:
                results = executor.map(self.get_template_transform, self.filenames)

        transforms = dict(zip(self.filenames, results))

        if self.cache:
            self.save_transforms("template_transforms.pkl", transforms)

        return transforms

    # @statictypes.enforce TODO: Fix the statictypes bug
    def merge_transforms(self, transforms_list: List[Dict[str, skimage.transform.EuclideanTransform]], skip_unavailable: bool = False
                         ) -> Dict[str, skimage.transform.EuclideanTransform]:
        """
        Get the resultant transform from applying each transform in the order given by the input list.

        The output transform will be like applying them separately in the same order.
        TODO: There might be a much simpler way of doing this. I just know that this approach works.

        param: transforms_list: A list of filename:transform dictionaries.
        param: skip_unavailable: Whether or not to skip a key if it does not exist in one of the transform dictionaries.

        return: merged_transforms: A filename:transform dictionary.
        """
        # Get source coordinates in [[x, y], [x, y] etc.]
        # The coordinates themselves are arbitrary, and are just used to estimate a Euclidean transform later
        source_coords = np.array(list(self.fiducial_locations.values()))[:, [1, 0]]

        merged_transforms: Dict[str, skimage.transform.EuclideanTransform] = {}  # Instantiate the empty output variable
        if self.verbose:
            print("Merging transforms")
        # Go through all files (it is assumed that all dictionaries in the lists have the same keys)
        for filename in tqdm(transforms_list[0].keys(), disable=(not self.verbose)):
            if skip_unavailable:
                missing = any([filename in transforms.keys() for transforms in transforms_list])
                if missing:
                    continue
            # These will be modified soon
            destination_coords = source_coords.copy()
            # For each transform, apply it to the destination coordinates in order.
            for transforms in transforms_list:
                transform = transforms[filename]
                destination_coords = transform(destination_coords)

            # Estimate the resultant transform without outlier removal
            merged_transform = skimage.transform.estimate_transform("euclidean", source_coords, destination_coords)
            merged_transforms[filename] = merged_transform

        return merged_transforms

    @statictypes.enforce
    def evaluate_transforms(self, transforms: Dict[str, skimage.transform.EuclideanTransform]) -> float:
        """
        Compare the manually derived transforms with the given transforms and return the RMSE in px.

        param: transforms: A dictionary of filename:transform pairs.

        return: rmse: The RMSE between the manual transforms and the given transforms in px.
        """
        # Get source coordinates in [[x, y], [x, y] etc.]
        # The coordinates are approximately where the fiducials are.
        source_coords = np.array(list(self.fiducial_locations.values()))[:, [1, 0]]

        rmses = []
        for filename, transform in transforms.items():
            # Transform the (approximate) fiducial coordinates manually
            if filename not in self.manual_transforms.keys():
                continue
            manually_transformed_coords = self.manual_transforms[filename](source_coords)
            # Transform the (approximate) fiducial coordinates with the given transform
            transformed_coords = transform.inverse(source_coords)

            # Calculate the RMSE of the coordinate difference in pixels.
            rmse = np.sqrt(np.mean(np.square(np.linalg.norm(manually_transformed_coords - transformed_coords, axis=0))))
            rmses.append(rmse)

        return np.mean(rmses)

    def train_orb_attributes(self, nums=400) -> None:
        """
        Find the best attribute combinations by randomly trying them out.

        Saves the result to *temporary folder*/orb_param_errors.csv

        param: nums: Number of iterations (parameter combinations) to run.
        """
        # Parameter values to try
        params = {
            "max_orb_features": np.linspace(2000, 20000, num=nums, dtype=int),
            "orb_patch_size": np.linspace(10, 200, num=nums, dtype=int),
            "max_image_offset": np.linspace(50, 300, num=nums),
            "ransac_reprojection_threshold": np.linspace(1, 10, num=nums),
            "ransac_min_samples": np.linspace(50, 500, num=nums, dtype=int),
            "ransac_max_trials": np.linspace(200, 1500, num=nums, dtype=int),
        }
        # Randomly shuffle each array
        for arr in params.values():
            np.random.shuffle(arr)

        # Write the header (and overwrite a previous file)
        with open(os.path.join(TEMP_DIRECTORY, "orb_param_errors.csv"), "w") as file:
            file.write("error," + ",".join(list(params.keys())) + "\n")

        # Verbose will temporarily be disabled, so the initial value should be stored.
        was_verbose = self.verbose
        self.verbose = False

        for i in tqdm(range(nums)):
            # Set the parameter values in question
            for key in params:
                self.__setattr__(key, params[key][i])

            self.orb = cv2.ORB_create(nfeatures=self.max_orb_features, patchSize=self.orb_patch_size)
            # Attempt to get a solution (not all combinations succeed)
            try:
                orb_transforms = self.get_orb_transforms()
            except ValueError:
                continue

            # Calculate the resultant error
            error = self.evaluate_transforms(orb_transforms)

            # Format a csv line
            values_to_write = ",".join([error] + [params[key][i] for key in params])

            # Write the line
            with open(os.path.join(TEMP_DIRECTORY, "orb_param_errors.csv"), "a+") as file:
                file.write(values_to_write + "\n")

        # Set the verbose setting back to its default
        self.verbose = was_verbose

    @statictypes.enforce
    def train(self) -> None:
        """Run all algorithms in order on a training dataset."""
        if not self.cache:
            warnings.warn("Caching is disabled during the training stage. This will not save the results.",
                          RuntimeWarning)

        if self.cache:
            try:
                print("Fetching cached ORB transforms")
                orb_transforms = self.load_transforms("orb_transforms.pkl")
            except FileNotFoundError:
                print("Cache fetching failed. Generating new ones")
                orb_transforms = self.get_orb_transforms()

            print("Fetching cached fiducial templates")
            templates = self.load_fiducial_templates(generate_if_not_existing=True)

            self.latest_transforms = orb_transforms
            try:
                print("Fetching cached template transforms")
                template_transforms = self.load_transforms("template_transforms.pkl")
            except FileNotFoundError:
                print("Cache fetching failed. Generating new ones")
                template_transforms = self.get_template_transforms(templates)
        else:
            orb_transforms = self.get_orb_transforms()
            self.latest_transforms = orb_transforms
            templates = self.generate_fiducial_templates()
            template_transforms = self.get_template_transforms(templates=templates)

        merged_transforms = self.merge_transforms([orb_transforms, template_transforms])
        # Add the manually placed reference transform to the output
        merged_transforms[self.orb_reference_filename] = self.invert_transform(
            self.manual_transforms[self.orb_reference_filename])

        if self.cache:
            self.save_transforms("merged_transforms.pkl", merged_transforms)

        if self.verbose:
            orb_error = self.evaluate_transforms(orb_transforms)
            merged_error = self.evaluate_transforms(merged_transforms)

            print(f"""
                  Resulting manual-to-automatic comparison RMSEs:
                  ORB: {round(orb_error, 2)}
                  ORB + template: {round(merged_error, 2)}
                  """)

    @ statictypes.enforce
    def estimate(self, batch_size: int = 500) -> None:
        """
        Estimate frame transforms on all the available images.

        param: batch_size: The amount of pictures to process between saving the results.
        """

        if not self.cache:
            raise AssertionError("Caching is off. Cannot estimate transforms without a cached reference.")

        print("Loading existing cached transforms")
        try:  # Implicitly check if self.train() has been run
            existing_transforms = self.load_transforms("merged_transforms.pkl")
        except FileNotFoundError:
            print("\nNo existing transforms found. Has the matcher been trained?")
            return

        # Get the initial count of transforms (just for status messages)
        initial_count = len(existing_transforms)

        # Copy the full list of filenames and convert it to an array for easier indexing
        all_filenames = np.array(self.filenames)

        # Extract all images that do not already have a transform
        filenames_to_process = all_filenames[~np.isin(
            all_filenames, list(existing_transforms.keys()))]

        if len(filenames_to_process) == 0:
            print("\nAll transforms are already estimated.")
            print(f"{initial_count} transforms exist in the cache.")
            return

        templates = self.load_fiducial_templates(generate_if_not_existing=False)

        indices = np.repeat(np.arange(0, 20000), batch_size)[:len(filenames_to_process)]

        for index in np.arange(indices.min(), indices.max() + 1):
            if indices.max() > 0:
                print(f"Dividing task in batches of {batch_size} frames")
                print(f"Running batch {index + 1} / {indices.max()}")
            filenames_in_batch = filenames_to_process[indices == index]

            self.filenames = filenames_in_batch

            orb_transforms = self.get_orb_transforms()
            self.latest_transforms = orb_transforms

            template_transforms = self.get_template_transforms(templates=templates)

            merged_transforms = self.merge_transforms([orb_transforms, template_transforms], skip_unavailable=False)

            for key in merged_transforms:
                existing_transforms[key] = merged_transforms[key]

            self.save_transforms("merged_transforms.pkl", existing_transforms)

        print(f"Finshed {len(merged_transforms) - initial_count} frame transforms. Total: {len(merged_transforms)}.")

    @ statictypes.enforce
    def transform_images(self, transforms_file: str = "merged_transforms.pkl", output_format: str = "jpg") -> None:
        """
        Transform all images using the provided (cached) transforms file.

        It skips the images whose filenames do not exist in the transforms dictionary (usually the ORB reference).

        param: transforms_file: The pickled cached transforms dictionary.
        param: output_format: The output file extension for the images.

        return: None
        """
        transforms = self.load_transforms(transforms_file)
        # Make the folder if it doesn't already exist
        if not os.path.isdir(CACHE_FILES["transformed_image_dir"]):
            os.makedirs(CACHE_FILES["transformed_image_dir"])

        if self.verbose:
            print("Transforming and saving images")

        def apply_transform(filename):
            # Transform the image using the provided transforms
            transformed_image = self.transform_image(self.read_image(filename),
                                                     transforms[filename],
                                                     self.reference_frame.shape)
            cv2.imwrite(os.path.join(CACHE_FILES["transformed_image_dir"],
                                     filename.replace(".tif", f".{output_format}")),
                        transformed_image)
            self.progress_bar.update()

        with tqdm(total=len(transforms.keys()), disable=(not self.verbose)) as self.progress_bar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_open_files) as executor:
                executor.map(apply_transform, transforms.keys())

    def clear_cache(self):
        """Remove the folders corresponding to cached results."""
        if self.verbose:
            print("CLEARING CACHE")

        for folder in [CACHE_FILES["transforms_dir"], CACHE_FILES["fiducial_template_dir"]]:
            if not os.path.isdir(folder):
                continue
            shutil.rmtree(folder)

    def get_fiducial_reference_positions(self):
        """
        Get the fiducial positions in the reference frame.

        return: transformed_fiducials: Fiducial pixel positions (y, x).
        """

        initial_fiducials = self.manual_fiducials[self.manual_fiducials["camera"]
                                                  == self.orb_reference_filename]

        transformed_fiducials = self.manual_transforms[self.orb_reference_filename](
            initial_fiducials[["x_px", "y_px"]].values)[:, ::-1]

        return transformed_fiducials

    def calculate_fiducial_projections(self):
        fiducial_reference_positions = self.get_fiducial_reference_positions()

        Fiducial = namedtuple("Fiducial", field_names=["name", "x", "y"])
        Fiducials = namedtuple("Fiducials", field_names=["top", "right", "bottom", "left"])

        fiducials = {}
        for filename, transform in self.load_transforms("merged_transforms.pkl").items():
            transformed_fiducial_positions = transform(fiducial_reference_positions[:, ::-1])

            fiducials[filename] = Fiducials(
                Fiducial("top", transformed_fiducial_positions[0, 0], transformed_fiducial_positions[0, 1]),
                Fiducial("right", transformed_fiducial_positions[1, 0], transformed_fiducial_positions[1, 1]),
                Fiducial("bottom", transformed_fiducial_positions[2, 0], transformed_fiducial_positions[2, 1]),
                Fiducial("left", transformed_fiducial_positions[3, 0], transformed_fiducial_positions[3, 1]))

        return fiducials

    def calculate_fiducial_coordinates(self):

        raise NotImplementedError()


def generate_fiducial_animation(output=os.path.join(TEMP_DIRECTORY, "fiducial_template_corr_animation.mp4")):
    """
    Animate the top fiducial location by rapidly switching between rectified images.
    """
    matcher = FrameMatcher()
    fiducial_coords = matcher.fiducial_locations
    image_folder = matcher.transformed_image_folder
    # Make frame directory if necessary
    frame_temp_folder = os.path.join(TEMP_DIRECTORY, "frames")
    if not os.path.isdir(frame_temp_folder):
        os.makedirs(frame_temp_folder)

    window_size = 500

    fiducials = {key: [] for key in fiducial_coords.keys()}

    mean_fiducials = []
    progress_bar: Optional[tqdm] = None

    filenames = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder)]

    def extract_fiducials(filename):
        all_fiducials = np.empty((window_size * 2, window_size * 2), dtype=np.uint8)
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"File {filename} returned None on cv2.imread")

        for corner in fiducial_coords.keys():
            # Extract the fiducial
            y_center, x_center = fiducial_coords[corner]
            top = y_center - window_size // 2
            bottom = y_center + window_size // 2
            left = x_center - window_size // 2
            right = x_center + window_size // 2

            fiducial = img[top:bottom, left:right].astype(np.int32)
            # Set the luminance of the frame as zero
            fiducial -= find_frame_luminance_threshold(fiducial)
            # Fix negative values
            fiducial[fiducial < 0] = 0
            fiducials[corner].append(fiducial)
            if corner == "top":
                all_fiducials[:window_size, :window_size] = fiducial
            elif corner == "left":
                all_fiducials[window_size:, :window_size] = fiducial
            elif corner == "right":
                all_fiducials[:window_size, window_size:] = fiducial
            else:
                all_fiducials[window_size:, window_size:] = fiducial

        i = filenames.index(filename)
        cv2.imwrite(os.path.join(frame_temp_folder, f"frame_{i}.jpg"), all_fiducials)

        progress_bar.update()

    with tqdm(total=len(filenames)) as progress_bar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(extract_fiducials, filenames)

    if False:
        # Go over each file, extract the fiducial, normalise the zero exposure value, then export the frame
        for i, file in enumerate(tqdm(os.listdir(image_folder))):
            all_fiducials = np.empty((window_size * 2, window_size * 2), dtype=np.uint8)
            img = cv2.imread(os.path.join(image_folder, file), cv2.IMREAD_GRAYSCALE)

            for corner in fiducial_coords.keys():
                # Extract the fiducial
                y_center, x_center = fiducial_coords[corner]
                top = y_center - window_size // 2
                bottom = y_center + window_size // 2
                left = x_center - window_size // 2
                right = x_center + window_size // 2

                fiducial = img[top:bottom, left:right].astype(np.int32)
                # Set the luminance of the frame as zero
                fiducial -= find_frame_luminance_threshold(fiducial)
                # Fix negative values
                fiducial[fiducial < 0] = 0
                fiducials[corner].append(fiducial)
                if corner == "top":
                    all_fiducials[:window_size, :window_size] = fiducial
                elif corner == "left":
                    all_fiducials[window_size:, :window_size] = fiducial
                elif corner == "right":
                    all_fiducials[:window_size, window_size:] = fiducial
                else:
                    all_fiducials[window_size:, window_size:] = fiducial
            cv2.imwrite(os.path.join(frame_temp_folder, f"frame_{i + 1}.jpg"), all_fiducials)

    # Use ffmpeg to encode an easy to watch mp4
    subprocess.run(f"ffmpeg -framerate 10 -i {frame_temp_folder}/frame_%000d.jpg -c:v libx264 " +
                   f"-profile:v high -crf 20 -pix_fmt yuv420p {output} -y", check=True, shell=True)

    return output


if __name__ == "__main__":
    matcher = FrameMatcher(cache=True)

    matcher.calculate_fiducial_projections()
