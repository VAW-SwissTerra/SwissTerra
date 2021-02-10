"""Wrapper functions for Agisoft Metashape."""
from __future__ import annotations

import concurrent.futures
import itertools
import os
import pickle
import tempfile
import warnings
from enum import Enum
from typing import Optional

import Metashape as ms
import numpy as np
import pandas as pd
from tqdm import tqdm

from terra import dem_tools, files
from terra.constants import CONSTANTS
from terra.preprocessing import fiducials, masks
from terra.processing import inputs, processing_tools
from terra.utilities import is_gpu_available, no_stdout

CACHE_FILES = {}


# To fix a bug in Metashape 1.7.0
if ms.version == "1.7.0":
    ms.app.settings.setValue("main/depth_pm_enable", "False")

# Add dataset metashape project paths
for _dataset in inputs.DATASETS:
    CACHE_FILES[f"{_dataset}_metashape_project"] = os.path.join(
        inputs.CACHE_FILES[f"{_dataset}_dir"], f"{_dataset}.psx")


class Quality(Enum):
    """Dense cloud quality."""

    ULTRA: int = 1
    HIGH: int = 2
    MEDIUM: int = 4
    LOW: int = 8


class Filtering(Enum):
    """Dense cloud point filtering."""

    AGGRESSIVE: ms.FilterMode = ms.FilterMode.AggressiveFiltering
    MODERATE: ms.FilterMode = ms.FilterMode.ModerateFiltering
    MILD: ms.FilterMode = ms.FilterMode.MildFiltering
    NOFILTER: ms.FilterMode = ms.FilterMode.NoFiltering


class Step(Enum):
    """Pipeline processing step."""

    ALIGNMENT = 1
    DENSE_CLOUD = 2
    DEM = 3
    ORTHOMOSAIC = 4


def is_document(dataset: str) -> bool:
    """
    Check if a Metashape document exists.

    :param dataset: The name of the dataset.

    :returns: Whether the document exists or not.
    """
    cache_file_key = f"{dataset}_metashape_project"
    if cache_file_key not in CACHE_FILES:
        return False
    return os.path.isfile(CACHE_FILES[cache_file_key])


def new_document(dataset: str) -> ms.Document:
    """
    Create a new Metashape document.

    :param dataset: The dataset name.

    :returns: The newly created Metashape document.
    """
    # Make the temporary directory (also makes the project root directory)
    os.makedirs(inputs.CACHE_FILES[f"{dataset}_temp_dir"], exist_ok=True)
    with no_stdout():
        doc = ms.Document()

        doc.save(CACHE_FILES[f"{dataset}_metashape_project"])

    if doc.read_only:
        raise AssertionError("New document is in read-only mode. Is it open in another process?")

    doc.meta["dataset"] = dataset

    return doc


def load_document(dataset: str) -> ms.Document:
    """
    Load an already existing Metashape document.

    :param dataset: The dataset name.

    :returns: The newly created Metashape document.
    """
    with no_stdout():
        doc = ms.Document()
        doc.open(CACHE_FILES[f"{dataset}_metashape_project"])

    if doc.read_only:
        raise AssertionError("The loaded document is in read-only mode. Is it open in another process?\
                Remove lock file if not using 'terra cache remove-locks'.")

    doc.meta["dataset"] = dataset

    return doc


def save_document(doc: ms.Document) -> None:
    """Save the metashape document."""
    with no_stdout():
        doc.save()


def has_alignment(chunk: ms.Chunk) -> bool:
    """
    Check whether some or all cameras in a chunk are aligned.

    :param chunk: The chunk to check.

    :returns: Whether at least one camera is aligned.
    """
    any_alignment = any([camera.transform is not None for camera in chunk.cameras])
    return any_alignment


def import_reference(chunk: ms.Chunk, filepath: str) -> None:
    """
    Import camera location and orientation information into Metashape.

    The camera location CRS is assumed to be the same as the chunk CRS.
    The camera orientation format is assumed to be in Yaw, Pitch, Roll

    :param chunk: The chunk to add the reference to.
    :param filepath: The filepath of the reference csv.
    """
    # Use Omega Phi Kappa as the rotation system (yaw pitch roll has some bug that makes it incorrect)
    chunk.euler_angles = ms.EulerAnglesOPK

    # Load the reference position and rotation information.
    reference_data = pd.read_csv(filepath, index_col="label").astype(float)
    reference_data.index = reference_data.index.str.replace(".tif", "")

    for camera in chunk.cameras:
        cam_data = reference_data.loc[camera.label]
        camera.reference.location = cam_data[["easting", "northing", "altitude"]].values

        # Convert from yaw, pitch roll to Omega Phi Kappa (again due to the YPR bug)
        camera.reference.rotation = ms.Utils.mat2opk(ms.Utils.ypr2mat(
            ms.Vector(cam_data[["yaw", "pitch", "roll"]].values)))


def import_fiducials(chunk: ms.Chunk, sensor: ms.Sensor) -> None:
    """
    Generate fiducials for the specified sensor from a file with coordinates.

    :param chunk: The active chunk.
    :param sensor: The sensor (instrument) to assign the fiducials to.
    """
    # Find which instrument the dataset belongs to using its name, e.g. Wild1_1924 -> Wild1
    instrument = chunk.meta["dataset"].split("_")[0]
    fiducial_marks = pd.read_csv(
        os.path.join(fiducials.CACHE_FILES["fiducial_location_dir"], f"fiducials_{instrument}.csv"),
        index_col=0)

    # Add one fiducial mark for each corner.
    new_fiducials = {}
    for corner in ["top", "right", "bottom", "left"]:
        fiducial = chunk.addMarker()
        fiducial.type = ms.Marker.Type.Fiducial
        fiducial.label = f"{sensor.label}_{corner}"

        fiducial.sensor = sensor

        # Take the x/y mm coordinate of the first matching camera
        x_mm, y_mm = fiducial_marks.loc[chunk.cameras[0].label + ".tif", [f"{corner}_x_mm", f"{corner}_y_mm"]]
        fiducial.reference.location = ms.Vector([x_mm, y_mm, 0])

        new_fiducials[corner] = fiducial

    # Add the corresponding fiducial projections of each corner to the cameras.
    for camera in chunk.cameras:
        projection_row = fiducial_marks.loc[camera.label + ".tif"]

        for corner, fiducial in new_fiducials.items():
            fiducial.projections[camera] = ms.Marker.Projection(
                ms.Vector([
                    projection_row[f"{corner}_x"],
                    projection_row[f"{corner}_y"]
                ]),
                True)


def new_chunk(doc: ms.Document, filenames: list[str], chunk_label: Optional[str] = None) -> ms.Chunk:
    """
    Create a new chunk in a document, add photos, and set appropriate parameters for them.

    :param doc: The active Metashape document.
    :param filenames: The names of the files to be added to the chunk.

    :returns: The newly created Metashape chunk.
    """
    with no_stdout():
        chunk = doc.addChunk()

    chunk.meta["dataset"] = doc.meta["dataset"]

    if chunk_label is not None:
        chunk.label = chunk_label

    chunk.crs = ms.CoordinateSystem(CONSTANTS.crs_epsg)
    chunk.camera_location_accuracy = ms.Vector([CONSTANTS.position_accuracy] * 3)
    chunk.camera_rotation_accuracy = ms.Vector([CONSTANTS.rotation_accuracy] * 3)

    # Get the full filepaths for the images that should be used.
    # TODO: Maybe replace the filenames argument with filename derivation from the image_meta below?
    filepaths = [os.path.join(files.INPUT_DIRECTORIES["image_dir"], filename) for filename in filenames]

    # Read the metadata for each image in the dataset
    image_meta = inputs.get_dataset_metadata(doc.meta["dataset"])

    with no_stdout():
        chunk.addPhotos(filepaths)

    # Add each instrument in the dataset as a different sensor
    # TODO: Maybe remove this as instruments are now always processed separately?
    sensors = {}
    for focal_length, dataframe in image_meta.groupby("focal_length"):
        sensor = chunk.addSensor()
        sensor.film_camera = True
        sensor.label = dataframe["Instrument"].iloc[0] + "_" + str(int(focal_length)) + "mm"

        sensor.pixel_size = ms.Vector([CONSTANTS.scanning_resolution * 1e3] * 2)
        sensor.focal_length = focal_length  # dataframe.iloc[0]["focal_length"]
        import_fiducials(chunk, sensor)

        sensors[sensor.label] = sensor

    for camera in chunk.cameras:
        cam_meta = image_meta[image_meta["Image file"].str.replace(".tif", "") == camera.label].iloc[0]
        sensor_name = f"{cam_meta['Instrument']}_{int(cam_meta['focal_length'])}mm"
        camera.sensor = sensors[sensor_name]

    # This should not be done anymore as the import_fiducials gives the right mm size data now
    # for sensor in sensors.values():
    #    sensor.calibrateFiducials(resolution=CONSTANTS.scanning_resolution * 1e3)

    # Import position and rotation data
    import_reference(chunk, inputs.CACHE_FILES[f"{doc.meta['dataset']}_camera_orientations"])

    # Separate all stereo-pairs and their left/right positions into unique groups
    groups: dict[str, ms.CameraGroup] = {}
    for camera in chunk.cameras:
        # Find the row corresponding to the camera
        camera_row = image_meta[image_meta["Image file"].str.replace(".tif", "") == camera.label].squeeze()
        # Specify what group label it should have
        group_label = f"station_{str(camera_row['Base number'])}_{camera_row['Position']}"

        # Create the group if it doesn't yet exist
        if group_label not in groups:
            group = chunk.addCameraGroup()
            group.label = group_label
            group.type = ms.CameraGroup.Type.Station
            groups[group_label] = group

        camera.group = groups[group_label]

        camera.reference.enabled = True
        camera.reference.rotation_enabled = True

    with no_stdout():
        chunk.generateMasks(path=os.path.join(masks.CACHE_FILES["mask_dir"], "{filename}.tif"),
                            masking_mode=ms.MaskingMode.MaskingModeFile, mask_operation=ms.MaskOperation.MaskOperationReplacement)
    return chunk


def align_cameras(chunk: ms.Chunk, fixed_sensor: bool = False) -> bool:
    """
    Align all cameras in a chunk.

    :param chunk: The chunk whose images shall be aligned.
    :param fixed_sensor: Whether to fix the sensors during the alignment.

    :returns: Whether the alignment was successful or not.
    """
    # If it should be a fixed sensor alignment, save which sensors were fixed to begin with (and restore that later)
    fixed_sensors = {sensor: sensor.fixed_calibration for sensor in chunk.sensors}
    if fixed_sensor:
        for sensor in chunk.sensors:
            sensor.fixed_calibration = True

    with no_stdout():
        chunk.matchPhotos(reference_preselection=True, filter_mask=True)
        chunk.alignCameras()

        # Check if no cameras were aligned (if all transforms are None)
        if all([camera.transform is None for camera in chunk.cameras]):
            if fixed_sensor:
                for sensor in chunk.sensors:
                    sensor.fixed_calibration = fixed_sensors[sensor]
            return False

        # Add all tie points below a maximum reprojection error of 10 pixels
        chunk.triangulatePoints(max_error=10)
        chunk.optimizeCameras()

    if fixed_sensor:
        for sensor in chunk.sensors:
            sensor.fixed_calibration = fixed_sensors[sensor]
    return True


def merge_chunks(doc: ms.Document, chunks: list[ms.Chunk], optimize: bool = True, remove_old: bool = False) -> ms.Chunk:
    """
    Merge Metashape chunks.

    :param doc: The Metashape document.
    :param chunks: A list of chunks to merge.
    :param optimize: Whether to optimize cameras after merging.
    :param remove_old: Whether to remove the original chunks after merging.

    :returns: The merged chunk.
    """
    # Get the integer positions of all chunks in the provided chunk list
    chunk_positions = []
    for i, chunk in enumerate(doc.chunks):
        if chunk in chunks:
            chunk_positions.append(i)

    with no_stdout():
        doc.mergeChunks(chunks=chunk_positions, merge_markers=True)

    merged_chunk: Optional[ms.Chunk] = None
    for chunk in doc.chunks:
        if chunk.label == "Merged Chunk":
            merged_chunk = chunk
            break

    if merged_chunk is None:
        raise RuntimeError("Couldn't find the merged chunk")

    if remove_old:
        for chunk in chunks:
            doc.remove(chunk)

    sensors = {}
    # Merge all sensors with the same label
    for sensor in merged_chunk.sensors:
        # If a new label is found, add that as the reference
        if not sensor.label in sensors:
            sensors[sensor.label] = sensor
            continue

        # Move the fiducials in the duplicate sensor to the reference
        for fiducial in sensor.fiducials:
            for ref_fiducial in sensors[sensor.label].fiducials:
                if ref_fiducial.label == fiducial.label:
                    break

            for camera, projection in fiducial.projections.items():
                ref_fiducial.projections[camera] = projection

            merged_chunk.remove(fiducial)

        # Move all the cameras of the duplicate sensor to the reference
        for camera in merged_chunk.cameras:
            if camera.sensor.label == sensor.label:
                camera.sensor = sensors[sensor.label]

        # Remove the duplicate sensor
        merged_chunk.remove(sensor)

    # Assign the markers to the new merged sensors.
    for marker in merged_chunk.markers:
        if not marker.type == ms.Marker.Type.Fiducial:
            continue
        for sensor_label in sensors:
            if sensor_label in marker.label:
                marker.sensor = sensors[sensor_label]

    # TODO: Remove this? Sensors are already nicely calibrated using new fiducial scripts.
    for label, sensor in sensors.items():
        if label == "unknown":
            continue
        sensor.calibrateFiducials(resolution=CONSTANTS.scanning_resolution * 1e3)

    if optimize:
        with no_stdout():
            merged_chunk.optimizeCameras(adaptive_fitting=True)

    return merged_chunk


def load_or_remake_chunk(doc: ms.Document, dataset: str) -> ms.Chunk:
    """
    Load or or remake chunks based on if they already exist or not.

    If the chunk should be redone:
    1. Align all stereo-pairs in separate chunks.
    2. Merge the chunks.

    :param doc: The Metashape document to check
    :param dataset: The name of the dataset.

    :returns: The chunk with all stereo-pairs.
    """
    if not is_gpu_available():
        warnings.warn("GPU processing turned off")
        ms.app.gpu_mask = 0
    # Load image metadata to get the station numbers
    image_meta = inputs.get_dataset_metadata(dataset)

    # First check if the Merged Chunk already exists.
    for chunk in doc.chunks:
        if chunk.label == "Merged Chunk":
            merged_chunk = chunk
            merged_chunk.meta["dataset"] = dataset
            return merged_chunk

    aligned_chunks: list[ms.Chunk] = []
    progress_bar = tqdm(total=np.unique(image_meta["Base number"]).shape[0])
    # Loop through all stations (stereo-pairs) and align them if they don't already exist
    for station_number, station_meta in image_meta.groupby("Base number"):
        progress_bar.desc = f"Aligning station {station_number}"
        # Check if a "stereo-pair" only has the right or left position (so it's not a stereo-pair)
        if station_meta["Position"].unique().shape[0] < 2:
            print(f"Station {station_number} only has position {station_meta['Position'].iloc[0]}. Skipping.")
            progress_bar.update()
            continue

        chunk_label = f"station_{station_number}"

        # Check if the chunk already exists
        if any([chunk.label == chunk_label for chunk in doc.chunks]):
            chunk = [chunk for chunk in doc.chunks if chunk.label == chunk_label][0]
            # Check if the chunk is aligned
            aligned = has_alignment(chunk)

        # Create and try to align the chunk if it doesn't exist
        else:
            chunk = new_chunk(doc, filenames=list(
                station_meta["Image file"].values), chunk_label=chunk_label)
            # Try to align the cameras and check if it worked.
            aligned = align_cameras(chunk, fixed_sensor=False)
            save_document(doc)

        # Append the chunk to the to-be-merged chunk list if it got aligned.
        if aligned:
            aligned_chunks.append(chunk)

        progress_bar.update()

    print("Merging chunks")
    merged_chunk = merge_chunks(doc, aligned_chunks, remove_old=True, optimize=True)
    merged_chunk.meta["dataset"] = dataset

    return merged_chunk


def get_marker_reprojection_error(camera: ms.Camera, marker: ms.Marker) -> np.float64:
    """
    Get the reprojection error between a marker's projection and a camera's reprojected position.

    :param camera: The camera to reproject the marker position to.
    :param marker: The marker to compare the projection vs. reprojection on.
    """
    # Validate that both the marker and the camera is not None
    if None in [camera, marker]:
        return np.NaN
    projected_position = marker.projections[camera].coord
    reprojected_position = camera.project(marker.position)

    if reprojected_position is None:
        return np.NaN

    diff = (projected_position - reprojected_position).norm()

    image_size = max(int(camera.photo.meta["File/ImageWidth"]), int(camera.photo.meta["File/ImageHeight"]))
    # Check if the reprojected position difference is too high
    if diff > image_size:
        return np.NaN

    return diff


def get_rms_marker_reprojection_errors(markers: list[ms.Marker]) -> dict[ms.Marker, np.float64]:
    """
    Calculate the mean reprojection error for a marker, checked on all pinned projections.
    """
    errors: dict[ms.Marker, list[np.float64]] = {marker: []
                                                 for marker in markers if marker.type == ms.Marker.Type.Regular}

    for marker in errors:
        for camera, projection in marker.projections.items():
            if not projection.pinned:
                continue
            errors[marker].append(get_marker_reprojection_error(camera, marker))

    def rms(values: list[np.float64]):
        if np.count_nonzero(~np.isnan(values)) == 0:
            return np.nan
        return np.sqrt(np.nanmean(np.square(values))).astype(np.float64)

    mean_errors = {marker: rms(error_list) for marker, error_list in errors.items()}

    return mean_errors


def get_chunk_stereo_pairs(chunk: ms.Chunk) -> list[str]:
    """
    Get a list of stereo-pair group names.

    :param chunk: The chunk to analyse.

    :returns: A list of stereo-pair group names.

    """
    pairs: list[str] = []
    for group in chunk.camera_groups:
        pair = group.label.replace("_R", "").replace("_L", "")

        if pair not in pairs:
            pairs.append(pair)

    return pairs


def get_unfinished_pairs(chunk: ms.Chunk, step: Step) -> list[str]:
    """
    list all stereo-pairs that have not yet finished a step.

    :param chunk: The chunk to analyse.
    :param step: The step to check for.

    :returns: A list of stereo-pairs that are unfinished.
    """
    pairs = get_chunk_stereo_pairs(chunk)

    if step == Step.DENSE_CLOUD:
        labels_to_check = [cloud.label for cloud in chunk.dense_clouds]
    elif step == Step.DEM:
        labels_to_check = [dem.label for dem in chunk.elevations]
    elif step == Step.ORTHOMOSAIC:
        labels_to_check = [ortho.label for ortho in chunk.orthomosaics]
    else:
        raise NotImplementedError(f"Step {step} not implemented")

    unfinished_pairs = [pair for pair in pairs if pair not in labels_to_check]

    return unfinished_pairs


def optimize_cameras(chunk: ms.Chunk, fixed_sensors: bool = False) -> None:
    """
    Optimize the chunk camera alignment.

    :param chunk: The chunk whose camera alignment should be optimized.
    :param fixed_sensors: Whether to temporarily fix the sensor values on optimization.
    """
    # Parameters to either solve for or not to solve for in camera optimization
    parameters = ["fit_f", "fit_cx", "fit_cy", "fit_b1", "fit_b2",
                  "fit_k1", "fit_k2", "fit_k3", "fit_k4", "fit_p1", "fit_p2"]

    if fixed_sensors:
        # Get the initial values for the user calibration, fixed flag, and adjusted calibration.
        # Get the NoneType if the calibration doesn't exist, otherwise copy the calibration.
        old_initial_calibrations = {
            sensor: sensor.user_calib if not sensor.user_calib else sensor.user_calib.copy() for sensor in chunk.sensors
        }
        old_fixed_flags = {sensor: sensor.fixed_calibration for sensor in chunk.sensors}
        old_calibrations = {sensor: sensor.calibration.copy() for sensor in chunk.sensors}

        # Set the adjusted calibration to be the user calibration (to use as fixed) and then fix the calibration.
        for sensor in chunk.sensors:
            sensor.user_calib = old_calibrations[sensor]
            sensor.fixed_calibration = True

        # Optimize cameras with all optimization parameters turned off.
        with no_stdout():
            chunk.optimizeCameras(**{key: False for key in parameters})

        # Return the fixed flags and the user calibration to their initial values.
        for sensor in chunk.sensors:
            sensor.fixed_calibration = old_fixed_flags[sensor]
            sensor.user_calib = old_initial_calibrations[sensor]

    # If sensors should not be fixed.
    else:
        # Optimize cameras with all parameters turned on.
        with no_stdout():
            chunk.optimizeCameras(**{key: True for key in parameters})


def remove_bad_markers(chunk, marker_error_threshold: float = 4.0):
    """
    Remove every marker in the chunk that has is above the reprojection threshold.

    If an ICP tie point is too high, all of its family members will be removed as well.

    :param chunk: The chunk to analyse.
    :param marker_error_threshold: The marker reprojection error threshold to accept.
    """
    # Get the rms reprojection error for each marker
    errors = get_rms_marker_reprojection_errors(chunk.markers)
    # Find the markers whose values are too high
    too_high = [marker for marker in errors if errors[marker] > marker_error_threshold]

    # Loop through each marker and check if it belongs to a tie point family
    for marker in too_high:
        if "tie_" in marker.label:
            # If it is, remove all of its friends
            tie_family = marker.label[:marker.label.index("_num_")]

            # Find the friends with the same marker label family
            for marker2 in errors:
                if marker2 in too_high:
                    continue
                if tie_family in marker2.label:
                    too_high.append(marker2)

    for marker in too_high:
        chunk.remove(marker)


def build_dense_clouds(doc: ms.Document, chunk: ms.Chunk, pairs: list[str], quality: Quality = Quality.HIGH,
                       filtering: Filtering = Filtering.AGGRESSIVE, intermediate_saving: bool = True,
                       all_together: bool = False) -> list[str]:
    """
    Generate dense clouds for all stereo-pairs in a given chunk.

    :param doc: The document that the chunk is in.
    :param chunk: The chunk to process.
    :param pairs: A list of the stereo-pairs to process.
    :param quality: The quality of the dense cloud.
    :param filtering: The dense cloud filtering setting.
    :param intermediate_saving: Whether to save the document between dense cloud generations.
    :param all_together: Whether to process every stereo-pair together.
    :returns: A list of stereo-pairs where the dense cloud generation succeeded.
    """
    successful_pairs: list[str] = []

    def generate_dense_cloud(cameras: list[ms.Camera], label) -> bool:
        for camera in chunk.cameras:
            # Set it as enabled if the camera group label fits with the stereo pair label
            camera.enabled = camera in cameras
        try:
            with no_stdout():
                chunk.buildDepthMaps(downscale=quality.value, filter_mode=filtering.value)
        except RuntimeError as exception:
            if "Assertion 23910910009 failed" in str(exception):
                raise ValueError("Zero resolution error. Last time it was because of Metashape 1.7.0.")
            raise exception
        try:
            with no_stdout(disable=False):
                chunk.buildDenseCloud(point_confidence=True)
        except MemoryError:
            warnings.warn(f"Memory error on dense cloud for {label} in {chunk.meta['dataset']}")
            return False
        except Exception as exception:
            if "Zero resolution" in str(exception):
                return False
            raise exception

        with no_stdout():
            chunk.dense_cloud.label = label

            # Remove all points with a low confidence
            chunk.dense_cloud.setConfidenceFilter(0, CONSTANTS.dense_cloud_min_confidence - 1)
            chunk.dense_cloud.removePoints(list(range(128)))
            chunk.dense_cloud.resetFilters()
        return True

    if all_together:
        print("Generating dense cloud from all stereo-pairs")
        generate_dense_cloud(chunk.cameras, label="all_pairs")
        successful_pairs = pairs

    else:
        with tqdm(total=len(pairs), desc="Building dense clouds") as progress_bar:
            for pair in pairs:
                progress_bar.desc = f"Building dense cloud for {pair}"
                progress_bar.update(n=0)  # Update the new text

                cameras_to_process = [camera for camera in chunk.cameras if pair in camera.group.label]
                successful = generate_dense_cloud(cameras_to_process, label=pair)
                if successful:
                    successful_pairs.append(pair)
                # Unset the dense cloud as default to allow for more dense clouds to be constructed
                chunk.dense_cloud = None

                progress_bar.update()
                if intermediate_saving:
                    save_document(doc)

        # Reset the enabled flags
        for camera in chunk.cameras:
            camera.enabled = True

    return successful_pairs


def build_dems(chunk: ms.Chunk, pairs: list[str], redo: bool = False,
               resolution: float = 5.0, interpolate_pixels: int = 0) -> dict[str, str]:
    """
    Build DEMs for each stereo-pair.

    :param chunk: The chunk to export from.
    :param pairs: Which stereo-pairs to use.
    :param redo: Whether to remake dense clouds or DEMs if they already exist.
    :param resolution: The DEM resolution in metres.
    :param interpolate_pixels: The amount of small gaps to interpolate in the DEMs

    :returns: The filepaths of the exported clouds for each stereo-pair.
    """
    assert chunk.meta["dataset"] is not None

    filepaths: dict[str, str] = {}

    dense_clouds = [cloud for cloud in chunk.dense_clouds if cloud.label in pairs]

    for cloud in tqdm(dense_clouds, desc="Exporting dense clouds"):
        if not cloud.label in pairs:
            continue
        cloud_filepath = os.path.join(
            inputs.CACHE_FILES["{dataset}_temp_dir".format(dataset=chunk.meta["dataset"])],
            "{cloud_label}_dense.ply".format(cloud_label=cloud.label))
        # Set the pair name (same as cloud.label) and its corresponding cloud filepath
        filepaths[cloud.label] = cloud_filepath

        if redo or not os.path.isfile(cloud_filepath):
            chunk.dense_cloud = cloud
            with no_stdout():
                chunk.exportPoints(cloud_filepath, crs=chunk.crs, source_data=ms.DataSource.DenseCloudData)

    chunk.dense_cloud = None

    progress_bar = tqdm(total=len(filepaths), desc="Generating DEMs")

    def build_dem(pair_and_filepath: tuple[str, str]) -> tuple[str, str]:
        """
        Generate a DEM from a point cloud.

        :param pair: The stereo-pair label
        :param filepath: The path to the input point cloud.

        :returns: The stereo-pair label and the path to the output DEM.
        """
        pair, filepath = pair_and_filepath
        progress_bar.desc = f"Generating DEMs for {pair}"
        progress_bar.update(n=0)  # Update the text
        output_filepath = os.path.splitext(filepath)[0] + "_DEM.tif"
        try:
            if redo or not os.path.isfile(output_filepath):
                processing_tools.generate_dem(filepath, output_dem_path=output_filepath,
                                              resolution=resolution, interpolate_pixels=interpolate_pixels)
        # TODO: Fix better exception handling
        except Exception as exception:
            print(exception)
            output_filepath = None

        progress_bar.update()

        return pair, output_filepath

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        dem_filepaths = dict(executor.map(build_dem, filepaths.items()))

    # Remove all entries where DEM generation was unsuccessful.
    for pair in list(dem_filepaths.keys()):
        if dem_filepaths[pair] is None:
            del dem_filepaths[pair]

    for dem_label, dem_filepath in dem_filepaths.items():
        with no_stdout():
            chunk.importRaster(path=dem_filepath, crs=chunk.crs, raster_type=ms.DataSource.ElevationData)
        chunk.elevations[-1].label = dem_label
        chunk.elevation = None

    progress_bar.close()
    return dem_filepaths


def build_orthomosaics(chunk: ms.Chunk, pairs: list[str], resolution: float) -> list[str]:
    """
    Build orthomosaics for each chunk.

    :param chunks: The active chunk.
    :param pairs: The stereo-pairs to build orthomosaics for.
    :param resolution: The orthomosaic resolution in metres.
    :returns: A list of stereo-pairs where the orthomosaic generation succeeded.
    """
    # Make a list of all of the stereo-pairs that successfully got orthomosaics
    successful_pairs: list[str] = []

    progress_bar = tqdm(total=len(pairs))
    for pair in pairs:
        progress_bar.desc = f"Generating orthomosaic for {pair}"
        # Set all cameras of the stereo-pair to be enabled and disable all others.
        for camera in chunk.cameras:
            camera.enabled = pair in camera.group.label

        # Match a DEM with the stereo-pair name. Has a len() of 1 if it exists or 0 if it doesn't.
        corresponding_dem_list = [dem for dem in chunk.elevations if pair == dem.label]

        # Skip if the stereo-pair has no corresponding DEM
        if len(corresponding_dem_list) == 0:
            print(f"Pair {pair} has no DEM. Skipping orthomosaic generation")
            progress_bar.update()
            continue
        # Set the "default" DEM to the one corresponding to the stereo-pair
        chunk.elevation = corresponding_dem_list[0]

        try:
            with no_stdout():
                chunk.buildOrthomosaic(surface_data=ms.DataSource.ElevationData, resolution=resolution)
        except Exception as exception:
            if "Zero resolution" in str(exception):
                print(f"Pair {pair} orthomosaic got zero resolution")
                progress_bar.update()
                continue
            # If Zero resolution is not in exception:
            raise exception

        # Set the label to equal the stereo-pair label
        chunk.orthomosaics[-1].label = pair
        # Unset the "default" orthomosaic
        chunk.orthomosaic = None

        successful_pairs.append(pair)
        progress_bar.update()

    # Unset the "default" DEM
    chunk.elevation = None
    progress_bar.close()

    return successful_pairs


def export_orthomosaics(chunk: ms.Chunk, pairs: list[str], directory: str, overwrite: bool = False):
    """
    Export all orthomosaics of the given pairs that exist.

    Orthomosaics that do not exist are silently skipped.

    :param chunk: The current Metashape chunk.
    :param pairs: The stereo-pairs to export the orthomosaics from.
    :param directory: The output directory to export the orthomosaics in.
    :param overwrite: Overwrite already existing files flag.
    """
    os.makedirs(directory, exist_ok=True)

    # Set the compression type to be JPEG quality 90% for TIFFs
    compression_type = ms.ImageCompression()
    compression_type.tiff_compression = ms.ImageCompression.TiffCompressionJPEG
    compression_type.jpeg_quality = 90

    progress_bar = tqdm(total=len(pairs))
    for pair in pairs:
        progress_bar.desc = f"Exporting orthomosaic for {pair}"
        output_filepath = os.path.join(directory, f"{pair}_orthomosaic.tif")

        # Skip if it shouldn't overwrite an already existing orthomosaic
        if not overwrite and os.path.isfile(output_filepath):
            progress_bar.update()
            continue

        # A list of matched orthomosaic(s). It always has a len() of 1 or 0 (meaning it exists or it doesn't)
        corresponding_ortho_list = [ortho for ortho in chunk.orthomosaics if ortho.label == pair]

        # Skip if no orthomosaic exists of that pair
        if len(corresponding_ortho_list) == 0:
            progress_bar.update()
            continue
        # Set the "default" orthomosaic to be the one corresponding to the stereo-pair
        chunk.orthomosaic = corresponding_ortho_list[0]

        with no_stdout():
            chunk.exportRaster(
                path=output_filepath,
                source_data=ms.DataSource.OrthomosaicData,
                image_compression=compression_type
            )
        progress_bar.update()


def coalign_stereo_pairs(chunk: ms.Chunk, pairs: list[str], max_fitness: float = 20.0,
                         tie_group_radius: float = 30.0, marker_pixel_accuracy=4.0):
    """
    Use DEM ICP coaligning to align combinations of stereo-pairs.

    :param chunk: The chunk to analyse.
    :param pairs: The stereo-pairs to coaling.
    :param max_fitness: The maximum allowed PDAL ICP fitness parameters (presumed to be C2C distance in m)
    :param tie_group_radius: The distance of all tie points from the centroid of the alignment.
    :param marker_pixel_accuracy: The approximate accuracy in pixels to give to Metashape.
    """
    if chunk.meta["dataset"] is None:
        raise ValueError("Chunk dataset meta is undefined")
    # Build DEMs to subsequently coalign
    dem_paths = build_dems(chunk, pairs=pairs)

    # Find all combinations of pairs that had DEMs successfully generated.
    pair_combinations = list(itertools.combinations([pair for pair in pairs if pair in dem_paths], r=2))
    # Get the same combinations but with DEM paths instead.
    path_combinations = [(dem_paths[first], dem_paths[second]) for first, second in pair_combinations]

    # Start a progress bar for the DEM coaligning
    progress_bar = tqdm(total=len(path_combinations), desc="Coaligning DEM pairs")

    def coalign_dems(path_combination: tuple[str, str]):
        """Coalign two DEMs in one thread."""
        path_1, path_2 = path_combination
        result = processing_tools.coalign_dems(reference_path=path_1, aligned_path=path_2)
        progress_bar.update()

        return result

    # Coaling all DEM combinations
    # The results variable is a list of transforms from pair1 to pair2
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        results = list(executor.map(coalign_dems, path_combinations))

    progress_bar.close()

    # Run through each combination of stereo-pairs and try to coalign the DEMs
    progress_bar = tqdm(total=len(pair_combinations))
    for i, (pair_1, pair_2) in enumerate(pair_combinations):
        # Update the progress bar description
        progress_bar.desc = f"Processing results for {pair_1} and {pair_2}"
        # Exctract the corresponding result
        result = results[i]
        # Skip if coalignment was not possible
        if result is None:
            progress_bar.update()
            continue

        # Skip if the coalignment fitness was poor. I think 10 means 10 m of offset
        if result["fitness"] > max_fitness:
            progress_bar.update()
            print(f"{pair_1} to {pair_2} fitness exceeded threshold: {result['fitness']}")
            continue

        # Get the ICP centroid as a numpy array (x, y, z) and create tie points from it
        centroid = np.array([float(value) for value in result["centroid"].splitlines()])
        # Offsets to change the centroid (first offsets the x coordinate, third offsets the y coordinate, etc.)
        offsets = [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 0]
        ]
        # Create rows for pandas DataFrames by multiplying the centroid with corresponding offsets
        rows = [centroid + np.array(offset) * tie_group_radius for offset in offsets]
        # Make an aligned point position DataFrame from the centroid-centered points
        aligned_point_positions = pd.DataFrame(data=rows, columns=["X", "Y", "Z"])
        # Transform the points to the reference coordinates using the ICP transform
        # The order of these may seem unintuitive (since the reference points are transformed, not vice versa)
        # The transform is a "recipe" for how to get points from an aligned POV to a reference POV
        # Therefore, by transforming the aligned points to a reference POV, we get the corresponding reference points.
        reference_point_positions = processing_tools.transform_points(
            aligned_point_positions, result["composed"], inverse=False)

        def global_to_local(row: pd.Series) -> ms.Vector:
            """Convert global coordinates in a pandas row to local coordinates."""
            global_coord = ms.Vector(row[["X", "Y", "Z"]].values)

            # Transform it using the inverse transform matrix of the unprojected coordinate. Geez.
            local_coord = chunk.transform.matrix.inv().mulp(chunk.crs.unproject(global_coord))

            # Make a new identical row, but with new values. This may be replaced by just new_row = list(local_coord)
            new_row = pd.Series(data=list(local_coord), index=["X", "Y", "Z"])

            return new_row

        # Convert the point positions to local coordinates (so point projections can be made)
        local_aligned_positions = aligned_point_positions.apply(global_to_local, axis=1)
        local_reference_positions = reference_point_positions.apply(global_to_local, axis=1)

        def is_projection_valid(camera: ms.Camera, projected_position: ms.Vector) -> bool:
            """
            Check if a camera projection is valid, e.g. if it's not outside the camera bounds.

            :param camera: The camera to check the projection validity on.
            :param projected_position: The pixel position of the projection to evaluate.

            :returns: If the projection seems valid or not.


            """
            if projected_position is None:
                return False
            # Check if the pixel locations are negative (meaning it's invalid)
            if projected_position.x < 0 or projected_position.y < 0:
                return False

            # Check if the projected x position is bigger than the image width
            if projected_position.x > int(camera.photo.meta["File/ImageWidth"]):
                return False
            # Check if the projected y position is bigger than the image height
            if projected_position.y > int(camera.photo.meta["File/ImageHeight"]):
                return False

            # Assume that the projection is valid if it didn't fill any of the above criterai
            return True

        def find_good_cameras(pair: str, positions: pd.DataFrame) -> list[ms.Camera]:
            """
            Find a camera that can "see" all the given positions.

            :param pair: Which stereo-pair to look for a camera in.
            :param positions: The positions to check if they are visible or not.

            :returns: A camera that can "see" all the given positions.
            """
            cameras_in_pair = [camera for camera in chunk.cameras if pair in camera.group.label]

            # Make a count for how many positions the camera can "see", starting at zero
            n_valid = {camera: 0 for camera in cameras_in_pair}

            # Go through each camera and count the visibilities
            for camera in cameras_in_pair:
                # Go through each given position
                for _, position in positions.iterrows():
                    # Project the position onto the camera coordinate system
                    projection = camera.project(position[["X", "Y", "Z"]].values)
                    # Check if the projection was valid
                    if is_projection_valid(camera, projection):
                        n_valid[camera] += 1

            # Find the cameras with valid at least n-1 projections valid (5 positions => 4 valid projections)
            good_cameras = [camera for camera, n_valid_projections in n_valid.items(
            ) if n_valid_projections >= (positions.shape[0] - 1)]

            return good_cameras

        # Find two represenative cameras. One in the "reference" pair and one in the "aligned"
        # Note that the "reference" and "aligned" labels only matter here, and have no effect on the result
        reference_cameras = find_good_cameras(pair_1, local_reference_positions)
        aligned_cameras = find_good_cameras(pair_2, local_aligned_positions)

        # Set the marker pixel accuracy
        chunk.marker_projection_accuracy = marker_pixel_accuracy

        # Go over each point position and make a Metashape marker from it if it's valid
        for j in range(local_aligned_positions.shape[0]):
            # Extract the reference and aligned point position
            reference_position = local_reference_positions.iloc[j]
            aligned_position = local_aligned_positions.iloc[j]

            # Project the positions
            reference_projections = [camera.project(reference_position) for camera in reference_cameras]
            aligned_projections = [camera.project(aligned_position) for camera in aligned_cameras]

            # TODO: Check if this is still necessary (should be redundant due to find_good_cameras())
            # Double check that the projections are valid
            for label, cameras, projections in [
                    ("Reference", reference_cameras, reference_projections),
                    ("Aligned", aligned_cameras, aligned_projections)
            ]:
                for camera, projection in zip(cameras, projections):
                    if not is_projection_valid(camera, projection):
                        print(f"{label} projection was invalid: {projection}")

            # Add a marker and set an appropriate label
            marker = chunk.addMarker()
            error = round(float(result["fitness"]), 2)  # The "fitness" is presumed to be in m
            marker.label = f"tie_{pair_1}_to_{pair_2}_num_{j}_error_{error}_m"

            # Set the projections
            for cameras, projections in [
                    (reference_cameras, reference_projections),
                    (aligned_cameras, aligned_projections)
            ]:

                for camera, projection in zip(cameras, projections):
                    marker.projections[camera] = ms.Marker.Projection(projection, True)

        progress_bar.update()

    progress_bar.close()


def stable_ground_registration(chunk: ms.Chunk, pairs: list[str], max_fitness: float = 20.0,
                               tie_group_radius: float = 30.0, marker_pixel_accuracy=4.0):
    """
    Use DEM ICP coaligning to align stereo-pairs to a stable ground reference.

    :param chunk: The chunk to analyse.
    :param pairs: The stereo-pairs to coaling.
    :param max_fitness: The maximum allowed PDAL ICP fitness parameters (presumed to be C2C distance in m)
    :param tie_group_radius: The distance of all tie points from the centroid of the alignment.
    :param marker_pixel_accuracy: The approximate accuracy in pixels to give to Metashape.
    """
    if chunk.meta["dataset"] is None:
        raise ValueError("Chunk dataset meta is undefined")
    # Build DEMs to subsequently coalign
    dem_paths = build_dems(chunk, pairs=pairs)
    pairs_with_dem = list(dem_paths.keys())

    temp_dir = tempfile.TemporaryDirectory()
    base_dem_paths: dict[str, str] = {}
    for pair, dem_path in tqdm(dem_paths.items(), desc="Extracting base DEM subsets"):
        base_dem_path = os.path.join(temp_dir.name, f"{pair}_base_dem.tif")
        dem_tools.extract_base_stable_ground(dem_path, base_dem_path, buffer=40)
        base_dem_paths[pair] = base_dem_path

    # Start a progress bar for the DEM coaligning
    progress_bar = tqdm(total=len(dem_paths), desc="Registering to base DEM")

    def register_dem(pair: str):
        """Register a DEM in one thread."""
        result = processing_tools.coalign_dems(
            reference_path=base_dem_paths[pair], aligned_path=dem_paths[pair], pixel_buffer=10)
        progress_bar.update()

        return result

    # Coaling all DEM combinations
    # The results variable is a list of all resultant transforms.
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(register_dem, pairs_with_dem))

    progress_bar.close()

    # Run through each combination of stereo-pairs and try to apply the registration results
    progress_bar = tqdm(total=len(pairs_with_dem))
    for i, pair in enumerate(pairs_with_dem):
        # Update the progress bar description
        progress_bar.desc = f"Processing results for {pair}"
        # Exctract the corresponding result
        result = results[i]
        # Skip if coalignment was not possible
        if result is None:
            progress_bar.update()
            print("Registration not possible")
            continue

        # Skip if the coalignment fitness was poor. I think e.g. 10 means 10 m of offset
        if result["fitness"] > max_fitness:
            progress_bar.update()
            print(f"{pair} fitness exceeded threshold: {result['fitness']}")
            continue

        # Get the ICP centroid as a numpy array (x, y, z) and create tie points from it
        centroid = np.array([float(value) for value in result["centroid"].splitlines()])
        # Offsets to change the centroid (first offsets the x coordinate, third offsets the y coordinate, etc.)
        offsets = [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 0]
        ]
        # Create rows for pandas DataFrames by multiplying the centroid with corresponding offsets
        rows = [centroid + np.array(offset) * tie_group_radius for offset in offsets]
        # Make a source point position DataFrame from the centroid-centered points
        # These will be the initial positions of the points (after which reference information is given).
        source_point_positions = pd.DataFrame(data=rows, columns=["X", "Y", "Z"])

        # Transform the source points to the reference coordinates using the ICP transform
        # These will be the GCP location data.
        destination_point_positions = processing_tools.transform_points(
            source_point_positions, result["composed"], inverse=False)

        def global_to_local(row: pd.Series) -> pd.Series:
            """Convert global coordinates in a pandas row to local coordinates."""
            global_coord = ms.Vector(row[["X", "Y", "Z"]].values)

            # Transform it using the inverse transform matrix of the unprojected coordinate. Geez.
            local_coord = chunk.transform.matrix.inv().mulp(chunk.crs.unproject(global_coord))

            # Make a new identical row, but with new values. This may be replaced by just new_row = list(local_coord)
            new_row = pd.Series(data=list(local_coord), index=["X", "Y", "Z"])

            return new_row

        # Convert the point positions to local coordinates (so point projections can be made)
        local_source_positions = source_point_positions.apply(global_to_local, axis=1)

        def is_projection_valid(camera: ms.Camera, projected_position: ms.Vector) -> bool:
            """
            Check if a camera projection is valid, e.g. if it's not outside the camera bounds.

            :param camera: The camera to check the projection validity on.
            :param projected_position: The pixel position of the projection to evaluate.

            :returns: If the projection seems valid or not.
            """
            if projected_position is None:
                return False
            # Check if the pixel locations are negative (meaning it's invalid)
            if projected_position.x < 0 or projected_position.y < 0:
                return False

            # Check if the projected x position is bigger than the image width
            if projected_position.x > int(camera.photo.meta["File/ImageWidth"]):
                return False
            # Check if the projected y position is bigger than the image height
            if projected_position.y > int(camera.photo.meta["File/ImageHeight"]):
                return False

            # Assume that the projection is valid if it didn't fill any of the above criterai
            return True

        def find_good_cameras(pair: str, positions: pd.DataFrame) -> list[ms.Camera]:
            """
            Find a camera that can "see" all the given positions.

            :param pair: Which stereo-pair to look for a camera in.
            :param positions: The positions to check if they are visible or not.

            :returns: A camera that can "see" all the given positions.
            """
            cameras_in_pair = [camera for camera in chunk.cameras if pair in camera.group.label]

            # Make a count for how many positions the camera can "see", starting at zero
            n_valid = {camera: 0 for camera in cameras_in_pair}

            # Go through each camera and count the visibilities
            for camera in cameras_in_pair:
                # Go through each given position
                for _, position in positions.iterrows():
                    # Project the position onto the camera coordinate system
                    projection = camera.project(position[["X", "Y", "Z"]].values)
                    # Check if the projection was valid
                    if is_projection_valid(camera, projection):
                        n_valid[camera] += 1

            # Find the cameras with valid at least n-1 projections valid (5 positions => 4 valid projections)
            good_cameras = [camera for camera, n_valid_projections in n_valid.items(
            ) if n_valid_projections >= (positions.shape[0] - 1)]

            return good_cameras

        # Find all the cameras that can "see" the local source coordinates.
        cameras = find_good_cameras(pair, local_source_positions)

        assert len(cameras) > 0

        # Set the marker pixel accuracy
        chunk.marker_projection_accuracy = marker_pixel_accuracy
        chunk.marker_location_accuracy = [CONSTANTS.stable_ground_accuracy] * 3

        # Go over each point position and make a Metashape marker from it if it's valid
        for j in range(local_source_positions.shape[0]):
            # Extract the local source position and the "global" destination position.
            local_source_position = local_source_positions.iloc[j]
            destination_position = destination_point_positions.iloc[j]

            # Project the source positions
            source_projections = [camera.project(local_source_position) for camera in cameras]

            # Double check that the projections are valid
            projection_validity: dict[ms.Camera, bool] = {}
            for camera, projection in zip(cameras, source_projections):
                projection_validity[camera] = is_projection_valid(camera, projection)

            # Check that at least two camera projections exist for a marker.
            if np.count_nonzero(list(projection_validity.values())) < 2:
                continue

            error = round(float(result["fitness"]), 2)  # The "fitness" is presumed to be in m
            marker_label = f"gcp_base_to_{pair}_num_{j}_fitness_{error}"
            for marker in chunk.markers:
                if marker_label[:marker_label.index("fitness")] in marker.label:
                    chunk.remove(marker)
            # Add a marker and set an appropriate label
            marker = chunk.addMarker()
            marker.label = marker_label

            marker.reference.location = destination_position

            # Set the projections
            for camera, projection in zip(cameras, source_projections):
                if not projection_validity[camera]:
                    continue
                marker.projections[camera] = ms.Marker.Projection(projection, True)

        progress_bar.update()

    progress_bar.close()
