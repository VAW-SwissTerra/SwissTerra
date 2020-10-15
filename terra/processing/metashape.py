"""Wrapper functions for Agisoft Metashape."""
import concurrent.futures
import itertools
import os
import random
import time
import warnings
from collections import deque, namedtuple
from enum import Enum
from typing import Dict, List, Optional, Tuple

import Metashape as ms
import numpy as np
import pandas as pd
import statictypes
from tqdm import tqdm

from terra import fiducials, files, metadata
from terra.constants import CONSTANTS
from terra.preprocessing import masks
from terra.processing import inputs, processing
from terra.utilities import no_stdout

CACHE_FILES = {}

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


@statictypes.enforce
def is_document(dataset: str) -> bool:
    """
    Check if a Metashape document exists.

    param: dataset: The name of the dataset.

    return: exists: Whether the document exists or not.
    """
    return os.path.isfile(CACHE_FILES[f"{dataset}_metashape_project"])


@statictypes.enforce
def new_document(dataset: str) -> ms.Document:
    """
    Create a new Metashape document.

    param: dataset: The dataset name.

    return: doc: The newly created Metashape document.
    """
    # Make the temporary directory (also makes the project root directory)
    os.makedirs(inputs.CACHE_FILES[f"{dataset}_temp_dir"], exist_ok=True)
    with no_stdout():
        doc = ms.Document()

        doc.save(CACHE_FILES[f"{dataset}_metashape_project"])

    if doc.read_only:
        raise AssertionError("New document is in read-only mode. Is it open?")

    doc.meta["dataset"] = dataset

    return doc


@statictypes.enforce
def load_document(dataset: str) -> ms.Document:
    """
    Load an already existing Metashape document.

    param: dataset: The dataset name.

    return: doc: The newly created Metashape document.
    """
    with no_stdout():
        doc = ms.Document()
        doc.open(CACHE_FILES[f"{dataset}_metashape_project"])

    if doc.read_only:
        raise AssertionError("The loaded document is in read-only mode. Is it open? Remove lock file if not.")

    doc.meta["dataset"] = dataset

    return doc


def save_document(doc: ms.Document) -> None:
    """Save the metashape document."""
    with no_stdout():
        doc.save()


@statictypes.convert
def get_unfinished_chunks(chunks: List[ms.Chunk], step: Step) -> List[ms.Chunk]:
    """
    Check whether a step is finished.

    param: chunks: The chunks to check.
    param: step: The step to check.

    return: unfinished: The chunks whose step is unfinished.
    """
    unfinished: List[ms.Chunk] = []
    for chunk in chunks:
        if step == Step.ALIGNMENT and any(["AlignCameras" not in meta for meta in chunk.meta.keys()]):
            unfinished.append(chunk)

        elif step == Step.DENSE_CLOUD and chunk.dense_cloud is None:
            unfinished.append(chunk)

        elif step == Step.DEM and chunk.elevation is None:
            unfinished.append(chunk)

        elif step == Step.ORTHOMOSAIC and chunk.orthomosaic is None:
            unfinished.append(chunk)

    # Check whether all chunks' step is valid
    return unfinished


@statictypes.enforce
def has_alignment(chunk: ms.Chunk) -> bool:
    """
    Check whether some or all cameras in a chunk are aligned.

    param: chunk: The chunk to check.

    return: any_alignment: Whether at least one camera is aligned.
    """
    any_alignment = any([camera.transform is not None for camera in chunk.cameras])
    return any_alignment


@ statictypes.enforce
def new_chunk(doc: ms.Document, filenames: List[str], chunk_label: Optional[str] = None) -> ms.Chunk:
    """
    Create a new chunk in a document, add photos, and set appropriate parameters for them.

    param: doc: The active Metashape document.
    param: filenames: The names of the files to be added to the chunk.

    return: The newly created Metashape chunk.
    """
    with no_stdout():
        chunk = doc.addChunk()

    if chunk_label is not None:
        chunk.label = chunk_label

    chunk.crs = ms.CoordinateSystem(CONSTANTS["crs_epsg"])
    chunk.camera_location_accuracy = ms.Vector([2] * 3)
    chunk.camera_rotation_accuracy = ms.Vector([1] * 3)

    filepaths = [os.path.join(files.INPUT_DIRECTORIES["image_dir"], filename) for filename in filenames]

    image_meta = inputs.get_dataset_metadata(doc.meta["dataset"])

    with no_stdout():

        chunk.addPhotos(filepaths)

    sensors = {}
    for instrument, dataframe in image_meta.groupby("Instrument"):

        sensor = chunk.addSensor()
        sensor.film_camera = True
        sensor.label = instrument

        sensor.pixel_size = ms.Vector([CONSTANTS["scanning_resolution"] * 1e3] * 2)
        sensor.focal_length = dataframe.iloc[0]["focal_length"]
        generate_fiducials(chunk, sensor)

        sensors[sensor.label] = sensor

    for camera in chunk.cameras:
        sensor_label = image_meta[image_meta["Image file"].str.replace(
            ".tif", "") == camera.label].iloc[0]["Instrument"]
        camera.sensor = sensors[sensor_label]

    for sensor in sensors.values():
        sensor.calibrateFiducials(resolution=CONSTANTS["scanning_resolution"] * 1e3)

    import_reference(chunk, inputs.CACHE_FILES[f"{doc.meta['dataset']}_camera_orientations"])

    groups = {}
    for camera in chunk.cameras:
        camera_row = image_meta[image_meta["Image file"].str.replace(".tif", "") == camera.label].squeeze()
        assert isinstance(camera_row, pd.Series), "Camera row has incorrect filetype"
        group_label = f"station_{str(camera_row['Base number'])}_{camera_row['Position']}"

        if group_label not in groups:
            group = chunk.addCameraGroup()
            group.label = group_label
            group.type = ms.CameraGroup.Type.Station
            groups[group_label] = group

        camera.group = groups[group_label]

        camera.reference.enabled = True
        camera.reference.rotation_enabled = True

    with no_stdout():
        chunk.importMasks(path=os.path.join(masks.CACHE_FILES["mask_dir"], "{filename}.tif"),
                          source=ms.MaskSource.MaskSourceFile, operation=ms.MaskOperation.MaskOperationReplacement)
    return chunk


@statictypes.enforce
def import_reference(chunk: ms.Chunk, filepath: str) -> None:
    """
    Import camera location and orientation information into Metashape.

    The camera location CRS is assumed to be the same as the chunk CRS.
    The camera orientation format is assumed to be in Yaw, Pitch, Roll

    param: chunk: The chunk to add the reference to.
    param: filepath: The filepath of the reference csv.
    """
    chunk.euler_angles = ms.EulerAnglesOPK

    reference_data = pd.read_csv(filepath, index_col="label").astype(float)
    reference_data.index = reference_data.index.str.replace(".tif", "")

    for camera in chunk.cameras:
        cam_data = reference_data.loc[camera.label]
        camera.reference.location = cam_data[["easting", "northing", "altitude"]].values

        camera.reference.rotation = ms.Utils.mat2opk(ms.Utils.ypr2mat(
            ms.Vector(cam_data[["yaw", "pitch", "roll"]].values)))


@statictypes.enforce
def align_cameras(chunk: ms.Chunk, fixed_sensor: bool = False) -> bool:
    """
    Align all cameras in a chunk.

    param: chunk: The chunk whose images shall be aligned.
    param: fixed_sensor: Whether to fix the sensors during the alignment.

    return: aligned: Whether the alignment was successful or not.
    """
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

        # Add extra tie points
        chunk.triangulatePoints()
        chunk.optimizeCameras()

    if fixed_sensor:
        for sensor in chunk.sensors:
            sensor.fixed_calibration = fixed_sensors[sensor]
    return True


@statictypes.enforce
def generate_fiducials(chunk: ms.Chunk, sensor: ms.Sensor) -> None:
    """
    Generate fiducials for the specified sensor.

    param: chunk: The active chunk.
    param: sensor: The sensor to assign the fiducials to.
    """
    matcher = fiducials.fiducials.FrameMatcher(verbose=False)
    new_fiducials = {}
    for corner in ["top", "right", "bottom", "left"]:
        fiducial = chunk.addMarker()
        fiducial.type = ms.Marker.Type.Fiducial
        fiducial.label = f"{sensor.label}_{corner}"

        fiducial.sensor = sensor

        new_fiducials[corner] = fiducial

    fiducial_projections = matcher.calculate_fiducial_projections()

    for camera in chunk.cameras:
        projections = fiducial_projections[camera.label + ".tif"]

        for corner, fiducial in new_fiducials.items():
            fid_projection = projections._asdict()[corner]
            fiducial.projections[camera] = ms.Marker.Projection(
                ms.Vector([fid_projection.x, fid_projection.y]), True)


@statictypes.convert
def build_dense_clouds(chunk: ms.Chunk, pairs: List[str], quality: Quality = Quality.HIGH,
                       filtering: Filtering = Filtering.AGGRESSIVE) -> None:
    """
    Generate dense clouds for all stereo-pairs in a given chunk.

    param: chunk: The chunk to process.
    param: pairs: A list of the stereo-pairs to process.
    param: quality: The quality of the dense cloud.
    param: filtering: The dense cloud filtering setting.
    """
    for pair in tqdm(pairs, desc="Building dense clouds for stereo-pairs"):
        for camera in chunk.cameras:
            # Set it as enabled if the camera group label fits with the stereo pair label
            camera.enabled = pair in camera.group.label

        with no_stdout():
            chunk.buildDepthMaps(downscale=quality.value, filter_mode=filtering.value)
            try:
                chunk.buildDenseCloud(point_confidence=True)
            except Exception as exception:
                if "Zero resolution" in str(exception):
                    continue
                raise exception

        chunk.dense_cloud.label = pair

        # Remove all points with a low confidence
        chunk.dense_cloud.setConfidenceFilter(0, CONSTANTS["dense_cloud_min_confidence"] - 1)
        chunk.dense_cloud.removePoints(list(range(128)))
        chunk.dense_cloud.resetFilters()

        # Unset the dense cloud as default to allow for more dense clouds to be constructed
        chunk.dense_cloud = None

    # Reset the enabled flags
    for camera in chunk.cameras:
        camera.enabled = True


@statictypes.convert
def old_build_dems(chunks: List[ms.Chunk], dataset: str) -> None:
    """
    Build DEMs using PDAL and GDAL.

    param: chunks: The chunks to build DEMs for.
    param: dataset: The name of the dataset.
    """
    warnings.warn("DEM function is outdated", Warning)
    # Fill this with tuples: (dense_cloud_path, dem_path)
    filepaths: List[Tuple[str, str]] = []

    print("Exporting dense clouds")
    for chunk in tqdm(chunks):
        dense_cloud_path = os.path.join(inputs.CACHE_FILES[f"{dataset}_temp_dir"], f"{chunk.label}_dense.ply")
        dem_path = os.path.join(inputs.CACHE_FILES[f"{dataset}_temp_dir"], f"{chunk.label}_DEM.tif")
        filepaths.append((dense_cloud_path, dem_path))
        # Export points from metashape
        with no_stdout(disable=False):
            chunk.exportPoints(dense_cloud_path, crs=chunk.crs)

    print("Generating DEMs")
    time.sleep(0.2)

    progress_bar = tqdm(total=len(chunks))

    def dem_thread_process(cloud_and_dem_paths: Tuple[str, str]) -> None:
        """
        Execute DEM generation in a thread.

        param: cloud_and_dem_paths: A tuple of (dense_cloud_path, dem_path).
        """
        processing.generate_dem(*cloud_and_dem_paths)
        progress_bar.update()

    # Generate DEMs for all point clouds in multiple threads.
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONSTANTS["max_point_cloud_threads"]) as executor:
        # Evaluate the generators into an empty deque
        deque(executor.map(dem_thread_process, filepaths), maxlen=0)

    # Import the results to Metashape
    for i, chunk in enumerate(chunks):
        with no_stdout():
            chunk.importRaster(path=filepaths[i][1], crs=chunk.crs, raster_type=ms.DataSource.ElevationData)


@statictypes.convert
def build_orthomosaics(chunks: List[ms.Chunk], resolution: float) -> None:
    """
    Build orthomosaics for each chunk.

    param: chunks: The chunks to build orthomosaics for.
    param: resolution: The orthomosaic resolution in metres.
    """
    warnings.warn("Orthomosaic function is outdated", Warning)
    for chunk in tqdm(chunks):
        with no_stdout(disable=False):
            chunk.buildOrthomosaic(surface_data=ms.DataSource.ElevationData, resolution=resolution)


def merge_chunks(doc: ms.Document, chunks: List[ms.Chunk], optimize: bool = True, remove_old: bool = False) -> ms.Chunk:
    """
    Merge Metashape chunks.

    param: doc: The Metashape document.
    param: chunks: A list of chunks to merge.
    param: optimize: Whether to optimize cameras after merging.
    param: remove_old: Whether to remove

    return: merged_chunk: The merged chunk.

    """
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

    for marker in merged_chunk.markers:
        if not marker.type == ms.Marker.Type.Fiducial:
            continue
        for sensor_label in sensors:
            if sensor_label in marker.label:
                marker.sensor = sensors[sensor_label]

    for label, sensor in sensors.items():
        if label == "unknown":
            continue
        sensor.calibrateFiducials(resolution=CONSTANTS["scanning_resolution"] * 1e3)

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

    param: doc: The Metashape document to check
    param: dataset: The name of the dataset.

    return: merged_chunk: The chunk with all stereo-pairs.
    """
    # Load image metadata to get the station numbers
    image_meta = inputs.get_dataset_metadata(dataset)

    # First check if the Merged Chunk already exists.
    for chunk in doc.chunks:
        if chunk.label == "Merged Chunk":
            merged_chunk = chunk
            merged_chunk.meta["dataset"] = dataset
            return merged_chunk

    aligned_chunks: List[ms.Chunk] = []
    print("Aligning stations")
    # Loop through all stations (stereo-pairs) and align them if they don't already exist
    for station_number, station_meta in tqdm(image_meta.groupby("Base number")):
        # Check if a "stereo-pair" only has the right or left position (so it's not a stereo-pair)
        if station_meta["Position"].unique().shape[0] < 2:
            print(f"Station {station_number} only has position {station_meta['Position'].iloc[0]}. Skipping.")
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

    print("Merging chunks")
    merged_chunk = merge_chunks(doc, aligned_chunks, remove_old=True, optimize=True)
    merged_chunk.meta["dataset"] = dataset

    return merged_chunk


@statictypes.enforce
def get_marker_reprojection_error(camera: ms.Camera, marker: ms.Marker) -> np.float64:
    """
    Get the reprojection error between a marker's projection and a camera's reprojected position.

    param: camera: The camera to reproject the marker position to.
    param: marker: The marker to compare the projection vs. reprojection on.
    """
    projected_position = marker.projections[camera].coord
    reprojected_position = camera.project(marker.position)

    if reprojected_position is None:
        return np.NaN

    diff = (projected_position - reprojected_position).norm()

    image_size = max(camera.photo.meta["File/ImageWidth"], camera.photo.meta["File/ImageHeight"])
    # Check if the reprojected position difference is too high
    if diff > image_size:
        return np.NaN

    return diff


@statictypes.enforce
def get_mean_marker_reprojection_errors(markers: List[ms.Marker]) -> np.float64:
    """
    Calculate the mean reprojection error for a marker, checked on all pinned projections.
    """
    errors = {marker: List[np.float64]() for marker in markers}

    for marker in markers:
        for camera, projection in marker.projections.items():
            if not projection.pinned:
                continue
            errors[marker].append(get_marker_reprojection_error(camera, marker))

    mean_errors = {marker: np.nanmean(error_list, dtype=np.float64) for marker, error_list in errors.items()}

    return mean_errors


def get_asift_markers(chunk):
    """Get markers from ASIFT matching."""
    warnings.warn(DeprecationWarning)
    all_candidates = metadata.image_meta.get_matching_candidates()

    cameras = {camera.label + ".tif": camera for camera in chunk.cameras}
    matching_candidates = [candidates for candidates in all_candidates if candidates[0]
                           in cameras and candidates[1] in cameras]

    for filename1, filename2 in tqdm(matching_candidates):
        filepath1 = os.path.join(files.INPUT_DIRECTORIES["image_dir"], filename1)
        filepath2 = os.path.join(files.INPUT_DIRECTORIES["image_dir"], filename2)

        matches = processing.match_asift(filepath1, filepath2, verbose=False)

        for i, match in matches.iterrows():
            marker = chunk.addMarker()

            for tag, filename in zip(["img1", "img2"], [filename1, filename2]):
                marker.projections[cameras[filename]] = ms.Marker.Projection(
                    ms.Vector(match[[f"{tag}_x", f"{tag}_y"]].values), True)

            errors = []
            for filename in [filename1, filename2]:
                errors.append(get_marker_reprojection_error(cameras[filename], marker))

            if np.mean(errors) == np.NaN:
                chunk.remove(marker)
                continue

            marker.label = f"asift_{filename1}_{filename2}_{i}"


@ statictypes.enforce
def get_chunk_stereo_pairs(chunk: ms.Chunk) -> List[str]:
    """
    Get a list of stereo-pair group names.

    param: chunk: The chunk to analyse.

    return: pairs: A list of stereo-pair group names.

    """
    pairs: List[str] = []
    for group in chunk.camera_groups:
        pair = group.label.replace("_R", "").replace("_L", "")

        if pair not in pairs:
            pairs.append(pair)

    return pairs


@ statictypes.enforce
def get_unfinished_pairs(chunk: ms.Chunk, step: Step) -> List[str]:
    """
    List all stereo-pairs that have not yet finished a step.

    param: chunk: The chunk to analyse.
    param: step: The step to check for.

    return: unfinished_pairs: A list of stereo-pairs that are unfinished.
    """
    pairs = get_chunk_stereo_pairs(chunk)

    if step == Step.DENSE_CLOUD:
        labels_to_check = [cloud.label for cloud in chunk.dense_clouds]
    else:
        raise NotImplementedError()

    unfinished_pairs = [pair for pair in pairs if pair not in labels_to_check]

    return unfinished_pairs


def build_dems(chunk: ms.Chunk, pairs: List[str], redo: bool = False) -> Dict[str, str]:
    """
    Build DEMs in local coordinates.

    param: chunk: The chunk to export from.
    param: pairs: Which stereo-pairs to use.
    param: redo: Whether to remake dense clouds or DEMs if they already exist.

    return: filepaths: The filepaths of the exported clouds for each stereo-pair.
    """
    assert chunk.meta["dataset"] is not None

    filepaths: Dict[str, str] = {}

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
                chunk.exportPoints(cloud_filepath, crs=chunk.crs)

    chunk.dense_cloud = None

    progress_bar = tqdm(total=len(filepaths), desc="Generating DEMs")

    def build_dem(pair_and_filepath: Tuple[str, str]) -> Tuple[str, str]:
        """
        Generate a DEM from a point cloud.

        param: pair: The stereo-pair label
        param: filepath: The path to the input point cloud.

        return: (pair, output_filepath): The stereo-pair label and the path to the output DEM.
        """
        pair, filepath = pair_and_filepath
        output_filepath = os.path.splitext(filepath)[0] + "_DEM.tif"
        if redo or not os.path.isfile(output_filepath):
            processing.generate_dem(filepath, output_dem_path=output_filepath, resolution=5.0, interpolate_pixels=0)

        progress_bar.update()

        return pair, output_filepath

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        dem_filepaths = dict(executor.map(build_dem, filepaths.items()))

    progress_bar.close()
    return dem_filepaths


def coalign_stereo_pairs(chunk: ms.Chunk, pairs: List[str]):
    """
    Use DEM ICP coaligning to align combinations of stereo-pairs.

    param: chunk: The chunk to analyse.
    param: pairs: The stereo-pairs to coaling.
    """
    # Build local DEMs to subsequently coalign
    dem_paths = build_dems(chunk, pairs=pairs)

    pair_combinations = list(itertools.combinations([pair for pair in pairs if pair in dem_paths], r=2))
    path_combinations = [(dem_paths[first], dem_paths[second]) for first, second in pair_combinations]

    progress_bar = tqdm(total=len(path_combinations), desc="Coaligning DEM pairs")

    def coalign_dems(path_combination):
        path_1, path_2 = path_combination
        result = processing.coalign_dems(path_1, path_2)
        progress_bar.update()

        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(coalign_dems, path_combinations))

    progress_bar.close()

    # Run through each combination of stereo-pairs and try to coalign the DEMs
    progress_bar = tqdm(total=len(pair_combinations))
    for i, (pair_1, pair_2) in enumerate(pair_combinations):
        progress_bar.desc = f"Processing results for {pair_1} and {pair_2}"
        result = results[i]
        # Skip if coalignment was not possible
        if result is None:
            progress_bar.update()
            continue

        if result["fitness"] > 10:
            progress_bar.update()
            print(f"{pair_1} to {pair_2} fitness exceeded threshold: {result['fitness']}")
            continue

        # TODO: Use the alignment centroid to make points instead of a random tie point
        # TODO: Add fitting threshold

        # Load the first image in both stereo-pairs
        # The first pair acts as reference and the second pair acts as aligned
        # This only matters for the script here.
        centroid = np.array([float(value) for value in result["centroid"].splitlines()])
        offsets = [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1]
        ]
        rows = [centroid + np.array(offset) * 30 for offset in offsets]
        aligned_point_positions = pd.DataFrame(data=rows, columns=["X", "Y", "Z"])
        reference_point_positions = processing.transform_points(aligned_point_positions, result["composed"])

        def global_to_local(row: pd.Series) -> ms.Vector:
            global_coord = ms.Vector(row[["X", "Y", "Z"]].values)

            local_coord = chunk.transform.matrix.inv().mulp(chunk.crs.unproject(global_coord))

            new_row = pd.Series(data=list(local_coord), index=["X", "Y", "Z"])

            return new_row

        local_aligned_positions = aligned_point_positions.apply(global_to_local, axis=1)
        local_reference_positions = reference_point_positions.apply(global_to_local, axis=1)

        def projection_valid(camera: ms.Camera, projected_position: ms.Vector):
            if projected_position is None:
                return False
            if projected_position.x < 0 or projected_position.y < 0:
                return False

            if projected_position.x > int(camera.photo.meta["File/ImageWidth"]):
                return False
            if projected_position.y > int(camera.photo.meta["File/ImageHeight"]):
                return False

            return True

        @ statictypes.enforce
        def find_good_image(pair: str, positions: pd.DataFrame) -> ms.Camera:
            images = [camera for camera in chunk.cameras if pair in camera.group.label]

            n_valid = {image: 0 for image in images}

            for image in images:
                for i, position in positions.iterrows():
                    projection = image.project(position[["X", "Y", "Z"]].values)
                    if projection_valid(image, projection):
                        n_valid[image] += 1

            best_image = max(n_valid, key=n_valid.get)

            return best_image

        reference_image = find_good_image(pair_1, local_reference_positions)
        aligned_image = find_good_image(pair_2, local_aligned_positions)

        for j in range(aligned_point_positions.shape[0]):
            reference_position = local_reference_positions.iloc[j]
            aligned_position = local_aligned_positions.iloc[j]

            reference_projection = reference_image.project(reference_position)
            aligned_projection = aligned_image.project(aligned_position)

            if not projection_valid(reference_image, reference_projection):
                print(f"Reference projection was invalid: {reference_projection}")
                continue
            if not projection_valid(aligned_image, aligned_projection):
                print(f"Aligned projection invalid: {aligned_projection}")
                continue

            marker = chunk.addMarker()
            error = round(float(result["fitness"]), 3)
            marker.label = f"tie_{pair_1}_to_{pair_2}_{j}_error_{error}"

            marker.projections[reference_image] = ms.Marker.Projection(reference_projection, True)
            marker.projections[aligned_image] = ms.Marker.Projection(aligned_projection, True)

        progress_bar.update()

    progress_bar.close()

    with no_stdout():
        chunk.optimizeCameras()
