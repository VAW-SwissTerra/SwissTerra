"""Wrapper functions for Agisoft Metashape."""
import concurrent.futures
import os
import time
from collections import deque
from enum import Enum
from typing import List, Optional, Tuple

import Metashape as ms
import pandas as pd
import statictypes
from tqdm import tqdm

from terra import fiducials, files, metadata
from terra.constants import CONSTANTS
from terra.preprocessing import masks
from terra.processing import inputs, main, processing
from terra.utilities import no_stdout

CACHE_FILES = {
}


for _dataset in inputs.DATASETS:
    CACHE_FILES[f"{_dataset}_metashape_project"] = os.path.join(
        inputs.CACHE_FILES[f"{_dataset}_dir"], f"{_dataset}.psx")


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

    print(f"Created new Metashape project for {dataset}")

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

    with no_stdout():
        doc.save()


class Step(Enum):

    ALIGNMENT = 1
    DENSE_CLOUD = 2
    DEM = 3
    ORTHOMOSAIC = 4


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

    return: alignment: Whether at least one camera is aligned.
    """

    return any([camera.transform is not None for camera in chunk.cameras])


@ statictypes.enforce
def new_chunk(doc: ms.Document, filenames: List[str], chunk_label: Optional[str] = None) -> ms.Chunk:
    """
    Create a new chunk in a document, add photos, and set appropriate parameters for them.

    param: doc: The active Metashape document.
    param: filenames: The names of the files to be added to the chunk.

    return: The newly created Metashape chunk.
    """
    # TODO: Add explicit yaw, pitch, roll as rotation format.
    with no_stdout():
        chunk = doc.addChunk()

    if chunk_label is not None:
        chunk.label = chunk_label

    chunk.crs = ms.CoordinateSystem(CONSTANTS["crs_epsg"])

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

    with no_stdout():
        chunk.importReference(inputs.CACHE_FILES[f"{doc.meta['dataset']}_camera_orientations"],
                              delimiter=",", create_markers=False, columns="nxyzabc",
                              items=ms.ReferenceItems.ReferenceItemsCameras, format=ms.ReferenceFormat.ReferenceFormatCSV,
                              crs=chunk.crs)

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
def align_cameras(chunk: ms.Chunk) -> bool:
    """
    Align all cameras in a chunk.

    param: chunk: The chunk whose images shall be aligned.

    return: aligned: Whether the alignment was successful or not.
    """

    with no_stdout():
        chunk.matchPhotos(reference_preselection=True, filter_mask=True)
        chunk.alignCameras()

        # Check if no cameras were aligned (if all transforms are None)
        if all([camera.transform is None for camera in chunk.cameras]):
            return False

        # Add extra tie points
        chunk.triangulatePoints()
        chunk.optimizeCameras()
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


class Quality(Enum):
    """Dense cloud quality."""

    ULTRA: int = 1
    HIGH: int = 2
    MEDIUM: int = 4
    LOW: int = 8


@statictypes.convert
def build_dense_clouds(chunks: List[ms.Chunk], quality: Quality = Quality.HIGH) -> None:
    """
    Generate dense clouds for all given chunks.

    param: chunks: The list of chunks to process.
    param: quality: The quality of the dense cloud.
    """
    for chunk in tqdm(chunks):
        with no_stdout():
            chunk.buildDepthMaps(downscale=quality.value, filter_mode=ms.FilterMode.AggressiveFiltering)
            chunk.buildDenseCloud(point_confidence=True)


@statictypes.convert
def build_dems(chunks: List[ms.Chunk], dataset: str) -> None:
    """
    Build DEMs using PDAL and GDAL.

    param: chunks: The chunks to build DEMs for.
    param: dataset: The name of the dataset.
    """
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

    for chunk in tqdm(chunks):
        with no_stdout(disable=False):
            chunk.buildOrthomosaic(surface_data=ms.DataSource.ElevationData, resolution=resolution)
