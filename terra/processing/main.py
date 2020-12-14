"""The main processing pipeline."""

import os
import shutil
import warnings

import Metashape as ms
import statictypes

from terra import files
from terra.constants import CONSTANTS  # pylint: disable=no-name-in-module
from terra.processing import inputs, metashape_tools
from terra.utilities import no_stdout, notify


@statictypes.enforce
def process_dataset(dataset: str, redo: bool = False) -> None:
    """
    Process a dataset from start to finish.

    param: dataset: The name of the dataset to process
    param: redo: Whether to redo steps that already exist.

    """
    dataset = "Wild109"
    warnings.warn("Hardcoded instrument Wild109")

    # Load the metashape document or create a new one
    if not redo and metashape_tools.is_document(dataset):
        doc = metashape_tools.load_document(dataset)
        print(f"Loaded Metashape document for {dataset}")
    else:
        inputs.export_camera_orientation_csv(dataset)
        doc = metashape_tools.new_document(dataset)
        shutil.rmtree(inputs.CACHE_FILES[f"{dataset}_temp_dir"])
        os.makedirs(inputs.CACHE_FILES[f"{dataset}_temp_dir"])
        print(f"Created new Metashape document for {dataset}")

    # Load or create a chunk with all the stereo-pairs
    chunk = metashape_tools.load_or_remake_chunk(doc, dataset)

    # Get the names of the stereo-pairs
    pairs = metashape_tools.get_chunk_stereo_pairs(chunk)

    metashape_tools.save_document(doc)

    # TODO: Remove this when ICP seems to be working.
    # Import reference markers and fit the camera model
    if False:
        for marker in chunk.markers:
            if "Tie " in marker.label:
                break
        else:
            chunk.importMarkers(os.path.join(files.INPUT_ROOT_DIRECTORY, "datasets", "rhone", "tie_points.xml"))
            metashape_tools.optimize_cameras(chunk)

    # Remove the reference markers but keep the camera model fixed onward.
    if False:
        for marker in chunk.markers:
            if "Tie " in marker.label:
                chunk.remove(marker)
    metashape_tools.optimize_cameras(chunk, fixed_sensors=True)

    # Check which pairs do not yet have a dense cloud
    unfinished_pairs = metashape_tools.get_unfinished_pairs(chunk, metashape_tools.Step.DENSE_CLOUD)
    # Make missing dense clouds
    if len(unfinished_pairs) > 0:
        metashape_tools.build_dense_clouds(chunk, pairs=unfinished_pairs, quality=metashape_tools.Quality.ULTRA,
                                           filtering=metashape_tools.Filtering.MILD)
    metashape_tools.save_document(doc)

    # Coalign stereo-pair DEMs with each other and generate markers from their alignment
    metashape_tools.coalign_stereo_pairs(chunk, pairs=pairs, marker_pixel_accuracy=2)
    metashape_tools.optimize_cameras(chunk, fixed_sensors=True)
    metashape_tools.remove_bad_markers(chunk, marker_error_threshold=3)
    metashape_tools.optimize_cameras(chunk, fixed_sensors=True)

    metashape_tools.build_dense_clouds(chunk, pairs=pairs, quality=metashape_tools.Quality.HIGH,
                                       filtering=metashape_tools.Filtering.MILD, all_together=True)

    print("Generating DEM")
    with no_stdout():
        chunk.buildDem()

    # metashape.build_dense_clouds(chunk, pairs=pairs, quality=metashape.Quality.HIGH,
    #                             filtering=metashape.Filtering.MILD)

    # Run a bundle adjustment with these new markers.
    # metashape.optimize_cameras(chunk)

    # Remove markers that still have an unreasonable error and run another bundle adjustment.
    # metashape.optimize_cameras(chunk)
    metashape_tools.save_document(doc)

    notify(f"{dataset} finished")
    return

    # Rebuild the dense clouds with the proper pose
    # metashape.build_dense_clouds(chunk, pairs=pairs, quality=metashape.Quality.ULTRA,
    #                             filtering=metashape.Filtering.MILD)
    metashape_tools.save_document(doc)

    metashape_tools.build_dems(chunk=chunk, pairs=pairs, resolution=CONSTANTS.dem_resolution)
    metashape_tools.save_document(doc)

    chunks_to_process = metashape_tools.get_unfinished_chunks(aligned_chunks, metashape_tools.Step.DENSE_CLOUD)
    if len(chunks_to_process) > 0:
        print("Building dense clouds")
        metashape_tools.build_dense_clouds(chunks_to_process, quality=metashape_tools.Quality.HIGH)
        metashape_tools.save_document(doc)

    chunks_to_process = metashape_tools.get_unfinished_chunks(aligned_chunks, metashape_tools.Step.DEM)
    if len(chunks_to_process) > 0:
        print("Building DEMs")
        metashape_tools.build_dems(chunks_to_process, dataset)
        metashape_tools.save_document(doc)

    return

    chunks_to_process = metashape_tools.get_unfinished_chunks(aligned_chunks, metashape_tools.Step.ORTHOMOSAIC)
    if len(chunks_to_process) > 0:
        print("Building orthomosaics")
        metashape_tools.build_orthomosaics(chunks_to_process, resolution=1)
        metashape_tools.save_document(doc)
