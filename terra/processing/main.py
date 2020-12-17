"""The main processing pipeline."""

import csv
import os
import shutil
import time

from terra.processing import inputs, metashape_tools
from terra.utilities import notify


def log(dataset: str, event: str):
    """
    Log an event to a file to keep track of the progress.

    :param dataset: The name of the current dataset.
    :param event: Text describing the event.
    """
    current_time = time.strftime("UTC %Y/%m/%d %H:%M:%S", time.gmtime(time.time()))
    os.makedirs(os.path.dirname(inputs.CACHE_FILES["log_filepath"]), exist_ok=True)
    with open(inputs.CACHE_FILES["log_filepath"], "a+") as outfile:
        writer = csv.writer(outfile)
        writer.writerow([current_time, dataset, event])


def run_processing_pipeline(dataset: str, redo: bool = False) -> None:
    """
    Process a dataset from start to finish.

    param: dataset: The name of the dataset to process
    param: redo: Whether to redo steps that already exist.
    """
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
    log(dataset, "Dataset is aligned")

    # Get the names of the stereo-pairs
    pairs = metashape_tools.get_chunk_stereo_pairs(chunk)

    metashape_tools.save_document(doc)

    # Check if coalignment should be done. If there is no "tie_station*" marker, it hasn't been done yet
    coalign = not any(["tie_station" in marker.label for marker in chunk.markers])
    if coalign:
        # Check which pairs do not yet have a dense cloud
        missing_clouds_pairs = metashape_tools.get_unfinished_pairs(chunk, metashape_tools.Step.DENSE_CLOUD)
        # Make missing dense clouds
        if len(missing_clouds_pairs) > 0:
            metashape_tools.build_dense_clouds(chunk, pairs=missing_clouds_pairs, quality=metashape_tools.Quality.ULTRA,
                                               filtering=metashape_tools.Filtering.MILD)
        metashape_tools.save_document(doc)

        # Coalign stereo-pair DEMs with each other and generate markers from their alignment
        metashape_tools.coalign_stereo_pairs(chunk, pairs=pairs, marker_pixel_accuracy=2)
        metashape_tools.optimize_cameras(chunk, fixed_sensors=True)
        metashape_tools.remove_bad_markers(chunk, marker_error_threshold=3)
        metashape_tools.optimize_cameras(chunk, fixed_sensors=True)

        # Remove all coalignment DEMs in the temp folder to not confuse them with the better output DEMs
        for filename in os.listdir(inputs.CACHE_FILES[f"{dataset}_temp_dir"]):
            if filename.endswith(".tif"):
                os.remove(os.path.join(inputs.CACHE_FILES[f"{dataset}_temp_dir"], filename))

        log(dataset, "Coalignment successful")
    else:
        log(dataset, "Coalignment already exists")

    # Generate dense clouds for all stereo-pairs that do not yet have one.
    missing_clouds_pairs = metashape_tools.get_unfinished_pairs(chunk, step=metashape_tools.Step.DENSE_CLOUD)
    if len(missing_clouds_pairs) > 0:
        print(f"Building {len(missing_clouds_pairs)} dense clouds")
        time.sleep(0.3)  # This is to make the above statement come before the code below. Why? I have no idea!
        successful = metashape_tools.build_dense_clouds(
            chunk=chunk,
            pairs=missing_clouds_pairs,
            quality=metashape_tools.Quality.ULTRA,
            filtering=metashape_tools.Filtering.MILD
        )
        log(dataset, f"Made {len(successful)} dense clouds")
    metashape_tools.save_document(doc)

    # Generate DEMs for all stereo-pairs that do not yet have one.
    missing_dem_pairs = metashape_tools.get_unfinished_pairs(chunk, step=metashape_tools.Step.DEM)
    if len(missing_dem_pairs) > 0:
        print(f"Building {len(missing_dem_pairs)} DEMs")
        dem_filepaths = metashape_tools.build_dems(chunk=chunk, pairs=missing_dem_pairs)

        # Copy the DEMs to the export directory
        os.makedirs("export/dems", exist_ok=True)
        for filepath in dem_filepaths.values():
            shutil.copyfile(filepath, os.path.join("export/dems", os.path.basename(filepath)))
        log(dataset, f"Made {len(dem_filepaths)} DEMs")
    metashape_tools.save_document(doc)

    # Generate orthomosaics for all stereo-pairs that do not yet have one.
    missing_ortho_pairs = metashape_tools.get_unfinished_pairs(chunk, step=metashape_tools.Step.ORTHOMOSAIC)
    if len(missing_ortho_pairs) > 0:
        successful = metashape_tools.build_orthomosaics(chunk=chunk, pairs=missing_ortho_pairs, resolution=1)
        print(f"Made {len(successful)} orthomosaics")
    metashape_tools.save_document(doc)
    metashape_tools.export_orthomosaics(chunk=chunk, pairs=pairs, directory="export/orthos", overwrite=False)

    # Remove all ply files (dense clouds that take up a lot of space and are saved in Metashape anyway)
    for filename in os.listdir(inputs.CACHE_FILES[f"{dataset}_temp_dir"]):
        if filename.endswith(".ply"):
            os.remove(os.path.join(inputs.CACHE_FILES[f"{dataset}_temp_dir"], filename))

    notify(f"{dataset} finished")
    log(dataset, "Processing finished")


def process_dataset(dataset: str, redo: bool = False):
    """
    Run the processing pipeline and log when it starts, fails or finishes.

    :param dataset: The name of the dataset to process.
    :param redo: Whether to start the analysis from scratch.
    """
    log(dataset, "Processing started")

    try:
        run_processing_pipeline(dataset, redo=redo)
    except Exception as exception:
        log(dataset, "Processing failed")
        raise exception
