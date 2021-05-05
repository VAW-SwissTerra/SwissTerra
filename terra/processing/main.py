"""The main processing pipeline."""

import csv
import os
import shutil
import time

import Metashape as ms
from tqdm import tqdm

from terra import files
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

    print(f"\n{current_time} ({dataset}): {event}\n")


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
    try:
        chunk = metashape_tools.load_or_remake_chunk(doc, dataset)
    except Exception as exception:
        # One of these errors come up if no chunks were successfully aligned.
        for error in ["Empty chunk list", "No aligned cameras", "Not enough cameras"]:
            if error in str(exception):
                log(dataset, "Alignment not possible")
                return
        raise exception
    log(dataset, "Dataset is aligned")

    # Make sure that the region is representative of the dataset.
    chunk.resetRegion()
    chunk.region.size *= 2

    # Get the names of the stereo-pairs
    pairs = metashape_tools.get_chunk_stereo_pairs(chunk)

    metashape_tools.save_document(doc)

    # Check if coalignment should be done. If there is no "tie_station*" marker, it hasn't been done yet
    coalign = not any(["tie_station" in marker.label for marker in chunk.markers])
    if coalign:
        # Check which pairs do not yet have a dense cloud
        pairs_missing_clouds = metashape_tools.get_unfinished_pairs(chunk, metashape_tools.Step.DENSE_CLOUD)
        # Make missing dense clouds
        if len(pairs_missing_clouds) > 0:
            metashape_tools.build_dense_clouds(doc, chunk, pairs=pairs_missing_clouds,
                                               quality=metashape_tools.Quality.ULTRA,
                                               filtering=metashape_tools.Filtering.MILD)
            metashape_tools.save_document(doc)

        # Coalign stereo-pair DEMs with each other and generate markers from their alignment
        metashape_tools.stable_ground_registration(chunk, pairs=pairs, marker_pixel_accuracy=2)
        metashape_tools.coalign_stereo_pairs(chunk, pairs=pairs, marker_pixel_accuracy=2)
        metashape_tools.optimize_cameras(chunk, fixed_sensors=True)
        # 23 pixels is half a millimeter.
        metashape_tools.remove_bad_markers(chunk, marker_error_threshold=23)
        # Remove cameras with unreasonable estimated pitch values.
        metashape_tools.remove_bad_cameras(chunk, pitch_error_threshold=40)
        metashape_tools.optimize_cameras(chunk, fixed_sensors=False)

        metashape_tools.save_document(doc)

        # Remove all coalignment DEMs in the temp folder to not confuse them with the subsequent output DEMs
        for filename in os.listdir(inputs.CACHE_FILES[f"{dataset}_temp_dir"]):
            if filename.endswith(".tif"):
                os.remove(os.path.join(inputs.CACHE_FILES[f"{dataset}_temp_dir"], filename))

        # Remove all intermediate coalignment DEMs in the Metashape project
        for dem in chunk.elevations:
            chunk.remove(dem)

        chunk.resetRegion()
        chunk.region.size *= 2
        log(dataset, "Coalignment successful")
    else:
        log(dataset, "Coalignment already exists")

    # Generate dense clouds for all stereo-pairs that do not yet have one.
    pairs_missing_clouds = metashape_tools.get_unfinished_pairs(chunk, step=metashape_tools.Step.DENSE_CLOUD)
    if len(pairs_missing_clouds) > 0:
        print(f"Building {len(pairs_missing_clouds)} dense clouds")
        time.sleep(0.3)  # This is to make the above statement come before the code below. Why? I have no idea!
        successful = metashape_tools.build_dense_clouds(
            doc=doc,
            chunk=chunk,
            pairs=pairs_missing_clouds,
            quality=metashape_tools.Quality.ULTRA,
            filtering=metashape_tools.Filtering.MILD
        )
        log(dataset, f"Made {len(successful)} dense clouds")
    metashape_tools.save_document(doc)

    # Generate DEMs for all stereo-pairs.
    pairs_with_cloud = [cloud.label for cloud in chunk.dense_clouds]
    if len(pairs_with_cloud) > 0:
        print(f"Building {len(pairs_with_cloud)} DEMs")
        dem_filepaths = metashape_tools.build_dems(chunk=chunk, pairs=pairs_with_cloud, redo=True)
        log(dataset, f"Made {len(dem_filepaths)} DEMs")

        # Copy the DEMs to the export directory
        os.makedirs(os.path.join(inputs.TEMP_DIRECTORY, "output/dems"), exist_ok=True)
        for filepath in dem_filepaths.values():
            shutil.copyfile(filepath, os.path.join(inputs.TEMP_DIRECTORY, "output/dems", os.path.basename(filepath)))
    metashape_tools.save_document(doc)

    # Generate orthomosaics for all stereo-pairs that do not yet have one.
    missing_ortho_pairs = metashape_tools.get_unfinished_pairs(chunk, step=metashape_tools.Step.ORTHOMOSAIC)
    if len(missing_ortho_pairs) > 0:
        successful = metashape_tools.build_orthomosaics(chunk=chunk, pairs=missing_ortho_pairs, resolution=1)
        print(f"Made {len(successful)} orthomosaics")
    metashape_tools.save_document(doc)
    metashape_tools.export_orthomosaics(chunk=chunk, pairs=pairs,
                                        directory=os.path.join(inputs.TEMP_DIRECTORY, "output/orthos"), overwrite=True)

    # Remove all ply files (dense clouds that take up a lot of space)
    for filename in os.listdir(inputs.CACHE_FILES[f"{dataset}_temp_dir"]):
        if filename.endswith(".ply"):
            os.remove(os.path.join(inputs.CACHE_FILES[f"{dataset}_temp_dir"], filename))

    # Remove all dense clouds in the Metashape project.
    for dense_cloud in chunk.dense_clouds:
        chunk.remove(dense_cloud)

    metashape_tools.save_document(doc)

    notify(f"{dataset} finished")
    print(f"\n\n{dataset} finished\n\n")
    log(dataset, "Processing finished")


def process_failed_stereo_pairs(redo=False):
    """Try to reprocess all stereo pairs that failed in the main workflow."""

    if redo or not metashape_tools.is_document("failed_pairs"):
        processed_datasets = [fp for fp in os.listdir(inputs.TEMP_DIRECTORY) if ("Wild" in fp) or ("Zeiss" in fp)]
        # Load all documents, find the pairs without alignment, save the chunk, append the chunks to a big project.
        doc = metashape_tools.new_document("failed_pairs")

        # This will be filled with sensor calibrations from the "Merged Chunk"s of all datasets
        sensor_calibrations: dict[str, dict[str, ms.Calibration]] = {}

        for dataset in tqdm(processed_datasets, desc="Fetching failed stations from datasets."):
            # Load the document corresponding to the dataset.
            dataset_doc = metashape_tools.load_document(dataset)

            sensor_calibrations[dataset] = {}

            # Instantiate the list of chunks to append to the larger document.
            chunks_to_save: list[ms.Chunk] = []
            for chunk in dataset_doc.chunks:

                chunk.meta["dataset"] = dataset
                # If it's the merged chunk, extract all estimated sensor calibrations.
                if "Merged" in chunk.label:
                    for sensor in chunk.sensors:
                        if "unknown" in sensor.label:
                            continue

                        sensor_calibrations[dataset][sensor.label] = sensor.calibration.copy()

                # If it's a failed station (it was never merged), append it to the chunks to try again.
                if chunk.label.startswith("station"):
                    chunks_to_save.append(chunk)
                    continue

                # Now, loop through each chunk and find which stations have cameras without alignment.
                non_aligned_stations = set()
                for camera in chunk.cameras:
                    if camera.transform is None:
                        non_aligned_stations.add(camera.group.label.replace("_R", "").replace("_L", ""))

                # Loop over the non- or poorly aligned stations and subset a chunk with only those images.
                for station in non_aligned_stations:
                    station_cameras = [camera.label for camera in chunk.cameras if station in camera.group.label]
                    station_chunk = metashape_tools.copy_chunk(chunk, station, camera_subset=station_cameras)

                    chunks_to_save.append(station_chunk)

            # Loop over all chunks to save. Remove unused data and set a fixed calibration.
            for chunk in chunks_to_save:
                # Remove all markers with no projections.
                for marker in chunk.markers:
                    if len(marker.projections.keys()) == 0:
                        chunk.remove(marker)

                # Make a set of the groups that are used (to remove the unused ones)
                used_groups = set()
                # "Reset" the cameras in the chunk.
                for camera in chunk.cameras:
                    used_groups.add(camera.group.label)
                    camera.transform = None
                    camera.enabled = True

                    # Set the path appropriately (it might be wrong since datasets were processed on multiple machines)
                    camera.photo.path = os.path.join(
                        files.INPUT_DIRECTORIES["image_dir"], os.path.basename(camera.photo.path))
                    assert os.path.isfile(camera.photo.path)
                # Remove all groups that are unused.
                for group in chunk.camera_groups:
                    if group.label not in used_groups:
                        chunk.remove(group)

                # Reset the transform and region (latter might be redundant?)
                chunk.transform = None
                chunk.resetRegion()

            doc.append(dataset_doc, chunks_to_save)

        for chunk in doc.chunks:
            # For some reason, a few "Merged chunk"s get appended, so these should be removed.
            if "Merged" in chunk.label:
                doc.remove(chunk)
            # Assign a fixed calibration from the "Merged chunk"
            for sensor in chunk.sensors:
                if "unknown" in sensor.label:
                    continue

                # Try to fetch a calibration from the "Merged chunk"
                calib = sensor_calibrations[chunk.meta["dataset"]].get(sensor.label)
                # If the key didn't exist, look in the other datasets to find one.
                if calib is None:
                    for other_dataset in sensor_calibrations:
                        # Try to fetch a calibration from another dataset
                        calib = sensor_calibrations[other_dataset].get(sensor.label)

                        # If one was found, stop looking.
                        if calib is not None:
                            break
                # If a calibration was found (either from the active dataset or from another dataset)
                # then set the calibration and make it fixed.
                if calib is not None:
                    sensor.user_calib = calib.copy()
                    sensor.fixed_calibration = True

        metashape_tools.save_document(doc)
    else:
        doc = metashape_tools.load_document("failed_pairs")
        metashape_tools.validate_image_filepaths(doc)
        print("Loaded Metashape document for failed_pairs")

    chunks_not_aligned = [chunk for chunk in doc.chunks if not metashape_tools.has_alignment(chunk)]
    for chunk in tqdm(chunks_not_aligned, desc="Aligning stations"):
        continue  # I've fixed some things manually so I don't want this to be redone.
        # Align cameras (if the calibration was set as fixed before, it will still be fixed.
        # If no "Merged Chunk" calibration was found, a calibration will be estimated)
        success = metashape_tools.align_cameras(chunk, fixed_sensor=False)
        if not success:
            continue
        chunk.resetRegion()
        chunk.region.size *= 2
    metashape_tools.save_document(doc)

    # Generate dense clouds for all stereo-pairs that do not yet have one.
    chunks_missing_clouds = [chunk for chunk in doc.chunks if (
        chunk.dense_cloud is None and metashape_tools.has_alignment(chunk))]
    for chunk in tqdm(chunks_missing_clouds, desc="Generating dense clouds."):
        continue
        successful = metashape_tools.build_dense_clouds(
            doc=doc,
            chunk=chunk,
            pairs=[chunk.label],
            quality=metashape_tools.Quality.ULTRA,
            filtering=metashape_tools.Filtering.MILD,
            all_together=True
        )
        if chunk.dense_cloud is not None:
            chunk.dense_cloud.label = chunk.label

        metashape_tools.save_document(doc)

    # Generate DEMs for all stereo-pairs.
    chunks_with_clouds = [chunk for chunk in doc.chunks if chunk.dense_cloud is not None]
    for chunk in tqdm(chunks_with_clouds, desc="Building DEMs"):
        continue
        if len(chunk.elevations) > 0:
            continue
        # FOR FUTURE IMPLEMENTATION:
        # This raises a "RuntimeError: Unsupported datum transformation" if a proper transformation matrix is not available
        # I just ended up doing this as a batch process in the GUI instead.
        # Build a DEM for the chunk (it's probably just one, but I'll keep the structure as it was in the top)
        dem_filepaths = metashape_tools.build_dems(chunk=chunk, pairs=[chunk.label], redo=True)
        # Copy the DEMs to the export directory
        os.makedirs(os.path.join(inputs.TEMP_DIRECTORY, "output/dems"), exist_ok=True)
        for filepath in dem_filepaths.values():
            shutil.copyfile(filepath, os.path.join(inputs.TEMP_DIRECTORY, "output/dems", os.path.basename(filepath)))

        if len(chunk.elevations) > 0:
            chunk.elevation = chunk.elevations[0]

    metashape_tools.save_document(doc)

    # Generate orthomosaics for all stereo-pairs that do not yet have one.
    chunks_missing_orthos = [chunk for chunk in doc.chunks if (
        chunk.orthomosaic is None and chunk.elevation is not None)]
    for chunk in tqdm(chunks_missing_orthos, desc="Building orthomosaics"):
        chunk.elevation.crs = chunk.crs
        #successful = metashape_tools.build_orthomosaics(chunk=chunk, pairs=[chunk.label], resolution=1)

        try:
            metashape_tools.export_orthomosaics(
                chunk=chunk,
                pairs=[chunk.label],
                directory=os.path.join(inputs.TEMP_DIRECTORY, "output/orthos"),
                overwrite=True
            )
        except Exception as exception:
            print(exception)

    metashape_tools.save_document(doc)

    # Remove all ply files (dense clouds that take up a lot of space)
    for filename in os.listdir(inputs.CACHE_FILES[f"failed_pairs_temp_dir"]):
        if filename.endswith(".ply"):
            os.remove(os.path.join(inputs.CACHE_FILES[f"failed_pairs_temp_dir"], filename))

    for chunk in doc.chunks:
        # Remove all dense clouds in the Metashape project.
        for dense_cloud in chunk.dense_clouds:
            chunk.remove(dense_cloud)

    metashape_tools.save_document(doc)

    notify(f"failed_pairs finished")
    print(f"\n\nfailed_pairs finished\n\n")


def process_dataset(dataset: str, redo: bool = False):
    """
    Run the processing pipeline and log when it starts, fails or finishes.

    :param dataset: The name of the dataset to process.
    :param redo: Whether to start the analysis from scratch.
    """
    log(dataset, "Processing started")

    # This is basically just a wrapper for run_processing_pipeline, but with logging functionality for exceptions.
    try:
        run_processing_pipeline(dataset, redo=redo)
    # A weird error comes up sometimes on invalid (?) dense clouds.
    except KeyboardInterrupt as exception:
        log(dataset, "Processing cancelled")
        raise exception
    except RuntimeError as exception:
        if "Assertion 23910910127 failed" in str(exception):
            print(exception)
            log(dataset, "Processing failed")
            return
        raise exception
    except Exception as exception:
        if "No aligned cameras" in str(exception):  # Just go on if the exception was due to a failed alignment.
            log(dataset, "Alignment not possible")
            return
        log(dataset, "Processing failed")
        raise exception
