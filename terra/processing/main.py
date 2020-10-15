"""The main processing pipeline."""

import statictypes

from terra.constants import CONSTANTS
from terra.processing import inputs, metashape


@statictypes.enforce
def process_dataset(dataset: str, redo: bool = False) -> None:
    """
    Process a dataset from start to finish.

    param: dataset: The name of the dataset to process
    param: redo: Whether to redo steps that already exist.

    """
    inputs.export_camera_orientation_csv(dataset)

    # Load the metashape document or create a new one
    if not redo and metashape.is_document(dataset):
        doc = metashape.load_document(dataset)
        print(f"Loaded Metashape document for {dataset}")
    else:
        doc = metashape.new_document(dataset)
        print(f"Created new Metashape document for {dataset}")

    # Load or create a chunk with all the stereo-pairs
    chunk = metashape.load_or_remake_chunk(doc, dataset)

    # Get the names of the stereo-pairs
    pairs = metashape.get_chunk_stereo_pairs(chunk)

    # Check which pairs do not yet have a dense cloud
    unfinished_pairs = metashape.get_unfinished_pairs(chunk, metashape.Step.DENSE_CLOUD)
    # Make missing dense clouds
    if len(unfinished_pairs) > 0:
        metashape.build_dense_clouds(chunk, pairs=unfinished_pairs, quality=metashape.Quality.HIGH,
                                     filtering=metashape.Filtering.MILD)
    metashape.save_document(doc)

    # Coalign stereo-pair DEMs with each other and generate markers from their alignment
    metashape.coalign_stereo_pairs(chunk, pairs=pairs)
    # Run a bundle adjustment with these new markers.
    metashape.optimize_cameras(chunk)
    # Remove markers that still have an unreasonable error and run another bundle adjustment.
    metashape.remove_bad_markers(chunk)
    metashape.optimize_cameras(chunk)
    metashape.save_document(doc)

    # Rebuild the dense clouds with the proper pose
    metashape.build_dense_clouds(chunk, pairs=pairs, quality=metashape.Quality.ULTRA,
                                 filtering=metashape.Filtering.MILD)
    metashape.save_document(doc)

    metashape.build_dems(chunk=chunk, pairs=pairs, resolution=CONSTANTS["resolution"])
    metashape.save_document(doc)


#    chunks_to_process = metashape.get_unfinished_chunks(aligned_chunks, metashape.Step.DENSE_CLOUD)
#    if len(chunks_to_process) > 0:
#        print("Building dense clouds")
#        metashape.build_dense_clouds(chunks_to_process, quality=metashape.Quality.HIGH)
#        metashape.save_document(doc)
#
#    chunks_to_process = metashape.get_unfinished_chunks(aligned_chunks, metashape.Step.DEM)
#    if len(chunks_to_process) > 0:
#        print("Building DEMs")
#        metashape.build_dems(chunks_to_process, dataset)
#        metashape.save_document(doc)
#
#    return
#
#    chunks_to_process = metashape.get_unfinished_chunks(aligned_chunks, metashape.Step.ORTHOMOSAIC)
#    if len(chunks_to_process) > 0:
#        print("Building orthomosaics")
#        metashape.build_orthomosaics(chunks_to_process, resolution=1)
#        metashape.save_document(doc)
