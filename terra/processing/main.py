import os
from collections import namedtuple
from typing import List

import Metashape as ms
import numpy as np
import pandas as pd
import statictypes
from tqdm import tqdm

from terra.processing import inputs, metashape


@statictypes.enforce
def process_dataset(dataset: str, redo: bool = False) -> None:
    """
    Process a dataset from start to finish.

    param: dataset: The name of the dataset to process
    param: redo: Whether to redo steps that already exist.

    """

    inputs.export_camera_orientation_csv(dataset)

    if not redo and metashape.is_document(dataset):
        doc = metashape.load_document(dataset)
        print(f"Loaded Metashape document for {dataset}")
    else:
        doc = metashape.new_document(dataset)
        print(f"Created new Metashape document for {dataset}")

    image_meta = inputs.get_dataset_metadata(dataset)

    aligned_chunks: List[ms.Chunk] = []
    for station_number, station_meta in tqdm(image_meta.groupby("Base number")):
        if station_meta["Position"].unique().shape[0] < 2:
            print(f"Station {station_number} only has position {station_meta['Position'].iloc[0]}. Skipping.")
            continue

        chunk_label = f"station_{station_number}"

        # Check if the chunk already exists
        if any([chunk.label == chunk_label for chunk in doc.chunks]):
            chunk = [chunk for chunk in doc.chunks if chunk.label == chunk_label][0]
            aligned = metashape.has_alignment(chunk)

        else:
            chunk = metashape.new_chunk(doc, filenames=list(
                station_meta["Image file"].values), chunk_label=chunk_label)

            aligned = metashape.align_cameras(chunk, fixed_sensor=True)
            metashape.save_document(doc)

        if aligned:
            aligned_chunks.append(chunk)

    chunks_to_process = metashape.get_unfinished_chunks(aligned_chunks, metashape.Step.DENSE_CLOUD)
    if len(chunks_to_process) > 0:
        print("Building dense clouds")
        metashape.build_dense_clouds(chunks_to_process, quality=metashape.Quality.HIGH)
        metashape.save_document(doc)

    chunks_to_process = metashape.get_unfinished_chunks(aligned_chunks, metashape.Step.DEM)
    if len(chunks_to_process) > 0:
        print("Building DEMs")
        metashape.build_dems(chunks_to_process, dataset)
        metashape.save_document(doc)

    return

    chunks_to_process = metashape.get_unfinished_chunks(aligned_chunks, metashape.Step.ORTHOMOSAIC)
    if len(chunks_to_process) > 0:
        print("Building orthomosaics")
        metashape.build_orthomosaics(chunks_to_process, resolution=1)
        metashape.save_document(doc)
