"""Wrapper functions for Agisoft Metashape."""
import os
from typing import List

import Metashape as ms
import statictypes

from terra import files
from terra.processing import inputs, main
from terra.utilities import no_stdout

CACHE_FILES = {
}


for dataset in inputs.DATASETS:
    CACHE_FILES[f"{dataset}_metashape_project"] = os.path.join(inputs.CACHE_FILES[f"{dataset}_dir"], f"{dataset}.psx")


@statictypes.enforce
def new_document(dataset: str) -> ms.Document:

    with no_stdout():
        doc = ms.Document()

        doc.save(CACHE_FILES[f"{dataset}_metashape_project"])

    print(f"Created new Metashape project for {dataset}")

    return doc


@statictypes.enforce
def new_chunk(doc: ms.Document, filenames: List[str]) -> ms.Chunk:

    with no_stdout():
        chunk = doc.addChunk()

    filepaths = [os.path.join(files.INPUT_DIRECTORIES["image_dir"], filename) for filename in filenames]

    with no_stdout():

        chunk.addPhotos(filepaths)
        doc.save()

    return chunk

    print("NEW CHUNK")
