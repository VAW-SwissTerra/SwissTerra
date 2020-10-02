import os
from collections import namedtuple

import numpy as np
import pandas as pd

from terra.processing import inputs, metashape


def process_dataset(dataset: str):

    inputs.export_camera_orientation_csv(dataset)

    doc = metashape.new_document(dataset)

    chunk = metashape.new_chunk(doc, filenames=inputs.get_dataset_filenames(dataset))


def process_rhone():
    print("I FINSHED IT WAS THIS SIMPLE ALL ALONG")
