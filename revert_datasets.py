import os
import shutil

import Metashape as ms
import numpy as np

from terra.processing import metashape_tools


def revert_datasets():

    for dataset in os.listdir("temp/processing/"):
        if not dataset[:4] in ["Wild", "Zeis"]:
            continue

        if dataset == "Wild10_1927":  # This one is good
            continue

        doc = ms.Document()
        doc.open(f"temp/processing/{dataset}/{dataset}.psx")

        shutil.rmtree(f"temp/processing/{dataset}/temp", ignore_errors=True)
        os.makedirs(f"temp/processing/{dataset}/temp", exist_ok=True)
        try:
            chunk = [ch for ch in doc.chunks if "Merged" in ch.label][0]
        except IndexError:
            continue

        for marker in chunk.markers:
            if marker.type == ms.Marker.Type.Fiducial:
                continue
            chunk.remove(marker)

        for item in np.r_[chunk.elevations, chunk.orthomosaics]:
            chunk.remove(item)

        for sensor in chunk.sensors:
            metashape_tools.import_fiducials(chunk, sensor)

        # Make sure that fiducials were not removed.
        assert len(chunk.markers) > 0

        doc.save()


if __name__ == "__main__":
    revert_datasets()
