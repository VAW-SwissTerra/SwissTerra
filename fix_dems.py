import os
import shutil

import Metashape as ms

from terra.processing import inputs, metashape_tools, processing_tools


def fix_dems():

    for dataset in os.listdir("temp/processing/"):
        if dataset[:4] not in ["Wild", "Zeis"]:
            continue

        if not processing_tools.is_dataset_finished(dataset):
            continue

        doc = ms.Document()
        doc.open(f"temp/processing/{dataset}/{dataset}.psx")

        chunk = [ch for ch in doc.chunks if "Merged" in ch.label][0]
        # Get the names of the stereo-pairs
        pairs = metashape_tools.get_chunk_stereo_pairs(chunk)
        pairs_with_cloud = [cloud.label for cloud in chunk.dense_clouds]

        for elevation in chunk.elevations:
            chunk.remove(elevation)

        for ortho in chunk.orthomosaics:
            chunk.remove(ortho)

        # Generate DEMs for all stereo-pairs that do not yet have one.
        pairs_with_cloud = [cloud.label for cloud in chunk.dense_clouds]
        print(f"Building {len(pairs_with_cloud)} DEMs")
        dem_filepaths = metashape_tools.build_dems(chunk=chunk, pairs=pairs_with_cloud, redo=True)

        # Copy the DEMs to the export directory
        os.makedirs(os.path.join(inputs.TEMP_DIRECTORY, "output/dems"), exist_ok=True)
        for filepath in dem_filepaths.values():
            shutil.copyfile(filepath, os.path.join(inputs.TEMP_DIRECTORY, "output/dems", os.path.basename(filepath)))

        # Generate orthomosaics for all stereo-pairs that do not yet have one.
        missing_ortho_pairs = metashape_tools.get_unfinished_pairs(chunk, step=metashape_tools.Step.ORTHOMOSAIC)
        if len(missing_ortho_pairs) > 0:
            successful = metashape_tools.build_orthomosaics(chunk=chunk, pairs=missing_ortho_pairs, resolution=1)
            print(f"Made {len(successful)} orthomosaics")
        metashape_tools.export_orthomosaics(chunk=chunk, pairs=pairs,
                                            directory=os.path.join(inputs.TEMP_DIRECTORY, "output/orthos"), overwrite=True)

        # Remove all ply files (dense clouds that take up a lot of space)
        for filename in os.listdir(inputs.CACHE_FILES[f"{dataset}_temp_dir"]):
            if filename.endswith(".ply"):
                os.remove(os.path.join(inputs.CACHE_FILES[f"{dataset}_temp_dir"], filename))
        metashape_tools.save_document(doc)


if __name__ == "__main__":
    fix_dems()
