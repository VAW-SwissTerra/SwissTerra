import os
import tempfile
import time

import graphviz

from terra import files

EXTERNAL_INPUT_STYLE = dict(fillcolor="skyblue", shape="box", style="filled", fontcolor="black")
INTERNAL_INPUT_STYLE = dict(fillcolor="plum", style="filled", fontcolor="black")
INTERMEDIATE_FILE_STYLE = dict(fillcolor="lightgray", style="filled", fontcolor="black")
PROCESS_STYLE = dict(fillcolor="lightgreen", style="filled", fontcolor="black")
OUTPUT_STYLE = dict(fillcolor="lightpink", style="filled", fontcolor="black")


TEMP_DIRECTORY = os.path.join(files.TEMP_DIRECTORY, "figures/")


def show_dot(dot: graphviz.Digraph):
    temp_dir = tempfile.TemporaryDirectory()
    graph_path = os.path.join(temp_dir.name, "graph.gv")

    dot.render(graph_path, view=True)
    while True:
        time.sleep(1)
        continue


def preprocessing_flowchart():
    dot = graphviz.Digraph()

    dot.attr(size="8.3,11.7!", ratio="fill")

    with dot.subgraph(name="cluster_inputs_0") as cluster:
        cluster.attr(color="none", label="")
        cluster.node("input-meta", "Image metadata files", **EXTERNAL_INPUT_STYLE)
        cluster.node("input-images", "Images", **EXTERNAL_INPUT_STYLE)
        cluster.node("input-dem", "Modern DEM (swissAlti3D)", **EXTERNAL_INPUT_STYLE)
        cluster.node("input-glacier_mask", "~1935 glacier outlines", **EXTERNAL_INPUT_STYLE)

    dot.node("process-slope_aspect_map", "Slope/aspect map generation", **PROCESS_STYLE)
    dot.node("interm-slope_aspect_map", "Slope/aspect maps", **INTERMEDIATE_FILE_STYLE)

    dot.node("process-meta_to_dem_comparison", "Periglacial recorded vs.\n DEM height comparison", **PROCESS_STYLE)
    dot.node("interm-offset_fields", "X/Y/Z 1x1 km gridded offset fields", **INTERMEDIATE_FILE_STYLE)
    dot.node("process-position_correction", "Offset correction", **PROCESS_STYLE)

    dot.node("process-fiducial_marking", "Manual fiducial matching", **PROCESS_STYLE)
    dot.node("interm-marked_fiducials", "Manual fiducial marks", **INTERMEDIATE_FILE_STYLE)
    dot.node("process-fiducial_matching", "Supervised fiducial matching", **PROCESS_STYLE)

    dot.node("process-mask_generation", "Image frame\nmask generation", **PROCESS_STYLE)

    with dot.subgraph(name="cluster_outputs_0") as cluster:
        cluster.attr(color="none", label="")
        cluster.node("output-internal_image_transforms",
                     "Internal image transforms\n(Fiducial marks)", **OUTPUT_STYLE)
        cluster.node("output-frame_masks", "Image frame masks", **OUTPUT_STYLE)
        cluster.node("interm-corrected_position", "Corrected position data", **OUTPUT_STYLE)

    # Edges
    dot.edge("input-dem", "process-slope_aspect_map")
    dot.edge("process-slope_aspect_map", "interm-slope_aspect_map")
    dot.edge("input-meta", "process-meta_to_dem_comparison")
    dot.edge("interm-slope_aspect_map", "process-meta_to_dem_comparison")
    dot.edge("input-dem", "process-meta_to_dem_comparison")
    dot.edge("input-glacier_mask", "process-meta_to_dem_comparison")
    dot.edge("process-meta_to_dem_comparison", "interm-offset_fields")

    dot.edge("interm-offset_fields", "process-position_correction")
    dot.edge("process-position_correction", "interm-corrected_position")

    dot.edge("input-images", "process-fiducial_marking")
    dot.edge("process-fiducial_marking", "interm-marked_fiducials")
    dot.edge("interm-marked_fiducials", "process-fiducial_matching")
    dot.edge("input-images", "process-fiducial_matching")
    dot.edge("process-fiducial_matching", "output-internal_image_transforms")

    dot.edge("input-images", "process-mask_generation")
    dot.edge("output-internal_image_transforms", "process-mask_generation")
    dot.edge("process-mask_generation", "output-frame_masks")

    dot.render(os.path.join(TEMP_DIRECTORY, "preprocessing_flowchart"))
    # show_dot(dot)


def processing_flowchart():
    dot = graphviz.Digraph()

    dot.attr(label="* = Not yet implemented", size="8.3,11.7!", ratio="fill")
    metashape_attrs = dict(color="blue", label="Metashape")

    with dot.subgraph(name="cluster_inputs_0") as cluster:
        cluster.attr(color="none", label="")
        cluster.node("input-images", "Images", **EXTERNAL_INPUT_STYLE)
        cluster.node("input-corrected_position", "Corrected position data", **INTERNAL_INPUT_STYLE)
        cluster.node("input-internal_image_transforms",
                     "Internal image transforms\n(Fiducial marks)", **INTERNAL_INPUT_STYLE)
        cluster.node("input-frame_masks", "Image frame masks", **INTERNAL_INPUT_STYLE)

    with dot.subgraph(name="cluster_inputs_1") as cluster:
        cluster.node("input-glacier_mask", "Stable terrain mask", **EXTERNAL_INPUT_STYLE)
        cluster.attr(color="none", label="")
        cluster.node("input-dem", "Modern DEM (swissAlti3D)", **EXTERNAL_INPUT_STYLE)

    with dot.subgraph(name="cluster_ms_0") as cluster:
        cluster.attr(**metashape_attrs)

        cluster.node("process-image_alignment", "Stereo-pair-wise image alignment", **PROCESS_STYLE)
        cluster.node("interm-initial_alignment", "Roughly aligned stereo-pairs", **INTERMEDIATE_FILE_STYLE)

        cluster.node("process-first_dense_cloud", "Dense cloud generation", **PROCESS_STYLE)
        cluster.node("interm-first_dense_cloud", "Initial dense clouds", **INTERMEDIATE_FILE_STYLE)
    dot.node("process-dems_for_coalignment", "DEM generation", **PROCESS_STYLE)
    dot.node("interm-dems_for_coalignment", "DEMs", **INTERMEDIATE_FILE_STYLE)
    dot.node("process-stereo_pair_coalignment", "Inter-stereo-pair coalignment", **PROCESS_STYLE)
    dot.node("interm-coaligned_stereo_pairs", "Coaligned stereo pairs", **INTERMEDIATE_FILE_STYLE)

    with dot.subgraph(name="cluster_ms_1") as cluster:
        cluster.attr(**metashape_attrs)
        cluster.node("process-second_dense_cloud", "Dense cloud generation", **PROCESS_STYLE)
        cluster.node("interm-second_dense_cloud", "Dense clouds", **INTERMEDIATE_FILE_STYLE)

    dot.node("process-dem_generation", "DEM generation", **PROCESS_STYLE)
    dot.node("output-dems", "DEMs", **INTERMEDIATE_FILE_STYLE)

    with dot.subgraph(name="cluster_ms_2") as cluster:
        cluster.attr(**metashape_attrs)
        cluster.node("process-orthomosaic_generation", "Orthomosaic generation", **PROCESS_STYLE)
        cluster.node("output-orthomosaics", "Orthomosaics", **INTERMEDIATE_FILE_STYLE)

    dot.node("process-dem_coregistration", "Stable-ground ICP coregistration", **PROCESS_STYLE)
    dot.node("interm-offsets", "ICP stats/transforms", **INTERMEDIATE_FILE_STYLE)

    dot.node("process-raster_transform", "Raster transformation", **PROCESS_STYLE)

    with dot.subgraph(name="cluster_outputs_0") as cluster:
        cluster.attr(color="none", label="")
        cluster.node("output-final_orthomosaics", "*Corrected orthomosaics", **OUTPUT_STYLE)
        cluster.node("output-final_dems", "Corrected DEMs", **OUTPUT_STYLE)

    # Edges

    dot.edge("input-images", "process-image_alignment")
    dot.edge("input-corrected_position", "process-image_alignment")
    dot.edge("input-frame_masks", "process-image_alignment")
    dot.edge("input-internal_image_transforms", "process-image_alignment")
    dot.edge("process-image_alignment", "interm-initial_alignment")

    dot.edge("interm-initial_alignment", "process-first_dense_cloud")
    dot.edge("process-first_dense_cloud", "interm-first_dense_cloud")
    dot.edge("interm-first_dense_cloud", "process-dems_for_coalignment")
    dot.edge("process-dems_for_coalignment", "interm-dems_for_coalignment")
    dot.edge("interm-dems_for_coalignment", "process-stereo_pair_coalignment")
    dot.edge("process-stereo_pair_coalignment", "interm-coaligned_stereo_pairs")

    dot.edge("interm-coaligned_stereo_pairs", "process-second_dense_cloud")
    dot.edge("process-second_dense_cloud", "interm-second_dense_cloud")
    dot.edge("interm-second_dense_cloud", "process-dem_generation")
    dot.edge("process-dem_generation", "output-dems")

    dot.edge("output-dems", "process-orthomosaic_generation")
    dot.edge("process-orthomosaic_generation", "output-orthomosaics")

    dot.edge("output-dems", "process-dem_coregistration")
    dot.edge("input-glacier_mask", "process-dem_coregistration")
    dot.edge("input-dem", "process-dem_coregistration")
    dot.edge("process-dem_coregistration", "interm-offsets")

    dot.edge("interm-offsets", "process-raster_transform")
    dot.edge("output-dems", "process-raster_transform")
    dot.edge("output-orthomosaics", "process-raster_transform")
    dot.edge("process-raster_transform", "output-final_dems")
    dot.edge("process-raster_transform", "output-final_orthomosaics")

    dot.render(os.path.join(TEMP_DIRECTORY, "processing_flowchart"))
    # show_dot(dot)


def evaluation_flowchart():
    dot = graphviz.Digraph()
    dot.attr(label="* = Not yet implemented/finished")

    with dot.subgraph(name="cluster_inputs_0") as cluster:
        cluster.attr(color="none", label="")
        cluster.node("input-corrected_dems", "Corrected DEMs", **INTERNAL_INPUT_STYLE)
        cluster.node("input-glacier_mask", "~1935 glacier outlines", **EXTERNAL_INPUT_STYLE)
        cluster.node("input-dem", "Modern DEM (swissAlti3D)", **EXTERNAL_INPUT_STYLE)

    dot.node("process-ddem_generation", "dDEM generation", **PROCESS_STYLE)
    dot.node("interm-all_ddems", "dDEMs", **INTERMEDIATE_FILE_STYLE)

    dot.node("process-evaluation", "Supervised dDEM quality evaluation", **PROCESS_STYLE)

    dot.node("interm-optimal_ddem_combination", "Optimal dDEM combination", **INTERMEDIATE_FILE_STYLE)

    dot.node("process-ddem_merging", "Yearly dDEM conversion\nand merging", **PROCESS_STYLE)
    dot.node("interm-merged_ddem", "Merged yearly dDEM\nwith gaps", **INTERMEDIATE_FILE_STYLE)
    dot.node("process-gap_filling", "dDEM gap filling*", **PROCESS_STYLE)
    dot.node("interm-gap_filled_ddem", "Gap-filled merged yearly dDEM*", **INTERMEDIATE_FILE_STYLE)

    dot.node("process-glacier_change_calculation", "Glacier change calculations*", **PROCESS_STYLE)
    dot.node("output-glacier_specific_change", "Glacier-specific\nvolume change*", **OUTPUT_STYLE)
    dot.node("output-swiss_wide_hypsometry_change", "Swiss-wide volume- and\nhypsometric change*", **OUTPUT_STYLE)

    dot.edge("input-dem", "process-ddem_generation")
    dot.edge("input-corrected_dems", "process-ddem_generation")
    dot.edge("process-ddem_generation", "interm-all_ddems")

    dot.edge("interm-all_ddems", "process-evaluation")
    dot.edge("input-glacier_mask", "process-evaluation")
    dot.edge("process-evaluation", "interm-optimal_ddem_combination")

    dot.edge("interm-optimal_ddem_combination", "process-ddem_merging")
    dot.edge("process-ddem_merging", "interm-merged_ddem")
    dot.edge("interm-merged_ddem", "process-gap_filling")
    dot.edge("process-gap_filling", "interm-gap_filled_ddem")

    dot.edge("input-glacier_mask", "process-glacier_change_calculation")
    dot.edge("input-dem", "process-glacier_change_calculation")
    dot.edge("interm-gap_filled_ddem", "process-glacier_change_calculation")
    dot.edge("process-glacier_change_calculation", "output-glacier_specific_change")
    dot.edge("process-glacier_change_calculation", "output-swiss_wide_hypsometry_change")

    dot.render(os.path.join(TEMP_DIRECTORY, "evaluation_flowchart"))
    # show_dot(dot)


if __name__ == "__main__":
    preprocessing_flowchart()
    processing_flowchart()
    evaluation_flowchart()
