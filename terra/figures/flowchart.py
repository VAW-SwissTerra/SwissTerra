import os
import tempfile

import graphviz


def generate_flowchart():
    dot = graphviz.Digraph(comment="Hello there")

    dot.attr(label="* = Not yet implemented", size="8,12")

    input_style = dict(fillcolor="skyblue", shape="box", style="filled", fontcolor="black")
    intermediate_file_style = dict(fillcolor="lightgray", style="filled", fontcolor="black")
    process_style = dict(fillcolor="lightgreen", style="filled", fontcolor="black")
    output_style = dict(fillcolor="lightpink", style="filled", fontcolor="black")

    dot.node("input-dem", "Modern DEM (swissAlti3D)", **input_style)
    dot.node("input-meta", "Image metadata files", **input_style)
    dot.node("input-images", "Images", **input_style)

    dot.node("process-slope_aspect_map", "Slope/aspect map generation", **process_style)
    dot.node("interm-slope_aspect_map", "Slope/aspect maps", **intermediate_file_style)

    dot.node("process-meta_to_dem_comparison", "Recorded vs. DEM height comparison", **process_style)
    dot.node("interm-offset_fields", "X/Y/Z 1x1 km gridded offset fields", **intermediate_file_style)

    dot.node("process-position_correction", "Offset correction", **process_style)
    dot.node("interm-corrected_position", "Corrected position data", **intermediate_file_style)

    dot.node("process-image_alignment", "Stereo-pair-wise image alignment", **process_style)
    dot.node("interm-initial_alignment", "Roughly aligned stereo-pairs", **intermediate_file_style)

    dot.node("process-first_dense_cloud", "Dense cloud generation", **process_style)
    dot.node("interm-first_dense_cloud", "Initial dense clouds", **intermediate_file_style)
    dot.node("process-dems_for_coalignment", "DEM generation", **process_style)
    dot.node("interm-dems_for_coalignment", "DEMs", **intermediate_file_style)
    dot.node("process-stereo_pair_coalignment", "Inter-stereo-pair coalignment", **process_style)
    dot.node("interm-coaligned_stereo_pairs", "Coaligned stereo pairs", **intermediate_file_style)

    dot.node("process-second_dense_cloud", "Dense cloud generation", **process_style)
    dot.node("interm-second_dense_cloud", "Dense clouds", **intermediate_file_style)
    dot.node("process-dem_generation", "DEM generation", **process_style)
    dot.node("output-dems", "DEMs", **intermediate_file_style)

    dot.node("process-orthomosaic_generation", "Orthomosaic generation", **process_style)
    dot.node("output-orthomosaics", "Orthomosaics", **intermediate_file_style)

    dot.node("process-dem_coregistration", "Stable-ground coregistration", **process_style)
    dot.node("interm-offsets", "DEM offsets", **intermediate_file_style)

    dot.node("process-raster_transform", "Raster transformation", **process_style)
    dot.node("output-final_dems", "Corrected DEMs", **output_style)
    dot.node("output-final_orthomosaics", "*Corrected orthomosaics", **output_style)

    dot.edge("input-dem", "process-slope_aspect_map")
    dot.edge("process-slope_aspect_map", "interm-slope_aspect_map")
    dot.edge("input-meta", "process-meta_to_dem_comparison")
    dot.edge("interm-slope_aspect_map", "process-meta_to_dem_comparison")
    dot.edge("input-dem", "process-meta_to_dem_comparison")
    dot.edge("process-meta_to_dem_comparison", "interm-offset_fields")

    dot.edge("interm-offset_fields", "process-position_correction")
    dot.edge("process-position_correction", "interm-corrected_position")

    dot.edge("input-images", "process-image_alignment")
    dot.edge("interm-corrected_position", "process-image_alignment")
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
    dot.edge("input-dem", "process-dem_coregistration")
    dot.edge("process-dem_coregistration", "interm-offsets")

    dot.edge("interm-offsets", "process-raster_transform")
    dot.edge("output-dems", "process-raster_transform")
    dot.edge("output-orthomosaics", "process-raster_transform")
    dot.edge("process-raster_transform", "output-final_dems")
    dot.edge("process-raster_transform", "output-final_orthomosaics")

    print(dot.source)
    temp_dir = tempfile.TemporaryDirectory()
    graph_path = os.path.join(temp_dir.name, "graph.gv")

    dot.render(graph_path, view=True)


if __name__ == "__main__":
    generate_flowchart()
    while True:
        continue
