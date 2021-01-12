import os
import tempfile
import time

import graphviz


def generate_flowchart():
    dot = graphviz.Digraph()

    dot.attr(label="* = Not yet implemented", size="8.3,11.7!", ratio="fill")
    metashape_attrs = dict(color="blue", label="Metashape")

    input_style = dict(fillcolor="skyblue", shape="box", style="filled", fontcolor="black")
    intermediate_file_style = dict(fillcolor="lightgray", style="filled", fontcolor="black")
    process_style = dict(fillcolor="lightgreen", style="filled", fontcolor="black")
    output_style = dict(fillcolor="lightpink", style="filled", fontcolor="black")

    with dot.subgraph(name="cluster_inputs_0") as cluster:
        cluster.attr(color="none", label="")
        cluster.node("input-meta", "Image metadata files", **input_style)
        cluster.node("input-images", "Images", **input_style)
        cluster.node("input-dem", "Modern DEM (swissAlti3D)", **input_style)

    dot.node("process-slope_aspect_map", "Slope/aspect map generation", **process_style)
    dot.node("interm-slope_aspect_map", "Slope/aspect maps", **intermediate_file_style)

    dot.node("process-meta_to_dem_comparison", "Recorded vs. DEM height comparison", **process_style)
    dot.node("interm-offset_fields", "X/Y/Z 1x1 km gridded offset fields", **intermediate_file_style)
    dot.node("process-position_correction", "Offset correction", **process_style)
    dot.node("interm-corrected_position", "Corrected position data", **intermediate_file_style)

    with dot.subgraph(name="cluster_ms_0") as cluster:
        cluster.attr(**metashape_attrs)

        cluster.node("process-image_alignment", "Stereo-pair-wise image alignment", **process_style)
        cluster.node("interm-initial_alignment", "Roughly aligned stereo-pairs", **intermediate_file_style)

        cluster.node("process-first_dense_cloud", "Dense cloud generation", **process_style)
        cluster.node("interm-first_dense_cloud", "Initial dense clouds", **intermediate_file_style)
    dot.node("process-dems_for_coalignment", "DEM generation", **process_style)
    dot.node("interm-dems_for_coalignment", "DEMs", **intermediate_file_style)
    dot.node("process-stereo_pair_coalignment", "Inter-stereo-pair coalignment", **process_style)
    dot.node("interm-coaligned_stereo_pairs", "Coaligned stereo pairs", **intermediate_file_style)

    with dot.subgraph(name="cluster_ms_1") as cluster:
        cluster.attr(**metashape_attrs)
        cluster.node("process-second_dense_cloud", "Dense cloud generation", **process_style)
        cluster.node("interm-second_dense_cloud", "Dense clouds", **intermediate_file_style)

    dot.node("process-dem_generation", "DEM generation", **process_style)
    dot.node("output-dems", "DEMs", **intermediate_file_style)

    with dot.subgraph(name="cluster_ms_2") as cluster:
        cluster.attr(**metashape_attrs)
        cluster.node("process-orthomosaic_generation", "Orthomosaic generation", **process_style)
        cluster.node("output-orthomosaics", "Orthomosaics", **intermediate_file_style)

    dot.node("process-dem_coregistration", "Stable-ground ICP coregistration", **process_style)
    dot.node("interm-offsets", "ICP stats/transforms", **intermediate_file_style)

    dot.node("process-evaluation", "DEM quality evaluation", **process_style)

    dot.node("process-raster_transform", "Raster transformation", **process_style)
    dot.node("output-final_dems", "Corrected DEMs", **intermediate_file_style)

    with dot.subgraph(name="cluster_outputs_0") as cluster:
        cluster.attr(color="none", label="")
        cluster.node("output-final_orthomosaics", "*Corrected orthomosaics", **output_style)
        cluster.node("output-final_dem_combination", "Optimal DEM combination", **output_style)

    # Edges

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

    dot.edge("output-final_dems", "process-evaluation")
    dot.edge("output-dems", "process-evaluation")
    dot.edge("interm-offsets", "process-evaluation")

    dot.edge("process-evaluation", "output-final_dem_combination")

    print(dot.source)
    temp_dir = tempfile.TemporaryDirectory()
    graph_path = os.path.join(temp_dir.name, "graph.gv")

    dot.render(graph_path, view=True)
    while True:
        time.sleep(1)
        continue


if __name__ == "__main__":
    generate_flowchart()
