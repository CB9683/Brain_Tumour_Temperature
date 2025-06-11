# main.py
import argparse
import logging
import os
import time
import numpy as np

from src import config_manager, io_utils, utils
from src import data_structures # For type hinting and initial object creation
from src import vascular_growth, angiogenesis, perfusion_solver, visualization
from src.constants import DEFAULT_VOXEL_SIZE_MM


def setup_logging(log_level_str: str, log_file: str):
    """Configures logging for the simulation."""
    numeric_level = getattr(logging, log_level_str.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level_str}")

    # Make sure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'), # Overwrite log file each run
            logging.StreamHandler() # Also print to console
        ]
    )
    # Suppress overly verbose PyVista/VTK logs if not in DEBUG
    if numeric_level > logging.DEBUG:
        logging.getLogger('pyvista').setLevel(logging.WARNING)
        # May need to find other noisy vtk loggers if they appear

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="GBO Brain Vasculature Simulation Framework")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config file."
    )
    return parser.parse_args()

def load_initial_data(config: dict) -> tuple[dict, data_structures.nx.DiGraph | None]:
    """
    Loads all necessary input data based on the configuration.
    - Tissue segmentations (WM, GM, Tumor, CSF)
    - Initial arterial centerlines
    
    Returns:
        tuple: (tissue_data_dict, initial_arterial_graph)
               tissue_data_dict contains segmentation arrays, affine, voxel_volume, etc.
               initial_arterial_graph is a NetworkX graph of the input arteries.
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading initial data...")
    
    # Initialize tissue_data dictionary
    tissue_data = {
        'WM': None, 'GM': None, 'CSF': None, 'Tumor': None,
        'affine': None, 'voxel_volume': None, 
        'domain_mask': None, 'metabolic_demand_map': None,
        'world_coords_flat': None, 'voxel_indices_flat': None,
        'shape': None
    }

    # Load segmentations
    paths = config_manager.get_param(config, "paths")
    wm_data, affine_wm, _ = io_utils.load_nifti_image(paths["wm_nifti"])
    gm_data, affine_gm, _ = io_utils.load_nifti_image(paths["gm_nifti"])
    
    # Use the first available affine and assume all are co-registered
    # A more robust approach would check for co-registration or resample.
    if affine_wm is not None:
        tissue_data['affine'] = affine_wm
    elif affine_gm is not None:
        tissue_data['affine'] = affine_gm
    else:
        logger.warning("No WM or GM NIfTI found to determine affine. Using default identity affine and 1mm voxel size.")
        tissue_data['affine'] = np.eye(4)
        tissue_data['affine'][0,0] = tissue_data['affine'][1,1] = tissue_data['affine'][2,2] = DEFAULT_VOXEL_SIZE_MM
        # This case might require specifying image dimensions if no NIFTIs are loaded.

    tissue_data['voxel_volume'] = utils.get_voxel_volume_from_affine(tissue_data['affine'])
    
    # Determine a common shape (assuming co-registration)
    common_shape = None
    if wm_data is not None:
        common_shape = wm_data.shape
        tissue_data['WM'] = wm_data.astype(bool) # Ensure binary
    if gm_data is not None:
        if common_shape is not None and gm_data.shape != common_shape:
            logger.error("GM segmentation shape mismatch with WM. Co-registration issue?")
            # Handle error or attempt resampling
        elif common_shape is None:
            common_shape = gm_data.shape
        tissue_data['GM'] = gm_data.astype(bool) # Ensure binary

    if common_shape is None:
        logger.error("Could not determine tissue domain shape. At least WM or GM NIfTI must be provided.")
        # Or, allow specifying shape in config if no NIFTIs are used (e.g. for synthetic phantoms)
        # For now, let's assume this is a fatal error for real data processing.
        raise ValueError("Cannot determine tissue domain shape from inputs.")
    tissue_data['shape'] = common_shape

    # Optional segmentations
    if "csf_nifti" in paths and paths["csf_nifti"]:
        csf_data, _, _ = io_utils.load_nifti_image(paths["csf_nifti"])
        if csf_data is not None and csf_data.shape == common_shape:
            tissue_data['CSF'] = csf_data.astype(bool)
        elif csf_data is not None:
            logger.warning(f"CSF segmentation shape {csf_data.shape} mismatch with domain {common_shape}. Skipping CSF.")

    if "tumor_nifti" in paths and paths["tumor_nifti"]:
        tumor_data, _, _ = io_utils.load_nifti_image(paths["tumor_nifti"])
        if tumor_data is not None and tumor_data.shape == common_shape:
            tissue_data['Tumor'] = tumor_data.astype(bool)
        elif tumor_data is not None:
            logger.warning(f"Tumor segmentation shape {tumor_data.shape} mismatch with domain {common_shape}. Skipping Tumor.")

    # Create domain mask (voxels considered for simulation)
    # For now, union of WM and GM. Could be extended to include tumor or other ROIs.
    domain_mask = np.zeros(common_shape, dtype=bool)
    if tissue_data['WM'] is not None:
        domain_mask = np.logical_or(domain_mask, tissue_data['WM'])
    if tissue_data['GM'] is not None:
        domain_mask = np.logical_or(domain_mask, tissue_data['GM'])
    if tissue_data['Tumor'] is not None and config_manager.get_param(config, "tumor_angiogenesis.enabled", False):
        domain_mask = np.logical_or(domain_mask, tissue_data['Tumor'])
    
    tissue_data['domain_mask'] = domain_mask
    logger.info(f"Domain mask created with {np.sum(domain_mask)} active voxels.")

    # Precompute world coordinates for active voxels
    voxel_indices = np.array(np.where(domain_mask)).T
    if voxel_indices.size > 0:
        tissue_data['voxel_indices_flat'] = voxel_indices
        tissue_data['world_coords_flat'] = utils.voxel_to_world(voxel_indices, tissue_data['affine'])
    else: # Handle case of empty domain_mask gracefully, though it's unlikely for valid inputs
        tissue_data['voxel_indices_flat'] = np.empty((0,3), dtype=int)
        tissue_data['world_coords_flat'] = np.empty((0,3), dtype=float)
        logger.warning("Domain mask is empty. No voxels will be processed.")

    # Create metabolic demand map (q_met * dV per voxel)
    segmentations_for_demand = {
        'WM': tissue_data['WM'], 'GM': tissue_data['GM'],
        'CSF': tissue_data['CSF'], 'Tumor': tissue_data['Tumor']
    }
    # Filter out None segmentations before passing
    segmentations_for_demand = {k: v for k, v in segmentations_for_demand.items() if v is not None}
    
    tissue_data['metabolic_demand_map'] = data_structures.get_metabolic_demand_map(
        segmentations_for_demand, config, tissue_data['voxel_volume']
    )
    if tissue_data['metabolic_demand_map'] is None:
        logger.error("Failed to generate metabolic demand map.")
        # This could be a fatal error depending on workflow
        # For now, allow continuation but GBO might fail or do nothing.
        tissue_data['metabolic_demand_map'] = np.zeros(common_shape, dtype=np.float32)


    # Load initial arterial centerlines
    initial_arterial_graph = None
    centerline_path = paths.get("arterial_centerlines")
    if centerline_path:
        poly_data = None
        if centerline_path.endswith(".vtp"):
            poly_data = io_utils.load_arterial_centerlines_vtp(centerline_path)
        elif centerline_path.endswith(".txt"):
            poly_data = io_utils.load_arterial_centerlines_txt(centerline_path) # Add default radius from config if needed
        else:
            logger.warning(f"Unsupported arterial centerline file format: {centerline_path}")

        if poly_data and poly_data.n_points > 0:
            # Convert PyVista PolyData to NetworkX graph
            # This is a simplified conversion; more sophisticated handling of branches might be needed
            # based on VTP structure (e.g., using vtkvmtk libraries for full tree parsing).
            # For now, assumes poly_data.lines represents connected segments.
            initial_arterial_graph = data_structures.create_empty_vascular_graph()
            node_id_counter = 0
            pv_point_to_nx_node = {}

            # Add points as nodes
            for pt_idx in range(poly_data.n_points):
                pos = poly_data.points[pt_idx]
                radius = poly_data.point_data.get('radius', [config_manager.get_param(config,"vascular_properties.min_radius",0.01)])[pt_idx] \
                           if 'radius' in poly_data.point_data else config_manager.get_param(config,"vascular_properties.min_radius",0.01)
                
                # Assign a unique ID (can be more meaningful if VTP has IDs)
                current_node_id = f"m_{node_id_counter}" # 'm' for measured
                data_structures.add_node_to_graph(initial_arterial_graph, current_node_id,
                                                  pos=pos, radius=radius, type='measured_root_or_segment')
                pv_point_to_nx_node[pt_idx] = current_node_id
                node_id_counter += 1
            
            # Add lines as edges
            # PyVista lines array is [n_pts_cell0, pt0_idx, pt1_idx, ..., n_pts_cell1, ptA_idx, ptB_idx, ...]
            lines_array = poly_data.lines
            current_idx = 0
            while current_idx < len(lines_array):
                num_pts_in_segment = lines_array[current_idx]
                # Assuming line segments (2 points) for now
                if num_pts_in_segment == 2:
                    pt1_original_idx = lines_array[current_idx + 1]
                    pt2_original_idx = lines_array[current_idx + 2]
                    
                    node_u = pv_point_to_nx_node.get(pt1_original_idx)
                    node_v = pv_point_to_nx_node.get(pt2_original_idx)

                    if node_u and node_v:
                        # Determine direction if possible (e.g. decreasing radius, or assume from root)
                        # For now, add undirected or assume order in VTP implies direction
                        # GBO often implies a rooted tree, so direction matters.
                        # Simplistic: add edge, direction can be refined or assumed during GBO init.
                        # Add as DiGraph: assume u->v if radius[u] >= radius[v], else v->u (very heuristic)
                        # A better way is if VTP implies hierarchy or use flow simulation.
                        # For now, let's add one direction and let GBO sort it out or assume root points are specified.
                        # A robust solution would use more advanced VTP parsing or input conventions.
                        rad_u = initial_arterial_graph.nodes[node_u]['radius']
                        rad_v = initial_arterial_graph.nodes[node_v]['radius']
                        if rad_u > rad_v: # Flow from larger to smaller
                             data_structures.add_edge_to_graph(initial_arterial_graph, node_u, node_v, type='measured_segment')
                        elif rad_v > rad_u:
                             data_structures.add_edge_to_graph(initial_arterial_graph, node_v, node_u, type='measured_segment')
                        else: # Equal radii, use original order
                             data_structures.add_edge_to_graph(initial_arterial_graph, node_u, node_v, type='measured_segment')
                    current_idx += (num_pts_in_segment + 1)
                else: # Skip polylines with != 2 points for now or handle them
                    logger.warning(f"Skipping polyline segment with {num_pts_in_segment} points in VTP. Expected 2.")
                    current_idx += (num_pts_in_segment + 1)
            
            # Identify root nodes (in-degree 0) and terminal nodes (out-degree 0) of this measured graph
            for node_id in list(initial_arterial_graph.nodes()): # list() to avoid issues if modifying graph
                if initial_arterial_graph.in_degree(node_id) == 0 and initial_arterial_graph.out_degree(node_id) > 0 :
                    initial_arterial_graph.nodes[node_id]['type'] = 'measured_root'
                elif initial_arterial_graph.out_degree(node_id) == 0 and initial_arterial_graph.in_degree(node_id) > 0:
                    initial_arterial_graph.nodes[node_id]['type'] = 'measured_terminal'
                elif initial_arterial_graph.in_degree(node_id) > 0 and initial_arterial_graph.out_degree(node_id) > 0:
                    initial_arterial_graph.nodes[node_id]['type'] = 'measured_bifurcation_or_segment' # Needs refinement
            
            logger.info(f"Converted arterial centerlines to NetworkX graph: {initial_arterial_graph.number_of_nodes()} nodes, {initial_arterial_graph.number_of_edges()} edges.")
            if not nx.is_forest(nx.to_undirected(initial_arterial_graph)): # Check for cycles
                 logger.warning("Initial arterial graph contains cycles. GBO expects a tree-like structure.")
        else:
            logger.warning("No arterial centerline data loaded or data is empty.")
    else:
        logger.warning("Arterial centerline file path not specified in config.")
        # Potentially start GBO from a single seed point if no arteries given (synthetic organoid mode)

    return tissue_data, initial_arterial_graph


def main():
    args = parse_arguments()
    
    # Load configuration
    try:
        config = config_manager.load_config(args.config)
    except Exception as e:
        print(f"CRITICAL: Failed to load configuration: {e}")
        return

    # Setup output directory
    sim_name = config_manager.get_param(config, "simulation.simulation_name", "gbo_sim_run")
    if args.output_dir: # Command line override
        base_output_dir = args.output_dir
    else:
        base_output_dir = config_manager.get_param(config, "paths.output_dir", "output")
    
    output_dir = utils.create_output_directory(base_output_dir, sim_name, timestamp=True)
    
    # Setup logging
    log_level = config_manager.get_param(config, "simulation.log_level", "INFO")
    log_file_path = os.path.join(output_dir, f"{sim_name}.log")
    setup_logging(log_level, log_file_path)
    
    main_logger = logging.getLogger(__name__) # Get logger for main script after setup
    main_logger.info(f"Simulation started. Output directory: {output_dir}")
    main_logger.info(f"Using configuration file: {os.path.abspath(args.config)}")

    # Save the used configuration to the output directory
    io_utils.save_simulation_parameters(config, os.path.join(output_dir, "config_used.yaml"))

    # Set RNG seed
    seed = config_manager.get_param(config, "simulation.random_seed", None)
    if seed is not None:
        utils.set_rng_seed(seed)

    # --- Simulation Core ---
    start_time = time.time()

    # 1. Load Inputs
    try:
        tissue_data, initial_arterial_graph = load_initial_data(config)
    except Exception as e:
        main_logger.critical(f"Failed during data loading: {e}", exc_info=True)
        return
    
    # Save loaded initial data for inspection if in debug or requested
    if config_manager.get_param(config, "visualization.save_intermediate_steps", False):
        for key, arr in tissue_data.items():
            if isinstance(arr, np.ndarray) and arr.ndim == 3 : # Save 3D arrays
                 if arr.dtype == bool: arr = arr.astype(np.uint8) # nibabel compatibility
                 io_utils.save_nifti_image(arr, tissue_data['affine'], os.path.join(output_dir, f"initial_tissue_{key}.nii.gz"))
        if initial_arterial_graph:
            io_utils.save_vascular_tree_vtp(initial_arterial_graph, os.path.join(output_dir, "initial_arterial_graph.vtp"))

    # 2. Healthy Vascular Development (GBO)
    main_logger.info("Starting healthy vascular development (GBO)...")
    # The GBO module will need the initial graph (or just terminals from it) and tissue data.
    # It will return the grown healthy vascular tree.
    healthy_vascular_tree = vascular_growth.grow_healthy_vasculature(
        config=config,
        tissue_data=tissue_data,
        initial_graph=initial_arterial_graph,
        output_dir=output_dir # For intermediate saves
    )
    if healthy_vascular_tree is None:
        main_logger.error("Healthy vascular growth failed or returned no result.")
        # Decide whether to proceed or terminate
    else:
        main_logger.info(f"Healthy vascular growth finished. Tree has {healthy_vascular_tree.number_of_nodes()} nodes, {healthy_vascular_tree.number_of_edges()} segments.")
        io_utils.save_vascular_tree_vtp(healthy_vascular_tree, os.path.join(output_dir, "healthy_vascular_tree.vtp"))


    # 3. Tumor Angiogenesis (if enabled and tumor exists)
    final_vascular_tree = healthy_vascular_tree
    if config_manager.get_param(config, "tumor_angiogenesis.enabled", False) and \
       tissue_data.get('Tumor') is not None and np.any(tissue_data['Tumor']):
        main_logger.info("Starting tumor angiogenesis...")
        final_vascular_tree = angiogenesis.grow_tumor_vessels(
            config=config,
            tissue_data=tissue_data,
            base_vascular_tree=healthy_vascular_tree, # Start from the healthy tree
            output_dir=output_dir
        )
        if final_vascular_tree is None:
            main_logger.error("Tumor angiogenesis failed. Using healthy tree for subsequent steps.")
            final_vascular_tree = healthy_vascular_tree # Fallback
        else:
            main_logger.info(f"Tumor angiogenesis finished. Final tree has {final_vascular_tree.number_of_nodes()} nodes, {final_vascular_tree.number_of_edges()} segments.")
            io_utils.save_vascular_tree_vtp(final_vascular_tree, os.path.join(output_dir, "tumor_vascular_tree.vtp"))
    else:
        main_logger.info("Tumor angiogenesis skipped (disabled or no tumor data).")


    # 4. Perfusion Modeling
    perfusion_map = None
    pressure_map_tissue = None
    if config_manager.get_param(config, "perfusion_solver.enabled", False) and final_vascular_tree:
        main_logger.info("Starting perfusion modeling...")
        # Perfusion solver updates tree with pressures/flows and returns tissue maps
        final_vascular_tree, perfusion_map, pressure_map_tissue = perfusion_solver.calculate_perfusion(
            config=config,
            tissue_data=tissue_data,
            vascular_graph=final_vascular_tree, # Pass the most up-to-date tree
            output_dir=output_dir
        )
        if perfusion_map is not None:
            io_utils.save_nifti_image(perfusion_map, tissue_data['affine'], os.path.join(output_dir, "perfusion_map.nii.gz"))
            main_logger.info("Perfusion map generated and saved.")
        if pressure_map_tissue is not None:
            io_utils.save_nifti_image(pressure_map_tissue, tissue_data['affine'], os.path.join(output_dir, "tissue_pressure_map.nii.gz"))
            main_logger.info("Tissue pressure map generated and saved.")
        # The final_vascular_tree might have updated pressure/flow attributes
        io_utils.save_vascular_tree_vtp(final_vascular_tree, os.path.join(output_dir, "final_vascular_tree_with_perfusion_data.vtp"))

    else:
        main_logger.info("Perfusion modeling skipped (disabled or no vascular tree).")


    # 5. Visualization
    main_logger.info("Generating final visualizations...")
    visualization.generate_final_visualizations(
        config=config,
        output_dir=output_dir,
        tissue_data=tissue_data,
        vascular_graph=final_vascular_tree,
        perfusion_map=perfusion_map,
        pressure_map_tissue=pressure_map_tissue
    )

    end_time = time.time()
    main_logger.info(f"Simulation finished. Total execution time: {end_time - start_time:.2f} seconds.")
    main_logger.info(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    # Create dummy placeholder files for modules to allow main.py to import them
    # This is for initial skeleton setup. These will be replaced by actual implementations.
    placeholder_modules = ["vascular_growth", "angiogenesis", "perfusion_solver", "visualization"]
    for mod_name in placeholder_modules:
        mod_path = os.path.join("src", f"{mod_name}.py")
        if not os.path.exists(mod_path):
            with open(mod_path, "w") as f:
                if mod_name == "vascular_growth":
                    f.write("import networkx as nx\nimport logging\nlogger = logging.getLogger(__name__)\ndef grow_healthy_vasculature(config, tissue_data, initial_graph, output_dir):\n    logger.info('vascular_growth.grow_healthy_vasculature called (placeholder)')\n    if initial_graph: return initial_graph.copy()\n    return nx.DiGraph()\n")
                elif mod_name == "angiogenesis":
                    f.write("import networkx as nx\nimport logging\nlogger = logging.getLogger(__name__)\ndef grow_tumor_vessels(config, tissue_data, base_vascular_tree, output_dir):\n    logger.info('angiogenesis.grow_tumor_vessels called (placeholder)')\n    if base_vascular_tree: return base_vascular_tree.copy()\n    return nx.DiGraph()\n")
                elif mod_name == "perfusion_solver":
                    f.write("import numpy as np\nimport logging\nlogger = logging.getLogger(__name__)\ndef calculate_perfusion(config, tissue_data, vascular_graph, output_dir):\n    logger.info('perfusion_solver.calculate_perfusion called (placeholder)')\n    shape = tissue_data.get('shape', (10,10,10))\n    return vascular_graph, np.zeros(shape), np.zeros(shape)\n")
                elif mod_name == "visualization":
                    f.write("import logging\nlogger = logging.getLogger(__name__)\ndef generate_final_visualizations(config, output_dir, tissue_data, vascular_graph, perfusion_map, pressure_map_tissue):\n    logger.info('visualization.generate_final_visualizations called (placeholder)')\n    pass\n")
    
    # Create dummy __init__.py in src if it doesn't exist
    init_py_path = os.path.join("src", "__init__.py")
    if not os.path.exists(init_py_path):
        with open(init_py_path, "w") as f:
            f.write("# src package\n")

    # Create a default config.yaml if it doesn't exist
    if not os.path.exists("config.yaml"):
        config_manager.create_default_config("config.yaml")
        print("Created default config.yaml. Please populate 'data/' directory or update paths in config.yaml.")
    
    # Create data and output directories if they don't exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    # Add READMEs to data/ and output/
    if not os.path.exists("data/README.md"):
        with open("data/README.md", "w") as f: f.write("Place input NIfTI (.nii.gz) and VTP/TXT centerline files here.\nUpdate config.yaml to point to these files.\n")
    if not os.path.exists("output/README.md"):
        with open("output/README.md", "w") as f: f.write("This directory will contain simulation results.\nEach run creates a timestamped subfolder.\n")


    main()