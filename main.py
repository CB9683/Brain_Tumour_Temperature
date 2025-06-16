# main.py
import argparse
import logging
import os
import time
import numpy as np
import networkx as nx # For type hinting
from typing import Optional
from src import config_manager, io_utils, utils
from src import data_structures
from src import vascular_growth, angiogenesis, perfusion_solver, visualization
from src.constants import DEFAULT_VOXEL_SIZE_MM, Q_MET_TUMOR_RIM_PER_ML # Added Q_MET_TUMOR_RIM_PER_ML

def setup_logging(log_level_str: str, log_file: str):
    """Configures logging for the simulation."""
    numeric_level = getattr(logging, log_level_str.upper(), logging.INFO)
    if not isinstance(numeric_level, int): # Fallback if level is invalid
        print(f"Warning: Invalid log level '{log_level_str}'. Defaulting to INFO.")
        numeric_level = logging.INFO

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    if numeric_level > logging.DEBUG:
        logging.getLogger('pyvista').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)

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

def load_initial_data(config: dict) -> tuple[dict, Optional[nx.DiGraph]]:
    """
    Loads all necessary input data based on the configuration.
    """
    logger = logging.getLogger(__name__) # Get logger for this function
    logger.info("Loading initial data...")
    
    tissue_data = {
        'WM': None, 'GM': None, 'CSF': None, 
        'Tumor_Max_Extent': None, 'Tumor': None, # Active tumor
        'tumor_rim_mask': None, 'tumor_core_mask': None,
        'VEGF_field': None, 'affine': None, 'voxel_volume': None, 
        'domain_mask': None, 'metabolic_demand_map': None,
        'world_coords_flat': None, 'voxel_indices_flat': None,
        'shape': None
    }

    paths = config_manager.get_param(config, "paths", {}) # Default to empty dict
    wm_data, affine_wm, _ = io_utils.load_nifti_image(paths.get("wm_nifti",""))
    gm_data, affine_gm, _ = io_utils.load_nifti_image(paths.get("gm_nifti",""))
    
    if affine_wm is not None: tissue_data['affine'] = affine_wm
    elif affine_gm is not None: tissue_data['affine'] = affine_gm
    
    common_shape = None
    if wm_data is not None:
        common_shape = wm_data.shape
        tissue_data['WM'] = wm_data.astype(bool)
    if gm_data is not None:
        if common_shape and gm_data.shape != common_shape:
            logger.error(f"GM shape {gm_data.shape} mismatch WM {common_shape}. Check inputs.")
        elif not common_shape: common_shape = gm_data.shape
        tissue_data['GM'] = gm_data.astype(bool)

    # Try to get shape from Tumor_Max_Extent if GM/WM are missing
    if common_shape is None and paths.get("tumor_nifti"):
        logger.info("Attempting to derive shape and affine from tumor_nifti as GM/WM are missing/invalid.")
        tumor_final_data_for_shape, affine_tumor_for_shape, _ = io_utils.load_nifti_image(paths["tumor_nifti"])
        if tumor_final_data_for_shape is not None:
            common_shape = tumor_final_data_for_shape.shape
            if tissue_data['affine'] is None and affine_tumor_for_shape is not None:
                tissue_data['affine'] = affine_tumor_for_shape
            logger.info(f"Derived common_shape {common_shape} from tumor_nifti.")
        else:
            logger.error("Could not load tumor_nifti to derive shape. GM/WM also missing/invalid.")
    
    if tissue_data['affine'] is None: # Absolute fallback if no NIfTI provided affine
        logger.warning("No NIfTI found to determine affine. Using default identity affine and 1mm voxel size.")
        tissue_data['affine'] = np.eye(4)
        for i in range(3): tissue_data['affine'][i,i] = DEFAULT_VOXEL_SIZE_MM
        if common_shape is None: # If shape is still None, this is a problem
             raise ValueError("Cannot determine tissue domain shape. No valid NIfTI inputs and no default shape specified.")

    tissue_data['voxel_volume'] = utils.get_voxel_volume_from_affine(tissue_data['affine'])
    if common_shape is None: # If still none after trying tumor, it's an issue
        raise ValueError("Critical: Could not determine a common shape for tissue data.")
    tissue_data['shape'] = common_shape

    # Initialize all mask keys to prevent KeyErrors, ensuring they have the common_shape
    for key_mask in ['CSF', 'Tumor_Max_Extent', 'Tumor', 'tumor_rim_mask', 'tumor_core_mask', 'VEGF_field']:
        if key_mask not in tissue_data or tissue_data[key_mask] is None: # Check if None also
            if key_mask == 'VEGF_field':
                tissue_data[key_mask] = np.zeros(common_shape, dtype=float)
            else:
                tissue_data[key_mask] = np.zeros(common_shape, dtype=bool)

    if paths.get("csf_nifti"):
        csf_data, _, _ = io_utils.load_nifti_image(paths["csf_nifti"])
        if csf_data is not None and csf_data.shape == common_shape: tissue_data['CSF'] = csf_data.astype(bool)
        elif csf_data is not None: logger.warning(f"CSF shape {csf_data.shape} mismatch domain {common_shape}. Skipping.")

    if paths.get("tumor_nifti"): # This is the FINAL tumor extent
        tumor_data_final, _, _ = io_utils.load_nifti_image(paths["tumor_nifti"]) # Affine already set or taken from this if needed
        if tumor_data_final is not None and tumor_data_final.shape == common_shape:
            tissue_data['Tumor_Max_Extent'] = tumor_data_final.astype(bool)
            logger.info(f"Loaded Tumor_Max_Extent segmentation: {np.sum(tissue_data['Tumor_Max_Extent'])} voxels.")
        elif tumor_data_final is not None:
            logger.warning(f"Tumor_Max_Extent shape {tumor_data_final.shape} mismatch domain {common_shape}. Using empty.")

    # Domain mask for *healthy* GBO growth.
    # Tumor growth will occur *within* Tumor_Max_Extent, potentially converting these healthy regions.
    domain_mask_healthy = np.zeros(common_shape, dtype=bool)
    if tissue_data['WM'] is not None: domain_mask_healthy = np.logical_or(domain_mask_healthy, tissue_data['WM'])
    if tissue_data['GM'] is not None: domain_mask_healthy = np.logical_or(domain_mask_healthy, tissue_data['GM'])
    # Ensure healthy domain does not initially include the pre-defined final tumor area if you want tumor to "invade" it.
    # Or, if healthy GBO should also vascularize the future tumor area, include it.
    # For Option 1 (tumor grows into pre-existing healthy tissue), domain_mask_healthy can overlap Tumor_Max_Extent.
    tissue_data['domain_mask'] = domain_mask_healthy
    logger.info(f"Initial healthy GBO domain_mask: {np.sum(domain_mask_healthy)} voxels.")

    voxel_indices = np.array(np.where(tissue_data['domain_mask'])).T
    if voxel_indices.size > 0:
        tissue_data['voxel_indices_flat'] = voxel_indices
        tissue_data['world_coords_flat'] = utils.voxel_to_world(voxel_indices, tissue_data['affine'])
    else:
        tissue_data['voxel_indices_flat'] = np.empty((0,3), dtype=int)
        tissue_data['world_coords_flat'] = np.empty((0,3), dtype=float)
        logger.warning("Initial healthy domain_mask is empty. Healthy GBO might not run effectively.")

    # Initial metabolic demand map for HEALTHY tissue only.
    segmentations_for_healthy_demand = {k: tissue_data[k] for k in ['WM', 'GM', 'CSF'] if tissue_data.get(k) is not None and np.any(tissue_data[k])}
    tissue_data['metabolic_demand_map'] = data_structures.get_metabolic_demand_map(
        segmentations_for_healthy_demand, config, tissue_data['voxel_volume']
    )
    if tissue_data['metabolic_demand_map'] is None:
        tissue_data['metabolic_demand_map'] = np.zeros(common_shape, dtype=np.float32)


    initial_arterial_graph = None
    centerline_path = paths.get("arterial_centerlines")
    if centerline_path:
        poly_data = None
        if centerline_path.endswith(".vtp"):
            poly_data = io_utils.load_arterial_centerlines_vtp(centerline_path)
        elif centerline_path.endswith(".txt"):
            default_radius_centerline = config_manager.get_param(config, "vascular_properties.centerline_default_radius", 0.1)
            poly_data = io_utils.load_arterial_centerlines_txt(centerline_path, radius_default=default_radius_centerline)
        else: logger.warning(f"Unsupported arterial centerline file format: {centerline_path}")

        if poly_data and poly_data.n_points > 0:
            initial_arterial_graph = data_structures.create_empty_vascular_graph()
            node_id_counter = 0; pv_point_to_nx_node = {}
            default_min_radius = config_manager.get_param(config, "vascular_properties.min_radius", 0.01)

            for pt_idx in range(poly_data.n_points):
                pos = poly_data.points[pt_idx]
                radius_array = poly_data.point_data.get('radius')
                radius = radius_array[pt_idx] if radius_array is not None and pt_idx < len(radius_array) else default_min_radius
                current_node_id = f"m_{node_id_counter}"; node_id_counter += 1
                data_structures.add_node_to_graph(initial_arterial_graph, current_node_id, pos=pos, radius=radius, type='measured_segment_point')
                pv_point_to_nx_node[pt_idx] = current_node_id
            
            lines_array = poly_data.lines; current_idx = 0
            while current_idx < len(lines_array):
                num_pts_in_segment = lines_array[current_idx]
                if num_pts_in_segment == 2:
                    node_u_id = pv_point_to_nx_node.get(lines_array[current_idx + 1])
                    node_v_id = pv_point_to_nx_node.get(lines_array[current_idx + 2])
                    if node_u_id and node_v_id:
                        rad_u = initial_arterial_graph.nodes[node_u_id]['radius']
                        rad_v = initial_arterial_graph.nodes[node_v_id]['radius']
                        # Prefer direction from larger to smaller radius, default u->v
                        if rad_u < rad_v: # Swap if v is larger
                            node_u_id, node_v_id = node_v_id, node_u_id
                        data_structures.add_edge_to_graph(initial_arterial_graph, node_u_id, node_v_id, type='measured_segment')
                else: logger.warning(f"Skipping polyline segment with {num_pts_in_segment} points in VTP. Expected 2.")
                current_idx += (num_pts_in_segment + 1)
            
            for node_id in list(initial_arterial_graph.nodes()): # Refine node types
                in_deg = initial_arterial_graph.in_degree(node_id)
                out_deg = initial_arterial_graph.out_degree(node_id)
                if in_deg == 0 and out_deg > 0 : initial_arterial_graph.nodes[node_id]['type'] = 'measured_root'
                elif out_deg == 0 and in_deg > 0: initial_arterial_graph.nodes[node_id]['type'] = 'measured_terminal'
                elif in_deg > 0 and out_deg > 1 : initial_arterial_graph.nodes[node_id]['type'] = 'measured_bifurcation'
                elif in_deg > 1 and out_deg > 0 : initial_arterial_graph.nodes[node_id]['type'] = 'measured_convergence' # Or bifurcation if undirected
                elif in_deg == 1 and out_deg == 1: initial_arterial_graph.nodes[node_id]['type'] = 'measured_segment_point'
                # else: isolated node or complex junction
            logger.info(f"Loaded initial arterial graph: {initial_arterial_graph.number_of_nodes()} N, {initial_arterial_graph.number_of_edges()} E.")
    else: logger.info("No arterial centerline file specified. GBO will start from config seeds or fallback.")

    return tissue_data, initial_arterial_graph


def main():
    args = parse_arguments()
    try: config = config_manager.load_config(args.config)
    except Exception as e: print(f"CRITICAL: Failed to load config: {e}"); return

    sim_name = config_manager.get_param(config, "simulation.simulation_name", "gbo_sim")
    base_output_dir = args.output_dir if args.output_dir else config_manager.get_param(config, "paths.output_dir", "output")
    output_dir = utils.create_output_directory(base_output_dir, sim_name, timestamp=True)
    
    log_level = config_manager.get_param(config, "simulation.log_level", "INFO")
    setup_logging(log_level, os.path.join(output_dir, f"{sim_name}.log"))
    
    main_logger = logging.getLogger(__name__)
    main_logger.info(f"Simulation started. Output: {output_dir}. Config: {os.path.abspath(args.config)}")
    io_utils.save_simulation_parameters(config, os.path.join(output_dir, "config_used.yaml"))
    
    seed_val = config_manager.get_param(config, "simulation.random_seed", None)
    if seed_val is not None: utils.set_rng_seed(seed_val)

    start_time = time.time()
    try:
        tissue_data, initial_arterial_graph = load_initial_data(config)
    except Exception as e:
        main_logger.critical(f"Data loading failed: {e}", exc_info=True)
        return
    
    if config_manager.get_param(config, "visualization.save_initial_masks", False):
        for key in ['WM', 'GM', 'CSF', 'Tumor_Max_Extent', 'domain_mask', 'metabolic_demand_map']:
            if key in tissue_data and tissue_data[key] is not None and np.any(tissue_data[key]):
                arr_to_save = tissue_data[key]
                if arr_to_save.dtype == bool: arr_to_save = arr_to_save.astype(np.uint8)
                try:
                    io_utils.save_nifti_image(arr_to_save.astype(np.float32 if key == 'metabolic_demand_map' else np.uint8), # Ensure correct dtype
                                              tissue_data['affine'], 
                                              os.path.join(output_dir, f"initial_tissue_{key}.nii.gz"))
                except Exception as e_save:
                    main_logger.error(f"Could not save initial mask {key}: {e_save}")


    healthy_vascular_tree = None
    if config_manager.get_param(config, "gbo_growth.enabled", True):
        main_logger.info("Starting healthy vascular development (GBO)...")
        healthy_vascular_tree = vascular_growth.grow_healthy_vasculature(
            config=config, tissue_data=tissue_data, initial_graph=initial_arterial_graph, output_dir=output_dir
        )
        if healthy_vascular_tree:
            main_logger.info(f"Healthy GBO finished. Tree: {healthy_vascular_tree.number_of_nodes()} N, {healthy_vascular_tree.number_of_edges()} E.")
            io_utils.save_vascular_tree_vtp(healthy_vascular_tree, os.path.join(output_dir, "healthy_vascular_tree.vtp"))
        else: main_logger.error("Healthy GBO failed or returned no tree.")
    else:
        main_logger.info("Healthy GBO growth skipped by config.")
        if initial_arterial_graph: # Use initial graph if GBO is skipped but centerlines were provided
            healthy_vascular_tree = initial_arterial_graph
            main_logger.info("Using provided initial arterial graph as base for subsequent steps.")


    final_vascular_tree = healthy_vascular_tree if healthy_vascular_tree else data_structures.create_empty_vascular_graph()

    if config_manager.get_param(config, "tumor_angiogenesis.enabled", False):
        if tissue_data.get('Tumor_Max_Extent') is not None and np.any(tissue_data['Tumor_Max_Extent']):
            main_logger.info("Starting tumor growth and angiogenesis simulation...")
            # Pass a copy of the healthy tree to angiogenesis module
            base_for_angiogenesis = healthy_vascular_tree.copy() if healthy_vascular_tree else data_structures.create_empty_vascular_graph()
            final_vascular_tree = angiogenesis.simulate_tumor_angiogenesis_fixed_extent(
                config=config, tissue_data=tissue_data, # tissue_data is modified in-place
                base_vascular_tree=base_for_angiogenesis,
                output_dir=output_dir,
                perfusion_solver_func=perfusion_solver.solve_1d_poiseuille_flow
            )
            if final_vascular_tree:
                main_logger.info(f"Tumor angiogenesis finished. Final tree: {final_vascular_tree.number_of_nodes()} N, {final_vascular_tree.number_of_edges()} E.")
                io_utils.save_vascular_tree_vtp(final_vascular_tree, os.path.join(output_dir, "final_tumor_vascular_tree.vtp"))
            else: # Fallback if angiogenesis returns None
                main_logger.error("Tumor angiogenesis returned no tree. Using pre-angiogenesis tree.")
                final_vascular_tree = base_for_angiogenesis # Or healthy_vascular_tree
        else: main_logger.info("Tumor angiogenesis skipped (No Tumor_Max_Extent defined or empty).")
    else: main_logger.info("Tumor angiogenesis disabled in config.")

    perfusion_map_3d, pressure_map_3d_tissue = None, None # For 3D tissue grid perfusion
    # The 1D solver updates the graph directly.
    # A more advanced solver would produce 3D tissue perfusion maps.
    if config_manager.get_param(config, "perfusion_solver.run_final_1d_flow_solve", True) and final_vascular_tree and final_vascular_tree.number_of_nodes() > 0:
        main_logger.info("Running final 1D flow solve on the final vascular tree...")
        # TODO: Robustly update Q_flow demands for ALL terminals (healthy, tumor) based on final territories
        # This is a complex step involving re-assigning territories (e.g. Voronoi) to all terminals
        # and summing metabolic_demand_map from tissue_data for each territory.
        # For now, the Q_flow values set during GBO/Angiogenesis will be used if not updated here.
        # Example of simple update (could be more sophisticated):
        for node_id_f, data_f in final_vascular_tree.nodes(data=True):
            if final_vascular_tree.out_degree(node_id_f) == 0 and final_vascular_tree.in_degree(node_id_f) > 0: # Is a terminal
                # Placeholder: Re-evaluate demand based on its final type/location
                if data_f.get('is_tumor_vessel'):
                    data_f['Q_flow'] = config_manager.get_param(config, "tumor_angiogenesis.min_tumor_terminal_demand", DEFAULT_MIN_TUMOR_TERMINAL_DEMAND)
                # else: (for healthy terminals, their demand might have been set by GBO)
        
        final_vascular_tree_with_flow = perfusion_solver.solve_1d_poiseuille_flow(
            final_vascular_tree.copy(), config, None, None # Use Q_flow from nodes
        )
        if final_vascular_tree_with_flow:
            final_vascular_tree = final_vascular_tree_with_flow
            io_utils.save_vascular_tree_vtp(final_vascular_tree, os.path.join(output_dir, "final_vascular_tree_with_flowdata.vtp"))
            main_logger.info("Final 1D flow solution computed and saved.")
        else:
            main_logger.error("Final 1D flow solution failed.")
    else:
        main_logger.info("Final 1D flow solve skipped.")

    main_logger.info("Generating final visualizations...")
    visualization.generate_final_visualizations(
        config=config, output_dir=output_dir, tissue_data=tissue_data,
        vascular_graph=final_vascular_tree,
        perfusion_map=perfusion_map_3d, # This would be from an advanced solver
        pressure_map_tissue=pressure_map_3d_tissue, # This also
        plot_context_masks=config_manager.get_param(config, "visualization.plot_context_masks_final", True)
    )

    main_logger.info(f"Simulation finished. Total time: {time.time() - start_time:.2f}s. Output: {output_dir}")

if __name__ == "__main__":
    # Basic setup for running directly, assuming config.yaml and data dir exist
    if not os.path.exists("config.yaml"):
        # Attempt to create a default config if config_manager supports it
        try:
            config_manager.create_default_config("config.yaml") # You'd need to implement this
            print("Created default config.yaml. Please review and populate paths, especially for NIfTI files.")
        except AttributeError:
            print("Warning: config_manager.create_default_config not found. Please ensure config.yaml exists.")
        except Exception as e:
            print(f"Warning: Could not create default config.yaml: {e}")

    os.makedirs(config_manager.get_param(config_manager.load_config("config.yaml" if os.path.exists("config.yaml") else {}), "paths.input_data_dir", "data"), exist_ok=True)
    os.makedirs(config_manager.get_param(config_manager.load_config("config.yaml" if os.path.exists("config.yaml") else {}), "paths.output_dir", "output"), exist_ok=True)
    main()