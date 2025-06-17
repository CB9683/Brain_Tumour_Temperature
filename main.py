# main.py
import argparse
import logging
import os
import time
import numpy as np
import networkx as nx # For type hinting and graph operations
from typing import Optional

# PyVista import for VTP parsing, ensure it's available (used by io_utils indirectly)
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None

from src import config_manager, io_utils, utils
from src import data_structures
from src import vascular_growth, angiogenesis, perfusion_solver, visualization
from src.constants import DEFAULT_VOXEL_SIZE_MM, Q_MET_TUMOR_RIM_PER_ML, INITIAL_TERMINAL_FLOW_Q # Added more constants

def setup_logging(log_level_str: str, log_file: str):
    """Configures logging for the simulation."""
    numeric_level = getattr(logging, log_level_str.upper(), logging.INFO)
    if not isinstance(numeric_level, int): # Fallback if level is invalid
        print(f"Warning: Invalid log level '{log_level_str}'. Defaulting to INFO.")
        numeric_level = logging.INFO

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
    # Suppress overly verbose logs if not in DEBUG
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
    logger = logging.getLogger(__name__)
    logger.info("Loading initial data...")
    
    tissue_data = {
        'WM': None, 'GM': None, 'CSF': None, 
        'Tumor_Max_Extent': None, 'Tumor': None, 
        'tumor_rim_mask': None, 'tumor_core_mask': None,
        'VEGF_field': None, 'affine': None, 'voxel_volume': None, 
        'domain_mask': None, 'metabolic_demand_map': None,
        'world_coords_flat': None, 'voxel_indices_flat': None,
        'shape': None
    }

    paths = config_manager.get_param(config, "paths", {})
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

    if common_shape is None and paths.get("tumor_nifti"):
        logger.info("Attempting to derive shape and affine from tumor_nifti as GM/WM are missing/invalid.")
        tumor_final_data_for_shape, affine_tumor_for_shape, _ = io_utils.load_nifti_image(paths["tumor_nifti"])
        if tumor_final_data_for_shape is not None:
            common_shape = tumor_final_data_for_shape.shape
            if tissue_data['affine'] is None and affine_tumor_for_shape is not None:
                tissue_data['affine'] = affine_tumor_for_shape
            logger.info(f"Derived common_shape {common_shape} from tumor_nifti.")
    
    if tissue_data['affine'] is None:
        logger.warning("No NIfTI found for affine. Using default identity affine, 1mm voxel size.")
        tissue_data['affine'] = np.eye(4)
        for i in range(3): tissue_data['affine'][i,i] = DEFAULT_VOXEL_SIZE_MM
        if common_shape is None:
             raise ValueError("Cannot determine tissue domain shape and no affine available.")

    tissue_data['voxel_volume'] = utils.get_voxel_volume_from_affine(tissue_data['affine'])
    if common_shape is None:
        raise ValueError("Critical: Could not determine a common shape for tissue data.")
    tissue_data['shape'] = common_shape

    for key_mask in ['CSF', 'Tumor_Max_Extent', 'Tumor', 'tumor_rim_mask', 'tumor_core_mask', 'VEGF_field']:
        if key_mask not in tissue_data or tissue_data[key_mask] is None:
            tissue_data[key_mask] = np.zeros(common_shape, dtype=float if key_mask == 'VEGF_field' else bool)

    if paths.get("csf_nifti"):
        csf_data, _, _ = io_utils.load_nifti_image(paths["csf_nifti"])
        if csf_data is not None and csf_data.shape == common_shape: tissue_data['CSF'] = csf_data.astype(bool)
        elif csf_data is not None: logger.warning(f"CSF shape {csf_data.shape} mismatch domain {common_shape}. Skipping.")

    if paths.get("tumor_nifti"):
        tumor_data_final, affine_tumor, _ = io_utils.load_nifti_image(paths["tumor_nifti"])
        if tumor_data_final is not None and tumor_data_final.shape == common_shape:
            tissue_data['Tumor_Max_Extent'] = tumor_data_final.astype(bool)
            # Affine should be set by now, but double check if only tumor was provided
            if tissue_data['affine'] is None and affine_tumor is not None:
                tissue_data['affine'] = affine_tumor
                tissue_data['voxel_volume'] = utils.get_voxel_volume_from_affine(tissue_data['affine'])
            logger.info(f"Loaded Tumor_Max_Extent segmentation: {np.sum(tissue_data['Tumor_Max_Extent'])} voxels.")
        elif tumor_data_final is not None:
            logger.warning(f"Tumor_Max_Extent shape {tumor_data_final.shape} mismatch domain {common_shape}. Using empty.")

    domain_mask_healthy = np.zeros(common_shape, dtype=bool)
    if tissue_data.get('WM') is not None: domain_mask_healthy = np.logical_or(domain_mask_healthy, tissue_data['WM'])
    if tissue_data.get('GM') is not None: domain_mask_healthy = np.logical_or(domain_mask_healthy, tissue_data['GM'])
    tissue_data['domain_mask'] = domain_mask_healthy
    logger.info(f"Initial healthy GBO domain_mask set with {np.sum(domain_mask_healthy)} voxels (may overlap future tumor area).")

    voxel_indices = np.array(np.where(tissue_data['domain_mask'])).T
    if voxel_indices.size > 0:
        tissue_data['voxel_indices_flat'] = voxel_indices
        tissue_data['world_coords_flat'] = utils.voxel_to_world(voxel_indices, tissue_data['affine'])
    else:
        tissue_data['voxel_indices_flat'] = np.empty((0,3), dtype=int)
        tissue_data['world_coords_flat'] = np.empty((0,3), dtype=float)
        logger.warning("Initial healthy domain_mask is empty. Healthy GBO might not run or will rely on fallback seeding if enabled.")

    segmentations_for_healthy_demand = {k: tissue_data[k] for k in ['WM', 'GM', 'CSF'] if tissue_data.get(k) is not None and np.any(tissue_data[k])}
    tissue_data['metabolic_demand_map'] = data_structures.get_metabolic_demand_map(
        segmentations_for_healthy_demand, config, tissue_data['voxel_volume']
    )
    if tissue_data['metabolic_demand_map'] is None: # Should be initialized by get_metabolic_demand_map
        tissue_data['metabolic_demand_map'] = np.zeros(common_shape, dtype=np.float32)


    initial_arterial_graph = None
    centerline_path_key = "arterial_centerlines" # Key in config
    centerline_file_str = paths.get(centerline_path_key)
    
    if centerline_file_str:
        input_data_base_dir = paths.get("input_data_dir", ".")
        if not os.path.isabs(centerline_file_str):
            full_centerline_path = os.path.join(input_data_base_dir, centerline_file_str)
        else:
            full_centerline_path = centerline_file_str
        
        logger.info(f"Attempting to load arterial centerlines from: {full_centerline_path}")
        poly_data = None
        if full_centerline_path.endswith((".vtp", ".vtk")): # Allow .vtk too
            poly_data = io_utils.load_arterial_centerlines_vtp(full_centerline_path)
        elif full_centerline_path.endswith(".txt"):
            default_radius_centerline = config_manager.get_param(config, "vascular_properties.centerline_default_radius", 0.1)
            poly_data = io_utils.load_arterial_centerlines_txt(full_centerline_path, radius_default=default_radius_centerline)
        else: logger.warning(f"Unsupported arterial centerline file format: {full_centerline_path}")

        if poly_data and poly_data.n_points > 0:
            logger.info(f"Processing VTP/PolyData with {poly_data.n_points} points and {poly_data.n_cells} cells.")
            initial_arterial_graph = data_structures.create_empty_vascular_graph()
            pv_point_to_nx_node = {} 
            node_id_counter = 0
            default_min_radius = config_manager.get_param(config, "vascular_properties.min_radius", 0.01)

            radii_array = poly_data.point_data.get('Radius') # Case-sensitive
            if radii_array is None:
                logger.warning("'Radius' point data array not found in VTP. Using default_min_radius for all points.")
            elif len(radii_array) != poly_data.n_points:
                logger.warning(f"Mismatch in length of 'Radius' array ({len(radii_array)}) and n_points ({poly_data.n_points}). Using default_min_radius.")
                radii_array = None

            for pt_idx in range(poly_data.n_points):
                pos = poly_data.points[pt_idx]
                radius = float(radii_array[pt_idx]) if radii_array is not None else default_min_radius
                current_node_id = f"m_{node_id_counter}"; node_id_counter += 1
                data_structures.add_node_to_graph(initial_arterial_graph, current_node_id,
                                                  pos=pos, radius=radius, type='measured_point')
                pv_point_to_nx_node[pt_idx] = current_node_id
            
            logger.debug(f"Added {initial_arterial_graph.number_of_nodes()} nodes from VTP points.")

            vtp_polyline_start_nodes = set()
            vtp_polyline_end_nodes = set()

            for i_cell in range(poly_data.n_cells):
                cell_point_indices = poly_data.get_cell(i_cell).point_ids
                if len(cell_point_indices) < 1: continue # Should not happen for valid cells
                
                start_node_idx_original_vtp = cell_point_indices[0]
                start_node_id_nx = pv_point_to_nx_node.get(start_node_idx_original_vtp)
                if start_node_id_nx: vtp_polyline_start_nodes.add(start_node_id_nx)

                if len(cell_point_indices) >= 2:
                    end_node_idx_original_vtp = cell_point_indices[-1]
                    end_node_id_nx = pv_point_to_nx_node.get(end_node_idx_original_vtp)
                    if end_node_id_nx: vtp_polyline_end_nodes.add(end_node_id_nx)

                    for k_edge in range(len(cell_point_indices) - 1):
                        node_u_id = pv_point_to_nx_node.get(cell_point_indices[k_edge])
                        node_v_id = pv_point_to_nx_node.get(cell_point_indices[k_edge + 1])
                        if node_u_id and node_v_id and node_u_id != node_v_id:
                            rad_u = initial_arterial_graph.nodes[node_u_id]['radius']
                            rad_v = initial_arterial_graph.nodes[node_v_id]['radius']
                            source_node, target_node = node_u_id, node_v_id
                            if (rad_u < rad_v and not np.isclose(rad_u, rad_v)): # Prefer flow from larger to smaller
                                source_node, target_node = node_v_id, node_u_id
                            if not initial_arterial_graph.has_edge(source_node, target_node):
                                data_structures.add_edge_to_graph(initial_arterial_graph, source_node, target_node, type='measured_segment')
            
            logger.info(f"Added {initial_arterial_graph.number_of_edges()} edges from VTP cells.")
            logger.info(f"Identified {len(vtp_polyline_start_nodes)} VTP polyline start nodes and {len(vtp_polyline_end_nodes)} VTP polyline end nodes.")
            
            for node_id in list(initial_arterial_graph.nodes()):
                in_deg = initial_arterial_graph.in_degree(node_id)
                out_deg = initial_arterial_graph.out_degree(node_id)
                node_data = initial_arterial_graph.nodes[node_id]
                is_polyline_end = node_id in vtp_polyline_end_nodes

                if in_deg == 0 and out_deg > 0 : node_data['type'] = 'measured_root'
                elif out_deg == 0 and in_deg > 0:
                    if is_polyline_end: node_data['type'] = 'measured_terminal'
                    else: node_data['type'] = 'measured_segment_point'; logger.debug(f"Node {node_id} out_deg=0 but not VTP end, type: seg_point.")
                elif out_deg > 1: node_data['type'] = 'measured_bifurcation'
                elif in_deg > 1 and out_deg == 1: node_data['type'] = 'measured_convergence'
                elif in_deg == 1 and out_deg == 1: node_data['type'] = 'measured_segment_point'
                elif in_deg == 0 and out_deg == 0: node_data['type'] = 'measured_isolated_point'
                else: node_data['type'] = 'measured_complex_junction'; logger.debug(f"Node {node_id} complex: in={in_deg}, out={out_deg}.")

            num_roots = sum(1 for _, data in initial_arterial_graph.nodes(data=True) if data['type'] == 'measured_root')
            num_terminals = sum(1 for _, data in initial_arterial_graph.nodes(data=True) if data['type'] == 'measured_terminal')
            logger.info(f"Refined node types: {num_roots} roots, {num_terminals} measured_terminals.")
    else: logger.info("No 'arterial_centerlines' path specified in config.")

    return tissue_data, initial_arterial_graph


def main():
    args = parse_arguments()
    try:
        config = config_manager.load_config(args.config)
    except Exception as e:
        print(f"CRITICAL: Failed to load configuration file '{args.config}': {e}")
        try: # Attempt to set up basic logging for more details on config load failure
            os.makedirs("output/error_logs", exist_ok=True)
            setup_logging("ERROR", "output/error_logs/config_load_error.log") # Log to a fixed error file
            logging.getLogger(__name__).critical(f"Failed to load configuration file '{args.config}': {e}", exc_info=True)
        except Exception as log_e: print(f"Additionally, failed to set up error logging: {log_e}")
        return

    sim_name = config_manager.get_param(config, "simulation.simulation_name", "gbo_sim")
    base_output_dir_config = config_manager.get_param(config, "paths.output_dir", "output")
    base_output_dir = args.output_dir if args.output_dir else base_output_dir_config
    
    os.makedirs(base_output_dir, exist_ok=True) # Ensure base output dir exists
    output_dir = utils.create_output_directory(base_output_dir, sim_name, timestamp=True)
    
    log_level = config_manager.get_param(config, "simulation.log_level", "INFO")
    log_file_path = os.path.join(output_dir, f"{sim_name}.log")
    setup_logging(log_level, log_file_path)
    
    main_logger = logging.getLogger(__name__)
    main_logger.info(f"Simulation started. Output directory: {output_dir}")
    main_logger.info(f"Using configuration file: {os.path.abspath(args.config)}")

    try: io_utils.save_simulation_parameters(config, os.path.join(output_dir, "config_used.yaml"))
    except Exception as e_save_config: main_logger.error(f"Could not save used configuration file: {e_save_config}")

    seed_val = config_manager.get_param(config, "simulation.random_seed", None)
    if seed_val is not None:
        try: utils.set_rng_seed(int(seed_val))
        except ValueError: main_logger.warning(f"Invalid random_seed value '{seed_val}'. Using system default.")

    start_time = time.time()
    main_logger.info("--- Loading Initial Data ---")
    try:
        tissue_data, initial_arterial_graph = load_initial_data(config)
    except ValueError as ve:
        main_logger.critical(f"Critical error during data loading: {ve}", exc_info=True); return
    except Exception as e:
        main_logger.critical(f"Unexpected error during data loading: {e}", exc_info=True); return
    
    if config_manager.get_param(config, "visualization.plot_initial_setup", True):
        main_logger.info("--- Visualizing Initial Setup ---")
        try:
            visualization.visualize_initial_setup(config=config, output_dir=output_dir,
                                                  tissue_data=tissue_data, initial_arterial_graph=initial_arterial_graph)
        except Exception as e_viz_init: main_logger.error(f"Failed to generate initial setup visualization: {e_viz_init}", exc_info=True)
    
    if config_manager.get_param(config, "visualization.save_initial_masks", False):
        main_logger.info("--- Saving Initial Masks (as NIfTI) ---")
        for key in ['WM', 'GM', 'CSF', 'Tumor_Max_Extent', 'domain_mask', 'metabolic_demand_map', 'Tumor', 'tumor_rim_mask', 'tumor_core_mask', 'VEGF_field']:
            if key in tissue_data and isinstance(tissue_data[key], np.ndarray) and np.any(tissue_data[key]):
                arr_to_save = tissue_data[key]
                dtype_to_save = np.float32 if key in ['metabolic_demand_map', 'VEGF_field'] else np.uint8
                if arr_to_save.dtype == bool: arr_to_save = arr_to_save.astype(np.uint8)
                if tissue_data.get('affine') is None: main_logger.warning(f"Cannot save '{key}': Affine missing."); continue
                try: io_utils.save_nifti_image(arr_to_save.astype(dtype_to_save), tissue_data['affine'], os.path.join(output_dir, f"debug_initial_tissue_{key}.nii.gz"))
                except Exception as e_save: main_logger.error(f"Could not save initial mask {key}: {e_save}")
        if initial_arterial_graph and initial_arterial_graph.number_of_nodes() > 0 :
            try: io_utils.save_vascular_tree_vtp(initial_arterial_graph, os.path.join(output_dir, "debug_initial_arterial_graph_parsed.vtp"))
            except Exception as e_save_vtp: main_logger.error(f"Could not save initial parsed arterial graph: {e_save_vtp}")

    healthy_vascular_tree = None
    if config_manager.get_param(config, "gbo_growth.enabled", True):
        main_logger.info("--- Starting Healthy Vascular Development (GBO) ---")
        healthy_vascular_tree = vascular_growth.grow_healthy_vasculature(
            config=config, tissue_data=tissue_data, initial_graph=initial_arterial_graph, output_dir=output_dir
        )
        if healthy_vascular_tree:
            main_logger.info(f"Healthy GBO finished. Tree: {healthy_vascular_tree.number_of_nodes()} N, {healthy_vascular_tree.number_of_edges()} E.")
            io_utils.save_vascular_tree_vtp(healthy_vascular_tree, os.path.join(output_dir, "healthy_vascular_tree.vtp"))
        else: main_logger.error("Healthy GBO failed or returned no tree.")
    else:
        main_logger.info("Healthy GBO growth skipped by config.")
        healthy_vascular_tree = initial_arterial_graph if initial_arterial_graph else data_structures.create_empty_vascular_graph()
        if initial_arterial_graph: main_logger.info("Using provided initial arterial graph as base for subsequent steps.")
        else: main_logger.info("No initial arterial graph and GBO skipped. Starting with empty tree.")


    final_vascular_tree = healthy_vascular_tree if healthy_vascular_tree else data_structures.create_empty_vascular_graph()

    if config_manager.get_param(config, "tumor_angiogenesis.enabled", False):
        if tissue_data.get('Tumor_Max_Extent') is not None and np.any(tissue_data['Tumor_Max_Extent']):
            main_logger.info("--- Starting Tumor Growth and Angiogenesis Simulation ---")
            base_for_angiogenesis = final_vascular_tree.copy() # Use the result of GBO (or initial graph if GBO skipped)
            final_vascular_tree = angiogenesis.simulate_tumor_angiogenesis_fixed_extent(
                config=config, tissue_data=tissue_data,
                base_vascular_tree=base_for_angiogenesis,
                output_dir=output_dir,
                perfusion_solver_func=perfusion_solver.solve_1d_poiseuille_flow
            )
            if final_vascular_tree:
                main_logger.info(f"Tumor angiogenesis finished. Final tree: {final_vascular_tree.number_of_nodes()} N, {final_vascular_tree.number_of_edges()} E.")
                io_utils.save_vascular_tree_vtp(final_vascular_tree, os.path.join(output_dir, "final_tumor_vascular_tree.vtp"))
            else:
                main_logger.error("Tumor angiogenesis returned no tree. Using pre-angiogenesis tree.")
                final_vascular_tree = base_for_angiogenesis 
        else: main_logger.info("Tumor angiogenesis skipped (No Tumor_Max_Extent defined or empty).")
    else: main_logger.info("Tumor angiogenesis disabled in config.")

    perfusion_map_3d, pressure_map_3d_tissue = None, None
    if config_manager.get_param(config, "perfusion_solver.run_final_1d_flow_solve", True) and \
       final_vascular_tree and final_vascular_tree.number_of_nodes() > 0:
        main_logger.info("--- Running Final 1D Flow Solve ---")
        
        for node_id_f, data_f in final_vascular_tree.nodes(data=True):
            if final_vascular_tree.out_degree(node_id_f) == 0 and final_vascular_tree.in_degree(node_id_f) > 0:
                term_pos_vox = np.round(utils.world_to_voxel(data_f['pos'], tissue_data['affine'])).astype(int)
                demand = data_f.get('Q_flow', 0.0) 
                if utils.is_voxel_in_bounds(term_pos_vox, tissue_data['shape']):
                    if data_f.get('is_tumor_vessel') and tissue_data.get('Tumor') is not None and tissue_data['Tumor'][tuple(term_pos_vox)]:
                        demand = config_manager.get_param(config, "tumor_angiogenesis.min_tumor_terminal_demand", DEFAULT_MIN_TUMOR_TERMINAL_DEMAND)
                    elif not data_f.get('is_tumor_vessel') and tissue_data.get('metabolic_demand_map') is not None:
                        # For healthy terminals, try to get demand from the metabolic map at their location
                        # This is a simplification; proper territory demand is better.
                        demand = tissue_data['metabolic_demand_map'][tuple(term_pos_vox)]
                        if demand < constants.EPSILON: # If healthy tissue demand is zero here for some reason
                            demand = constants.INITIAL_TERMINAL_FLOW_Q # Fallback small flow
                data_f['Q_flow'] = demand
        
        final_tree_copy_for_solve = final_vascular_tree.copy()
        final_vascular_tree_with_flow = perfusion_solver.solve_1d_poiseuille_flow(
            final_tree_copy_for_solve, config, None, None
        )
        if final_vascular_tree_with_flow:
            final_vascular_tree = final_vascular_tree_with_flow
            io_utils.save_vascular_tree_vtp(final_vascular_tree, os.path.join(output_dir, "final_vascular_tree_with_flowdata.vtp"))
            main_logger.info("Final 1D flow solution computed and saved on tree.")
        else: main_logger.error("Final 1D flow solution failed.")
    else: main_logger.info("Final 1D flow solve skipped.")

    main_logger.info("--- Generating Final Visualizations ---")
    visualization.generate_final_visualizations(
        config=config, output_dir=output_dir, tissue_data=tissue_data,
        vascular_graph=final_vascular_tree,
        perfusion_map=perfusion_map_3d, 
        pressure_map_tissue=pressure_map_3d_tissue,
        plot_context_masks=config_manager.get_param(config, "visualization.plot_context_masks_final", True)
    )

    main_logger.info(f"Simulation finished. Total time: {time.time() - start_time:.2f}s. Output: {output_dir}")

if __name__ == "__main__":
    temp_config_path = "config.yaml"
    if not os.path.exists(temp_config_path):
        try:
            # Assuming config_manager can create a default if you implement it
            # config_manager.create_default_config(temp_config_path)
            print(f"Warning: {temp_config_path} not found. Attempting to run with default parameters where possible.")
            print("Please create a config.yaml or provide one via --config argument.")
            # Create a minimal dummy config if none exists, so get_param doesn't fail immediately
            if not os.path.exists(temp_config_path):
                 with open(temp_config_path, "w") as f_cfg:
                    f_cfg.write("paths:\n  input_data_dir: \"data\"\n  output_dir: \"output/simulation_results\"\n")
                 print(f"Created minimal dummy {temp_config_path}. Please customize it.")
        except Exception as e: print(f"Could not create dummy config: {e}")

    # Ensure data and output directories are attempted to be created based on a minimal config load
    loaded_temp_config = {}
    if os.path.exists(temp_config_path):
        try: loaded_temp_config = config_manager.load_config(temp_config_path)
        except: print(f"Could not load {temp_config_path} for directory creation.")
    
    os.makedirs(config_manager.get_param(loaded_temp_config, "paths.input_data_dir", "data"), exist_ok=True)
    os.makedirs(config_manager.get_param(loaded_temp_config, "paths.output_dir", "output"), exist_ok=True)
    
    main()