# tests/test_vascular_growth.py
import pytest

import logging
import re
import numpy as np
import networkx as nx
import os
import shutil # For cleaning up output directory
from typing import Tuple, List, Optional # For type hinting
from src import vascular_growth, data_structures, io_utils, utils, constants, config_manager

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None

logger = logging.getLogger(__name__) # <<<<<<<<<<<<<< ADD THIS DEFINITION

@pytest.fixture(scope="function")
def simple_gbo_config_and_output():
    # Create a temporary directory for this test module's outputs
    test_module_output_dir = "temp_test_vascular_growth_output"
    if os.path.exists(test_module_output_dir): # Clean up from previous runs
        shutil.rmtree(test_module_output_dir)
    os.makedirs(test_module_output_dir, exist_ok=True)
    
    # Simplified config for a small test run
    config = {
        "paths": {
            "output_dir": test_module_output_dir 
        },
        "simulation": {
            "simulation_name": "gbo_simple_test",
            "random_seed": 42,
            "log_level": "DEBUG", # Use DEBUG to see detailed logs from growth
             "units": {"length": "mm", "pressure": "Pa", "flow_rate": "mm^3/s"}
        },
        "tissue_properties": {
            "metabolic_rates": {"gm": 0.01, "wm": 0.003}, # Simplified rates
            "permeability": {"gm": 1.0e-7, "wm": 5.0e-8}
        },
        "vascular_properties": {
            "blood_viscosity": constants.DEFAULT_BLOOD_VISCOSITY,
            "murray_law_exponent": 3.0,
            "initial_terminal_flow": 1e-5, # Slightly larger initial flow for small domain
            "min_radius": 0.005, # mm
            "k_murray_scaling_factor": 0.5, # s^(1/3)
            "min_segment_length": 0.05, # mm
        },
        "gbo_growth": {
            "max_iterations": 5, # Run for very few iterations
            "energy_coefficient_C_met_vessel_wall": 1.0e-5, # (Units: 10^-9 W / mm^3 or consistent)
            "initial_territory_radius": 0.2, # mm, for Ω_init
            "frontier_search_radius_factor": 3.0, # Smaller factor for small domain
            "frontier_search_radius_fixed": 0.5,  # mm
            "max_voxels_for_Rip": 50, # Small Rip
            "branch_radius_increase_threshold": 1.1, # More sensitive branching
            "max_flow_single_terminal": 0.0001, # Lower threshold for branching
            "target_domain_perfusion_fraction": 0.90, # Lower target for short test
            "bifurcation_candidate_points": 5, # Fewer candidates for speed
            "min_iterations_before_no_growth_stop": 2,
            "stop_criteria": {"max_radius_factor_measured": 1.0} # Not relevant for seed test
        },
        "tumor_angiogenesis": {"enabled": False},
        "perfusion_solver": {"enabled": False},
        "visualization": {
            "save_intermediate_steps": True, # Save steps for inspection
            "intermediate_step_interval": 1
        }
    }
    # Create a unique output dir for this specific test function run
    run_output_dir = utils.create_output_directory(test_module_output_dir, "run_simple_gbo", timestamp=False)

    yield config, run_output_dir # Provide config and specific run output_dir

    # Teardown: Optionally remove the test_module_output_dir after all tests in module are done
    # shutil.rmtree(test_module_output_dir) # Comment out if you want to inspect outputs

@pytest.fixture
def minimal_tissue_data():
    shape = (10, 10, 10) # Small 3D volume
    affine = np.eye(4) 
    affine[0,0] = affine[1,1] = affine[2,2] = 0.1 # Voxel size 0.1 mm (total volume 1mm^3)
    
    domain_mask = np.zeros(shape, dtype=bool)
    domain_mask[2:8, 2:8, 2:8] = True # A central block of tissue
    
    # Define GM and WM regions within the domain_mask
    gm_mask = np.zeros(shape, dtype=bool)
    gm_mask[2:5, 2:8, 2:8] = True # Lower part is GM
    gm_mask = gm_mask & domain_mask # Ensure it's within domain

    wm_mask = np.zeros(shape, dtype=bool)
    wm_mask[5:8, 2:8, 2:8] = True # Upper part is WM
    wm_mask = wm_mask & domain_mask

    metabolic_demand_map_qmet = np.zeros(shape, dtype=np.float32)
    # Use simplified config rates for this test
    metabolic_demand_map_qmet[gm_mask] = 0.01 # q_met for GM (e.g., 1/s)
    metabolic_demand_map_qmet[wm_mask] = 0.003 # q_met for WM
    
    voxel_volume = utils.get_voxel_volume_from_affine(affine)

    # Get flat coordinates and indices for voxels *within domain_mask*
    voxel_indices_3d_domain = np.array(np.where(domain_mask)).T
    world_coords_domain_flat = utils.voxel_to_world(voxel_indices_3d_domain, affine)

    tissue = {
        'shape': shape,
        'affine': affine,
        'voxel_volume': voxel_volume,
        'domain_mask': domain_mask,
        'GM': gm_mask, # Provide GM/WM for seeding logic if used
        'WM': wm_mask,
        # metabolic_demand_map should store q_met (rate per unit volume), 
        # actual demand (q_met * dV) is calculated where needed
        'metabolic_demand_map': metabolic_demand_map_qmet, 
        'world_coords_flat': world_coords_domain_flat, # Coords of voxels IN domain_mask
        'voxel_indices_flat': voxel_indices_3d_domain  # 3D indices of voxels IN domain_mask
    }
    return tissue

def test_vascular_growth_single_seed(simple_gbo_config_and_output, minimal_tissue_data, caplog):
    """
    Test GBO growth starting from a single seed point in a minimal tissue.
    """
    config, output_dir = simple_gbo_config_and_output
    tissue_data = minimal_tissue_data

    # Set log level for more detail during this test if not globally set
    caplog.set_level(logging.DEBUG, logger="src.vascular_growth")
    caplog.set_level(logging.DEBUG, logger="src.energy_model")

    # Ensure RNG seed for deterministic test
    utils.set_rng_seed(config_manager.get_param(config, "simulation.random_seed", 42))

    initial_graph = None # Start with no measured arteries for this test

    # --- Run the growth simulation ---
    final_graph = vascular_growth.grow_healthy_vasculature(
        config=config,
        tissue_data=tissue_data,
        initial_graph=initial_graph,
        output_dir=output_dir
    )
    
     # Determine the actual last iteration for which files would be saved
    # In tests/test_vascular_growth.py

        # ... (after final_graph = vascular_growth.grow_healthy_vasculature(...)) ...

    max_iterations_config = config['gbo_growth']['max_iterations']
        # Default to assuming it ran all iterations
    expected_save_file_iter_num = max_iterations_config 
        
    found_stop_message = False
    for record in caplog.records:
        # logger.debug(f"Test log scan: {record.levelname} - {record.message}") # Uncomment for extreme debug
        if record.levelname == "INFO":
            if "GBO Stopping after iteration" in record.message:
                match = re.search(r"GBO Stopping after iteration (\d+):", record.message)
                if match:
                    expected_save_file_iter_num = int(match.group(1))
                    logger.info(f"Found stop message: '{record.message}'. Expecting files for iter_{expected_save_file_iter_num}.")
                    found_stop_message = True
                    break 
            elif "GBO Stopping:" in record.message and not found_stop_message: # Fallback for other stop messages
                # This fallback is less precise; it will take the iter # of the GBO Iteration log line
                # that appeared *before* this generic stop message.
                # This might be off by one if the generic stop happens after all processing for an iter.
                logger.warning(f"Found generic stop message: '{record.message}'. Attempting to find last processed iteration.")
                # Search backwards for the last "--- GBO Iteration X / Y ---"
                temp_iter_num = 1 
                current_record_index = caplog.records.index(record)
                for i_rec in range(current_record_index - 1, -1, -1):
                    if "--- GBO Iteration" in caplog.records[i_rec].message:
                        match_iter = re.search(r"--- GBO Iteration (\d+)", caplog.records[i_rec].message)
                        if match_iter:
                            temp_iter_num = int(match_iter.group(1))
                            break
                expected_save_file_iter_num = temp_iter_num
                logger.info(f"Fallback: Expecting files for iter_{expected_save_file_iter_num} based on generic stop.")
                found_stop_message = True # Consider it found for overriding the default
                break

    if not found_stop_message:
        logger.info(f"No 'GBO Stopping' message found or parsed. Defaulting to check for iter_{max_iterations_config} files.")
        # expected_save_file_iter_num remains max_iterations_config

    last_iter_perfused_mask_path = os.path.join(output_dir, f"perfused_mask_iter_{expected_save_file_iter_num}.nii.gz")
    logger.info(f"FINAL CHECK for saved mask: {last_iter_perfused_mask_path}")
    assert os.path.exists(last_iter_perfused_mask_path), \
        f"Expected {os.path.basename(last_iter_perfused_mask_path)} was not saved. Check logs for save errors or actual stop iteration."

    last_iter_graph_path = os.path.join(output_dir, f"gbo_graph_iter_{expected_save_file_iter_num}.vtp")
    assert os.path.exists(last_iter_graph_path), \
        f"Expected {os.path.basename(last_iter_graph_path)} was not saved."

    # --- Assertions if files exist ---
    assert final_graph is not None, "Growth function returned None"
    num_initial_nodes = 0 
    assert final_graph.number_of_nodes() > num_initial_nodes, \
        f"Graph did not grow. Started with {num_initial_nodes} nodes, ended with {final_graph.number_of_nodes()}."

    if os.path.exists(last_iter_perfused_mask_path):
        perf_mask_data, _, _ = io_utils.load_nifti_image(last_iter_perfused_mask_path)
        assert perf_mask_data is not None
        min_expected_perf_count = 10 
        if final_graph.number_of_nodes() > 1: 
            assert np.sum(perf_mask_data) > min_expected_perf_count, \
                f"Very few voxels perfused. Perfused count: {np.sum(perf_mask_data)}"
    else: # If the mask file wasn't found, this part of the test can't run.
            # The assertion above would have already failed.
        pass

    logger.info(f"Test `test_vascular_growth_single_seed` completed. Final graph: {final_graph.number_of_nodes()} nodes, {final_graph.number_of_edges()} edges.")
    logger.info(f"Outputs saved in: {output_dir}")

@pytest.fixture(scope="module") # Use module scope if fixture is complex and shared
def larger_synthetic_tissue_and_config():
    test_module_output_dir = "temp_test_larger_growth_output"
    if os.path.exists(test_module_output_dir):
        shutil.rmtree(test_module_output_dir)
    os.makedirs(test_module_output_dir, exist_ok=True)

    # Config for a slightly larger run
    config = {
        "paths": {"output_dir": test_module_output_dir},
        "simulation": {
            "simulation_name": "gbo_larger_test", "random_seed": 42, "log_level": "INFO", # INFO to reduce log spam
            "units": {"length": "mm", "pressure": "Pa", "flow_rate": "mm^3/s"}
        },
        "tissue_properties": {"metabolic_rates": {"gm": 0.016, "wm": 0.005}, "permeability": {}},
        "vascular_properties": {
            "blood_viscosity": constants.DEFAULT_BLOOD_VISCOSITY, "murray_law_exponent": 3.0,
            "initial_terminal_flow": 1e-4, "min_radius": 0.005, "k_murray_scaling_factor": 0.5,
            "min_segment_length": 0.1, "max_segment_length": 1.0, # Smaller max seg length
        },
        "gbo_growth": {
            "max_iterations": 7, # More iterations
            "energy_coefficient_C_met_vessel_wall": 1.0e-5,
            "initial_territory_radius": 0.3, # Larger initial claim for larger domain
            "frontier_search_radius_factor": 3.0,
            "frontier_search_radius_fixed": 0.4, 
            "max_voxels_for_Rip": 50,
            "branch_radius_increase_threshold": 1.15,
            "max_flow_single_terminal": 0.001, 
            "target_domain_perfusion_fraction": 0.80, # Aim for 80%
            "bifurcation_candidate_points": 10,
            "min_iterations_before_no_growth_stop": 5,
            "stop_criteria": {"max_radius_factor_measured": 1.0}
        },
        "visualization": {"save_intermediate_steps": True, "intermediate_step_interval": 3} # Save every 3 iters
    }
    run_output_dir = utils.create_output_directory(test_module_output_dir, "run_larger_gbo", timestamp=False)

    # Tissue data
    shape = (40, 40, 40) # Larger grid
    affine = np.eye(4)
    affine[0,0] = affine[1,1] = affine[2,2] = 0.1 # 0.1mm voxels (4mm x 4mm x 4mm cube)
    
    domain_mask = np.zeros(shape, dtype=bool)
    domain_mask[5:35, 5:35, 5:35] = True # Central active tissue block
    
    gm_mask = np.zeros(shape, dtype=bool) # e.g., a shell or specific region
    gm_mask[5:15, 5:35, 5:35] = True 
    gm_mask = gm_mask & domain_mask

    metabolic_demand_map_qmet = np.zeros(shape, dtype=np.float32)
    metabolic_demand_map_qmet[gm_mask] = config["tissue_properties"]["metabolic_rates"]["gm"]
    # Fill rest of domain with WM rate
    wm_fill_mask = domain_mask & (~gm_mask)
    metabolic_demand_map_qmet[wm_fill_mask] = config["tissue_properties"]["metabolic_rates"]["wm"]
    
    voxel_volume = utils.get_voxel_volume_from_affine(affine)
    voxel_indices_3d_domain = np.array(np.where(domain_mask)).T
    world_coords_domain_flat = utils.voxel_to_world(voxel_indices_3d_domain, affine)

    tissue = {
        'shape': shape, 'affine': affine, 'voxel_volume': voxel_volume,
        'domain_mask': domain_mask, 'GM': gm_mask, 'WM': wm_fill_mask,
        'metabolic_demand_map': metabolic_demand_map_qmet, 
        'world_coords_flat': world_coords_domain_flat,
        'voxel_indices_flat': voxel_indices_3d_domain
    }

    # Initial "measured" artery - a simple line segment
    initial_artery_graph = data_structures.create_empty_vascular_graph()
    # Root point (e.g., edge of domain)
    # Voxel (0, 20, 20) -> world coords
    p_root_vox = np.array([0, shape[1]//2, shape[2]//2]) 
    p_root_world = utils.voxel_to_world(p_root_vox.reshape(1,-1), affine)[0]
    # Terminal point (e.g., just inside the domain_mask)
    p_term_vox = np.array([6, shape[1]//2, shape[2]//2]) # Inside domain_mask[5:35,...]
    p_term_world = utils.voxel_to_world(p_term_vox.reshape(1,-1), affine)[0]

    data_structures.add_node_to_graph(initial_artery_graph, "m_root_0", pos=p_root_world, radius=0.3, type='measured_root')
    data_structures.add_node_to_graph(initial_artery_graph, "m_term_0", pos=p_term_world, radius=0.25, type='measured_terminal')
    data_structures.add_edge_to_graph(initial_artery_graph, "m_root_0", "m_term_0", type='measured_segment')
    
    return config, run_output_dir, tissue, initial_artery_graph

def test_vascular_growth_larger_phantom_with_artery(larger_synthetic_tissue_and_config, caplog):
    config, output_dir, tissue_data, initial_graph = larger_synthetic_tissue_and_config
    
    caplog.set_level(logging.INFO) # Keep logs less verbose for this potentially longer test
    utils.set_rng_seed(config_manager.get_param(config, "simulation.random_seed", 42))

    logger.info(f"Starting larger phantom test. Output will be in: {output_dir}")
    logger.info(f"Initial graph has {initial_graph.number_of_nodes()} nodes, {initial_graph.number_of_edges()} edges.")
    logger.info(f"Tissue domain size: {tissue_data['shape']}, Voxel size: {np.diag(tissue_data['affine'])[:3]}mm")
    logger.info(f"Number of voxels in domain_mask: {np.sum(tissue_data['domain_mask'])}")


    final_graph = vascular_growth.grow_healthy_vasculature(
        config=config,
        tissue_data=tissue_data,
        initial_graph=initial_graph,
        output_dir=output_dir
    )

    assert final_graph is not None
    assert final_graph.number_of_nodes() > initial_graph.number_of_nodes(), "Graph did not grow from initial artery."

    # Check if some files were saved (at least the first intermediate or final)
    # This relies on the save logic in grow_healthy_vasculature correctly identifying when to save.
    # For this test, we are mostly interested in visual inspection of the output.
    # Let's just check if the output directory contains VTP files.
    vtp_files_found = [f for f in os.listdir(output_dir) if f.endswith(".vtp")]
    nii_files_found = [f for f in os.listdir(output_dir) if f.endswith(".nii.gz")]

    assert len(vtp_files_found) > 0, "No VTP graph files were saved."
    assert len(nii_files_found) > 0, "No NIfTI mask files were saved."
    
    logger.info(f"Larger phantom test completed. Final graph has {final_graph.number_of_nodes()} nodes, {final_graph.number_of_edges()} edges.")
    logger.info(f"Inspect outputs in: {output_dir}")
    
    # Call the main visualization function (which might produce a PyVista plot/screenshot)
    # Ensure visualization settings in config are appropriate (e.g., background color)
    if config_manager.get_param(config,"visualization.pyvista_enabled_for_test", True) and PYVISTA_AVAILABLE : # Add a flag to enable/disable for CI
        from src import visualization # Import here to use the updated one
        visualization.plot_vascular_tree_pyvista(
            final_graph, 
            title="Larger Phantom GBO Result",
            output_screenshot_path=os.path.join(output_dir, "larger_phantom_final_tree.png")
        )
    else:
        logger.info("Skipping PyVista plot in test_vascular_growth_larger_phantom_with_artery.")