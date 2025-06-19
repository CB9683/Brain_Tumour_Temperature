# src/angiogenesis.py
from __future__ import annotations # Must be first line

import numpy as np
import networkx as nx
import logging
import os
from scipy.spatial import KDTree
from scipy.ndimage import binary_erosion, binary_dilation, gaussian_filter
from typing import Tuple, List, Dict, Optional, Set, Callable

from src import utils, data_structures, constants, config_manager, io_utils
from src.vascular_growth import GBOIterationData

logger = logging.getLogger(__name__)

DEFAULT_RIM_THICKNESS_VOXELS = 3
DEFAULT_VEGF_PRODUCTION_RIM = 1.0
DEFAULT_MIN_TUMOR_TERMINAL_DEMAND = 1e-5

# --- Tumor Morphology and State Update Functions ---
# (initialize_active_tumor_from_seed, update_tumor_rim_and_core,
#  update_metabolic_demand_for_tumor, update_vegf_field_rim_driven,
#  grow_tumor_mass_within_defined_segmentation, coopt_and_modify_vessels,
#  find_angiogenic_sprouting_candidates - These functions remain the same as the previous complete version)
# For brevity, I will assume these functions are present as in the previous complete response.
# If you need them explicitly here, let me know. I'll paste them from the previous version.

# --- PASTE THE FOLLOWING FUNCTIONS FROM THE PREVIOUS RESPONSE HERE ---
# initialize_active_tumor_from_seed
# update_tumor_rim_and_core
# update_metabolic_demand_for_tumor
# update_vegf_field_rim_driven
# grow_tumor_mass_within_defined_segmentation
# coopt_and_modify_vessels
# find_angiogenic_sprouting_candidates
# --- END OF PASTED FUNCTIONS ---

# --- Angiogenesis Core Functions (Sprouting, Growth, Co-option) ---
# (Functions like find_angiogenic_sprouting_candidates, coopt_and_modify_vessels are assumed to be here
#  from the previous "complete" version provided)

# --- PASTE find_angiogenic_sprouting_candidates and coopt_and_modify_vessels here from previous response ---
def initialize_active_tumor_from_seed(tissue_data: dict, config: dict) -> bool:
    """
    Initializes a small 'active' tumor (tissue_data['Tumor'])
    within the bounds of a pre-defined 'Tumor_Max_Extent'.
    If specific seed coordinates are not provided in config, it attempts to find a seed automatically.
    """
    tumor_max_extent_array = tissue_data.get('Tumor_Max_Extent')
    if tumor_max_extent_array is None or not np.any(tumor_max_extent_array):
        logger.error("Cannot initialize seed: 'Tumor_Max_Extent' not found, is None, or is empty in tissue_data.")
        if 'shape' in tissue_data and tissue_data['shape'] is not None:
            tissue_data['Tumor'] = np.zeros(tissue_data['shape'], dtype=bool)
        return False

    tumor_seed_config = config_manager.get_param(config, "tumor_angiogenesis.initial_tumor_seed", {})
    seed_strategy = tumor_seed_config.get("strategy", "auto_center") # New: "auto_center", "auto_random", "manual"
    radius_vox = tumor_seed_config.get('radius_voxels', 3)
    shape = tissue_data['shape']
    center_vox_ijk = None

    if seed_strategy == "manual":
        center_vox_ijk_manual = tumor_seed_config.get('center_voxel_ijk_relative_to_image_grid')
        if not center_vox_ijk_manual or not isinstance(center_vox_ijk_manual, list) or len(center_vox_ijk_manual) != 3:
            logger.error("Tumor seed strategy is 'manual' but 'center_voxel_ijk_relative_to_image_grid' not properly defined. Aborting seed.")
            tissue_data['Tumor'] = np.zeros(shape, dtype=bool)
            return False
        center_vox_ijk = np.array(center_vox_ijk_manual)
        if not utils.is_voxel_in_bounds(center_vox_ijk, shape):
             logger.error(f"Manual tumor seed center {center_vox_ijk} is outside image dimensions {shape}.")
             tissue_data['Tumor'] = np.zeros(shape, dtype=bool)
             return False
    elif seed_strategy == "auto_center":
        from scipy.ndimage import center_of_mass
        if np.any(tumor_max_extent_array):
            # Calculate the center of mass of the Tumor_Max_Extent mask
            # Ensure it's integer coordinates and within bounds
            com_float = center_of_mass(tumor_max_extent_array)
            center_vox_ijk = np.round(com_float).astype(int)
            # Clip to ensure it's within array bounds strictly (for safety with radius)
            for d in range(3):
                center_vox_ijk[d] = np.clip(center_vox_ijk[d], radius_vox, shape[d] - 1 - radius_vox)
            logger.info(f"Automatic seed strategy 'auto_center': calculated COM {com_float}, using seed center {center_vox_ijk}.")
        else: # Should have been caught by the first check, but for safety
            logger.error("'auto_center' seed strategy failed: Tumor_Max_Extent is empty.")
            tissue_data['Tumor'] = np.zeros(shape, dtype=bool)
            return False
    elif seed_strategy == "auto_random":
        true_indices = np.array(np.where(tumor_max_extent_array)).T
        if true_indices.shape[0] > 0:
            random_idx = np.random.choice(true_indices.shape[0])
            center_vox_ijk = true_indices[random_idx]
            logger.info(f"Automatic seed strategy 'auto_random': selected random seed center {center_vox_ijk}.")
        else: # Should have been caught, but for safety
            logger.error("'auto_random' seed strategy failed: Tumor_Max_Extent is empty.")
            tissue_data['Tumor'] = np.zeros(shape, dtype=bool)
            return False
    else:
        logger.error(f"Unknown tumor seed strategy: {seed_strategy}. Choose 'manual', 'auto_center', or 'auto_random'.")
        tissue_data['Tumor'] = np.zeros(shape, dtype=bool)
        return False

    # Create a spherical seed mask at the determined center
    coords = np.ogrid[:shape[0], :shape[1], :shape[2]]
    distance_sq = ((coords[0] - center_vox_ijk[0])**2 +
                   (coords[1] - center_vox_ijk[1])**2 +
                   (coords[2] - center_vox_ijk[2])**2)
    initial_seed_mask = distance_sq <= radius_vox**2

    # Active tumor is the seed AND where it overlaps with the max extent
    tissue_data['Tumor'] = initial_seed_mask & tumor_max_extent_array

    if not np.any(tissue_data['Tumor']):
        logger.warning(f"Initial tumor seed (strategy: {seed_strategy}) at {center_vox_ijk} with radius {radius_vox} "
                       f"resulted in an empty active tumor region within Tumor_Max_Extent. "
                       f"This might happen if Tumor_Max_Extent is very thin or small at the chosen seed location, "
                       f"or if the seed radius is too small to capture any 'True' voxels of Tumor_Max_Extent.")
        # tissue_data['Tumor'] is already an all-False array of the right shape here
        return False

    logger.info(f"Initialized active tumor seed (strategy: {seed_strategy}) within Tumor_Max_Extent: "
                f"center_vox={center_vox_ijk}, radius_vox={radius_vox}, "
                f"num_active_tumor_voxels={np.sum(tissue_data['Tumor'])}")
    return True

def update_tumor_rim_and_core(tissue_data: dict, config: dict):
    """Identifies tumor rim and core based on current active tissue_data['Tumor'] mask."""
    active_tumor_mask_local = tissue_data.get('Tumor') # Get the array first
    if active_tumor_mask_local is None or not np.any(active_tumor_mask_local): # Check if None or all False
        # Ensure these keys exist even if tumor is empty, to prevent KeyErrors later
        if 'shape' in tissue_data and tissue_data['shape'] is not None:
            tissue_data['tumor_rim_mask'] = np.zeros(tissue_data['shape'], dtype=bool)
            tissue_data['tumor_core_mask'] = np.zeros(tissue_data['shape'], dtype=bool)
        else:
            logger.error("Cannot initialize empty rim/core masks as tissue_data['shape'] is missing.")
        return

    # Now we know active_tumor_mask_local is a valid NumPy array with at least one True value
    active_tumor_mask_bool = active_tumor_mask_local.astype(bool) # Ensure boolean for morphology operations
    params = config_manager.get_param(config, "tumor_angiogenesis.tumor_morphology", {})
    rim_thickness = params.get("rim_thickness_voxels", DEFAULT_RIM_THICKNESS_VOXELS)

    if rim_thickness <= 0: # No rim, all core (or all rim if tumor is very small and erosion makes it disappear)
        # If no rim thickness, the whole active tumor is considered rim for VEGF production,
        # and there's no distinct core by erosion.
        # Or, another interpretation: all is core. Let's assume all is rim for VEGF.
        eroded_core = np.zeros_like(active_tumor_mask_bool)
        # If tumor is smaller than rim_thickness, erosion might make it disappear.
        # In such a case, consider the whole tumor as rim.
        if np.sum(active_tumor_mask_bool) > 0 and np.sum(binary_erosion(active_tumor_mask_bool, iterations=rim_thickness, border_value=0)) == 0:
             # If erosion results in nothing, but tumor exists, the whole thing is effectively rim
             tissue_data['tumor_rim_mask'] = active_tumor_mask_bool.copy()
             tissue_data['tumor_core_mask'] = np.zeros_like(active_tumor_mask_bool)
        else:
             eroded_core = binary_erosion(active_tumor_mask_bool, iterations=rim_thickness, border_value=0)
             tissue_data['tumor_core_mask'] = eroded_core
             tissue_data['tumor_rim_mask'] = active_tumor_mask_bool & (~eroded_core)

    else: # rim_thickness > 0
        eroded_core = binary_erosion(active_tumor_mask_bool, iterations=rim_thickness, border_value=0)
        tissue_data['tumor_core_mask'] = eroded_core
        tissue_data['tumor_rim_mask'] = active_tumor_mask_bool & (~eroded_core)
    
    logger.debug(f"Updated tumor rim ({np.sum(tissue_data.get('tumor_rim_mask', np.array([])))} vox) and core ({np.sum(tissue_data.get('tumor_core_mask', np.array([])))} vox).")

def update_metabolic_demand_for_tumor(tissue_data: dict, config: dict):
    """Updates metabolic_demand_map based on tumor rim and core."""
    active_tumor_mask_local = tissue_data.get('Tumor') # Get the array first
    if active_tumor_mask_local is None or not np.any(active_tumor_mask_local): # Check if None or all False
        # No active tumor, so no specific tumor metabolic demand to set.
        # The metabolic_demand_map should reflect healthy tissue or be zero in these areas.
        logger.debug("No active tumor present; skipping tumor-specific metabolic demand update.")
        return

    # Ensure metabolic_demand_map exists and has the correct shape
    if 'metabolic_demand_map' not in tissue_data or \
       tissue_data.get('metabolic_demand_map') is None or \
       tissue_data['metabolic_demand_map'].shape != tissue_data['shape']:
        # If it's missing or wrong shape, it should have been initialized correctly in load_initial_data
        # or after the first tumor growth step. For safety, ensure it's there.
        if 'shape' in tissue_data and tissue_data['shape'] is not None:
            tissue_data['metabolic_demand_map'] = np.zeros(tissue_data['shape'], dtype=float)
            logger.warning("Re-initialized metabolic_demand_map during tumor demand update due to mismatch or absence.")
        else:
            logger.error("Cannot update metabolic demand for tumor: tissue_data['shape'] is missing.")
            return


    rates = config_manager.get_param(config, "tissue_properties.metabolic_rates", {})
    q_met_rim = rates.get("tumor_rim", constants.Q_MET_TUMOR_RIM_PER_ML)
    q_met_core = rates.get("tumor_core", constants.Q_MET_TUMOR_CORE_PER_ML)
    voxel_vol = tissue_data['voxel_volume']

    # metabolic_demand_map should already exist and be initialized (e.g. with healthy demands)
    # We are now OVERWRITING the demand in tumor regions.

    if tissue_data.get('tumor_rim_mask') is not None and np.any(tissue_data['tumor_rim_mask']):
        tissue_data['metabolic_demand_map'][tissue_data['tumor_rim_mask']] = q_met_rim * voxel_vol
    
    if tissue_data.get('tumor_core_mask') is not None and np.any(tissue_data['tumor_core_mask']):
        tissue_data['metabolic_demand_map'][tissue_data['tumor_core_mask']] = q_met_core * voxel_vol
    
    # What about active tumor voxels that are neither rim nor core (e.g., if rim_thickness is 0 or tumor is tiny)?
    # The current update_tumor_rim_and_core logic tries to ensure rim+core covers the active tumor.
    # If there are any 'Tumor' voxels not covered by rim or core (shouldn't happen with current logic),
    # their metabolic rate would remain unchanged (i.e., healthy rate).
    # This is generally fine, as rim/core should define the tumor's metabolic activity.

    logger.debug("Updated metabolic demand map for active tumor regions (rim/core).")

def update_vegf_field_rim_driven(tissue_data: dict, config: dict) -> bool:
    """VEGF produced primarily by the tumor rim. Updates tissue_data['VEGF_field']."""
    tumor_rim_mask_local = tissue_data.get('tumor_rim_mask') # Get the array first
    if tumor_rim_mask_local is None or not np.any(tumor_rim_mask_local): # Check if None or all False
        logger.debug("No tumor rim to produce VEGF. Setting VEGF field to zero.")
        # Ensure VEGF_field exists even if empty
        if 'shape' in tissue_data and tissue_data['shape'] is not None:
            tissue_data['VEGF_field'] = np.zeros(tissue_data['shape'], dtype=float)
        else:
            logger.error("Cannot initialize empty 'VEGF_field' as tissue_data['shape'] is missing.")
        return True # Or False if this state is considered an error for VEGF generation

    # Now we know tumor_rim_mask_local is a valid NumPy array with at least one True value
    vegf_config = config_manager.get_param(config, "tumor_angiogenesis.vegf_settings", {})
    vegf_prod_rim = vegf_config.get("production_rate_rim", DEFAULT_VEGF_PRODUCTION_RIM)
    
    vegf_field = np.zeros(tissue_data['shape'], dtype=float) # Initialize with correct shape
    vegf_field[tumor_rim_mask_local] = vegf_prod_rim # Use the fetched local variable
    
    # Optional: Core contribution (if you add this logic back)
    # vegf_prod_core = vegf_config.get("production_rate_core", DEFAULT_VEGF_PRODUCTION_CORE)
    # tumor_core_mask_local = tissue_data.get('tumor_core_mask')
    # if tumor_core_mask_local is not None and np.any(tumor_core_mask_local):
    #     vegf_field[tumor_core_mask_local] += vegf_prod_core # Additive or max?

    if vegf_config.get("apply_diffusion_blur", True):
        sigma = vegf_config.get("diffusion_blur_sigma", 2.5)
        if sigma > 0 and np.any(vegf_field): # Only blur if there's something to blur
            try:
                vegf_field = gaussian_filter(vegf_field, sigma=sigma)
                logger.debug(f"Applied Gaussian blur (sigma={sigma}) to VEGF field.")
            except Exception as e: # Catch potential errors from gaussian_filter if field is weird
                logger.warning(f"Could not apply Gaussian blur to VEGF field: {e}")
    
    tissue_data['VEGF_field'] = vegf_field
    logger.info(f"Updated rim-driven VEGF field. Max VEGF: {np.max(vegf_field) if np.any(vegf_field) else 0:.2e}")
    return True

def grow_tumor_mass_within_defined_segmentation(tissue_data: dict, config: dict) -> bool:
    tumor_growth_params = config_manager.get_param(config, "tumor_angiogenesis.tumor_growth", {})
    expansion_voxels_per_step = tumor_growth_params.get("expansion_voxels_per_step", 100)
    
    # Get masks
    current_active_tumor_mask = tissue_data.get('Tumor')
    tumor_max_extent_mask = tissue_data.get('Tumor_Max_Extent')
    active_tumor_rim_mask = tissue_data.get('tumor_rim_mask') # This is calculated based on 'Tumor'

    # Check for None before np.any or other operations
    if current_active_tumor_mask is None or \
       tumor_max_extent_mask is None or \
       active_tumor_rim_mask is None: # active_tumor_rim_mask could be None if 'Tumor' was None when it was calculated
        logger.error("grow_tumor_mass: One or more required masks (Tumor, Tumor_Max_Extent, tumor_rim_mask) is None.")
        return False

    if np.all(current_active_tumor_mask == tumor_max_extent_mask): # This is fine, compares two arrays
        logger.info("Tumor Growth: Active tumor has filled 'Tumor_Max_Extent'.")
        return False
    
    # Determine source for dilation based on rim or whole active tumor
    source_for_dilation = active_tumor_rim_mask if np.any(active_tumor_rim_mask) else current_active_tumor_mask
    if not np.any(source_for_dilation): # Correctly checks if the chosen source is empty
        logger.info("Tumor Growth: Source for dilation (rim or active tumor) is empty. Cannot grow.")
        return False # Added this return

    # Ensure domain_mask is also valid before using it in the bitwise AND
    domain_mask_for_growth = tissue_data.get('domain_mask')
    if domain_mask_for_growth is None:
        logger.error("grow_tumor_mass: 'domain_mask' is None. Cannot determine growth candidates.")
        return False

    growth_candidates_mask = binary_dilation(source_for_dilation) & \
                             (~current_active_tumor_mask) & \
                             tumor_max_extent_mask & \
                             domain_mask_for_growth # Use the fetched domain_mask

    candidate_voxel_indices = np.array(np.where(growth_candidates_mask)).T
    if candidate_voxel_indices.shape[0] == 0:
        logger.info("Tumor Growth: No suitable healthy voxels for expansion within constraints.")
        return False
        
    num_to_convert = min(expansion_voxels_per_step, candidate_voxel_indices.shape[0])
    if num_to_convert > 0:
        chosen_indices_idx = np.random.choice(candidate_voxel_indices.shape[0], num_to_convert, replace=False)
        voxels_to_add = candidate_voxel_indices[chosen_indices_idx]
        
        new_active_tumor_mask = current_active_tumor_mask.copy()
        gm_mask = tissue_data.get('GM') # Fetch GM and WM once
        wm_mask = tissue_data.get('WM')

        for vox_idx_tuple in map(tuple, voxels_to_add):
            new_active_tumor_mask[vox_idx_tuple] = True
            if gm_mask is not None and utils.is_voxel_in_bounds(vox_idx_tuple, gm_mask.shape) and gm_mask[vox_idx_tuple]: # Check bounds for safety
                gm_mask[vox_idx_tuple] = False
            if wm_mask is not None and utils.is_voxel_in_bounds(vox_idx_tuple, wm_mask.shape) and wm_mask[vox_idx_tuple]: # Check bounds for safety
                wm_mask[vox_idx_tuple] = False
        
        tissue_data['Tumor'] = new_active_tumor_mask
        # Update GM and WM in tissue_data if they were modified
        if gm_mask is not None: tissue_data['GM'] = gm_mask
        if wm_mask is not None: tissue_data['WM'] = wm_mask

        logger.info(f"Tumor Growth: Expanded by {num_to_convert} vox. Active: {np.sum(new_active_tumor_mask)}, Max: {np.sum(tumor_max_extent_mask)}")
        return True
    return False

def coopt_and_modify_vessels(graph: nx.DiGraph, tissue_data: dict, config: dict):
    active_tumor_mask_local = tissue_data.get('Tumor') # Get the array
    if active_tumor_mask_local is None or not np.any(active_tumor_mask_local): # Corrected check
        logger.debug("Co-option: No active tumor to co-opt vessels from.")
        return

    affine = tissue_data.get('affine')
    if affine is None:
        logger.error("Co-option: Affine matrix not found in tissue_data. Cannot perform co-option.")
        return

    cooption_params = config_manager.get_param(config, "tumor_angiogenesis.cooption", {})
    radius_dilation_factor_mean = cooption_params.get("radius_dilation_factor_mean", 1.1)
    radius_dilation_factor_std = cooption_params.get("radius_dilation_factor_std", 0.05)
    permeability_Lp_tumor_factor = cooption_params.get("permeability_Lp_factor_tumor", 10.0)
    
    nodes_coopted_this_step: Set[str] = set()
    for node_id, data in graph.nodes(data=True):
        if data.get('is_tumor_vessel', False): continue
        
        node_pos = data.get('pos')
        if node_pos is None: continue # Skip nodes without position

        pos_vox_int = np.round(utils.world_to_voxel(node_pos, affine)).astype(int)
        if utils.is_voxel_in_bounds(pos_vox_int, active_tumor_mask_local.shape) and \
           active_tumor_mask_local[tuple(pos_vox_int)]:
            nodes_coopted_this_step.add(node_id)
            
    if not nodes_coopted_this_step:
        logger.debug("Co-option: No new healthy vessels found within active tumor for co-option this step.")
        return

    for node_id in nodes_coopted_this_step:
        node_data = graph.nodes[node_id] # Get node data once
        node_data['is_tumor_vessel'] = True
        node_data['vessel_origin_type'] = 'coopted_healthy'
        
        original_radius = node_data.get('radius', constants.MIN_VESSEL_RADIUS_MM) # Use default if radius missing
        dilation = max(0.5, np.random.normal(radius_dilation_factor_mean, radius_dilation_factor_std))
        new_radius = max(constants.MIN_VESSEL_RADIUS_MM, original_radius * dilation)
        node_data['radius'] = new_radius
        # logger.debug(f"Co-opted node {node_id}. Type: {node_data.get('type')}->coopted_healthy. R: {original_radius:.4f}->{new_radius:.4f}") # Already logged in main loop

        # Mark connected edges and update their radii if they originate from this node
        for u, v, edge_data in graph.out_edges(node_id, data=True): # Edges where this node is the source
            edge_data['is_tumor_vessel'] = True
            edge_data['radius'] = new_radius # Edge takes radius of its (now co-opted) source node
            edge_data['permeability_Lp_factor'] = permeability_Lp_tumor_factor
        for u, v, edge_data in graph.in_edges(node_id, data=True): # Edges where this node is the target
            edge_data['is_tumor_vessel'] = True # The segment is now within tumor influence
            edge_data['permeability_Lp_factor'] = permeability_Lp_tumor_factor
            # Radius of incoming edge is determined by its source node (u), which might also get co-opted

    logger.info(f"Co-opted and modified {len(nodes_coopted_this_step)} nodes and their adjacent edges.")

def find_angiogenic_sprouting_candidates(graph: nx.DiGraph, tissue_data: dict, config: dict) -> List[Tuple[str, str, np.ndarray, np.ndarray]]:
    candidates = []
    vegf_field_local = tissue_data.get('VEGF_field') # Get the array
    if vegf_field_local is None or not np.any(vegf_field_local): # Corrected check
        logger.debug("Sprouting candidates: No VEGF field or VEGF field is all zero.")
        return candidates

    affine = tissue_data.get('affine')
    if affine is None:
        logger.error("Sprouting candidates: Affine matrix not found in tissue_data.")
        return candidates

    sprouting_params = config_manager.get_param(config, "tumor_angiogenesis.sprouting", {})
    min_vegf = sprouting_params.get("min_vegf_concentration", 0.2)
    min_parent_r = sprouting_params.get("min_parent_vessel_radius_mm", 0.02)
    
    grad_ax0, grad_ax1, grad_ax2 = np.gradient(vegf_field_local) # Use local copy

    for u, v, edge_data in graph.edges(data=True):
        node_u_data = graph.nodes[u]
        node_v_data = graph.nodes[v] # Get v data too for position

        parent_radius = node_u_data.get('radius', 0) # Use default 0 if radius missing
        if parent_radius < min_parent_r: 
            continue
        
        pos_u = node_u_data.get('pos')
        pos_v = node_v_data.get('pos')
        if pos_u is None or pos_v is None: continue # Skip if positions are missing

        sprout_origin = (pos_u + pos_v) / 2.0
        sprout_vox_int = np.round(utils.world_to_voxel(sprout_origin, affine)).astype(int)

        if not utils.is_voxel_in_bounds(sprout_vox_int, vegf_field_local.shape): 
            continue
            
        if vegf_field_local[tuple(sprout_vox_int)] >= min_vegf:
            g_ax0 = grad_ax0[tuple(sprout_vox_int)]
            g_ax1 = grad_ax1[tuple(sprout_vox_int)]
            g_ax2 = grad_ax2[tuple(sprout_vox_int)]
            sprout_dir_vox = np.array([g_ax0, g_ax1, g_ax2])
            
            # Transform gradient to world space direction
            # affine[:3,:3] is the rotation/scaling part
            sprout_dir_world = utils.normalize_vector(affine[:3,:3] @ sprout_dir_vox) 

            if np.linalg.norm(sprout_dir_world) > constants.EPSILON:
                candidates.append((u, v, sprout_origin, sprout_dir_world))
    
    max_sprouts = sprouting_params.get("max_new_sprouts_per_iteration", 5)
    if len(candidates) > max_sprouts:
        indices = np.random.choice(len(candidates), max_sprouts, replace=False)
        selected_candidates = [candidates[i] for i in indices]
        logger.info(f"Selected {len(selected_candidates)} sprouts from {len(candidates)} original candidates.")
        return selected_candidates
    elif candidates: 
        logger.info(f"Found {len(candidates)} sprouting candidates.")
    else:
        logger.debug("No sprouting candidates found this step.")
    return candidates


def attempt_anastomosis_tip_to_segment(
    term_gbo: GBOIterationData,
    graph: nx.DiGraph, # The main angiogenic graph
    vessel_kdtree: KDTree, # KDTree of (positions of midpoints of all non-parent segments)
    segment_midpoints_data: List[Dict], # List of {'pos': mid_pos, 'u': seg_u, 'v': seg_v, 'radius': seg_radius}
    config: dict,
    next_synthetic_node_id_ref: List[int] # Pass as list to modify in place
) -> bool:
    """
    Attempts to anastomose the given angiogenic terminal (term_gbo) to a nearby existing segment.
    Modifies graph and term_gbo if successful.
    Returns True if anastomosis occurred, False otherwise.
    """
    anastomosis_params = config_manager.get_param(config, "tumor_angiogenesis.anastomosis", {})
    search_radius = term_gbo.radius * anastomosis_params.get("search_radius_factor", 3.0)
    min_angle_deg = anastomosis_params.get("min_fusion_angle_deg", 120.0) # Angle between tip's last segment and segment to target
    max_dist_to_midpoint_factor = anastomosis_params.get("max_dist_to_midpoint_factor", 1.5) # Tip must be close to midpoint

    if vessel_kdtree is None or not segment_midpoints_data: return False

    nearby_indices = vessel_kdtree.query_ball_point(term_gbo.pos, r=search_radius)
    if not nearby_indices: return False

    parent_of_tip_pos = graph.nodes[term_gbo.parent_id]['pos']
    tip_growth_vector = term_gbo.pos - parent_of_tip_pos # Vector of the last segment of the tip

    best_target_seg_info = None
    min_dist_sq = float('inf')

    for idx in nearby_indices:
        target_seg = segment_midpoints_data[idx]
        target_midpoint = target_seg['pos']
        target_u, target_v = target_seg['u'], target_seg['v']

        # Avoid self-anastomosis or anastomosis with immediate parent segment from bifurcation
        if term_gbo.parent_id == target_u or term_gbo.parent_id == target_v: continue
        # Avoid if target is the segment the tip just grew from (if parent_id was a midpoint)
        # This check needs to be more robust if parent_id can be a segment point.
        # For now, assume parent_id is a node.

        dist_sq = utils.distance_squared(term_gbo.pos, target_midpoint)
        if dist_sq < min_dist_sq and dist_sq < (term_gbo.radius * max_dist_to_midpoint_factor)**2 :
            # Check angle: vector from tip to target_midpoint vs. tip_growth_vector
            vec_tip_to_target = target_midpoint - term_gbo.pos
            if np.linalg.norm(tip_growth_vector) > constants.EPSILON and np.linalg.norm(vec_tip_to_target) > constants.EPSILON:
                cos_angle = np.dot(tip_growth_vector, vec_tip_to_target) / \
                            (np.linalg.norm(tip_growth_vector) * np.linalg.norm(vec_tip_to_target))
                angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                if angle_deg >= min_angle_deg: # Tip should be "aiming away" or sideways relative to target for good fusion
                    min_dist_sq = dist_sq
                    best_target_seg_info = target_seg
    
    if best_target_seg_info:
        target_u = best_target_seg_info['u']
        target_v = best_target_seg_info['v']
        target_midpoint_pos = best_target_seg_info['pos'] # This will be the new anastomosis node

        logger.info(f"Anastomosis: Tip {term_gbo.id} (parent {term_gbo.parent_id}) fusing with segment {target_u}-{target_v} at {np.round(target_midpoint_pos,2)}.")

        # 1. Create new anastomosis node at target_midpoint_pos
        anastomosis_node_id = f"s_{next_synthetic_node_id_ref[0]}"; next_synthetic_node_id_ref[0] += 1
        # Radius of anastomosis node can be average or based on fusing vessels
        fused_radius = (term_gbo.radius + best_target_seg_info['radius']) / 2.0
        data_structures.add_node_to_graph(graph, anastomosis_node_id, pos=target_midpoint_pos, radius=fused_radius,
                                          type='anastomosis_point', is_tumor_vessel=True, vessel_origin_type='anastomosis')

        # 2. Remove old target segment (target_u, target_v)
        original_target_edge_data = graph.edges[target_u, target_v].copy() # Assuming directed u->v
        graph.remove_edge(target_u, target_v)

        # 3. Add new segments: target_u -> anastomosis_node, anastomosis_node -> target_v
        data_structures.add_edge_to_graph(graph, target_u, anastomosis_node_id, **original_target_edge_data) # Update length/radius
        graph.edges[target_u, anastomosis_node_id]['length'] = utils.distance(graph.nodes[target_u]['pos'], target_midpoint_pos)
        graph.edges[target_u, anastomosis_node_id]['radius'] = graph.nodes[target_u]['radius'] # Takes radius of upstream node

        data_structures.add_edge_to_graph(graph, anastomosis_node_id, target_v, **original_target_edge_data) # Update length/radius
        graph.edges[anastomosis_node_id, target_v]['length'] = utils.distance(target_midpoint_pos, graph.nodes[target_v]['pos'])
        graph.edges[anastomosis_node_id, target_v]['radius'] = fused_radius # Takes radius of new anastomosis node

        # 4. Connect the parent of the fusing tip to the anastomosis_node
        # The segment was (term_gbo.parent_id) -> term_gbo.id (which is at term_gbo.pos)
        # New segment is (term_gbo.parent_id) -> anastomosis_node_id
        tip_parent_node_id = term_gbo.parent_id
        data_structures.add_edge_to_graph(graph, tip_parent_node_id, anastomosis_node_id,
                                          length=utils.distance(graph.nodes[tip_parent_node_id]['pos'], target_midpoint_pos),
                                          radius=graph.nodes[tip_parent_node_id]['radius'], # Or fused_radius?
                                          type='angiogenic_segment', is_tumor_vessel=True,
                                          permeability_Lp_factor=graph.edges[tip_parent_node_id, term_gbo.id].get('permeability_Lp_factor'))


        # 5. Remove the old tip node (term_gbo.id) and its incoming segment
        graph.remove_edge(tip_parent_node_id, term_gbo.id)
        graph.remove_node(term_gbo.id)
        
        term_gbo.stop_growth = True # Mark this GBOIterationData as done
        return True
    return False


# --- Main Angiogenesis Orchestration ---
def simulate_tumor_angiogenesis_fixed_extent(
    config: dict,
    tissue_data: dict,
    base_vascular_tree: nx.DiGraph,
    output_dir: str,
    perfusion_solver_func: Callable[[nx.DiGraph, dict, Optional[float], Optional[Dict[str, float]]], Optional[nx.DiGraph]]
) -> nx.DiGraph:
    logger.info("--- Starting Tumor Angiogenesis Simulation (Growing within Fixed Extent) ---")
    
    main_params = config_manager.get_param(config, "tumor_angiogenesis", {})
    num_macro_iterations = main_params.get("num_macro_iterations", 20)
    tumor_growth_steps_per_macro = main_params.get("tumor_growth.steps_per_macro_iter", 1)
    angiogenesis_steps_per_macro = main_params.get("angiogenesis.steps_per_macro_iter", 3)
    flow_solve_interval_macro = main_params.get("flow_solve_interval_macro_iters", 5)
    save_interval_macro = main_params.get("save_intermediate_interval_macro_iters", 1)

    sprouting_params = main_params.get("sprouting", {})
    initial_sprout_radius = sprouting_params.get("initial_sprout_radius_mm", constants.MIN_VESSEL_RADIUS_MM * 1.1)
    initial_sprout_length = sprouting_params.get("initial_sprout_length_mm", initial_sprout_radius * 4)
    
    extension_params = main_params.get("extension", {})
    extension_step_length = extension_params.get("step_length_mm", initial_sprout_radius * 2)
    
    branching_params = main_params.get("angiogenic_branching", {})
    branch_probability_factor_vegf = branching_params.get("branch_probability_factor_vegf", 0.1) # Prob = factor * vegf_norm
    branch_angle_spread_deg = branching_params.get("branch_angle_spread_deg", 60.0)


    if not initialize_active_tumor_from_seed(tissue_data, config):
        logger.error("Failed to initialize active tumor seed. Aborting angiogenesis.")
        return base_vascular_tree.copy()

    angiogenic_graph = base_vascular_tree.copy()
    # Pass next_synthetic_node_id as a list so its modification is seen by caller
    max_id_num = 0
    for node_id_str_val in angiogenic_graph.nodes():
        if isinstance(node_id_str_val, str) and node_id_str_val.startswith('s_'):
            try: max_id_num = max(max_id_num, int(node_id_str_val.split('_')[1]))
            except (ValueError, IndexError): pass
    next_synthetic_node_id_ref = [max_id_num + 10000] # List to pass by reference
    
    active_angiogenic_terminals: List[GBOIterationData] = []

    for macro_iter in range(num_macro_iterations):
        logger.info(f"===== Macro Iteration {macro_iter + 1} / {num_macro_iterations} =====")

        any_tumor_growth_this_macro = False
        for _ in range(tumor_growth_steps_per_macro):
            update_tumor_rim_and_core(tissue_data, config)
            if grow_tumor_mass_within_defined_segmentation(tissue_data, config): any_tumor_growth_this_macro = True
            else: break 
        
        update_tumor_rim_and_core(tissue_data, config); update_metabolic_demand_for_tumor(tissue_data, config)
        update_vegf_field_rim_driven(tissue_data, config)
        coopt_and_modify_vessels(angiogenic_graph, tissue_data, config)

        for ag_step in range(angiogenesis_steps_per_macro):
            logger.debug(f"  Angiogenesis Step {ag_step + 1} (Active Tips Before Sprouting: {len(active_angiogenic_terminals)})")
            
            # --- 2a. Sprouting ---
            new_sprouts_info = find_angiogenic_sprouting_candidates(angiogenic_graph, tissue_data, config)
            if new_sprouts_info:
                logger.debug(f"    Found {len(new_sprouts_info)} new sprout candidates this AG step.")

            for parent_u, parent_v, sprout_origin, sprout_dir in new_sprouts_info:
                bif_node_id = f"s_{next_synthetic_node_id_ref[0]}"; next_synthetic_node_id_ref[0] += 1
                
                parent_u_data = angiogenic_graph.nodes[parent_u]
                parent_v_data = angiogenic_graph.nodes[parent_v] # Get parent_v data as well
                parent_u_radius = parent_u_data['radius']
                
                # Determine vessel_origin_type for the new bifurcation node
                # If parent_u was already a tumor vessel (coopted or angiogenic), the bif is too.
                # Otherwise, it's a bifurcation on a healthy vessel that's now leading to tumor growth.
                bif_origin_type = parent_u_data.get('vessel_origin_type', 'healthy_parent_of_sprout')
                if parent_u_data.get('is_tumor_vessel'):
                    bif_origin_type = parent_u_data.get('vessel_origin_type', 'coopted_healthy') # Default if type was missing

                data_structures.add_node_to_graph(
                    angiogenic_graph, bif_node_id,
                    pos=sprout_origin,
                    radius=parent_u_radius, # Bifurcation point takes radius of the parent segment it's on
                    type='angiogenic_bifurcation',
                    is_tumor_vessel=True, # The bifurcation itself is part of the tumor response
                    vessel_origin_type=bif_origin_type
                )
                
                # Get original edge data before removing
                # Important: Check if edge still exists, could have been modified by another sprout from same segment in same AG step (unlikely but good check)
                if not angiogenic_graph.has_edge(parent_u, parent_v):
                    logger.warning(f"Sprouting target edge {parent_u}-{parent_v} no longer exists. Skipping this sprout.")
                    # Rollback bif_node_id? Or just let it be an isolated node that might get pruned.
                    # For now, continue, but this indicates a potential complex interaction.
                    # To properly handle, would need to process sprouts sequentially and update graph immediately.
                    # Current find_angiogenic_sprouting_candidates finds all then processes.
                    if angiogenic_graph.has_node(bif_node_id): angiogenic_graph.remove_node(bif_node_id) # Clean up unused bif
                    next_synthetic_node_id_ref[0] -=1 # Decrement counter
                    continue

                edge_data_orig = angiogenic_graph.edges[parent_u, parent_v].copy()
                angiogenic_graph.remove_edge(parent_u, parent_v)

                # Add new segments: parent_u -> bif_node_id and bif_node_id -> parent_v
                # These new segments inherit properties and get updated lengths/radii
                data_structures.add_edge_to_graph(angiogenic_graph, parent_u, bif_node_id, **edge_data_orig)
                angiogenic_graph.edges[parent_u, bif_node_id]['length'] = utils.distance(parent_u_data['pos'], sprout_origin)
                angiogenic_graph.edges[parent_u, bif_node_id]['radius'] = parent_u_radius # Takes radius of parent_u

                data_structures.add_edge_to_graph(angiogenic_graph, bif_node_id, parent_v, **edge_data_orig)
                angiogenic_graph.edges[bif_node_id, parent_v]['length'] = utils.distance(sprout_origin, parent_v_data['pos'])
                angiogenic_graph.edges[bif_node_id, parent_v]['radius'] = parent_u_radius # New segment from bif also takes bif radius

                # Mark these split segments as tumor vessels if the original parent was, or if the bif is
                # The bifurcation node itself is marked is_tumor_vessel=True
                # Segments connected to it that are part of the original path should also be marked.
                perm_factor_to_set = default_permeability_factor # Default for new tumor-related segments
                if parent_u_data.get('is_tumor_vessel'):
                    # If parent_u was already a tumor vessel, its perm factor might be already set
                    perm_factor_to_set = edge_data_orig.get('permeability_Lp_factor', default_permeability_factor)

                for e_start, e_end in [(parent_u, bif_node_id), (bif_node_id, parent_v)]:
                    angiogenic_graph.edges[e_start, e_end]['is_tumor_vessel'] = True # Part of the angiogenic event path
                    angiogenic_graph.edges[e_start, e_end]['permeability_Lp_factor'] = perm_factor_to_set


                # Create the new angiogenic sprout (terminal node and its GBOIterationData)
                sprout_tip_id = f"s_{next_synthetic_node_id_ref[0]}"; next_synthetic_node_id_ref[0] += 1
                sprout_tip_pos = sprout_origin + sprout_dir * initial_sprout_length
                
                # Initial flow for angiogenic sprout can be very small or based on a minimal tumor demand
                sprout_initial_flow = DEFAULT_MIN_TUMOR_TERMINAL_DEMAND 
                
                sprout_gbo = GBOIterationData(
                    terminal_id=sprout_tip_id,
                    pos=sprout_tip_pos,
                    radius=initial_sprout_radius,
                    flow=sprout_initial_flow, 
                    parent_id=bif_node_id
                )
                sprout_gbo.length_from_parent = initial_sprout_length
                active_angiogenic_terminals.append(sprout_gbo)

                data_structures.add_node_to_graph(
                    angiogenic_graph, sprout_tip_id,
                    pos=sprout_tip_pos,
                    radius=initial_sprout_radius,
                    type='angiogenic_terminal',
                    is_tumor_vessel=True,
                    vessel_origin_type='angiogenic_sprout', # Clearly mark its origin
                    parent_id=bif_node_id, 
                    Q_flow=sprout_gbo.flow
                )
                data_structures.add_edge_to_graph(
                    angiogenic_graph, bif_node_id, sprout_tip_id,
                    length=initial_sprout_length,
                    radius=initial_sprout_radius, # Edge to new tip takes tip's radius
                    type='angiogenic_segment',
                    is_tumor_vessel=True,
                    permeability_Lp_factor=default_permeability_factor # New angiogenic segments are leaky
                )
                logger.debug(f"    Created new sprout: {sprout_tip_id} from new bif {bif_node_id} on original edge {parent_u}-{parent_v}.")
            
            
            # --- Growth of Active Angiogenic Terminals ---
            next_iter_active_terminals_this_ag_step = []
            
            # Prepare KDTree for anastomosis (only if there are terminals and potential targets)
            vessel_kdtree = None
            segment_midpoints_data = []
            if active_angiogenic_terminals and angiogenic_graph.number_of_edges() > 0:
                midpoints_pos_list = []
                for u, v, data in angiogenic_graph.edges(data=True):
                    # Exclude very new segments connected to active tips to avoid self-anastomosis with parent segment immediately
                    # This check might need refinement
                    is_parent_of_active_tip = any(term.parent_id == u and term.id == v for term in active_angiogenic_terminals)
                    if not is_parent_of_active_tip:
                        mid_pos = (angiogenic_graph.nodes[u]['pos'] + angiogenic_graph.nodes[v]['pos']) / 2.0
                        midpoints_pos_list.append(mid_pos)
                        segment_midpoints_data.append({'pos': mid_pos, 'u': u, 'v': v, 'radius': data.get('radius', constants.MIN_VESSEL_RADIUS_MM)})
                if midpoints_pos_list:
                    vessel_kdtree = KDTree(np.array(midpoints_pos_list))

            newly_branched_terminals_this_ag_step = [] # To hold children from branching
            for term_gbo in active_angiogenic_terminals:
                if term_gbo.stop_growth: continue
                
                # Attempt Anastomosis
                if attempt_anastomosis_tip_to_segment(term_gbo, angiogenic_graph, vessel_kdtree, segment_midpoints_data, config, next_synthetic_node_id_ref):
                    # term_gbo.stop_growth is set by the function
                    continue # Fused, so process next terminal

                # Attempt Branching (Simplified Stochastic)
                current_pos_vox_int = np.round(utils.world_to_voxel(term_gbo.pos, tissue_data['affine'])).astype(int)
                if not utils.is_voxel_in_bounds(current_pos_vox_int, tissue_data['VEGF_field'].shape):
                    term_gbo.stop_growth = True; continue
                
                vegf_at_tip = tissue_data['VEGF_field'][tuple(current_pos_vox_int)]
                normalized_vegf = vegf_at_tip / (np.max(tissue_data['VEGF_field']) + constants.EPSILON)
                prob_branch = branch_probability_factor_vegf * normalized_vegf
                
                if np.random.rand() < prob_branch:
                    logger.debug(f"Angiogenic terminal {term_gbo.id} branching (VEGF: {vegf_at_tip:.2f}, P_branch: {prob_branch:.2f})")
                    # Change current terminal to bifurcation
                    angiogenic_graph.nodes[term_gbo.id]['type'] = 'angiogenic_bifurcation'
                    # Create two new child GBOIterationData objects
                    parent_growth_dir = utils.normalize_vector(term_gbo.pos - angiogenic_graph.nodes[term_gbo.parent_id]['pos'])
                    
                    for i_child in range(2):
                        child_id = f"s_{next_synthetic_node_id_ref[0]}"; next_synthetic_node_id_ref[0] += 1
                        # Perturb direction slightly
                        angle_offset = np.deg2rad(np.random.uniform(-branch_angle_spread_deg/2, branch_angle_spread_deg/2))
                        # This is a 2D rotation logic, needs proper 3D random vector perturbation
                        # For simplicity, create a random perturbation and add to parent_growth_dir then renormalize
                        random_perturb = utils.normalize_vector(np.random.rand(3) - 0.5) * 0.5 # Scale of perturbation
                        child_dir = utils.normalize_vector(parent_growth_dir + random_perturb)
                        if np.linalg.norm(child_dir) < constants.EPSILON: child_dir = parent_growth_dir # Fallback

                        child_pos = term_gbo.pos + child_dir * extension_step_length # Initial small extension
                        child_radius = term_gbo.radius # Or slightly smaller
                        child_flow = term_gbo.flow / 2 # Split flow (very rough)

                        child_gbo = GBOIterationData(child_id, child_pos, child_radius, child_flow, parent_id=term_gbo.id)
                        child_gbo.length_from_parent = extension_step_length
                        newly_branched_terminals_this_ag_step.append(child_gbo)

                        data_structures.add_node_to_graph(angiogenic_graph, child_id, pos=child_pos, radius=child_radius,
                                                          type='angiogenic_terminal', is_tumor_vessel=True, vessel_origin_type='angiogenic_sprout',
                                                          parent_id=term_gbo.id, Q_flow=child_flow)
                        data_structures.add_edge_to_graph(angiogenic_graph, term_gbo.id, child_id, length=extension_step_length,
                                                          radius=child_radius, type='angiogenic_segment', is_tumor_vessel=True,
                                                          permeability_Lp_factor=cooption_params.get("permeability_Lp_factor_tumor", 10.0))
                    term_gbo.stop_growth = True # Parent tip stops, children take over
                    continue

                # Extension (if not anastomosed or branched)
                grad_ax0_f, grad_ax1_f, grad_ax2_f = np.gradient(tissue_data['VEGF_field'])
                g_ax0 = grad_ax0_f[tuple(current_pos_vox_int)]; g_ax1 = grad_ax1_f[tuple(current_pos_vox_int)]; g_ax2 = grad_ax2_f[tuple(current_pos_vox_int)]
                growth_dir_vox = np.array([g_ax0, g_ax1, g_ax2])
                growth_dir_world = utils.normalize_vector(tissue_data['affine'][:3,:3] @ growth_dir_vox)

                if np.linalg.norm(growth_dir_world) > constants.EPSILON:
                    new_pos = term_gbo.pos + growth_dir_world * extension_step_length
                    old_tip_id = term_gbo.id
                    angiogenic_graph.nodes[old_tip_id]['type'] = 'angiogenic_segment_point'
                    new_tip_id = f"s_{next_synthetic_node_id_ref[0]}"; next_synthetic_node_id_ref[0] += 1
                    
                    term_gbo.id = new_tip_id; term_gbo.pos = new_pos; term_gbo.parent_id = old_tip_id
                    term_gbo.length_from_parent = extension_step_length
                    
                    data_structures.add_node_to_graph(angiogenic_graph, new_tip_id, pos=new_pos, radius=term_gbo.radius,
                                                      type='angiogenic_terminal', is_tumor_vessel=True, vessel_origin_type='angiogenic_sprout',
                                                      parent_id=old_tip_id, Q_flow=term_gbo.flow)
                    data_structures.add_edge_to_graph(angiogenic_graph, old_tip_id, new_tip_id, length=extension_step_length,
                                                      radius=term_gbo.radius, type='angiogenic_segment', is_tumor_vessel=True,
                                                      permeability_Lp_factor=cooption_params.get("permeability_Lp_factor_tumor", 10.0))
                    next_iter_active_terminals_this_ag_step.append(term_gbo)
                else:
                    next_iter_active_terminals_this_ag_step.append(term_gbo) # Stalled, keep for next try
            
            active_angiogenic_terminals = [t for t in next_iter_active_terminals_this_ag_step if not t.stop_growth]
            active_angiogenic_terminals.extend(newly_branched_terminals_this_ag_step) # Add new children from branching

            if not active_angiogenic_terminals and not new_sprouts_info : break # from ag_steps
        
        # --- Flow Solve & Adaptation ---
        if (macro_iter + 1) % flow_solve_interval_macro == 0:
            logger.info(f"Running global flow solver and adaptation (Macro Iter {macro_iter + 1})...")
            # ... (Flow solve and differentiated radius adaptation logic - as in previous complete version) ...
            # This part needs careful Q_flow assignment to all terminals (healthy, coopted, angiogenic)
            all_terminals_in_graph = [nid for nid, data in angiogenic_graph.nodes(data=True) if angiogenic_graph.out_degree(nid) == 0 and angiogenic_graph.in_degree(nid) > 0]
            for term_id in all_terminals_in_graph:
                term_node_data = angiogenic_graph.nodes[term_id]
                term_pos_vox = np.round(utils.world_to_voxel(term_node_data['pos'], tissue_data['affine'])).astype(int)
                demand = term_node_data.get('Q_flow', 0.0) # Keep existing Q_flow if not overridden

                if utils.is_voxel_in_bounds(term_pos_vox, tissue_data['shape']):
                    if tissue_data.get('Tumor') is not None and tissue_data['Tumor'][tuple(term_pos_vox)]:
                        # TODO: Better demand based on local tumor voxel demand sum
                        demand = config_manager.get_param(config, "tumor_angiogenesis.min_tumor_terminal_demand", DEFAULT_MIN_TUMOR_TERMINAL_DEMAND)
                    elif not term_node_data.get('is_tumor_vessel'): # Healthy terminal in healthy tissue
                        # This demand should come from GBO's Voronoi refinement for healthy tissue
                        # For now, if not set, use a default. This part needs robust integration with healthy GBO state.
                        demand = term_node_data.get('Q_flow', constants.INITIAL_TERMINAL_FLOW_Q)
                term_node_data['Q_flow'] = demand
            
            temp_graph_for_solver = angiogenic_graph.copy()
            solved_graph = perfusion_solver_func(temp_graph_for_solver, config, None, None)

            if solved_graph:
                angiogenic_graph = solved_graph
                min_r_healthy = config_manager.get_param(config, "vascular_properties.min_radius")
                k_m = config_manager.get_param(config, "vascular_properties.k_murray_scaling_factor")
                m_exp = config_manager.get_param(config, "vascular_properties.murray_law_exponent")
                adapt_params = main_params.get("adaptation", {})
                tumor_radius_factor = adapt_params.get("tumor_radius_factor", 1.0)
                min_r_tumor = adapt_params.get("min_tumor_vessel_radius_mm", min_r_healthy * 1.1)

                for node_id, data in angiogenic_graph.nodes(data=True):
                    if data.get('type') == 'measured_root' or data.get('is_flow_root'): continue
                    
                    actual_node_flow = 0.0 # Recalculate based on solved edge flows
                    if angiogenic_graph.out_degree(node_id) == 0 and angiogenic_graph.in_degree(node_id) > 0: # Sink
                        for _, _, edge_data_in in angiogenic_graph.in_edges(node_id, data=True):
                            actual_node_flow += abs(edge_data_in.get('flow_solver', 0.0))
                    elif angiogenic_graph.out_degree(node_id) > 0: # Source-like or bifurcation
                        for _, _, edge_data_out in angiogenic_graph.out_edges(node_id, data=True):
                            actual_node_flow += abs(edge_data_out.get('flow_solver', 0.0))
                    
                    if abs(actual_node_flow) > constants.EPSILON:
                        target_r = k_m * (abs(actual_node_flow) ** (1.0 / m_exp))
                        if data.get('is_tumor_vessel'):
                            target_r *= tumor_radius_factor
                            data['radius'] = max(min_r_tumor, target_r)
                        else: data['radius'] = max(min_r_healthy, target_r)
                    else: data['radius'] = min_r_tumor if data.get('is_tumor_vessel') else min_r_healthy
                logger.info("Global flow solve and differentiated radius adaptation complete.")


        if (macro_iter + 1) % save_interval_macro == 0:
            # ... (saving logic as before) ...
            logger.info(f"Saving intermediate state for macro iteration {macro_iter + 1}...")
            io_utils.save_vascular_tree_vtp(angiogenic_graph, os.path.join(output_dir, f"angiogenesis_iter_{macro_iter+1}.vtp"))
            io_utils.save_nifti_image(tissue_data['Tumor'].astype(np.uint8), tissue_data['affine'], os.path.join(output_dir, f"active_tumor_mask_iter_{macro_iter+1}.nii.gz"))
            if 'VEGF_field' in tissue_data and tissue_data['VEGF_field'] is not None:
                 io_utils.save_nifti_image(tissue_data['VEGF_field'].astype(np.float32), tissue_data['affine'], os.path.join(output_dir, f"vegf_field_iter_{macro_iter+1}.nii.gz"))

        if np.all(tissue_data['Tumor'] == tissue_data['Tumor_Max_Extent']) and not any_tumor_growth_this_macro:
            logger.info(f"Tumor filled Tumor_Max_Extent. Stopping macro iterations at {macro_iter + 1}.")
            break

    logger.info(f"--- Tumor Angiogenesis (Fixed Extent) Finished. Final graph: {angiogenic_graph.number_of_nodes()} N, {angiogenic_graph.number_of_edges()} E ---")
    return angiogenic_graph