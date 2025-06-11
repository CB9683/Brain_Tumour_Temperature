# src/vascular_growth.py
import numpy as np
import networkx as nx
import logging
import os
from scipy.spatial import KDTree # For finding nearby unperfused voxels
from typing import Tuple, List, Dict, Set, Optional # For type hinting

from src import utils,io_utils, data_structures, constants, config_manager
from src import energy_model 

logger = logging.getLogger(__name__)

class GBOIterationData:
    def __init__(self, terminal_id: str, pos: np.ndarray, radius: float, flow: float, 
                 is_synthetic: bool = True, original_measured_radius: Optional[float] = None,
                 parent_id: Optional[str] = None, parent_measured_terminal_id: Optional[str] = None):
        self.id: str = terminal_id
        self.pos: np.ndarray = np.array(pos)
        self.radius: float = float(radius)
        self.flow: float = float(flow) # Current flow supplied by this terminal to its territory
        self.parent_id: Optional[str] = parent_id
        self.parent_measured_terminal_id: Optional[str] = parent_measured_terminal_id
        self.original_measured_terminal_radius: Optional[float] = original_measured_radius
        self.length_from_parent: float = 0.0 # Will be set if parent_id exists
        self.is_synthetic: bool = is_synthetic
        self.stop_growth: bool = False
        # Flat indices (into tissue_data['world_coords_flat']) of voxels currently supplied
        self.current_territory_voxel_indices_flat: List[int] = [] 
        self.current_territory_demand: float = 0.0 # Demand from current_territory_voxel_indices_flat

def initialize_perfused_territory_and_terminals(
    config: dict,
    initial_graph: Optional[nx.DiGraph], # This will be None if using seeds only
    tissue_data: dict
) -> Tuple[np.ndarray, List[GBOIterationData], int, nx.DiGraph]:
    logger.info("Initializing perfused territory and active terminals...")
    perfused_tissue_mask = np.zeros(tissue_data['shape'], dtype=bool)
    active_terminals: List[GBOIterationData] = []
    
    initial_synthetic_radius_default = config_manager.get_param(config, "vascular_properties.min_radius", 0.005)
    # initial_flow_q_default will be calculated from seed_radius using Murray's if not specified per seed
    k_murray_factor = config_manager.get_param(config, "vascular_properties.k_murray_scaling_factor", 0.5)
    murray_exponent = config_manager.get_param(config, "vascular_properties.murray_law_exponent", 3.0)

    next_synthetic_node_id = 0 # Used for synthetic sprouts from measured, or for synthetic seeds
    
    # Create gbo_graph: start with initial_graph if provided, else empty
    gbo_graph = initial_graph.copy() if initial_graph else data_structures.create_empty_vascular_graph()

    # --- Priority 1: Sprout from measured terminals if initial_graph is valid ---
    processed_from_initial_graph = False
    if initial_graph and initial_graph.number_of_nodes() > 0:
        for node_id, data in initial_graph.nodes(data=True):
            if data.get('type') == 'measured_terminal':
                measured_pos = np.array(data['pos'])
                measured_radius_at_terminal = data.get('radius', initial_synthetic_radius_default)

                # Synthetic sprout starts with min_radius, its flow/radius will update after Ω_init
                term_gbo_data = GBOIterationData(
                    terminal_id=f"s_{next_synthetic_node_id}",
                    pos=measured_pos,
                    radius=initial_synthetic_radius_default, 
                    flow=config_manager.get_param(config, "vascular_properties.initial_terminal_flow", constants.INITIAL_TERMINAL_FLOW_Q), # Seed flow
                    original_measured_radius=measured_radius_at_terminal,
                    parent_id=node_id,
                    parent_measured_terminal_id=node_id
                )
                active_terminals.append(term_gbo_data)
                
                data_structures.add_node_to_graph(
                    gbo_graph, term_gbo_data.id, pos=term_gbo_data.pos, radius=term_gbo_data.radius,
                    type='synthetic_terminal', Q_flow=term_gbo_data.flow, parent_id=node_id,
                    parent_measured_terminal_id=node_id, is_synthetic=True, stop_growth=False,
                    original_measured_terminal_radius=measured_radius_at_terminal
                )
                data_structures.add_edge_to_graph(gbo_graph, node_id, term_gbo_data.id, length=0.0, type='synthetic_sprout')
                if gbo_graph.has_node(node_id): 
                    gbo_graph.nodes[node_id]['type'] = 'measured_bifurcation_or_segment'
                
                logger.info(f"Initialized synthetic terminal {term_gbo_data.id} from measured {node_id} (orig_R={measured_radius_at_terminal:.4f}).")
                next_synthetic_node_id += 1
                processed_from_initial_graph = True
        
        if not processed_from_initial_graph and initial_graph.number_of_nodes() > 0:
            logger.warning("Initial graph provided but no 'measured_terminal' nodes found. Checking for seed points.")


    # --- Priority 2: Use seed points from config if no terminals from initial_graph ---
    if not processed_from_initial_graph:
        seed_points_config = config_manager.get_param(config, "gbo_growth.seed_points", [])
        if seed_points_config and isinstance(seed_points_config, list):
            logger.info(f"No measured terminals processed. Using {len(seed_points_config)} seed points from configuration.")
            for seed_info in seed_points_config:
                seed_id_base = seed_info.get('id', f"cfg_seed_{next_synthetic_node_id}")
                seed_pos = np.array(seed_info.get('position'))
                # Use specified seed radius, or default to min_radius
                seed_initial_radius = float(seed_info.get('initial_radius', initial_synthetic_radius_default))
                
                # Calculate an initial flow for this seed based on its given radius using inverse Murray's
                # Q = (R / K_murray)^(gamma)
                seed_initial_flow = (seed_initial_radius / k_murray_factor) ** murray_exponent \
                                    if seed_initial_radius > 0 and k_murray_factor > 0 \
                                    else config_manager.get_param(config, "vascular_properties.initial_terminal_flow", constants.INITIAL_TERMINAL_FLOW_Q)

                term_gbo_data = GBOIterationData(
                    terminal_id=seed_id_base, # Use ID from config or generated
                    pos=seed_pos,
                    radius=seed_initial_radius, 
                    flow=seed_initial_flow,
                    original_measured_radius=None # No measured parent for these seeds
                )
                active_terminals.append(term_gbo_data)
                
                # Add this seed point as a root node in the gbo_graph
                data_structures.add_node_to_graph(
                    gbo_graph, term_gbo_data.id, pos=term_gbo_data.pos, radius=term_gbo_data.radius,
                    type='synthetic_terminal', # Initially a terminal, also acts as a root of its tree
                    Q_flow=term_gbo_data.flow, 
                    is_synthetic=True, stop_growth=False
                    # No parent_id, parent_measured_terminal_id for these pure seeds
                )
                logger.info(f"Initialized synthetic terminal from config seed: {term_gbo_data.id} at {seed_pos} "
                            f"with R={seed_initial_radius:.4f}, initial Q={seed_initial_flow:.2e}")
                next_synthetic_node_id += 1 # Ensure unique IDs if mixing with other s_X
            processed_from_initial_graph = True # Mark that we got starting points
        else:
            logger.warning("No measured terminals and no valid seed_points found in configuration.")

    # --- Fallback: One seed at domain center (if still no starting points) ---
    if not active_terminals: # If still no terminals after checking graph and config seeds
        logger.warning("No starting points from graph or config seeds. Attempting one fallback seed at domain center.")
        # (Fallback seed logic as before - this is a last resort)
        if tissue_data.get('domain_mask') is not None and np.any(tissue_data['domain_mask']):
            # ... (seed placement logic from your previous version) ...
            # For brevity, assuming it can place a seed:
            # Example:
            seed_point_world = np.mean(tissue_data['world_coords_flat'], axis=0) if tissue_data['world_coords_flat'].shape[0] > 0 else np.array([0,0,0])
            fallback_seed_radius = initial_synthetic_radius_default
            fallback_seed_flow = config_manager.get_param(config, "vascular_properties.initial_terminal_flow", constants.INITIAL_TERMINAL_FLOW_Q)
            term_gbo_data = GBOIterationData(
                terminal_id=f"s_fallback_{next_synthetic_node_id}", pos=seed_point_world, 
                radius=fallback_seed_radius, flow=fallback_seed_flow
            )
            active_terminals.append(term_gbo_data)
            data_structures.add_node_to_graph(
                gbo_graph, term_gbo_data.id, pos=term_gbo_data.pos, radius=term_gbo_data.radius,
                type='synthetic_terminal', Q_flow=term_gbo_data.flow, is_synthetic=True, stop_growth=False
            )
            logger.info(f"Initialized fallback seed terminal {term_gbo_data.id} at {np.round(seed_point_world,2)}.")
            next_synthetic_node_id +=1


    if not active_terminals:
        logger.error("CRITICAL: No terminals (from graph, config seeds, or fallback) to initialize growth. Aborting.")
        return perfused_tissue_mask, [], 0, gbo_graph # Return empty/initial graph

    # --- Define Ω_init (Initial Perfused Territory) for ALL active_terminals ---
    # (The rest of this function: KDTree setup, looping through active_terminals to claim Ω_init,
    #  updating their flow/radius, moving them to centroid, remains largely the same as before)
    # ...
    if tissue_data['world_coords_flat'].shape[0] == 0:
        logger.error("tissue_data['world_coords_flat'] is empty. Cannot initialize Ω_init.")
        return perfused_tissue_mask, active_terminals, next_synthetic_node_id, gbo_graph # Return current state
        
    kdtree_all_domain_voxels = KDTree(tissue_data['world_coords_flat'])
    initial_territory_radius_search = config_manager.get_param(config, "gbo_growth.initial_territory_radius", 0.2) 

    for term_data in active_terminals:
        # ... (Ω_init claiming logic - unchanged from your working version) ...
        # Example snippet (ensure this part is complete from your working version):
        nearby_domain_voxel_flat_indices = kdtree_all_domain_voxels.query_ball_point(
            term_data.pos, r=initial_territory_radius_search
        )
        actual_demand_in_init_territory = 0.0
        voxels_for_this_terminal_init_flat: List[int] = []
        for v_idx_flat in nearby_domain_voxel_flat_indices:
            v_3d_idx_tuple = tuple(tissue_data['voxel_indices_flat'][v_idx_flat])
            if not perfused_tissue_mask[v_3d_idx_tuple]: # Check if not already globally perfused
                perfused_tissue_mask[v_3d_idx_tuple] = True
                actual_demand_in_init_territory += tissue_data['metabolic_demand_map'][v_3d_idx_tuple] * tissue_data['voxel_volume']
                voxels_for_this_terminal_init_flat.append(v_idx_flat)
        
        term_data.current_territory_voxel_indices_flat = voxels_for_this_terminal_init_flat
        term_data.current_territory_demand = actual_demand_in_init_territory
        
        if actual_demand_in_init_territory > constants.EPSILON:
            term_data.flow = actual_demand_in_init_territory
            new_r = k_murray_factor * (term_data.flow ** (1.0 / murray_exponent))
            # If it was a config seed, its initial radius was set from config, don't shrink below that here.
            # If it was a sprout from measured, it started at min_radius.
            # term_data.radius was already set (either min_radius or seed_initial_radius)
            term_data.radius = max(term_data.radius, new_r, initial_synthetic_radius_default) 
        else:
            # If no demand claimed, flow remains its initial seed flow (or default initial_terminal_flow)
            # And radius remains its initial seed radius (or default min_radius)
            pass # term_data.flow and term_data.radius already set
            
        if gbo_graph.has_node(term_data.id):
            gbo_graph.nodes[term_data.id]['Q_flow'] = term_data.flow
            gbo_graph.nodes[term_data.id]['radius'] = term_data.radius
        
        # Move to centroid of Ω_init
        if term_data.current_territory_voxel_indices_flat:
            initial_territory_coords = tissue_data['world_coords_flat'][term_data.current_territory_voxel_indices_flat]
            if initial_territory_coords.shape[0] > 0:
                new_initial_pos = np.mean(initial_territory_coords, axis=0)
                logger.debug(f"Terminal {term_data.id} (Ω_init): Moving from {np.round(term_data.pos,3)} to centroid {np.round(new_initial_pos,3)}.")
                term_data.pos = new_initial_pos
                if gbo_graph.has_node(term_data.id):
                    gbo_graph.nodes[term_data.id]['pos'] = term_data.pos
                    if term_data.parent_id and gbo_graph.has_edge(term_data.parent_id, term_data.id):
                        parent_node_pos = gbo_graph.nodes[term_data.parent_id]['pos']
                        new_len = utils.distance(parent_node_pos, term_data.pos)
                        gbo_graph.edges[term_data.parent_id, term_data.id]['length'] = new_len
                        term_data.length_from_parent = new_len
        
        logger.debug(f"Terminal {term_data.id} (Ω_init final): Pos={np.round(term_data.pos,3)}, Claimed {len(voxels_for_this_terminal_init_flat)} voxels, "
                    f"Demand={term_data.current_territory_demand:.2e}, Flow={term_data.flow:.2e}, Radius={term_data.radius:.4f}")

    logger.info(f"Initialization complete. Perfused {np.sum(perfused_tissue_mask)} initial voxels. "
                f"{len(active_terminals)} active terminals.")
    return perfused_tissue_mask, active_terminals, next_synthetic_node_id, gbo_graph


# In src/vascular_growth.py

def find_growth_frontier_voxels(
    terminal_gbo_data: GBOIterationData, # Contains current terminal pos and radius
    kdtree_unperfused_domain_voxels: Optional[KDTree], 
    unperfused_global_flat_indices: np.ndarray, # Global flat indices for points in KDTree
    tissue_data: dict, 
    config: dict
) -> np.ndarray: # Returns global flat indices of voxels in Ri,p
    """Identifies a small region of unperfused tissue (Ri,p) adjacent/near the terminal."""
    
    logger.debug(f"Terminal {terminal_gbo_data.id}: Entering find_growth_frontier_voxels. "
                 f"Pos: {np.round(terminal_gbo_data.pos,3)}, Radius: {terminal_gbo_data.radius:.4f}")

    if kdtree_unperfused_domain_voxels is None or kdtree_unperfused_domain_voxels.n == 0:
        logger.debug(f"Terminal {terminal_gbo_data.id}: KDTree of unperfused voxels is empty or None. No frontier.")
        return np.array([], dtype=int)

    radius_factor = config_manager.get_param(config, "gbo_growth.frontier_search_radius_factor", 3.0)
    fixed_radius = config_manager.get_param(config, "gbo_growth.frontier_search_radius_fixed", 0.25) # mm
    voxel_dim = tissue_data['voxel_volume']**(1/3.0)

    # Effective search radius from the terminal's centerline
    search_r = max(radius_factor * terminal_gbo_data.radius, fixed_radius, voxel_dim * 1.5) # Ensure it's at least > 1 voxel diagonal
    
    logger.debug(f"Terminal {terminal_gbo_data.id}: Searching for frontier with effective radius {search_r:.3f}mm. "
                 f"(voxel_dim ~{voxel_dim:.3f}, term_radius_component ~{radius_factor * terminal_gbo_data.radius:.3f})")

    # Query for unperfused domain voxels near the terminal position
    # These are local indices into the array that built kdtree_unperfused_domain_voxels
    try:
        local_indices_in_kdtree = kdtree_unperfused_domain_voxels.query_ball_point(terminal_gbo_data.pos, r=search_r)
    except Exception as e:
        logger.error(f"Terminal {terminal_gbo_data.id}: KDTree query_ball_point failed: {e}")
        return np.array([], dtype=int)
        
    logger.debug(f"Terminal {terminal_gbo_data.id}: KDTree query_ball_point found {len(local_indices_in_kdtree)} local indices "
                 f"within {search_r:.3f}mm.")
    
    if not local_indices_in_kdtree: # Check if the list is empty
        return np.array([], dtype=int)
        
    # Map these local KDTree indices back to global flat indices
    # Ensure unperfused_global_flat_indices is not empty and indices are valid
    if unperfused_global_flat_indices.shape[0] == 0:
        logger.warning(f"Terminal {terminal_gbo_data.id}: unperfused_global_flat_indices is empty, cannot map KDTree results.")
        return np.array([], dtype=int)
    
    # Check if all local_indices_in_kdtree are valid for unperfused_global_flat_indices
    max_kdtree_idx = np.max(local_indices_in_kdtree) if len(local_indices_in_kdtree) > 0 else -1
    if max_kdtree_idx >= len(unperfused_global_flat_indices):
        logger.error(f"Terminal {terminal_gbo_data.id}: Invalid index from KDTree query. Max KDTree index {max_kdtree_idx} vs "
                     f"unperfused_global_flat_indices length {len(unperfused_global_flat_indices)}. This indicates a mismatch.")
        # This can happen if kdtree_unperfused_domain_voxels was built on a different set of points
        # than what unperfused_global_flat_indices refers to.
        return np.array([], dtype=int)

    frontier_voxels_global_flat_indices_initial = unperfused_global_flat_indices[local_indices_in_kdtree]
    logger.debug(f"Terminal {terminal_gbo_data.id}: Initial frontier (after mapping to global flat indices) "
                 f"contains {len(frontier_voxels_global_flat_indices_initial)} voxels.")

    # Filter: limit the size of Ri,p
    max_voxels_in_Rip = config_manager.get_param(config, "gbo_growth.max_voxels_for_Rip", 50)
    final_frontier_voxels_global_flat_indices = frontier_voxels_global_flat_indices_initial

    if len(frontier_voxels_global_flat_indices_initial) > max_voxels_in_Rip:
        logger.debug(f"Terminal {terminal_gbo_data.id}: Initial frontier size ({len(frontier_voxels_global_flat_indices_initial)}) "
                     f"exceeds max_voxels_for_Rip ({max_voxels_in_Rip}). Selecting closest ones.")
        # If too many, take the closest ones. Re-query KDTree with k=max_voxels_in_Rip.
        # This query returns (distances, indices)
        try:
            # Ensure k is not greater than number of points in tree
            k_val = min(max_voxels_in_Rip, kdtree_unperfused_domain_voxels.n)
            if k_val > 0 :
                distances, local_indices_k_closest = kdtree_unperfused_domain_voxels.query(terminal_gbo_data.pos, k=k_val)
                
                # query might return single int if k=1 and not a list/array
                if isinstance(local_indices_k_closest, (int, np.integer)): 
                    local_indices_k_closest = np.array([local_indices_k_closest])
                
                if len(local_indices_k_closest) > 0:
                     # Check for invalid indices again before mapping
                    max_k_closest_idx = np.max(local_indices_k_closest) if len(local_indices_k_closest) > 0 else -1
                    if max_k_closest_idx < len(unperfused_global_flat_indices):
                        final_frontier_voxels_global_flat_indices = unperfused_global_flat_indices[local_indices_k_closest]
                    else:
                        logger.error(f"Terminal {terminal_gbo_data.id}: Invalid index from KDTree k-closest query. Max index {max_k_closest_idx} vs "
                                     f"unperfused_global_flat_indices length {len(unperfused_global_flat_indices)}.")
                        # Fallback to initial query if k-closest had issues, or keep it empty
                        final_frontier_voxels_global_flat_indices = np.array([], dtype=int) # Safer to return empty
                else:
                    final_frontier_voxels_global_flat_indices = np.array([], dtype=int)
            else: # k_val is 0
                 final_frontier_voxels_global_flat_indices = np.array([], dtype=int)

        except Exception as e_kquery:
            logger.error(f"Terminal {terminal_gbo_data.id}: KDTree k-closest query failed: {e_kquery}. "
                         f"Using initial ball query result if valid, or empty.")
            # Fallback to the initial ball query result if it was valid, otherwise ensure it's empty
            if np.any(np.array(local_indices_in_kdtree) >= len(unperfused_global_flat_indices)):
                 final_frontier_voxels_global_flat_indices = np.array([], dtype=int) # Problem with original indices too
            # else: final_frontier_voxels_global_flat_indices remains from ball_query

        logger.debug(f"Terminal {terminal_gbo_data.id}: Frontier limited by max_voxels_for_Rip to "
                     f"{len(final_frontier_voxels_global_flat_indices)} voxels.")

    logger.info(f"Terminal {terminal_gbo_data.id} identified {len(final_frontier_voxels_global_flat_indices)} "
                 f"final frontier voxels (Ri,p).") # Changed to INFO for test visibility
    return final_frontier_voxels_global_flat_indices


# In src/vascular_growth.py
# (Assume GBOIterationData class, initialize_perfused_territory_and_terminals, 
#  and find_growth_frontier_voxels are defined as before)

def grow_healthy_vasculature(config: dict,
                             tissue_data: dict,
                             initial_graph: Optional[nx.DiGraph],
                             output_dir: str) -> Optional[nx.DiGraph]:
    logger.info("Starting GBO healthy vascular growth (Tissue-Led Model v2)...")

    # --- 1. Initialization ---
    perfused_tissue_mask, current_active_terminals, next_node_id, gbo_graph = \
        initialize_perfused_territory_and_terminals(config, initial_graph, tissue_data)

    if not current_active_terminals:
        return gbo_graph 

    max_iterations = config_manager.get_param(config, "gbo_growth.max_iterations", 100)
    min_radius = config_manager.get_param(config, "vascular_properties.min_radius", constants.MIN_VESSEL_RADIUS_MM)
    k_murray = config_manager.get_param(config, "vascular_properties.k_murray_scaling_factor", 0.5)
    murray_exp = config_manager.get_param(config, "vascular_properties.murray_law_exponent", 3.0)
    
    branch_radius_factor_thresh = config_manager.get_param(config, "gbo_growth.branch_radius_increase_threshold", 1.1)
    max_flow_single_term = config_manager.get_param(config, "gbo_growth.max_flow_single_terminal", 0.005)
    min_iters_no_growth_stop = config_manager.get_param(config, "gbo_growth.min_iterations_before_no_growth_stop", 10)


    total_voxels_in_domain = np.sum(tissue_data['domain_mask'])
    if total_voxels_in_domain == 0:
        logger.error("Domain mask is empty. Cannot proceed with growth.")
        return gbo_graph
        
    map_3d_to_flat_idx = -np.ones(tissue_data['shape'], dtype=np.int64)
    if tissue_data['voxel_indices_flat'].shape[0] > 0:
        map_3d_to_flat_idx[tissue_data['voxel_indices_flat'][:,0], 
                           tissue_data['voxel_indices_flat'][:,1], 
                           tissue_data['voxel_indices_flat'][:,2]] = np.arange(tissue_data['voxel_indices_flat'].shape[0])

    # --- Main Iteration Loop ---
    for iteration in range(max_iterations):
        logger.info(f"--- GBO Iteration {iteration + 1} / {max_iterations} ---")
        
        terminals_for_growth_attempt = [t for t in current_active_terminals if not t.stop_growth]
        if not terminals_for_growth_attempt:
            logger.info("GBO: No active terminals left for growth at start of iteration.")
            # This check is also done at the end, but good for early exit
            break 
        
        current_perfused_count = np.sum(perfused_tissue_mask)
        logger.info(f"Active terminals: {len(terminals_for_growth_attempt)}. Perfused voxels: {current_perfused_count}/{total_voxels_in_domain} "
                    f"({(current_perfused_count/total_voxels_in_domain)*100:.1f}%)")

        unperfused_mask_3d = tissue_data['domain_mask'] & (~perfused_tissue_mask)
        unperfused_voxels_3d_indices = np.array(np.where(unperfused_mask_3d)).T
        
        kdtree_unperfused: Optional[KDTree] = None
        unperfused_global_flat_indices_for_kdtree: np.ndarray = np.array([], dtype=int)

        if unperfused_voxels_3d_indices.shape[0] > 0:
            unperfused_voxels_world_coords = utils.voxel_to_world(unperfused_voxels_3d_indices, tissue_data['affine'])
            kdtree_unperfused = KDTree(unperfused_voxels_world_coords)
            
            flat_indices_temp = map_3d_to_flat_idx[unperfused_voxels_3d_indices[:,0],
                                                   unperfused_voxels_3d_indices[:,1],
                                                   unperfused_voxels_3d_indices[:,2]]
            valid_mask = flat_indices_temp != -1 # Ensure all mapped indices are valid
            unperfused_global_flat_indices_for_kdtree = flat_indices_temp[valid_mask]
            if not np.all(valid_mask): # If some were invalid, KDTree needs to be rebuilt on filtered coords
                logger.warning("Some unperfused 3D indices did not map to valid flat indices. Rebuilding KDTree on valid subset.")
                unperfused_voxels_world_coords = unperfused_voxels_world_coords[valid_mask]
                if unperfused_voxels_world_coords.shape[0] > 0:
                    kdtree_unperfused = KDTree(unperfused_voxels_world_coords)
                else:
                    kdtree_unperfused = None # No valid unperfused points left

        if kdtree_unperfused is None or kdtree_unperfused.n == 0:
            logger.info("GBO: No unperfused domain voxels left for KDTree. Target likely achieved.")
            break

        next_iter_terminals_manager: List[GBOIterationData] = []
        newly_perfused_in_iter_mask = np.zeros_like(perfused_tissue_mask)

        for term_p_gbo_data in terminals_for_growth_attempt:
            if term_p_gbo_data.stop_growth: # Should have been filtered, but double check
                next_iter_terminals_manager.append(term_p_gbo_data)
                continue

            logger.debug(f"Processing terminal {term_p_gbo_data.id}. Pos: {np.round(term_p_gbo_data.pos,3)}, "
                         f"Radius: {term_p_gbo_data.radius:.4f}, Current Flow: {term_p_gbo_data.flow:.2e}")
            
            frontier_voxels_global_flat_indices = find_growth_frontier_voxels(
                term_p_gbo_data, kdtree_unperfused, 
                unperfused_global_flat_indices_for_kdtree,
                tissue_data, config
            )

            if len(frontier_voxels_global_flat_indices) == 0:
                next_iter_terminals_manager.append(term_p_gbo_data)
                continue

            demand_map_3d_indices_frontier = tissue_data['voxel_indices_flat'][frontier_voxels_global_flat_indices]
            demand_of_frontier_voxels = tissue_data['metabolic_demand_map'][
                demand_map_3d_indices_frontier[:,0],
                demand_map_3d_indices_frontier[:,1],
                demand_map_3d_indices_frontier[:,2]
            ] * tissue_data['voxel_volume']
            demand_Rip = np.sum(demand_of_frontier_voxels)

            if demand_Rip < constants.EPSILON:
                next_iter_terminals_manager.append(term_p_gbo_data)
                continue
            
            potential_total_flow_if_extended = term_p_gbo_data.flow + demand_Rip 
            potential_radius_if_extended = k_murray * (potential_total_flow_if_extended ** (1.0 / murray_exp))
            
            attempt_branching = False
            if term_p_gbo_data.radius > 0 and \
               potential_radius_if_extended > term_p_gbo_data.radius * branch_radius_factor_thresh :
                attempt_branching = True
            if potential_total_flow_if_extended > max_flow_single_term:
                attempt_branching = True
            
            if attempt_branching and len(frontier_voxels_global_flat_indices) >= 1: # Min 1 voxel for KMeans fallback
                logger.debug(f"Terminal {term_p_gbo_data.id} attempting to branch for Ri,p (demand {demand_Rip:.2e}). Pot. new R={potential_radius_if_extended:.4f}")
                
                bifurcation_result = energy_model.find_optimal_bifurcation_for_new_region(
                    term_p_gbo_data, frontier_voxels_global_flat_indices, tissue_data, config,
                    k_murray, murray_exp
                )

                if bifurcation_result:
                    c1_pos, c1_rad, c1_flow_new, c2_pos, c2_rad, c2_flow_new, _ = bifurcation_result
                    
                    new_parent_flow = term_p_gbo_data.flow + c1_flow_new + c2_flow_new
                    new_parent_radius = k_murray * (new_parent_flow ** (1.0 / murray_exp))
                    new_parent_radius = max(min_radius, new_parent_radius) # Ensure parent radius not too small
                    
                    gbo_graph.nodes[term_p_gbo_data.id]['type'] = 'synthetic_bifurcation'
                    gbo_graph.nodes[term_p_gbo_data.id]['radius'] = new_parent_radius
                    gbo_graph.nodes[term_p_gbo_data.id]['Q_flow'] = new_parent_flow
                    
                    child1_id = f"s_{next_node_id}"
                    next_node_id += 1
                    c1_gbo = GBOIterationData(child1_id, c1_pos, c1_rad, c1_flow_new,
                                              original_measured_radius=term_p_gbo_data.original_measured_terminal_radius,
                                              parent_id=term_p_gbo_data.id, 
                                              parent_measured_terminal_id=term_p_gbo_data.parent_measured_terminal_id)
                    c1_gbo.length_from_parent = utils.distance(term_p_gbo_data.pos, c1_pos)
                    next_iter_terminals_manager.append(c1_gbo)
                    data_structures.add_node_to_graph(gbo_graph, c1_gbo.id, **vars(c1_gbo))
                    data_structures.add_edge_to_graph(gbo_graph, term_p_gbo_data.id, c1_gbo.id, 
                                                      length=c1_gbo.length_from_parent, type='synthetic_segment')

                    child2_id = f"s_{next_node_id}"
                    next_node_id += 1
                    c2_gbo = GBOIterationData(child2_id, c2_pos, c2_rad, c2_flow_new,
                                               original_measured_radius=term_p_gbo_data.original_measured_terminal_radius,
                                               parent_id=term_p_gbo_data.id,
                                               parent_measured_terminal_id=term_p_gbo_data.parent_measured_terminal_id)
                    c2_gbo.length_from_parent = utils.distance(term_p_gbo_data.pos, c2_pos)
                    next_iter_terminals_manager.append(c2_gbo)
                    data_structures.add_node_to_graph(gbo_graph, c2_gbo.id, **vars(c2_gbo))
                    data_structures.add_edge_to_graph(gbo_graph, term_p_gbo_data.id, c2_gbo.id, 
                                                      length=c2_gbo.length_from_parent, type='synthetic_segment')

                    for v_idx_flat in frontier_voxels_global_flat_indices:
                        v_3d_idx = tuple(tissue_data['voxel_indices_flat'][v_idx_flat])
                        newly_perfused_in_iter_mask[v_3d_idx] = True
                    logger.info(f"Terminal {term_p_gbo_data.id} branched into {child1_id} & {child2_id} to supply new demand {demand_Rip:.2e}.")
                else: 
                    attempt_branching = False 
            
            if not attempt_branching: 
                logger.debug(f"Terminal {term_p_gbo_data.id} (R={term_p_gbo_data.radius:.4f}, Q_curr={term_p_gbo_data.flow:.2e}) "
                             f"extending for Ri,p (demand {demand_Rip:.2e}).")
                
                old_pos = term_p_gbo_data.pos.copy()
                term_p_gbo_data.flow += demand_Rip 
                term_p_gbo_data.radius = k_murray * (term_p_gbo_data.flow ** (1.0 / murray_exp))
                term_p_gbo_data.radius = max(min_radius, term_p_gbo_data.radius)
                
                if len(frontier_voxels_global_flat_indices) > 0:
                    newly_acquired_coords = tissue_data['world_coords_flat'][frontier_voxels_global_flat_indices]
                    
                    current_territory_coords_list = []
                    if term_p_gbo_data.current_territory_voxel_indices_flat:
                        # Ensure indices are valid before fetching coordinates
                        valid_current_indices = [idx for idx in term_p_gbo_data.current_territory_voxel_indices_flat if idx < tissue_data['world_coords_flat'].shape[0]]
                        if valid_current_indices:
                            current_territory_coords_list.append(tissue_data['world_coords_flat'][valid_current_indices])
                    
                    if newly_acquired_coords.shape[0] > 0:
                        current_territory_coords_list.append(newly_acquired_coords)

                    if current_territory_coords_list:
                        all_supplied_coords = np.vstack(current_territory_coords_list)
                        if all_supplied_coords.shape[0] > 0:
                            new_target_pos = np.mean(all_supplied_coords, axis=0)
                            extension_vector = new_target_pos - old_pos
                            extension_length = np.linalg.norm(extension_vector)
                            max_segment_len_from_config = config_manager.get_param(config, "vascular_properties.max_segment_length", 2.0)

                            if extension_length > constants.EPSILON:
                                if extension_length > max_segment_len_from_config:
                                    term_p_gbo_data.pos = old_pos + extension_vector * (max_segment_len_from_config / extension_length)
                                else:
                                    term_p_gbo_data.pos = new_target_pos
                                
                                logger.debug(f"Terminal {term_p_gbo_data.id} moved from {np.round(old_pos,3)} to {np.round(term_p_gbo_data.pos,3)}")
                                if gbo_graph.has_node(term_p_gbo_data.id) and term_p_gbo_data.parent_id and \
                                   gbo_graph.has_edge(term_p_gbo_data.parent_id, term_p_gbo_data.id):
                                    parent_node_pos = gbo_graph.nodes[term_p_gbo_data.parent_id]['pos']
                                    new_len = utils.distance(parent_node_pos, term_p_gbo_data.pos)
                                    gbo_graph.edges[term_p_gbo_data.parent_id, term_p_gbo_data.id]['length'] = new_len
                                    term_p_gbo_data.length_from_parent = new_len
                            else:
                                logger.debug(f"Terminal {term_p_gbo_data.id} extension target is current pos. No physical move.")
                        else:
                             logger.debug(f"Terminal {term_p_gbo_data.id}: No valid coordinates to determine movement target during extension.")
                    else:
                        logger.debug(f"Terminal {term_p_gbo_data.id}: No coordinates (current or new) for movement calculation.")

                if gbo_graph.has_node(term_p_gbo_data.id): # Update pos, Q, R
                    gbo_graph.nodes[term_p_gbo_data.id]['pos'] = term_p_gbo_data.pos 
                    gbo_graph.nodes[term_p_gbo_data.id]['Q_flow'] = term_p_gbo_data.flow
                    gbo_graph.nodes[term_p_gbo_data.id]['radius'] = term_p_gbo_data.radius
                
                for v_idx_flat in frontier_voxels_global_flat_indices:
                    v_3d_idx = tuple(tissue_data['voxel_indices_flat'][v_idx_flat])
                    newly_perfused_in_iter_mask[v_3d_idx] = True
                
                term_p_gbo_data.current_territory_voxel_indices_flat.extend(list(frontier_voxels_global_flat_indices))
                next_iter_terminals_manager.append(term_p_gbo_data)

        current_active_terminals = next_iter_terminals_manager
        perfused_tissue_mask = perfused_tissue_mask | newly_perfused_in_iter_mask
        num_newly_perfused_this_iter = np.sum(newly_perfused_in_iter_mask)
        logger.info(f"Perfused {num_newly_perfused_this_iter} new voxels in this iteration.")
        
        # --- Territory Refinement (Weighted Voronoi on *current* perfused_tissue_mask) ---
        if current_active_terminals and np.any(perfused_tissue_mask):
            live_terminals_for_voronoi = [t for t in current_active_terminals if not t.stop_growth]
            if live_terminals_for_voronoi:
                perfused_3d_indices = np.array(np.where(perfused_tissue_mask)).T
                
                # Get global flat indices for these currently perfused voxels
                perfused_global_flat_indices = map_3d_to_flat_idx[perfused_3d_indices[:,0], 
                                                                  perfused_3d_indices[:,1], 
                                                                  perfused_3d_indices[:,2]]
                valid_flat_mask_for_perf = perfused_global_flat_indices != -1
                perfused_global_flat_indices = perfused_global_flat_indices[valid_flat_mask_for_perf]
                
                # Get world coords for only the validly mapped perfused voxels
                perfused_world_coords_for_voronoi = tissue_data['world_coords_flat'][perfused_global_flat_indices]


                if perfused_global_flat_indices.shape[0] > 0 and perfused_world_coords_for_voronoi.shape[0] > 0:
                    term_positions_vor = np.array([t.pos for t in live_terminals_for_voronoi])
                    term_flows_vor = np.array([t.flow if t.flow > constants.EPSILON else constants.INITIAL_TERMINAL_FLOW_Q 
                                               for t in live_terminals_for_voronoi])

                    assigned_local_term_indices_to_perf_vox = np.full(perfused_world_coords_for_voronoi.shape[0], -1, dtype=int)
                    for i_pvox, p_vox_wc in enumerate(perfused_world_coords_for_voronoi):
                        distances_sq = np.sum((term_positions_vor - p_vox_wc)**2, axis=1)
                        weighted_distances = distances_sq / term_flows_vor
                        assigned_local_term_indices_to_perf_vox[i_pvox] = np.argmin(weighted_distances)
                    
                    for t_data in live_terminals_for_voronoi:
                        t_data.current_territory_voxel_indices_flat = []
                        t_data.current_territory_demand = 0.0

                    for i_pvox, local_term_idx in enumerate(assigned_local_term_indices_to_perf_vox):
                        if local_term_idx != -1:
                            term_obj = live_terminals_for_voronoi[local_term_idx]
                            global_flat_v_idx = perfused_global_flat_indices[i_pvox]
                            term_obj.current_territory_voxel_indices_flat.append(global_flat_v_idx)
                            
                            v_3d_idx_for_demand = tuple(tissue_data['voxel_indices_flat'][global_flat_v_idx])
                            term_obj.current_territory_demand += tissue_data['metabolic_demand_map'][v_3d_idx_for_demand] * tissue_data['voxel_volume']
                    
                    for t_data in live_terminals_for_voronoi:
                        if t_data.current_territory_demand > constants.EPSILON:
                            t_data.flow = t_data.current_territory_demand
                        else:
                            t_data.flow = config_manager.get_param(config, "vascular_properties.initial_terminal_flow", constants.INITIAL_TERMINAL_FLOW_Q)
                        
                        new_r = k_murray * (t_data.flow ** (1.0 / murray_exp))
                        t_data.radius = max(min_radius, new_r)
                        
                        if gbo_graph.has_node(t_data.id):
                            gbo_graph.nodes[t_data.id]['Q_flow'] = t_data.flow
                            gbo_graph.nodes[t_data.id]['radius'] = t_data.radius
                        logger.debug(f"Term {t_data.id} (Refined Terr.): Q={t_data.flow:.2e}, R={t_data.radius:.4f}, "
                                     f"Supplying {len(t_data.current_territory_voxel_indices_flat)} voxels.")
        
        # --- Update Stop Flags for next iteration's filter ---
        active_terminals_still_growing = 0
        for term_data in current_active_terminals: # Check all, including newly added/modified ones
            if term_data.stop_growth: continue # Already decided to stop

            if term_data.radius < min_radius + constants.EPSILON :
                term_data.stop_growth = True
                logger.debug(f"Terminal {term_data.id} stopping: min radius ({min_radius:.4f}) reached (R={term_data.radius:.4f}).")
            
            if term_data.original_measured_terminal_radius is not None:
                stop_factor = config_manager.get_param(config, "gbo_growth.stop_criteria.max_radius_factor_measured", 1.0)
                if term_data.radius > term_data.original_measured_terminal_radius * stop_factor:
                    logger.info(f"Terminal {term_data.id} (R={term_data.radius:.4f}) stopping: exceeded {stop_factor*100:.0f}% "
                                 f"of its origin measured R ({term_data.original_measured_terminal_radius:.4f}).")
                    term_data.stop_growth = True
            
            # If a terminal has no territory after refinement, it might also stop (or be pruned later)
            if not term_data.current_territory_voxel_indices_flat and term_data.current_territory_demand < constants.EPSILON:
                logger.debug(f"Terminal {term_data.id} stopping: no current territory after refinement.")
                term_data.stop_growth = True

            if gbo_graph.has_node(term_data.id):
                gbo_graph.nodes[term_data.id]['stop_growth'] = term_data.stop_growth
            
            if not term_data.stop_growth:
                active_terminals_still_growing += 1

        # --- Intermediate Save ---
        save_this_iter = False
        if config_manager.get_param(config, "visualization.save_intermediate_steps", False):
            interval = config_manager.get_param(config, "visualization.intermediate_step_interval", 1)
            if ((iteration + 1) % interval == 0) or \
               (iteration == max_iterations - 1) or \
               (num_newly_perfused_this_iter == 0 and iteration > min_iters_no_growth_stop) or \
               (np.sum(perfused_tissue_mask) >= total_voxels_in_domain * config_manager.get_param(config, "gbo_growth.target_domain_perfusion_fraction", 0.99)) or \
               (active_terminals_still_growing == 0 and iteration > 0):
                save_this_iter = True

            if save_this_iter:
                logger.info(f"Saving intermediate results for iteration {iteration + 1}...")
                io_utils.save_vascular_tree_vtp(gbo_graph, os.path.join(output_dir, f"gbo_graph_iter_{iteration+1}.vtp"))
                io_utils.save_nifti_image(perfused_tissue_mask.astype(np.uint8), tissue_data['affine'], 
                                          os.path.join(output_dir, f"perfused_mask_iter_{iteration+1}.nii.gz"))

        # --- Global Stop Conditions ---
        if np.sum(perfused_tissue_mask) >= total_voxels_in_domain * config_manager.get_param(config, "gbo_growth.target_domain_perfusion_fraction", 0.99):
            logger.info(f"GBO Stopping after iteration {iteration + 1}: Target domain perfusion fraction reached.")
            break
        
        if active_terminals_still_growing == 0 and iteration > 0:
             logger.info(f"GBO Stopping after iteration {iteration + 1}: No terminals actively growing.")
             break
        
        if num_newly_perfused_this_iter == 0 and iteration >= min_iters_no_growth_stop: # Use >= for more robust stop
            logger.info(f"GBO Stopping after iteration {iteration + 1}: No new tissue perfused for {iteration - min_iters_no_growth_stop +1} iterations beyond grace period.")
            break
            
    # End of main iteration loop
    logger.info("GBO healthy vascular growth (Tissue-Led Model v2) finished.")
    return gbo_graph