# src/vascular_growth.py
from __future__ import annotations # Must be first line for postponed evaluation of annotations

import numpy as np
import networkx as nx
import logging
import os
from scipy.spatial import KDTree 
from typing import Tuple, List, Dict, Set, Optional 

from src import utils, data_structures, constants, config_manager
from src import energy_model 
from src import perfusion_solver

logger = logging.getLogger(__name__)

class GBOIterationData: # Definition should be here
    def __init__(self, terminal_id: str, pos: np.ndarray, radius: float, flow: float, 
                 is_synthetic: bool = True, original_measured_radius: Optional[float] = None,
                 parent_id: Optional[str] = None, parent_measured_terminal_id: Optional[str] = None):
        self.id: str = terminal_id
        self.pos: np.ndarray = np.array(pos, dtype=float)
        self.radius: float = float(radius)
        self.flow: float = float(flow) 
        self.parent_id: Optional[str] = parent_id
        self.parent_measured_terminal_id: Optional[str] = parent_measured_terminal_id
        self.original_measured_terminal_radius: Optional[float] = original_measured_radius
        self.length_from_parent: float = 0.0 
        self.is_synthetic: bool = is_synthetic
        self.stop_growth: bool = False
        self.current_territory_voxel_indices_flat: List[int] = [] 
        self.current_territory_demand: float = 0.0

def initialize_perfused_territory_and_terminals(
    config: dict,
    initial_graph: Optional[nx.DiGraph],
    tissue_data: dict
) -> Tuple[np.ndarray, List[GBOIterationData], int, nx.DiGraph]:
    logger.info("Initializing perfused territory and active terminals...")
    perfused_tissue_mask = np.zeros(tissue_data['shape'], dtype=bool)
    active_terminals: List[GBOIterationData] = []
    
    initial_synthetic_radius_default = config_manager.get_param(config, "vascular_properties.min_radius", 0.005)
    k_murray_factor = config_manager.get_param(config, "vascular_properties.k_murray_scaling_factor", 0.5)
    murray_exponent = config_manager.get_param(config, "vascular_properties.murray_law_exponent", 3.0)
    default_initial_flow = config_manager.get_param(config, "vascular_properties.initial_terminal_flow", constants.INITIAL_TERMINAL_FLOW_Q)

    next_synthetic_node_id = 0
    gbo_graph = initial_graph.copy() if initial_graph else data_structures.create_empty_vascular_graph()
    
    if initial_graph:
        for node_id, data in initial_graph.nodes(data=True): # Ensure iteration over gbo_graph if it's a copy
            if gbo_graph.nodes[node_id].get('type') == 'measured_root':
                gbo_graph.nodes[node_id]['is_flow_root'] = True
                logger.debug(f"Marked measured_root {node_id} as is_flow_root=True.")

    processed_from_initial_graph = False
    if initial_graph and initial_graph.number_of_nodes() > 0:
        for node_id, data in initial_graph.nodes(data=True): # Iterate over original to find terminals
            if data.get('type') == 'measured_terminal':
                measured_pos = np.array(data['pos'], dtype=float)
                measured_radius_at_terminal = data.get('radius', initial_synthetic_radius_default)
                term_gbo_data = GBOIterationData(
                    terminal_id=f"s_{next_synthetic_node_id}", pos=measured_pos,
                    radius=initial_synthetic_radius_default, flow=default_initial_flow,
                    original_measured_radius=measured_radius_at_terminal,
                    parent_id=node_id, parent_measured_terminal_id=node_id
                )
                active_terminals.append(term_gbo_data)
                # Add node to gbo_graph (which is a copy of initial_graph or empty)
                # Attributes from GBOIterationData are primary, is_flow_root is False for these sprouts
                node_attrs_sprout = vars(term_gbo_data).copy()
                node_attrs_sprout['type'] = 'synthetic_terminal' # Explicitly set type
                node_attrs_sprout.pop('current_territory_voxel_indices_flat', None) # Not a graph attribute
                node_attrs_sprout.pop('current_territory_demand', None) # Not a graph attribute
                data_structures.add_node_to_graph(gbo_graph, term_gbo_data.id, **node_attrs_sprout)

                data_structures.add_edge_to_graph(gbo_graph, node_id, term_gbo_data.id, length=0.0, type='synthetic_sprout')
                if gbo_graph.has_node(node_id): gbo_graph.nodes[node_id]['type'] = 'measured_bifurcation_or_segment'
                
                logger.info(f"Initialized synthetic terminal {term_gbo_data.id} from measured {node_id} (orig_R={measured_radius_at_terminal:.4f}).")
                next_synthetic_node_id += 1
                processed_from_initial_graph = True
        if not processed_from_initial_graph and initial_graph.number_of_nodes() > 0:
            logger.warning("Initial graph provided but no 'measured_terminal' nodes found. Checking for config seed points.")

    if not processed_from_initial_graph:
        seed_points_config = config_manager.get_param(config, "gbo_growth.seed_points", [])
        if seed_points_config and isinstance(seed_points_config, list):
            logger.info(f"No measured terminals processed. Using {len(seed_points_config)} seed points from configuration.")
            for seed_info in seed_points_config:
                seed_id_base = seed_info.get('id', f"cfg_seed_{next_synthetic_node_id}")
                seed_pos = np.array(seed_info.get('position'), dtype=float)
                seed_initial_radius = float(seed_info.get('initial_radius', initial_synthetic_radius_default))
                seed_initial_flow = (seed_initial_radius / k_murray_factor) ** murray_exponent \
                                    if seed_initial_radius > constants.EPSILON and k_murray_factor > constants.EPSILON \
                                    else default_initial_flow
                term_gbo_data = GBOIterationData(
                    terminal_id=seed_id_base, pos=seed_pos, radius=seed_initial_radius, flow=seed_initial_flow
                )
                active_terminals.append(term_gbo_data)
                node_attributes_seed = vars(term_gbo_data).copy()
                node_attributes_seed['is_flow_root'] = True 
                node_attributes_seed['type'] = 'synthetic_root_terminal' # Distinguish from sprouts
                node_attributes_seed.pop('current_territory_voxel_indices_flat', None)
                node_attributes_seed.pop('current_territory_demand', None)
                data_structures.add_node_to_graph(gbo_graph, term_gbo_data.id, **node_attributes_seed)
                logger.info(f"Initialized synthetic terminal from config seed: {term_gbo_data.id} at {np.round(seed_pos,2)} "
                            f"with R={seed_initial_radius:.4f}, initial Q={seed_initial_flow:.2e}")
                next_synthetic_node_id += 1
            # processed_from_initial_graph = True # Not needed to set here
        elif seed_points_config : # Not a list or empty
             logger.warning("`gbo_growth.seed_points` in config is not a valid list or is empty. Checking fallback.")


    if not active_terminals: # Fallback seed
        logger.warning("No starting points from graph or config seeds. Attempting one fallback seed.")
        if tissue_data.get('domain_mask') is not None and np.any(tissue_data['domain_mask']):
            seed_mask = tissue_data.get('GM', tissue_data.get('WM', tissue_data['domain_mask']))
            valid_seed_points_vox = np.array(np.where(seed_mask)).T
            if valid_seed_points_vox.shape[0] > 0:
                seed_point_vox = valid_seed_points_vox[np.random.choice(valid_seed_points_vox.shape[0])]
                seed_point_world = utils.voxel_to_world(seed_point_vox.reshape(1,-1), tissue_data['affine'])[0]
                term_gbo_data = GBOIterationData(
                    terminal_id=f"s_fallback_{next_synthetic_node_id}", pos=seed_point_world, 
                    radius=initial_synthetic_radius_default, flow=default_initial_flow
                )
                active_terminals.append(term_gbo_data)
                node_attrs_fallback = vars(term_gbo_data).copy()
                node_attrs_fallback['is_flow_root'] = True
                node_attrs_fallback['type'] = 'synthetic_root_terminal'
                node_attrs_fallback.pop('current_territory_voxel_indices_flat', None)
                node_attrs_fallback.pop('current_territory_demand', None)
                data_structures.add_node_to_graph(gbo_graph, term_gbo_data.id, **node_attrs_fallback)
                logger.info(f"Initialized fallback seed terminal {term_gbo_data.id} at {np.round(seed_point_world,2)}.")
                next_synthetic_node_id +=1
            else: logger.error("Cannot find a valid seed point within domain_mask/GM/WM for fallback.")
        else: logger.error("No domain_mask available for fallback seeding.")

    if not active_terminals:
        logger.error("CRITICAL: No terminals to initialize growth. Aborting.")
        return perfused_tissue_mask, [], 0, gbo_graph

    if tissue_data['world_coords_flat'].shape[0] == 0:
        logger.error("tissue_data['world_coords_flat'] is empty. Cannot initialize Ω_init.")
        return perfused_tissue_mask, active_terminals, next_synthetic_node_id, gbo_graph
        
    kdtree_all_domain_voxels = KDTree(tissue_data['world_coords_flat'])
    initial_territory_radius_search = config_manager.get_param(config, "gbo_growth.initial_territory_radius", 0.2) 

    for term_data in active_terminals:
        nearby_domain_voxel_flat_indices = kdtree_all_domain_voxels.query_ball_point(
            term_data.pos, r=initial_territory_radius_search
        )
        actual_demand_in_init_territory = 0.0
        voxels_for_this_terminal_init_flat: List[int] = []
        for v_idx_flat in nearby_domain_voxel_flat_indices: # v_idx_flat is index into tissue_data['voxel_indices_flat']
            v_3d_idx_tuple = tuple(tissue_data['voxel_indices_flat'][v_idx_flat])
            if not perfused_tissue_mask[v_3d_idx_tuple]: 
                perfused_tissue_mask[v_3d_idx_tuple] = True
                actual_demand_in_init_territory += tissue_data['metabolic_demand_map'][v_3d_idx_tuple] * tissue_data['voxel_volume']
                voxels_for_this_terminal_init_flat.append(v_idx_flat)
        
        term_data.current_territory_voxel_indices_flat = voxels_for_this_terminal_init_flat
        term_data.current_territory_demand = actual_demand_in_init_territory
        
        current_radius_before_omega_init_update = term_data.radius 
        if actual_demand_in_init_territory > constants.EPSILON:
            term_data.flow = actual_demand_in_init_territory
            new_r = k_murray_factor * (term_data.flow ** (1.0 / murray_exponent))
            term_data.radius = max(current_radius_before_omega_init_update, new_r, initial_synthetic_radius_default) 
        # If no demand, flow and radius remain as initialized
            
        if gbo_graph.has_node(term_data.id):
            gbo_graph.nodes[term_data.id]['Q_flow'] = term_data.flow
            gbo_graph.nodes[term_data.id]['radius'] = term_data.radius
        
        if term_data.current_territory_voxel_indices_flat:
            # Ensure indices are valid for world_coords_flat
            valid_indices_for_centroid = [idx for idx in term_data.current_territory_voxel_indices_flat 
                                          if idx < tissue_data['world_coords_flat'].shape[0]]
            if valid_indices_for_centroid:
                initial_territory_coords = tissue_data['world_coords_flat'][valid_indices_for_centroid]
                if initial_territory_coords.shape[0] > 0:
                    new_initial_pos = np.mean(initial_territory_coords, axis=0)
                    old_pos_for_log = term_data.pos.copy()
                    term_data.pos = new_initial_pos
                    logger.debug(f"Terminal {term_data.id} (Ω_init): Moved from {np.round(old_pos_for_log,3)} to centroid {np.round(new_initial_pos,3)}.")
                    if gbo_graph.has_node(term_data.id):
                        gbo_graph.nodes[term_data.id]['pos'] = term_data.pos
                        if term_data.parent_id and gbo_graph.has_edge(term_data.parent_id, term_data.id):
                            parent_node_pos = gbo_graph.nodes[term_data.parent_id]['pos']
                            new_len = utils.distance(parent_node_pos, term_data.pos)
                            gbo_graph.edges[term_data.parent_id, term_data.id]['length'] = new_len
                            term_data.length_from_parent = new_len
            else:
                logger.warning(f"Terminal {term_data.id} claimed territory voxels but could not get coordinates for centroid calculation.")

        logger.debug(f"Terminal {term_data.id} (Ω_init final): Pos={np.round(term_data.pos,3)}, Claimed {len(voxels_for_this_terminal_init_flat)} voxels, "
                     f"Demand={term_data.current_territory_demand:.2e}, Flow={term_data.flow:.2e}, Radius={term_data.radius:.4f}")

    logger.info(f"Initialization complete. Perfused {np.sum(perfused_tissue_mask)} initial voxels. "
                f"{len(active_terminals)} active terminals.")
    return perfused_tissue_mask, active_terminals, next_synthetic_node_id, gbo_graph


def find_growth_frontier_voxels(
    terminal_gbo_data: GBOIterationData,
    kdtree_unperfused_domain_voxels: Optional[KDTree], 
    unperfused_global_flat_indices: np.ndarray,
    tissue_data: dict, 
    config: dict
) -> np.ndarray:
    logger.debug(f"Terminal {terminal_gbo_data.id}: Entering find_growth_frontier_voxels. "
                 f"Pos: {np.round(terminal_gbo_data.pos,3)}, Radius: {terminal_gbo_data.radius:.4f}")

    if kdtree_unperfused_domain_voxels is None or kdtree_unperfused_domain_voxels.n == 0:
        logger.debug(f"Terminal {terminal_gbo_data.id}: KDTree of unperfused voxels is empty or None. No frontier.")
        return np.array([], dtype=int)

    radius_factor = config_manager.get_param(config, "gbo_growth.frontier_search_radius_factor", 3.0)
    fixed_radius = config_manager.get_param(config, "gbo_growth.frontier_search_radius_fixed", 0.25)
    voxel_dim = tissue_data['voxel_volume']**(1/3.0)
    search_r = max(radius_factor * terminal_gbo_data.radius, fixed_radius, voxel_dim * 1.5)
    
    logger.debug(f"Terminal {terminal_gbo_data.id}: Searching for frontier with effective radius {search_r:.3f}mm. "
                 f"(voxel_dim ~{voxel_dim:.3f}, term_radius_component ~{radius_factor * terminal_gbo_data.radius:.3f})")

    try:
        local_indices_in_kdtree = kdtree_unperfused_domain_voxels.query_ball_point(terminal_gbo_data.pos, r=search_r)
    except Exception as e:
        logger.error(f"Terminal {terminal_gbo_data.id}: KDTree query_ball_point failed: {e}", exc_info=True)
        return np.array([], dtype=int)
        
    logger.debug(f"Terminal {terminal_gbo_data.id}: KDTree query_ball_point found {len(local_indices_in_kdtree)} local indices "
                 f"within {search_r:.3f}mm.")
    
    if not local_indices_in_kdtree: return np.array([], dtype=int)
    if unperfused_global_flat_indices.shape[0] == 0:
        logger.warning(f"Terminal {terminal_gbo_data.id}: unperfused_global_flat_indices is empty.")
        return np.array([], dtype=int)
    
    # Ensure all indices from kdtree are valid for unperfused_global_flat_indices
    valid_kdtree_indices = [idx for idx in local_indices_in_kdtree if idx < len(unperfused_global_flat_indices)]
    if len(valid_kdtree_indices) != len(local_indices_in_kdtree):
        logger.warning(f"Terminal {terminal_gbo_data.id}: Some KDTree indices were out of bounds for unperfused_global_flat_indices. "
                       f"Original: {len(local_indices_in_kdtree)}, Valid: {len(valid_kdtree_indices)}")
    if not valid_kdtree_indices: return np.array([], dtype=int)

    frontier_voxels_global_flat_indices_initial = unperfused_global_flat_indices[valid_kdtree_indices]
    logger.debug(f"Terminal {terminal_gbo_data.id}: Initial frontier (after mapping) "
                 f"contains {len(frontier_voxels_global_flat_indices_initial)} voxels.")

    max_voxels_in_Rip = config_manager.get_param(config, "gbo_growth.max_voxels_for_Rip", 50)
    final_frontier_voxels_global_flat_indices = frontier_voxels_global_flat_indices_initial

    if len(frontier_voxels_global_flat_indices_initial) > max_voxels_in_Rip:
        logger.debug(f"Terminal {terminal_gbo_data.id}: Initial frontier ({len(frontier_voxels_global_flat_indices_initial)}) "
                     f"> max_voxels_for_Rip ({max_voxels_in_Rip}). Selecting closest.")
        try:
            k_val = min(max_voxels_in_Rip, kdtree_unperfused_domain_voxels.n)
            if k_val > 0 :
                _, local_indices_k_closest = kdtree_unperfused_domain_voxels.query(terminal_gbo_data.pos, k=k_val)
                if isinstance(local_indices_k_closest, (int, np.integer)): 
                    local_indices_k_closest = np.array([local_indices_k_closest])
                
                if len(local_indices_k_closest) > 0:
                    valid_k_closest_indices = [idx for idx in local_indices_k_closest if idx < len(unperfused_global_flat_indices)]
                    if len(valid_k_closest_indices) != len(local_indices_k_closest):
                         logger.warning(f"Terminal {terminal_gbo_data.id}: Some k-closest KDTree indices out of bounds.")
                    if valid_k_closest_indices:
                        final_frontier_voxels_global_flat_indices = unperfused_global_flat_indices[valid_k_closest_indices]
                    else: final_frontier_voxels_global_flat_indices = np.array([], dtype=int)
                else: final_frontier_voxels_global_flat_indices = np.array([], dtype=int)
            else: final_frontier_voxels_global_flat_indices = np.array([], dtype=int)
        except Exception as e_kquery:
            logger.error(f"Terminal {terminal_gbo_data.id}: KDTree k-closest query failed: {e_kquery}. Using initial ball query if valid.", exc_info=True)
            if np.any(np.array(valid_kdtree_indices) >= len(unperfused_global_flat_indices)): # Check original indices used for initial array
                 final_frontier_voxels_global_flat_indices = np.array([], dtype=int)
        logger.debug(f"Terminal {terminal_gbo_data.id}: Frontier limited to {len(final_frontier_voxels_global_flat_indices)} voxels.")

    logger.info(f"Terminal {terminal_gbo_data.id} identified {len(final_frontier_voxels_global_flat_indices)} "
                 f"final frontier voxels (Ri,p).")
    return final_frontier_voxels_global_flat_indices


def grow_healthy_vasculature(config: dict,
                             tissue_data: dict,
                             initial_graph: Optional[nx.DiGraph],
                             output_dir: str) -> Optional[nx.DiGraph]:
    logger.info("Starting GBO healthy vascular growth (Tissue-Led Model v2 with Flow Solver Integration)...")

    # --- 1. Initialization ---
    perfused_tissue_mask, current_active_terminals, next_node_id, gbo_graph = \
        initialize_perfused_territory_and_terminals(config, initial_graph, tissue_data)

    if not current_active_terminals: 
        logger.error("GBO Aborted: No active terminals after initialization.")
        return gbo_graph 

    # --- Load parameters used throughout the main loop ---
    max_iterations = config_manager.get_param(config, "gbo_growth.max_iterations", 100)
    min_radius = config_manager.get_param(config, "vascular_properties.min_radius", constants.MIN_VESSEL_RADIUS_MM)
    k_murray = config_manager.get_param(config, "vascular_properties.k_murray_scaling_factor", 0.5)
    murray_exp = config_manager.get_param(config, "vascular_properties.murray_law_exponent", 3.0)
    branch_radius_factor_thresh = config_manager.get_param(config, "gbo_growth.branch_radius_increase_threshold", 1.1)
    max_flow_single_term = config_manager.get_param(config, "gbo_growth.max_flow_single_terminal", 0.005)
    min_iters_no_growth_stop = config_manager.get_param(config, "gbo_growth.min_iterations_before_no_growth_stop", 10)
    min_demand_rip_bif_factor = config_manager.get_param(config, "gbo_growth.min_frontier_demand_factor_for_bifurcation", 0.3)
    default_initial_flow = config_manager.get_param(config, "vascular_properties.initial_terminal_flow", constants.INITIAL_TERMINAL_FLOW_Q)
    
    # Flow solver integration parameters
    flow_solver_interval = config_manager.get_param(config, "gbo_growth.flow_solver_interval", 1)

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
            logger.info("GBO: No active terminals for growth at start of iteration.")
            break 
        
        current_perfused_count = np.sum(perfused_tissue_mask)
        logger.info(f"Active terminals: {len(terminals_for_growth_attempt)}. Perfused voxels: {current_perfused_count}/{total_voxels_in_domain} "
                    f"({(current_perfused_count/total_voxels_in_domain)*100:.1f}%)")

        unperfused_mask_3d = tissue_data['domain_mask'] & (~perfused_tissue_mask)
        unperfused_voxels_3d_indices = np.array(np.where(unperfused_mask_3d)).T
        kdtree_unperfused: Optional[KDTree] = None
        unperfused_global_flat_indices_for_kdtree: np.ndarray = np.array([], dtype=int)

        if unperfused_voxels_3d_indices.shape[0] > 0:
            unperfused_voxels_world_coords_kdt = utils.voxel_to_world(unperfused_voxels_3d_indices, tissue_data['affine'])
            flat_indices_temp = map_3d_to_flat_idx[unperfused_voxels_3d_indices[:,0],
                                                   unperfused_voxels_3d_indices[:,1],
                                                   unperfused_voxels_3d_indices[:,2]]
            valid_mask_kdt = flat_indices_temp != -1
            unperfused_global_flat_indices_for_kdtree = flat_indices_temp[valid_mask_kdt]
            if np.any(~valid_mask_kdt): 
                unperfused_voxels_world_coords_kdt = unperfused_voxels_world_coords_kdt[valid_mask_kdt]
            if unperfused_voxels_world_coords_kdt.shape[0] > 0:
                kdtree_unperfused = KDTree(unperfused_voxels_world_coords_kdt)
            else: logger.debug("No valid unperfused world coordinates to build KDTree.")
        
        if kdtree_unperfused is None or kdtree_unperfused.n == 0:
            logger.info("GBO: No unperfused domain voxels for KDTree. Target likely achieved this iteration.")
            # No break here yet, allow Voronoi, flow solve, and stop conditions to run
            # break # Old: break here

        next_iter_terminals_manager: List[GBOIterationData] = []
        newly_perfused_in_iter_mask = np.zeros_like(perfused_tissue_mask)

        # --- Growth/Branching/Extension Phase for each terminal ---
        for term_p_gbo_data in terminals_for_growth_attempt:
            logger.debug(f"Processing terminal {term_p_gbo_data.id}. Pos: {np.round(term_p_gbo_data.pos,3)}, "
                         f"Radius: {term_p_gbo_data.radius:.4f}, Current Flow (demand): {term_p_gbo_data.flow:.2e}")
            
            frontier_voxels_global_flat_indices = find_growth_frontier_voxels(
                term_p_gbo_data, kdtree_unperfused, unperfused_global_flat_indices_for_kdtree,
                tissue_data, config
            )

            if len(frontier_voxels_global_flat_indices) == 0:
                next_iter_terminals_manager.append(term_p_gbo_data); continue

            demand_map_3d_indices_frontier = tissue_data['voxel_indices_flat'][frontier_voxels_global_flat_indices]
            demand_of_frontier_voxels = tissue_data['metabolic_demand_map'][
                demand_map_3d_indices_frontier[:,0], demand_map_3d_indices_frontier[:,1],
                demand_map_3d_indices_frontier[:,2]] * tissue_data['voxel_volume']
            demand_Rip = np.sum(demand_of_frontier_voxels)

            if demand_Rip < constants.EPSILON:
                next_iter_terminals_manager.append(term_p_gbo_data); continue
            
            potential_total_flow_if_extended = term_p_gbo_data.flow + demand_Rip 
            potential_radius_if_extended = k_murray * (potential_total_flow_if_extended ** (1.0 / murray_exp))
            attempt_branching = False
            if term_p_gbo_data.radius > constants.EPSILON and \
               potential_radius_if_extended > term_p_gbo_data.radius * branch_radius_factor_thresh : attempt_branching = True
            if potential_total_flow_if_extended > max_flow_single_term: attempt_branching = True
            if demand_Rip > term_p_gbo_data.flow * min_demand_rip_bif_factor and term_p_gbo_data.flow > constants.EPSILON: attempt_branching = True

            if attempt_branching and len(frontier_voxels_global_flat_indices) > 0: # Need some new region
                logger.debug(f"Terminal {term_p_gbo_data.id} evaluating branching. New frontier demand: {demand_Rip:.2e}.")
                old_territory_indices = np.array(term_p_gbo_data.current_territory_voxel_indices_flat, dtype=int)
                combined_territory_indices = np.unique(np.concatenate((old_territory_indices, frontier_voxels_global_flat_indices))) \
                                             if len(old_territory_indices) > 0 else np.unique(frontier_voxels_global_flat_indices)

                if len(combined_territory_indices) < 2: # Min 2 voxels to split between children
                    logger.debug(f"Combined territory for {term_p_gbo_data.id} too small ({len(combined_territory_indices)} voxels). Attempting extension.")
                    attempt_branching = False
                else:
                    bifurcation_result = energy_model.find_optimal_bifurcation_for_combined_territory(
                        term_p_gbo_data, combined_territory_indices, tissue_data, config, k_murray, murray_exp
                    )
                    if bifurcation_result:
                        c1_pos, c1_rad, c1_total_flow, c2_pos, c2_rad, c2_total_flow, _ = bifurcation_result
                        new_parent_total_flow = c1_total_flow + c2_total_flow
                        new_parent_radius = max(min_radius, k_murray * (new_parent_total_flow ** (1.0 / murray_exp)))
                        
                        gbo_graph.nodes[term_p_gbo_data.id].update(type='synthetic_bifurcation', radius=new_parent_radius, Q_flow=new_parent_total_flow)
                        
                        for i_child, (child_pos, child_rad, child_flow) in enumerate([(c1_pos, c1_rad, c1_total_flow), (c2_pos, c2_rad, c2_total_flow)]):
                            child_id = f"s_{next_node_id}"
                            next_node_id += 1
                            child_gbo = GBOIterationData(
                                child_id, child_pos, child_rad, child_flow,
                                original_measured_radius=term_p_gbo_data.original_measured_terminal_radius,
                                parent_id=term_p_gbo_data.id, parent_measured_terminal_id=term_p_gbo_data.parent_measured_terminal_id
                            )
                            child_gbo.length_from_parent = utils.distance(term_p_gbo_data.pos, child_pos)
                            next_iter_terminals_manager.append(child_gbo)
                            # Use **vars(child_gbo) to pass all attributes to add_node_to_graph
                            data_structures.add_node_to_graph(gbo_graph, child_gbo.id, **vars(child_gbo),type='synthetic_terminal', is_flow_root=True)
                            data_structures.add_edge_to_graph(gbo_graph, term_p_gbo_data.id, child_gbo.id, 
                                                              length=child_gbo.length_from_parent, radius=new_parent_radius, # Edge radius from parent
                                                              type='synthetic_segment')
                        
                        for v_idx_flat in frontier_voxels_global_flat_indices:
                            v_3d_idx = tuple(tissue_data['voxel_indices_flat'][v_idx_flat])
                            newly_perfused_in_iter_mask[v_3d_idx] = True
                        logger.info(f"Terminal {term_p_gbo_data.id} branched. Children flows: {c1_total_flow:.2e}, {c2_total_flow:.2e}.")
                    else: attempt_branching = False
            
            if not attempt_branching: 
                old_pos_ext = term_p_gbo_data.pos.copy()
                logger.debug(f"Terminal {term_p_gbo_data.id} extending for Ri,p (demand {demand_Rip:.2e}).")
                term_p_gbo_data.flow += demand_Rip 
                term_p_gbo_data.radius = max(min_radius, k_murray * (term_p_gbo_data.flow ** (1.0 / murray_exp)))
                
                if len(frontier_voxels_global_flat_indices) > 0:
                    newly_acquired_coords = tissue_data['world_coords_flat'][frontier_voxels_global_flat_indices]
                    all_coords_for_centroid_list = [newly_acquired_coords] # Start with new
                    if term_p_gbo_data.current_territory_voxel_indices_flat:
                        valid_curr_idx = [idx for idx in term_p_gbo_data.current_territory_voxel_indices_flat 
                                          if idx < tissue_data['world_coords_flat'].shape[0]]
                        if valid_curr_idx: 
                            all_coords_for_centroid_list.insert(0, tissue_data['world_coords_flat'][valid_curr_idx])
                    
                    all_supplied_coords = np.vstack(all_coords_for_centroid_list)
                    if all_supplied_coords.shape[0] > 0:
                        new_target_pos = np.mean(all_supplied_coords, axis=0)
                        extension_vector = new_target_pos - old_pos_ext
                        extension_length = np.linalg.norm(extension_vector)
                        max_seg_len = config_manager.get_param(config, "vascular_properties.max_segment_length", 2.0)

                        if extension_length > constants.EPSILON:
                            move_dist = min(extension_length, max_seg_len)
                            term_p_gbo_data.pos = old_pos_ext + extension_vector * (move_dist / extension_length)
                            logger.debug(f"Terminal {term_p_gbo_data.id} moved from {np.round(old_pos_ext,3)} to {np.round(term_p_gbo_data.pos,3)}")
                            if gbo_graph.has_node(term_p_gbo_data.id) and term_p_gbo_data.parent_id and \
                               gbo_graph.has_edge(term_p_gbo_data.parent_id, term_p_gbo_data.id):
                                parent_pos = gbo_graph.nodes[term_p_gbo_data.parent_id]['pos']
                                new_len = utils.distance(parent_pos, term_p_gbo_data.pos)
                                gbo_graph.edges[term_p_gbo_data.parent_id, term_p_gbo_data.id]['length'] = new_len
                                gbo_graph.edges[term_p_gbo_data.parent_id, term_p_gbo_data.id]['radius'] = gbo_graph.nodes[term_p_gbo_data.parent_id]['radius'] # Update edge radius
                                term_p_gbo_data.length_from_parent = new_len
                        else: logger.debug(f"Terminal {term_p_gbo_data.id} extension target is current pos.")
                
                if gbo_graph.has_node(term_p_gbo_data.id):
                    gbo_graph.nodes[term_p_gbo_data.id].update(pos=term_p_gbo_data.pos, Q_flow=term_p_gbo_data.flow, radius=term_p_gbo_data.radius)
                for v_idx_flat in frontier_voxels_global_flat_indices:
                    v_3d_idx = tuple(tissue_data['voxel_indices_flat'][v_idx_flat])
                    newly_perfused_in_iter_mask[v_3d_idx] = True
                term_p_gbo_data.current_territory_voxel_indices_flat.extend(list(frontier_voxels_global_flat_indices))
                next_iter_terminals_manager.append(term_p_gbo_data)

        current_active_terminals = next_iter_terminals_manager
        perfused_tissue_mask = perfused_tissue_mask | newly_perfused_in_iter_mask
        num_newly_perfused_this_iter = np.sum(newly_perfused_in_iter_mask)
        logger.info(f"Perfused {num_newly_perfused_this_iter} new voxels in this iteration's growth phase.")
        
        # --- Global Adaptation Phase ---
        if current_active_terminals and np.any(perfused_tissue_mask):
            live_terminals_for_adaptation = [t for t in current_active_terminals if not t.stop_growth]
            if live_terminals_for_adaptation:
                # 1. Voronoi Refinement (updates GBOIterationData[t].flow to current_territory_demand)
                perfused_3d_indices = np.array(np.where(perfused_tissue_mask)).T
                perfused_global_flat_indices = map_3d_to_flat_idx[perfused_3d_indices[:,0], perfused_3d_indices[:,1], perfused_3d_indices[:,2]]
                valid_flat_mask_for_perf = perfused_global_flat_indices != -1
                perfused_global_flat_indices = perfused_global_flat_indices[valid_flat_mask_for_perf]
                if perfused_global_flat_indices.shape[0] > 0 :
                    perfused_world_coords_for_voronoi = tissue_data['world_coords_flat'][perfused_global_flat_indices]
                    term_positions_vor = np.array([t.pos for t in live_terminals_for_adaptation])
                    term_flows_vor = np.array([t.flow if t.flow > constants.EPSILON else default_initial_flow for t in live_terminals_for_adaptation])
                    assigned_local_term_indices = np.full(perfused_world_coords_for_voronoi.shape[0], -1, dtype=int)
                    for i_pvox, p_vox_wc in enumerate(perfused_world_coords_for_voronoi):
                        distances_sq = np.sum((term_positions_vor - p_vox_wc)**2, axis=1)
                        weighted_distances = distances_sq / term_flows_vor # Weight by current flow capacity
                        assigned_local_term_indices[i_pvox] = np.argmin(weighted_distances)
                    for t_data in live_terminals_for_adaptation: t_data.current_territory_voxel_indices_flat, t_data.current_territory_demand = [], 0.0
                    for i_pvox, local_term_idx in enumerate(assigned_local_term_indices):
                        if local_term_idx != -1:
                            term_obj = live_terminals_for_adaptation[local_term_idx]
                            global_flat_v_idx = perfused_global_flat_indices[i_pvox]
                            term_obj.current_territory_voxel_indices_flat.append(global_flat_v_idx)
                            v_3d_idx_for_demand = tuple(tissue_data['voxel_indices_flat'][global_flat_v_idx])
                            term_obj.current_territory_demand += tissue_data['metabolic_demand_map'][v_3d_idx_for_demand] * tissue_data['voxel_volume']
                    for t_data in live_terminals_for_adaptation: # Update GBOIterationData flow based on new territory demand
                        t_data.flow = t_data.current_territory_demand if t_data.current_territory_demand > constants.EPSILON else default_initial_flow
                        # Radii will be updated after flow solve, or if no flow solve this iter, update now
                        if not ((iteration + 1) % flow_solver_interval == 0 or iteration == max_iterations - 1):
                            new_r = k_murray * (t_data.flow ** (1.0 / murray_exp))
                            t_data.radius = max(min_radius, new_r)
                            if gbo_graph.has_node(t_data.id): gbo_graph.nodes[t_data.id].update(Q_flow=t_data.flow, radius=t_data.radius)
                        logger.debug(f"Term {t_data.id} (Voronoi Refined): Target Q_demand={t_data.flow:.2e}, Current R_demand_based={t_data.radius:.4f}")
                logger.info("Completed Voronoi refinement for active terminals.")

                # 2. Solve Network Flow (conditionally)
                run_flow_solver_this_iteration = ((iteration + 1) % flow_solver_interval == 0) or \
                                                 (iteration == max_iterations - 1)
                if run_flow_solver_this_iteration:
                    logger.info(f"Running 1D network flow solver for iteration {iteration + 1}...")
                    for term_obj in live_terminals_for_adaptation: # Ensure graph nodes have latest target Q_flow for terminals
                        if gbo_graph.has_node(term_obj.id): gbo_graph.nodes[term_obj.id]['Q_flow'] = term_obj.flow
                    
                    temp_graph_for_solver = gbo_graph.copy() # Solver might modify in place, or we update from its return
                    gbo_graph_with_flow = perfusion_solver.solve_1d_poiseuille_flow(temp_graph_for_solver, config)
                    
                    if gbo_graph_with_flow:
                        gbo_graph = gbo_graph_with_flow
                        logger.info("Flow solution obtained. Starting global radius adaptation...")
                        synthetic_nodes_to_adapt = [
                            n for n, data in gbo_graph.nodes(data=True)
                            if data.get('is_synthetic', False) or data.get('type', '').startswith('synthetic_')
                        ]
                        for node_id_adapt in synthetic_nodes_to_adapt:
                            node_data_adapt = gbo_graph.nodes[node_id_adapt]
                            actual_node_flow = 0.0
                            
                            # --- FETCH THE ORIGINAL RADIUS AT THE START OF THE LOOP ITERATION ---
                            # This ensures it's defined before any conditional blocks that might use it.
                            original_radius_before_adapt = node_data_adapt.get('radius', min_radius) 
                            # --- END OF FETCH ---

                            node_type_for_adapt = node_data_adapt.get('type')

                            if node_type_for_adapt == 'synthetic_terminal':
                                actual_node_flow = node_data_adapt.get('Q_flow', 0.0) 
                                # Optional: Refine with incoming solved flow if available and reliable
                                for u_edge_in, _, edge_data_in in gbo_graph.in_edges(node_id_adapt, data=True):
                                    solved_in_flow = edge_data_in.get('flow_solver')
                                    if solved_in_flow is not None and np.isfinite(solved_in_flow):
                                        actual_node_flow = solved_in_flow 
                                        break 
                            
                            elif node_type_for_adapt == 'synthetic_bifurcation':
                                for _, _, edge_data_out in gbo_graph.out_edges(node_id_adapt, data=True):
                                    flow_on_edge = edge_data_out.get('flow_solver', 0.0)
                                    if np.isnan(flow_on_edge): 
                                        logger.warning(f"GlobalAdapt: Edge from {node_id_adapt} has NaN flow. Treating as 0 for sum.")
                                        flow_on_edge = 0.0 
                                    actual_node_flow += flow_on_edge
                                
                                # Calculate potential new radius just for the debug log message
                                potential_new_r_debug = min_radius 
                                if abs(actual_node_flow) > constants.EPSILON:
                                    potential_new_r_debug = max(min_radius, k_murray * (abs(actual_node_flow) ** (1.0 / murray_exp)))
                                
                                logger.info(f"MurrayDebug - Bifurcation {node_id_adapt}: Children solved flows sum to actual_node_flow={actual_node_flow:.3e}. Original R={original_radius_before_adapt:.4f}. New R will be {potential_new_r_debug:.4f}")
                            else: 
                                # logger.debug(f"GlobalAdapt: Skipping node {node_id_adapt} of type {node_type_for_adapt}")
                                continue # Skip other types for now for radius adaptation
                                
                            # --- Actual Radius Update Logic ---
                            if abs(actual_node_flow) > constants.EPSILON:
                                new_radius_adapted = k_murray * (abs(actual_node_flow) ** (1.0 / murray_exp))
                                new_radius_adapted = max(min_radius, new_radius_adapted)
                                
                                if not np.isclose(original_radius_before_adapt, new_radius_adapted, rtol=1e-2, atol=1e-5):
                                    logger.info(f"GlobalAdapt: Node {node_id_adapt} (type: {node_type_for_adapt}) R: {original_radius_before_adapt:.4f} -> {new_radius_adapted:.4f} (Q_actual={actual_node_flow:.2e})")
                                    node_data_adapt['radius'] = new_radius_adapted
                                    # Also update corresponding GBOIterationData object if it's an active terminal
                                    for term_obj_adapt in live_terminals_for_adaptation: # live_terminals_for_adaptation should be in scope
                                        if term_obj_adapt.id == node_id_adapt:
                                            term_obj_adapt.radius = new_radius_adapted
                                            term_obj_adapt.flow = actual_node_flow 
                                            break
                            elif original_radius_before_adapt > min_radius + constants.EPSILON : 
                                logger.info(f"GlobalAdapt: Node {node_id_adapt} (type: {node_type_for_adapt}) Q_actual near zero or NaN. Shrinking R from {original_radius_before_adapt:.4f} to {min_radius:.4f}.")
                                node_data_adapt['radius'] = min_radius
                                # Also update corresponding GBOIterationData object if it's an active terminal
                                for term_obj_adapt in live_terminals_for_adaptation:
                                    if term_obj_adapt.id == node_id_adapt: 
                                        term_obj_adapt.radius = min_radius
                                        term_obj_adapt.flow = 0.0 
                                        break
                        logger.info("Global radius adaptation based on solved flows complete.")
                    else: logger.error("Flow solver did not return a graph. Skipping radius adaptation.")
                else: logger.info(f"Skipping flow solver and global radius adaptation for iteration {iteration + 1}.")
        
        # --- Update Stop Flags ---
        active_terminals_still_growing = 0
        for term_data in current_active_terminals:
            if term_data.stop_growth: continue
            if term_data.radius < min_radius + constants.EPSILON : term_data.stop_growth = True; logger.debug(f"T{term_data.id} stop:minR")
            if term_data.original_measured_terminal_radius is not None:
                stop_factor = config_manager.get_param(config, "gbo_growth.stop_criteria.max_radius_factor_measured", 1.0)
                if term_data.radius > term_data.original_measured_terminal_radius * stop_factor:
                    term_data.stop_growth = True; logger.info(f"T{term_data.id} stop:>measR_factor")
            if not term_data.current_territory_voxel_indices_flat and term_data.current_territory_demand < constants.EPSILON:
                term_data.stop_growth = True; logger.debug(f"T{term_data.id} stop:no_terr")
            if gbo_graph.has_node(term_data.id): gbo_graph.nodes[term_data.id]['stop_growth'] = term_data.stop_growth
            if not term_data.stop_growth: active_terminals_still_growing += 1

        # --- Intermediate Save & Global Stop Conditions ---
        save_this_iter = False
        stop_due_to_target_perfusion = (np.sum(perfused_tissue_mask) >= total_voxels_in_domain * 
                                        config_manager.get_param(config, "gbo_growth.target_domain_perfusion_fraction", 0.99))
        stop_due_to_no_active_terminals = (active_terminals_still_growing == 0 and iteration > 0)
        stop_due_to_no_new_growth = (num_newly_perfused_this_iter == 0 and iteration >= min_iters_no_growth_stop)

        if config_manager.get_param(config, "visualization.save_intermediate_steps", False):
            interval = config_manager.get_param(config, "visualization.intermediate_step_interval", 1)
            if ((iteration + 1) % interval == 0) or \
               (iteration == max_iterations - 1) or \
               stop_due_to_no_new_growth or stop_due_to_target_perfusion or stop_due_to_no_active_terminals:
                save_this_iter = True
            if save_this_iter:
                logger.info(f"Saving intermediate results for iteration {iteration + 1}...")
                io_utils.save_vascular_tree_vtp(gbo_graph, os.path.join(output_dir, f"gbo_graph_iter_{iteration+1}.vtp"))
                io_utils.save_nifti_image(perfused_tissue_mask.astype(np.uint8), tissue_data['affine'], 
                                          os.path.join(output_dir, f"perfused_mask_iter_{iteration+1}.nii.gz"))

        if stop_due_to_target_perfusion: logger.info(f"GBO Stopping after iteration {iteration + 1}: Target perfusion reached."); break
        if stop_due_to_no_active_terminals: logger.info(f"GBO Stopping after iteration {iteration + 1}: No active terminals."); break
        if stop_due_to_no_new_growth: logger.info(f"GBO Stopping after iteration {iteration + 1}: No new growth."); break
            
    logger.info(f"GBO healthy vascular growth (with flow solver) finished. Final tree: {gbo_graph.number_of_nodes()} nodes, {gbo_graph.number_of_edges()} edges.")
    return gbo_graph