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
from src import io_utils # Added missing import

logger = logging.getLogger(__name__)

class GBOIterationData:
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
    initial_graph: Optional[nx.DiGraph], # Parsed VTP graph from main.py
    tissue_data: dict # Contains various masks and affine
) -> Tuple[np.ndarray, List[GBOIterationData], int, nx.DiGraph]:
    logger.info("Initializing perfused territory and active GBO terminals...")
    perfused_tissue_mask = np.zeros(tissue_data['shape'], dtype=bool)
    active_terminals: List[GBOIterationData] = []

    initial_synthetic_radius_default = config_manager.get_param(config, "vascular_properties.min_radius", 0.005)
    k_murray_factor = config_manager.get_param(config, "vascular_properties.k_murray_scaling_factor", 0.5)
    murray_exponent = config_manager.get_param(config, "vascular_properties.murray_law_exponent", 3.0)
    default_initial_flow = config_manager.get_param(config, "vascular_properties.initial_terminal_flow", constants.INITIAL_TERMINAL_FLOW_Q)

    next_synthetic_node_id = 0

    gbo_graph = initial_graph.copy() if initial_graph and initial_graph.number_of_nodes() > 0 else data_structures.create_empty_vascular_graph()
    if initial_graph is None or initial_graph.number_of_nodes() == 0:
        logger.info("No initial VTP graph provided or it's empty. GBO will rely on config seeds or fallback.")

    # Mark VTP roots from input graph as flow roots for the solver
    if initial_graph:
        for node_id, data in initial_graph.nodes(data=True):
            if data.get('type') == 'measured_root':
                if gbo_graph.has_node(node_id):
                    gbo_graph.nodes[node_id]['is_flow_root'] = True
                    gbo_graph.nodes[node_id]['Q_flow'] = 0.0 # Pressure BC
                    logger.info(f"Marked VTP input node {node_id} (type: measured_root) as is_flow_root=True in gbo_graph.")

    processed_from_vtp_terminals = False
    if initial_graph:
        vtp_terminals_in_anatomical_domain = []
        # This is the GM+WM mask, where GBO growth is allowed
        gbo_growth_domain_mask = tissue_data.get('gbo_growth_domain_mask') 
        affine = tissue_data.get('affine')

        if gbo_growth_domain_mask is None or affine is None:
            logger.error("GBO growth domain mask or affine missing in tissue_data. Cannot reliably sprout from VTP terminals.")
        else:
            for node_id, data in initial_graph.nodes(data=True):
                # In main.py, VTP terminals within GM/WM/CSF were typed as 'measured_terminal_in_anatomical_domain'
                if data.get('type') == 'measured_terminal_in_anatomical_domain':
                    # Now, further check if this terminal is within the GBO growth domain (GM/WM)
                    pos_world = data['pos']
                    pos_vox_int = np.round(utils.world_to_voxel(pos_world, affine)).astype(int)

                    if utils.is_voxel_in_bounds(pos_vox_int, gbo_growth_domain_mask.shape) and \
                       gbo_growth_domain_mask[tuple(pos_vox_int)]:
                        vtp_terminals_in_anatomical_domain.append(node_id)
                    else:
                        logger.info(f"VTP terminal {node_id} (in anatomical domain) is outside GBO growth domain (GM/WM). Not sprouting GBO from it.")
                        # Optionally change its type in gbo_graph if it won't be a sprouting point
                        if gbo_graph.has_node(node_id):
                            gbo_graph.nodes[node_id]['type'] = 'measured_terminal_non_parenchymal' # e.g. ends in CSF

        if vtp_terminals_in_anatomical_domain:
            logger.info(f"Found {len(vtp_terminals_in_anatomical_domain)} VTP terminals within GBO growth domain (GM/WM) to sprout from.")
            for measured_terminal_id in vtp_terminals_in_anatomical_domain:
                measured_data = initial_graph.nodes[measured_terminal_id] # Data from original VTP graph
                measured_pos = np.array(measured_data['pos'], dtype=float)
                original_measured_radius = measured_data.get('radius', initial_synthetic_radius_default) 

                gbo_sprout_id = f"s_{next_synthetic_node_id}"; next_synthetic_node_id += 1

                term_gbo_data_obj = GBOIterationData(
                    terminal_id=gbo_sprout_id, pos=measured_pos,
                    radius=initial_synthetic_radius_default, flow=default_initial_flow,
                    original_measured_radius=original_measured_radius,
                    parent_id=measured_terminal_id,
                    parent_measured_terminal_id=measured_terminal_id
                )
                active_terminals.append(term_gbo_data_obj)

                node_attrs_for_s_node = vars(term_gbo_data_obj).copy()
                node_attrs_for_s_node['type'] = 'synthetic_terminal'
                node_attrs_for_s_node['is_flow_root'] = False
                node_attrs_for_s_node['Q_flow'] = term_gbo_data_obj.flow # Initial demand for solver
                node_attrs_for_s_node.pop('current_territory_voxel_indices_flat', None)
                node_attrs_for_s_node.pop('current_territory_demand', None)
                data_structures.add_node_to_graph(gbo_graph, gbo_sprout_id, **node_attrs_for_s_node)

                if gbo_graph.has_node(measured_terminal_id):
                     gbo_graph.nodes[measured_terminal_id]['type'] = 'measured_to_synthetic_junction'
                     if 'Q_flow' in gbo_graph.nodes[measured_terminal_id]: # Remove if it was typed as sink by main.py
                         del gbo_graph.nodes[measured_terminal_id]['Q_flow'] 
                
                edge_length_vtp_to_gbo = config_manager.get_param(config, "gbo_growth.vtp_sprout_connection_length", constants.EPSILON)
                data_structures.add_edge_to_graph(
                    gbo_graph, measured_terminal_id, gbo_sprout_id, 
                    length=edge_length_vtp_to_gbo, 
                    radius=initial_synthetic_radius_default, # Edge takes radius of the new GBO sprout
                    type='synthetic_sprout_from_measured'
                )
                logger.info(f"GBO Init: Sprouted synthetic_terminal '{gbo_sprout_id}' from VTP node '{measured_terminal_id}' "
                            f"(VTP R={original_measured_radius:.4f}, Sprout R_init={initial_synthetic_radius_default:.4f}).")
                processed_from_vtp_terminals = True
        
        if not processed_from_vtp_terminals and initial_graph.number_of_nodes() > 0 :
             if any(data.get('type') == 'measured_terminal_in_anatomical_domain' for _, data in initial_graph.nodes(data=True)):
                logger.warning("VTP terminals were found in anatomical domain, but none were within GBO growth domain (GM/WM). Checking config seeds.")
             else: # No 'measured_terminal_in_anatomical_domain' types were found at all
                logger.warning("Initial graph (VTP) provided but no 'measured_terminal_in_anatomical_domain' nodes found/processed by GBO init. Checking config seeds.")


    if not processed_from_vtp_terminals:
        seed_points_config = config_manager.get_param(config, "gbo_growth.seed_points", [])
        if seed_points_config and isinstance(seed_points_config, list):
            logger.info(f"No VTP terminals processed for GBO, or none valid. Using {len(seed_points_config)} GBO seed points from configuration.")
            for seed_info in seed_points_config:
                seed_id_base = seed_info.get('id', f"cfg_seed_{next_synthetic_node_id}")
                next_synthetic_node_id +=1

                seed_pos = np.array(seed_info.get('position'), dtype=float)
                seed_initial_radius = float(seed_info.get('initial_radius', initial_synthetic_radius_default))
                # Flow for config seeds is for radius calc, actual Q_flow on node for solver is 0 if is_flow_root
                seed_flow_for_radius = (seed_initial_radius / k_murray_factor) ** murray_exponent \
                                    if seed_initial_radius > constants.EPSILON and k_murray_factor > constants.EPSILON \
                                    else default_initial_flow
                term_gbo_data_obj = GBOIterationData(
                    terminal_id=seed_id_base, pos=seed_pos, radius=seed_initial_radius, flow=seed_flow_for_radius
                )
                active_terminals.append(term_gbo_data_obj)

                node_attrs_seed = vars(term_gbo_data_obj).copy()
                node_attrs_seed['is_flow_root'] = True 
                node_attrs_seed['type'] = 'synthetic_root_terminal'
                node_attrs_seed['Q_flow'] = 0.0 # Pressure BC for solver
                node_attrs_seed.pop('current_territory_voxel_indices_flat', None)
                node_attrs_seed.pop('current_territory_demand', None)
                data_structures.add_node_to_graph(gbo_graph, term_gbo_data_obj.id, **node_attrs_seed)
                logger.info(f"Initialized GBO seed terminal from config: {term_gbo_data_obj.id} at {np.round(seed_pos,2)} "
                            f"with R={seed_initial_radius:.4f}. Marked as is_flow_root.")
        elif not processed_from_vtp_terminals: # Only log if no VTP terminals AND no config seeds
             logger.info("No VTP terminals processed and no GBO seed points found in configuration. Checking fallback.")

    if not active_terminals: # Fallback seed
        logger.warning("No GBO starting points from VTP or config seeds. Attempting one fallback seed.")
        # Ensure 'domain_mask' for fallback is gbo_growth_domain_mask (GM/WM)
        fallback_domain_mask = tissue_data.get('gbo_growth_domain_mask', tissue_data.get('domain_mask'))
        if fallback_domain_mask is not None and np.any(fallback_domain_mask):
            seed_point_world = utils.get_random_point_in_mask(fallback_domain_mask, tissue_data['affine'])
            if seed_point_world is not None:
                fallback_id = f"s_fallback_{next_synthetic_node_id}"; next_synthetic_node_id +=1
                term_gbo_data_obj = GBOIterationData(fallback_id, seed_point_world, initial_synthetic_radius_default, default_initial_flow)
                active_terminals.append(term_gbo_data_obj)
                node_attrs_fallback = vars(term_gbo_data_obj).copy()
                node_attrs_fallback['is_flow_root'] = True; node_attrs_fallback['type'] = 'synthetic_root_terminal'
                node_attrs_fallback['Q_flow'] = 0.0 # Pressure BC
                node_attrs_fallback.pop('current_territory_voxel_indices_flat', None); node_attrs_fallback.pop('current_territory_demand', None)
                data_structures.add_node_to_graph(gbo_graph, fallback_id, **node_attrs_fallback)
                logger.info(f"Initialized fallback GBO seed terminal {fallback_id} at {np.round(seed_point_world,2)}.")
            else: logger.error("Cannot find a valid random seed point within GBO growth domain for fallback.")
        else: logger.error("No GBO growth domain_mask available for fallback seeding.")

    if not active_terminals:
        logger.error("CRITICAL: No GBO terminals to initialize growth. Aborting GBO.")
        return perfused_tissue_mask, [], next_synthetic_node_id, gbo_graph

    # Initial Territory Assignment uses 'world_coords_flat' derived from 'gbo_growth_domain_mask'
    if tissue_data.get('world_coords_flat') is None or tissue_data['world_coords_flat'].size == 0:
        logger.error("tissue_data['world_coords_flat'] (from GBO growth domain) is empty. Cannot initialize GBO territories.")
        for term_data in active_terminals: term_data.stop_growth = True
        return perfused_tissue_mask, active_terminals, next_synthetic_node_id, gbo_graph
        
    kdtree_gbo_domain_voxels = KDTree(tissue_data['world_coords_flat']) # KDTree of GM/WM voxels
    initial_territory_radius_search = config_manager.get_param(config, "gbo_growth.initial_territory_radius", 0.2) 

    for term_gbo_obj in active_terminals: # These are GBOIterationData objects (s_XXX)
        # Query against GM/WM kdtree
        nearby_flat_indices_in_gbo_domain = kdtree_gbo_domain_voxels.query_ball_point(term_gbo_obj.pos, r=initial_territory_radius_search)
        actual_demand_init = 0.0
        voxels_for_term_init_flat: List[int] = [] # These will be indices into tissue_data['world_coords_flat']

        if nearby_flat_indices_in_gbo_domain: # Check if any nearby voxels were found
            for local_kdtree_idx in nearby_flat_indices_in_gbo_domain:
                # The index from query_ball_point is an index into the points used to build kdtree_gbo_domain_voxels
                # which are tissue_data['world_coords_flat']. So, local_kdtree_idx IS the global_flat_idx.
                global_flat_idx = local_kdtree_idx
                
                v_3d_idx_tuple = tuple(tissue_data['voxel_indices_flat'][global_flat_idx]) # Get 3D index from global flat index
                
                if not perfused_tissue_mask[v_3d_idx_tuple]: 
                    perfused_tissue_mask[v_3d_idx_tuple] = True
                    actual_demand_init += tissue_data['metabolic_demand_map'][v_3d_idx_tuple] 
                    voxels_for_term_init_flat.append(global_flat_idx)
        
        term_gbo_obj.current_territory_voxel_indices_flat = voxels_for_term_init_flat
        term_gbo_obj.current_territory_demand = actual_demand_init
        
        if actual_demand_init > constants.EPSILON:
            term_gbo_obj.flow = actual_demand_init
            new_r = k_murray_factor * (term_gbo_obj.flow ** (1.0 / murray_exponent))
            term_gbo_obj.radius = max(initial_synthetic_radius_default, new_r) 
        else: 
            term_gbo_obj.flow = default_initial_flow 
            term_gbo_obj.radius = initial_synthetic_radius_default

        if gbo_graph.has_node(term_gbo_obj.id):
            gbo_node_data = gbo_graph.nodes[term_gbo_obj.id]
            if gbo_node_data.get('is_flow_root'): gbo_node_data['Q_flow'] = 0.0
            else: gbo_node_data['Q_flow'] = term_gbo_obj.flow
            gbo_node_data['radius'] = term_gbo_obj.radius
            
            if term_gbo_obj.current_territory_voxel_indices_flat:
                valid_indices_centroid = [idx for idx in term_gbo_obj.current_territory_voxel_indices_flat 
                                          if idx < tissue_data['world_coords_flat'].shape[0]]
                if valid_indices_centroid:
                    initial_coords = tissue_data['world_coords_flat'][valid_indices_centroid]
                    if initial_coords.shape[0] > 0:
                        new_pos_centroid = np.mean(initial_coords, axis=0)
                        if utils.distance_squared(term_gbo_obj.pos, new_pos_centroid) > constants.EPSILON**2 : # Only move if significant
                            old_pos_log = term_gbo_obj.pos.copy()
                            term_gbo_obj.pos = new_pos_centroid 
                            gbo_node_data['pos'] = new_pos_centroid 
                            logger.debug(f"GBO Terminal {term_gbo_obj.id} (Ω_init): Moved from {np.round(old_pos_log,3)} to centroid {np.round(new_pos_centroid,3)}.")
                            if term_gbo_obj.parent_id and gbo_graph.has_edge(term_gbo_obj.parent_id, term_gbo_obj.id):
                                parent_pos = gbo_graph.nodes[term_gbo_obj.parent_id]['pos']
                                new_len = utils.distance(parent_pos, term_gbo_obj.pos)
                                gbo_graph.edges[term_gbo_obj.parent_id, term_gbo_obj.id]['length'] = new_len
                                term_gbo_obj.length_from_parent = new_len
        else:
            logger.error(f"GBO terminal {term_gbo_obj.id} from GBOIterationData not found in gbo_graph during territory init.")

        logger.debug(f"GBO Terminal {term_gbo_obj.id} (Ω_init final): Pos={np.round(term_gbo_obj.pos,3)}, Claimed {len(voxels_for_term_init_flat)} voxels, "
                     f"Demand={term_gbo_obj.current_territory_demand:.2e}, Target Flow={term_gbo_obj.flow:.2e}, Radius={term_gbo_obj.radius:.4f}")

    logger.info(f"GBO Initialization complete. Perfused {np.sum(perfused_tissue_mask)} initial voxels within GBO domain. "
                f"{len(active_terminals)} active GBO terminals.")
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
            if np.any(np.array(valid_kdtree_indices) >= len(unperfused_global_flat_indices)):
                 final_frontier_voxels_global_flat_indices = np.array([], dtype=int)
        logger.debug(f"Terminal {terminal_gbo_data.id}: Frontier limited to {len(final_frontier_voxels_global_flat_indices)} voxels.")

    logger.info(f"Terminal {terminal_gbo_data.id} identified {len(final_frontier_voxels_global_flat_indices)} "
                 f"final frontier voxels (Ri,p).")
    return final_frontier_voxels_global_flat_indices


# In src/vascular_growth.py

import numpy as np
import networkx as nx
import logging
import os
from scipy.spatial import KDTree
from typing import Tuple, List, Dict, Optional 

from src import utils, data_structures, constants, config_manager
from src import energy_model
from src import perfusion_solver
from src import io_utils # For saving intermediate VTPs/NIfTIs

# Assuming GBOIterationData class and initialize_perfused_territory_and_terminals,
# find_growth_frontier_voxels are defined above this function in the same file or imported.

# If GBOIterationData is not defined elsewhere in this file, it should be:
# class GBOIterationData:
#     def __init__(self, terminal_id: str, pos: np.ndarray, radius: float, flow: float,
#                  is_synthetic: bool = True, original_measured_radius: Optional[float] = None,
#                  parent_id: Optional[str] = None, parent_measured_terminal_id: Optional[str] = None):
#         self.id: str = terminal_id
#         self.pos: np.ndarray = np.array(pos, dtype=float)
#         self.radius: float = float(radius)
#         self.flow: float = float(flow)
#         self.parent_id: Optional[str] = parent_id
#         self.parent_measured_terminal_id: Optional[str] = parent_measured_terminal_id
#         self.original_measured_terminal_radius: Optional[float] = original_measured_radius
#         self.length_from_parent: float = 0.0
#         self.is_synthetic: bool = is_synthetic
#         self.stop_growth: bool = False
#         self.current_territory_voxel_indices_flat: List[int] = []
#         self.current_territory_demand: float = 0.0

# Assume initialize_perfused_territory_and_terminals and find_growth_frontier_voxels
# are correctly defined in this file (as provided in previous responses).

logger = logging.getLogger(__name__)


def grow_healthy_vasculature(config: dict,
                             tissue_data: dict,
                             initial_graph: Optional[nx.DiGraph],
                             output_dir: str) -> Optional[nx.DiGraph]:
    logger.info("Starting GBO healthy vascular growth (Tissue-Led Model v2 with Flow Solver Integration)...")

    perfused_tissue_mask, current_active_terminals, next_node_id, gbo_graph = \
        initialize_perfused_territory_and_terminals(config, initial_graph, tissue_data)

    if not current_active_terminals:
        logger.error("GBO Aborted: No active GBO terminals after initialization.")
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
    flow_solver_interval = config_manager.get_param(config, "gbo_growth.flow_solver_interval", 1)
    max_segment_length_gbo = config_manager.get_param(config, "vascular_properties.max_segment_length", 2.0)


    total_voxels_in_domain = 0
    # Use healthy domain_mask for GBO total voxels count
    if tissue_data.get('domain_mask') is not None and np.any(tissue_data['domain_mask']):
        total_voxels_in_domain = np.sum(tissue_data['domain_mask'])
    
    if total_voxels_in_domain == 0:
        logger.warning("Healthy GBO: Domain mask for healthy tissue is empty. Growth might be limited or rely on fallback.")

    map_3d_to_flat_idx = -np.ones(tissue_data['shape'], dtype=np.int64)
    if tissue_data.get('voxel_indices_flat') is not None and tissue_data['voxel_indices_flat'].size > 0:
        valid_indices = tissue_data['voxel_indices_flat']
        valid_mask = (valid_indices[:,0] < tissue_data['shape'][0]) & \
                     (valid_indices[:,1] < tissue_data['shape'][1]) & \
                     (valid_indices[:,2] < tissue_data['shape'][2]) & \
                     (valid_indices[:,0] >= 0) & \
                     (valid_indices[:,1] >= 0) & \
                     (valid_indices[:,2] >= 0)
        valid_indices = valid_indices[valid_mask]
        if valid_indices.size > 0: # Check after filtering
            map_3d_to_flat_idx[valid_indices[:,0], valid_indices[:,1], valid_indices[:,2]] = np.arange(valid_indices.shape[0])

    # --- Main Iteration Loop ---
    for iteration in range(max_iterations):
        logger.info(f"--- GBO Iteration {iteration + 1} / {max_iterations} ---")

        terminals_for_growth_attempt = [t for t in current_active_terminals if not t.stop_growth]
        if not terminals_for_growth_attempt:
            logger.info("GBO: No active GBO terminals for growth at start of iteration.")
            break

        current_perfused_count = np.sum(perfused_tissue_mask)
        if total_voxels_in_domain > 0:
            perf_percentage = (current_perfused_count / total_voxels_in_domain) * 100
            logger.info(f"Active GBO terminals: {len(terminals_for_growth_attempt)}. Perfused voxels: {current_perfused_count}/{total_voxels_in_domain} ({perf_percentage:.1f}%)")
        else:
            logger.info(f"Active GBO terminals: {len(terminals_for_growth_attempt)}. Perfused voxels: {current_perfused_count} (Total domain voxels is 0).")

        # --- Build KDTree of Unperfused Voxels (within healthy GBO domain) ---
        unperfused_mask_3d = tissue_data.get('domain_mask', np.zeros(tissue_data['shape'], dtype=bool)) & (~perfused_tissue_mask)
        unperfused_voxels_3d_indices = np.array(np.where(unperfused_mask_3d)).T
        kdtree_unperfused: Optional[KDTree] = None
        # unperfused_kdtree_global_flat_indices maps indices from kdtree_unperfused back to global flat indices
        unperfused_kdtree_global_flat_indices: np.ndarray = np.array([], dtype=int)

        if unperfused_voxels_3d_indices.shape[0] > 0:
            unperfused_voxels_world_coords_for_kdt_build = utils.voxel_to_world(unperfused_voxels_3d_indices, tissue_data['affine'])
            
            # Get the global flat indices corresponding to these unperfused 3D voxel indices
            # These are indices into tissue_data['voxel_indices_flat'] / tissue_data['world_coords_flat']
            temp_flat_indices = map_3d_to_flat_idx[unperfused_voxels_3d_indices[:,0],
                                                   unperfused_voxels_3d_indices[:,1],
                                                   unperfused_voxels_3d_indices[:,2]]
            valid_for_kdtree_mask = (temp_flat_indices != -1)
            
            if np.any(valid_for_kdtree_mask):
                unperfused_kdtree_global_flat_indices = temp_flat_indices[valid_for_kdtree_mask]
                unperfused_voxels_world_coords_for_kdt_build = unperfused_voxels_world_coords_for_kdt_build[valid_for_kdtree_mask]
                if unperfused_voxels_world_coords_for_kdt_build.shape[0] > 0:
                    kdtree_unperfused = KDTree(unperfused_voxels_world_coords_for_kdt_build)
            else:
                logger.debug("No unperfused voxels mapped to valid flat indices for KDTree.")
        
        if kdtree_unperfused is None or kdtree_unperfused.n == 0:
            logger.info("GBO: No unperfused domain voxels for KDTree this iteration.")

        next_iter_terminals_manager: List[GBOIterationData] = []
        newly_perfused_in_iter_mask = np.zeros_like(perfused_tissue_mask)

        # --- Growth/Branching/Extension Phase for each GBO terminal ---
        for term_p_gbo_data in terminals_for_growth_attempt:
            logger.debug(f"Processing GBO terminal {term_p_gbo_data.id}. Pos: {np.round(term_p_gbo_data.pos,3)}, "
                         f"R: {term_p_gbo_data.radius:.4f}, Current Q_target: {term_p_gbo_data.flow:.2e}")

            # find_growth_frontier_voxels returns indices LOCAL to the kdtree_unperfused
            local_indices_in_kdt = find_growth_frontier_voxels(
                term_p_gbo_data, kdtree_unperfused, 
                np.arange(kdtree_unperfused.n if kdtree_unperfused else 0), # Pass local indices 0..N-1
                tissue_data, config
            )

            if kdtree_unperfused is None or kdtree_unperfused.n == 0 or local_indices_in_kdt.size == 0:
                logger.debug(f"Terminal {term_p_gbo_data.id} found no frontier voxels.")
                next_iter_terminals_manager.append(term_p_gbo_data)
                continue
            
            # Map these local KDTree indices back to the global flat indices
            current_frontier_global_flat_indices = unperfused_kdtree_global_flat_indices[local_indices_in_kdt]
            unique_frontier_global_flat_indices = np.unique(current_frontier_global_flat_indices) # Ensure unique

            if unique_frontier_global_flat_indices.size == 0:
                next_iter_terminals_manager.append(term_p_gbo_data)
                continue

            demand_map_3d_indices_frontier = tissue_data['voxel_indices_flat'][unique_frontier_global_flat_indices]
            demand_of_frontier_voxels = tissue_data['metabolic_demand_map'][
                demand_map_3d_indices_frontier[:,0], demand_map_3d_indices_frontier[:,1],
                demand_map_3d_indices_frontier[:,2]] # Demand map already has * dV
            demand_Rip = np.sum(demand_of_frontier_voxels)

            if demand_Rip < constants.EPSILON:
                next_iter_terminals_manager.append(term_p_gbo_data)
                continue
            
            potential_total_flow_if_extended = term_p_gbo_data.flow + demand_Rip
            potential_radius_if_extended = k_murray * (potential_total_flow_if_extended ** (1.0 / murray_exp))
            attempt_branching = False
            if term_p_gbo_data.radius > constants.EPSILON and \
               potential_radius_if_extended > term_p_gbo_data.radius * branch_radius_factor_thresh : attempt_branching = True
            if potential_total_flow_if_extended > max_flow_single_term: attempt_branching = True
            if demand_Rip > term_p_gbo_data.flow * min_demand_rip_bif_factor and term_p_gbo_data.flow > constants.EPSILON: attempt_branching = True

            if attempt_branching and unique_frontier_global_flat_indices.size > 0:
                logger.debug(f"Terminal {term_p_gbo_data.id} evaluating branching. New frontier demand: {demand_Rip:.2e}.")
                old_territory_indices_flat = np.array(term_p_gbo_data.current_territory_voxel_indices_flat, dtype=int)
                
                combined_territory_indices_flat = np.unique(np.concatenate((old_territory_indices_flat, unique_frontier_global_flat_indices))) \
                                             if old_territory_indices_flat.size > 0 else unique_frontier_global_flat_indices

                if len(combined_territory_indices_flat) < 2:
                    logger.debug(f"Combined territory for {term_p_gbo_data.id} too small. Attempting extension.")
                    attempt_branching = False
                else:
                    bifurcation_result = energy_model.find_optimal_bifurcation_for_combined_territory(
                        term_p_gbo_data, combined_territory_indices_flat, tissue_data, config, k_murray, murray_exp
                    )
                    if bifurcation_result:
                        c1_pos, c1_rad, c1_total_flow, c2_pos, c2_rad, c2_total_flow, _ = bifurcation_result
                        new_parent_total_flow = c1_total_flow + c2_total_flow
                        new_parent_radius = max(min_radius, k_murray * (new_parent_total_flow ** (1.0 / murray_exp)))
                        
                        parent_node_gbo_graph_data = gbo_graph.nodes[term_p_gbo_data.id]
                        parent_is_flow_root = parent_node_gbo_graph_data.get('is_flow_root', False)
                        parent_node_gbo_graph_data.update(
                            type='synthetic_bifurcation', radius=new_parent_radius,
                            Q_flow=new_parent_total_flow if not parent_is_flow_root else 0.0,
                            is_flow_root=parent_is_flow_root
                        )
                        logger.debug(f"Bifurcating node {term_p_gbo_data.id}: is_flow_root remains {parent_is_flow_root}")

                        for _, (child_pos, child_rad, child_flow) in enumerate([(c1_pos, c1_rad, c1_total_flow), (c2_pos, c2_rad, c2_total_flow)]):
                            child_id = f"s_{next_node_id}"; next_node_id += 1
                            child_gbo_obj = GBOIterationData(
                                child_id, child_pos, child_rad, child_flow,
                                original_measured_radius=term_p_gbo_data.original_measured_terminal_radius,
                                parent_id=term_p_gbo_data.id, parent_measured_terminal_id=term_p_gbo_data.parent_measured_terminal_id
                            )
                            child_gbo_obj.length_from_parent = utils.distance(parent_node_gbo_graph_data['pos'], child_pos)
                            next_iter_terminals_manager.append(child_gbo_obj)

                            child_attrs = vars(child_gbo_obj).copy()
                            child_attrs['type'] = 'synthetic_terminal'; child_attrs['is_flow_root'] = False
                            child_attrs['Q_flow'] = child_gbo_obj.flow
                            child_attrs.pop('current_territory_voxel_indices_flat', None); child_attrs.pop('current_territory_demand', None)
                            data_structures.add_node_to_graph(gbo_graph, child_id, **child_attrs)
                            data_structures.add_edge_to_graph(gbo_graph, term_p_gbo_data.id, child_id,
                                                              length=child_gbo_obj.length_from_parent, radius=new_parent_radius,
                                                              type='synthetic_segment')
                        
                        for v_idx_flat in unique_frontier_global_flat_indices:
                            v_3d_idx = tuple(tissue_data['voxel_indices_flat'][v_idx_flat])
                            newly_perfused_in_iter_mask[v_3d_idx] = True
                        # Parent GBOIterationData is done, children are in next_iter_terminals_manager
                        term_p_gbo_data.stop_growth = True # Mark parent GBO object as stopped
                        logger.info(f"Terminal {term_p_gbo_data.id} branched. Children target flows: {c1_total_flow:.2e}, {c2_total_flow:.2e}.")
                    else: attempt_branching = False 
            
            if not attempt_branching: # Extend
                old_pos_ext = term_p_gbo_data.pos.copy()
                logger.debug(f"Terminal {term_p_gbo_data.id} extending for Ri,p (demand {demand_Rip:.2e}).")
                term_p_gbo_data.flow += demand_Rip 
                term_p_gbo_data.radius = max(min_radius, k_murray * (term_p_gbo_data.flow ** (1.0 / murray_exp)))
                
                if unique_frontier_global_flat_indices.size > 0:
                    current_territory_coords_list = []
                    if term_p_gbo_data.current_territory_voxel_indices_flat:
                         valid_curr_idx_ext = [idx for idx in term_p_gbo_data.current_territory_voxel_indices_flat if idx < tissue_data['world_coords_flat'].shape[0]]
                         if valid_curr_idx_ext: current_territory_coords_list.append(tissue_data['world_coords_flat'][valid_curr_idx_ext])
                    
                    newly_acquired_coords = tissue_data['world_coords_flat'][unique_frontier_global_flat_indices]
                    current_territory_coords_list.append(newly_acquired_coords)
                    
                    all_supplied_coords = np.vstack(current_territory_coords_list)
                    if all_supplied_coords.shape[0] > 0:
                        new_target_pos = np.mean(all_supplied_coords, axis=0)
                        extension_vector = new_target_pos - old_pos_ext
                        extension_length = np.linalg.norm(extension_vector)

                        if extension_length > constants.EPSILON:
                            move_dist = min(extension_length, max_segment_length_gbo)
                            term_p_gbo_data.pos = old_pos_ext + extension_vector * (move_dist / extension_length)
                            if term_p_gbo_data.parent_id and gbo_graph.has_edge(term_p_gbo_data.parent_id, term_p_gbo_data.id):
                                parent_pos = gbo_graph.nodes[term_p_gbo_data.parent_id]['pos']
                                new_len = utils.distance(parent_pos, term_p_gbo_data.pos)
                                gbo_graph.edges[term_p_gbo_data.parent_id, term_p_gbo_data.id]['length'] = new_len
                                if gbo_graph.has_node(term_p_gbo_data.parent_id):
                                    gbo_graph.edges[term_p_gbo_data.parent_id, term_p_gbo_data.id]['radius'] = gbo_graph.nodes[term_p_gbo_data.parent_id]['radius']
                                term_p_gbo_data.length_from_parent = new_len
                
                if gbo_graph.has_node(term_p_gbo_data.id):
                    gbo_graph.nodes[term_p_gbo_data.id].update(pos=term_p_gbo_data.pos, 
                                                               Q_flow=term_p_gbo_data.flow if not gbo_graph.nodes[term_p_gbo_data.id].get('is_flow_root') else 0.0, 
                                                               radius=term_p_gbo_data.radius)
                
                for v_idx_flat in unique_frontier_global_flat_indices:
                    v_3d_idx = tuple(tissue_data['voxel_indices_flat'][v_idx_flat])
                    newly_perfused_in_iter_mask[v_3d_idx] = True
                term_p_gbo_data.current_territory_voxel_indices_flat.extend(list(unique_frontier_global_flat_indices))
                next_iter_terminals_manager.append(term_p_gbo_data)
        # --- End of loop for terminals_for_growth_attempt ---

        current_active_terminals = next_iter_terminals_manager # Update active terminals list
        perfused_tissue_mask = perfused_tissue_mask | newly_perfused_in_iter_mask
        num_newly_perfused_this_iter = np.sum(newly_perfused_in_iter_mask)
        logger.info(f"Perfused {num_newly_perfused_this_iter} new voxels in this GBO iteration's growth/branching phase.")
        
        # --- Global Adaptation Phase ---
        if current_active_terminals and np.any(perfused_tissue_mask):
            live_terminals_for_adaptation = [t for t in current_active_terminals if not t.stop_growth]
            if live_terminals_for_adaptation:
                # 1. Voronoi Refinement
                perfused_3d_indices_vor = np.array(np.where(perfused_tissue_mask)).T
                if perfused_3d_indices_vor.shape[0] > 0:
                    perfused_global_flat_indices_vor = map_3d_to_flat_idx[perfused_3d_indices_vor[:,0], perfused_3d_indices_vor[:,1], perfused_3d_indices_vor[:,2]]
                    valid_flat_mask_for_perf_vor = perfused_global_flat_indices_vor != -1
                    perfused_global_flat_indices_vor = perfused_global_flat_indices_vor[valid_flat_mask_for_perf_vor]
                    
                    if perfused_global_flat_indices_vor.size > 0 :
                        perfused_world_coords_for_voronoi = tissue_data['world_coords_flat'][perfused_global_flat_indices_vor]
                        term_positions_vor = np.array([t.pos for t in live_terminals_for_adaptation])
                        term_flows_capacity_vor = np.array([t.radius**murray_exp if t.radius > constants.EPSILON else default_initial_flow for t in live_terminals_for_adaptation]) # Weight by capacity (r^gamma)
                        
                        assigned_local_term_indices = np.full(perfused_world_coords_for_voronoi.shape[0], -1, dtype=int)
                        for i_pvox, p_vox_wc in enumerate(perfused_world_coords_for_voronoi):
                            distances_sq = np.sum((term_positions_vor - p_vox_wc)**2, axis=1)
                            weighted_distances = distances_sq / (term_flows_capacity_vor + constants.EPSILON) 
                            assigned_local_term_indices[i_pvox] = np.argmin(weighted_distances)
                        
                        for t_data_vor in live_terminals_for_adaptation: t_data_vor.current_territory_voxel_indices_flat, t_data_vor.current_territory_demand = [], 0.0
                        for i_pvox, local_term_idx in enumerate(assigned_local_term_indices):
                            if local_term_idx != -1 and local_term_idx < len(live_terminals_for_adaptation):
                                term_obj_vor = live_terminals_for_adaptation[local_term_idx]
                                global_flat_v_idx_vor = perfused_global_flat_indices_vor[i_pvox]
                                term_obj_vor.current_territory_voxel_indices_flat.append(global_flat_v_idx_vor)
                                v_3d_idx_for_demand_vor = tuple(tissue_data['voxel_indices_flat'][global_flat_v_idx_vor])
                                term_obj_vor.current_territory_demand += tissue_data['metabolic_demand_map'][v_3d_idx_for_demand_vor]
                        
                        for t_data_vor in live_terminals_for_adaptation: 
                            t_data_vor.flow = t_data_vor.current_territory_demand if t_data_vor.current_territory_demand > constants.EPSILON else default_initial_flow
                            if not ((iteration + 1) % flow_solver_interval == 0 or iteration == max_iterations - 1):
                                new_r_vor = k_murray * (t_data_vor.flow ** (1.0 / murray_exp))
                                t_data_vor.radius = max(min_radius, new_r_vor)
                                if gbo_graph.has_node(t_data_vor.id): 
                                    node_to_update = gbo_graph.nodes[t_data_vor.id]
                                    node_to_update['radius'] = t_data_vor.radius
                                    if not node_to_update.get('is_flow_root'):
                                        node_to_update['Q_flow'] = t_data_vor.flow
                            logger.debug(f"Term {t_data_vor.id} (Voronoi Refined): Target Q_demand={t_data_vor.flow:.2e}, Current R_after_voronoi_adapt={t_data_vor.radius:.4f}")
                    logger.info("Completed Voronoi refinement for active GBO terminals.")

                # 2. Solve Network Flow & Global Radius Adaptation
                run_flow_solver_this_iteration = ((iteration + 1) % flow_solver_interval == 0) or \
                                                 (iteration == max_iterations - 1 and iteration > 0)

                if run_flow_solver_this_iteration and gbo_graph.number_of_nodes() > 0 :
                    logger.info(f"Running 1D network flow solver for GBO iteration {iteration + 1}...")
                    for term_obj_flow_set in live_terminals_for_adaptation: # Update demands for sinks
                        if gbo_graph.has_node(term_obj_flow_set.id):
                            node_data_fs = gbo_graph.nodes[term_obj_flow_set.id]
                            if not node_data_fs.get('is_flow_root', False):
                                node_data_fs['Q_flow'] = term_obj_flow_set.flow
                            else: # Roots are pressure BCs, their Q_flow for B_vector is 0
                                node_data_fs['Q_flow'] = 0.0
                    
                    temp_graph_for_solver = gbo_graph.copy()
                    gbo_graph_with_flow = perfusion_solver.solve_1d_poiseuille_flow(temp_graph_for_solver, config, None, None) # Use Q_flow on nodes
                    
                    if gbo_graph_with_flow:
                        gbo_graph = gbo_graph_with_flow
                        logger.info("Flow solution obtained. Starting global radius adaptation for GBO tree...")
                        # Adapt radii of synthetic GBO nodes (and potentially measured junctions if desired)
                        nodes_to_adapt_gbo = [n for n, data_gbo_adapt in gbo_graph.nodes(data=True)
                                              if data_gbo_adapt.get('type','').startswith('synthetic_') or \
                                                 data_gbo_adapt.get('type') == 'measured_to_synthetic_junction'] # Adapt junctions too

                        for node_id_adapt in nodes_to_adapt_gbo:
                            node_data_adapt = gbo_graph.nodes[node_id_adapt]
                            actual_node_flow = 0.0
                            original_radius_before_adapt = node_data_adapt.get('radius', min_radius)
                            node_type_for_adapt = node_data_adapt.get('type')

                            is_sink_node = gbo_graph.out_degree(node_id_adapt) == 0 and gbo_graph.in_degree(node_id_adapt) > 0
                            is_source_like_node = gbo_graph.out_degree(node_id_adapt) > 0 # Roots, Bifs

                            if is_sink_node: # e.g. synthetic_terminal
                                for _, _, edge_data_in in gbo_graph.in_edges(node_id_adapt, data=True):
                                    solved_in_flow = edge_data_in.get('flow_solver')
                                    if solved_in_flow is not None and np.isfinite(solved_in_flow):
                                        actual_node_flow += abs(solved_in_flow) # Sum of magnitudes of incoming flows
                            elif is_source_like_node: # e.g. synthetic_bifurcation, synthetic_root_terminal, measured_to_synthetic_junction
                                for _, _, edge_data_out in gbo_graph.out_edges(node_id_adapt, data=True):
                                    solved_out_flow = edge_data_out.get('flow_solver')
                                    if solved_out_flow is not None and np.isfinite(solved_out_flow):
                                        actual_node_flow += abs(solved_out_flow) # Sum of magnitudes of outgoing flows
                            
                            if node_data_adapt.get('is_flow_root'): # Don't adapt radius of fixed pressure roots based on this flow sum
                                logger.debug(f"GlobalAdapt GBO: Node {node_id_adapt} is flow_root. Actual flow sum (for info): {actual_node_flow:.2e}. Radius not adapted by flow.")
                                continue # Skip radius adaptation for roots based on summed child flows

                            if abs(actual_node_flow) > constants.EPSILON:
                                new_radius_adapted = k_murray * (abs(actual_node_flow) ** (1.0 / murray_exp))
                                new_radius_adapted = max(min_radius, new_radius_adapted)
                                if not np.isclose(original_radius_before_adapt, new_radius_adapted, rtol=1e-2, atol=1e-5):
                                    logger.info(f"GlobalAdapt GBO: Node {node_id_adapt} ({node_type_for_adapt}) R: {original_radius_before_adapt:.4f} -> {new_radius_adapted:.4f} (Q_actual_sum={actual_node_flow:.2e})")
                                node_data_adapt['radius'] = new_radius_adapted
                            elif original_radius_before_adapt > min_radius + constants.EPSILON :
                                logger.info(f"GlobalAdapt GBO: Node {node_id_adapt} ({node_type_for_adapt}) Q_actual_sum near zero. Shrinking R from {original_radius_before_adapt:.4f} to {min_radius:.4f}.")
                                node_data_adapt['radius'] = min_radius
                        
                        # Sync radii from gbo_graph back to GBOIterationData objects
                        for term_obj_sync in live_terminals_for_adaptation: # These are GBOIterationData
                            if gbo_graph.has_node(term_obj_sync.id): # s_XXX nodes
                                term_obj_sync.radius = gbo_graph.nodes[term_obj_sync.id]['radius']
                                # Also update GBOIterationData flow if it was derived from solved edge flows (e.g. for terminals)
                                # For now, term_obj_sync.flow is primarily driven by Voronoi demand.
                        logger.info("Global radius adaptation for GBO tree complete.")
                    else: logger.error("Flow solver did not return a graph. Skipping GBO radius adaptation.")
                else: logger.info(f"Skipping GBO flow solver and global radius adaptation for iteration {iteration + 1}.")

        # --- Update Stop Flags (includes the new criterion) ---
        logger.info("Updating stop flags for GBO terminals...")
        active_terminals_still_growing = 0
        for term_data_stop_check in current_active_terminals:
            if term_data_stop_check.stop_growth:
                if gbo_graph.has_node(term_data_stop_check.id): gbo_graph.nodes[term_data_stop_check.id]['stop_growth'] = True
                continue

            if term_data_stop_check.radius < (min_radius + constants.EPSILON) : 
                term_data_stop_check.stop_growth = True
                logger.info(f"Terminal {term_data_stop_check.id} stopped: minRadius (R={term_data_stop_check.radius:.4f} <= {min_radius:.4f})")
            
            if not term_data_stop_check.stop_growth and term_data_stop_check.original_measured_terminal_radius is not None:
                stop_radius_factor_measured = config_manager.get_param(config, "gbo_growth.stop_criteria.radius_match_factor_measured", 0.95)
                stop_radius_factor_measured = np.clip(stop_radius_factor_measured, 0.1, 2.0)
                target_stop_radius = term_data_stop_check.original_measured_terminal_radius * stop_radius_factor_measured
                if term_data_stop_check.radius >= target_stop_radius:
                    term_data_stop_check.stop_growth = True
                    logger.info(f"Terminal {term_data_stop_check.id} (from VTP {term_data_stop_check.parent_measured_terminal_id}) "
                                f"stopped: GBO R {term_data_stop_check.radius:.4f} reached target {target_stop_radius:.4f} "
                                f"(orig_measured_R={term_data_stop_check.original_measured_terminal_radius:.4f}, factor={stop_radius_factor_measured:.2f}).")
            
            if not term_data_stop_check.stop_growth and \
               (not term_data_stop_check.current_territory_voxel_indices_flat and \
                term_data_stop_check.current_territory_demand < constants.EPSILON):
                term_data_stop_check.stop_growth = True
                logger.info(f"Terminal {term_data_stop_check.id} stopped: no territory/demand.")

            if gbo_graph.has_node(term_data_stop_check.id):
                gbo_graph.nodes[term_data_stop_check.id]['stop_growth'] = term_data_stop_check.stop_growth
            
            if not term_data_stop_check.stop_growth: 
                active_terminals_still_growing += 1
        
        logger.info(f"End of GBO iteration {iteration + 1}: {active_terminals_still_growing} GBO terminals still active.")

        # --- Intermediate Save & Global Stop Conditions ---
        stop_due_to_target_perfusion = False
        if total_voxels_in_domain > 0:
             stop_due_to_target_perfusion = (np.sum(perfused_tissue_mask) >= total_voxels_in_domain * 
                                        config_manager.get_param(config, "gbo_growth.target_domain_perfusion_fraction", 0.99))
        stop_due_to_no_active_terminals = (active_terminals_still_growing == 0 and iteration > 0)
        stop_due_to_no_new_growth = (num_newly_perfused_this_iter == 0 and iteration >= min_iters_no_growth_stop)

        if config_manager.get_param(config, "visualization.save_intermediate_steps", False):
            interval = config_manager.get_param(config, "visualization.intermediate_step_interval", 1)
            if ((iteration + 1) % interval == 0) or (iteration == max_iterations - 1) or \
               stop_due_to_no_new_growth or stop_due_to_target_perfusion or stop_due_to_no_active_terminals:
                logger.info(f"Saving intermediate GBO results for iteration {iteration + 1}...")
                io_utils.save_vascular_tree_vtp(gbo_graph, os.path.join(output_dir, f"gbo_graph_iter_{iteration+1}.vtp"))
                if np.any(perfused_tissue_mask):
                    io_utils.save_nifti_image(perfused_tissue_mask.astype(np.uint8), tissue_data['affine'],
                                              os.path.join(output_dir, f"gbo_perfused_mask_iter_{iteration+1}.nii.gz"))

        if stop_due_to_target_perfusion: logger.info(f"GBO Stopping after iteration {iteration + 1}: Target perfusion reached."); break
        if stop_due_to_no_active_terminals: logger.info(f"GBO Stopping after iteration {iteration + 1}: No active GBO terminals."); break
        if stop_due_to_no_new_growth: logger.info(f"GBO Stopping after iteration {iteration + 1}: No new growth for specified iters."); break
            
    logger.info(f"GBO healthy vascular growth finished. Final GBO tree component(s) added to graph. Total graph: {gbo_graph.number_of_nodes()} nodes, {gbo_graph.number_of_edges()} edges.")
    return gbo_graph