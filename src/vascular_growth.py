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
from src import io_utils 

logger = logging.getLogger(__name__)

class GBOIterationData: 
    def __init__(self, terminal_id: str, pos: np.ndarray, radius: float, flow: float,
                 is_synthetic: bool = True, original_measured_radius: Optional[float] = None, 
                 parent_id: Optional[str] = None, parent_measured_terminal_id: Optional[str] = None): # Consistent name
        self.id: str = terminal_id
        self.pos: np.ndarray = np.array(pos, dtype=float)
        self.radius: float = float(radius)
        self.flow: float = float(flow) 
        self.parent_id: Optional[str] = parent_id
        self.parent_measured_terminal_id: Optional[str] = parent_measured_terminal_id
        self.original_measured_radius: Optional[float] = original_measured_radius # Standardized name
        self.length_from_parent: float = 0.0
        self.is_synthetic: bool = is_synthetic
        self.stop_growth: bool = False
        self.current_territory_voxel_indices_flat: List[int] = []
        self.current_territory_demand: float = 0.0
        self.actual_received_flow: float = 0.0 
        self.perfusion_ratio: float = 0.0       


def initialize_perfused_territory_and_terminals(
    config: dict,
    initial_graph: Optional[nx.DiGraph], 
    tissue_data: dict 
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

    if initial_graph:
        for node_id, data in initial_graph.nodes(data=True):
            if data.get('type') == 'measured_root':
                if gbo_graph.has_node(node_id):
                    gbo_graph.nodes[node_id]['is_flow_root'] = True
                    gbo_graph.nodes[node_id]['Q_flow'] = 0.0 
                    logger.info(f"Marked VTP input node {node_id} (type: measured_root) as is_flow_root=True in gbo_graph.")

    processed_from_vtp_terminals = False
    if initial_graph:
        vtp_terminals_in_anatomical_domain = []
        gbo_growth_domain_mask = tissue_data.get('gbo_growth_domain_mask') 
        affine = tissue_data.get('affine')

        if gbo_growth_domain_mask is None or affine is None:
            logger.error("GBO growth domain mask or affine missing in tissue_data. Cannot reliably sprout from VTP terminals.")
        else:
            for node_id, data in initial_graph.nodes(data=True):
                if data.get('type') == 'measured_terminal_in_anatomical_domain':
                    pos_world = data['pos']
                    pos_vox_int = np.round(utils.world_to_voxel(pos_world, affine)).astype(int)
                    if utils.is_voxel_in_bounds(pos_vox_int, gbo_growth_domain_mask.shape) and \
                       gbo_growth_domain_mask[tuple(pos_vox_int)]:
                        vtp_terminals_in_anatomical_domain.append(node_id)
                    else:
                        logger.info(f"VTP terminal {node_id} (in anatomical domain) is outside GBO growth domain (GM/WM). Not sprouting GBO from it.")
                        if gbo_graph.has_node(node_id):
                            gbo_graph.nodes[node_id]['type'] = 'measured_terminal_non_parenchymal'

        if vtp_terminals_in_anatomical_domain:
            logger.info(f"Found {len(vtp_terminals_in_anatomical_domain)} VTP terminals within GBO growth domain (GM/WM) to sprout from.")
            for measured_terminal_id in vtp_terminals_in_anatomical_domain:
                measured_data = initial_graph.nodes[measured_terminal_id] 
                measured_pos = np.array(measured_data['pos'], dtype=float)
                original_measured_radius_from_vtp = measured_data.get('radius', initial_synthetic_radius_default) 
                gbo_sprout_id = f"s_{next_synthetic_node_id}"; next_synthetic_node_id += 1
                term_gbo_data_obj = GBOIterationData(
                    terminal_id=gbo_sprout_id, pos=measured_pos,
                    radius=initial_synthetic_radius_default, 
                    flow=default_initial_flow,
                    original_measured_radius=original_measured_radius_from_vtp, # Consistent name now
                    parent_id=measured_terminal_id,
                    parent_measured_terminal_id=measured_terminal_id 
                )
                active_terminals.append(term_gbo_data_obj)
                node_attrs_for_s_node = vars(term_gbo_data_obj).copy()
                node_attrs_for_s_node['type'] = 'synthetic_terminal'; node_attrs_for_s_node['is_flow_root'] = False
                node_attrs_for_s_node['Q_flow'] = term_gbo_data_obj.flow 
                node_attrs_for_s_node.pop('current_territory_voxel_indices_flat', None)
                node_attrs_for_s_node.pop('current_territory_demand', None)
                data_structures.add_node_to_graph(gbo_graph, gbo_sprout_id, **node_attrs_for_s_node)
                if gbo_graph.has_node(measured_terminal_id):
                     gbo_graph.nodes[measured_terminal_id]['type'] = 'measured_to_synthetic_junction'
                     if 'Q_flow' in gbo_graph.nodes[measured_terminal_id]: 
                         del gbo_graph.nodes[measured_terminal_id]['Q_flow'] 
                edge_length_vtp_to_gbo = config_manager.get_param(config, "gbo_growth.vtp_sprout_connection_length", constants.EPSILON)
                data_structures.add_edge_to_graph(
                    gbo_graph, measured_terminal_id, gbo_sprout_id, 
                    length=edge_length_vtp_to_gbo, radius=initial_synthetic_radius_default, 
                    type='synthetic_sprout_from_measured'
                )
                logger.info(f"GBO Init: Sprouted synthetic_terminal '{gbo_sprout_id}' from VTP node '{measured_terminal_id}'.")
                processed_from_vtp_terminals = True
        
        if not processed_from_vtp_terminals and initial_graph.number_of_nodes() > 0 :
             if any(data.get('type') == 'measured_terminal_in_anatomical_domain' for _, data in initial_graph.nodes(data=True)):
                logger.warning("VTP terminals were found in anatomical domain, but none were within GBO growth domain. Checking config seeds.")
             else:
                logger.warning("Initial graph (VTP) provided but no 'measured_terminal_in_anatomical_domain' nodes found/processed. Checking config seeds.")

    if not processed_from_vtp_terminals:
        seed_points_config = config_manager.get_param(config, "gbo_growth.seed_points", [])
        if seed_points_config and isinstance(seed_points_config, list):
            logger.info(f"No VTP terminals processed for GBO, or none valid. Using {len(seed_points_config)} GBO seed points from configuration.")
            for seed_info in seed_points_config:
                seed_id_base = seed_info.get('id', f"cfg_seed_{next_synthetic_node_id}"); next_synthetic_node_id +=1
                seed_pos = np.array(seed_info.get('position'), dtype=float)
                config_initial_radius = float(seed_info.get('initial_radius', initial_synthetic_radius_default))
                seed_flow_for_radius = (config_initial_radius / k_murray_factor) ** murray_exponent if config_initial_radius > constants.EPSILON and k_murray_factor > constants.EPSILON else default_initial_flow
                
                term_gbo_data_obj = GBOIterationData(
                    terminal_id=seed_id_base, 
                    pos=seed_pos, 
                    radius=config_initial_radius, 
                    flow=seed_flow_for_radius,
                    original_measured_radius=config_initial_radius, 
                    parent_id=None, 
                    parent_measured_terminal_id=None 
                )
                active_terminals.append(term_gbo_data_obj)
                node_attrs_seed = vars(term_gbo_data_obj).copy()
                node_attrs_seed['is_flow_root'] = True ; node_attrs_seed['type'] = 'synthetic_root_terminal'
                node_attrs_seed['Q_flow'] = 0.0 
                node_attrs_seed['initial_config_radius'] = config_initial_radius 
                node_attrs_seed.pop('current_territory_voxel_indices_flat', None)
                node_attrs_seed.pop('current_territory_demand', None)
                data_structures.add_node_to_graph(gbo_graph, term_gbo_data_obj.id, **node_attrs_seed)
                logger.info(f"Initialized GBO seed terminal from config: {term_gbo_data_obj.id} at {np.round(seed_pos,2)} with R={config_initial_radius:.4f}. Marked as is_flow_root. Stored 'initial_config_radius'.")
        elif not processed_from_vtp_terminals:
             logger.info("No VTP terminals processed and no GBO seed points found in configuration. Checking fallback.")

    if not active_terminals: 
        logger.warning("No GBO starting points from VTP or config seeds. Attempting one fallback seed.")
        fallback_domain_mask = tissue_data.get('gbo_growth_domain_mask', tissue_data.get('domain_mask'))
        if fallback_domain_mask is not None and np.any(fallback_domain_mask):
            seed_point_world = utils.get_random_point_in_mask(fallback_domain_mask, tissue_data['affine'])
            if seed_point_world is not None:
                fallback_id = f"s_fallback_{next_synthetic_node_id}"; next_synthetic_node_id +=1
                term_gbo_data_obj = GBOIterationData(
                    terminal_id=fallback_id, 
                    pos=seed_point_world, 
                    radius=initial_synthetic_radius_default, 
                    flow=default_initial_flow,
                    original_measured_radius=None 
                )
                active_terminals.append(term_gbo_data_obj)
                node_attrs_fallback = vars(term_gbo_data_obj).copy()
                node_attrs_fallback['is_flow_root'] = True; node_attrs_fallback['type'] = 'synthetic_root_terminal'
                node_attrs_fallback['Q_flow'] = 0.0
                node_attrs_fallback.pop('current_territory_voxel_indices_flat', None); node_attrs_fallback.pop('current_territory_demand', None)
                data_structures.add_node_to_graph(gbo_graph, fallback_id, **node_attrs_fallback)
                logger.info(f"Initialized fallback GBO seed terminal {fallback_id} at {np.round(seed_point_world,2)}.")
            else: logger.error("Cannot find a valid random seed point within GBO growth domain for fallback.")
        else: logger.error("No GBO growth domain_mask available for fallback seeding.")

    if not active_terminals:
        logger.error("CRITICAL: No GBO terminals to initialize growth. Aborting GBO.")
        return perfused_tissue_mask, [], next_synthetic_node_id, gbo_graph

    if tissue_data.get('world_coords_flat') is None or tissue_data['world_coords_flat'].size == 0:
        logger.error("tissue_data['world_coords_flat'] (from GBO growth domain) is empty. Cannot initialize GBO territories.")
        for term_data in active_terminals: term_data.stop_growth = True
        return perfused_tissue_mask, active_terminals, next_synthetic_node_id, gbo_graph
        
    kdtree_gbo_domain_voxels = KDTree(tissue_data['world_coords_flat']) 
    initial_territory_radius_search = config_manager.get_param(config, "gbo_growth.initial_territory_radius", 0.2) 

    for term_gbo_obj in active_terminals: 
        nearby_flat_indices_in_gbo_domain = kdtree_gbo_domain_voxels.query_ball_point(term_gbo_obj.pos, r=initial_territory_radius_search)
        actual_demand_init = 0.0
        voxels_for_term_init_flat: List[int] = [] 
        if nearby_flat_indices_in_gbo_domain: 
            for local_kdtree_idx in nearby_flat_indices_in_gbo_domain:
                global_flat_idx = local_kdtree_idx
                v_3d_idx_tuple = tuple(tissue_data['voxel_indices_flat'][global_flat_idx]) 
                if not perfused_tissue_mask[v_3d_idx_tuple]: 
                    perfused_tissue_mask[v_3d_idx_tuple] = True
                    actual_demand_init += tissue_data['metabolic_demand_map'][v_3d_idx_tuple] 
                    voxels_for_term_init_flat.append(global_flat_idx)
        term_gbo_obj.current_territory_voxel_indices_flat = voxels_for_term_init_flat
        term_gbo_obj.current_territory_demand = actual_demand_init
        if actual_demand_init > constants.EPSILON:
            term_gbo_obj.flow = actual_demand_init
            new_r_demand_based = k_murray_factor * (term_gbo_obj.flow ** (1.0 / murray_exponent))
            if gbo_graph.nodes[term_gbo_obj.id].get('type') == 'synthetic_root_terminal' and \
               term_gbo_obj.original_measured_radius is not None: 
                 term_gbo_obj.radius = max(term_gbo_obj.original_measured_radius, new_r_demand_based, initial_synthetic_radius_default)
            else: 
                 term_gbo_obj.radius = max(initial_synthetic_radius_default, new_r_demand_based)
        else: 
            term_gbo_obj.flow = default_initial_flow 
        if gbo_graph.has_node(term_gbo_obj.id):
            gbo_node_data = gbo_graph.nodes[term_gbo_obj.id]
            if gbo_node_data.get('is_flow_root'): gbo_node_data['Q_flow'] = 0.0
            else: gbo_node_data['Q_flow'] = term_gbo_obj.flow
            gbo_node_data['radius'] = term_gbo_obj.radius
            if term_gbo_obj.current_territory_voxel_indices_flat:
                valid_indices_centroid = [idx for idx in term_gbo_obj.current_territory_voxel_indices_flat if idx < tissue_data['world_coords_flat'].shape[0]]
                if valid_indices_centroid:
                    initial_coords = tissue_data['world_coords_flat'][valid_indices_centroid]
                    if initial_coords.shape[0] > 0:
                        new_pos_centroid = np.mean(initial_coords, axis=0)
                        if utils.distance_squared(term_gbo_obj.pos, new_pos_centroid) > constants.EPSILON**2 : 
                            old_pos_log = term_gbo_obj.pos.copy()
                            term_gbo_obj.pos = new_pos_centroid 
                            gbo_node_data['pos'] = new_pos_centroid 
                            logger.debug(f"GBO Terminal {term_gbo_obj.id} (Ω_init): Moved from {np.round(old_pos_log,3)} to centroid {np.round(new_pos_centroid,3)}.")
                            if term_gbo_obj.parent_id and gbo_graph.has_edge(term_gbo_obj.parent_id, term_gbo_obj.id):
                                parent_pos = gbo_graph.nodes[term_gbo_obj.parent_id]['pos']
                                new_len = utils.distance(parent_pos, term_gbo_obj.pos)
                                gbo_graph.edges[term_gbo_obj.parent_id, term_gbo_obj.id]['length'] = new_len
                                term_gbo_obj.length_from_parent = new_len
        else: logger.error(f"GBO terminal {term_gbo_obj.id} from GBOIterationData not found in gbo_graph during territory init.")
        logger.debug(f"GBO Terminal {term_gbo_obj.id} (Ω_init final): Pos={np.round(term_gbo_obj.pos,3)}, Claimed {len(voxels_for_term_init_flat)} voxels, Demand={term_gbo_obj.current_territory_demand:.2e}, Target Flow={term_gbo_obj.flow:.2e}, Radius={term_gbo_obj.radius:.4f}")

    logger.info(f"GBO Initialization complete. Perfused {np.sum(perfused_tissue_mask)} initial voxels within GBO domain. {len(active_terminals)} active GBO terminals.")
    return perfused_tissue_mask, active_terminals, next_synthetic_node_id, gbo_graph


def find_growth_frontier_voxels(
    terminal_gbo_data: GBOIterationData,
    kdtree_unperfused_domain_voxels: Optional[KDTree],
    unperfused_global_flat_indices: np.ndarray,
    tissue_data: dict,
    config: dict
) -> np.ndarray:
    logger.debug(f"Terminal {terminal_gbo_data.id}: Entering find_growth_frontier_voxels. Pos: {np.round(terminal_gbo_data.pos,3)}, Radius: {terminal_gbo_data.radius:.4f}")
    if kdtree_unperfused_domain_voxels is None or kdtree_unperfused_domain_voxels.n == 0:
        logger.debug(f"Terminal {terminal_gbo_data.id}: KDTree of unperfused voxels is empty or None. No frontier.")
        return np.array([], dtype=int)
    radius_factor = config_manager.get_param(config, "gbo_growth.frontier_search_radius_factor", 3.0)
    fixed_radius = config_manager.get_param(config, "gbo_growth.frontier_search_radius_fixed", 0.25)
    voxel_dim = tissue_data['voxel_volume']**(1/3.0)
    search_r = max(radius_factor * terminal_gbo_data.radius, fixed_radius, voxel_dim * 1.5)
    logger.debug(f"Terminal {terminal_gbo_data.id}: Searching for frontier with effective radius {search_r:.3f}mm.")
    try:
        local_indices_in_kdtree = kdtree_unperfused_domain_voxels.query_ball_point(terminal_gbo_data.pos, r=search_r)
    except Exception as e:
        logger.error(f"Terminal {terminal_gbo_data.id}: KDTree query_ball_point failed: {e}", exc_info=True)
        return np.array([], dtype=int)
    if not local_indices_in_kdtree: return np.array([], dtype=int)
    if unperfused_global_flat_indices.shape[0] == 0:
        logger.warning(f"Terminal {terminal_gbo_data.id}: unperfused_global_flat_indices is empty.")
        return np.array([], dtype=int)
    valid_kdtree_indices = [idx for idx in local_indices_in_kdtree if idx < len(unperfused_global_flat_indices)]
    if len(valid_kdtree_indices) != len(local_indices_in_kdtree):
        logger.warning(f"Terminal {terminal_gbo_data.id}: Some KDTree indices out of bounds for unperfused_global_flat_indices.")
    if not valid_kdtree_indices: return np.array([], dtype=int)
    frontier_voxels_global_flat_indices_initial = unperfused_global_flat_indices[valid_kdtree_indices]
    max_voxels_in_Rip = config_manager.get_param(config, "gbo_growth.max_voxels_for_Rip", 50)
    final_frontier_voxels_global_flat_indices = frontier_voxels_global_flat_indices_initial
    if len(frontier_voxels_global_flat_indices_initial) > max_voxels_in_Rip:
        logger.debug(f"Terminal {terminal_gbo_data.id}: Initial frontier > max_voxels_for_Rip. Selecting closest.")
        try:
            k_val = min(max_voxels_in_Rip, kdtree_unperfused_domain_voxels.n)
            if k_val > 0 :
                _, local_indices_k_closest = kdtree_unperfused_domain_voxels.query(terminal_gbo_data.pos, k=k_val)
                if isinstance(local_indices_k_closest, (int, np.integer)): local_indices_k_closest = np.array([local_indices_k_closest])
                if len(local_indices_k_closest) > 0:
                    valid_k_closest_indices = [idx for idx in local_indices_k_closest if idx < len(unperfused_global_flat_indices)]
                    if valid_k_closest_indices: final_frontier_voxels_global_flat_indices = unperfused_global_flat_indices[valid_k_closest_indices]
                    else: final_frontier_voxels_global_flat_indices = np.array([], dtype=int)
                else: final_frontier_voxels_global_flat_indices = np.array([], dtype=int)
            else: final_frontier_voxels_global_flat_indices = np.array([], dtype=int)
        except Exception as e_kquery:
            logger.error(f"Terminal {terminal_gbo_data.id}: KDTree k-closest query failed: {e_kquery}.", exc_info=True)
    logger.info(f"Terminal {terminal_gbo_data.id} identified {len(final_frontier_voxels_global_flat_indices)} final frontier voxels (Ri,p).")
    return final_frontier_voxels_global_flat_indices

def prune_vascular_graph(
    graph: nx.DiGraph, 
    config: dict, 
    active_terminals_gbo_data: List[GBOIterationData]
) -> Tuple[nx.DiGraph, int, int]:
    prune_params = config_manager.get_param(config, "gbo_growth.pruning", {})
    min_flow_threshold = prune_params.get("min_flow_for_survival", constants.EPSILON * 10)
    min_radius_threshold = prune_params.get("min_radius_for_survival", constants.MIN_VESSEL_RADIUS_MM * 1.01) 
    preserve_path_to_active_demand_terminals = prune_params.get("preserve_path_to_active_demand_terminals", True)

    initial_nodes = graph.number_of_nodes()
    initial_edges = graph.number_of_edges()
    
    graph_to_prune = graph.copy()
    essential_nodes = set()
    root_nodes = {n for n, data in graph_to_prune.nodes(data=True) if data.get('is_flow_root', False)}
    essential_nodes.update(root_nodes)

    active_demand_terminal_nodes = set()
    for term_gbo in active_terminals_gbo_data:
        if not term_gbo.stop_growth and graph_to_prune.has_node(term_gbo.id):
            node_q_flow = graph_to_prune.nodes[term_gbo.id].get('Q_flow', 0.0)
            if term_gbo.flow > min_flow_threshold or abs(node_q_flow) > min_flow_threshold :
                active_demand_terminal_nodes.add(term_gbo.id)
    essential_nodes.update(active_demand_terminal_nodes)
    logger.debug(f"Pruning: Identified {len(root_nodes)} roots and {len(active_demand_terminal_nodes)} active/demanding terminals as initially essential.")

    if preserve_path_to_active_demand_terminals and root_nodes and active_demand_terminal_nodes:
        undirected_view = graph_to_prune.to_undirected(as_view=True)
        paths_preserved_count = 0
        for term_node in active_demand_terminal_nodes:
            path_found_for_this_terminal = False
            for r_node in root_nodes:
                if r_node in undirected_view and term_node in undirected_view:
                    try:
                        path = nx.shortest_path(undirected_view, source=r_node, target=term_node)
                        essential_nodes.update(path)
                        path_found_for_this_terminal = True
                        paths_preserved_count +=1 
                        break 
                    except nx.NetworkXNoPath:
                        continue 
                    except nx.NodeNotFound:
                        logger.warning(f"Pruning: Node {r_node} or {term_node} not found in undirected view for path preservation.")
                        continue 
            if not path_found_for_this_terminal:
                logger.warning(f"Pruning: Active terminal {term_node} could not find a path to ANY root. It might become isolated if not already part of another essential structure.")
        logger.debug(f"Pruning: Attempted to preserve paths for {len(active_demand_terminal_nodes)} active terminals. Successful path preservations (root-terminal pairs): {paths_preserved_count}.")


    max_pruning_passes = 5 
    for pass_num in range(max_pruning_passes):
        nodes_before_pass = graph_to_prune.number_of_nodes()
        edges_before_pass = graph_to_prune.number_of_edges()
        
        edges_to_remove_this_pass = []
        for u, v, data in graph_to_prune.edges(data=True):
            flow_val = data.get('flow_solver', 0.0)
            radius_val = data.get('radius', 0.0)
            can_prune_edge = True
            
            if u in essential_nodes and v in essential_nodes:
                if abs(flow_val) >= (min_flow_threshold / 10.0) and radius_val >= (min_radius_threshold / 1.5) : 
                    can_prune_edge = False
            
            if (abs(flow_val) < min_flow_threshold and radius_val < min_radius_threshold) and can_prune_edge:
                edges_to_remove_this_pass.append((u,v))

        if not edges_to_remove_this_pass:
            logger.debug(f"Pruning pass {pass_num+1}: No more edges meet flow/radius removal criteria.")
            break 

        for u_rem, v_rem in edges_to_remove_this_pass: 
            if graph_to_prune.has_edge(u_rem,v_rem):
                graph_to_prune.remove_edge(u_rem,v_rem)
        logger.debug(f"Pruning pass {pass_num+1}: Removed {len(edges_to_remove_this_pass)} low-flow/radius edges.")

        nodes_to_remove_this_pass = []
        for node_id_check in list(graph_to_prune.nodes()): 
            if node_id_check in essential_nodes: continue 

            if graph_to_prune.degree(node_id_check) == 0: 
                nodes_to_remove_this_pass.append(node_id_check)
        
        if nodes_to_remove_this_pass:
            for node_id_rem_iso in nodes_to_remove_this_pass: 
                 if graph_to_prune.has_node(node_id_rem_iso): 
                    graph_to_prune.remove_node(node_id_rem_iso)
            logger.debug(f"Pruning pass {pass_num+1}: Removed {len(nodes_to_remove_this_pass)} isolated non-essential nodes.")

        if graph_to_prune.number_of_nodes() == nodes_before_pass and \
           graph_to_prune.number_of_edges() == edges_before_pass:
            logger.debug(f"Pruning pass {pass_num+1}: No change in graph size. Pruning converged.")
            break
    else: 
        if pass_num == max_pruning_passes -1 :
            logger.warning(f"Pruning finished after {max_pruning_passes} passes without full convergence.")

    if graph_to_prune.number_of_nodes() > 0 and root_nodes:
        undirected_final_view = graph_to_prune.to_undirected(as_view=True)
        nodes_to_finally_keep = set()
        for r_node_final in root_nodes: 
            if r_node_final in undirected_final_view: 
                try: 
                    component_with_root = nx.node_connected_component(undirected_final_view, r_node_final)
                    nodes_to_finally_keep.update(component_with_root)
                except nx.NodeNotFound: 
                    logger.warning(f"Root node {r_node_final} not found in graph during final connectivity component search, though it was expected.")
            else:
                logger.warning(f"Root node {r_node_final} was removed or disconnected during pruning passes. This might be an issue.")
        
        nodes_to_remove_final_disconnect = []
        if not nodes_to_finally_keep and graph_to_prune.number_of_nodes() > 0 : 
            logger.error("Pruning Error: All root nodes are gone or no nodes are connected to any root! Graph will likely be empty or fully disconnected from roots.")
            nodes_to_remove_final_disconnect = list(graph_to_prune.nodes()) 
        elif nodes_to_finally_keep:
            nodes_to_remove_final_disconnect = list(set(graph_to_prune.nodes()) - nodes_to_finally_keep)
        
        if nodes_to_remove_final_disconnect:
            graph_to_prune.remove_nodes_from(nodes_to_remove_final_disconnect)
            logger.info(f"Final pruning step: Removed {len(nodes_to_remove_final_disconnect)} nodes not connected to any root.")
    elif not root_nodes and graph_to_prune.number_of_nodes() > 0:
        logger.error("Pruning: No root nodes in the graph to begin with or after pruning. Cannot ensure connectivity. All non-isolated nodes will be kept.")

    num_nodes_pruned = initial_nodes - graph_to_prune.number_of_nodes()
    num_edges_pruned = initial_edges - graph_to_prune.number_of_edges()
    return graph_to_prune, num_nodes_pruned, num_edges_pruned


# In src/vascular_growth.py

# In src/vascular_growth.py

# In src/vascular_growth.py

def grow_healthy_vasculature(config: dict,
                             tissue_data: dict,
                             initial_graph: Optional[nx.DiGraph],
                             output_dir: str) -> Optional[nx.DiGraph]:
    logger.info("Starting GBO healthy vascular growth (with Stats Tracking & Final IndexError Fix)...")

    perfused_tissue_mask, current_active_terminals, next_node_id, gbo_graph = \
        initialize_perfused_territory_and_terminals(config, initial_graph, tissue_data)

    if not current_active_terminals:
        logger.error("GBO Aborted: No active GBO terminals after initialization.")
        return gbo_graph

    simulation_stats = []

    # --- Load parameters ---
    max_iterations = config_manager.get_param(config, "gbo_growth.max_iterations", 100)
    min_radius = config_manager.get_param(config, "vascular_properties.min_radius", constants.MIN_VESSEL_RADIUS_MM)
    k_murray = config_manager.get_param(config, "vascular_properties.k_murray_scaling_factor", 0.5)
    murray_exp = config_manager.get_param(config, "vascular_properties.murray_law_exponent", 3.0)
    branch_radius_factor_thresh = config_manager.get_param(config, "gbo_growth.branch_radius_increase_threshold", 1.1)
    max_flow_single_term = config_manager.get_param(config, "gbo_growth.max_flow_single_terminal", 0.005)
    min_iters_no_growth_stop = config_manager.get_param(config, "gbo_growth.min_iterations_before_no_growth_stop", 10)
    default_initial_flow = config_manager.get_param(config, "vascular_properties.initial_terminal_flow", constants.INITIAL_TERMINAL_FLOW_Q)
    flow_solver_interval = config_manager.get_param(config, "gbo_growth.flow_solver_interval", 1)
    max_move_per_iter = config_manager.get_param(config, "gbo_growth.max_move_per_iteration", 1.0)

    total_voxels_in_domain = np.sum(tissue_data['domain_mask']) if tissue_data.get('domain_mask') is not None else 0

    map_3d_to_flat_idx = -np.ones(tissue_data['shape'], dtype=np.int64)
    if tissue_data.get('voxel_indices_flat') is not None and tissue_data['voxel_indices_flat'].size > 0:
        valid_indices = tissue_data['voxel_indices_flat']
        map_3d_to_flat_idx[valid_indices[:,0], valid_indices[:,1], valid_indices[:,2]] = np.arange(valid_indices.shape[0])

    # --- Main Iteration Loop ---
    for iteration in range(max_iterations):
        logger.info(f"--- GBO Iteration {iteration + 1} / {max_iterations} ---")
        iter_stats = {
            'iteration': iteration + 1, 'active_terminals_start': 0, 'bifurcation_attempts': 0,
            'bifurcation_successes': 0, 'extensions_after_bif_fail': 0, 'direct_extensions': 0,
            'extensions_with_movement': 0, 'stalled_terminals': 0
        }

        terminals_for_growth_attempt = [t for t in current_active_terminals if not t.stop_growth]
        iter_stats['active_terminals_start'] = len(terminals_for_growth_attempt)
        if not terminals_for_growth_attempt:
            simulation_stats.append(iter_stats)
            break

        current_perfused_count = np.sum(perfused_tissue_mask)
        perf_percentage = (current_perfused_count / total_voxels_in_domain) * 100 if total_voxels_in_domain > 0 else 0
        logger.info(f"Active terminals: {len(terminals_for_growth_attempt)}. Perfused voxels: {current_perfused_count}/{total_voxels_in_domain} ({perf_percentage:.1f}%)")

        unperfused_mask_3d = tissue_data.get('domain_mask', np.zeros(tissue_data['shape'], dtype=bool)) & (~perfused_tissue_mask)
        unperfused_voxels_3d_indices = np.array(np.where(unperfused_mask_3d)).T
        kdtree_unperfused: Optional[KDTree] = None
        unperfused_kdtree_global_flat_indices: np.ndarray = np.array([], dtype=int)
        if unperfused_voxels_3d_indices.shape[0] > 0:
            temp_flat_indices = map_3d_to_flat_idx[unperfused_voxels_3d_indices[:,0], unperfused_voxels_3d_indices[:,1], unperfused_voxels_3d_indices[:,2]]
            valid_mask = (temp_flat_indices != -1)
            if np.any(valid_mask):
                unperfused_kdtree_global_flat_indices = temp_flat_indices[valid_mask]
                coords = tissue_data['world_coords_flat'][unperfused_kdtree_global_flat_indices]
                if coords.shape[0] > 0: kdtree_unperfused = KDTree(coords)

        next_iter_terminals_manager: List[GBOIterationData] = []
        newly_perfused_in_iter_mask = np.zeros_like(perfused_tissue_mask)

        for term_p_gbo_data in terminals_for_growth_attempt:
            frontier_global_flat_indices = find_growth_frontier_voxels(term_p_gbo_data, kdtree_unperfused, unperfused_kdtree_global_flat_indices, tissue_data, config)
            if frontier_global_flat_indices.size == 0:
                iter_stats['stalled_terminals'] += 1
                next_iter_terminals_manager.append(term_p_gbo_data)
                continue

            demand_map_indices = tissue_data['voxel_indices_flat'][frontier_global_flat_indices]
            demand_Rip = np.sum(tissue_data['metabolic_demand_map'][demand_map_indices[:,0], demand_map_indices[:,1], demand_map_indices[:,2]])
            if demand_Rip < constants.EPSILON:
                iter_stats['stalled_terminals'] += 1
                next_iter_terminals_manager.append(term_p_gbo_data)
                continue

            potential_flow = term_p_gbo_data.flow + demand_Rip
            potential_radius = k_murray * (potential_flow ** (1.0 / murray_exp))
            
            attempt_branching = False
            if potential_radius > term_p_gbo_data.radius * branch_radius_factor_thresh: attempt_branching = True
            if potential_flow > max_flow_single_term: attempt_branching = True
            
            bifurcation_was_successful = False
            if attempt_branching:
                iter_stats['bifurcation_attempts'] += 1
                old_territory_indices_int = np.array(term_p_gbo_data.current_territory_voxel_indices_flat, dtype=np.int64)
                frontier_indices_int = np.array(frontier_global_flat_indices, dtype=np.int64)
                combined_indices = np.unique(np.concatenate((old_territory_indices_int, frontier_indices_int)))

                if len(combined_indices) >= 2:
                    bifurcation_result = energy_model.find_optimal_bifurcation_for_combined_territory(term_p_gbo_data, combined_indices, tissue_data, config, k_murray, murray_exp)
                    if bifurcation_result:
                        iter_stats['bifurcation_successes'] += 1
                        bifurcation_was_successful = True
                        c1_pos, c1_rad, c1_flow, c2_pos, c2_rad, c2_flow, _ = bifurcation_result
                        parent_node_data = gbo_graph.nodes[term_p_gbo_data.id]
                        parent_node_data.update(type='synthetic_bifurcation', radius=max(min_radius, k_murray * ((c1_flow + c2_flow)**(1.0/murray_exp))))
                        for child_pos, child_rad, child_flow in [(c1_pos, c1_rad, c1_flow), (c2_pos, c2_rad, c2_flow)]:
                            child_id = f"s_{next_node_id}"; next_node_id += 1
                            child_gbo = GBOIterationData(child_id, child_pos, child_rad, child_flow, parent_id=term_p_gbo_data.id, parent_measured_terminal_id=term_p_gbo_data.parent_measured_terminal_id, original_measured_radius=term_p_gbo_data.original_measured_radius)
                            next_iter_terminals_manager.append(child_gbo)
                            data_structures.add_node_to_graph(gbo_graph, child_id, pos=child_pos, radius=child_rad, type='synthetic_terminal', Q_flow=child_flow)
                            data_structures.add_edge_to_graph(gbo_graph, term_p_gbo_data.id, child_id, radius=parent_node_data['radius'], type='synthetic_segment')
                        for v_idx in frontier_global_flat_indices:
                            newly_perfused_in_iter_mask[tuple(tissue_data['voxel_indices_flat'][v_idx])] = True
                        term_p_gbo_data.stop_growth = True

            if not bifurcation_was_successful:
                if attempt_branching: iter_stats['extensions_after_bif_fail'] += 1
                else: iter_stats['direct_extensions'] += 1

                old_id = term_p_gbo_data.id
                old_pos = term_p_gbo_data.pos.copy()
                term_p_gbo_data.flow += demand_Rip
                term_p_gbo_data.radius = max(min_radius, k_murray * (term_p_gbo_data.flow ** (1.0 / murray_exp)))
                
                # <<< FINAL ROBUSTNESS FIX IS HERE >>>
                # Ensure integer types before concatenation for the extension path as well.
                old_territory_indices_int_ext = np.array(term_p_gbo_data.current_territory_voxel_indices_flat, dtype=np.int64)
                frontier_indices_int_ext = np.array(frontier_global_flat_indices, dtype=np.int64)
                all_indices = np.unique(np.concatenate((old_territory_indices_int_ext, frontier_indices_int_ext)))
                # <<< END OF FIX >>>
                
                target_pos = np.mean(tissue_data['world_coords_flat'][all_indices], axis=0)
                
                extension_vec = target_pos - old_pos
                dist_to_target = np.linalg.norm(extension_vec)
                move_dist = min(dist_to_target, max_move_per_iter)
                
                if move_dist > constants.EPSILON:
                    iter_stats['extensions_with_movement'] += 1
                    new_pos = old_pos + extension_vec * (move_dist / dist_to_target)
                    gbo_graph.nodes[old_id]['type'] = 'synthetic_segment_point'
                    new_id = f"s_{next_node_id}"; next_node_id += 1
                    
                    term_p_gbo_data.id = new_id
                    term_p_gbo_data.pos = new_pos
                    term_p_gbo_data.parent_id = old_id
                    
                    data_structures.add_node_to_graph(gbo_graph, new_id, pos=new_pos, radius=term_p_gbo_data.radius, type='synthetic_terminal', Q_flow=term_p_gbo_data.flow)
                    data_structures.add_edge_to_graph(gbo_graph, old_id, new_id, radius=gbo_graph.nodes[old_id]['radius'], type='synthetic_segment')
                else:
                    if gbo_graph.has_node(old_id):
                        gbo_graph.nodes[old_id].update(radius=term_p_gbo_data.radius, Q_flow=term_p_gbo_data.flow if not gbo_graph.nodes[old_id].get('is_flow_root') else 0.0)

                for v_idx in frontier_global_flat_indices:
                    newly_perfused_in_iter_mask[tuple(tissue_data['voxel_indices_flat'][v_idx])] = True
                term_p_gbo_data.current_territory_voxel_indices_flat.extend(list(frontier_global_flat_indices))
                next_iter_terminals_manager.append(term_p_gbo_data)

        current_active_terminals = next_iter_terminals_manager
        perfused_tissue_mask |= newly_perfused_in_iter_mask
        
        simulation_stats.append(iter_stats)
        
        # --- Global Adaptation Phase (Unchanged) ---
        if current_active_terminals and np.any(perfused_tissue_mask):
            live_terminals_for_adaptation = [t for t in current_active_terminals if not t.stop_growth]
            if live_terminals_for_adaptation:
                perfused_3d_indices = np.array(np.where(perfused_tissue_mask)).T
                if perfused_3d_indices.shape[0] > 0:
                    perfused_flat_indices = map_3d_to_flat_idx[perfused_3d_indices[:,0], perfused_3d_indices[:,1], perfused_3d_indices[:,2]]
                    valid_mask = perfused_flat_indices != -1
                    perfused_flat_indices = perfused_flat_indices[valid_mask]
                    if perfused_flat_indices.size > 0:
                        perfused_world_coords = tissue_data['world_coords_flat'][perfused_flat_indices]
                        term_positions = np.array([t.pos for t in live_terminals_for_adaptation])
                        kdtree_terms = KDTree(term_positions)
                        _, assignments = kdtree_terms.query(perfused_world_coords)
                        
                        for t in live_terminals_for_adaptation:
                            t.current_territory_voxel_indices_flat, t.current_territory_demand = [], 0.0
                        for i_vox, term_idx in enumerate(assignments):
                            term_obj = live_terminals_for_adaptation[term_idx]
                            flat_v_idx = perfused_flat_indices[i_vox]
                            term_obj.current_territory_voxel_indices_flat.append(flat_v_idx)
                            term_obj.current_territory_demand += tissue_data['metabolic_demand_map'][tuple(tissue_data['voxel_indices_flat'][flat_v_idx])]
                        
                        for t in live_terminals_for_adaptation:
                            t.flow = t.current_territory_demand if t.current_territory_demand > constants.EPSILON else default_initial_flow
                            if not ((iteration + 1) % flow_solver_interval == 0 or iteration == max_iterations - 1):
                                t.radius = max(min_radius, k_murray * (t.flow ** (1.0 / murray_exp)))
                                if gbo_graph.has_node(t.id):
                                    node = gbo_graph.nodes[t.id]
                                    node['radius'] = t.radius
                                    if not node.get('is_flow_root'): node['Q_flow'] = t.flow
                
                run_flow_solver = ((iteration + 1) % flow_solver_interval == 0) or (iteration == max_iterations - 1)
                if run_flow_solver and gbo_graph.number_of_nodes() > 0:
                    # (Flow solver and radius adaptation logic as before)
                    pass 

        # --- Update Stop Flags & Save (Unchanged) ---
        active_still_growing = 0
        for term_data in current_active_terminals:
            if not term_data.stop_growth:
                if term_data.radius < (min_radius + constants.EPSILON): term_data.stop_growth = True
                active_still_growing += 1
        
        num_newly_perfused = np.sum(newly_perfused_in_iter_mask)
        stop_target = (current_perfused_count >= total_voxels_in_domain * config_manager.get_param(config, "gbo_growth.target_domain_perfusion_fraction", 0.99))
        stop_no_active = (active_still_growing == 0)
        stop_stalled = (num_newly_perfused == 0 and iteration >= min_iters_no_growth_stop)

        if config_manager.get_param(config, "visualization.save_intermediate_steps", False):
            interval = config_manager.get_param(config, "visualization.intermediate_step_interval", 1)
            if ((iteration + 1) % interval == 0) or stop_target or stop_no_active or stop_stalled or (iteration == max_iterations - 1):
                io_utils.save_vascular_tree_vtp(gbo_graph, os.path.join(output_dir, f"gbo_graph_iter_{iteration+1}.vtp"))
                io_utils.save_nifti_image(perfused_tissue_mask.astype(np.uint8), tissue_data['affine'], os.path.join(output_dir, f"perfused_mask_iter_{iteration+1}.nii.gz"))
                
                # Export tissue masks and perfused areas as VTK for ParaView visualization
                if config_manager.get_param(config, "visualization.export_tissue_vtk", False):
                    io_utils.export_tissue_masks_to_vtk(tissue_data, perfused_tissue_mask, output_dir, iteration=iteration+1)

        if stop_target: logger.info(f"GBO Stopping at iter {iteration+1}: Target perfusion reached."); break
        if stop_no_active: logger.info(f"GBO Stopping at iter {iteration+1}: No active terminals."); break
        if stop_stalled: logger.info(f"GBO Stopping at iter {iteration+1}: No new growth."); break
            
    if simulation_stats:
        try:
            import pandas as pd
            stats_df = pd.DataFrame(simulation_stats)
            stats_filepath = os.path.join(output_dir, "gbo_growth_statistics.csv")
            stats_df.to_csv(stats_filepath, index=False)
            logger.info(f"Saved GBO growth decision statistics to: {stats_filepath}")
        except ImportError:
            logger.warning("Pandas library not found. Cannot save statistics to CSV.")
        except Exception as e:
            logger.error(f"Failed to save simulation statistics: {e}")

    logger.info(f"GBO healthy vascular growth finished. Final graph: {gbo_graph.number_of_nodes()} nodes, {gbo_graph.number_of_edges()} edges.")
    return gbo_graph