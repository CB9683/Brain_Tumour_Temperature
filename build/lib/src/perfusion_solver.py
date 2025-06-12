# src/perfusion_solver.py
from __future__ import annotations
import numpy as np
import networkx as nx
import logging
from typing import Dict, Tuple, Optional, List

from src import constants, config_manager, utils 

logger = logging.getLogger(__name__)

def calculate_segment_resistance(length: float, radius: float, viscosity: float) -> float:
    """Calculates hydraulic resistance of a cylindrical segment using Poiseuille's law.
    R = (8 * mu * L) / (pi * r^4)
    """
    if radius < constants.EPSILON:
        return np.inf 
    if length < constants.EPSILON:
        return constants.EPSILON # Return a tiny resistance to avoid division by zero for conductance
    return (8.0 * viscosity * length) / (constants.PI * (radius**4))

def solve_1d_poiseuille_flow(
    graph: nx.DiGraph, 
    config: dict, 
    root_pressure_val: Optional[float] = None, 
    terminal_flows_val: Optional[Dict[str, float]] = None 
    ) -> nx.DiGraph:
    """Solves for nodal pressures and segmental flows in a vascular graph.

    Modifies the input graph in-place by adding:
    - 'pressure' attribute to nodes.
    - 'flow_solver' attribute to edges.

    Nodes should have an 'is_flow_root' (bool) attribute if they are pressure inlets.
    Nodes acting as flow sinks should have a 'Q_flow' attribute (demand) and be identifiable as terminals.
    Edges require 'length' and 'radius' attributes (radius of the upstream segment).
    """
    if graph.number_of_nodes() == 0:
        logger.warning("Flow solver: Graph is empty. Nothing to solve.")
        return graph

    logger.info(f"Starting 1D Poiseuille flow solution for graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")

    viscosity = config_manager.get_param(config, "vascular_properties.blood_viscosity", constants.DEFAULT_BLOOD_VISCOSITY)
    if root_pressure_val is None:
        inlet_pressure = config_manager.get_param(config, "perfusion_solver.inlet_pressure", 10000.0) # Pa
    else:
        inlet_pressure = root_pressure_val

    node_list = list(graph.nodes())
    node_to_idx = {node_id: i for i, node_id in enumerate(node_list)}
    num_nodes = len(node_list)

    if num_nodes == 0: return graph

    A_matrix = np.zeros((num_nodes, num_nodes), dtype=float)
    B_vector = np.zeros(num_nodes, dtype=float)
    num_defined_pressure_bcs = 0
    num_defined_flow_sinks = 0

    logger.debug("--- Flow Solver: Assembling System Matrix A and Vector B ---")
    for i, node_id_i in enumerate(node_list):
        node_i_data = graph.nodes[node_id_i]
        
        if node_i_data.get('is_flow_root', False):
            A_matrix[i, :] = 0.0 
            A_matrix[i, i] = 1.0
            B_vector[i] = inlet_pressure
            num_defined_pressure_bcs += 1
            logger.debug(f"  BC Set: Node {node_id_i} (type: {node_i_data.get('type')}) is flow_root. Row {i}: A[{i},{i}]=1, B[{i}]={inlet_pressure:.2f}")
            continue

        sum_conductances_at_i = 0.0
        physical_neighbors = set(graph.predecessors(node_id_i)) | set(graph.successors(node_id_i))

        current_row_terms = [] # For debugging matrix row
        for neighbor_id in physical_neighbors:
            if neighbor_id not in node_to_idx: 
                logger.error(f"  Error: Neighbor {neighbor_id} of {node_id_i} not in node_to_idx map. Skipping.")
                continue
            j = node_to_idx[neighbor_id]
            
            edge_data = None
            # Check both u->v and v->u as physical connection implies one edge exists
            if graph.has_edge(node_id_i, neighbor_id):
                edge_data = graph.edges[node_id_i, neighbor_id]
            elif graph.has_edge(neighbor_id, node_id_i):
                edge_data = graph.edges[neighbor_id, node_id_i]
            
            if edge_data:
                length = edge_data.get('length', 0.0)
                radius = edge_data.get('radius', constants.MIN_VESSEL_RADIUS_MM) 
                radius = max(radius, constants.MIN_VESSEL_RADIUS_MM * 0.01) # Ensure tiny positive
                if length < constants.EPSILON: length = constants.EPSILON # Use tiny length for zero-length

                resistance = calculate_segment_resistance(length, radius, viscosity)
                conductance = 0.0
                if resistance < np.inf and resistance > constants.EPSILON:
                    conductance = 1.0 / resistance
                elif resistance <= constants.EPSILON: 
                    conductance = 1.0 / (constants.EPSILON * 10) 
                    logger.debug(f"  Edge involving {node_id_i}-{neighbor_id}: R~0, using high G={conductance:.1e}")
                
                if conductance > 0:
                    A_matrix[i, j] -= conductance
                    sum_conductances_at_i += conductance
                    current_row_terms.append(f"G_({node_id_i}-{neighbor_id})={conductance:.2e} (to P_{neighbor_id})")
            else:
                 logger.warning(f"  No edge data found between physically connected {node_id_i} and {neighbor_id}.")
        
        A_matrix[i, i] = sum_conductances_at_i
        logger.debug(f"  Matrix Row {i} ({node_id_i}): Diag A[{i},{i}]={A_matrix[i,i]:.2e}. Off-diag terms: {', '.join(current_row_terms) if current_row_terms else 'None'}")


        node_type = node_i_data.get('type', '')
        # A terminal for flow sink is a node with no outgoing segments in the *current graph structure being solved*
        # AND it's not a pressure root.
        is_graph_terminal_for_flow = (graph.out_degree(node_id_i) == 0 and not node_i_data.get('is_flow_root', False))
        
        if (node_type == 'synthetic_terminal' and is_graph_terminal_for_flow) or \
           node_i_data.get('is_flow_terminal', False): # Explicitly marked
            outflow = 0.0
            if terminal_flows_val and node_id_i in terminal_flows_val:
                outflow = terminal_flows_val[node_id_i]
            elif 'Q_flow' in node_i_data: 
                outflow = node_i_data['Q_flow'] 
            else:
                logger.warning(f"  Terminal node {node_id_i} (type: {node_type}) has no Q_flow or provided outflow. Assuming zero.")
            
            B_vector[i] -= outflow # Flow leaving node i is a source term -Q on RHS
            num_defined_flow_sinks +=1
            logger.debug(f"  BC Set: Node {node_id_i} is flow_sink. B[{i}] -= {outflow:.2e} (Total B[{i}]={B_vector[i]:.2e})")
        elif B_vector[i] == 0 and not is_graph_terminal_for_flow and not node_i_data.get('is_flow_root', False): # Internal node
            logger.debug(f"  Matrix Row {i} ({node_id_i}): Internal node. B[{i}]=0.")


    logger.debug(f"--- Flow Solver: System Matrix A (shape {A_matrix.shape}):\n{A_matrix}")
    logger.debug(f"--- Flow Solver: System Vector B (shape {B_vector.shape}):\n{B_vector}")
    logger.debug(f"Flow Solver: Total defined pressure BCs (is_flow_root): {num_defined_pressure_bcs}")
    logger.debug(f"Flow Solver: Total defined flow sinks (terminals with Q_flow): {num_defined_flow_sinks}")

    if num_defined_pressure_bcs == 0 and num_nodes > 0:
        logger.error("Flow solver: No pressure boundary conditions (is_flow_root=True) defined. Cannot solve.")
        for node_id_err in node_list: graph.nodes[node_id_err]['pressure'] = np.nan
        for u, v, data_err in graph.edges(data=True): data_err['flow_solver'] = np.nan
        return graph

    node_pressures_vec = np.full(num_nodes, np.nan) # Initialize with NaNs

    try:
        rank_A = 0
        if num_nodes > 0 : rank_A = np.linalg.matrix_rank(A_matrix)
        logger.debug(f"Flow Solver: Matrix A Rank = {rank_A} for {num_nodes} nodes.")

        if num_nodes > 0 and rank_A < num_nodes:
            logger.error(f"Flow solver: Matrix A is singular or rank-deficient (rank {rank_A} for {num_nodes} nodes). Cannot solve. "
                         "Check for disconnected graph components not attached to a pressure BC, or insufficient outflow conditions.")
            # graph is already initialized with NaNs effectively, so just return
        else:    
            node_pressures_vec = np.linalg.solve(A_matrix, B_vector)
            logger.info("Flow solver: Successfully solved for nodal pressures.")
            # logger.debug(f"Solved Node Pressures Vector:\n{node_pressures_vec}")

    except np.linalg.LinAlgError as e:
        logger.error(f"Flow solver: Linear algebra error during pressure solution: {e}", exc_info=False)
        logger.error(f"Matrix A Condition Number: {np.linalg.cond(A_matrix) if num_nodes > 0 and rank_A == num_nodes else 'N/A or singular'}")
        # NaNs are already default for node_pressures_vec if solve fails here

    for i, node_id in enumerate(node_list):
        graph.nodes[node_id]['pressure'] = node_pressures_vec[i] # Will be NaN if solve failed

    for u_id, v_id, data in graph.edges(data=True):
        P_u = graph.nodes[u_id]['pressure']
        P_v = graph.nodes[v_id]['pressure']
        
        # If pressures are NaN, flow will be NaN
        if np.isnan(P_u) or np.isnan(P_v):
            data['flow_solver'] = np.nan
            continue

        length = data.get('length', 0.0) 
        radius = data.get('radius', constants.MIN_VESSEL_RADIUS_MM)
        radius = max(radius, constants.MIN_VESSEL_RADIUS_MM * 0.01) 
        if length < constants.EPSILON: length = constants.EPSILON 

        resistance = calculate_segment_resistance(length, radius, viscosity)
        flow = 0.0
        if resistance < np.inf and resistance > constants.EPSILON:
            flow = (P_u - P_v) / resistance
        elif resistance <= constants.EPSILON: 
            if not np.isclose(P_u, P_v, atol=1e-6): 
                 logger.warning(f"Edge {u_id}->{v_id} has R~0 ({resistance:.1e}) but P_diff {(P_u-P_v):.2e}. Flow may be large/ill-defined.")
                 flow = (P_u - P_v) / (constants.EPSILON * 10) 
        data['flow_solver'] = flow
        if logger.isEnabledFor(logging.DEBUG): # Avoid string formatting if not needed
             logger.debug(f"Edge {u_id}->{v_id}: P_u={P_u:.2f}, P_v={P_v:.2f}, R={resistance:.2e}, Q_solver={flow:.2e}")
            
    logger.info("Flow solver: Finished annotating graph with pressures and flows.")
    return graph