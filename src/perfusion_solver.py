# src/perfusion_solver.py
from __future__ import annotations
import numpy as np
import networkx as nx
import logging
from typing import Dict, Tuple, Optional, List

from src import constants, config_manager, utils, data_structures

logger = logging.getLogger(__name__)

def calculate_segment_resistance(length: float, radius: float, viscosity: float) -> float:
    """Calculates hydraulic resistance of a cylindrical segment using Poiseuille's law.

    R = (8 * mu * L) / (pi * r^4)

    Args:
        length (float): Length of the segment.
        radius (float): Radius of the segment.
        viscosity (float): Dynamic viscosity of the fluid.

    Returns:
        float: Hydraulic resistance. Returns np.inf if radius is near zero.
    """
    if radius < constants.EPSILON: # Avoid division by zero or extremely small radius
        # logger.warning(f"Near-zero radius ({radius}) encountered in resistance calculation for segment of length {length}. Returning infinite resistance.")
        return np.inf
    if length < constants.EPSILON: # Zero length segment has zero resistance if radius is valid
        return 0.0
        
    return (8.0 * viscosity * length) / (constants.PI * (radius**4))

def solve_1d_poiseuille_flow(
    graph: nx.DiGraph, 
    config: dict, 
    root_pressure_val: Optional[float] = None, 
    terminal_flows_val: Optional[Dict[str, float]] = None 
    ) -> nx.DiGraph:
    """Solves for nodal pressures and segmental flows in a vascular graph using Poiseuille's law.

    Assumes a directed graph where edges represent vessel segments.
    Nodes require 'type' attribute ('measured_root', 'synthetic_terminal', 'synthetic_bifurcation', etc.).
    Edges require 'length' and 'radius' attributes.
    Root nodes get fixed pressure boundary conditions.
    Terminal nodes act as flow sinks with specified outflow (demand).

    Args:
        graph (nx.DiGraph): The vascular graph.
        config (dict): Simulation configuration dictionary.
        root_pressure_val (Optional[float]): If provided, use this pressure for all roots,
                                             overriding config.
        terminal_flows_val (Optional[Dict[str, float]]): If provided, use this dict of 
                                                       {terminal_id: flow_out} for terminals,
                                                       overriding 'Q_flow' attribute on nodes.

    Returns:
        nx.DiGraph: The input graph annotated with 'pressure' on nodes 
                    and 'flow_solver' on edges.
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

    A_matrix = np.zeros((num_nodes, num_nodes), dtype=float)
    B_vector = np.zeros(num_nodes, dtype=float)

    for i, node_id_i in enumerate(node_list):
        node_i_data = graph.nodes[node_id_i]
        node_i_type = node_i_data.get('type', 'unknown')
        is_root_node = (node_i_type == 'measured_root' or 
                        (node_i_type == 'synthetic_terminal' and graph.in_degree(node_id_i) == 0 and not node_i_data.get('parent_id')) or # Seed point
                        node_i_data.get('is_flow_root', False))


        if is_root_node:
            A_matrix[i, i] = 1.0
            B_vector[i] = inlet_pressure
            logger.debug(f"Flow solver: Root node {node_id_i} set to pressure {inlet_pressure:.2f} Pa.")
            continue 

        sum_conductances_at_i = 0.0
        neighbors = list(graph.predecessors(node_id_i)) + list(graph.successors(node_id_i))
        unique_neighbors = list(set(neighbors)) # Ensure each neighbor processed once for matrix contribution

        for neighbor_id in unique_neighbors:
            if neighbor_id not in node_to_idx: # Should not happen if graph is consistent
                logger.error(f"Flow solver: Neighbor {neighbor_id} of {node_id_i} not in node_to_idx map. Skipping.")
                continue
            j = node_to_idx[neighbor_id]
            
            edge_data = None
            if graph.has_edge(node_id_i, neighbor_id):
                edge_data = graph.edges[node_id_i, neighbor_id]
            elif graph.has_edge(neighbor_id, node_id_i):
                edge_data = graph.edges[neighbor_id, node_id_i]
            
            if edge_data:
                length = edge_data.get('length', constants.EPSILON)
                radius = edge_data.get('radius', constants.MIN_VESSEL_RADIUS_MM)
                
                # Ensure radius used for resistance is from the segment itself, usually parent's radius
                # If edge_data has 'radius', use it, else infer from one of its nodes
                # The graph construction should ensure edge 'radius' is appropriate (e.g., parent node's radius)
                
                resistance = calculate_segment_resistance(length, radius, viscosity)
                
                if resistance < np.inf and resistance > constants.EPSILON:
                    conductance = 1.0 / resistance
                    A_matrix[i, j] -= conductance
                    sum_conductances_at_i += conductance
                elif resistance <= constants.EPSILON: # Near zero resistance
                    logger.warning(f"Edge between {node_id_i} and {neighbor_id} has near-zero resistance ({resistance}). Using high conductance.")
                    conductance = 1.0 / (constants.EPSILON * 10) # Very high conductance
                    A_matrix[i, j] -= conductance
                    sum_conductances_at_i += conductance
            else:
                 logger.warning(f"Flow solver: No edge data found between {node_id_i} and {neighbor_id} (neighbors from graph).")


        A_matrix[i, i] = sum_conductances_at_i # Diagonal term

        is_terminal_node = (node_i_type == 'synthetic_terminal' and graph.out_degree(node_id_i) == 0 and not is_root_node) or \
                           node_i_data.get('is_flow_terminal', False)

        if is_terminal_node:
            outflow = 0.0
            if terminal_flows_val and node_id_i in terminal_flows_val:
                outflow = terminal_flows_val[node_id_i]
            elif 'Q_flow' in node_i_data: 
                outflow = node_i_data['Q_flow'] 
            else:
                logger.warning(f"Flow solver: Terminal node {node_id_i} has no Q_flow attribute or provided outflow. Assuming zero outflow.")
            
            B_vector[i] -= outflow 
            logger.debug(f"Flow solver: Terminal node {node_id_i} set with outflow {outflow:.2e}.")

    try:
        # Check for singularity or ill-conditioning before solving
        if num_nodes > 0 and np.linalg.matrix_rank(A_matrix) < num_nodes:
            logger.error(f"Flow solver: Matrix A is singular or rank-deficient (rank {np.linalg.matrix_rank(A_matrix)} for {num_nodes} nodes). Cannot solve. "
                         "Check for disconnected graph components not attached to a pressure BC or floating sections.")
            # Set NaNs and return
            for node_id_err in node_list: graph.nodes[node_id_err]['pressure'] = np.nan
            for u, v, data_err in graph.edges(data=True): data_err['flow_solver'] = np.nan
            return graph
            
        node_pressures_vec = np.linalg.solve(A_matrix, B_vector)
        logger.info("Flow solver: Successfully solved for nodal pressures.")
    except np.linalg.LinAlgError as e:
        logger.error(f"Flow solver: Linear algebra error during pressure solution: {e}")
        logger.error(f"Matrix A Condition Number: {np.linalg.cond(A_matrix) if num_nodes > 0 and np.linalg.matrix_rank(A_matrix) == num_nodes else 'N/A or singular'}")
        for node_id_err in node_list: graph.nodes[node_id_err]['pressure'] = np.nan
        for u, v, data_err in graph.edges(data=True): data_err['flow_solver'] = np.nan
        return graph

    for i, node_id in enumerate(node_list):
        graph.nodes[node_id]['pressure'] = node_pressures_vec[i]

    for u_id, v_id, data in graph.edges(data=True):
        # Edges are directed; flow is from u to v if P_u > P_v
        # The graph's edge direction should ideally align with flow direction for interpretation,
        # but the solver calculates based on pressure difference regardless of DiGraph edge direction.
        # For consistent Q_ij = (P_i - P_j) / R_ij, i should be upstream of j.
        # Let's assume the graph edges u->v represent potential flow direction for now.
        P_u = graph.nodes[u_id]['pressure']
        P_v = graph.nodes[v_id]['pressure']
        
        length = data.get('length', constants.EPSILON)
        radius = data.get('radius', constants.MIN_VESSEL_RADIUS_MM) # Radius of the segment (often parent's radius)
        resistance = calculate_segment_resistance(length, radius, viscosity)
        
        flow = 0.0
        if resistance < np.inf and resistance > constants.EPSILON:
            flow = (P_u - P_v) / resistance
        elif resistance <= constants.EPSILON: # Near zero resistance
            if not np.isclose(P_u, P_v):
                 logger.warning(f"Edge {u_id}->{v_id} has zero resistance but pressure diff {P_u-P_v:.2e}. Flow could be ill-defined.")
                 # This can indicate short circuit, flow would be very large.
                 # Or if P_u approx P_v, flow is small.
                 flow = (P_u - P_v) / (constants.EPSILON *10) # Assign a very small resistance
            # else flow remains 0.0
        # If resistance is np.inf, flow remains 0.0
            
        data['flow_solver'] = flow
        logger.debug(f"Edge {u_id}->{v_id}: P_u={P_u:.2f}, P_v={P_v:.2f}, R={resistance:.2e}, Q_solver={flow:.2e}")

    logger.info("Flow solver: Finished annotating graph with pressures and flows.")
    return graph