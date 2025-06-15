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
    if radius < constants.EPSILON: # Avoid division by zero or extremely small radius
        return np.inf
    if length < constants.EPSILON: # Segment with negligible length
        # Return a very small resistance to ensure conductance is high but finite
        return constants.EPSILON
    return (8.0 * viscosity * length) / (constants.PI * (radius**4))

def solve_1d_poiseuille_flow(
    graph: nx.DiGraph,
    config: dict,
    root_pressure_val: Optional[float] = None,
    terminal_flows_val: Optional[Dict[str, float]] = None
) -> nx.DiGraph:
    """Solves for nodal pressures and segmental flows in a vascular graph.

    Modifies the input graph in-place by adding/updating:
    - 'pressure' attribute to nodes.
    - 'flow_solver' attribute to edges.

    Assumptions on input graph:
    - Nodes:
        - Must have a unique ID.
        - Pressure inlet nodes must have `is_flow_root=True`.
        - Flow sink nodes (terminals) must have a 'Q_flow' attribute (target demand/outflow)
          and be identifiable (e.g., `type='synthetic_terminal'` and `graph.out_degree(node_id) == 0`,
          or an explicit `is_flow_terminal=True`).
    - Edges:
        - Must have 'length' and 'radius' attributes. The 'radius' attribute on an edge
          typically represents the radius of the upstream segment/node.
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

    # Initialize graph attributes in case of early exit or errors
    for node_id_init in node_list:
        graph.nodes[node_id_init]['pressure'] = np.nan
    for u_init, v_init, data_init in graph.edges(data=True):
        data_init['flow_solver'] = np.nan

    if num_nodes == 0: # Should have been caught, but as a safeguard
        return graph

    A_matrix = np.zeros((num_nodes, num_nodes), dtype=float)
    B_vector = np.zeros(num_nodes, dtype=float)
    num_defined_pressure_bcs = 0
    num_defined_flow_sinks = 0

    logger.debug("--- Flow Solver: Assembling System Matrix A and Vector B ---")
    for i, node_id_i in enumerate(node_list):
        node_i_data = graph.nodes[node_id_i]

        # --- Set up equation for node i ---
        if node_i_data.get('is_flow_root', False):
            # Pressure Boundary Condition: P_i = inlet_pressure
            A_matrix[i, :] = 0.0
            A_matrix[i, i] = 1.0
            B_vector[i] = inlet_pressure
            num_defined_pressure_bcs += 1
            logger.debug(f"  BC Set (Pressure): Node {node_id_i} (type: {node_i_data.get('type')}) is flow_root. Row {i}: A[{i},{i}]=1, B[{i}]={inlet_pressure:.2f}")
            continue # Move to the next node

        # Flow Conservation Equation for non-root nodes: sum(G_ij * (P_j - P_i)) = Q_source_sink_i
        # Rearranged: sum(G_ij * P_j) - (sum(G_ij)) * P_i = Q_source_sink_i
        # So, A[i,i] = -sum(G_ij) and A[i,j] = G_ij.
        # Or, (sum(G_ij)) * P_i - sum(G_ij * P_j) = -Q_source_sink_i
        # So, A[i,i] = sum_over_j(G_ij) and A[i,j] = -G_ij
        # The current implementation uses the second form for A_matrix terms.

        sum_conductances_at_i = 0.0
        # Consider all physical neighbors (connected by an edge, regardless of direction in DiGraph)
        physical_neighbors = set(graph.predecessors(node_id_i)) | set(graph.successors(node_id_i))
        current_row_terms_log = []

        for neighbor_id_j in physical_neighbors:
            if neighbor_id_j not in node_to_idx:
                logger.error(f"  Error: Neighbor {neighbor_id_j} of {node_id_i} not in node_to_idx map. Skipping.")
                continue
            j = node_to_idx[neighbor_id_j]

            edge_data = None
            if graph.has_edge(node_id_i, neighbor_id_j):
                edge_data = graph.edges[node_id_i, neighbor_id_j]
            elif graph.has_edge(neighbor_id_j, node_id_i):
                edge_data = graph.edges[neighbor_id_j, node_id_i]

            if edge_data:
                length = edge_data.get('length', 0.0)
                # Radius for resistance calculation is taken from the edge attribute
                radius = edge_data.get('radius', constants.MIN_VESSEL_RADIUS_MM)
                radius = max(radius, constants.MIN_VESSEL_RADIUS_MM * 0.01) # Ensure tiny positive
                if length < constants.EPSILON:
                    length = constants.EPSILON # Use tiny length for zero-length segments

                resistance = calculate_segment_resistance(length, radius, viscosity)
                conductance = 0.0
                if resistance < np.inf and resistance > constants.EPSILON:
                    conductance = 1.0 / resistance
                elif resistance <= constants.EPSILON: # Effectively zero resistance
                    conductance = 1.0 / (constants.EPSILON * 10) # Very high conductance
                    logger.debug(f"  Edge involving {node_id_i}-{neighbor_id_j}: R~0, using high G={conductance:.1e}")

                if conductance > 0:
                    A_matrix[i, j] -= conductance # Off-diagonal term: -G_ij
                    sum_conductances_at_i += conductance
                    current_row_terms_log.append(f"G_({node_id_i}-{neighbor_id_j})={conductance:.2e} (to P_{neighbor_id_j})")
            else:
                 logger.warning(f"  No edge data found between physically connected {node_id_i} and {neighbor_id_j}.")

        A_matrix[i, i] = sum_conductances_at_i # Diagonal term: sum_over_j(G_ij)

        # Handle flow sinks (outflow boundary condition)
        # A terminal for flow sink is a node with no outgoing segments AND it's not a pressure root.
        is_graph_terminal_for_flow = (graph.out_degree(node_id_i) == 0 and not node_i_data.get('is_flow_root', False))

        if (node_i_data.get('type') == 'synthetic_terminal' and is_graph_terminal_for_flow) or \
           node_i_data.get('is_flow_terminal', False): # Explicitly marked as a flow terminal
            outflow_demand = 0.0
            if terminal_flows_val and node_id_i in terminal_flows_val: # Prioritize explicitly passed terminal flows
                outflow_demand = terminal_flows_val[node_id_i]
            elif 'Q_flow' in node_i_data: # Fallback to Q_flow attribute on the node
                outflow_demand = node_i_data['Q_flow']
            else:
                logger.warning(f"  Terminal node {node_id_i} (type: {node_i_data.get('type')}) has no Q_flow or provided outflow. Assuming zero demand.")

            B_vector[i] -= outflow_demand # Flow leaving node i is -Q on RHS (sum(G_ij*P_j) - sum(G_ij)*P_i = -Q_out)
            num_defined_flow_sinks +=1
            logger.debug(f"  BC Set (Flow Sink): Node {node_id_i}. B[{i}] -= {outflow_demand:.2e} (Total B[{i}]={B_vector[i]:.2e})")
        elif B_vector[i] == 0.0 and not is_graph_terminal_for_flow and not node_i_data.get('is_flow_root', False): # Internal node with no explicit source/sink
            logger.debug(f"  Matrix Row {i} ({node_id_i}, type: {node_i_data.get('type')}): Internal node. B[{i}]=0.")
        
        logger.debug(f"  Matrix Row {i} ({node_id_i}, type: {node_i_data.get('type')}): Diag A[{i},{i}]={A_matrix[i,i]:.2e}. Off-diag terms: {', '.join(current_row_terms_log) if current_row_terms_log else 'None'}. B[{i}]={B_vector[i]:.2e}")


    logger.debug(f"--- Flow Solver: System Matrix A (shape {A_matrix.shape}):\n{A_matrix if num_nodes < 10 else 'Too large to print'}")
    logger.debug(f"--- Flow Solver: System Vector B (shape {B_vector.shape}):\n{B_vector if num_nodes < 10 else 'Too large to print'}")
    logger.info(f"Flow Solver: Total defined pressure BCs (is_flow_root): {num_defined_pressure_bcs}")
    logger.info(f"Flow Solver: Total defined flow sinks (terminals with Q_flow): {num_defined_flow_sinks}")

    if num_defined_pressure_bcs == 0 and num_nodes > 0:
        logger.error("Flow solver: No pressure boundary conditions (is_flow_root=True) defined. Cannot solve. Pressures and flows will be NaN.")
        return graph # Graph attributes already initialized to NaN

    node_pressures_vec = np.full(num_nodes, np.nan) # Default to NaN

    try:
        rank_A = 0
        if num_nodes > 0:
            rank_A = np.linalg.matrix_rank(A_matrix)
        logger.debug(f"Flow Solver: Matrix A Rank = {rank_A} for {num_nodes} nodes.")

        if num_nodes > 0 and rank_A < num_nodes:
            logger.error(f"Flow solver: Matrix A is singular or rank-deficient (rank {rank_A} for {num_nodes} nodes). Cannot solve. "
                         "Check for disconnected graph components not attached to a pressure BC, or insufficient/inconsistent outflow conditions. "
                         "Pressures and flows will be NaN.")
            # Graph attributes already initialized to NaN
        else:
            node_pressures_vec = np.linalg.solve(A_matrix, B_vector)
            logger.info("Flow solver: Successfully solved for nodal pressures.")
            # logger.debug(f"Solved Node Pressures Vector:\n{node_pressures_vec}")

    except np.linalg.LinAlgError as e:
        logger.error(f"Flow solver: Linear algebra error during pressure solution: {e}", exc_info=False)
        # Log condition number only if matrix is likely well-formed enough for it
        if num_nodes > 0 and rank_A == num_nodes and not np.all(A_matrix == 0):
             logger.error(f"Matrix A Condition Number: {np.linalg.cond(A_matrix)}")
        # Graph attributes already initialized to NaN

    # Assign solved pressures to graph nodes
    for i, node_id in enumerate(node_list):
        graph.nodes[node_id]['pressure'] = node_pressures_vec[i] # Will be NaN if solve failed or matrix was singular

    # Calculate flows on edges based on solved pressures
    for u_id, v_id, data in graph.edges(data=True):
        P_u = graph.nodes[u_id]['pressure']
        P_v = graph.nodes[v_id]['pressure']

        if np.isnan(P_u) or np.isnan(P_v): # If pressures couldn't be solved, flow is also unknown
            data['flow_solver'] = np.nan
            continue

        length = data.get('length', 0.0)
        radius = data.get('radius', constants.MIN_VESSEL_RADIUS_MM)
        radius = max(radius, constants.MIN_VESSEL_RADIUS_MM * 0.01)
        if length < constants.EPSILON:
            length = constants.EPSILON

        resistance = calculate_segment_resistance(length, radius, viscosity)
        flow_on_edge = 0.0
        if resistance < np.inf and resistance > constants.EPSILON:
            flow_on_edge = (P_u - P_v) / resistance
        elif resistance <= constants.EPSILON: # Near zero resistance
            # If pressures are different, flow could be very large.
            # This case implies P_u should be very close to P_v.
            if not np.isclose(P_u, P_v, atol=1e-3): # Allow small tolerance for numerical error
                 logger.warning(f"Edge {u_id}->{v_id} has R~0 ({resistance:.1e}) but P_diff {(P_u-P_v):.2e}. Flow may be very large/ill-defined.")
                 # Use a very high conductance to estimate flow, but this situation is often problematic.
                 flow_on_edge = (P_u - P_v) / (constants.EPSILON * 10)
            # else, if P_u is close to P_v, flow is near zero, which is fine.
        elif resistance == np.inf: # Infinite resistance, zero flow
            flow_on_edge = 0.0


        data['flow_solver'] = flow_on_edge
        if logger.isEnabledFor(logging.DEBUG):
             logger.debug(f"Edge {u_id}->{v_id}: P_u={P_u:.2f}, P_v={P_v:.2f}, L={length:.3f}, R_edge={radius:.4f}, Res={resistance:.2e}, Q_solver={flow_on_edge:.2e}")

    logger.info("Flow solver: Finished annotating graph with pressures and flows.")
    return graph