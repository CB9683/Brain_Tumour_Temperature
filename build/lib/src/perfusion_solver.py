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

    for node_id_init in node_list:
        graph.nodes[node_id_init]['pressure'] = np.nan
    for u_init, v_init, data_init in graph.edges(data=True):
        data_init['flow_solver'] = np.nan

    if num_nodes == 0:
        return graph

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
            logger.debug(f"  BC Set (Pressure): Node {node_id_i} (type: {node_i_data.get('type')}) is flow_root. Row {i}: A[{i},{i}]=1, B[{i}]={inlet_pressure:.2f}")
            continue

        sum_conductances_at_i = 0.0
        physical_neighbors = set(graph.predecessors(node_id_i)) | set(graph.successors(node_id_i))
        current_row_terms_log = []

        for neighbor_id_j in physical_neighbors:
            if neighbor_id_j not in node_to_idx:
                logger.error(f"  Error: Neighbor {neighbor_id_j} of {node_id_i} not in node_to_idx map. Skipping.")
                continue
            j = node_to_idx[neighbor_id_j]

            edge_data = graph.get_edge_data(node_id_i, neighbor_id_j) or graph.get_edge_data(neighbor_id_j, node_id_i)

            if edge_data:
                length = edge_data.get('length', 0.0)
                radius = edge_data.get('radius', constants.MIN_VESSEL_RADIUS_MM)
                radius = max(radius, constants.MIN_VESSEL_RADIUS_MM * 0.01) 
                if length < constants.EPSILON:
                    length = constants.EPSILON

                resistance = calculate_segment_resistance(length, radius, viscosity)
                conductance = 0.0
                if resistance < np.inf and resistance > constants.EPSILON:
                    conductance = 1.0 / resistance
                elif resistance <= constants.EPSILON: 
                    conductance = 1.0 / (constants.EPSILON * 10)
                    logger.debug(f"  Edge involving {node_id_i}-{neighbor_id_j}: R~0, using high G={conductance:.1e}")

                if conductance > 0:
                    A_matrix[i, j] -= conductance 
                    sum_conductances_at_i += conductance
                    current_row_terms_log.append(f"G_({node_id_i}-{neighbor_id_j})={conductance:.2e} (to P_{neighbor_id_j})")
            else:
                 logger.warning(f"  No edge data found between physically connected {node_id_i} and {neighbor_id_j}.")

        A_matrix[i, i] = sum_conductances_at_i 

        is_graph_terminal_for_flow = (graph.out_degree(node_id_i) == 0 and not node_i_data.get('is_flow_root', False))

        if (node_i_data.get('type') == 'synthetic_terminal' and is_graph_terminal_for_flow) or \
           node_i_data.get('is_flow_terminal', False): 
            outflow_demand = 0.0
            if terminal_flows_val and node_id_i in terminal_flows_val:
                outflow_demand = terminal_flows_val[node_id_i]
            elif 'Q_flow' in node_i_data:
                outflow_demand = node_i_data['Q_flow']
            else:
                logger.warning(f"  Terminal node {node_id_i} (type: {node_i_data.get('type')}) has no Q_flow or provided outflow. Assuming zero demand.")

            B_vector[i] -= outflow_demand 
            num_defined_flow_sinks +=1
            logger.debug(f"  BC Set (Flow Sink): Node {node_id_i}. B[{i}] -= {outflow_demand:.2e} (Total B[{i}]={B_vector[i]:.2e})")
        elif B_vector[i] == 0.0 and not is_graph_terminal_for_flow and not node_i_data.get('is_flow_root', False): 
            logger.debug(f"  Matrix Row {i} ({node_id_i}, type: {node_i_data.get('type')}): Internal node. B[{i}]=0.")
        
        logger.debug(f"  Matrix Row {i} ({node_id_i}, type: {node_i_data.get('type')}): Diag A[{i},{i}]={A_matrix[i,i]:.2e}. Off-diag terms: {', '.join(current_row_terms_log) if current_row_terms_log else 'None'}. B[{i}]={B_vector[i]:.2e}")

    logger.debug(f"--- Flow Solver: System Matrix A (shape {A_matrix.shape}):\n{A_matrix if num_nodes < 10 else 'Too large to print'}")
    logger.debug(f"--- Flow Solver: System Vector B (shape {B_vector.shape}):\n{B_vector if num_nodes < 10 else 'Too large to print'}")
    logger.info(f"Flow Solver: Total defined pressure BCs (is_flow_root): {num_defined_pressure_bcs}")
    logger.info(f"Flow Solver: Total defined flow sinks (terminals with Q_flow): {num_defined_flow_sinks}")

    if num_defined_pressure_bcs == 0 and num_nodes > 0:
        logger.error("Flow solver: No pressure boundary conditions (is_flow_root=True) defined. Cannot solve. Pressures and flows will be NaN.")
        return graph

    node_pressures_vec = np.full(num_nodes, np.nan)

    try:
        rank_A = 0
        if num_nodes > 0:
            if np.any(np.isnan(A_matrix)) or np.any(np.isinf(A_matrix)):
                logger.critical("CRITICAL DIAGNOSIS: A_matrix contains NaN/Inf values BEFORE rank calculation. Problem with conductances (NaN radii/lengths?).")
                print("CRITICAL DIAGNOSIS: A_matrix contains NaN/Inf values BEFORE rank calculation.") # Force print
            else:
                 rank_A = np.linalg.matrix_rank(A_matrix)
        logger.debug(f"Flow Solver: Matrix A Rank = {rank_A} for {num_nodes} nodes.")

        if num_nodes > 0 and rank_A < num_nodes:
            logger.error(f"Flow solver: Matrix A is singular or rank-deficient (rank {rank_A} for {num_nodes} nodes). Pressures and flows will be NaN.")
            print(f"Flow solver: Matrix A is singular or rank-deficient (rank {rank_A} for {num_nodes} nodes).") # Force print
            logger.error("Investigating cause of rank deficiency:")
            print("Investigating cause of rank deficiency:") # Force print
            
            if np.any(np.isnan(A_matrix)) or np.any(np.isinf(A_matrix)):
                logger.error("  DIAGNOSIS (NAN/INF A): A_matrix contains NaN/Inf values. Problem with input graph data (radii, lengths).")
                print("  DIAGNOSIS (NAN/INF A): A_matrix contains NaN/Inf values.")
            if np.any(np.isnan(B_vector)) or np.any(np.isinf(B_vector)):
                logger.error("  DIAGNOSIS (NAN/INF B): B_vector contains NaN/Inf values. Problem with Q_flow for terminals.")
                print("  DIAGNOSIS (NAN/INF B): B_vector contains NaN/Inf values.")

            problem_rows_found = False
            for i_diag in range(num_nodes):
                node_id_diag = node_list[i_diag]
                is_root_diag = graph.nodes[node_id_diag].get('is_flow_root', False)
                
                # Check for all-zero rows (excluding root node equations which are P_i = C)
                if not is_root_diag and np.all(np.abs(A_matrix[i_diag, :]) < constants.EPSILON):
                    problem_rows_found = True
                    logger.error(f"  DIAGNOSIS (ZERO ROW): Row {i_diag} for node '{node_id_diag}' (type: {graph.nodes[node_id_diag].get('type')}) is all zeros in A_matrix.")
                    print(f"  DIAGNOSIS (ZERO ROW): Row {i_diag} for node '{node_id_diag}' is all zeros.")
                    # ... (rest of zero row logging as before)
                
                # Check for zero diagonal for non-root nodes
                elif not is_root_diag and abs(A_matrix[i_diag, i_diag]) < constants.EPSILON:
                    problem_rows_found = True
                    logger.error(f"  DIAGNOSIS (ZERO DIAG): Row {i_diag} for node '{node_id_diag}' (type: {graph.nodes[node_id_diag].get('type')}) has A[{i_diag},{i_diag}]=0. Sum of conductances is zero.")
                    print(f"  DIAGNOSIS (ZERO DIAG): Row {i_diag} for node '{node_id_diag}' has zero diagonal.")
                    logger.error(f"    Node '{node_id_diag}' Data: {graph.nodes[node_id_diag]}")
                    logger.error(f"    Node '{node_id_diag}' Degree: {graph.degree(node_id_diag)}, In: {graph.in_degree(node_id_diag)}, Out: {graph.out_degree(node_id_diag)}")
                    for neighbor_diag_zd in list(graph.predecessors(node_id_diag)) + list(graph.successors(node_id_diag)):
                        edge_diag_zd = graph.get_edge_data(node_id_diag, neighbor_diag_zd) or graph.get_edge_data(neighbor_diag_zd, node_id_diag)
                        if edge_diag_zd:
                             logger.error(f"      Connected to '{neighbor_diag_zd}' via edge with data: {edge_diag_zd}")
            
            if not problem_rows_found:
                logger.info("  DIAGNOSIS: No rows in A_matrix were found to be all-zero or have zero diagonals (for non-roots). Problem might be more complex graph structure or multiple interacting issues.")
                print("  DIAGNOSIS: No rows in A_matrix were found to be all-zero or have zero diagonals (for non-roots).")


            logger.info("  DIAGNOSIS: Checking for graph components without pressure BCs...")
            print("  DIAGNOSIS: Checking for graph components without pressure BCs...")
            try:
                if graph.is_directed():
                    undi_graph = graph.to_undirected(as_view=True) 
                else: 
                    undi_graph = graph 

                components_found_count = 0
                floating_components_found_count = 0
                for component_nodes in nx.connected_components(undi_graph):
                    components_found_count +=1
                    has_pressure_bc_in_comp = any(graph.nodes[comp_node_id].get('is_flow_root', False) for comp_node_id in component_nodes)
                    if not has_pressure_bc_in_comp:
                        floating_components_found_count += 1
                        comp_node_list_str = list(component_nodes)
                        logger.error(f"  DIAGNOSIS (FLOATING COMP): Component {floating_components_found_count} (size {len(comp_node_list_str)}) is FLOATING (no pressure BC). Nodes: {comp_node_list_str[:10]}{'...' if len(comp_node_list_str)>10 else ''}")
                        print(f"  DIAGNOSIS (FLOATING COMP): Component {floating_components_found_count} (size {len(comp_node_list_str)}) is FLOATING. Nodes: {comp_node_list_str[:5]}...")
                        # Check if this floating component has any flow sinks
                        has_flow_sinks_in_comp = any(
                            (graph.nodes[cn_id].get('type') == 'synthetic_terminal' and graph.out_degree(cn_id) == 0) or \
                            graph.nodes[cn_id].get('is_flow_terminal', False)
                            for cn_id in component_nodes if not graph.nodes[cn_id].get('is_flow_root', False)
                        )
                        if has_flow_sinks_in_comp:
                            logger.error(f"    Floating Component {floating_components_found_count} HAS flow sinks.")
                            print(f"    Floating Component {floating_components_found_count} HAS flow sinks.")
                        else:
                            logger.error(f"    Floating Component {floating_components_found_count} HAS NO flow sinks.")
                            print(f"    Floating Component {floating_components_found_count} HAS NO flow sinks.")
                if components_found_count == 0:
                    logger.error("  DIAGNOSIS: nx.connected_components found NO components (graph might be empty or unusual).")
                    print("  DIAGNOSIS: nx.connected_components found NO components.")
                elif floating_components_found_count == 0:
                    logger.info(f"  DIAGNOSIS: All {components_found_count} graph component(s) appear to have a pressure BC.")
                    print(f"  DIAGNOSIS: All {components_found_count} graph component(s) appear to have a pressure BC.")
            except Exception as e_comp:
                logger.error(f"  DIAGNOSIS: Error during connected components check: {e_comp}")
                print(f"  DIAGNOSIS: Error during connected components check: {e_comp}")
        else: 
            node_pressures_vec = np.linalg.solve(A_matrix, B_vector)
            logger.info("Flow solver: Successfully solved for nodal pressures.")

    except np.linalg.LinAlgError as e:
        logger.error(f"Flow solver: Linear algebra error during pressure solution: {e}", exc_info=False)
        if num_nodes > 0 and not (np.any(np.isnan(A_matrix)) or np.any(np.isinf(A_matrix))):
             try:
                 # Check if A_matrix is square and finite before calling cond
                 if A_matrix.shape[0] == A_matrix.shape[1] and np.all(np.isfinite(A_matrix)):
                     logger.error(f"Matrix A Condition Number: {np.linalg.cond(A_matrix)}")
                 else:
                     logger.error("Matrix A is not square or contains non-finite values, cannot compute condition number.")
             except Exception as e_cond:
                 logger.error(f"Could not compute condition number for A: {e_cond}")

    for i, node_id in enumerate(node_list):
        graph.nodes[node_id]['pressure'] = node_pressures_vec[i]

    for u_id, v_id, data in graph.edges(data=True):
        P_u = graph.nodes[u_id]['pressure']
        P_v = graph.nodes[v_id]['pressure']

        if np.isnan(P_u) or np.isnan(P_v):
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
        elif resistance <= constants.EPSILON: 
            if not np.isclose(P_u, P_v, atol=1e-3):
                 logger.warning(f"Edge {u_id}->{v_id} has R~0 ({resistance:.1e}) but P_diff {(P_u-P_v):.2e}. Flow may be very large/ill-defined.")
                 flow_on_edge = (P_u - P_v) / (constants.EPSILON * 10)
        elif resistance == np.inf:
            flow_on_edge = 0.0

        data['flow_solver'] = flow_on_edge
        if logger.isEnabledFor(logging.DEBUG):
             logger.debug(f"Edge {u_id}->{v_id}: P_u={P_u:.2f}, P_v={P_v:.2f}, L={length:.3f}, R_edge={radius:.4f}, Res={resistance:.2e}, Q_solver={flow_on_edge:.2e}")

    logger.info("Flow solver: Finished annotating graph with pressures and flows.")
    return graph