# tests/test_perfusion_solver.py
import pytest
import numpy as np
import networkx as nx
import logging

from src import perfusion_solver, data_structures, constants, config_manager # Added config_manager

@pytest.fixture
def basic_config_flow_test(): # Renamed for clarity
    """Provides a basic config for perfusion solver tests."""
    # Ensure all necessary keys that get_param might look for are present, even if using defaults
    return {
        "vascular_properties": {
            "blood_viscosity": 0.0035, 
            "min_radius": 0.001,
            # Add other vascular_properties if get_param in solver might access them with defaults
            "k_murray_scaling_factor": 0.5, # Example, not directly used by solver but good practice
            "murray_law_exponent": 3.0,
             "min_segment_length": 0.01,
             "max_segment_length": 2.0,
             "initial_terminal_flow": 1e-6,
        },
        "perfusion_solver": {
            "inlet_pressure": 10000.0 
        },
        "gbo_growth": { # Add this section even if empty, as get_param might traverse
             "energy_coefficient_C_met_vessel_wall": 1.0e-5,
        }
    }

def test_calculate_segment_resistance():
    viscosity = 0.0035
    expected_R = (8 * viscosity * 1.0) / (constants.PI * (0.1**4))
    assert np.isclose(perfusion_solver.calculate_segment_resistance(1.0, 0.1, viscosity), expected_R)
    assert perfusion_solver.calculate_segment_resistance(1.0, 0.0, viscosity) == np.inf
    assert perfusion_solver.calculate_segment_resistance(1.0, constants.EPSILON/10, viscosity) == np.inf
    assert perfusion_solver.calculate_segment_resistance(0.0, 0.1, viscosity) == 0.0

def test_solve_single_segment_pressure_bc(basic_config_flow_test):
    graph = data_structures.create_empty_vascular_graph()
    config = basic_config_flow_test
    viscosity = config_manager.get_param(config, "vascular_properties.blood_viscosity")
    
    P_in_val = 100.0
    P_out_val = 50.0 # Target outlet pressure
    L_val, R_val = 10.0, 0.5 

    data_structures.add_node_to_graph(graph, "n0", pos=np.array([0,0,0]), radius=R_val, type='measured_root')
    # For this test to work with current solver (flow sinks), n1 needs to be a flow sink.
    # We calculate the expected flow if P_out was 50, and set that as the sink.
    data_structures.add_node_to_graph(graph, "n1", pos=np.array([L_val,0,0]), radius=R_val, type='synthetic_terminal')
    
    # Edge radius should be that of the upstream node for consistency with how GBO might set it
    data_structures.add_edge_to_graph(graph, "n0", "n1", length=L_val, radius=R_val) 

    resistance = perfusion_solver.calculate_segment_resistance(L_val, R_val, viscosity)
    expected_flow = (P_in_val - P_out_val) / resistance
    
    graph.nodes['n1']['Q_flow'] = expected_flow # Set n1 as a flow sink with this value
    
    solved_graph = perfusion_solver.solve_1d_poiseuille_flow(graph, config, root_pressure_val=P_in_val)

    assert np.isclose(solved_graph.nodes['n0']['pressure'], P_in_val)
    # The pressure at n1 will be P_in - Q*R. If Q is set correctly, P_n1 should be P_out_val.
    assert np.isclose(solved_graph.nodes['n1']['pressure'], P_out_val, atol=1e-1) 
    assert np.isclose(solved_graph.edges[('n0','n1')]['flow_solver'], expected_flow, rtol=1e-3)


def test_solve_y_bifurcation_flow_sinks(basic_config_flow_test):
    graph = data_structures.create_empty_vascular_graph()
    config = basic_config_flow_test
    P_in = config_manager.get_param(config, "perfusion_solver.inlet_pressure")
    visc = config_manager.get_param(config, "vascular_properties.blood_viscosity")

    L01, R01 = 5.0, 0.3
    L1T1, R1T1 = 3.0, 0.2
    L1T2, R1T2 = 4.0, 0.25

    data_structures.add_node_to_graph(graph, "P0", pos=np.array([0,0,0]), radius=R01, type='measured_root') # CORRECTED
    data_structures.add_node_to_graph(graph, "P1", pos=np.array([L01,0,0]), radius=R01, type='synthetic_bifurcation') # CORRECTED
    data_structures.add_node_to_graph(graph, "T1", pos=np.array([L01+L1T1,1,0]), radius=R1T1, type='synthetic_terminal') # CORRECTED
    data_structures.add_node_to_graph(graph, "T2", pos=np.array([L01+L1T2,-1,0]), radius=R1T2, type='synthetic_terminal') # 

    # Edges store their own radius (typically parent's radius or average)
    data_structures.add_edge_to_graph(graph, "P0", "P1", length=L01, radius=R01)
    data_structures.add_edge_to_graph(graph, "P1", "T1", length=L1T1, radius=R1T1) # Child segment gets its own radius
    data_structures.add_edge_to_graph(graph, "P1", "T2", length=L1T2, radius=R1T2) # Child segment gets its own radius

    Q_out_T1 = 0.002 
    Q_out_T2 = 0.003 
    graph.nodes["T1"]['Q_flow'] = Q_out_T1
    graph.nodes["T2"]['Q_flow'] = Q_out_T2

    solved_graph = perfusion_solver.solve_1d_poiseuille_flow(graph, config)

    Res01 = perfusion_solver.calculate_segment_resistance(L01, R01, visc)
    Res1T1 = perfusion_solver.calculate_segment_resistance(L1T1, R1T1, visc)
    Res1T2 = perfusion_solver.calculate_segment_resistance(L1T2, R1T2, visc)

    exp_Q01 = Q_out_T1 + Q_out_T2
    exp_P_P1 = P_in - exp_Q01 * Res01
    exp_P_T1 = exp_P_P1 - Q_out_T1 * Res1T1
    exp_P_T2 = exp_P_P1 - Q_out_T2 * Res1T2

    assert np.isclose(solved_graph.nodes['P0']['pressure'], P_in, rtol=1e-3)
    assert np.isclose(solved_graph.nodes['P1']['pressure'], exp_P_P1, rtol=1e-3)
    assert np.isclose(solved_graph.nodes['T1']['pressure'], exp_P_T1, rtol=1e-3)
    assert np.isclose(solved_graph.nodes['T2']['pressure'], exp_P_T2, rtol=1e-3)

    assert np.isclose(solved_graph.edges[('P0','P1')]['flow_solver'], exp_Q01, rtol=1e-3)
    assert np.isclose(solved_graph.edges[('P1','T1')]['flow_solver'], Q_out_T1, rtol=1e-3)
    assert np.isclose(solved_graph.edges[('P1','T2')]['flow_solver'], Q_out_T2, rtol=1e-3)

def test_solve_disconnected_component_no_bc(basic_config_flow_test, caplog): # Add caplog
    """Test that solver handles a component not connected to a pressure BC."""
    graph = data_structures.create_empty_vascular_graph()
    config = basic_config_flow_test

    # Component 1 (with BC)
    data_structures.add_node_to_graph(graph, "R0", pos=np.array([0,0,0]), radius=0.1, type='measured_root')
    data_structures.add_node_to_graph(graph, "T0", pos=np.array([1,0,0]), radius=0.1, type='synthetic_terminal', Q_flow=0.001)
    data_structures.add_edge_to_graph(graph, "R0", "T0", length=1.0, radius=0.1)

    # Component 2 (floating, no BC)
    data_structures.add_node_to_graph(graph, "F1", pos=np.array([10,0,0]), radius=0.1, type='synthetic_bifurcation')
    data_structures.add_node_to_graph(graph, "F2", pos=np.array([11,0,0]), radius=0.1, type='synthetic_terminal', Q_flow=0.0005)
    data_structures.add_edge_to_graph(graph, "F1", "F2", length=1.0, radius=0.1)

    # Capture logs to verify the singularity message
    with caplog.at_level(logging.ERROR, logger="src.perfusion_solver"):
        solved_graph = perfusion_solver.solve_1d_poiseuille_flow(graph, config)

    # Check if the singularity error was logged
    assert any("Matrix A is singular or rank-deficient" in record.message for record in caplog.records), \
        "Expected singularity error was not logged."

    # When matrix is singular, current solver sets all pressures and flows to NaN
    assert np.isnan(solved_graph.nodes['R0']['pressure'])
    assert np.isnan(solved_graph.nodes['T0']['pressure'])
    assert np.isnan(solved_graph.nodes['F1']['pressure'])
    assert np.isnan(solved_graph.nodes['F2']['pressure'])
    
    assert np.isnan(solved_graph.edges[('R0','T0')]['flow_solver'])
    assert np.isnan(solved_graph.edges[('F1','F2')]['flow_solver'])