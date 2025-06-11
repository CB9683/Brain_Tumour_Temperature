# tests/test_energy_model.py
import pytest
import numpy as np
from typing import Optional
import logging
from src import energy_model, constants, config_manager, utils

# Define a common mock config for tests in this file
@pytest.fixture
def mock_energy_config():
    return {
        "vascular_properties": {
            "blood_viscosity": 0.004,  # Pa.s - Use a slightly different value for testing
            "min_radius": 0.001,       # mm
            "k_murray_scaling_factor": 0.6, # s^(1/3)
            "murray_law_exponent": 3.0,
            "min_segment_length": 0.01
        },
        "gbo_growth": {
            # Ensure this C_met is in units consistent with E_flow from calculate_segment_flow_energy
            # If E_flow inputs (L,R,Q) are mm, mm, mm^3/s and mu is Pa.s, E_flow is in (Pa.mm^3/s) or (10^-9 W)
            # Then C_met here should be in (Pa.mm^3/s) / mm^3 = Pa/s (if we treat energy as power)
            # Or if energy is just a "cost value", units need to be balanced by user.
            # Let's pick a value for testing.
            "energy_coefficient_C_met_vessel_wall": 100.0, # Arbitrary units, assuming consistency
            "bifurcation_candidate_points": 5 
        }
    }

# --- Tests for calculate_segment_flow_energy ---
def test_calc_flow_energy_known_values(mock_energy_config):
    viscosity = mock_energy_config["vascular_properties"]["blood_viscosity"]
    # L=1mm, R=0.1mm, Q=0.01 mm^3/s
    length, radius, flow = 1.0, 0.1, 0.01
    # Expected: (8 * 0.004 * 1.0 * 0.01^2) / (pi * 0.1^4)
    # = (8 * 0.004 * 1e-4) / (pi * 1e-4) = (3.2e-6) / (pi * 1e-4) = 3.2e-2 / pi 
    expected_energy = (8 * viscosity * length * flow**2) / (constants.PI * radius**4)
    assert np.isclose(
        energy_model.calculate_segment_flow_energy(length, radius, flow, viscosity),
        expected_energy
    )

def test_calc_flow_energy_zero_radius(mock_energy_config):
    viscosity = mock_energy_config["vascular_properties"]["blood_viscosity"]
    assert energy_model.calculate_segment_flow_energy(1.0, 0.0, 0.01, viscosity) == np.inf
    assert energy_model.calculate_segment_flow_energy(1.0, constants.EPSILON / 2, 0.01, viscosity) == np.inf

def test_calc_flow_energy_zero_flow(mock_energy_config):
    viscosity = mock_energy_config["vascular_properties"]["blood_viscosity"]
    assert energy_model.calculate_segment_flow_energy(1.0, 0.1, 0.0, viscosity) == 0.0

def test_calc_flow_energy_zero_length(mock_energy_config):
    viscosity = mock_energy_config["vascular_properties"]["blood_viscosity"]
    assert energy_model.calculate_segment_flow_energy(0.0, 0.1, 0.01, viscosity) == 0.0

# --- Tests for calculate_segment_metabolic_energy ---
def test_calc_metabolic_energy_known_values(mock_energy_config):
    c_met = mock_energy_config["gbo_growth"]["energy_coefficient_C_met_vessel_wall"]
    length, radius = 1.0, 0.1
    # Expected: c_met * pi * 0.1^2 * 1.0
    expected_energy = c_met * constants.PI * radius**2 * length
    assert np.isclose(
        energy_model.calculate_segment_metabolic_energy(length, radius, c_met),
        expected_energy
    )

def test_calc_metabolic_energy_zero_radius(mock_energy_config):
    c_met = mock_energy_config["gbo_growth"]["energy_coefficient_C_met_vessel_wall"]
    assert energy_model.calculate_segment_metabolic_energy(1.0, 0.0, c_met) == 0.0

def test_calc_metabolic_energy_zero_length(mock_energy_config):
    c_met = mock_energy_config["gbo_growth"]["energy_coefficient_C_met_vessel_wall"]
    assert energy_model.calculate_segment_metabolic_energy(0.0, 0.1, c_met) == 0.0

# --- Tests for calculate_bifurcation_loss ---
def test_calc_bifurcation_loss_basic(mock_energy_config):
    parent_pos = np.array([0,0,0])
    c1_pos, c1_r, c1_q = np.array([1,0,0]), 0.1, 0.01
    c2_pos, c2_r, c2_q = np.array([0,1,0]), 0.08, 0.008
    
    viscosity = mock_energy_config["vascular_properties"]["blood_viscosity"]
    c_met = mock_energy_config["gbo_growth"]["energy_coefficient_C_met_vessel_wall"]

    l_c1 = utils.distance(parent_pos, c1_pos)
    e_flow_c1 = energy_model.calculate_segment_flow_energy(l_c1, c1_r, c1_q, viscosity)
    e_met_c1 = energy_model.calculate_segment_metabolic_energy(l_c1, c1_r, c_met)

    l_c2 = utils.distance(parent_pos, c2_pos)
    e_flow_c2 = energy_model.calculate_segment_flow_energy(l_c2, c2_r, c2_q, viscosity)
    e_met_c2 = energy_model.calculate_segment_metabolic_energy(l_c2, c2_r, c_met)
    
    expected_loss = e_flow_c1 + e_met_c1 + e_flow_c2 + e_met_c2
    
    calculated_loss = energy_model.calculate_bifurcation_loss(
        parent_pos, c1_pos, c1_r, c1_q, c2_pos, c2_r, c2_q, mock_energy_config
    )
    assert np.isclose(calculated_loss, expected_loss)

def test_calc_bifurcation_loss_one_child_zero_flow(mock_energy_config):
    parent_pos = np.array([0,0,0])
    c1_pos, c1_r, c1_q = np.array([1,0,0]), 0.1, 0.01
    c2_pos, c2_r, c2_q = np.array([0,1,0]), 0.08, 0.0 # Child 2 has zero flow
    
    viscosity = mock_energy_config["vascular_properties"]["blood_viscosity"]
    c_met = mock_energy_config["gbo_growth"]["energy_coefficient_C_met_vessel_wall"]

    l_c1 = utils.distance(parent_pos, c1_pos)
    e_flow_c1 = energy_model.calculate_segment_flow_energy(l_c1, c1_r, c1_q, viscosity)
    e_met_c1 = energy_model.calculate_segment_metabolic_energy(l_c1, c1_r, c_met)

    # For child 2, flow energy should be 0, metabolic energy depends on radius/length
    l_c2 = utils.distance(parent_pos, c2_pos)
    e_flow_c2 = 0.0 
    e_met_c2 = energy_model.calculate_segment_metabolic_energy(l_c2, c2_r, c_met)
    
    expected_loss = e_flow_c1 + e_met_c1 + e_flow_c2 + e_met_c2
    
    calculated_loss = energy_model.calculate_bifurcation_loss(
        parent_pos, c1_pos, c1_r, c1_q, c2_pos, c2_r, c2_q, mock_energy_config
    )
    assert np.isclose(calculated_loss, expected_loss)

# --- Sanity Tests for find_optimal_bifurcation_for_new_region ---
@pytest.fixture
def mock_parent_terminal():
    # Using a simple class or dict for parent_terminal_gbo_data for the test
    class MockParent:
        id = "p_test"
        pos = np.array([0.,0.,0.])
        radius = 0.2 # Not directly used by find_optimal, but good for context
        flow = 0.01  # Flow to *existing* territory, also not directly used by find_optimal for *new* region
    return MockParent()

@pytest.fixture
def mock_tissue_data_for_bif_search():
    # Minimal tissue data for bifurcation search
    # Two distinct demand points in the "new growth region"
    voxel_vol = 0.001 # mm^3 (0.1mm sided voxels)
    
    # These are global flat indices for the new growth region
    new_growth_indices_flat = np.array([0, 1]) 
    
    # World coords for these two specific voxels
    # Place them such that KMeans or random selection should pick them or points near them
    new_growth_world_coords = np.array([
        [1.0, 0.5, 0.0], # Demand point 1
        [1.0, -0.5, 0.0] # Demand point 2
    ])
    
    # Corresponding 3D indices (dummy for this test, must match shape of demand map)
    new_growth_3d_indices = np.array([
        [0,0,0],
        [0,1,0]
    ])
    
    # Metabolic demand (q_met) for these points
    demand_q_met_values = np.array([0.02, 0.02]) # 1/s
    
    # Create a minimal 3D demand map that contains these
    demand_map_3d = np.zeros((1,2,1))
    demand_map_3d[new_growth_3d_indices[:,0], 
                  new_growth_3d_indices[:,1], 
                  new_growth_3d_indices[:,2]] = demand_q_met_values

    return {
        'world_coords_flat': new_growth_world_coords, # For this test, it *only* contains the new region voxels
        'voxel_indices_flat': new_growth_3d_indices, # Corresponding 3D indices for the new region voxels
        'metabolic_demand_map': demand_map_3d, # q_met map
        'voxel_volume': voxel_vol,
        'shape': demand_map_3d.shape # Shape of the minimal demand map
    }

def test_find_optimal_bif_empty_new_region(mock_parent_terminal, mock_tissue_data_for_bif_search, mock_energy_config):
    result = energy_model.find_optimal_bifurcation_for_new_region(
        mock_parent_terminal, 
        np.array([], dtype=int), # Empty new growth region
        mock_tissue_data_for_bif_search, 
        mock_energy_config,
        k_murray_factor=mock_energy_config["vascular_properties"]["k_murray_scaling_factor"],
        murray_exponent=mock_energy_config["vascular_properties"]["murray_law_exponent"]
    )
    assert result is None

def test_find_optimal_bif_negligible_demand(mock_parent_terminal, mock_tissue_data_for_bif_search, mock_energy_config):
    # Modify tissue data to have zero demand
    tissue_data_zero_demand = mock_tissue_data_for_bif_search.copy()
    tissue_data_zero_demand['metabolic_demand_map'] = np.zeros_like(tissue_data_zero_demand['metabolic_demand_map'])
    
    new_growth_indices = np.array([0,1]) # Referring to indices in this zero-demand setup

    result = energy_model.find_optimal_bifurcation_for_new_region(
        mock_parent_terminal, 
        new_growth_indices,
        tissue_data_zero_demand, 
        mock_energy_config,
        k_murray_factor=mock_energy_config["vascular_properties"]["k_murray_scaling_factor"],
        murray_exponent=mock_energy_config["vascular_properties"]["murray_law_exponent"]
    )
    assert result is None

def test_find_optimal_bif_basic_search(mock_parent_terminal, mock_tissue_data_for_bif_search, mock_energy_config, caplog):
    caplog.set_level(logging.DEBUG, logger="src.energy_model") # For verbose output from this test
    
    new_growth_indices = np.array([0,1]) # The two demand points defined in fixture

    result = energy_model.find_optimal_bifurcation_for_new_region(
        mock_parent_terminal, 
        new_growth_indices,
        mock_tissue_data_for_bif_search, 
        mock_energy_config,
        k_murray_factor=mock_energy_config["vascular_properties"]["k_murray_scaling_factor"],
        murray_exponent=mock_energy_config["vascular_properties"]["murray_law_exponent"]
    )
    assert result is not None, "Expected a bifurcation result for simple two-point demand"
    c1_pos, c1_r, c1_q, c2_pos, c2_r, c2_q, loss = result

    # Check child positions are somewhat reasonable (e.g., near the demand points)
    # Demand points were [1,0.5,0] and [1,-0.5,0]. Parent at [0,0,0].
    # Children should be somewhere around X=1.
    assert c1_pos[0] > 0.5 and c2_pos[0] > 0.5 
    
    # Check flows sum to total demand of the new region
    total_demand_q_met = np.sum(mock_tissue_data_for_bif_search['metabolic_demand_map'])
    total_demand_flow = total_demand_q_met * mock_tissue_data_for_bif_search['voxel_volume']
    assert np.isclose(c1_q + c2_q, total_demand_flow), "Child flows should sum to new region's total demand"

    # Check radii are positive and follow Murray's scaling roughly
    min_r = mock_energy_config["vascular_properties"]["min_radius"]
    assert c1_r >= min_r and c2_r >= min_r
    if c1_q > constants.EPSILON:
        expected_r1 = mock_energy_config["vascular_properties"]["k_murray_scaling_factor"] * (c1_q ** (1.0/3.0))
        assert np.isclose(c1_r, max(min_r, expected_r1), rtol=0.05) # Allow some tolerance due to min_radius clamping
    if c2_q > constants.EPSILON:
        expected_r2 = mock_energy_config["vascular_properties"]["k_murray_scaling_factor"] * (c2_q ** (1.0/3.0))
        assert np.isclose(c2_r, max(min_r, expected_r2), rtol=0.05)

    assert loss < np.inf and loss > 0 # Loss should be a finite positive number# tests/test_energy_model.py
import pytest
import numpy as np
from typing import Optional

from src import energy_model, constants, config_manager, utils

# Define a common mock config for tests in this file
@pytest.fixture
def mock_energy_config():
    return {
        "vascular_properties": {
            "blood_viscosity": 0.004,  # Pa.s - Use a slightly different value for testing
            "min_radius": 0.001,       # mm
            "k_murray_scaling_factor": 0.6, # s^(1/3)
            "murray_law_exponent": 3.0,
            "min_segment_length": 0.01
        },
        "gbo_growth": {
            # Ensure this C_met is in units consistent with E_flow from calculate_segment_flow_energy
            # If E_flow inputs (L,R,Q) are mm, mm, mm^3/s and mu is Pa.s, E_flow is in (Pa.mm^3/s) or (10^-9 W)
            # Then C_met here should be in (Pa.mm^3/s) / mm^3 = Pa/s (if we treat energy as power)
            # Or if energy is just a "cost value", units need to be balanced by user.
            # Let's pick a value for testing.
            "energy_coefficient_C_met_vessel_wall": 100.0, # Arbitrary units, assuming consistency
            "bifurcation_candidate_points": 5 
        }
    }

# --- Tests for calculate_segment_flow_energy ---
def test_calc_flow_energy_known_values(mock_energy_config):
    viscosity = mock_energy_config["vascular_properties"]["blood_viscosity"]
    # L=1mm, R=0.1mm, Q=0.01 mm^3/s
    length, radius, flow = 1.0, 0.1, 0.01
    # Expected: (8 * 0.004 * 1.0 * 0.01^2) / (pi * 0.1^4)
    # = (8 * 0.004 * 1e-4) / (pi * 1e-4) = (3.2e-6) / (pi * 1e-4) = 3.2e-2 / pi 
    expected_energy = (8 * viscosity * length * flow**2) / (constants.PI * radius**4)
    assert np.isclose(
        energy_model.calculate_segment_flow_energy(length, radius, flow, viscosity),
        expected_energy
    )

def test_calc_flow_energy_zero_radius(mock_energy_config):
    viscosity = mock_energy_config["vascular_properties"]["blood_viscosity"]
    assert energy_model.calculate_segment_flow_energy(1.0, 0.0, 0.01, viscosity) == np.inf
    assert energy_model.calculate_segment_flow_energy(1.0, constants.EPSILON / 2, 0.01, viscosity) == np.inf

def test_calc_flow_energy_zero_flow(mock_energy_config):
    viscosity = mock_energy_config["vascular_properties"]["blood_viscosity"]
    assert energy_model.calculate_segment_flow_energy(1.0, 0.1, 0.0, viscosity) == 0.0

def test_calc_flow_energy_zero_length(mock_energy_config):
    viscosity = mock_energy_config["vascular_properties"]["blood_viscosity"]
    assert energy_model.calculate_segment_flow_energy(0.0, 0.1, 0.01, viscosity) == 0.0

# --- Tests for calculate_segment_metabolic_energy ---
def test_calc_metabolic_energy_known_values(mock_energy_config):
    c_met = mock_energy_config["gbo_growth"]["energy_coefficient_C_met_vessel_wall"]
    length, radius = 1.0, 0.1
    # Expected: c_met * pi * 0.1^2 * 1.0
    expected_energy = c_met * constants.PI * radius**2 * length
    assert np.isclose(
        energy_model.calculate_segment_metabolic_energy(length, radius, c_met),
        expected_energy
    )

def test_calc_metabolic_energy_zero_radius(mock_energy_config):
    c_met = mock_energy_config["gbo_growth"]["energy_coefficient_C_met_vessel_wall"]
    assert energy_model.calculate_segment_metabolic_energy(1.0, 0.0, c_met) == 0.0

def test_calc_metabolic_energy_zero_length(mock_energy_config):
    c_met = mock_energy_config["gbo_growth"]["energy_coefficient_C_met_vessel_wall"]
    assert energy_model.calculate_segment_metabolic_energy(0.0, 0.1, c_met) == 0.0

# --- Tests for calculate_bifurcation_loss ---
def test_calc_bifurcation_loss_basic(mock_energy_config):
    parent_pos = np.array([0,0,0])
    c1_pos, c1_r, c1_q = np.array([1,0,0]), 0.1, 0.01
    c2_pos, c2_r, c2_q = np.array([0,1,0]), 0.08, 0.008
    
    viscosity = mock_energy_config["vascular_properties"]["blood_viscosity"]
    c_met = mock_energy_config["gbo_growth"]["energy_coefficient_C_met_vessel_wall"]

    l_c1 = utils.distance(parent_pos, c1_pos)
    e_flow_c1 = energy_model.calculate_segment_flow_energy(l_c1, c1_r, c1_q, viscosity)
    e_met_c1 = energy_model.calculate_segment_metabolic_energy(l_c1, c1_r, c_met)

    l_c2 = utils.distance(parent_pos, c2_pos)
    e_flow_c2 = energy_model.calculate_segment_flow_energy(l_c2, c2_r, c2_q, viscosity)
    e_met_c2 = energy_model.calculate_segment_metabolic_energy(l_c2, c2_r, c_met)
    
    expected_loss = e_flow_c1 + e_met_c1 + e_flow_c2 + e_met_c2
    
    calculated_loss = energy_model.calculate_bifurcation_loss(
        parent_pos, c1_pos, c1_r, c1_q, c2_pos, c2_r, c2_q, mock_energy_config
    )
    assert np.isclose(calculated_loss, expected_loss)

def test_calc_bifurcation_loss_one_child_zero_flow(mock_energy_config):
    parent_pos = np.array([0,0,0])
    c1_pos, c1_r, c1_q = np.array([1,0,0]), 0.1, 0.01
    c2_pos, c2_r, c2_q = np.array([0,1,0]), 0.08, 0.0 # Child 2 has zero flow
    
    viscosity = mock_energy_config["vascular_properties"]["blood_viscosity"]
    c_met = mock_energy_config["gbo_growth"]["energy_coefficient_C_met_vessel_wall"]

    l_c1 = utils.distance(parent_pos, c1_pos)
    e_flow_c1 = energy_model.calculate_segment_flow_energy(l_c1, c1_r, c1_q, viscosity)
    e_met_c1 = energy_model.calculate_segment_metabolic_energy(l_c1, c1_r, c_met)

    # For child 2, flow energy should be 0, metabolic energy depends on radius/length
    l_c2 = utils.distance(parent_pos, c2_pos)
    e_flow_c2 = 0.0 
    e_met_c2 = energy_model.calculate_segment_metabolic_energy(l_c2, c2_r, c_met)
    
    expected_loss = e_flow_c1 + e_met_c1 + e_flow_c2 + e_met_c2
    
    calculated_loss = energy_model.calculate_bifurcation_loss(
        parent_pos, c1_pos, c1_r, c1_q, c2_pos, c2_r, c2_q, mock_energy_config
    )
    assert np.isclose(calculated_loss, expected_loss)

# --- Sanity Tests for find_optimal_bifurcation_for_new_region ---
@pytest.fixture
def mock_parent_terminal():
    # Using a simple class or dict for parent_terminal_gbo_data for the test
    class MockParent:
        id = "p_test"
        pos = np.array([0.,0.,0.])
        radius = 0.2 # Not directly used by find_optimal, but good for context
        flow = 0.01  # Flow to *existing* territory, also not directly used by find_optimal for *new* region
    return MockParent()

@pytest.fixture
def mock_tissue_data_for_bif_search():
    # Minimal tissue data for bifurcation search
    # Two distinct demand points in the "new growth region"
    voxel_vol = 0.001 # mm^3 (0.1mm sided voxels)
    
    # These are global flat indices for the new growth region
    new_growth_indices_flat = np.array([0, 1]) 
    
    # World coords for these two specific voxels
    # Place them such that KMeans or random selection should pick them or points near them
    new_growth_world_coords = np.array([
        [1.0, 0.5, 0.0], # Demand point 1
        [1.0, -0.5, 0.0] # Demand point 2
    ])
    
    # Corresponding 3D indices (dummy for this test, must match shape of demand map)
    new_growth_3d_indices = np.array([
        [0,0,0],
        [0,1,0]
    ])
    
    # Metabolic demand (q_met) for these points
    demand_q_met_values = np.array([0.02, 0.02]) # 1/s
    
    # Create a minimal 3D demand map that contains these
    demand_map_3d = np.zeros((1,2,1))
    demand_map_3d[new_growth_3d_indices[:,0], 
                  new_growth_3d_indices[:,1], 
                  new_growth_3d_indices[:,2]] = demand_q_met_values

    return {
        'world_coords_flat': new_growth_world_coords, # For this test, it *only* contains the new region voxels
        'voxel_indices_flat': new_growth_3d_indices, # Corresponding 3D indices for the new region voxels
        'metabolic_demand_map': demand_map_3d, # q_met map
        'voxel_volume': voxel_vol,
        'shape': demand_map_3d.shape # Shape of the minimal demand map
    }

def test_find_optimal_bif_empty_new_region(mock_parent_terminal, mock_tissue_data_for_bif_search, mock_energy_config):
    result = energy_model.find_optimal_bifurcation_for_new_region(
        mock_parent_terminal, 
        np.array([], dtype=int), # Empty new growth region
        mock_tissue_data_for_bif_search, 
        mock_energy_config,
        k_murray_factor=mock_energy_config["vascular_properties"]["k_murray_scaling_factor"],
        murray_exponent=mock_energy_config["vascular_properties"]["murray_law_exponent"]
    )
    assert result is None

def test_find_optimal_bif_negligible_demand(mock_parent_terminal, mock_tissue_data_for_bif_search, mock_energy_config):
    # Modify tissue data to have zero demand
    tissue_data_zero_demand = mock_tissue_data_for_bif_search.copy()
    tissue_data_zero_demand['metabolic_demand_map'] = np.zeros_like(tissue_data_zero_demand['metabolic_demand_map'])
    
    new_growth_indices = np.array([0,1]) # Referring to indices in this zero-demand setup

    result = energy_model.find_optimal_bifurcation_for_new_region(
        mock_parent_terminal, 
        new_growth_indices,
        tissue_data_zero_demand, 
        mock_energy_config,
        k_murray_factor=mock_energy_config["vascular_properties"]["k_murray_scaling_factor"],
        murray_exponent=mock_energy_config["vascular_properties"]["murray_law_exponent"]
    )
    assert result is None

def test_find_optimal_bif_basic_search(mock_parent_terminal, mock_tissue_data_for_bif_search, mock_energy_config, caplog):
    caplog.set_level(logging.DEBUG, logger="src.energy_model") # For verbose output from this test
    
    new_growth_indices = np.array([0,1]) # The two demand points defined in fixture

    result = energy_model.find_optimal_bifurcation_for_new_region(
        mock_parent_terminal, 
        new_growth_indices,
        mock_tissue_data_for_bif_search, 
        mock_energy_config,
        k_murray_factor=mock_energy_config["vascular_properties"]["k_murray_scaling_factor"],
        murray_exponent=mock_energy_config["vascular_properties"]["murray_law_exponent"]
    )
    assert result is not None, "Expected a bifurcation result for simple two-point demand"
    c1_pos, c1_r, c1_q, c2_pos, c2_r, c2_q, loss = result

    # Check child positions are somewhat reasonable (e.g., near the demand points)
    # Demand points were [1,0.5,0] and [1,-0.5,0]. Parent at [0,0,0].
    # Children should be somewhere around X=1.
    assert c1_pos[0] > 0.5 and c2_pos[0] > 0.5 
    
    # Check flows sum to total demand of the new region
    total_demand_q_met = np.sum(mock_tissue_data_for_bif_search['metabolic_demand_map'])
    total_demand_flow = total_demand_q_met * mock_tissue_data_for_bif_search['voxel_volume']
    assert np.isclose(c1_q + c2_q, total_demand_flow), "Child flows should sum to new region's total demand"

    # Check radii are positive and follow Murray's scaling roughly
    min_r = mock_energy_config["vascular_properties"]["min_radius"]
    assert c1_r >= min_r and c2_r >= min_r
    if c1_q > constants.EPSILON:
        expected_r1 = mock_energy_config["vascular_properties"]["k_murray_scaling_factor"] * (c1_q ** (1.0/3.0))
        assert np.isclose(c1_r, max(min_r, expected_r1), rtol=0.05) # Allow some tolerance due to min_radius clamping
    if c2_q > constants.EPSILON:
        expected_r2 = mock_energy_config["vascular_properties"]["k_murray_scaling_factor"] * (c2_q ** (1.0/3.0))
        assert np.isclose(c2_r, max(min_r, expected_r2), rtol=0.05)

    assert loss < np.inf and loss > 0 # Loss should be a finite positive number# tests/test_energy_model.py
import pytest
import numpy as np
from typing import Optional

from src import energy_model, constants, config_manager, utils

# Define a common mock config for tests in this file
@pytest.fixture
def mock_energy_config():
    return {
        "vascular_properties": {
            "blood_viscosity": 0.004,  # Pa.s - Use a slightly different value for testing
            "min_radius": 0.001,       # mm
            "k_murray_scaling_factor": 0.6, # s^(1/3)
            "murray_law_exponent": 3.0,
            "min_segment_length": 0.01
        },
        "gbo_growth": {
            # Ensure this C_met is in units consistent with E_flow from calculate_segment_flow_energy
            # If E_flow inputs (L,R,Q) are mm, mm, mm^3/s and mu is Pa.s, E_flow is in (Pa.mm^3/s) or (10^-9 W)
            # Then C_met here should be in (Pa.mm^3/s) / mm^3 = Pa/s (if we treat energy as power)
            # Or if energy is just a "cost value", units need to be balanced by user.
            # Let's pick a value for testing.
            "energy_coefficient_C_met_vessel_wall": 100.0, # Arbitrary units, assuming consistency
            "bifurcation_candidate_points": 5 
        }
    }

# --- Tests for calculate_segment_flow_energy ---
def test_calc_flow_energy_known_values(mock_energy_config):
    viscosity = mock_energy_config["vascular_properties"]["blood_viscosity"]
    # L=1mm, R=0.1mm, Q=0.01 mm^3/s
    length, radius, flow = 1.0, 0.1, 0.01
    # Expected: (8 * 0.004 * 1.0 * 0.01^2) / (pi * 0.1^4)
    # = (8 * 0.004 * 1e-4) / (pi * 1e-4) = (3.2e-6) / (pi * 1e-4) = 3.2e-2 / pi 
    expected_energy = (8 * viscosity * length * flow**2) / (constants.PI * radius**4)
    assert np.isclose(
        energy_model.calculate_segment_flow_energy(length, radius, flow, viscosity),
        expected_energy
    )

def test_calc_flow_energy_zero_radius(mock_energy_config):
    viscosity = mock_energy_config["vascular_properties"]["blood_viscosity"]
    assert energy_model.calculate_segment_flow_energy(1.0, 0.0, 0.01, viscosity) == np.inf
    assert energy_model.calculate_segment_flow_energy(1.0, constants.EPSILON / 2, 0.01, viscosity) == np.inf

def test_calc_flow_energy_zero_flow(mock_energy_config):
    viscosity = mock_energy_config["vascular_properties"]["blood_viscosity"]
    assert energy_model.calculate_segment_flow_energy(1.0, 0.1, 0.0, viscosity) == 0.0

def test_calc_flow_energy_zero_length(mock_energy_config):
    viscosity = mock_energy_config["vascular_properties"]["blood_viscosity"]
    assert energy_model.calculate_segment_flow_energy(0.0, 0.1, 0.01, viscosity) == 0.0

# --- Tests for calculate_segment_metabolic_energy ---
def test_calc_metabolic_energy_known_values(mock_energy_config):
    c_met = mock_energy_config["gbo_growth"]["energy_coefficient_C_met_vessel_wall"]
    length, radius = 1.0, 0.1
    # Expected: c_met * pi * 0.1^2 * 1.0
    expected_energy = c_met * constants.PI * radius**2 * length
    assert np.isclose(
        energy_model.calculate_segment_metabolic_energy(length, radius, c_met),
        expected_energy
    )

def test_calc_metabolic_energy_zero_radius(mock_energy_config):
    c_met = mock_energy_config["gbo_growth"]["energy_coefficient_C_met_vessel_wall"]
    assert energy_model.calculate_segment_metabolic_energy(1.0, 0.0, c_met) == 0.0

def test_calc_metabolic_energy_zero_length(mock_energy_config):
    c_met = mock_energy_config["gbo_growth"]["energy_coefficient_C_met_vessel_wall"]
    assert energy_model.calculate_segment_metabolic_energy(0.0, 0.1, c_met) == 0.0

# --- Tests for calculate_bifurcation_loss ---
def test_calc_bifurcation_loss_basic(mock_energy_config):
    parent_pos = np.array([0,0,0])
    c1_pos, c1_r, c1_q = np.array([1,0,0]), 0.1, 0.01
    c2_pos, c2_r, c2_q = np.array([0,1,0]), 0.08, 0.008
    
    viscosity = mock_energy_config["vascular_properties"]["blood_viscosity"]
    c_met = mock_energy_config["gbo_growth"]["energy_coefficient_C_met_vessel_wall"]

    l_c1 = utils.distance(parent_pos, c1_pos)
    e_flow_c1 = energy_model.calculate_segment_flow_energy(l_c1, c1_r, c1_q, viscosity)
    e_met_c1 = energy_model.calculate_segment_metabolic_energy(l_c1, c1_r, c_met)

    l_c2 = utils.distance(parent_pos, c2_pos)
    e_flow_c2 = energy_model.calculate_segment_flow_energy(l_c2, c2_r, c2_q, viscosity)
    e_met_c2 = energy_model.calculate_segment_metabolic_energy(l_c2, c2_r, c_met)
    
    expected_loss = e_flow_c1 + e_met_c1 + e_flow_c2 + e_met_c2
    
    calculated_loss = energy_model.calculate_bifurcation_loss(
        parent_pos, c1_pos, c1_r, c1_q, c2_pos, c2_r, c2_q, mock_energy_config
    )
    assert np.isclose(calculated_loss, expected_loss)

def test_calc_bifurcation_loss_one_child_zero_flow(mock_energy_config):
    parent_pos = np.array([0,0,0])
    c1_pos, c1_r, c1_q = np.array([1,0,0]), 0.1, 0.01
    c2_pos, c2_r, c2_q = np.array([0,1,0]), 0.08, 0.0 # Child 2 has zero flow
    
    viscosity = mock_energy_config["vascular_properties"]["blood_viscosity"]
    c_met = mock_energy_config["gbo_growth"]["energy_coefficient_C_met_vessel_wall"]

    l_c1 = utils.distance(parent_pos, c1_pos)
    e_flow_c1 = energy_model.calculate_segment_flow_energy(l_c1, c1_r, c1_q, viscosity)
    e_met_c1 = energy_model.calculate_segment_metabolic_energy(l_c1, c1_r, c_met)

    # For child 2, flow energy should be 0, metabolic energy depends on radius/length
    l_c2 = utils.distance(parent_pos, c2_pos)
    e_flow_c2 = 0.0 
    e_met_c2 = energy_model.calculate_segment_metabolic_energy(l_c2, c2_r, c_met)
    
    expected_loss = e_flow_c1 + e_met_c1 + e_flow_c2 + e_met_c2
    
    calculated_loss = energy_model.calculate_bifurcation_loss(
        parent_pos, c1_pos, c1_r, c1_q, c2_pos, c2_r, c2_q, mock_energy_config
    )
    assert np.isclose(calculated_loss, expected_loss)

# --- Sanity Tests for find_optimal_bifurcation_for_new_region ---
@pytest.fixture
def mock_parent_terminal():
    # Using a simple class or dict for parent_terminal_gbo_data for the test
    class MockParent:
        id = "p_test"
        pos = np.array([0.,0.,0.])
        radius = 0.2 # Not directly used by find_optimal, but good for context
        flow = 0.01  # Flow to *existing* territory, also not directly used by find_optimal for *new* region
    return MockParent()

@pytest.fixture
def mock_tissue_data_for_bif_search():
    # Minimal tissue data for bifurcation search
    # Two distinct demand points in the "new growth region"
    voxel_vol = 0.001 # mm^3 (0.1mm sided voxels)
    
    # These are global flat indices for the new growth region
    new_growth_indices_flat = np.array([0, 1]) 
    
    # World coords for these two specific voxels
    # Place them such that KMeans or random selection should pick them or points near them
    new_growth_world_coords = np.array([
        [1.0, 0.5, 0.0], # Demand point 1
        [1.0, -0.5, 0.0] # Demand point 2
    ])
    
    # Corresponding 3D indices (dummy for this test, must match shape of demand map)
    new_growth_3d_indices = np.array([
        [0,0,0],
        [0,1,0]
    ])
    
    # Metabolic demand (q_met) for these points
    demand_q_met_values = np.array([0.02, 0.02]) # 1/s
    
    # Create a minimal 3D demand map that contains these
    demand_map_3d = np.zeros((1,2,1))
    demand_map_3d[new_growth_3d_indices[:,0], 
                  new_growth_3d_indices[:,1], 
                  new_growth_3d_indices[:,2]] = demand_q_met_values

    return {
        'world_coords_flat': new_growth_world_coords, # For this test, it *only* contains the new region voxels
        'voxel_indices_flat': new_growth_3d_indices, # Corresponding 3D indices for the new region voxels
        'metabolic_demand_map': demand_map_3d, # q_met map
        'voxel_volume': voxel_vol,
        'shape': demand_map_3d.shape # Shape of the minimal demand map
    }

def test_find_optimal_bif_empty_new_region(mock_parent_terminal, mock_tissue_data_for_bif_search, mock_energy_config):
    result = energy_model.find_optimal_bifurcation_for_new_region(
        mock_parent_terminal, 
        np.array([], dtype=int), # Empty new growth region
        mock_tissue_data_for_bif_search, 
        mock_energy_config,
        k_murray_factor=mock_energy_config["vascular_properties"]["k_murray_scaling_factor"],
        murray_exponent=mock_energy_config["vascular_properties"]["murray_law_exponent"]
    )
    assert result is None

def test_find_optimal_bif_negligible_demand(mock_parent_terminal, mock_tissue_data_for_bif_search, mock_energy_config):
    # Modify tissue data to have zero demand
    tissue_data_zero_demand = mock_tissue_data_for_bif_search.copy()
    tissue_data_zero_demand['metabolic_demand_map'] = np.zeros_like(tissue_data_zero_demand['metabolic_demand_map'])
    
    new_growth_indices = np.array([0,1]) # Referring to indices in this zero-demand setup

    result = energy_model.find_optimal_bifurcation_for_new_region(
        mock_parent_terminal, 
        new_growth_indices,
        tissue_data_zero_demand, 
        mock_energy_config,
        k_murray_factor=mock_energy_config["vascular_properties"]["k_murray_scaling_factor"],
        murray_exponent=mock_energy_config["vascular_properties"]["murray_law_exponent"]
    )
    assert result is None

def test_find_optimal_bif_basic_search(mock_parent_terminal, mock_tissue_data_for_bif_search, mock_energy_config, caplog):
    caplog.set_level(logging.DEBUG, logger="src.energy_model") # For verbose output from this test
    
    new_growth_indices = np.array([0,1]) # The two demand points defined in fixture

    result = energy_model.find_optimal_bifurcation_for_new_region(
        mock_parent_terminal, 
        new_growth_indices,
        mock_tissue_data_for_bif_search, 
        mock_energy_config,
        k_murray_factor=mock_energy_config["vascular_properties"]["k_murray_scaling_factor"],
        murray_exponent=mock_energy_config["vascular_properties"]["murray_law_exponent"]
    )
    assert result is not None, "Expected a bifurcation result for simple two-point demand"
    c1_pos, c1_r, c1_q, c2_pos, c2_r, c2_q, loss = result

    # Check child positions are somewhat reasonable (e.g., near the demand points)
    # Demand points were [1,0.5,0] and [1,-0.5,0]. Parent at [0,0,0].
    # Children should be somewhere around X=1.
    assert c1_pos[0] > 0.5 and c2_pos[0] > 0.5 
    
    # Check flows sum to total demand of the new region
    total_demand_q_met = np.sum(mock_tissue_data_for_bif_search['metabolic_demand_map'])
    total_demand_flow = total_demand_q_met * mock_tissue_data_for_bif_search['voxel_volume']
    assert np.isclose(c1_q + c2_q, total_demand_flow), "Child flows should sum to new region's total demand"

    # Check radii are positive and follow Murray's scaling roughly
    min_r = mock_energy_config["vascular_properties"]["min_radius"]
    assert c1_r >= min_r and c2_r >= min_r
    if c1_q > constants.EPSILON:
        expected_r1 = mock_energy_config["vascular_properties"]["k_murray_scaling_factor"] * (c1_q ** (1.0/3.0))
        assert np.isclose(c1_r, max(min_r, expected_r1), rtol=0.05) # Allow some tolerance due to min_radius clamping
    if c2_q > constants.EPSILON:
        expected_r2 = mock_energy_config["vascular_properties"]["k_murray_scaling_factor"] * (c2_q ** (1.0/3.0))
        assert np.isclose(c2_r, max(min_r, expected_r2), rtol=0.05)

    assert loss < np.inf and loss > 0 # Loss should be a finite positive number