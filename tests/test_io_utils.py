# tests/test_io_utils.py
import pytest
import nibabel as nib
import numpy as np
import pyvista as pv
import networkx as nx
import os
import yaml
import shutil # For cleaning up
from src import io_utils, data_structures # For graph creation helpers
from src import constants # For default values

@pytest.fixture(scope="module") # Use module scope for tmp_path to reduce overhead
def test_output_dir(tmp_path_factory):
    # Create a single temporary directory for all tests in this module
    tdir = tmp_path_factory.mktemp("io_test_data")
    return tdir

# --- NIfTI I/O Tests ---
def test_nifti_save_load_roundtrip(test_output_dir):
    filepath = test_output_dir / "test.nii.gz"
    shape = (5, 5, 5)
    affine = np.diag([0.5, 0.5, 0.5, 1.0]) # 0.5mm isotropic
    data = np.random.rand(*shape).astype(np.float32)
    
    io_utils.save_nifti_image(data, affine, str(filepath))
    assert os.path.exists(filepath)
    
    loaded_data, loaded_affine, _ = io_utils.load_nifti_image(str(filepath))
    assert loaded_data is not None
    assert np.allclose(data, loaded_data)
    assert np.allclose(affine, loaded_affine)


def test_nifti_save_load_boolean(test_output_dir):
    filepath = test_output_dir / "test_bool.nii.gz"
    shape = (5, 5, 5)
    affine = np.eye(4)
    original_bool_data = np.random.choice([True, False], size=shape)

    # In save_nifti_image, boolean data is cast to uint8
    io_utils.save_nifti_image(original_bool_data, affine, str(filepath))
    
    # load_nifti_image casts to float32
    loaded_data_float, _, _ = io_utils.load_nifti_image(str(filepath)) 

    assert loaded_data_float is not None
    assert loaded_data_float.dtype == np.float32 # Expect float32 after loading

    # Check if the values are preserved (0.0 for False, 1.0 for True)
    # Cast original boolean data to float for comparison
    expected_float_values = original_bool_data.astype(np.float32)
    assert np.all(loaded_data_float == expected_float_values)

def test_load_nifti_non_existent(caplog):
    data, affine, header = io_utils.load_nifti_image("non_existent_file.nii.gz")
    assert data is None
    assert affine is None
    assert header is None
    assert "NIfTI file not found" in caplog.text

# --- VTP Arterial Centerlines I/O Tests ---
@pytest.fixture
def sample_vtp_file(test_output_dir):
    filepath = test_output_dir / "sample_arteries.vtp"
    points = np.array([[0,0,0], [1,1,0], [2,0,0]], dtype=float)
    lines = np.array([2, 0, 1, 2, 1, 2]) # Connect point 0-1, then 1-2
    poly = pv.PolyData(points, lines=lines)
    poly.point_data['radius'] = np.array([0.5, 0.4, 0.3])
    poly.save(str(filepath))
    return str(filepath)

def test_load_arterial_centerlines_vtp_valid(sample_vtp_file):
    poly_data = io_utils.load_arterial_centerlines_vtp(sample_vtp_file)
    assert poly_data is not None
    assert poly_data.n_points == 3
    assert 'radius' in poly_data.point_data
    assert np.allclose(poly_data.point_data['radius'], [0.5, 0.4, 0.3])

def test_load_arterial_centerlines_vtp_no_radius(test_output_dir, caplog):
    filepath_no_radius = test_output_dir / "no_radius.vtp"
    points = np.array([[0,0,0], [1,0,0]], dtype=float)
    lines = np.array([2,0,1])
    poly = pv.PolyData(points, lines=lines)
    poly.save(str(filepath_no_radius))
    
    poly_data = io_utils.load_arterial_centerlines_vtp(str(filepath_no_radius))
    assert poly_data is not None
    assert 'radius' not in poly_data.point_data
    assert "does not contain 'radius' point data" in caplog.text

def test_load_arterial_centerlines_vtp_non_existent(caplog):
    poly_data = io_utils.load_arterial_centerlines_vtp("non_existent.vtp")
    assert poly_data is None
    assert "Arterial centerline file not found" in caplog.text

# --- TXT Arterial Centerlines I/O Tests ---
@pytest.fixture
def sample_txt_file(test_output_dir):
    filepath = test_output_dir / "sample_arteries.txt"
    content = (
        "# Test TXT file\n"
        "0 0 0 0.5\n"  # x y z radius
        "1 1 0 0.4\n"
        "2 0 0      \n"  # x y z (use default radius)
        "3 1 1 0.2\n"
    )
    with open(filepath, "w") as f:
        f.write(content)
    return str(filepath)

def test_load_arterial_centerlines_txt_valid(sample_txt_file):
    poly_data = io_utils.load_arterial_centerlines_txt(sample_txt_file, radius_default=0.1)
    assert poly_data is not None
    assert poly_data.n_points == 4
    assert 'radius' in poly_data.point_data
    expected_radii = np.array([0.5, 0.4, 0.1, 0.2])
    assert np.allclose(poly_data.point_data['radius'], expected_radii)
    # Check lines (assumes single polyline connection)
    assert poly_data.n_cells == 3 # 3 segments for 4 points

def test_load_arterial_centerlines_txt_non_existent(caplog):
    poly_data = io_utils.load_arterial_centerlines_txt("non_existent.txt")
    assert poly_data is None
    assert "Arterial centerline TXT file not found" in caplog.text

# --- Vascular Tree (NetworkX to VTP) Save/Load Test ---
@pytest.fixture
def sample_nx_graph():
    graph = data_structures.create_empty_vascular_graph()
    data_structures.add_node_to_graph(graph, "n0", pos=np.array([0.,0.,0.]), radius=0.5, pressure=100.0, type="root")
    data_structures.add_node_to_graph(graph, "n1", pos=np.array([1.,0.,0.]), radius=0.4, pressure=90.0, type="segment")
    data_structures.add_node_to_graph(graph, "n2", pos=np.array([1.,1.,0.]), radius=0.3, pressure=80.0, type="terminal")
    
    data_structures.add_edge_to_graph(graph, "n0", "n1", flow=10.0, type="segment_edge")
    data_structures.add_edge_to_graph(graph, "n1", "n2", flow=8.0, type="segment_edge")
    return graph

def test_save_vascular_tree_vtp_valid(sample_nx_graph, test_output_dir):
    filepath = test_output_dir / "vascular_tree_output.vtp"
    io_utils.save_vascular_tree_vtp(sample_nx_graph, str(filepath))
    assert os.path.exists(filepath)

    # Load back with PyVista and verify
    loaded_poly = pv.read(filepath)
    assert loaded_poly.n_points == 3
    assert 'radius' in loaded_poly.point_data
    assert 'pressure' in loaded_poly.point_data
    assert np.allclose(loaded_poly.point_data['radius'], [0.5, 0.4, 0.3]) # Order dependent on node iteration
    
    # Check flow on cells (edges)
    assert loaded_poly.n_cells == 2 # number of edges
    assert 'flow' in loaded_poly.cell_data
    # The order of flows in cell_data depends on the order edges were processed.
    # For a robust check, one might need to map cells back to graph edges if order isn't guaranteed.
    # For now, check if values exist and are correct, assuming a small graph maintains order.
    # This might be brittle. A better check might involve querying specific cells if possible.
    # For now, let's check if the set of flows is correct.
    assert set(np.round(loaded_poly.cell_data['flow'], 5)) == {10.0, 8.0}


def test_save_vascular_tree_vtp_empty_graph(test_output_dir, caplog):
    filepath = test_output_dir / "empty_tree.vtp"
    empty_graph = data_structures.create_empty_vascular_graph()
    io_utils.save_vascular_tree_vtp(empty_graph, str(filepath))
    
    assert os.path.exists(filepath)
    assert "Saved an empty VTP file" in caplog.text # Check for our specific log message
    
    loaded_poly = pv.read(filepath)
    assert loaded_poly.n_points == 0
    assert loaded_poly.n_cells == 0

# --- Simulation Parameters Save Test ---
def test_save_simulation_parameters(test_output_dir):
    filepath = test_output_dir / "sim_params_test.yaml"
    test_config = {"param_A": 123, "group_B": {"sub_param_C": "value_test"}}
    
    io_utils.save_simulation_parameters(test_config, str(filepath))
    assert os.path.exists(filepath)
    
    with open(filepath, 'r') as f:
        loaded_params = yaml.safe_load(f)
    
    assert loaded_params == test_config