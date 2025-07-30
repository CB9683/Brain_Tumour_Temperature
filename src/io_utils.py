# src/io_utils.py
import nibabel as nib
import numpy as np
import pyvista as pv
import networkx as nx
import os
import logging
import yaml
from typing import Any, Union, Dict, List, Tuple

logger = logging.getLogger(__name__)

def load_nifti_image(filepath: str) -> tuple[np.ndarray, np.ndarray, Any] | tuple[None, None, None]:
    """
    Loads a NIfTI image.

    Args:
        filepath (str): Path to the .nii or .nii.gz file.

    Returns:
        tuple: (data_array, affine_matrix, header) or (None, None, None) if loading fails.
               data_array is usually in (L,P,I) or (R,A,S) depending on how it was saved.
               Affine maps voxel indices to world space (often RAS).
    """
    if not os.path.exists(filepath):
        logger.warning(f"NIfTI file not found: {filepath}. Skipping.")
        return None, None, None
    try:
        img = nib.load(filepath)
        data = img.get_fdata()
        affine = img.affine
        header = img.header
        logger.info(f"Loaded NIfTI image: {filepath}, Shape: {data.shape}, Voxel size from affine: {np.diag(affine)[:3]}")
        return data.astype(np.float32), affine, header # Cast to float for calculations
    except Exception as e:
        logger.error(f"Error loading NIfTI file {filepath}: {e}")
        return None, None, None

def save_nifti_image(data_array: np.ndarray, affine: np.ndarray, filepath: str, header: nib.Nifti1Header = None):
    """
    Saves a NumPy array as a NIfTI image.

    Args:
        data_array (np.ndarray): The image data.
        affine (np.ndarray): The affine matrix for the image.
        filepath (str): Path to save the .nii or .nii.gz file.
        header (nib.Nifti1Header, optional): NIfTI header. If None, a minimal one is created.
    """
    try:
        # Ensure data type is compatible, e.g. float32 or int16
        # Nifti1Image constructor will handle appropriate dtype based on data.
        # If data is boolean, convert to int8 or uint8
        if data_array.dtype == bool:
            data_array = data_array.astype(np.uint8)
            
        img = nib.Nifti1Image(data_array, affine, header=header)
        nib.save(img, filepath)
        logger.info(f"Saved NIfTI image to: {filepath}")
    except Exception as e:
        logger.error(f"Error saving NIfTI file {filepath}: {e}")
        raise

def load_arterial_centerlines_vtp(filepath: str) -> pv.PolyData | None:
    """
    Loads arterial centerlines from a VTP file.
    Assumes the VTP file contains points and lines representing vessel segments.
    It should ideally have a 'radius' point data array.

    Args:
        filepath (str): Path to the .vtp file.

    Returns:
        pyvista.PolyData: The loaded PolyData object or None if loading fails.
    """
    if not os.path.exists(filepath):
        logger.warning(f"Arterial centerline file not found: {filepath}. Skipping.")
        return None
    try:
        mesh = pv.read(filepath)
        logger.info(f"Loaded arterial centerlines from VTP: {filepath}")
        if 'radius' not in mesh.point_data:
            logger.warning(f"VTP file {filepath} does not contain 'radius' point data. Defaulting or errors might occur.")
        # Could add more checks here, e.g., for line connectivity
        return mesh
    except Exception as e:
        logger.error(f"Error loading VTP file {filepath}: {e}")
        return None

def load_arterial_centerlines_txt(filepath: str, radius_default: float = 0.1) -> pv.PolyData | None:
    """
    Loads arterial centerlines from a TXT file and converts to PyVista PolyData.
    Expected TXT format:
    Each line: x y z [radius]
    If radius is not present, radius_default is used.
    Segments are assumed to connect consecutive points.
    A more robust format might specify connectivity explicitly. For now, assume polylines.

    Args:
        filepath (str): Path to the .txt file.
        radius_default (float): Default radius if not specified in the file.

    Returns:
        pyvista.PolyData: A PolyData object representing the centerlines, or None if loading fails.
    """
    if not os.path.exists(filepath):
        logger.warning(f"Arterial centerline TXT file not found: {filepath}. Skipping.")
        return None
    
    points = []
    radii = []
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'): # Skip empty lines or comments
                    continue
                parts = list(map(float, line.split()))
                if len(parts) == 3:
                    points.append(parts)
                    radii.append(radius_default)
                elif len(parts) == 4:
                    points.append(parts[:3])
                    radii.append(parts[3])
                else:
                    logger.warning(f"Skipping malformed line {line_num+1} in {filepath}: {line}")
        
        if not points:
            logger.error(f"No valid points found in TXT file: {filepath}")
            return None

        points_np = np.array(points)
        radii_np = np.array(radii)

        # Create PolyData: assumes a single polyline for simplicity
        # For multiple disconnected arteries, the TXT format would need to be richer or
        # processed to identify separate polylines.
        num_points = len(points_np)
        lines = np.empty((num_points - 1, 3), dtype=int)
        lines[:, 0] = 2  # Each line segment has 2 points
        lines[:, 1] = np.arange(num_points - 1)
        lines[:, 2] = np.arange(1, num_points)
        
        poly = pv.PolyData(points_np, lines=lines)
        poly.point_data['radius'] = radii_np
        
        logger.info(f"Loaded arterial centerlines from TXT: {filepath}, {num_points} points.")
        return poly

    except Exception as e:
        logger.error(f"Error loading TXT file {filepath}: {e}")
        return None


def save_vascular_tree_vtp(graph: nx.DiGraph, filepath: str,
                           pos_attr='pos', radius_attr='radius', pressure_attr='pressure', flow_attr='flow_solver'):
    """
    Saves a vascular tree (NetworkX graph) to a VTP file.
    Nodes store positions and radii. Edges define connectivity.
    """
    points = []
    point_radii = []
    point_pressures = []
    lines_connectivity = [] # Changed name for clarity, this is for pv.PolyData(points, lines=HERE)
    edge_flows = [] 

    node_to_idx = {node_id: i for i, node_id in enumerate(graph.nodes())}

    for node_id, data in graph.nodes(data=True):
        if pos_attr not in data:
            logger.warning(f"Node {node_id} missing '{pos_attr}' attribute. Skipping for point data.")
            continue
        points.append(data[pos_attr])
        point_radii.append(data.get(radius_attr, 0.0))
        point_pressures.append(data.get(pressure_attr, np.nan)) 

    if not points:
        logger.error("No points to save in the vascular tree. VTP file will be empty or invalid.")
        empty_poly = pv.PolyData()
        empty_poly.save(filepath)
        logger.error(f"Saved an empty VTP file to: {filepath} due to no valid points in the graph.")
        return

    # Build the lines array for PolyData constructor
    # Format: [n_points_in_line0, pt0_idx, pt1_idx, n_points_in_line1, ptA_idx, ptB_idx, ...]
    raw_lines_for_pv = []
    for u, v, data in graph.edges(data=True):
        if u in node_to_idx and v in node_to_idx:
            raw_lines_for_pv.extend([2, node_to_idx[u], node_to_idx[v]]) # Each line segment has 2 points
            edge_flows.append(data.get(flow_attr, np.nan)) 
        else:
            logger.warning(f"Edge ({u}-{v}) references missing node. Skipping this edge for line connectivity.")

    # Create PolyData object
    # If there are lines, pass them to the constructor.
    # Otherwise, it's just a point cloud.
    if raw_lines_for_pv:
        poly_data = pv.PolyData(np.array(points), lines=np.array(raw_lines_for_pv))
    else:
        poly_data = pv.PolyData(np.array(points)) # Will be a point cloud if no edges

    logger.debug(f"PolyData created. Number of points: {poly_data.n_points}, Number of cells (lines): {poly_data.n_cells}")
    logger.debug(f"Number of edge_flows collected: {len(edge_flows)}")
    
    # Add point data
    if points: # Check if points list is not empty before trying to assign
        poly_data.point_data[radius_attr] = np.array(point_radii)
        if any(not np.isnan(p) for p in point_pressures): 
            poly_data.point_data[pressure_attr] = np.array(point_pressures)
    
    # Add cell data (flow) only if there are cells and corresponding flow data
    if edge_flows and poly_data.n_cells > 0:
        if poly_data.n_cells == len(edge_flows):
            poly_data.cell_data[flow_attr] = np.array(edge_flows)
        else:
            # This case should ideally not be hit if graph processing and polydata creation are correct
            logger.error(
                f"Critical mismatch assigning cell data! "
                f"PolyData n_cells: {poly_data.n_cells}, "
                f"Number of flow values: {len(edge_flows)}. "
                f"Flow data will NOT be saved for cells."
            )
            # Decide: either don't add flow data, or pad/truncate (not ideal)
            # For now, we won't add it if there's a mismatch.
    
    try:
        poly_data.save(filepath)
        logger.info(f"Saved vascular tree to VTP: {filepath}")
    except Exception as e:
        logger.error(f"Error saving vascular tree to VTP {filepath}: {e}")
        raise

def save_simulation_parameters(config: dict, filepath: str):
    """Saves the simulation configuration to a YAML file."""
    try:
        with open(filepath, 'w') as f:
            yaml.dump(config, f, sort_keys=False, indent=4)
        logger.info(f"Saved simulation parameters to: {filepath}")
    except Exception as e:
        logger.error(f"Error saving simulation parameters to {filepath}: {e}")
        raise


if __name__ == '__main__':
    # Setup basic logging for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create dummy data directory
    test_data_dir = "temp_io_test_data"
    os.makedirs(test_data_dir, exist_ok=True)

    # --- Test NIfTI I/O ---
    dummy_nifti_path = os.path.join(test_data_dir, "dummy.nii.gz")
    shape = (10, 10, 10)
    affine = np.eye(4)
    affine[0,0] = affine[1,1] = affine[2,2] = 0.5 # 0.5mm isotropic voxels
    data = np.random.rand(*shape).astype(np.float32)
    
    print(f"\n--- Testing NIfTI I/O ---")
    save_nifti_image(data, affine, dummy_nifti_path)
    loaded_data, loaded_affine, _ = load_nifti_image(dummy_nifti_path)
    
    if loaded_data is not None:
        assert np.allclose(data, loaded_data), "NIfTI data mismatch"
        assert np.allclose(affine, loaded_affine), "NIfTI affine mismatch"
        print("NIfTI I/O test successful.")
    else:
        print("NIfTI loading failed.")

    # Test loading non-existent NIfTI
    load_nifti_image("non_existent.nii.gz")


    # --- Test VTP I/O (arterial centerlines) ---
    print(f"\n--- Testing VTP I/O (arterial centerlines) ---")
    dummy_vtp_path = os.path.join(test_data_dir, "dummy_arteries.vtp")
    # Create a simple PyVista PolyData object
    points = np.array([[0,0,0], [1,1,0], [2,0,0]], dtype=float)
    lines = np.array([2, 0, 1, 2, 1, 2]) # Connect point 0-1, then 1-2
    poly = pv.PolyData(points, lines=lines)
    poly.point_data['radius'] = np.array([0.5, 0.4, 0.3])
    poly.save(dummy_vtp_path)
    
    loaded_poly = load_arterial_centerlines_vtp(dummy_vtp_path)
    if loaded_poly:
        assert loaded_poly.n_points == 3, "VTP point count mismatch"
        assert 'radius' in loaded_poly.point_data, "VTP radius data missing"
        print("VTP arterial centerline I/O test successful.")
    else:
        print("VTP loading failed.")
    
    # Test loading non-existent VTP
    load_arterial_centerlines_vtp("non_existent.vtp")

    # --- Test TXT I/O (arterial centerlines) ---
    print(f"\n--- Testing TXT I/O (arterial centerlines) ---")
    dummy_txt_path = os.path.join(test_data_dir, "dummy_arteries.txt")
    with open(dummy_txt_path, "w") as f:
        f.write("# Test TXT file\n")
        f.write("0 0 0 0.5\n")
        f.write("1 1 0 0.4\n")
        f.write("2 0 0\n") # Test with default radius
        f.write("3 1 1 0.2\n")
    
    loaded_poly_txt = load_arterial_centerlines_txt(dummy_txt_path, radius_default=0.1)
    if loaded_poly_txt:
        assert loaded_poly_txt.n_points == 4, "TXT point count mismatch"
        assert 'radius' in loaded_poly_txt.point_data, "TXT radius data missing"
        expected_radii = np.array([0.5, 0.4, 0.1, 0.2])
        assert np.allclose(loaded_poly_txt.point_data['radius'], expected_radii), "TXT radii mismatch"
        print(f"TXT loaded radii: {loaded_poly_txt.point_data['radius']}")
        print("TXT arterial centerline I/O test successful.")
    else:
        print("TXT loading failed.")

    # --- Test Vascular Tree (NetworkX to VTP) Save ---
    print(f"\n--- Testing Vascular Tree (NetworkX to VTP) Save ---")
    graph = nx.DiGraph()
    # Add nodes with positions and radii
    graph.add_node(0, pos=np.array([0,0,0]), radius=0.5, pressure=100.0)
    graph.add_node(1, pos=np.array([1,0,0]), radius=0.4, pressure=90.0)
    graph.add_node(2, pos=np.array([1,1,0]), radius=0.3, pressure=80.0)
    # Add edges with flow
    graph.add_edge(0, 1, flow=10.0)
    graph.add_edge(1, 2, flow=8.0)

    tree_vtp_path = os.path.join(test_data_dir, "vascular_tree.vtp")
    save_vascular_tree_vtp(graph, tree_vtp_path)
    
    # Verify by loading it back with PyVista
    if os.path.exists(tree_vtp_path):
        loaded_tree_poly = pv.read(tree_vtp_path)
        assert loaded_tree_poly.n_points == 3, "Saved tree VTP point count mismatch"
        assert 'radius' in loaded_tree_poly.point_data, "Saved tree VTP radius missing"
        assert 'pressure' in loaded_tree_poly.point_data, "Saved tree VTP pressure missing"
        # Check if flow is present as cell data
        if loaded_tree_poly.n_cells > 0 : # n_cells corresponds to number of lines/edges
             assert 'flow' in loaded_tree_poly.cell_data, "Saved tree VTP flow missing"
        print("Vascular tree (NetworkX to VTP) save test successful.")
    else:
        print("Vascular tree VTP save failed.")
        
    # Test saving an empty graph
    empty_graph = nx.DiGraph()
    empty_tree_vtp_path = os.path.join(test_data_dir, "empty_vascular_tree.vtp")
    save_vascular_tree_vtp(empty_graph, empty_tree_vtp_path)
    if os.path.exists(empty_tree_vtp_path):
        loaded_empty_tree_poly = pv.read(empty_tree_vtp_path)
        assert loaded_empty_tree_poly.n_points == 0, "Empty graph should result in VTP with 0 points."
        print("Saving empty graph to VTP test successful.")

    # --- Test Saving Simulation Parameters ---
    print(f"\n--- Testing Saving Simulation Parameters ---")
    dummy_params_path = os.path.join(test_data_dir, "sim_params.yaml")
    test_config = {"param1": 10, "nested": {"param2": "test"}}
    save_simulation_parameters(test_config, dummy_params_path)
    # Verify by loading
    with open(dummy_params_path, 'r') as f:
        loaded_params = yaml.safe_load(f)
    assert loaded_params["param1"] == 10, "Param save/load mismatch"
    print("Simulation parameter saving test successful.")

    # Clean up dummy data directory
    import shutil
    shutil.rmtree(test_data_dir)
    print(f"\nCleaned up temporary test directory: {test_data_dir}")

def export_tissue_masks_to_vtk(tissue_data: dict, perfused_mask: np.ndarray, output_dir: str, iteration: int = -1):
    """
    Exports tissue masks as VTK files for ParaView visualization.
    
    Args:
        tissue_data: Dictionary containing tissue masks and affine
        perfused_mask: Current perfusion state
        output_dir: Output directory
        iteration: Current iteration number (-1 for final)
    """
    try:
        import vtk
        from vtk.util import numpy_support
    except ImportError:
        logger.warning("VTK not available. Cannot export tissue masks for ParaView.")
        return
    
    affine = tissue_data.get('affine')
    if affine is None:
        logger.error("Cannot export tissue masks: affine matrix missing")
        return
    
    # Create subdirectory for VTK files
    vtk_dir = os.path.join(output_dir, "tissue_vtk")
    os.makedirs(vtk_dir, exist_ok=True)
    
    # Determine file suffix
    suffix = f"_iter_{iteration}" if iteration >= 0 else "_final"
    
    # List of masks to export
    masks_to_export = {
        'perfused': perfused_mask,
        'GM': tissue_data.get('GM'),
        'WM': tissue_data.get('WM'),
        'domain_mask': tissue_data.get('domain_mask'),
        'metabolic_demand': tissue_data.get('metabolic_demand_map')
    }
    
    # Get voxel spacing from affine
    spacing = np.abs(np.diag(affine)[:3])
    origin = affine[:3, 3]
    
    for mask_name, mask_data in masks_to_export.items():
        if mask_data is None or (isinstance(mask_data, np.ndarray) and not np.any(mask_data)):
            continue
            
        # Create VTK image data
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(mask_data.shape)
        image_data.SetSpacing(spacing)
        image_data.SetOrigin(origin)
        
        # Convert numpy array to VTK array
        if mask_data.dtype == bool:
            vtk_data = numpy_support.numpy_to_vtk(
                mask_data.astype(np.uint8).ravel(order='F'), 
                deep=True, 
                array_type=vtk.VTK_UNSIGNED_CHAR
            )
        else:
            vtk_data = numpy_support.numpy_to_vtk(
                mask_data.astype(np.float32).ravel(order='F'), 
                deep=True, 
                array_type=vtk.VTK_FLOAT
            )
        
        vtk_data.SetName(mask_name)
        image_data.GetPointData().SetScalars(vtk_data)
        
        # Write to file
        writer = vtk.vtkXMLImageDataWriter()
        filename = os.path.join(vtk_dir, f"{mask_name}{suffix}.vti")
        writer.SetFileName(filename)
        writer.SetInputData(image_data)
        writer.Write()
        
        logger.info(f"Exported {mask_name} to {filename}")