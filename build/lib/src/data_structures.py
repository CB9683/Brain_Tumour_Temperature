# src/data_structures.py
import networkx as nx
import numpy as np
import logging

logger = logging.getLogger(__name__)

# --- Vascular Tree (NetworkX Graph) Conventions ---
# Nodes in the graph represent points in 3D space (e.g., bifurcations, terminals, points along a segment).
# Edges represent vessel segments connecting these points.

# Node Attributes:
# - 'id': (any, unique) Unique identifier for the node. Often the NetworkX node key itself.
# - 'pos': (np.ndarray, shape (3,)) 3D coordinates [x, y, z] in physical units (e.g., mm). REQUIRED.
# - 'radius': (float) Radius of the vessel at this node/point in physical units. REQUIRED for many operations.
# - 'type': (str) Type of node, e.g., 'root', 'bifurcation', 'terminal', 'segment_point'.
# - 'pressure': (float) Blood pressure at this node (computed during perfusion modeling).
# - 'flow_demand': (float) For terminal nodes, the flow Q_i required by its territory.
# - 'territory_voxels': (list or np.ndarray) Indices or coordinates of voxels supplied by this terminal.
# - 'parent_measured_terminal_id': (any) For synthetic terminals, ID of the measured artery terminal they originated from.
# - 'is_tumor_vessel': (bool) True if this node is part of a tumor-induced vessel.

# Edge Attributes (for edge u -> v):
# - 'length': (float) Length of the vessel segment in physical units. Can be calculated from node positions.
# - 'radius': (float) Radius of the segment. Can be average of u and v radii, or u's radius if flow is from u to v.
#           Consistency needed: often derived from flow and Murray's law.
# - 'flow': (float) Blood flow rate through the segment (computed during perfusion modeling).
# - 'resistance': (float) Hydraulic resistance of the segment (computed for perfusion modeling).
# - 'is_tumor_vessel': (bool) True if this segment is part of a tumor-induced vessel.


def create_empty_vascular_graph() -> nx.DiGraph:
    """Creates an empty directed graph for the vascular tree."""
    return nx.DiGraph()

def add_node_to_graph(graph: nx.DiGraph, node_id: any, pos: np.ndarray, radius: float, 
                      node_type: str = 'default', **kwargs):
    """
    Adds a node with standard attributes to the vascular graph.
    
    Args:
        graph (nx.DiGraph): The graph to add the node to.
        node_id (any): Unique ID for the node.
        pos (np.ndarray): 3D position [x,y,z].
        radius (float): Vessel radius at this node.
        node_type (str): Type of node.
        **kwargs: Additional attributes to set for the node.
    """
    if not isinstance(pos, np.ndarray) or pos.shape != (3,):
        raise ValueError("Position 'pos' must be a 3-element NumPy array.")
    if not isinstance(radius, (int, float)) or radius < 0:
        raise ValueError("Radius must be a non-negative number.")

    attrs = {
        'pos': pos,
        'radius': radius,
        'type': node_type,
    }
    attrs.update(kwargs) # Add any extra attributes
    graph.add_node(node_id, **attrs)
    # logger.debug(f"Added node {node_id} with attributes: {attrs}")

def add_edge_to_graph(graph: nx.DiGraph, u_id: any, v_id: any, **kwargs):
    """
    Adds an edge with standard attributes to the vascular graph.
    Length is automatically calculated if node positions exist.
    
    Args:
        graph (nx.DiGraph): The graph to add the edge to.
        u_id (any): ID of the source node.
        v_id (any): ID of the target node.
        **kwargs: Additional attributes to set for the edge.
    """
    if not graph.has_node(u_id) or not graph.has_node(v_id):
        logger.error(f"Cannot add edge ({u_id}-{v_id}): one or both nodes do not exist.")
        raise ValueError(f"Nodes {u_id} or {v_id} not in graph.")

    attrs = {}
    # Calculate length
    pos_u = graph.nodes[u_id].get('pos')
    pos_v = graph.nodes[v_id].get('pos')
    if pos_u is not None and pos_v is not None:
        length = np.linalg.norm(pos_u - pos_v)
        attrs['length'] = length
    else:
        logger.warning(f"Could not calculate length for edge ({u_id}-{v_id}) due to missing node positions.")

    # Example: Edge radius could be based on upstream node or average
    # For now, let's assume it might be set explicitly or derived later
    # radius_u = graph.nodes[u_id].get('radius')
    # if radius_u is not None:
    #    attrs['radius'] = radius_u 

    attrs.update(kwargs) # Add any extra attributes
    graph.add_edge(u_id, v_id, **attrs)
    # logger.debug(f"Added edge ({u_id}-{v_id}) with attributes: {attrs}")


# --- Tissue Data Structure ---
# Represented as a dictionary of NumPy arrays, plus affine and voxel volume.
# Example:
# tissue_data = {
#     'WM': wm_array,         # (X, Y, Z) binary or fractional mask
#     'GM': gm_array,         # (X, Y, Z)
#     'Tumor': tumor_array,   # (X, Y, Z)
#     'CSF': csf_array,       # (X, Y, Z)
#     'domain_mask': combined_mask, # (X, Y, Z) boolean array defining relevant voxels
#     'metabolic_demand_map': demand_map, # (X, Y, Z) float array of q_met per voxel
#     'affine': affine_matrix, # 4x4 np.ndarray
#     'voxel_volume': volume_per_voxel, # float (mm^3 or m^3)
#     'world_coords_flat': world_coords_of_domain_voxels # (N_domain_voxels, 3)
#     'voxel_indices_flat': voxel_indices_of_domain_voxels # (N_domain_voxels, 3)
# }

def get_metabolic_demand_map(tissue_segmentations: dict, config: dict, voxel_volume: float) -> np.ndarray:
    """
    Generates a metabolic demand map (q_met * dV) from tissue segmentations.
    
    Args:
        tissue_segmentations (dict): Dictionary of tissue type arrays (e.g., 'WM', 'GM', 'Tumor').
                                     Values are masks (0 or 1, or fractional 0-1).
        config (dict): Configuration dictionary with metabolic rates.
        voxel_volume (float): Volume of a single voxel (e.g., in mm^3).

    Returns:
        np.ndarray: A 3D array of the same shape as segmentations, where each voxel
                    contains the total metabolic demand (e.g., in mm^3_blood/s).
    """
    from src import config_manager as cfg_mgr # to use get_param

    # Get shape from one of the segmentations
    shape = None
    for seg_name, seg_array in tissue_segmentations.items():
        if seg_array is not None:
            shape = seg_array.shape
            break
    if shape is None:
        logger.error("No valid tissue segmentations provided to create metabolic map.")
        return None

    demand_map = np.zeros(shape, dtype=np.float32)
    
    q_rates = cfg_mgr.get_param(config, "tissue_properties.metabolic_rates")

    if 'GM' in tissue_segmentations and tissue_segmentations['GM'] is not None:
        demand_map += tissue_segmentations['GM'] * q_rates.get('gm', 0.0)
    if 'WM' in tissue_segmentations and tissue_segmentations['WM'] is not None:
        demand_map += tissue_segmentations['WM'] * q_rates.get('wm', 0.0)
    if 'CSF' in tissue_segmentations and tissue_segmentations['CSF'] is not None:
        demand_map += tissue_segmentations['CSF'] * q_rates.get('csf', 0.0) # usually 0
    
    # Tumor can have rim/core distinction if available, or a single tumor type
    if 'Tumor' in tissue_segmentations and tissue_segmentations['Tumor'] is not None:
        # Simple model: use 'tumor_rim' for all tumor voxels if 'tumor_core' isn't specified
        # or if the tumor segmentation isn't further divided.
        # A more complex model would require separate Tumor_Rim and Tumor_Core segmentations.
        tumor_rate = q_rates.get('tumor_rim', q_rates.get('tumor', 0.0)) # Fallback to 'tumor' if 'tumor_rim' not there
        demand_map += tissue_segmentations['Tumor'] * tumor_rate
    elif 'Tumor_Rim' in tissue_segmentations and tissue_segmentations['Tumor_Rim'] is not None:
         demand_map += tissue_segmentations['Tumor_Rim'] * q_rates.get('tumor_rim', 0.0)
         if 'Tumor_Core' in tissue_segmentations and tissue_segmentations['Tumor_Core'] is not None:
             demand_map += tissue_segmentations['Tumor_Core'] * q_rates.get('tumor_core', 0.0)
             
    return demand_map * voxel_volume # Now it's total demand per voxel (e.g. mm^3/s)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # Test vascular graph functions
    g = create_empty_vascular_graph()
    add_node_to_graph(g, 0, pos=np.array([0.,0.,0.]), radius=0.5, node_type='root', pressure=100)
    add_node_to_graph(g, 1, pos=np.array([1.,0.,0.]), radius=0.4, node_type='bifurcation')
    add_node_to_graph(g, 2, pos=np.array([2.,1.,0.]), radius=0.3, node_type='terminal', flow_demand=5)
    add_node_to_graph(g, 3, pos=np.array([2.,-1.,0.]), radius=0.3, node_type='terminal', flow_demand=5)

    add_edge_to_graph(g, 0, 1, flow=10)
    add_edge_to_graph(g, 1, 2, flow=5)
    add_edge_to_graph(g, 1, 3, flow=5)

    print("\n--- Vascular Graph Test ---")
    print(f"Nodes: {g.nodes(data=True)}")
    print(f"Edges: {g.edges(data=True)}")
    assert np.isclose(g.edges[(0,1)]['length'], 1.0)
    assert g.nodes[2]['flow_demand'] == 5

    # Test metabolic demand map
    print("\n--- Metabolic Demand Map Test ---")
    dummy_config = {
        "tissue_properties": {
            "metabolic_rates": {
                "gm": 0.01, "wm": 0.003, "csf": 0.0, "tumor_rim": 0.02
            }
        }
    }
    gm_seg = np.zeros((3,3,3), dtype=np.uint8)
    gm_seg[1,1,1] = 1
    wm_seg = np.zeros((3,3,3), dtype=np.uint8)
    wm_seg[0,0,0] = 1
    
    tissue_segs = {'GM': gm_seg, 'WM': wm_seg}
    voxel_vol = 2.0 # mm^3

    demand_map = get_metabolic_demand_map(tissue_segs, dummy_config, voxel_vol)
    if demand_map is not None:
        print(f"Demand map at (1,1,1) (GM): {demand_map[1,1,1]} (Expected: 0.01 * 2.0 = 0.02)")
        assert np.isclose(demand_map[1,1,1], 0.01 * voxel_vol)
        print(f"Demand map at (0,0,0) (WM): {demand_map[0,0,0]} (Expected: 0.003 * 2.0 = 0.006)")
        assert np.isclose(demand_map[0,0,0], 0.003 * voxel_vol)
        print(f"Demand map at (2,2,2) (Background): {demand_map[2,2,2]} (Expected: 0.0)")
        assert np.isclose(demand_map[2,2,2], 0.0)
        print("Metabolic demand map test successful.")
    else:
        print("Metabolic demand map test failed.")