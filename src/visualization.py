# src/visualization.py
import logging
import os
import numpy as np
import networkx as nx
from typing import Optional
# Attempt to import PyVista, but make it optional for non-interactive/headless environments
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None 

from src import io_utils, config_manager

logger = logging.getLogger(__name__)

def plot_vascular_tree_pyvista(graph: nx.DiGraph, title: str = "Vascular Tree", 
                                 background_color: str = "white", cmap_radius: str = "viridis",
                                 output_screenshot_path: Optional[str] = None):
    """
    Uses PyVista to plot the vascular tree in 3D.
    Requires PyVista to be installed.
    """
    if not PYVISTA_AVAILABLE:
        logger.warning("PyVista is not available. Skipping 3D PyVista plot.")
        return
    if graph is None or graph.number_of_nodes() == 0:
        logger.info("Graph is empty or None. Skipping PyVista plot.")
        return

    points = []
    radii = []
    lines = []
    node_to_idx = {}
    idx_counter = 0

    for node_id, data in graph.nodes(data=True):
        if 'pos' in data and 'radius' in data:
            points.append(data['pos'])
            radii.append(data['radius'])
            node_to_idx[node_id] = idx_counter
            idx_counter += 1
        else:
            logger.warning(f"Node {node_id} missing 'pos' or 'radius'. Skipping for PyVista plot.")
            # Add a placeholder to keep indices consistent if some nodes are skipped
            # Or ensure all nodes always have pos/radius for robust plotting.
            # For now, if a node is skipped, edges to it will also be skipped.

    if not points:
        logger.info("No valid points with pos/radius found in graph. Skipping PyVista plot.")
        return
        
    points_np = np.array(points)
    radii_np = np.array(radii)

    for u, v, data in graph.edges(data=True):
        if u in node_to_idx and v in node_to_idx:
            lines.extend([2, node_to_idx[u], node_to_idx[v]]) # 2 points per line segment

    if not lines: # No valid edges connected to plottable nodes
        # Plot as a point cloud if no lines
        poly_data = pv.PolyData(points_np)
        logger.info("No valid edges for lines. Plotting nodes as point cloud.")
    else:
        poly_data = pv.PolyData(points_np, lines=np.array(lines))
    
    poly_data.point_data['radius'] = radii_np
    
    # Create tubes for visualization
    # The tube radius should be derived from the point_data 'radius'
    # PyVista's `tube` filter uses a single radius or varies by scalars.
    # To vary tube radius per point, we might need to generate lines and then glyph them or use tubes with varying radius if supported.
    # A simpler approach is to color by radius and use a representative tube size.
    
    plotter = pv.Plotter(off_screen=output_screenshot_path is not None, window_size=[800,600])
    plotter.background_color = background_color
    
    # Option 1: Plot lines colored by radius
    # plotter.add_mesh(poly_data, scalars='radius', line_width=5, cmap=cmap_radius, render_lines_as_tubes=True, scalar_bar_args={'title': 'Radius'})

    # Option 2: Create tubes directly (might be slow for large trees, radius might be constant for whole tube)
    # If we want varying tube radius along segments, it's more complex.
    # For now, let's make tubes from the lines, and scale by point radii.
    # This requires converting lines to tubes, perhaps using `glyph` with spheres at points scaled by radius,
    # and tubes for segments using average radius.

    # Simpler: Render lines as tubes, and also add spheres at nodes scaled by radius.
    if poly_data.n_lines > 0:
        plotter.add_mesh(poly_data.tube(radius=np.min(radii_np[radii_np > 0]) if np.any(radii_np > 0) else 0.001, n_sides=8), 
                         scalars='radius', cmap=cmap_radius, scalar_bar_args={'title': 'Radius (mm)'})
    elif poly_data.n_points > 0: # Only points if no lines
        spheres = pv.Sphere(radius=0.01).glyph(scale='radius', factor=1.0, geom=poly_data) # factor might need adjustment
        plotter.add_mesh(spheres, scalars='radius', cmap=cmap_radius, scalar_bar_args={'title': 'Radius (mm)'})


    plotter.add_title(title, font_size=16)
    plotter.camera_position = 'xy'


    if output_screenshot_path:
        plotter.show(auto_close=True, screenshot=output_screenshot_path)
        logger.info(f"Saved PyVista plot screenshot to {output_screenshot_path}")
    else:
        logger.info("Displaying PyVista plot. Close window to continue.")
        plotter.show() # Interactive plot

def generate_final_visualizations(
    config: dict,
    output_dir: str,
    tissue_data: dict, # For potential overlays
    vascular_graph: Optional[nx.DiGraph],
    perfusion_map: Optional[np.ndarray] = None,
    pressure_map_tissue: Optional[np.ndarray] = None):
    """
    Generates and saves final visualizations.
    """
    logger.info("Generating final visualizations...")
    
    if vascular_graph is None or vascular_graph.number_of_nodes() == 0:
        logger.warning("Vascular graph is empty. Skipping visualizations.")
        return

    # 1. Save the final vascular tree again (redundant if already saved, but good for clarity)
    final_tree_vtp_path = os.path.join(output_dir, "final_vascular_tree_visualization.vtp")
    io_utils.save_vascular_tree_vtp(vascular_graph, final_tree_vtp_path)
    logger.info(f"Final vascular tree saved for visualization: {final_tree_vtp_path}")

    # 2. Generate a 3D plot using PyVista (if available)
    if PYVISTA_AVAILABLE:
        pv_screenshot_path = os.path.join(output_dir, "final_vascular_tree_3D_plot.png")
        try:
            plot_vascular_tree_pyvista(vascular_graph, 
                                       title="Final Generated Vasculature",
                                       output_screenshot_path=pv_screenshot_path,
                                       background_color=config_manager.get_param(config, "visualization.pyvista_background_color", "white"),
                                       cmap_radius=config_manager.get_param(config, "visualization.pyvista_cmap_perfusion", "viridis") # Using perfusion cmap for radius for now
                                       )
        except Exception as e:
            logger.error(f"Error during PyVista plotting: {e}", exc_info=True)
    else:
        logger.info("PyVista not installed. Skipping 3D plot generation.")

    # 3. TODO: 2D slice plots overlaying perfusion/pressure on anatomy
    # This would use matplotlib and nibabel.
    # Example:
    # if perfusion_map is not None and tissue_data.get('GM') is not None:
    #     slice_idx = config_manager.get_param(config, "visualization.plot_slice_index", -1)
    #     axis = config_manager.get_param(config, "visualization.plot_slice_axis", "axial")
    #     # ... logic to select slice and plot using matplotlib ...
    #     logger.info(f"Generated 2D overlay plot for perfusion (axis: {axis}, slice: {slice_idx})")
    
    logger.info("Final visualizations generation attempt complete.")

if __name__ == '__main__':
    # Example usage for visualization.py itself (requires a dummy graph)
    logging.basicConfig(level=logging.DEBUG)
    if not PYVISTA_AVAILABLE:
        print("PyVista not available, cannot run visualization example.")
    else:
        print("Running visualization.py example...")
        dummy_graph = nx.DiGraph()
        data_structures.add_node_to_graph(dummy_graph, 0, pos=np.array([0,0,0]), radius=0.5)
        data_structures.add_node_to_graph(dummy_graph, 1, pos=np.array([1,0,0]), radius=0.4)
        data_structures.add_node_to_graph(dummy_graph, 2, pos=np.array([1,1,0]), radius=0.3)
        data_structures.add_node_to_graph(dummy_graph, 3, pos=np.array([1,-1,0]), radius=0.3)
        data_structures.add_edge_to_graph(dummy_graph, 0, 1)
        data_structures.add_edge_to_graph(dummy_graph, 1, 2)
        data_structures.add_edge_to_graph(dummy_graph, 1, 3)

        # Create a dummy output dir for this example
        example_output_dir = "temp_vis_example_output"
        os.makedirs(example_output_dir, exist_ok=True)
        
        plot_vascular_tree_pyvista(dummy_graph, title="Visualization Example", 
                                     output_screenshot_path=os.path.join(example_output_dir, "vis_example.png"))
        
        # Dummy config for generate_final_visualizations
        dummy_config_vis = {
            "visualization": {
                "pyvista_background_color": "lightgrey",
                "pyvista_cmap_perfusion": "coolwarm"
            }
        }
        # generate_final_visualizations might need more from tissue_data for full functionality
        dummy_tissue_data_vis = {'affine': np.eye(4)} 

        generate_final_visualizations(dummy_config_vis, example_output_dir, dummy_tissue_data_vis, dummy_graph)
        
        print(f"Visualization example outputs (if any) saved in {example_output_dir}")
        # For interactive, comment out output_screenshot_path in plot_vascular_tree_pyvista call.