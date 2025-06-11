# src/visualization.py
import logging
import os
import numpy as np
import networkx as nx
from typing import Optional, Dict, List, Tuple # Added Dict, List, Tuple

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None 

from src import io_utils, config_manager, utils # Added utils for voxel_to_world if needed
from src import constants

logger = logging.getLogger(__name__)

def plot_vascular_tree_pyvista(
    graph: Optional[nx.DiGraph], 
    title: str = "Vascular Tree", 
    background_color: str = "white", 
    cmap_radius: str = "viridis",
    output_screenshot_path: Optional[str] = None,
    tissue_masks: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None, # Dict: {"mask_name": (mask_data, affine)}
    seed_points_world: Optional[List[Tuple[np.ndarray, float, str]]] = None, # List: [(pos_array, radius, name), ...]
    domain_outline_color: str = 'gray',
    domain_outline_opacity: float = 0.1,
    seed_point_color: str = 'red',
    seed_point_radius_scale: float = 10.0 # Visual scale factor for seed point spheres
    ):
    """
    Uses PyVista to plot the vascular tree in 3D, optionally with tissue masks and seed points.
    """
    if not PYVISTA_AVAILABLE:
        logger.warning("PyVista is not available. Skipping 3D PyVista plot.")
        return

    plotter = pv.Plotter(off_screen=output_screenshot_path is not None, window_size=[1000,800]) # Larger window
    plotter.background_color = background_color
    plotter.add_title(title, font_size=16)

    # --- 1. Plot Tissue Masks (as contours or volumes) ---
    if tissue_masks:
        mask_colors = {"GM": "lightblue", "WM": "lightyellow", "domain_mask": "lightgray", "Tumor": "lightcoral"}
        mask_opacities = {"GM": 0.2, "WM": 0.2, "domain_mask": 0.1, "Tumor": 0.3}

        for mask_name, (mask_data, affine) in tissue_masks.items():
            if mask_data is None or not np.any(mask_data):
                logger.debug(f"Mask '{mask_name}' is empty or None. Skipping.")
                continue
            
            try:
                # Create a PyVista UniformGrid from the mask data and affine
                # We need to ensure the origin and spacing are set correctly from the affine.
                # PyVista's UniformGrid origin is the corner of the first voxel.
                # Spacing is voxel dimensions.
                dims = np.array(mask_data.shape)
                # Voxel dimensions (spacing)
                spacing = np.abs(np.diag(affine)[:3])
                # Origin: world coordinates of voxel (0,0,0)
                origin = affine[:3, 3] 
                
                # If affine has rotations/shears, UniformGrid is not ideal. StructuredGrid is better but more complex.
                # For simple scaling and translation affines (most common for masks):
                # Check if affine[:3,:3] is purely diagonal (or permutation with sign flips)
                is_simple_affine = np.count_nonzero(affine[:3,:3] - np.diag(np.diag(affine[:3,:3]))) == 0

                if not is_simple_affine:
                    logger.warning(f"Affine for mask '{mask_name}' has rotations/shears. "
                                   "Volumetric rendering might be misaligned. Consider using contours or point clouds.")
                    # Fallback: Plot point cloud of non-zero voxels for complex affines
                    voxel_indices = np.array(np.where(mask_data > 0)).T
                    if voxel_indices.shape[0] > 0:
                        world_coords = utils.voxel_to_world(voxel_indices, affine)
                        cloud = pv.PolyData(world_coords)
                        plotter.add_mesh(cloud, color=mask_colors.get(mask_name, "grey"), 
                                         opacity=mask_opacities.get(mask_name, 0.1), point_size=1)
                    continue


                grid = pv.UniformGrid(dims=dims, spacing=spacing, origin=origin)
                grid.point_data[mask_name] = mask_data.flatten(order="F") # Flatten in Fortran order for VTK
                
                # Option A: Volume rendering (can be slow, adjust opacity)
                # plotter.add_volume(grid, scalars=mask_name, cmap=[(0,0,0,0), mask_colors.get(mask_name, "grey")], opacity=[0,0.5], shade=False)

                # Option B: Contour surface (isosurface at value 0.5 for a binary mask)
                contour = grid.contour([0.5], scalars=mask_name, rng=[0,1])
                if contour.n_points > 0:
                    plotter.add_mesh(contour, color=mask_colors.get(mask_name, "grey"), 
                                     opacity=mask_opacities.get(mask_name, 0.1), style='surface')
                else:
                    logger.debug(f"No contour generated for mask '{mask_name}'. It might be too sparse or flat.")
            except Exception as e_mask:
                logger.error(f"Error processing/plotting mask '{mask_name}': {e_mask}", exc_info=True)
                
    # --- 2. Plot Seed Points ---
    if seed_points_world:
        for seed_pos, seed_radius, seed_name in seed_points_world:
            try:
                sphere = pv.Sphere(center=seed_pos, radius=seed_radius * seed_point_radius_scale) # Scale for visibility
                plotter.add_mesh(sphere, color=seed_point_color, opacity=0.8)
                # plotter.add_point_labels([seed_pos], [seed_name], font_size=10, point_color='yellow', text_color='black') # Optional label
            except Exception as e_seed:
                logger.error(f"Error plotting seed point '{seed_name}': {e_seed}", exc_info=True)

    # --- 3. Plot Vascular Tree ---
    if graph is not None and graph.number_of_nodes() > 0:
        points = []
        radii = []
        lines = []
        node_to_idx = {}
        idx_counter = 0

        min_valid_radius = constants.MIN_VESSEL_RADIUS_MM * 0.1 # for plotting small radii

        for node_id, data in graph.nodes(data=True):
            if 'pos' in data and 'radius' in data:
                points.append(data['pos'])
                # Ensure radius is positive for tube filter, even if very small
                radii.append(max(data['radius'], min_valid_radius)) 
                node_to_idx[node_id] = idx_counter
                idx_counter += 1
        
        if points:
            points_np = np.array(points)
            radii_np = np.array(radii)

            for u, v, _ in graph.edges(data=True): # Edge data not used for tubes directly here
                if u in node_to_idx and v in node_to_idx:
                    lines.extend([2, node_to_idx[u], node_to_idx[v]])
            
            if not lines:
                tree_mesh = pv.PolyData(points_np) # Point cloud
                if tree_mesh.n_points > 0:
                    # Glyph spheres at points, scaled by radius
                    spheres = pv.Sphere(radius=0.01).glyph(scale='radius', factor=0.5, geom=tree_mesh, progress_bar=True) 
                    plotter.add_mesh(spheres, scalars='radius', cmap=cmap_radius, scalar_bar_args={'title': 'Radius (mm)'})
            else:
                tree_mesh = pv.PolyData(points_np, lines=np.array(lines))
                tree_mesh.point_data['radius'] = radii_np # Assign radii to points for tube filter
                
                # Use the tube filter. The radius of the tube can be constant or vary if scalars are provided.
                # If tree_mesh.point_data['radius'] exists, tube() can use it if `vary_radius` is supported
                # or use it to scale. Let's try to make tubes based on point radii.
                # A simple way is to generate splines first, then tube them.
                try:
                    # Create tubes based on point radii. This might require some finesse.
                    # A common approach is to create a spline through connected points if segments are multi-point,
                    # then tube the spline. For simple line segments, just tube them.
                    # PyVista's tube filter radius is constant or scaled by a single scalar array.
                    # If radii vary significantly, we might need to create individual tubes per segment.
                    
                    # For simplicity, render lines as tubes and color by radius
                    # The 'radius' argument to tube() sets a base radius if scalars are not used effectively for varying radius.
                    # We want the tube's visual radius to reflect the data['radius'].
                    # This is tricky with one add_mesh call for a complex tree with varying radii.
                    
                    # Workaround: Create many small tube segments if PyVista's tube filter doesn't vary radius nicely.
                    # Or, use a line plot with thickness varying by radius if supported.
                    # For now: plot lines as tubes, colored by point radius, and add a scalar bar.
                    if tree_mesh.n_points > 0: # Ensure mesh is not empty
                        plotter.add_mesh(tree_mesh, scalars='radius', line_width=5, cmap=cmap_radius, 
                                         render_lines_as_tubes=True, 
                                         scalar_bar_args={'title': 'Radius (mm)', 'color':'black'})
                except Exception as e_tree_plot:
                     logger.error(f"Error during tree mesh plotting: {e_tree_plot}", exc_info=True)
                     # Fallback to simple lines if tube rendering fails
                     if tree_mesh.n_points > 0:
                        plotter.add_mesh(tree_mesh, scalars='radius', line_width=2, cmap=cmap_radius,
                                         scalar_bar_args={'title': 'Radius (mm)', 'color':'black'})

        else:
            logger.info("No valid points with pos/radius found in graph for tree plotting.")
    else:
        logger.info("Graph is empty or None. Skipping tree plotting.")

    plotter.camera_position = 'xy' # Or other preferred views: 'yz', 'xz', 'isometric'
    # plotter.enable_zoom_scaling() # Removed as it may not exist in newer PyVista

    if output_screenshot_path:
        plotter.show(auto_close=True, screenshot=output_screenshot_path)
        logger.info(f"Saved PyVista plot screenshot to {output_screenshot_path}")
    else:
        logger.info("Displaying PyVista plot. Close window to continue.")
        plotter.show()


def generate_final_visualizations(
    config: dict,
    output_dir: str,
    tissue_data: dict,
    vascular_graph: Optional[nx.DiGraph],
    perfusion_map: Optional[np.ndarray] = None,
    pressure_map_tissue: Optional[np.ndarray] = None):
    """
    Generates and saves final visualizations, including optional tissue masks and seed points.
    """
    logger.info("Generating final visualizations...")
    
    # --- Prepare tissue masks for plotting ---
    masks_to_plot: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if tissue_data.get('domain_mask') is not None and tissue_data.get('affine') is not None:
        # Plot domain_mask as a faint outline
        masks_to_plot["domain_mask"] = (tissue_data['domain_mask'], tissue_data['affine'])
    # Add GM, WM, Tumor if you want them visualized distinctly
    # if tissue_data.get('GM') is not None:
    #     masks_to_plot["GM"] = (tissue_data['GM'], tissue_data['affine'])
    # if tissue_data.get('Tumor') is not None:
    #     masks_to_plot["Tumor"] = (tissue_data['Tumor'], tissue_data['affine'])

    # --- Prepare seed points for plotting ---
    seed_points_viz_data: List[Tuple[np.ndarray, float, str]] = []
    config_seeds = config_manager.get_param(config, "gbo_growth.seed_points", [])
    if config_seeds:
        for i, seed_info in enumerate(config_seeds):
            pos = np.array(seed_info.get('position', [0,0,0]))
            radius = float(seed_info.get('initial_radius', 0.1)) # Visual radius for the seed marker
            name = seed_info.get('id', f"Seed_{i}")
            seed_points_viz_data.append((pos, radius, name))

    # --- Plotting ---
    if vascular_graph is None or vascular_graph.number_of_nodes() == 0:
        logger.warning("Vascular graph is empty. Limited visualizations will be produced.")
        # Still plot masks and seeds if available
        if (masks_to_plot or seed_points_viz_data) and PYVISTA_AVAILABLE:
            pv_screenshot_path = os.path.join(output_dir, "tissue_and_seeds_plot.png")
            plot_vascular_tree_pyvista(
                graph=None, # No graph to plot
                title="Tissue Masks and Seed Points",
                output_screenshot_path=pv_screenshot_path,
                tissue_masks=masks_to_plot,
                seed_points_world=seed_points_viz_data,
                background_color=config_manager.get_param(config, "visualization.pyvista_background_color", "white")
            )
        return # Exit if no graph

    final_tree_vtp_path = os.path.join(output_dir, "final_plot_vascular_tree.vtp") # Different name for clarity
    io_utils.save_vascular_tree_vtp(vascular_graph, final_tree_vtp_path)
    logger.info(f"Final vascular tree saved for visualization plotting: {final_tree_vtp_path}")

    if PYVISTA_AVAILABLE:
        pv_screenshot_path = os.path.join(output_dir, "final_vascular_tree_with_context_3D.png")
        try:
            plot_vascular_tree_pyvista(
                vascular_graph, 
                title="Final Generated Vasculature with Context",
                output_screenshot_path=pv_screenshot_path,
                tissue_masks=masks_to_plot,
                seed_points_world=seed_points_viz_data,
                background_color=config_manager.get_param(config, "visualization.pyvista_background_color", "white"),
                cmap_radius=config_manager.get_param(config, "visualization.pyvista_cmap_radius", "viridis") # New config param
            )
        except Exception as e:
            logger.error(f"Error during PyVista plotting: {e}", exc_info=True)
    else:
        logger.info("PyVista not installed. Skipping 3D plot generation.")
    
    logger.info("Final visualizations generation attempt complete.")

# Add to default config.yaml if not there:
# visualization:
#   pyvista_cmap_radius: "plasma" 