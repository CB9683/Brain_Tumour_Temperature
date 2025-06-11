# src/visualization.py
from __future__ import annotations # Must be first line

import logging
import os
import numpy as np
import networkx as nx
from typing import Optional, Dict, List, Tuple 
import pandas as pd # For saving data to CSV

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None 

# Make matplotlib an optional import for headless environments
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


from src import io_utils, config_manager, utils, constants # constants was missing

logger = logging.getLogger(__name__)

# --- Plotting helper for distributions ---
def _plot_histogram(data: List[float], title: str, xlabel: str, output_path: str, bins: int = 30):
    if not MATPLOTLIB_AVAILABLE:
        logger.warning(f"Matplotlib not available. Skipping histogram plot: {title}")
        return
    if not data:
        logger.warning(f"No data to plot for histogram: {title}")
        return

    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bins, color='skyblue', edgecolor='black', density=False) # Use density=True for PDF-like
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    try:
        plt.savefig(output_path)
        logger.info(f"Saved histogram '{title}' to {output_path}")
    except Exception as e:
        logger.error(f"Error saving histogram '{title}' to {output_path}: {e}")
    plt.close()

# --- Functions for Quantitative Analysis ---

def analyze_radii_distribution(graph: nx.DiGraph, output_dir: str, filename_prefix: str = "final_"):
    """Calculates and plots the distribution of radii for synthetic vessels."""
    if graph is None or graph.number_of_nodes() == 0: return
    
    radii = [data['radius'] for _, data in graph.nodes(data=True) 
             if 'radius' in data and data.get('is_synthetic', False)] # Consider only synthetic node radii
    
    if not radii:
        logger.info("No synthetic nodes with radii found for distribution analysis.")
        return

    # Save raw radii data
    df_radii = pd.DataFrame(radii, columns=['radius_mm'])
    csv_path = os.path.join(output_dir, f"{filename_prefix}radii_data.csv")
    df_radii.to_csv(csv_path, index=False)
    logger.info(f"Saved raw radii data to {csv_path}")

    # Plot histogram
    plot_path = os.path.join(output_dir, f"{filename_prefix}radii_distribution.png")
    _plot_histogram(radii, "Distribution of Synthetic Vessel Radii", "Radius (mm)", plot_path)

def analyze_segment_lengths(graph: nx.DiGraph, output_dir: str, filename_prefix: str = "final_"):
    """Calculates and plots the distribution of synthetic segment lengths."""
    if graph is None or graph.number_of_edges() == 0: return

    lengths = [data['length'] for u, v, data in graph.edges(data=True)
               if 'length' in data and 
                  (graph.nodes[u].get('is_synthetic', False) or graph.nodes[v].get('is_synthetic', False))] # Edge involving synthetic node
    
    if not lengths:
        logger.info("No synthetic segments with lengths found for distribution analysis.")
        return

    df_lengths = pd.DataFrame(lengths, columns=['length_mm'])
    csv_path = os.path.join(output_dir, f"{filename_prefix}segment_lengths_data.csv")
    df_lengths.to_csv(csv_path, index=False)
    logger.info(f"Saved raw segment length data to {csv_path}")

    plot_path = os.path.join(output_dir, f"{filename_prefix}segment_lengths_distribution.png")
    _plot_histogram(lengths, "Distribution of Synthetic Segment Lengths", "Length (mm)", plot_path)


def analyze_murray_law(graph: nx.DiGraph, output_dir: str, murray_exponent: float = 3.0, filename_prefix: str = "final_"):
    """Tests compliance with Murray's Law at synthetic bifurcations."""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available. Skipping Murray's Law plot.")
        return
    if graph is None or graph.number_of_nodes() == 0: return

    parent_powers: List[float] = []
    children_sum_powers: List[float] = []
    bifurcation_data_for_csv: List[Dict] = []

    for node_id, data in graph.nodes(data=True):
        # Only consider synthetic bifurcations
        if data.get('type') == 'synthetic_bifurcation' and graph.out_degree(node_id) == 2:
            children_ids = list(graph.successors(node_id))
            
            r_p_node = data.get('radius') # Radius of the bifurcation node itself
            if r_p_node is None or r_p_node < constants.EPSILON: continue

            r_c1_node = graph.nodes[children_ids[0]].get('radius')
            r_c2_node = graph.nodes[children_ids[1]].get('radius')

            if r_c1_node and r_c2_node and r_c1_node > constants.EPSILON and r_c2_node > constants.EPSILON:
                p_power = r_p_node**murray_exponent
                c_sum_power = r_c1_node**murray_exponent + r_c2_node**murray_exponent
                
                parent_powers.append(p_power)
                children_sum_powers.append(c_sum_power)
                bifurcation_data_for_csv.append({
                    'bifurcation_node_id': node_id,
                    'parent_radius_cubed': p_power,
                    'children_radii_cubed_sum': c_sum_power,
                    'parent_radius': r_p_node,
                    'child1_radius': r_c1_node,
                    'child2_radius': r_c2_node
                })
            
    if not parent_powers:
        logger.info("No valid synthetic bifurcations found for Murray's Law test.")
        return

    # Save raw data
    df_murray = pd.DataFrame(bifurcation_data_for_csv)
    csv_path = os.path.join(output_dir, f"{filename_prefix}murray_law_data.csv")
    df_murray.to_csv(csv_path, index=False)
    logger.info(f"Saved Murray's Law raw data to {csv_path}")

    # Plot
    plt.figure(figsize=(7, 7))
    plt.scatter(children_sum_powers, parent_powers, alpha=0.6, edgecolors='k', s=40, label="Bifurcations")
    
    max_val = 0
    if parent_powers and children_sum_powers: # Ensure lists are not empty
        max_val = max(max(parent_powers, default=0), max(children_sum_powers, default=0)) * 1.1
    
    plt.plot([0, max_val], [0, max_val], 'r--', label=f'Ideal Murray (y=x, exp={murray_exponent:.1f})')
    plt.xlabel(f"Sum of Children Radii^{murray_exponent:.0f} (r$_1^{murray_exponent:.0f}$ + r$_2^{murray_exponent:.0f}$)")
    plt.ylabel(f"Parent Radius^{murray_exponent:.0f} (r$_0^{murray_exponent:.0f}$)")
    plt.title("Murray's Law Compliance Test (Synthetic Bifurcations)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    if max_val > 0 : # Only set limits if there's data
        plt.xlim([0, max_val])
        plt.ylim([0, max_val])
    
    plot_path = os.path.join(output_dir, f"{filename_prefix}murray_law_compliance.png")
    try:
        plt.savefig(plot_path)
        logger.info(f"Saved Murray's Law plot to {plot_path}")
    except Exception as e:
        logger.error(f"Error saving Murray's Law plot to {plot_path}: {e}")
    plt.close()

def analyze_branching_angles(graph: nx.DiGraph, output_dir: str, filename_prefix: str = "final_"):
    """Calculates and plots distribution of branching angles at synthetic bifurcations."""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available. Skipping branching angle plot.")
        return
    if graph is None or graph.number_of_nodes() == 0: return

    branching_angles_c1_c2: List[float] = [] # Angle between two children
    # Angles relative to parent requires knowing parent segment direction
    # For now, just angle between children

    angle_data_for_csv: List[Dict] = []

    for node_id, data in graph.nodes(data=True):
        if data.get('type') == 'synthetic_bifurcation' and graph.out_degree(node_id) == 2:
            parent_pos = data.get('pos')
            if parent_pos is None: continue

            children_ids = list(graph.successors(node_id))
            child1_pos = graph.nodes[children_ids[0]].get('pos')
            child2_pos = graph.nodes[children_ids[1]].get('pos')

            if child1_pos is not None and child2_pos is not None:
                vec1 = child1_pos - parent_pos
                vec2 = child2_pos - parent_pos
                
                norm_vec1 = np.linalg.norm(vec1)
                norm_vec2 = np.linalg.norm(vec2)

                if norm_vec1 > constants.EPSILON and norm_vec2 > constants.EPSILON:
                    cosine_angle = np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)
                    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) # Clip for stability
                    angle_deg = np.degrees(angle_rad)
                    branching_angles_c1_c2.append(angle_deg)
                    angle_data_for_csv.append({
                        'bifurcation_node_id': node_id,
                        'angle_degrees': angle_deg
                    })
    
    if not branching_angles_c1_c2:
        logger.info("No valid synthetic bifurcations found for branching angle analysis.")
        return

    df_angles = pd.DataFrame(angle_data_for_csv)
    csv_path = os.path.join(output_dir, f"{filename_prefix}branching_angles_data.csv")
    df_angles.to_csv(csv_path, index=False)
    logger.info(f"Saved branching angle data to {csv_path}")

    plot_path = os.path.join(output_dir, f"{filename_prefix}branching_angles_distribution.png")
    _plot_histogram(branching_angles_c1_c2, "Distribution of Branching Angles (Child-Child)", 
                    "Angle (degrees)", plot_path, bins=18) # e.g. 10 degree bins up to 180

# --- Main Visualization Functions (plot_vascular_tree_pyvista remains mostly the same) ---
# (plot_vascular_tree_pyvista as previously defined, ensure 'constants' is imported and used correctly)
# Small correction in plot_vascular_tree_pyvista for min_valid_radius usage
def plot_vascular_tree_pyvista(
    graph: Optional[nx.DiGraph], 
    title: str = "Vascular Tree", 
    # ... other parameters ...
    seed_point_radius_scale: float = 10.0 
    ):
    if not PYVISTA_AVAILABLE: # ... (same as before) ...
        return
    # ... (plotter setup) ...
    # ... (tissue mask plotting - same as before) ...
    # ... (seed point plotting - same as before) ...

    if graph is not None and graph.number_of_nodes() > 0:
        points = []
        radii = [] # This will store radii for point_data
        lines = []
        node_to_idx = {}
        idx_counter = 0
        
        # Use a small positive value for radii if actual radius is zero or too small for PyVista filters
        min_plot_radius = constants.MIN_VESSEL_RADIUS_MM * 0.1 
        if min_plot_radius <= 0: min_plot_radius = 1e-5 # Absolute fallback

        for node_id, data in graph.nodes(data=True):
            if 'pos' in data and 'radius' in data:
                points.append(data['pos'])
                radii.append(max(data['radius'], min_plot_radius)) # Ensure positive for tube filter
                node_to_idx[node_id] = idx_counter
                idx_counter += 1
        
        if not points:
            logger.info("No valid points with pos/radius found in graph for tree plotting.")
        else:
            points_np = np.array(points)
            radii_np = np.array(radii) # This is now populated with positive radii

            for u, v, _ in graph.edges(data=True):
                if u in node_to_idx and v in node_to_idx:
                    lines.extend([2, node_to_idx[u], node_to_idx[v]])
            
            if not lines:
                tree_mesh = pv.PolyData(points_np)
                if tree_mesh.n_points > 0:
                    tree_mesh.point_data['radius'] = radii_np # Add radii for glyph scaling
                    tree_mesh.active_scalars_name = 'radius' # Explicitly set for glyph
                    try:
                        spheres = pv.Sphere(radius=0.01).glyph(scale='radius', factor=0.5, geom=tree_mesh)
                        plotter.add_mesh(spheres, scalars='radius', cmap=cmap_radius, scalar_bar_args={'title': 'Radius (mm)', 'color':'black'})
                    except KeyError: # Fallback if glyph still fails
                        logger.warning("Glyphing point cloud failed, plotting simple points.")
                        plotter.add_points(points_np, render_points_as_spheres=True, point_size=5, color='red')
            else:
                tree_mesh = pv.PolyData(points_np, lines=np.array(lines))
                tree_mesh.point_data['radius'] = radii_np # Assign radii to points
                
                if tree_mesh.n_points > 0:
                    try:
                        plotter.add_mesh(tree_mesh, scalars='radius', line_width=5, cmap=cmap_radius, 
                                         render_lines_as_tubes=True, 
                                         scalar_bar_args={'title': 'Radius (mm)', 'color':'black'})
                    except Exception as e_tree_plot: # Fallback
                        logger.error(f"Error during tree tube plotting: {e_tree_plot}", exc_info=False)
                        plotter.add_mesh(tree_mesh, scalars='radius', line_width=2, cmap=cmap_radius,
                                         scalar_bar_args={'title': 'Radius (mm)', 'color':'black'})
    # ... (rest of plotter.show() logic - same as before) ...


def generate_final_visualizations(
    config: dict,
    output_dir: str,
    tissue_data: dict,
    vascular_graph: Optional[nx.DiGraph],
    perfusion_map: Optional[np.ndarray] = None,
    pressure_map_tissue: Optional[np.ndarray] = None):
    logger.info("Generating final visualizations and quantitative analyses...")
    
    # --- Prepare common data ---
    masks_to_plot: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if tissue_data.get('domain_mask') is not None and tissue_data.get('affine') is not None:
        masks_to_plot["domain_mask"] = (tissue_data['domain_mask'], tissue_data['affine'])

    seed_points_viz_data: List[Tuple[np.ndarray, float, str]] = []
    config_seeds = config_manager.get_param(config, "gbo_growth.seed_points", [])
    if config_seeds:
        for i, seed_info in enumerate(config_seeds):
            pos = np.array(seed_info.get('position', [0,0,0]))
            radius = float(seed_info.get('initial_radius', 0.1)) 
            name = seed_info.get('id', f"Seed_{i}")
            seed_points_viz_data.append((pos, radius, name)) # Using actual initial_radius for marker size

    # --- 3D Plot ---
    if vascular_graph is None or vascular_graph.number_of_nodes() == 0:
        logger.warning("Vascular graph is empty for final visualization.")
        if (masks_to_plot or seed_points_viz_data) and PYVISTA_AVAILABLE:
            pv_screenshot_path = os.path.join(output_dir, "context_only_plot.png")
            plot_vascular_tree_pyvista(graph=None, title="Tissue & Seeds (No Vasculature)",
                                       output_screenshot_path=pv_screenshot_path, tissue_masks=masks_to_plot,
                                       seed_points_world=seed_points_viz_data)
    else:
        final_tree_vtp_path = os.path.join(output_dir, "final_plot_vascular_tree.vtp")
        io_utils.save_vascular_tree_vtp(vascular_graph, final_tree_vtp_path)
        logger.info(f"Final vascular tree saved for visualization plotting: {final_tree_vtp_path}")

        if PYVISTA_AVAILABLE:
            pv_screenshot_path = os.path.join(output_dir, "final_vascular_tree_with_context_3D.png")
            plot_vascular_tree_pyvista(
                vascular_graph, title="Final Generated Vasculature with Context",
                output_screenshot_path=pv_screenshot_path, tissue_masks=masks_to_plot,
                seed_points_world=seed_points_viz_data,
                background_color=config_manager.get_param(config, "visualization.pyvista_background_color", "white"),
                cmap_radius=config_manager.get_param(config, "visualization.pyvista_cmap_radius", "plasma"),
                seed_point_radius_scale=config_manager.get_param(config, "visualization.seed_marker_radius_scale", 5.0) # Configurable scale
            )
    
    # --- Quantitative Analysis Plots ---
    if vascular_graph and vascular_graph.number_of_nodes() > 0:
        analyze_radii_distribution(vascular_graph, output_dir)
        analyze_segment_lengths(vascular_graph, output_dir)
        analyze_murray_law(vascular_graph, output_dir, 
                           murray_exponent=config_manager.get_param(config, "vascular_properties.murray_law_exponent", 3.0))
        analyze_branching_angles(vascular_graph, output_dir)

    # TODO: Add 2D slice plots for perfusion/pressure if they exist

    logger.info("Final visualizations and analyses generation attempt complete.")

# Ensure __main__ block from previous version for testing visualization.py itself is kept if desired.