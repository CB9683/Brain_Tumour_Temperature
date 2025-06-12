# src/visualization.py
from __future__ import annotations # Must be first line

import logging
import os
import numpy as np
import networkx as nx
from typing import Optional, Dict, List, Tuple 
import pandas as pd 

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None 

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

from src import io_utils, config_manager, utils, constants

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
    plt.hist(data, bins=bins, color='skyblue', edgecolor='black', density=False)
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
    if graph is None or graph.number_of_nodes() == 0: return    
    radii = [data['radius'] for _, data in graph.nodes(data=True) 
             if 'radius' in data and data.get('is_synthetic', True)] # Default to True if 'is_synthetic' missing
    if not radii: logger.info("No nodes with radii found for distribution analysis."); return
    df_radii = pd.DataFrame(radii, columns=['radius_mm'])
    csv_path = os.path.join(output_dir, f"{filename_prefix}radii_data.csv")
    df_radii.to_csv(csv_path, index=False); logger.info(f"Saved raw radii data to {csv_path}")
    plot_path = os.path.join(output_dir, f"{filename_prefix}radii_distribution.png")
    _plot_histogram(radii, "Distribution of Vessel Radii (Synthetic Nodes)", "Radius (mm)", plot_path)

def analyze_segment_lengths(graph: nx.DiGraph, output_dir: str, filename_prefix: str = "final_"):
    if graph is None or graph.number_of_edges() == 0: return
    lengths = [data['length'] for u, v, data in graph.edges(data=True)
               if 'length' in data and 
                  (graph.nodes[u].get('is_synthetic', True) or graph.nodes[v].get('is_synthetic', True))]
    if not lengths: logger.info("No segments with lengths found for distribution analysis."); return
    df_lengths = pd.DataFrame(lengths, columns=['length_mm'])
    csv_path = os.path.join(output_dir, f"{filename_prefix}segment_lengths_data.csv")
    df_lengths.to_csv(csv_path, index=False); logger.info(f"Saved raw segment length data to {csv_path}")
    plot_path = os.path.join(output_dir, f"{filename_prefix}segment_lengths_distribution.png")
    _plot_histogram(lengths, "Distribution of Segment Lengths", "Length (mm)", plot_path)

def analyze_murray_law(graph: nx.DiGraph, output_dir: str, murray_exponent: float = 3.0, filename_prefix: str = "final_"):
    if not MATPLOTLIB_AVAILABLE: logger.warning("Matplotlib not available. Skipping Murray's Law plot."); return
    if graph is None or graph.number_of_nodes() == 0: return
    parent_powers: List[float] = []; children_sum_powers: List[float] = []; bifurcation_data_for_csv: List[Dict] = []
    for node_id, data in graph.nodes(data=True):
        if data.get('type') == 'synthetic_bifurcation' and graph.out_degree(node_id) == 2:
            children_ids = list(graph.successors(node_id))
            r_p_node = data.get('radius')
            if r_p_node is None or r_p_node < constants.EPSILON: continue
            r_c1_node = graph.nodes[children_ids[0]].get('radius')
            r_c2_node = graph.nodes[children_ids[1]].get('radius')
            if r_c1_node and r_c2_node and r_c1_node > constants.EPSILON and r_c2_node > constants.EPSILON:
                p_power = r_p_node**murray_exponent
                c_sum_power = r_c1_node**murray_exponent + r_c2_node**murray_exponent
                parent_powers.append(p_power); children_sum_powers.append(c_sum_power)
                bifurcation_data_for_csv.append({'b_id': node_id, 'rP^exp': p_power, 'sum_rC^exp': c_sum_power, 
                                                 'rP': r_p_node, 'rC1': r_c1_node, 'rC2': r_c2_node})
    if not parent_powers: logger.info("No valid synthetic bifurcations for Murray's Law test."); return
    df_murray = pd.DataFrame(bifurcation_data_for_csv); df_murray.to_csv(os.path.join(output_dir, f"{filename_prefix}murray_law_data.csv"), index=False)
    logger.info(f"Saved Murray's Law raw data to {os.path.join(output_dir, f'{filename_prefix}murray_law_data.csv')}")
    plt.figure(figsize=(7, 7)); plt.scatter(children_sum_powers, parent_powers, alpha=0.6, edgecolors='k', s=40, label="Bifurcations")
    max_val = max(max(parent_powers, default=0), max(children_sum_powers, default=0)) * 1.1 if parent_powers and children_sum_powers else 1.0
    plt.plot([0, max_val], [0, max_val], 'r--', label=f'Ideal (exp={murray_exponent:.1f})'); plt.xlabel(f"Sum(Child Radii^{murray_exponent:.0f})")
    plt.ylabel(f"Parent Radius^{murray_exponent:.0f}"); plt.title("Murray's Law Test"); plt.legend(); plt.grid(True)
    plt.axis('equal') 
    # Only set limits if max_val is meaningful and to avoid issues if data is all zero
    if max_val > constants.EPSILON: # Check if max_val is substantially greater than zero
        plt.xlim([0, max_val])
        plt.ylim([0, max_val])
    # else: let matplotlib auto-scale if all data is near zero.
    plot_path = os.path.join(output_dir, f"{filename_prefix}murray_law_compliance.png")
    try: plt.savefig(plot_path); logger.info(f"Saved Murray's Law plot to {plot_path}"); plt.close()
    except Exception as e: logger.error(f"Error saving Murray's Law plot: {e}"); plt.close()


def analyze_branching_angles(graph: nx.DiGraph, output_dir: str, filename_prefix: str = "final_"):
    if not MATPLOTLIB_AVAILABLE: logger.warning("Matplotlib not available. Skipping branching angle plot."); return
    if graph is None or graph.number_of_nodes() == 0: return
    branching_angles_c1_c2: List[float] = []; angle_data_for_csv: List[Dict] = []
    for node_id, data in graph.nodes(data=True):
        if data.get('type') == 'synthetic_bifurcation' and graph.out_degree(node_id) == 2:
            parent_pos = data.get('pos'); successors = list(graph.successors(node_id))
            if parent_pos is None or len(successors) != 2: continue
            child1_pos = graph.nodes[successors[0]].get('pos'); child2_pos = graph.nodes[successors[1]].get('pos')
            if child1_pos is not None and child2_pos is not None:
                vec1 = child1_pos - parent_pos; vec2 = child2_pos - parent_pos
                norm_vec1 = np.linalg.norm(vec1); norm_vec2 = np.linalg.norm(vec2)
                if norm_vec1 > constants.EPSILON and norm_vec2 > constants.EPSILON:
                    angle_deg = np.degrees(np.arccos(np.clip(np.dot(vec1, vec2) / (norm_vec1 * norm_vec2), -1.0, 1.0)))
                    branching_angles_c1_c2.append(angle_deg)
                    angle_data_for_csv.append({'b_id': node_id, 'angle_deg': angle_deg})
    if not branching_angles_c1_c2: logger.info("No bifurcations for branching angle analysis."); return
    df_angles = pd.DataFrame(angle_data_for_csv); df_angles.to_csv(os.path.join(output_dir, f"{filename_prefix}branching_angles_data.csv"), index=False)
    logger.info(f"Saved branching angle data to {os.path.join(output_dir, f'{filename_prefix}branching_angles_data.csv')}")
    plot_path = os.path.join(output_dir, f"{filename_prefix}branching_angles_distribution.png")
    _plot_histogram(branching_angles_c1_c2, "Distribution of Branching Angles (Child-Child)", "Angle (degrees)", plot_path, bins=18)


def plot_vascular_tree_pyvista(
    graph: Optional[nx.DiGraph], 
    title: str = "Vascular Tree", 
    background_color: str = "white", 
    cmap_radius: str = "viridis",
    output_screenshot_path: Optional[str] = None,
    tissue_masks: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    seed_points_world: Optional[List[Tuple[np.ndarray, float, str]]] = None,
    domain_outline_color: str = 'gray',
    domain_outline_opacity: float = 0.1,
    seed_point_color: str = 'red',
    seed_point_radius_scale: float = 10.0 
    ):
    if not PYVISTA_AVAILABLE: logger.warning("PyVista not available. Skipping 3D PyVista plot."); return
    plotter = pv.Plotter(off_screen=output_screenshot_path is not None, window_size=[1000,800])
    plotter.background_color = background_color; plotter.add_title(title, font_size=16)

    if tissue_masks:
        mask_colors = {"GM": "lightblue", "WM": "lightyellow", "domain_mask": domain_outline_color, "Tumor": "lightcoral"}
        mask_opacities = {"GM": 0.2, "WM": 0.2, "domain_mask": domain_outline_opacity, "Tumor": 0.3}
        for mask_name, (mask_data, affine) in tissue_masks.items():
            if mask_data is None or not np.any(mask_data): continue
            try:
                dims = np.array(mask_data.shape)
                spacing = np.abs(np.diag(affine)[:3])
                origin = affine[:3, 3] 
                
                # is_simple_affine = ... (keep this check) ...
                # if not is_simple_affine and mask_name=="domain_mask": ... (keep warning) ...

                # grid = pv.UniformGrid(dims=dims, spacing=spacing, origin=origin) # OLD LINE
                grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=origin) # NEW LINE
                
                grid.point_data[mask_name] = mask_data.flatten(order="F").astype(float)
                
                contour = grid.contour([0.5], scalars=mask_name, rng=[0,1])
                if contour.n_points > 0: 
                    plotter.add_mesh(contour, color=mask_colors.get(mask_name, "grey"), 
                                     opacity=mask_opacities.get(mask_name, 0.1), style='surface')
            except Exception as e_mask: logger.error(f"Error plotting mask '{mask_name}': {e_mask}", exc_info=True)
                
    if seed_points_world:
        for seed_pos, seed_initial_radius, seed_name in seed_points_world:
            try:
                marker_radius = max(seed_initial_radius * seed_point_radius_scale, 0.05)
                sphere = pv.Sphere(center=seed_pos, radius=marker_radius)
                plotter.add_mesh(sphere, color=seed_point_color, opacity=0.8)
            except Exception as e_seed: logger.error(f"Error plotting seed '{seed_name}': {e_seed}", exc_info=True)

    if graph is not None and graph.number_of_nodes() > 0:
        points, radii_for_plot, lines, node_to_idx, idx_counter = [], [], [], {}, 0
        min_plot_radius = max(constants.MIN_VESSEL_RADIUS_MM * 0.1, 1e-5)
        for node_id, data in graph.nodes(data=True):
            if 'pos' in data and 'radius' in data:
                points.append(data['pos']); radii_for_plot.append(max(data['radius'], min_plot_radius))
                node_to_idx[node_id] = idx_counter; idx_counter += 1
        if points:
            points_np, radii_np_for_plot = np.array(points), np.array(radii_for_plot)
            for u, v, _ in graph.edges(data=True):
                if u in node_to_idx and v in node_to_idx: lines.extend([2, node_to_idx[u], node_to_idx[v]])
            tree_mesh = pv.PolyData(); tree_mesh.points = points_np
            if lines: tree_mesh.lines = np.array(lines)
            if tree_mesh.n_points > 0:
                tree_mesh.point_data['radius'] = radii_np_for_plot
                if tree_mesh.n_cells > 0: # Lines exist
                    try: plotter.add_mesh(tree_mesh, scalars='radius', line_width=5, cmap=cmap_radius, 
                                          render_lines_as_tubes=True, scalar_bar_args={'title': 'Radius (mm)', 'color':'black'})
                    except Exception: plotter.add_mesh(tree_mesh, scalars='radius', line_width=2, cmap=cmap_radius, scalar_bar_args={'title': 'Radius (mm)', 'color':'black'}) # Fallback
                elif tree_mesh.n_points > 0: # Only points
                    logger.info("Graph has points but no lines. Plotting as scaled spheres.")
                    tree_mesh.active_scalars_name = 'radius'
                    try:
                        vis_radii = np.clip(radii_np_for_plot, min_plot_radius * 10, None)
                        spheres_geom = pv.PolyData(points_np) # Geom for glyph should just be points
                        spheres = pv.Sphere(radius=1.0).glyph(orient=False, scale=False, factor=1.0, geom=spheres_geom) 
                        spheres.points = spheres.points * vis_radii[:, np.newaxis] # Scale the glyphs
                        plotter.add_mesh(spheres, scalars=radii_np_for_plot, cmap=cmap_radius, scalar_bar_args={'title': 'Node Radius (mm)'})
                    except Exception as e_glyph: logger.error(f"Glyphing points failed: {e_glyph}. Plotting simple points.", exc_info=True); plotter.add_points(points_np, render_points_as_spheres=True, point_size=5, color='purple')
    plotter.camera_position = 'xy'
    if output_screenshot_path: plotter.show(auto_close=True, screenshot=output_screenshot_path); logger.info(f"Saved PyVista plot to {output_screenshot_path}")
    else: logger.info("Displaying PyVista plot."); plotter.show()


def generate_final_visualizations(
    config: dict, output_dir: str, tissue_data: dict, vascular_graph: Optional[nx.DiGraph],
    perfusion_map: Optional[np.ndarray] = None, pressure_map_tissue: Optional[np.ndarray] = None):
    logger.info("Generating final visualizations and quantitative analyses...")
    masks_to_plot: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if tissue_data.get('domain_mask') is not None and tissue_data.get('affine') is not None:
        masks_to_plot["domain_mask"] = (tissue_data['domain_mask'], tissue_data['affine'])
    seed_points_viz_data: List[Tuple[np.ndarray, float, str]] = []
    config_seeds = config_manager.get_param(config, "gbo_growth.seed_points", [])
    if config_seeds:
        for i, seed_info in enumerate(config_seeds):
            seed_points_viz_data.append((np.array(seed_info.get('position')), float(seed_info.get('initial_radius',0.1)), seed_info.get('id',f"S_{i}")))

    title_suffix = f" ({vascular_graph.number_of_nodes()} nodes, {vascular_graph.number_of_edges()} edges)" if vascular_graph else ""
    if vascular_graph is None or vascular_graph.number_of_nodes() == 0:
        logger.warning("Vascular graph empty for final visualization.")
        if (masks_to_plot or seed_points_viz_data) and PYVISTA_AVAILABLE:
            plot_vascular_tree_pyvista(graph=None, title="Tissue & Seeds (No Vasculature)" + title_suffix,
                                       output_screenshot_path=os.path.join(output_dir, "context_only_plot.png"), 
                                       tissue_masks=masks_to_plot, seed_points_world=seed_points_viz_data)
    else:
        io_utils.save_vascular_tree_vtp(vascular_graph, os.path.join(output_dir, "final_plot_vascular_tree.vtp"))
        if PYVISTA_AVAILABLE:
            plot_vascular_tree_pyvista(
                vascular_graph, title="Final Generated Vasculature" + title_suffix,
                output_screenshot_path=os.path.join(output_dir, "final_vascular_tree_with_context_3D.png"), 
                tissue_masks=masks_to_plot, seed_points_world=seed_points_viz_data,
                background_color=config_manager.get_param(config, "visualization.pyvista_background_color", "white"),
                cmap_radius=config_manager.get_param(config, "visualization.pyvista_cmap_radius", "plasma"),
                seed_point_radius_scale=config_manager.get_param(config, "visualization.seed_marker_radius_scale", 5.0))
    
    if vascular_graph and vascular_graph.number_of_nodes() > 0:
        analyze_radii_distribution(vascular_graph, output_dir)
        analyze_segment_lengths(vascular_graph, output_dir)
        analyze_murray_law(vascular_graph, output_dir, 
                           murray_exponent=config_manager.get_param(config, "vascular_properties.murray_law_exponent", 3.0))
        analyze_branching_angles(vascular_graph, output_dir)
    logger.info("Final visualizations and analyses generation complete.")