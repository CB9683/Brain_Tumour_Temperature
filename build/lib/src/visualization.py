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
    if graph is None or graph.number_of_nodes() == 0: 
        logger.warning("Radii analysis: Graph is empty or None.")
        return    
    radii = [data['radius'] for _, data in graph.nodes(data=True) 
             if 'radius' in data and data.get('is_synthetic', True)] 
    if not radii: logger.info("No synthetic nodes with radii found for distribution analysis."); return
    df_radii = pd.DataFrame(radii, columns=['radius_mm'])
    csv_path = os.path.join(output_dir, f"{filename_prefix}radii_data.csv")
    df_radii.to_csv(csv_path, index=False); logger.info(f"Saved raw radii data to {csv_path}")
    plot_path = os.path.join(output_dir, f"{filename_prefix}radii_distribution.png")
    _plot_histogram(radii, "Distribution of Vessel Radii (Synthetic Nodes)", "Radius (mm)", plot_path)

def analyze_segment_lengths(graph: nx.DiGraph, output_dir: str, filename_prefix: str = "final_"):
    if graph is None or graph.number_of_edges() == 0: 
        logger.warning("Segment length analysis: Graph is empty or has no edges.")
        return
    lengths = []
    for u, v, data in graph.edges(data=True):
        # Ensure nodes u and v exist before checking their attributes
        if u in graph.nodes and v in graph.nodes and 'length' in data and \
           (graph.nodes[u].get('is_synthetic', True) or graph.nodes[v].get('is_synthetic', True)):
            lengths.append(data['length'])
            
    if not lengths: logger.info("No synthetic segments with lengths found for distribution analysis."); return
    df_lengths = pd.DataFrame(lengths, columns=['length_mm'])
    csv_path = os.path.join(output_dir, f"{filename_prefix}segment_lengths_data.csv")
    df_lengths.to_csv(csv_path, index=False); logger.info(f"Saved raw segment length data to {csv_path}")
    plot_path = os.path.join(output_dir, f"{filename_prefix}segment_lengths_distribution.png")
    _plot_histogram(lengths, "Distribution of Segment Lengths", "Length (mm)", plot_path)

def analyze_murray_law(graph: nx.DiGraph, output_dir: str, murray_exponent: float = 3.0, filename_prefix: str = "final_"):
    if not MATPLOTLIB_AVAILABLE: logger.warning("Matplotlib not available. Skipping Murray's Law plot."); return
    if graph is None or graph.number_of_nodes() == 0: 
        logger.warning("Murray's Law analysis: Graph is empty or None.")
        return
        
    parent_powers: List[float] = []
    children_sum_powers: List[float] = []
    bifurcation_data_for_csv: List[Dict] = []
    
    logger.info(f"--- Starting Murray's Law Analysis (Exponent: {murray_exponent}) ---")
    bifurcation_nodes_processed = 0

    for node_id, data in graph.nodes(data=True):
        if data.get('type') == 'synthetic_bifurcation' and graph.out_degree(node_id) == 2:
            bifurcation_nodes_processed += 1
            children_ids = list(graph.successors(node_id))
            
            r_p_node = data.get('radius') 
            q_p_node = data.get('Q_flow') 

            if len(children_ids) != 2 or \
               children_ids[0] not in graph.nodes or \
               children_ids[1] not in graph.nodes:
                logger.warning(f"Murray Test: Bifurcation {node_id} has invalid children setup. Children: {children_ids}. Skipping.")
                continue

            data_c1 = graph.nodes[children_ids[0]]
            data_c2 = graph.nodes[children_ids[1]]
            r_c1_node = data_c1.get('radius')
            r_c2_node = data_c2.get('radius')
            q_c1_node = data_c1.get('Q_flow') 
            q_c2_node = data_c2.get('Q_flow')
            type_c1 = data_c1.get('type')
            type_c2 = data_c2.get('type')

            logger.debug(f"Murray Test - Bifurcation Candidate: {node_id} (Type: {data.get('type')})")
            logger.debug(f"  Parent {node_id}: R={r_p_node}, Q={q_p_node}") # Raw values before check
            logger.debug(f"  Child1 {children_ids[0]} (Type: {type_c1}): R={r_c1_node}, Q={q_c1_node}")
            logger.debug(f"  Child2 {children_ids[1]} (Type: {type_c2}): R={r_c2_node}, Q={q_c2_node}")

            if r_p_node is None or r_p_node < constants.EPSILON:
                logger.debug(f"  Skipping {node_id} for Murray: Parent radius ({r_p_node}) too small or None.")
                continue
            if r_c1_node is None or r_c1_node < constants.EPSILON or \
               r_c2_node is None or r_c2_node < constants.EPSILON:
                logger.debug(f"  Skipping {node_id} for Murray: Child1 R ({r_c1_node}) or Child2 R ({r_c2_node}) too small or None.")
                continue
                
            p_power = r_p_node**murray_exponent
            c_sum_power = r_c1_node**murray_exponent + r_c2_node**murray_exponent
            
            logger.debug(f"  Parent Power (r_p^{murray_exponent:.1f}): {p_power:.3e}, Children Sum Power (sum r_c^{murray_exponent:.1f}): {c_sum_power:.3e}")

            parent_powers.append(p_power); children_sum_powers.append(c_sum_power)
            bifurcation_data_for_csv.append({'b_id': node_id, f'rP^{murray_exponent:.1f}': p_power, 
                                             f'sum_rC^{murray_exponent:.1f}': c_sum_power, 
                                             'rP': r_p_node, 'rC1': r_c1_node, 'rC2': r_c2_node,
                                             'qP': q_p_node, 'qC1': q_c1_node, 'qC2': q_c2_node}) # Add flows
                                             
    logger.info(f"Murray's Law Analysis: Processed {bifurcation_nodes_processed} synthetic bifurcation nodes.")
    if not parent_powers: logger.info("No valid synthetic bifurcations passed filters for Murray's Law plot."); return
    
    df_murray = pd.DataFrame(bifurcation_data_for_csv)
    csv_path = os.path.join(output_dir, f"{filename_prefix}murray_law_data.csv")
    df_murray.to_csv(csv_path, index=False)
    logger.info(f"Saved Murray's Law raw data to {csv_path}")

    plt.figure(figsize=(7, 7))
    plt.scatter(children_sum_powers, parent_powers, alpha=0.6, edgecolors='k', s=40, label="Bifurcations")
    
    max_val_plot = 0.0
    if parent_powers and children_sum_powers:
        max_val_plot = max(max(parent_powers, default=0.0), max(children_sum_powers, default=0.0)) * 1.1
    if max_val_plot < constants.EPSILON : max_val_plot = 1.0 

    plt.plot([0, max_val_plot], [0, max_val_plot], 'r--', label=f'Ideal Murray (y=x, exp={murray_exponent:.1f})') 
    plt.xlabel(f"Sum of Children Radii^{murray_exponent:.1f} (r$_1^{murray_exponent:.1f}$ + r$_2^{murray_exponent:.1f}$)") 
    plt.ylabel(f"Parent Radius^{murray_exponent:.1f} (r$_0^{murray_exponent:.1f}$)")
    plt.title("Murray's Law Compliance Test (Synthetic Bifurcations)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal') 
    plt.xlim([0, max_val_plot])
    plt.ylim([0, max_val_plot])
    
    plot_path = os.path.join(output_dir, f"{filename_prefix}murray_law_compliance.png")
    try: plt.savefig(plot_path); logger.info(f"Saved Murray's Law plot to {plot_path}"); plt.close()
    except Exception as e: logger.error(f"Error saving Murray's Law plot: {e}"); plt.close()


def analyze_branching_angles(graph: nx.DiGraph, output_dir: str, filename_prefix: str = "final_"):
    if not MATPLOTLIB_AVAILABLE: logger.warning("Matplotlib not available. Skipping branching angle plot."); return
    if graph is None or graph.number_of_nodes() == 0: 
        logger.warning("Branching angle analysis: Graph is empty or None.")
        return
        
    branching_angles_c1_c2: List[float] = []; angle_data_for_csv: List[Dict] = []
    for node_id, data in graph.nodes(data=True):
        if data.get('type') == 'synthetic_bifurcation' and graph.out_degree(node_id) == 2:
            parent_pos = data.get('pos'); successors = list(graph.successors(node_id))
            if parent_pos is None or len(successors) != 2: continue
            
            if successors[0] not in graph.nodes or successors[1] not in graph.nodes: continue
            child1_pos = graph.nodes[successors[0]].get('pos'); child2_pos = graph.nodes[successors[1]].get('pos')
            
            if child1_pos is not None and child2_pos is not None:
                vec1 = child1_pos - parent_pos; vec2 = child2_pos - parent_pos
                norm_vec1 = np.linalg.norm(vec1); norm_vec2 = np.linalg.norm(vec2)
                if norm_vec1 > constants.EPSILON and norm_vec2 > constants.EPSILON:
                    cosine_angle = np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)
                    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
                    angle_deg = np.degrees(angle_rad)
                    branching_angles_c1_c2.append(angle_deg)
                    angle_data_for_csv.append({'b_id': node_id, 'angle_deg': angle_deg})
    if not branching_angles_c1_c2: logger.info("No valid bifurcations for branching angle analysis."); return
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
    seed_point_radius_scale: float = 5.0 # Default scale for seed markers
    ):
    if not PYVISTA_AVAILABLE: logger.warning("PyVista not available. Skipping 3D PyVista plot."); return
    
    plotter = pv.Plotter(off_screen=output_screenshot_path is not None, window_size=[1000,800])
    plotter.background_color = background_color
    plotter.add_title(title, font_size=16)

    spacing_for_seeds = np.array([1.0, 1.0, 1.0]) # Default if no masks

    if tissue_masks:
        mask_colors = {"GM": "lightblue", "WM": "lightyellow", "domain_mask": domain_outline_color, "Tumor": "lightcoral"}
        mask_opacities = {"GM": 0.2, "WM": 0.2, "domain_mask": domain_outline_opacity, "Tumor": 0.3}
        for mask_name, mask_data_tuple in tissue_masks.items():
            if not isinstance(mask_data_tuple, tuple) or len(mask_data_tuple) != 2:
                logger.error(f"Invalid format for tissue_masks entry '{mask_name}'. Skipping.")
                continue
            mask_data, affine = mask_data_tuple
            if mask_data is None or not np.any(mask_data) or affine is None: 
                logger.debug(f"Mask '{mask_name}' empty, None, or affine missing. Skipping.")
                continue
            
            if mask_name == "GM": 
                logger.debug(f"Plotting GM: Shape={mask_data.shape}, Affine=\n{affine}, Sum={np.sum(mask_data)}")
                # temp_gm_path = os.path.join(os.path.dirname(output_screenshot_path) if output_screenshot_path else ".", "debug_gm_to_plot.nii.gz")
                # io_utils.save_nifti_image(mask_data.astype(np.uint8), affine, temp_gm_path)

            try:
                dims = np.array(mask_data.shape); current_spacing = np.abs(np.diag(affine)[:3]); origin = affine[:3, 3] 
                if mask_name == "domain_mask": spacing_for_seeds = current_spacing # Use domain mask spacing for seed scaling

                grid = pv.ImageData(dimensions=dims, spacing=current_spacing, origin=origin)
                grid.point_data[mask_name] = mask_data.flatten(order="F").astype(float)
                contour = grid.contour([0.5], scalars=mask_name, rng=[0,1])
                if contour.n_points > 0: 
                    plotter.add_mesh(contour, color=mask_colors.get(mask_name, "grey"), 
                                     opacity=mask_opacities.get(mask_name, 0.1), style='surface')
                else: logger.debug(f"No contour for mask '{mask_name}'.")
            except Exception as e_mask: logger.error(f"Error plotting mask '{mask_name}': {e_mask}", exc_info=True)
                
    if seed_points_world:
        for seed_pos, seed_initial_radius, seed_name in seed_points_world:
            try:
                # Scale marker based on a fraction of average voxel dim or fixed small value
                marker_base_size = np.mean(spacing_for_seeds) * 0.5 
                visual_marker_radius = max(marker_base_size * seed_point_radius_scale, 
                                           seed_initial_radius * 2.0, # At least 2x actual radius
                                           0.05) # Absolute minimum display size
                sphere = pv.Sphere(center=seed_pos, radius=visual_marker_radius)
                plotter.add_mesh(sphere, color=seed_point_color, opacity=0.8)
            except Exception as e_seed: logger.error(f"Error plotting seed '{seed_name}': {e_seed}", exc_info=True)

    if graph is not None and graph.number_of_nodes() > 0:
        points, radii_for_plot, lines, node_to_idx, idx_counter = [], [], [], {}, 0
        min_plot_radius = max(constants.MIN_VESSEL_RADIUS_MM * 0.1, 1e-5) 
        for node_id, data in graph.nodes(data=True):
            if 'pos' in data and 'radius' in data:
                points.append(data['pos']); radii_for_plot.append(max(data['radius'], min_plot_radius))
                node_to_idx[node_id] = idx_counter; idx_counter += 1
        if not points: logger.info("No valid nodes with pos/radius in graph for tree plotting.")
        else:
            points_np, radii_np_for_plot = np.array(points), np.array(radii_for_plot)
            for u, v, _ in graph.edges(data=True):
                if u in node_to_idx and v in node_to_idx: lines.extend([2, node_to_idx[u], node_to_idx[v]])
            
            tree_mesh = pv.PolyData()
            if points_np.shape[0] > 0:
                tree_mesh.points = points_np
                tree_mesh.point_data['radius'] = radii_np_for_plot
            else: logger.warning("No points to create tree_mesh from.")

            if lines and tree_mesh.n_points > 0 and tree_mesh.n_points == len(radii_np_for_plot): 
                tree_mesh.lines = np.array(lines)
                if tree_mesh.n_cells > 0: 
                    try: plotter.add_mesh(tree_mesh, scalars='radius', line_width=3, cmap=cmap_radius, 
                                          render_lines_as_tubes=True, scalar_bar_args={'title': 'Radius (mm)', 'color':'black'})
                    except Exception: plotter.add_mesh(tree_mesh, scalars='radius', line_width=1.5, cmap=cmap_radius, scalar_bar_args={'title': 'Radius (mm)', 'color':'black'})
                elif tree_mesh.n_points > 0: 
                    logger.info("Graph has points but no line cells. Plotting as scaled spheres.")
                    if 'radius' in tree_mesh.point_data:
                        tree_mesh.active_scalars_name = 'radius'
                        try:
                            glyph_geom = pv.PolyData(points_np)
                            spheres = pv.Sphere(radius=1.0).glyph(orient=False, scale='radius', factor=0.05, geom=glyph_geom) 
                            plotter.add_mesh(spheres, scalars='radius', cmap=cmap_radius, scalar_bar_args={'title': 'Node Radius (mm)'})
                        except Exception as e_glyph: logger.error(f"Glyphing points failed: {e_glyph}. Plotting simple points.", exc_info=True); plotter.add_points(points_np, render_points_as_spheres=True, point_size=3, color='purple')
            elif tree_mesh.n_points > 0: # Only points, no lines
                 logger.info("Graph has points but no lines. Plotting as scaled spheres.")
                 if 'radius' in tree_mesh.point_data:
                    tree_mesh.active_scalars_name = 'radius'
                    try:
                        glyph_geom = pv.PolyData(points_np)
                        spheres = pv.Sphere(radius=1.0).glyph(orient=False, scale='radius', factor=0.05, geom=glyph_geom)
                        plotter.add_mesh(spheres, scalars='radius', cmap=cmap_radius, scalar_bar_args={'title': 'Node Radius (mm)'})
                    except Exception as e_glyph: logger.error(f"Glyphing points failed: {e_glyph}. Plotting simple points.", exc_info=True); plotter.add_points(points_np, render_points_as_spheres=True, point_size=3, color='purple')
                 else: plotter.add_points(points_np, render_points_as_spheres=True, point_size=3, color='purple')
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
    if tissue_data.get('GM') is not None and tissue_data.get('affine') is not None:
        masks_to_plot["GM"] = (tissue_data['GM'], tissue_data['affine'])
        logger.info("Added GM mask for plotting in generate_final_visualizations.")
    else: logger.info("GM mask not found in tissue_data for generate_final_visualizations.")

    seed_points_viz_data: List[Tuple[np.ndarray, float, str]] = []
    config_seeds = config_manager.get_param(config, "gbo_growth.seed_points", [])
    if config_seeds:
        for i, seed_info in enumerate(config_seeds):
            seed_points_viz_data.append((np.array(seed_info.get('position')), 
                                         float(seed_info.get('initial_radius',0.1)), 
                                         seed_info.get('id',f"S_{i}")))

    title_suffix = f" ({vascular_graph.number_of_nodes()} nodes, {vascular_graph.number_of_edges()} edges)" if vascular_graph else ""
    
    pv_plot_title = "Final Generated Vasculature" + title_suffix if vascular_graph and vascular_graph.number_of_nodes() > 0 else "Tissue & Seeds (No/Empty Vasculature)"
    pv_screenshot_filename = "final_vascular_tree_with_context_3D.png" if vascular_graph and vascular_graph.number_of_nodes() > 0 else "context_only_plot.png"
    pv_screenshot_path = os.path.join(output_dir, pv_screenshot_filename)

    if PYVISTA_AVAILABLE:
        plot_vascular_tree_pyvista(
            graph=vascular_graph, title=pv_plot_title,
            output_screenshot_path=pv_screenshot_path, 
            tissue_masks=masks_to_plot, seed_points_world=seed_points_viz_data,
            background_color=config_manager.get_param(config, "visualization.pyvista_background_color", "white"),
            cmap_radius=config_manager.get_param(config, "visualization.pyvista_cmap_radius", "plasma"),
            seed_point_radius_scale=config_manager.get_param(config, "visualization.seed_marker_radius_scale", 5.0),
            domain_outline_color=config_manager.get_param(config, "visualization.domain_mask_color", "lightgray"),
            domain_outline_opacity=config_manager.get_param(config, "visualization.domain_mask_opacity", 0.1)
        )
    else:
        logger.warning("PyVista not available, skipping 3D plot generation.")
    
    if vascular_graph and vascular_graph.number_of_nodes() > 0:
        # Save VTP (moved here to ensure it's only saved if graph is valid)
        final_tree_vtp_path = os.path.join(output_dir, "final_plot_vascular_tree.vtp")
        io_utils.save_vascular_tree_vtp(vascular_graph, final_tree_vtp_path)
        logger.info(f"Final vascular tree saved for analysis: {final_tree_vtp_path}")

        analyze_radii_distribution(vascular_graph, output_dir)
        analyze_segment_lengths(vascular_graph, output_dir)
        analyze_murray_law(vascular_graph, output_dir, 
                           murray_exponent=config_manager.get_param(config, "vascular_properties.murray_law_exponent", 3.0))
        analyze_branching_angles(vascular_graph, output_dir)
    else:
        logger.warning("Vascular graph is empty. Skipping VTP save and quantitative analyses.")
        
    logger.info("Final visualizations and analyses generation complete.")