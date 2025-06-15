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
def _plot_histogram(data: List[float], title: str, xlabel: str, output_path: str, bins: int = 30, density: bool = False):
    if not MATPLOTLIB_AVAILABLE:
        logger.warning(f"Matplotlib not available. Skipping histogram plot: {title}")
        return
    if not data:
        logger.warning(f"No data to plot for histogram: {title}")
        return
    valid_data = [x for x in data if np.isfinite(x)]
    if not valid_data:
        logger.warning(f"No finite data to plot for histogram (all NaN/Inf): {title}")
        return

    plt.figure(figsize=(8, 6))
    plt.hist(valid_data, bins=bins, color='skyblue', edgecolor='black', density=density)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency" if not density else "Density")
    plt.grid(axis='y', alpha=0.75)
    try:
        plt.savefig(output_path)
        logger.info(f"Saved histogram '{title}' to {output_path}")
    except Exception as e:
        logger.error(f"Error saving histogram '{title}' to {output_path}: {e}")
    finally:
        plt.close()

# --- Functions for Quantitative Analysis ---
def analyze_radii_distribution(graph: nx.DiGraph, output_dir: str, filename_prefix: str = "final_"):
    if graph is None or graph.number_of_nodes() == 0:
        logger.warning("Radii analysis: Graph is empty or None.")
        return
    radii = [data['radius'] for _, data in graph.nodes(data=True)
             if 'radius' in data and data.get('is_synthetic', True) and np.isfinite(data['radius'])]
    if not radii: logger.info("No synthetic nodes with valid radii found for distribution analysis."); return
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
        if u in graph.nodes and v in graph.nodes and 'length' in data and \
           (graph.nodes[u].get('is_synthetic', True) or graph.nodes[v].get('is_synthetic', True)) and \
           np.isfinite(data['length']):
            lengths.append(data['length'])

    if not lengths: logger.info("No synthetic segments with valid lengths found for distribution analysis."); return
    df_lengths = pd.DataFrame(lengths, columns=['length_mm'])
    csv_path = os.path.join(output_dir, f"{filename_prefix}segment_lengths_data.csv")
    df_lengths.to_csv(csv_path, index=False); logger.info(f"Saved raw segment length data to {csv_path}")
    plot_path = os.path.join(output_dir, f"{filename_prefix}segment_lengths_distribution.png")
    _plot_histogram(lengths, "Distribution of Segment Lengths", "Length (mm)", plot_path)


def analyze_bifurcation_geometry(graph: nx.DiGraph, output_dir: str, murray_exponent: float = 3.0, filename_prefix: str = "final_"):
    if graph is None or graph.number_of_nodes() == 0:
        logger.warning("Bifurcation geometry analysis: Graph is empty or None.")
        return

    murray_parent_powers: List[float] = []
    murray_children_sum_powers: List[float] = []
    area_ratios_alpha: List[float] = []
    daughter_asymmetry_ratios: List[float] = []
    branching_angles_c1_c2: List[float] = []
    bifurcation_data_for_csv: List[Dict] = []

    logger.info(f"--- Starting Bifurcation Geometry Analysis (Murray Exp: {murray_exponent}) ---")
    bifurcation_nodes_processed = 0

    for node_id, data in graph.nodes(data=True):
        node_type = data.get('type', '')
        is_potential_bifurcation = (node_type == 'synthetic_bifurcation') or \
                                   (node_type == 'synthetic_root_terminal' and data.get('is_flow_root', False))

        if is_potential_bifurcation and graph.out_degree(node_id) == 2:
            bifurcation_nodes_processed += 1
            parent_pos = data.get('pos')
            children_ids = list(graph.successors(node_id))

            r_p_node = data.get('radius')
            q_p_node = data.get('Q_flow') 

            if len(children_ids) != 2 or \
               children_ids[0] not in graph.nodes or \
               children_ids[1] not in graph.nodes:
                logger.debug(f"Bif. Geom. Test: Node {node_id} has invalid children setup. Skipping.")
                continue

            data_c1 = graph.nodes[children_ids[0]]
            data_c2 = graph.nodes[children_ids[1]]
            child1_pos = data_c1.get('pos')
            child2_pos = data_c2.get('pos')
            r_c1_node = data_c1.get('radius')
            r_c2_node = data_c2.get('radius')
            q_c1_node = data_c1.get('Q_flow')
            q_c2_node = data_c2.get('Q_flow')

            if not (r_p_node and np.isfinite(r_p_node) and r_p_node >= constants.EPSILON and
                    r_c1_node and np.isfinite(r_c1_node) and r_c1_node >= constants.EPSILON and
                    r_c2_node and np.isfinite(r_c2_node) and r_c2_node >= constants.EPSILON):
                logger.debug(f"Bif. Geom. Test: Node {node_id} or children have invalid radii. Skipping.")
                continue

            p_power = r_p_node**murray_exponent
            c_sum_power = r_c1_node**murray_exponent + r_c2_node**murray_exponent
            murray_parent_powers.append(p_power)
            murray_children_sum_powers.append(c_sum_power)

            alpha = (r_c1_node**2 + r_c2_node**2) / (r_p_node**2)
            area_ratios_alpha.append(alpha)

            asymmetry_ratio = min(r_c1_node, r_c2_node) / max(r_c1_node, r_c2_node) if max(r_c1_node, r_c2_node) > constants.EPSILON else 1.0
            daughter_asymmetry_ratios.append(asymmetry_ratio)
            
            angle_deg_val = np.nan # Initialize for CSV
            if parent_pos is not None and child1_pos is not None and child2_pos is not None:
                vec1 = child1_pos - parent_pos
                vec2 = child2_pos - parent_pos
                norm_vec1 = np.linalg.norm(vec1)
                norm_vec2 = np.linalg.norm(vec2)
                if norm_vec1 > constants.EPSILON and norm_vec2 > constants.EPSILON:
                    cosine_angle = np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)
                    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
                    angle_deg_val = np.degrees(angle_rad)
                    if np.isfinite(angle_deg_val):
                        branching_angles_c1_c2.append(angle_deg_val)

            bifurcation_data_for_csv.append({
                'b_id': node_id,
                f'rP^{murray_exponent:.1f}': p_power, f'sum_rC^{murray_exponent:.1f}': c_sum_power,
                'area_ratio_alpha': alpha, 'daughter_asymmetry_ratio': asymmetry_ratio,
                'angle_deg': angle_deg_val,
                'rP': r_p_node, 'rC1': r_c1_node, 'rC2': r_c2_node,
                'qP_graph': q_p_node, 'qC1_graph': q_c1_node, 'qC2_graph': q_c2_node
            })

    logger.info(f"Bifurcation Geometry Analysis: Processed {bifurcation_nodes_processed} potential bifurcation nodes.")

    if bifurcation_data_for_csv:
        df_bif_geom = pd.DataFrame(bifurcation_data_for_csv)
        csv_path = os.path.join(output_dir, f"{filename_prefix}bifurcation_geometry_data.csv")
        df_bif_geom.to_csv(csv_path, index=False)
        logger.info(f"Saved bifurcation geometry data to {csv_path}")
    else:
        logger.info("No valid bifurcation data collected for CSV. Skipping plots.")
        return

    if MATPLOTLIB_AVAILABLE and murray_parent_powers:
        plt.figure(figsize=(7, 7))
        plt.scatter(murray_children_sum_powers, murray_parent_powers, alpha=0.6, edgecolors='k', s=40, label="Bifurcations")
        max_val_plot = 0.0
        valid_parent_powers = [p for p in murray_parent_powers if np.isfinite(p)]
        valid_children_sum_powers = [c for c in murray_children_sum_powers if np.isfinite(c)]
        if valid_parent_powers and valid_children_sum_powers:
             max_val_plot = max(max(valid_parent_powers, default=0.0), max(valid_children_sum_powers, default=0.0)) * 1.1
        if max_val_plot < constants.EPSILON : max_val_plot = 1.0

        plt.plot([0, max_val_plot], [0, max_val_plot], 'r--', label=f'Ideal Murray (y=x, exp={murray_exponent:.1f})')
        plt.xlabel(f"Sum of Children Radii^{murray_exponent:.1f} (r$_1^{murray_exponent:.1f}$ + r$_2^{murray_exponent:.1f}$)")
        plt.ylabel(f"Parent Radius^{murray_exponent:.1f} (r$_0^{murray_exponent:.1f}$)")
        plt.title("Murray's Law Compliance Test")
        plt.legend()
        plt.grid(True)
        plt.xlim([0, max_val_plot]); plt.ylim([0, max_val_plot])
        plot_path = os.path.join(output_dir, f"{filename_prefix}murray_law_compliance.png")
        try: plt.savefig(plot_path); logger.info(f"Saved Murray's Law plot to {plot_path}");
        except Exception as e: logger.error(f"Error saving Murray's Law plot: {e}")
        finally: plt.close()
    else: logger.info("Skipping Murray's Law plot (Matplotlib unavailable or no data).")

    _plot_histogram(area_ratios_alpha, "Distribution of Bifurcation Area Ratios (α)", "Area Ratio α = (r₁²+r₂²)/r₀²",
                    os.path.join(output_dir, f"{filename_prefix}area_ratios_distribution.png"), bins=20)
    _plot_histogram(daughter_asymmetry_ratios, "Distribution of Daughter Radius Asymmetry Ratios", "Asymmetry Ratio min(r₁,r₂)/max(r₁,r₂)",
                    os.path.join(output_dir, f"{filename_prefix}daughter_asymmetry_distribution.png"), bins=20)
    _plot_histogram(branching_angles_c1_c2, "Distribution of Branching Angles (Child-Child)", "Angle (degrees)",
                    os.path.join(output_dir, f"{filename_prefix}branching_angles_distribution.png"), bins=18)


def analyze_degree_distribution(graph: nx.DiGraph, output_dir: str, filename_prefix: str = "final_"):
    if graph is None or graph.number_of_nodes() == 0:
        logger.warning("Degree distribution analysis: Graph is empty or None.")
        return
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available. Skipping degree distribution plots.")
        return

    degrees = [d for n, d in graph.degree()]
    in_degrees = [d for n, d in graph.in_degree()]
    out_degrees = [d for n, d in graph.out_degree()]

    df_degrees = pd.DataFrame({
        'node_id': list(graph.nodes()),
        'total_degree': degrees,
        'in_degree': in_degrees,
        'out_degree': out_degrees
    })
    csv_path = os.path.join(output_dir, f"{filename_prefix}degree_data.csv")
    df_degrees.to_csv(csv_path, index=False)
    logger.info(f"Saved node degree data to {csv_path}")

    max_bins_degree = 10 # Default max bins for degree plots
    if degrees: max_bins_degree = max(1, max(degrees))
    _plot_histogram(degrees, "Total Node Degree Distribution", "Degree",
                    os.path.join(output_dir, f"{filename_prefix}total_degree_distribution.png"),
                    bins=min(max_bins_degree, 50)) # Cap bins for very high degrees
    if in_degrees: max_bins_degree = max(1, max(in_degrees))
    _plot_histogram(in_degrees, "Node In-Degree Distribution", "In-Degree",
                    os.path.join(output_dir, f"{filename_prefix}in_degree_distribution.png"),
                    bins=min(max_bins_degree, 50))
    if out_degrees: max_bins_degree = max(1, max(out_degrees))
    _plot_histogram(out_degrees, "Node Out-Degree Distribution", "Out-Degree",
                    os.path.join(output_dir, f"{filename_prefix}out_degree_distribution.png"),
                    bins=min(max_bins_degree, 50))

def analyze_network_connectivity(graph: nx.DiGraph, output_dir: str, filename_prefix: str = "final_"):
    if graph is None or graph.number_of_nodes() == 0:
        logger.warning("Network connectivity analysis: Graph is empty or None.")
        return
    undirected_graph = graph.to_undirected()
    num_components = nx.number_connected_components(undirected_graph)
    logger.info(f"Network Connectivity: Number of connected components (undirected) = {num_components}")
    if num_components > 1:
        logger.warning(f"Network has {num_components} disconnected components. Expected 1 for a fully connected structure from seeds.")

def analyze_volumetric_densities(graph: nx.DiGraph, tissue_data: dict, output_dir: str, filename_prefix: str = "final_"):
    if graph is None or graph.number_of_nodes() == 0:
        logger.warning("Volumetric density analysis: Graph is empty or None.")
        return
    if 'domain_mask' not in tissue_data or 'voxel_volume' not in tissue_data:
        logger.warning("Volumetric density analysis: Missing 'domain_mask' or 'voxel_volume' in tissue_data.")
        return

    domain_mask = tissue_data['domain_mask']
    voxel_volume = tissue_data['voxel_volume'] 

    if domain_mask is None or voxel_volume <= 0:
        logger.warning("Volumetric density analysis: Invalid domain_mask or voxel_volume.")
        return

    domain_volume_mm3 = np.sum(domain_mask) * voxel_volume
    if domain_volume_mm3 < constants.EPSILON:
        logger.warning("Volumetric density analysis: Domain volume is zero. Cannot calculate densities.")
        return

    total_vessel_length_mm = sum(data['length'] for u, v, data in graph.edges(data=True) if 'length' in data and np.isfinite(data['length']))
    num_bifurcation_nodes = sum(1 for node_id, data in graph.nodes(data=True) if data.get('type') == 'synthetic_bifurcation')

    vessel_length_density_mm_per_mm3 = total_vessel_length_mm / domain_volume_mm3
    branchpoint_density_per_mm3 = num_bifurcation_nodes / domain_volume_mm3

    logger.info(f"--- Volumetric Densities (Domain Volume: {domain_volume_mm3:.2e} mm^3) ---")
    logger.info(f"  Total vessel length: {total_vessel_length_mm:.2e} mm")
    logger.info(f"  Vessel length density: {vessel_length_density_mm_per_mm3:.2e} mm/mm^3")
    logger.info(f"  Number of bifurcation points: {num_bifurcation_nodes}")
    logger.info(f"  Branchpoint density: {branchpoint_density_per_mm3:.2e} #/mm^3")

    density_data = {
        'domain_volume_mm3': domain_volume_mm3,
        'total_vessel_length_mm': total_vessel_length_mm,
        'vessel_length_density_mm_per_mm3': vessel_length_density_mm_per_mm3,
        'num_bifurcation_nodes': num_bifurcation_nodes,
        'branchpoint_density_per_mm3': branchpoint_density_per_mm3
    }
    df_density = pd.DataFrame([density_data])
    csv_path = os.path.join(output_dir, f"{filename_prefix}volumetric_density_data.csv")
    df_density.to_csv(csv_path, index=False)
    logger.info(f"Saved volumetric density data to {csv_path}")


def plot_vascular_tree_pyvista(
    graph: Optional[nx.DiGraph],
    title: str = "Vascular Tree",
    background_color: str = "white",
    cmap_radius: str = "viridis",
    output_screenshot_path: Optional[str] = None,
    tissue_masks: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None, # Modified for testing
    seed_points_world: Optional[List[Tuple[np.ndarray, float, str]]] = None,
    color_by_scalar: Optional[str] = 'radius',
    scalar_bar_title_override: Optional[str] = None,
    custom_cmap: Optional[str] = None
    ):
    if not PYVISTA_AVAILABLE: logger.warning("PyVista not available. Skipping 3D PyVista plot."); return

    plotter = pv.Plotter(off_screen=output_screenshot_path is not None, window_size=[1200,900])
    plotter.background_color = background_color
    plotter.add_title(title, font_size=16)

    spacing_for_seeds = np.array([1.0, 1.0, 1.0])

    if tissue_masks: # tissue_masks will be None or an empty dict for this test
        mask_colors_default = {"GM": "lightblue", "WM": "lightyellow", "domain_mask": "lightgray", "Tumor": "lightcoral"}
        mask_opacities_default = {"GM": 0.2, "WM": 0.2, "domain_mask": 0.1, "Tumor": 0.3}
        # These config gets will use defaults if config is None
        domain_outline_color = config_manager.get_param(None, "visualization.domain_mask_color", "lightgray")
        domain_outline_opacity = config_manager.get_param(None, "visualization.domain_mask_opacity", 0.1)

        for mask_name, mask_data_tuple in tissue_masks.items():
            if not isinstance(mask_data_tuple, tuple) or len(mask_data_tuple) != 2: continue
            mask_data, affine = mask_data_tuple
            if mask_data is None or not np.any(mask_data) or affine is None: continue

            logger.info(f"Processing mask '{mask_name}' for PyVista: Shape={mask_data.shape}, Sum={np.sum(mask_data)}")
            logger.debug(f"Mask '{mask_name}' Affine:\n{affine}")
            try:
                dims = np.array(mask_data.shape)
                current_spacing = np.abs(np.diag(affine)[:3])
                origin = affine[:3, 3]
                logger.debug(f"Mask '{mask_name}': Dims={dims}, Spacing={current_spacing}, Origin={origin}")
                if mask_name == "domain_mask": spacing_for_seeds = current_spacing

                grid = pv.ImageData(dimensions=dims, spacing=current_spacing, origin=origin)
                grid.point_data[mask_name] = mask_data.flatten(order="F").astype(float)
                contour = grid.contour([0.5], scalars=mask_name, rng=[0,1])

                if contour.n_points > 0:
                    logger.info(f"Mask '{mask_name}' contour generated: {contour.n_points} points. Bounds: {contour.bounds}")
                    if mask_name == "domain_mask" and output_screenshot_path:
                        debug_contour_path = os.path.join(os.path.dirname(output_screenshot_path), f"debug_{mask_name}_contour.vtk")
                        try: contour.save(debug_contour_path); logger.info(f"Saved debug contour for '{mask_name}' to {debug_contour_path}")
                        except Exception as e_save_contour: logger.error(f"Could not save debug contour for '{mask_name}': {e_save_contour}")

                    current_color = mask_colors_default.get(mask_name, "grey")
                    current_opacity = mask_opacities_default.get(mask_name, 0.1)
                    if mask_name == "domain_mask":
                        current_color = domain_outline_color
                        current_opacity = domain_outline_opacity
                    plotter.add_mesh(contour, color=current_color, opacity=current_opacity, style='surface')
                else: logger.warning(f"No contour generated for mask '{mask_name}'.")
            except Exception as e_mask: logger.error(f"Error plotting mask '{mask_name}': {e_mask}", exc_info=True)
    else:
        logger.info("No tissue masks provided to plot_vascular_tree_pyvista. Skipping mask rendering.")


    if seed_points_world:
        seed_point_color = config_manager.get_param(None, "visualization.seed_point_color", "red")
        seed_point_radius_scale = config_manager.get_param(None, "visualization.seed_marker_radius_scale", 5.0)
        for seed_pos, seed_initial_radius, seed_name in seed_points_world:
            try:
                marker_base_size = np.mean(spacing_for_seeds) * 0.5
                visual_marker_radius = max(marker_base_size * seed_point_radius_scale,
                                           seed_initial_radius * 2.0,
                                           0.05 * np.mean(spacing_for_seeds if np.any(spacing_for_seeds > 0) else np.array([1.0])))
                logger.debug(f"Plotting seed '{seed_name}' at {seed_pos} with display radius {visual_marker_radius:.3f}")
                sphere = pv.Sphere(center=seed_pos, radius=visual_marker_radius)
                plotter.add_mesh(sphere, color=seed_point_color, opacity=0.8)
            except Exception as e_seed: logger.error(f"Error plotting seed '{seed_name}': {e_seed}", exc_info=True)

    tree_mesh_bounds_logged = False
    if graph is not None and graph.number_of_nodes() > 0:
        points, lines, node_to_idx, idx_counter = [], [], {}, 0
        point_data_arrays: Dict[str, List[float]] = {'radius': [], 'pressure': []}
        edge_data_arrays: Dict[str, List[float]] = {'flow_solver': []}

        min_voxel_dim = np.min(spacing_for_seeds) if np.any(spacing_for_seeds > 0) else 0.01
        min_plot_radius = max(constants.MIN_VESSEL_RADIUS_MM * 0.1, min_voxel_dim * 0.05, 1e-4)

        for node_id, data in graph.nodes(data=True):
            if 'pos' in data and np.all(np.isfinite(data['pos'])):
                points.append(data['pos'])
                point_data_arrays['radius'].append(max(data.get('radius', min_plot_radius), min_plot_radius) if np.isfinite(data.get('radius', min_plot_radius)) else min_plot_radius)
                point_data_arrays['pressure'].append(data.get('pressure', np.nan))
                node_to_idx[node_id] = idx_counter
                idx_counter += 1

        if not points: logger.warning("No valid nodes with positions in graph for tree plotting.")
        else:
            points_np = np.array(points)
            for u, v, data in graph.edges(data=True):
                if u in node_to_idx and v in node_to_idx:
                    lines.extend([2, node_to_idx[u], node_to_idx[v]])
                    edge_data_arrays['flow_solver'].append(data.get('flow_solver', np.nan))

            tree_mesh = pv.PolyData()
            if points_np.shape[0] > 0:
                tree_mesh.points = points_np
                for key, arr in point_data_arrays.items():
                    if arr: tree_mesh.point_data[key] = np.array(arr)
                logger.info(f"Vascular tree mesh generated: {tree_mesh.n_points} points. Bounds: {tree_mesh.bounds}")
                tree_mesh_bounds_logged = True

                if lines and tree_mesh.n_points > 0:
                    tree_mesh.lines = np.array(lines)
                    if tree_mesh.n_cells > 0:
                        for key, arr in edge_data_arrays.items():
                             if arr and len(arr) == tree_mesh.n_cells: tree_mesh.cell_data[key] = np.array(arr)
                             elif arr: logger.warning(f"Mismatch in edge data '{key}' length ({len(arr)}) and n_cells ({tree_mesh.n_cells}). Skipping.")

                        active_scalars_on_points = color_by_scalar in tree_mesh.point_data and np.any(np.isfinite(tree_mesh.point_data[color_by_scalar]))
                        active_scalars_on_cells = color_by_scalar in tree_mesh.cell_data and np.any(np.isfinite(tree_mesh.cell_data[color_by_scalar]))


                        sargs = {'title': scalar_bar_title_override if scalar_bar_title_override else color_by_scalar.replace("_"," ").title(),
                                 'color':'black', 'vertical':True, 'position_y': 0.05, 'position_x': 0.85, 'height': 0.3, 'n_labels': 5}

                        current_cmap = custom_cmap if custom_cmap else (cmap_radius if color_by_scalar == 'radius' else 'coolwarm')
                        
                        preference = 'point' if active_scalars_on_points else ('cell' if active_scalars_on_cells else None)

                        if preference:
                            logger.info(f"Plotting tree colored by '{color_by_scalar}' using cmap '{current_cmap}'. Scalar preference: '{preference}'.")
                            plotter.add_mesh(tree_mesh, scalars=color_by_scalar, line_width=5, cmap=current_cmap,
                                             render_lines_as_tubes=True, scalar_bar_args=sargs,
                                             preference=preference)
                        else:
                            logger.warning(f"Scalar '{color_by_scalar}' not found or contains no finite values in point or cell data. Plotting with default radius coloring.")
                            if 'radius' in tree_mesh.point_data and np.any(np.isfinite(tree_mesh.point_data['radius'])):
                                plotter.add_mesh(tree_mesh, scalars='radius', line_width=5, cmap=cmap_radius,
                                                 render_lines_as_tubes=True, scalar_bar_args={'title': 'Radius (mm)', **sargs})
                            else: # Absolute fallback: no color, just structure
                                logger.warning("No valid 'radius' data either. Plotting tree structure with default color.")
                                plotter.add_mesh(tree_mesh, color="gray", line_width=3, render_lines_as_tubes=True)

                    elif tree_mesh.n_points > 0: # Points exist, but no lines were formed
                         logger.info("Graph has points but no line cells. Plotting as scaled spheres.")
                         if 'radius' in tree_mesh.point_data and np.any(np.isfinite(tree_mesh.point_data['radius'])):
                            tree_mesh.active_scalars_name = 'radius'
                            try:
                                glyph_factor = 0.1
                                spheres = tree_mesh.glyph(orient=False, scale='radius', factor=glyph_factor)
                                plotter.add_mesh(spheres, scalars='radius', cmap=cmap_radius,
                                                 scalar_bar_args={'title': 'Node Radius (mm)', 'color':'black', 'vertical':True, 'position_y': 0.05, 'position_x': 0.85, 'height': 0.3, 'n_labels': 5})
                            except Exception as e_glyph:
                                logger.error(f"Glyphing points failed: {e_glyph}. Plotting simple points.", exc_info=True)
                                plotter.add_points(points_np, render_points_as_spheres=True, point_size=5, color='purple')
                         else: plotter.add_points(points_np, render_points_as_spheres=True, point_size=5, color='purple')

            else: logger.warning("No points to create tree_mesh from for PyVista.")
    if not tree_mesh_bounds_logged:
         logger.info("No vascular graph provided or graph was empty. Only plotting context (masks/seeds).")


    plotter.camera_position = 'iso'
    plotter.enable_parallel_projection()
    plotter.add_axes(interactive=True)

    if output_screenshot_path:
        plotter.show(auto_close=True, screenshot=output_screenshot_path)
        logger.info(f"Saved PyVista plot to {output_screenshot_path}")
    else:
        logger.info("Displaying PyVista plot. Close window to continue.")
        plotter.show()


def generate_final_visualizations(
    config: dict, output_dir: str, tissue_data: dict, vascular_graph: Optional[nx.DiGraph],
    perfusion_map: Optional[np.ndarray] = None, pressure_map_tissue: Optional[np.ndarray] = None,
    plot_context_masks: bool = True # New argument for controlling mask plotting
    ):
    logger.info("Generating final visualizations and quantitative analyses...")
    
    masks_to_plot_for_pyvista: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]
    if plot_context_masks:
        masks_to_plot_for_pyvista = {}
        if tissue_data.get('domain_mask') is not None and tissue_data.get('affine') is not None:
            masks_to_plot_for_pyvista["domain_mask"] = (tissue_data['domain_mask'], tissue_data['affine'])
        if tissue_data.get('GM') is not None and tissue_data.get('affine') is not None:
            masks_to_plot_for_pyvista["GM"] = (tissue_data['GM'], tissue_data['affine'])
        if tissue_data.get('WM') is not None and tissue_data.get('affine') is not None:
            masks_to_plot_for_pyvista["WM"] = (tissue_data['WM'], tissue_data['affine'])
        if tissue_data.get('Tumor') is not None and tissue_data.get('affine') is not None:
            masks_to_plot_for_pyvista["Tumor"] = (tissue_data['Tumor'], tissue_data['affine'])
        if not masks_to_plot_for_pyvista: # If dict is still empty
            logger.info("No valid masks found in tissue_data to plot for context.")
            masks_to_plot_for_pyvista = None # Explicitly set to None
    else:
        logger.info("Context mask plotting is disabled for this visualization call.")
        masks_to_plot_for_pyvista = None


    seed_points_viz_data: List[Tuple[np.ndarray, float, str]] = []
    config_seeds = config_manager.get_param(config, "gbo_growth.seed_points", [])
    if config_seeds and isinstance(config_seeds, list):
        for i, seed_info in enumerate(config_seeds):
            if isinstance(seed_info, dict) and 'position' in seed_info:
                seed_points_viz_data.append((np.array(seed_info.get('position')),
                                             float(seed_info.get('initial_radius',0.1)),
                                             seed_info.get('id',f"S_{i}")))
            else: logger.warning(f"Skipping invalid seed_info in config: {seed_info}")

    if vascular_graph and vascular_graph.number_of_nodes() > 0:
        final_tree_vtp_path = os.path.join(output_dir, "final_plot_vascular_tree.vtp")
        io_utils.save_vascular_tree_vtp(vascular_graph, final_tree_vtp_path,
                                        radius_attr='radius', pressure_attr='pressure', flow_attr='flow_solver')
        logger.info(f"Final vascular tree saved for analysis: {final_tree_vtp_path}")

        analyze_radii_distribution(vascular_graph, output_dir)
        analyze_segment_lengths(vascular_graph, output_dir)
        analyze_bifurcation_geometry(vascular_graph, output_dir,
                                     murray_exponent=config_manager.get_param(config, "vascular_properties.murray_law_exponent", 3.0))
        analyze_degree_distribution(vascular_graph, output_dir)
        analyze_network_connectivity(vascular_graph, output_dir)
        analyze_volumetric_densities(vascular_graph, tissue_data, output_dir)

        # Plotting with Radius (always try this if graph exists)
        pv_plot_title_radius = f"Vasculature (Radius, {vascular_graph.number_of_nodes()}N, {vascular_graph.number_of_edges()}E)"
        pv_screenshot_path_radius = os.path.join(output_dir, f"final_vascular_tree_radius_3D{'' if plot_context_masks else '_no_context'}.png")
        if PYVISTA_AVAILABLE:
            plot_vascular_tree_pyvista(
                graph=vascular_graph, title=pv_plot_title_radius, output_screenshot_path=pv_screenshot_path_radius,
                tissue_masks=masks_to_plot_for_pyvista, seed_points_world=seed_points_viz_data,
                color_by_scalar='radius', custom_cmap=config_manager.get_param(config, "visualization.pyvista_cmap_radius", "viridis")
            )

            # Plotting with Flow
            if any('flow_solver' in data for _,_,data in vascular_graph.edges(data=True) if 'flow_solver' in data and np.any(np.isfinite(data['flow_solver']))):
                pv_plot_title_flow = f"Vasculature (Edge Flow, {vascular_graph.number_of_nodes()}N, {vascular_graph.number_of_edges()}E)"
                pv_screenshot_path_flow = os.path.join(output_dir, f"final_vascular_tree_flow_3D{'' if plot_context_masks else '_no_context'}.png")
                plot_vascular_tree_pyvista(
                    graph=vascular_graph, title=pv_plot_title_flow, output_screenshot_path=pv_screenshot_path_flow,
                    tissue_masks=masks_to_plot_for_pyvista, seed_points_world=seed_points_viz_data,
                    color_by_scalar='flow_solver', custom_cmap='coolwarm', scalar_bar_title_override="Flow (mm³/s)"
                )
            else: logger.info("Skipping flow plot: No valid 'flow_solver' data on edges.")

            # Plotting with Pressure
            if any('pressure' in data for _,data in vascular_graph.nodes(data=True) if 'pressure' in data and np.any(np.isfinite(data['pressure']))):
                pv_plot_title_pressure = f"Vasculature (Node Pressure, {vascular_graph.number_of_nodes()}N, {vascular_graph.number_of_edges()}E)"
                pv_screenshot_path_pressure = os.path.join(output_dir, f"final_vascular_tree_pressure_3D{'' if plot_context_masks else '_no_context'}.png")
                plot_vascular_tree_pyvista(
                    graph=vascular_graph, title=pv_plot_title_pressure, output_screenshot_path=pv_screenshot_path_pressure,
                    tissue_masks=masks_to_plot_for_pyvista, seed_points_world=seed_points_viz_data,
                    color_by_scalar='pressure', custom_cmap='coolwarm', scalar_bar_title_override="Pressure (Pa)"
                )
            else: logger.info("Skipping pressure plot: No valid 'pressure' data on nodes.")
        else: logger.warning("PyVista not available, skipping 3D plots.")
    else: # No valid vascular_graph
        logger.warning("Vascular graph is empty or None. Skipping VTP save and quantitative analyses.")
        if PYVISTA_AVAILABLE:
            plot_vascular_tree_pyvista(None, title=f"Tissue Context & Seeds (No Vasculature){' (No Context Masks)' if not plot_context_masks else ''}",
                                       output_screenshot_path=os.path.join(output_dir, f"context_only_plot{'' if plot_context_masks else '_no_context'}.png"),
                                       tissue_masks=masks_to_plot_for_pyvista, seed_points_world=seed_points_viz_data)

    logger.info("Final visualizations and analyses generation complete.")