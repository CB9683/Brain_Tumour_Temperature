# Vascular Network Growth and Perfusion Simulation

This repository contains a sophisticated, physics-based simulation pipeline for modeling the growth of vascular networks in biological tissue. It includes modules for geometry-based optimization (GBO) of healthy vasculature, a detailed model for tumor-induced angiogenesis, and a robust 1D Poiseuille flow solver to simulate blood perfusion.

## Overview

The primary goal of this project is to generate and analyze realistic vascular networks under both physiological and pathological (tumor) conditions. The simulation is driven by biophysical principles, including energy minimization for vessel structure, metabolic demand of the underlying tissue, and chemotactic guidance by growth factors like VEGF.

### Key Features

*   **Physics-Based Growth:** Implements a Geometry-Based Optimization (GBO) algorithm where vessel growth (extension and bifurcation) is driven by minimizing a total energy function (viscous dissipation + metabolic cost).
*   **Detailed Perfusion Model:** Includes a robust 1D Poiseuille flow solver that constructs and solves a linear system of equations to determine pressures and flows throughout the entire vascular network. It features extensive diagnostics for common numerical issues like singular matrices.
*   **Tumor Angiogenesis Module:** Simulates key aspects of tumor-driven vessel growth, including:
    *   **VEGF-guided sprouting:** New vessels sprout from existing ones towards high concentrations of Vascular Endothelial Growth Factor (VEGF).
    *   **Vessel Co-option:** The tumor hijacks and modifies existing healthy vessels.
    *   **Anastomosis:** Growing vessel tips can fuse with other segments to form loops, a hallmark of tumor vasculature.
*   **Modular and Configurable:** The entire simulation is controlled by a central `config.yaml` file, allowing for easy modification of physical constants, model parameters, and simulation settings.
*   **Multi-Format I/O:** Supports standard medical imaging (NIfTI) and 3D geometry (VTP, TXT) formats for input and output.
*   **Quantitative Analysis & Visualization:** Includes a powerful `visualization` module using PyVista and Matplotlib to generate 3D renderings of the vasculature and quantitative plots for analyzing network properties (radii distribution, Murray's Law, etc.).

## ⚙️ Pipeline Workflow

The simulation can be conceptualized as two main phases, which can be run sequentially: **1. Healthy Vasculature Growth** and **2. Tumor Angiogenesis**.

1.  **Configuration (`config_manager.py`):** The simulation starts by loading all parameters from `config.yaml`. This file defines everything from file paths and physical constants to algorithm-specific parameters.

2.  **Data Loading (`io_utils.py`):** The pipeline loads initial conditions, which typically include:
    *   NIfTI masks for different tissue types (Gray Matter, White Matter, Tumor Extent).
    *   Initial arterial centerlines from a VTP or TXT file, which serve as the starting point for the GBO growth.

3.  **Healthy Vasculature Growth (`vascular_growth.py`):**
    *   **Initialization:** The GBO process begins from the terminal points of the initial arterial tree or from specified seed points. An initial perfused territory is defined.
    *   **Iterative Growth Loop:**
        a.  **Frontier Identification:** For each growing terminal, identify nearby unperfused tissue (`find_growth_frontier_voxels`).
        b.  **Energy Optimization (`energy_model.py`):** Decide whether to extend the terminal or create a bifurcation. This choice is based on finding the configuration that minimizes the total energy (flow + metabolic). The optimal bifurcation is found using `find_optimal_bifurcation_for_combined_territory`.
        c.  **Territory Refinement:** The tissue territories supplied by each terminal are periodically re-assigned using a weighted Voronoi tessellation to ensure efficiency.
        d.  **Perfusion & Adaptation:** The `perfusion_solver` is called at set intervals to calculate real flows. The vessel radii are then adapted globally based on these flows, following Murray's Law.
        e.  **Pruning:** Non-functional or inefficient vessel segments are removed.
    *   This loop continues until the target tissue domain is adequately perfused or other stopping criteria are met.

4.  **Tumor Angiogenesis (`angiogenesis.py`):**
    *   **Tumor Initialization:** An active tumor mass is seeded within a predefined "max extent" mask.
    *   **Macro Iteration Loop:**
        a.  **Tumor State Update:** The tumor's rim and core are identified. The metabolic demand map and a VEGF concentration field are updated. The VEGF field is highest at the proliferative rim and diffuses outwards.
        b.  **Vessel Co-option:** Existing vessels engulfed by the growing tumor are "co-opted"—their properties (radius, leakiness) are altered.
        c.  **Angiogenic Growth:** In a series of micro-steps, new vessels are grown via:
            *   **Sprouting:** New vessel tips sprout from existing vessels, guided by the gradient of the VEGF field.
            *   **Extension:** Tips grow and follow the VEGF gradient.
            *   **Branching:** Tips can branch stochastically, influenced by local VEGF levels.
            *   **Anastomosis:** Tips can fuse with nearby segments to form loops.
        d.  **Flow Solver & Adaptation:** The perfusion solver is run periodically, and the tumor vasculature adapts to the resulting flow patterns, often becoming tortuous and inefficient.

5.  **Analysis and Visualization (`visualization.py`):**
    *   At the end of the simulation (and optionally at intermediate steps), the final vascular graph and tissue masks are saved.
    *   A suite of quantitative analyses is performed to generate plots of vessel radii, lengths, bifurcation angles, and more.
    *   3D visualizations of the final network, colored by pressure, flow, or radius, are generated using PyVista.

## 📄 File-by-File Functional Breakdown

*   **`config_manager.py`**: Manages all simulation parameters. Its `create_default_config()` function is invaluable as it documents every configurable parameter in the project, from physical constants like `blood_viscosity` to algorithmic details like `gbo_growth.max_iterations`.

*   **`io_utils.py`**: Handles all file input/output. It loads NIfTI masks (`load_nifti_image`), initial arterial trees from VTP/TXT files, and saves the final NetworkX graph structures as VTP files (`save_vascular_tree_vtp`) for 3D visualization.

*   **`data_structures.py`**: Defines the data conventions for the project. It specifies the standard attributes for nodes (e.g., `pos`, `radius`, `type`) and edges (e.g., `length`, `flow_solver`) in the `networkx.DiGraph` that represents the vasculature. It also provides helper functions for creating and modifying the graph.

*   **`utils.py`**: A collection of essential helper functions, including coordinate transformations (`voxel_to_world`, `world_to_voxel`), distance calculations, and output directory management.

*   **`constants.py`**: Centralizes hard-coded physical and numerical constants, such as default blood viscosity, metabolic rates for different tissues (`Q_MET_GM_PER_ML`), and Murray's Law exponent.

*   **`perfusion_solver.py`**: The heart of the blood flow simulation.
    *   `solve_1d_poiseuille_flow()`: Constructs a linear system of equations (`Ax = B`) representing flow conservation at every node in the vascular graph.
    *   It calculates hydraulic resistance for each segment using Poiseuille's law (`calculate_segment_resistance`).
    *   It sets up boundary conditions based on a root pressure inlet (`is_flow_root`) and terminal flow demands (`Q_flow`).
    *   Crucially, it contains extensive diagnostic code to detect and report on singular matrices (rank deficiency), which can be caused by disconnected ("floating") parts of the vascular graph. This makes it very robust.
    *   The function solves for nodal pressures and then calculates the flow in each segment, updating the graph in-place.

*   **`energy_model.py`**: Provides the objective function for the GBO algorithm.
    *   It defines the total energy of a vessel segment as the sum of viscous dissipation (`calculate_segment_flow_energy`) and the vessel's metabolic maintenance cost (`calculate_segment_metabolic_energy`).
    *   `find_optimal_bifurcation_for_combined_territory()` is the key function. It uses KMeans clustering to find candidate locations for two new child vessels and determines the split that minimizes the total energy cost of the new bifurcation.

*   **`vascular_growth.py`**: Implements the GBO algorithm for growing a "healthy" vascular tree.
    *   `grow_healthy_vasculature()`: The main orchestrator for this module.
    *   It iteratively extends and branches vessels to perfuse a target tissue domain, driven by the `energy_model`.
    *   It dynamically re-calculates tissue territories, runs the `perfusion_solver` to get realistic flows, and adapts vessel radii accordingly, creating a powerful feedback loop between structure and function.
    *   `prune_vascular_graph()` removes segments with low flow or radius, ensuring an efficient final structure.

*   **`angiogenesis.py`**: Implements the logic for tumor-induced vascular growth.
    *   `simulate_tumor_angiogenesis_fixed_extent()`: The main orchestrator for this module.
    *   It simulates tumor mass growth, updates a `VEGF_field` that acts as a chemical attractant, and modifies the vasculature through co-option, sprouting, extension, and anastomosis. This creates the characteristic chaotic and leaky tumor vascular network.

*   **`visualization.py`**: Responsible for all output plots and 3D scenes.
    *   `plot_vascular_tree_pyvista()`: A powerful function that creates 3D renderings of the vascular network and can overlay tissue masks. Vessels can be colored by various properties (radius, pressure, flow).
    *   `analyze_*` functions: A suite of functions that calculate and plot key network metrics, essential for validating the simulation results (e.g., `analyze_bifurcation_geometry` checks for Murray's Law compliance).

## 🚀 Usage

1.  **Setup Environment:**
    ```bash
    # It's highly recommended to use a virtual environment
    python -m venv venv
    source venv/bin/activate

    # Install required packages
    pip install numpy networkx pyvista nibabel pyyaml matplotlib scikit-learn
    ```

2.  **Prepare Input Data:**
    *   Create a `data/` directory.
    *   Place your NIfTI files (e.g., `gm.nii.gz`, `wm.nii.gz`, `tumor_extent.nii.gz`) inside.
    *   Place your initial arterial centerline file (e.g., `arteries.vtp` or `arteries.txt`) inside.

3.  **Configure the Simulation:**
    *   Create a `config.yaml` file in the root directory. You can generate a comprehensive template by running `python -c "from src.config_manager import create_default_config; create_default_config()"`.
    *   Edit `config.yaml` to point to your input files and adjust simulation parameters as needed.

4.  **Run the Simulation:**
    *   A main execution script (e.g., `main.py`) would orchestrate the pipeline. A possible structure for such a script would be:

    ```python
    # main.py (Example)
    import logging
    from src import config_manager, io_utils, vascular_growth, angiogenesis, perfusion_solver, visualization, utils

    # 1. Setup
    logging.basicConfig(level=logging.INFO)
    config = config_manager.load_config("config.yaml")
    output_dir = utils.create_output_directory("output", "sim_run")
    
    # 2. Load Data
    # ... code to load NIfTI masks and arterial centerlines ...
    # ... code to create tissue_data dictionary ...

    # 3. Run GBO for healthy vasculature
    healthy_graph = vascular_growth.grow_healthy_vasculature(config, tissue_data, initial_arterial_graph, output_dir)
    
    # 4. Run Tumor Angiogenesis
    final_graph = angiogenesis.simulate_tumor_angiogenesis_fixed_extent(config, tissue_data, healthy_graph, output_dir, perfusion_solver.solve_1d_poiseuille_flow)

    # 5. Generate Final Visualizations
    visualization.generate_final_visualizations(config, output_dir, tissue_data, final_graph)
    ```

5.  **Analyze Outputs:**
    *   Check the specified `output_dir`.
    *   View the `.vtp` files in a 3D viewer like ParaView.
    *   Examine the generated `.png` plots for quantitative analysis of the network structure.
    *   Inspect the intermediate NIfTI masks and logs for debugging.



    Territory Assignment (Voronoi) → Metabolic Demand → Flow Requirement
                                                           ↓
                                                    Murray's Law → Radius
                                                           ↓
                                          Energy Optimization → Bifurcation Decision
                                                           ↓
                                              Smooth Path Growth → Realistic Morphology