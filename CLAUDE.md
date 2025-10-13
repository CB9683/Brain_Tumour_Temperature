# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**IMPORTANT**: This project requires a specific conda environment to run properly.

### Conda Environment
- **Environment Path**: `/Users/c3495249/miniconda3/envs/thermo311`
- **Activation Command**: `conda activate /Users/c3495249/miniconda3/envs/thermo311`

**Always activate this environment before running any simulations or Python commands.**

### Running Simulations
```bash
# Activate the conda environment first
conda activate /Users/c3495249/miniconda3/envs/thermo311

# Then run the main simulation
python3 main.py
```

### Coordinate System and Visualization

**Important**: The project uses RAS (Right-Anterior-Superior) medical coordinates internally but applies coordinate transformations for visualization tools:

- **ParaView VTK files**: Automatically transformed using RAS-to-Blender transformation (X→X, Y→-Z, Z→Y)
- **Blender import**: Uses same transformation matrix for consistent alignment
- **This ensures measured vessels and synthetic vessels align properly in both ParaView and Blender**

The transformation is applied by default in both:
- `save_vascular_tree_vtp()` for vessel networks (can be disabled with `apply_coordinate_transform=False`)
- `export_tissue_masks_to_vtk()` for tissue masks (can be disabled with `apply_coordinate_transform=False`)

### ParaView Visualization Tips

**VTK File Display Issues - SOLVED:**
- If VTK (.vti) files show only wireframe cubes instead of actual data, this was due to incorrect memory layout in VTK export
- **Fixed**: VTK export now uses point-by-point data filling with `SetScalarComponentFromDouble()` for correct memory layout
- **Volume rendering**: Should work directly after loading VTK files
- **Surface rendering**: Use Contour filter with value 0.5 for binary masks (GM, WM, perfused areas)

**Coordinate Alignment:**
- Vessel (.vtp) and tissue (.vti) files use identical RAS-to-Blender transformation
- Load both file types in same ParaView session for perfect alignment
- Vessel terminals should align with tissue mask boundaries

## Project Overview

This is a sophisticated Python framework for simulating vascular network growth and perfusion in biological tissue, with specialized support for modeling tumor-induced angiogenesis. The simulation implements physics-based algorithms for healthy vasculature development using Geometry-Based Optimization (GBO) and tumor angiogenesis modeling with VEGF guidance.

## Key Architecture Components

### Core Pipeline Flow
1. **Configuration Management** (`config_manager.py`) - Centralized YAML-based parameter management
2. **Data Loading** (`io_utils.py`) - NIfTI medical imaging and VTP/VTK vessel data I/O
3. **Healthy Vasculature Growth** (`vascular_growth.py`) - GBO algorithm for optimal vessel branching
4. **Tumor Angiogenesis** (`angiogenesis.py`) - VEGF-guided sprouting, co-option, and anastomosis
5. **Flow Simulation** (`perfusion_solver.py`) - 1D Poiseuille flow solver with linear system solving
6. **Visualization** (`visualization.py`) - 3D PyVista rendering and quantitative analysis

### Data Structures
- **Vascular Graph**: NetworkX DiGraph with nodes (vessel junctions) and edges (vessel segments)
- **Node Attributes**: `pos` (3D coordinates), `radius`, `type` (measured_root, synthetic_terminal, etc.), `Q_flow`
- **Edge Attributes**: `length`, `radius`, `flow_solver`, vessel classification data
- **Tissue Data**: Dict containing NIfTI masks for GM/WM/CSF/Tumor, metabolic demand maps, VEGF fields

### Energy Model
The GBO algorithm minimizes total energy = viscous dissipation + metabolic maintenance cost:
- **Flow Energy**: 8μLQ²/(πr⁴) (Poiseuille's law)
- **Metabolic Energy**: C_met × π × r² × L (vessel wall maintenance)
- Optimal bifurcations found using KMeans clustering and energy minimization

## Development Commands

### Installation
```bash
# Install in editable mode with development dependencies
pip install -e .[dev]

# Or install from requirements.txt
pip install -r requirements.txt
```

### Running Simulations
```bash
# Run main simulation with default config
python main.py

# Run with custom config
python main.py --config path/to/config.yaml --output_dir custom_output

# Generate default config template
python -c "from src.config_manager import create_default_config; create_default_config('new_config.yaml')"
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_config_manager.py

# Run with verbose output
pytest -v tests/
```

### Key Configuration Sections

The `config.yaml` file controls all simulation parameters:

- **paths**: Input NIfTI files (GM/WM/CSF masks), arterial centerlines, output directories
- **gbo_growth**: GBO algorithm parameters including max_iterations, energy coefficients, branching triggers
- **tumor_angiogenesis**: VEGF field settings, sprouting parameters, anastomosis rules
- **vascular_properties**: Blood viscosity, Murray's law parameters, radius constraints
- **visualization**: PyVista rendering options, plot settings, intermediate saves

### Core Modules Interaction

1. **main.py** orchestrates the full pipeline: data loading → healthy GBO → tumor angiogenesis → visualization
2. **perfusion_solver.py** constructs and solves Ax=B linear systems for pressure/flow distribution
3. **energy_model.py** provides optimization objectives for vessel placement decisions
4. **data_structures.py** defines standard graph attributes and helper functions for vessel networks

### Input Data Requirements

- **NIfTI masks**: Gray matter, white matter, tumor extent (coordinate system: RAS orientation preferred)
- **Arterial centerlines**: VTP/VTK files with radius data, or TXT files with coordinates
- **Configuration**: YAML file specifying all simulation parameters

### Output Structure

Simulations create timestamped directories in `output/simulation_results/` containing:
- **VTP files**: 3D vessel networks for ParaView visualization
- **VTI files**: Tissue masks and perfused areas in `tissue_vtk/` subdirectory (ParaView-compatible)
- **NIfTI files**: Tissue masks, perfusion maps, intermediate states
- **Analysis plots**: Vessel radius distributions, Murray's law compliance, branching angles
- **Logs**: Detailed simulation progress and debugging information

**VTK Export Configuration:**
- Enable with `visualization.export_tissue_vtk: true` in config.yaml
- Generates tissue masks (GM, WM, CSF, perfused areas) as .vti files
- Coordinate transformation applied by default for alignment with vessel data

### Development Notes

- The codebase uses extensive logging - check log levels in config for debugging
- Flow solver includes diagnostic code for singular matrix detection (disconnected vessel components)
- Intermediate visualization saves are controlled by config parameters
- All coordinate transformations use affine matrices from NIfTI headers
- Random seed control for reproducible simulations via config.yaml