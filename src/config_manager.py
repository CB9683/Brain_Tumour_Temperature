# src/config_manager.py
import yaml
import os
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration parameters.
    
    Raises:
        FileNotFoundError: If the config file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from: {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {config_path}: {e}")
        raise

def get_param(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Retrieves a parameter from the config dictionary using a dot-separated key path.
    Example: get_param(config, "simulation.gbo.max_iterations")

    Args:
        config (Dict[str, Any]): The configuration dictionary.
        key_path (str): Dot-separated path to the key (e.g., "parent.child.key").
        default (Any, optional): Default value to return if key is not found. Defaults to None.

    Returns:
        Any: The parameter value or the default value.
    """
    keys = key_path.split('.')
    value = config
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        logger.warning(f"Parameter '{key_path}' not found in config. Using default: {default}")
        return default

def create_default_config(config_path: str = "config.yaml"):
    """
    Creates a default configuration file if it doesn't exist.
    """
    from src import constants # To access default values

    default_config_content = {
        "paths": {
            "output_dir": "output/simulation_results",
            "wm_nifti": "data/sample_brain_wm.nii.gz",
            "gm_nifti": "data/sample_brain_gm.nii.gz",
            "csf_nifti": "data/sample_brain_csf.nii.gz", # Optional
            "tumor_nifti": "data/sample_tumor.nii.gz", # Optional
            "arterial_centerlines": "data/sample_arteries.vtp" # or .txt
        },
        "simulation": {
            "random_seed": 42,
            "log_level": "INFO", # DEBUG, INFO, WARNING, ERROR
            "units": { # Define units used for consistency, e.g. 'mm' for length
                "length": "mm",
                "pressure": "Pa", # Pascal
                "flow_rate": "mm^3/s"
            }
        },
        "tissue_properties": {
            "metabolic_rates": { # in 1/s (ml_blood / s / ml_tissue)
                "gm": constants.Q_MET_GM_PER_ML,
                "wm": constants.Q_MET_WM_PER_ML,
                "tumor_rim": constants.Q_MET_TUMOR_RIM_PER_ML,
                "tumor_core": constants.Q_MET_TUMOR_CORE_PER_ML,
                "csf": constants.Q_MET_CSF_PER_ML
            },
            "permeability": { # in mm^2 (if length unit is mm)
                "gm": constants.DEFAULT_TISSUE_PERMEABILITY_GM,
                "wm": constants.DEFAULT_TISSUE_PERMEABILITY_WM,
                # Tumor permeability can be higher
                "tumor": 5e-7 # mm^2
            }
        },
        "vascular_properties": {
            "blood_viscosity": constants.DEFAULT_BLOOD_VISCOSITY, # Pa.s
            "murray_law_exponent": constants.MURRAY_LAW_EXPONENT,
            "initial_terminal_flow": constants.INITIAL_TERMINAL_FLOW_Q, # mm^3/s (if units are mm)
            "min_segment_length": 0.1, # mm
            "max_segment_length": 2.0, # mm
            "min_radius": 0.005 # mm (e.g. 5 microns)
        },
        "gbo_growth": {
            "max_iterations": 100,
            "energy_coefficient_C_met": constants.DEFAULT_C_MET_VESSEL_WALL, # W/m^3 or equivalent in chosen units
            "target_perfusion_level": 0.9, # Target fraction of demand to be met
            "branching_threshold_radius": 0.2, # mm (example: branch if terminal radius exceeds this)
            "bifurcation_angle_search_steps": 5, # Number of angles to test
            "bifurcation_length_factor": 0.5, # New segments are this factor of parent segment length (heuristic)
            "stop_criteria": {
                "max_radius_factor_measured": 1.0 # Stop if synth. radius = 1.0 * measured_terminal_radius
            }
        },
        "tumor_angiogenesis": {
            "enabled": True,
            "sprouting_vessel_min_radius": 0.1, # mm
            "growth_bias_strength": 0.5, # For biased random walk
            "max_tumor_vessels": 500,
            "segment_length_mean_tumor": 0.3, # mm
            "segment_length_std_tumor": 0.1, # mm
            "tortuosity_factor": 1.5 # How much more tortuous tumor vessels are
        },
        "perfusion_solver": {
            "enabled": True,
            "inlet_pressure": 10000, # Pa (approx 75 mmHg)
            "terminal_outlet_pressure": 2000, # Pa (approx 15 mmHg, for network solver if applicable)
            "coupling_beta": constants.DEFAULT_COUPLING_BETA, # mm^3 / (s*Pa)
            "use_flow_sinks_for_terminals": True, # Example, if we add options later
            "max_iterations_coupling": 20,
            "convergence_tolerance_coupling": 1e-4,
            "darcy_solver_max_iter": 1000,
            "darcy_solver_tolerance": 1e-6
        },
        "visualization":{
            "save_intermediate_steps": False,
            "plot_slice_axis": "axial", # axial, sagittal, coronal
            "plot_slice_index": -1 # -1 for middle slice
        }
    }
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            yaml.dump(default_config_content, f, sort_keys=False, indent=4)
        logger.info(f"Created default configuration file: {config_path}")
    else:
        logger.info(f"Configuration file already exists: {config_path}")

if __name__ == '__main__':
    # Example usage:
    logging.basicConfig(level=logging.INFO)
    
    # Create a dummy config.yaml if it doesn't exist in the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dummy_config_path = os.path.join(script_dir, "..", "config.yaml") # Assumes script is in src/
    
    if not os.path.exists(dummy_config_path):
        os.makedirs(os.path.dirname(dummy_config_path), exist_ok=True)
        create_default_config(dummy_config_path)

    try:
        config = load_config(dummy_config_path)
        print("Config loaded successfully.")
        print("Random seed:", get_param(config, "simulation.random_seed", 0))
        print("GM metabolic rate:", get_param(config, "tissue_properties.metabolic_rates.gm"))
        print("Non-existent param:", get_param(config, "foo.bar.baz", "default_val"))
        
        # Test output directory creation from config
        output_dir = get_param(config, "paths.output_dir", "output/default_sim")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {output_dir}")

    except Exception as e:
        print(f"Error in example usage: {e}")