# src/energy_model.py
import numpy as np
import networkx as nx # Not directly used in functions yet, but good for context if needed later
import logging
from typing import Tuple, List, Optional # For type hinting

# Attempt to import sklearn for KMeans, but make it optional
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    KMeans = None # Placeholder if not available

from src import constants, config_manager, utils

logger = logging.getLogger(__name__)

def calculate_segment_flow_energy(length: float, radius: float, flow: float, viscosity: float) -> float:
    """ 
    Calculates viscous energy dissipation (power) for a vessel segment.
    E_flow = (8 * mu * L * Q^2) / (pi * r^4) 
    Units: If mu (Pa.s), L (m), Q (m^3/s), r (m), then E_flow is in Watts (J/s).
           If mu (Pa.s), L (mm), Q (mm^3/s), r (mm):
           Pa.s * mm * (mm^3/s)^2 / mm^4 = (N/m^2).s * (1e-3 m) * (1e-9 m^3/s)^2 / (1e-12 m^4)
                                       = N.s/m^2 * 1e-3 m * 1e-18 m^6/s^2 / 1e-12 m^4
                                       = 1e-9 N.m/s = 1e-9 W.
           To get milliWatts (mW), multiply by 1000: 1e-6 mW.
           Users must ensure consistency or apply conversion factors.
    """
    if radius < constants.EPSILON: # Avoid division by zero
        # If there's flow through a zero-radius vessel, cost is infinite
        return np.inf if abs(flow) > constants.EPSILON else 0.0
    if abs(flow) < constants.EPSILON: # If flow is effectively zero, energy dissipation is zero
        return 0.0
    
    # For extremely small radii with non-zero flow, energy can become excessively large.
    # This is physically plausible (high resistance) but can cause numerical issues.
    if radius < 1e-7 and abs(flow) > constants.EPSILON: # e.g., radius < 0.1 micron
        logger.debug(f"Very small radius ({radius:.2e}) with non-zero flow ({flow:.2e}). Flow energy may be extreme.")

    return (8.0 * viscosity * length * (flow**2)) / (constants.PI * (radius**4))

def calculate_segment_metabolic_energy(length: float, radius: float, c_met_coeff: float) -> float:
    """ 
    Calculates metabolic maintenance cost (power) for a vessel segment.
    E_metabolic = C_met_coeff * pi * r^2 * L 
    Units: If C_met_coeff (W/m^3), r (m), L (m), then E_metabolic is in Watts.
           If C_met_coeff (mW/mm^3), r (mm), L (mm), then E_metabolic is in milliWatts.
           The config value for c_met_coeff should be in units consistent with E_flow.
    """
    if radius < constants.EPSILON or length < constants.EPSILON:
        return 0.0
    return c_met_coeff * constants.PI * (radius**2) * length

def calculate_bifurcation_loss(
    parent_pos: np.ndarray, # Position of the bifurcation point itself
    child1_pos: np.ndarray, child1_radius: float, child1_flow: float,
    child2_pos: np.ndarray, child2_radius: float, child2_flow: float,
    config: dict
) -> float:
    """
    Calculates the total loss (E_flow + E_metabolic) for the two new child segments
    originating from parent_pos. The flows are specific to these new child segments.
    """
    viscosity = config_manager.get_param(config, "vascular_properties.blood_viscosity", constants.DEFAULT_BLOOD_VISCOSITY)
    c_met = config_manager.get_param(config, "gbo_growth.energy_coefficient_C_met_vessel_wall", constants.DEFAULT_C_MET_VESSEL_WALL)

    l_c1 = utils.distance(parent_pos, child1_pos)
    l_c2 = utils.distance(parent_pos, child2_pos)

    # Calculate energy for child 1 segment
    if l_c1 < constants.EPSILON or child1_radius < constants.EPSILON:
        e_flow_c1 = np.inf if abs(child1_flow) > constants.EPSILON else 0.0
        e_met_c1 = 0.0
    else:
        e_flow_c1 = calculate_segment_flow_energy(l_c1, child1_radius, child1_flow, viscosity)
        e_met_c1 = calculate_segment_metabolic_energy(l_c1, child1_radius, c_met)

    # Calculate energy for child 2 segment
    if l_c2 < constants.EPSILON or child2_radius < constants.EPSILON:
        e_flow_c2 = np.inf if abs(child2_flow) > constants.EPSILON else 0.0
        e_met_c2 = 0.0
    else:
        e_flow_c2 = calculate_segment_flow_energy(l_c2, child2_radius, child2_flow, viscosity)
        e_met_c2 = calculate_segment_metabolic_energy(l_c2, child2_radius, c_met)
    
    total_loss = e_flow_c1 + e_met_c1 + e_flow_c2 + e_met_c2
    
    # logger.debug(f"Bifurcation candidate: L1={l_c1:.2f}, R1={child1_radius:.4f}, Q1={child1_flow:.2e} -> E_f1={e_flow_c1:.2e}, E_m1={e_met_c1:.2e}")
    # logger.debug(f"                     L2={l_c2:.2f}, R2={child2_radius:.4f}, Q2={child2_flow:.2e} -> E_f2={e_flow_c2:.2e}, E_m2={e_met_c2:.2e}")
    # logger.debug(f"                     Total Loss = {total_loss:.3e}")
    return total_loss



def find_optimal_bifurcation_for_combined_territory(
    parent_terminal_gbo_data: object, # GBOIterationData for the terminal that will bifurcate
    # Combined territory: parent's old territory + new frontier region
    combined_territory_voxel_indices_flat: np.ndarray, 
    tissue_data: dict,
    config: dict,
    k_murray_factor: float,
    murray_exponent: float
) -> Optional[Tuple[np.ndarray, float, float, np.ndarray, float, float, float]]:
    """
    Searches for an optimal bifurcation for the parent_terminal to supply a 
    COMBINED territory (its old territory + a new growth region).
    The children C1 and C2 will share the total demand of this combined_territory.
    
    Args:
        parent_terminal_gbo_data: GBOIterationData of the branching terminal.
        combined_territory_voxel_indices_flat: Flat indices of ALL voxels 
                                               (old territory + new frontier) to be supplied.
        tissue_data: Full tissue data dict.
        config: Simulation config.
        k_murray_factor, murray_exponent: For radius calculation.

    Returns:
        Tuple (child1_pos, child1_radius, child1_total_flow, 
               child2_pos, child2_radius, child2_total_flow, min_loss_for_new_segments) 
        or None if no suitable bifurcation found.
        The child flows are their respective total target flows.
    """
    parent_id = parent_terminal_gbo_data.id
    parent_pos = parent_terminal_gbo_data.pos # This is the bifurcation point
    logger.debug(f"Finding optimal bifurcation for {parent_id} at {np.round(parent_pos,3)} to supply combined "
                 f"territory of {len(combined_territory_voxel_indices_flat)} voxels.")

    if len(combined_territory_voxel_indices_flat) == 0:
        logger.debug(f"Terminal {parent_id}: Combined target territory is empty. No bifurcation.")
        return None

    # Get world coordinates and demand for ALL voxels in the combined territory
    combined_voxels_world_coords = tissue_data['world_coords_flat'][combined_territory_voxel_indices_flat]
    
    demand_map_3d_indices = tissue_data['voxel_indices_flat'][combined_territory_voxel_indices_flat]
    demand_per_combined_voxel_qmet = tissue_data['metabolic_demand_map'][
        demand_map_3d_indices[:,0],
        demand_map_3d_indices[:,1],
        demand_map_3d_indices[:,2]
    ]
    demand_of_combined_voxels_flow = demand_per_combined_voxel_qmet * tissue_data['voxel_volume']
    total_demand_of_combined_territory = np.sum(demand_of_combined_voxels_flow)

    if total_demand_of_combined_territory < constants.EPSILON:
        logger.debug(f"Terminal {parent_id}: Total demand in combined territory is negligible. No bifurcation.")
        return None

    num_candidate_location_sets = config_manager.get_param(config, "gbo_growth.bifurcation_candidate_points", 10)
    min_seg_len = config_manager.get_param(config, "vascular_properties.min_segment_length", 0.1)
    min_radius = config_manager.get_param(config, "vascular_properties.min_radius", constants.MIN_VESSEL_RADIUS_MM)

    best_bifurcation_params = None
    min_loss_found = np.inf

    if len(combined_voxels_world_coords) < 2:
        logger.debug(f"Terminal {parent_id}: Combined territory too small ({len(combined_voxels_world_coords)} voxels) "
                     "for meaningful bifurcation. Consider extension.")
        return None # Bifurcation needs to split demand between two children

    # Candidate child locations should be within or near the combined_territory
    # (KMeans or random sampling on combined_voxels_world_coords)
    for i in range(num_candidate_location_sets):
        c1_pos_candidate, c2_pos_candidate = None, None
        # --- 1. Generate candidate child locations (c1_pos, c2_pos) based on combined_territory ---
        # (Using KMeans as before, but on combined_voxels_world_coords)
        if SKLEARN_AVAILABLE and KMeans is not None:
            try:
                n_clust = min(2, len(combined_voxels_world_coords))
                if n_clust < 2: # Should be caught by len(combined_voxels_world_coords) < 2 above
                    continue 
                kmeans = KMeans(n_clusters=n_clust, random_state=i, n_init='auto').fit(combined_voxels_world_coords)
                c1_pos_candidate = kmeans.cluster_centers_[0]
                c2_pos_candidate = kmeans.cluster_centers_[1]
            except Exception as e_km:
                logger.warning(f"KMeans failed for combined territory (iter {i}): {e_km}. Fallback.")
                indices = np.random.choice(len(combined_voxels_world_coords), 2, replace=False)
                c1_pos_candidate = combined_voxels_world_coords[indices[0]]
                c2_pos_candidate = combined_voxels_world_coords[indices[1]]

                # <<<<<<<<<<<<<<< ADD THE FOLLOWING CODE BLOCK >>>>>>>>>>>>>>>>>
                # NEW: Add a constraint on the maximum distance for new child segments
                max_child_dist = config_manager.get_param(config, "gbo_growth.bifurcation_max_child_distance", 2.5)

                if utils.distance(parent_pos, c1_pos_candidate) > max_child_dist or \
                utils.distance(parent_pos, c2_pos_candidate) > max_child_dist:
                    # logger.debug(f"Candidate children too far from parent {parent_id}. Skipping.")
                    continue # Skip this candidate pair and try the next one
                # <<<<<<<<<<<<<<<<<<<<<< END OF NEW BLOCK >>>>>>>>>>>>>>>>>>>>>>
        
        else:
            if i == 0 and not SKLEARN_AVAILABLE : logger.warning("Sklearn KMeans not available for child placement.")
            indices = np.random.choice(len(combined_voxels_world_coords), 2, replace=False)
            c1_pos_candidate = combined_voxels_world_coords[indices[0]]
            c2_pos_candidate = combined_voxels_world_coords[indices[1]]

        if utils.distance(parent_pos, c1_pos_candidate) < min_seg_len or \
           utils.distance(parent_pos, c2_pos_candidate) < min_seg_len or \
           utils.distance(c1_pos_candidate, c2_pos_candidate) < min_seg_len:
            continue

        # --- 2. Assign ALL voxels in combined_territory to c1 or c2 ---
        # This determines Q_C1_total and Q_C2_total for the candidate children.
        q_c1_total_candidate = 0.0
        q_c2_total_candidate = 0.0
        
        for idx_in_combined, voxel_wc in enumerate(combined_voxels_world_coords):
            dist_sq_to_c1 = utils.distance_squared(voxel_wc, c1_pos_candidate)
            dist_sq_to_c2 = utils.distance_squared(voxel_wc, c2_pos_candidate)
            if dist_sq_to_c1 <= dist_sq_to_c2:
                q_c1_total_candidate += demand_of_combined_voxels_flow[idx_in_combined]
            else:
                q_c2_total_candidate += demand_of_combined_voxels_flow[idx_in_combined]
        
        if q_c1_total_candidate < constants.EPSILON or q_c2_total_candidate < constants.EPSILON:
            # This means one child would get (almost) no flow from the entire combined territory.
            # This might be a poor bifurcation unless the other child takes nearly all.
            # Forcing both to have substantial flow might be too restrictive.
            # Let it proceed if total demand is met.
            if abs(q_c1_total_candidate + q_c2_total_candidate - total_demand_of_combined_territory) > constants.EPSILON * total_demand_of_combined_territory:
                 logger.warning(f"Demand conservation issue in child assignment: sum_child_Q={q_c1_total_candidate+q_c2_total_candidate:.2e}, total_demand={total_demand_of_combined_territory:.2e}")
                 continue # Skip if total demand not conserved by split

        # --- 3. Calculate radii for c1, c2 based on their TOTAL flows ---
        r_c1_candidate = k_murray_factor * (q_c1_total_candidate ** (1.0 / murray_exponent)) if q_c1_total_candidate > constants.EPSILON else min_radius
        r_c2_candidate = k_murray_factor * (q_c2_total_candidate ** (1.0 / murray_exponent)) if q_c2_total_candidate > constants.EPSILON else min_radius
        r_c1_candidate = max(min_radius, r_c1_candidate)
        r_c2_candidate = max(min_radius, r_c2_candidate)

        # --- 4. Calculate loss for this candidate bifurcation (for the two new child segments) ---
        # Flows used here are q_c1_total_candidate and q_c2_total_candidate
        current_loss = calculate_bifurcation_loss(
            parent_pos, # Bifurcation point
            c1_pos_candidate, r_c1_candidate, q_c1_total_candidate,
            c2_pos_candidate, r_c2_candidate, q_c2_total_candidate,
            config
        )

        if current_loss < min_loss_found:
            min_loss_found = current_loss
            best_bifurcation_params = (c1_pos_candidate.copy(), r_c1_candidate, q_c1_total_candidate,
                                       c2_pos_candidate.copy(), r_c2_candidate, q_c2_total_candidate,
                                       min_loss_found)

    if best_bifurcation_params:
        logger.info(f"Optimal bifurcation for {parent_id} (supplying combined territory) chosen (Loss {best_bifurcation_params[6]:.3e}): "
                    f"C1 (R={best_bifurcation_params[1]:.4f}, Q_total={best_bifurcation_params[2]:.2e}), "
                    f"C2 (R={best_bifurcation_params[4]:.4f}, Q_total={best_bifurcation_params[5]:.2e})")
        return best_bifurcation_params
    else:
        logger.debug(f"No suitable bifurcation found for terminal {parent_id} to supply combined territory.")
        return None

# The __main__ test block in energy_model.py would need to be updated to call this new function
# and provide mock data for a "combined_territory".


def find_optimal_bifurcation_for_new_region(
    parent_terminal_gbo_data: object, 
    new_growth_region_voxel_indices_flat: np.ndarray, 
    tissue_data: dict,
    config: dict,
    k_murray_factor: float,
    murray_exponent: float
) -> Optional[Tuple[np.ndarray, float, float, np.ndarray, float, float, float]]:
    """
    Searches for an optimal bifurcation for the parent_terminal to supply a *new growth region (Ri,p)*.
    (Full implementation as previously provided)
    """
    parent_id = parent_terminal_gbo_data.id
    parent_pos = parent_terminal_gbo_data.pos
    logger.debug(f"Finding optimal bifurcation for terminal {parent_id} at {parent_pos} to supply a new "
                 f"growth region of {len(new_growth_region_voxel_indices_flat)} voxels.")

    if len(new_growth_region_voxel_indices_flat) == 0:
        logger.debug(f"Terminal {parent_id}: New growth region is empty. No bifurcation.")
        return None

    new_growth_voxels_world_coords = tissue_data['world_coords_flat'][new_growth_region_voxel_indices_flat]
    
    demand_map_3d_indices = tissue_data['voxel_indices_flat'][new_growth_region_voxel_indices_flat]
    demand_per_new_growth_voxel = tissue_data['metabolic_demand_map'][
        demand_map_3d_indices[:,0],
        demand_map_3d_indices[:,1],
        demand_map_3d_indices[:,2]
    ] # This is q_met per voxel, not q_met * dV yet
    
    # Multiply by voxel volume to get demand (flow rate)
    demand_of_new_growth_voxels = demand_per_new_growth_voxel * tissue_data['voxel_volume']
    total_demand_of_new_region = np.sum(demand_of_new_growth_voxels)

    if total_demand_of_new_region < constants.EPSILON:
        logger.debug(f"Terminal {parent_id}: Total demand in the new growth region is negligible. No bifurcation.")
        return None

    num_candidate_location_sets = config_manager.get_param(config, "gbo_growth.bifurcation_candidate_points", 10)
    min_seg_len = config_manager.get_param(config, "vascular_properties.min_segment_length", 0.1)
    min_radius = config_manager.get_param(config, "vascular_properties.min_radius", constants.MIN_VESSEL_RADIUS_MM)

    best_bifurcation_params = None
    min_loss_found = np.inf

    if len(new_growth_voxels_world_coords) < 2 and len(new_growth_voxels_world_coords) > 0: # Handle single voxel new region
         logger.debug(f"New growth region for {parent_id} has only {len(new_growth_voxels_world_coords)} voxel(s). Treating as extension.")
         # This case should ideally be handled by "extension" logic in vascular_growth.py
         # For now, find_optimal_bifurcation will attempt to make one child supply it.
    elif len(new_growth_voxels_world_coords) == 0: # Should be caught earlier
        return None


    for i in range(num_candidate_location_sets):
        c1_pos_candidate, c2_pos_candidate = None, None
        if len(new_growth_voxels_world_coords) >= 2:
            if SKLEARN_AVAILABLE and KMeans is not None:
                try:
                    # Ensure n_clusters is not more than n_samples
                    n_clust = min(2, len(new_growth_voxels_world_coords))
                    if n_clust < 2 : # Not enough for two distinct clusters
                        idx = np.random.choice(len(new_growth_voxels_world_coords), 1)[0]
                        c1_pos_candidate = new_growth_voxels_world_coords[idx]
                        # Create a dummy c2 for calculation, it will get ~0 flow from new region
                        c2_pos_candidate = c1_pos_candidate + utils.normalize_vector(np.random.rand(3)-0.5) * min_seg_len
                    else:
                        kmeans = KMeans(n_clusters=n_clust, random_state=i, n_init='auto').fit(new_growth_voxels_world_coords)
                        c1_pos_candidate = kmeans.cluster_centers_[0]
                        c2_pos_candidate = kmeans.cluster_centers_[1] if n_clust == 2 else kmeans.cluster_centers_[0] + utils.normalize_vector(np.random.rand(3)-0.5) * min_seg_len

                except Exception as e_km: # Catch any Kmeans error
                    logger.warning(f"KMeans clustering failed for bifurcation candidates (iter {i}): {e_km}. Falling back to random points.")
                    # Fallback to random points from the new growth region
                    indices = np.random.choice(len(new_growth_voxels_world_coords), 2, replace=len(new_growth_voxels_world_coords) < 2)
                    c1_pos_candidate = new_growth_voxels_world_coords[indices[0]]
                    c2_pos_candidate = new_growth_voxels_world_coords[indices[1]]

            else: # SKLEARN_AVAILABLE is False or KMeans is None (ImportError)
                if i == 0 : logger.warning("Scikit-learn not found. Using random points for bifurcation candidates.")
                indices = np.random.choice(len(new_growth_voxels_world_coords), 2, replace=len(new_growth_voxels_world_coords) < 2)
                c1_pos_candidate = new_growth_voxels_world_coords[indices[0]]
                c2_pos_candidate = new_growth_voxels_world_coords[indices[1]]
        
        elif len(new_growth_voxels_world_coords) == 1:
            c1_pos_candidate = new_growth_voxels_world_coords[0]
            c2_pos_candidate = c1_pos_candidate + utils.normalize_vector(np.random.rand(3)-0.5) * min_seg_len # Dummy c2

        if c1_pos_candidate is None or c2_pos_candidate is None : continue # Should not happen with fallbacks

        if utils.distance(parent_pos, c1_pos_candidate) < min_seg_len or \
           utils.distance(parent_pos, c2_pos_candidate) < min_seg_len or \
           (utils.distance(c1_pos_candidate, c2_pos_candidate) < min_seg_len and not np.allclose(c1_pos_candidate, c2_pos_candidate)):
            continue

        q_c1_candidate_from_new = 0.0
        q_c2_candidate_from_new = 0.0
        
        for idx_in_new_region, voxel_wc in enumerate(new_growth_voxels_world_coords):
            dist_sq_to_c1 = utils.distance_squared(voxel_wc, c1_pos_candidate)
            dist_sq_to_c2 = utils.distance_squared(voxel_wc, c2_pos_candidate)
            if dist_sq_to_c1 <= dist_sq_to_c2:
                q_c1_candidate_from_new += demand_of_new_growth_voxels[idx_in_new_region]
            else:
                q_c2_candidate_from_new += demand_of_new_growth_voxels[idx_in_new_region]
        
        # Ensure flow is not split into practically zero for both if there's demand
        if (q_c1_candidate_from_new < constants.EPSILON and q_c2_candidate_from_new < constants.EPSILON and total_demand_of_new_region > constants.EPSILON):
            # This might happen if c1_pos and c2_pos are poorly chosen relative to demand distribution.
            # For example, if both are far from all demand points.
            # Or if total_demand_of_new_region is tiny.
            # logger.debug(f"Candidate pair {i} results in zero flow for both children from new region. Skipping.")
            continue

        r_c1_candidate = k_murray_factor * (q_c1_candidate_from_new ** (1.0 / murray_exponent)) if q_c1_candidate_from_new > constants.EPSILON else min_radius
        r_c2_candidate = k_murray_factor * (q_c2_candidate_from_new ** (1.0 / murray_exponent)) if q_c2_candidate_from_new > constants.EPSILON else min_radius
        r_c1_candidate = max(min_radius, r_c1_candidate)
        r_c2_candidate = max(min_radius, r_c2_candidate)

        current_loss = calculate_bifurcation_loss(
            parent_pos,
            c1_pos_candidate, r_c1_candidate, q_c1_candidate_from_new,
            c2_pos_candidate, r_c2_candidate, q_c2_candidate_from_new,
            config
        )

        if current_loss < min_loss_found:
            min_loss_found = current_loss
            best_bifurcation_params = (c1_pos_candidate.copy(), r_c1_candidate, q_c1_candidate_from_new,
                                       c2_pos_candidate.copy(), r_c2_candidate, q_c2_candidate_from_new,
                                       min_loss_found)

    if best_bifurcation_params:
        logger.info(f"Optimal bifurcation for {parent_id} to supply new region chosen (Loss {best_bifurcation_params[6]:.3e}): "
                    f"C1 (R={best_bifurcation_params[1]:.4f}, Q_new={best_bifurcation_params[2]:.2e}), "
                    f"C2 (R={best_bifurcation_params[4]:.4f}, Q_new={best_bifurcation_params[5]:.2e})")
        return best_bifurcation_params
    else:
        logger.debug(f"No suitable (lower loss or valid) bifurcation found for terminal {parent_id} to supply new region.")
        return None

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG) # Changed to DEBUG for more verbose test output
    
    # Mock config
    test_config = {
        "vascular_properties": {
            "blood_viscosity": 0.0035, 
            "min_segment_length": 0.01, 
            "min_radius": 0.005, 
            "k_murray_scaling_factor": 0.5, 
            "murray_law_exponent": 3.0
        },
        "gbo_growth": {
            # C_met unit: e.g. mW/mm^3 (if E_flow target unit is mW)
            # If E_flow from calculate_segment_flow_energy is in 10^-9 W (nanoWatts) when inputs are mm, Pa.s, mm^3/s
            # And we want E_met to be in same units: C_met * mm^2 * mm = C_met * mm^3
            # So C_met should be in (10^-9 W) / mm^3.
            # If paper uses C_met for W/m^3, and we use mm:
            # C_met_paper (W/m^3) * (1m/1000mm)^3 = C_met_paper * 1e-9 (W/mm^3)
            # Let's use a value that makes E_met somewhat comparable to E_flow for typical values.
            # Example from before: L=1, R=0.1, Q=0.01 -> E_flow ~ 1.1e-7 (in 10^-9 W units, so 1.1e-16 W actual)
            # E_met = C_met * pi * (0.1)^2 * 1. If C_met = 1e-5 (in 10^-9W/mm^3 units), E_met ~ 3e-7.
            "energy_coefficient_C_met_vessel_wall": 1.0e-5, # (Units: 10^-9 W / mm^3 or equivalent)
            "bifurcation_candidate_points": 50, # Increased for better testing
        }
    }
    
    # Ensure constants are properly defined or mocked for standalone execution
    class MockConstants:
        PI = np.pi
        EPSILON = 1e-10 # Slightly smaller epsilon
        DEFAULT_BLOOD_VISCOSITY = 0.0035
        # This default C_MET_VESSEL_WALL in constants.py is likely in W/m^3.
        # The config value above is what's used by the functions.
        DEFAULT_C_MET_VESSEL_WALL = 1.0e5 
        MIN_VESSEL_RADIUS_MM = 0.005
    
    # Overwrite constants only if they are not the actual imported ones (e.g. running file directly)
    if 'constants' not in globals() or not hasattr(constants, 'MIN_VESSEL_RADIUS_MM'):
        constants = MockConstants()
        print("Using MockConstants for standalone test.")


    # Test calculate_segment_flow_energy
    # Using L (mm), R (mm), Q (mm^3/s), mu (Pa.s)
    # Expected output units: 10^-9 W (nanoWatts) or Pa.mm^3/s
    l, r_test, q_test, mu_test = 1.0, 0.1, 0.01, test_config["vascular_properties"]["blood_viscosity"]
    e_flow = calculate_segment_flow_energy(l, r_test, q_test, mu_test)
    
    # c_met_val from config is assumed to be in (10^-9 W)/mm^3 to match E_flow units
    c_met_val = test_config["gbo_growth"]["energy_coefficient_C_met_vessel_wall"] 
    e_met = calculate_segment_metabolic_energy(l, r_test, c_met_val)
    logger.info(f"Test segment: L={l}mm, R={r_test}mm, Q={q_test}mm^3/s, mu={mu_test}Pa.s")
    logger.info(f"Calculated E_flow = {e_flow:.3e} (expected units: Pa.mm^3/s or 10^-9 W)")
    logger.info(f"Calculated E_met (with C_met={c_met_val:.1e}) = {e_met:.3e} (expected units: same as E_flow)")
    # Expected E_flow = (8 * 0.0035 * 1 * 0.01^2) / (pi * 0.1^4) = (2.8e-6) / (pi * 1e-4) approx 2.8e-6 / 3.14e-4 = 0.0089 Pa.mm^3/s
    # My previous manual calc was off. Let's recheck:
    # (8 * 0.0035 * 1.0 * (0.01**2)) / (np.pi * (0.1**4)) = 0.008912676...
    # E_met = 1e-5 * np.pi * (0.1**2) * 1.0 = 3.14159e-7
    # These values seem reasonable relative to each other if C_met is chosen appropriately.

    # Mock parent terminal for find_optimal_bifurcation_for_new_region
    class MockGBOIterationData: # Simplified for this test
        def __init__(self, id, pos, radius, flow): # Removed territory_demand as it's not used by this func
            self.id = id
            self.pos = np.array(pos)
            self.radius = radius
            self.flow = flow # This is flow to *existing* territory, not used by find_optimal_bifurcation_for_new_region

    parent_term = MockGBOIterationData(
        id="p_test_0", 
        pos=np.array([0.,0.,0.]), 
        radius=0.2, 
        flow=0.00 # Flow to existing territory, not directly used for optimizing *new* region supply
    )

    # Mock tissue_data for the new growth region
    num_new_growth_voxels = 50
    # These are flat indices relative to the global tissue_data arrays
    # For the test, let's assume these are the first 'num_new_growth_voxels' in a hypothetical global list
    mock_new_growth_indices_flat = np.arange(num_new_growth_voxels) 
    
    # World coords for these specific new growth voxels
    # Place them in a cluster, e.g., around [1,1,0]
    mock_new_growth_world_coords = np.random.rand(num_new_growth_voxels, 3) * 0.5 + np.array([1.0, 1.0, 0.0])
    
    # For tissue_data, we need the *full* set of domain voxels
    # For this test, we can make tissue_data['world_coords_flat'] just be these new growth voxels
    # And correspondingly for voxel_indices_flat and metabolic_demand_map
    
    mock_tissue_voxel_indices_3d = np.zeros((num_new_growth_voxels, 3), dtype=int)
    for i in range(num_new_growth_voxels): # Dummy 3D indices
        mock_tissue_voxel_indices_3d[i] = [i // 10, i % 10, 0] 

    # Demand for these new growth voxels (q_met, not q_met * dV yet)
    mock_demand_q_met_for_new_voxels = np.random.uniform(low=0.01, high=0.02, size=num_new_growth_voxels) # 1/s (q_met)
    
    # Assume a voxel volume for calculating total demand from q_met
    mock_voxel_vol = 0.001 # mm^3 (e.g., 0.1mm x 0.1mm x 0.1mm)

    # Create the metabolic_demand_map (3D) that would contain these values
    # For simplicity, make it just large enough for our dummy 3D indices
    max_indices = np.max(mock_tissue_voxel_indices_3d, axis=0)
    mock_full_metabolic_demand_map_3d = np.zeros((max_indices[0]+1, max_indices[1]+1, max_indices[2]+1))
    mock_full_metabolic_demand_map_3d[mock_tissue_voxel_indices_3d[:,0],
                                      mock_tissue_voxel_indices_3d[:,1],
                                      mock_tissue_voxel_indices_3d[:,2]] = mock_demand_q_met_for_new_voxels
    
    mock_tissue_data = {
        'world_coords_flat': mock_new_growth_world_coords, # Only contains the new growth voxels for this test
        'voxel_indices_flat': mock_tissue_voxel_indices_3d, # Corresponding 3D indices
        'metabolic_demand_map': mock_full_metabolic_demand_map_3d, # Full 3D q_met map
        'voxel_volume': mock_voxel_vol
    }
    
    bifurcation_result = find_optimal_bifurcation_for_combined_territory(
        parent_term,
        mock_new_growth_indices_flat, # These are indices into the arrays in mock_tissue_data
        mock_tissue_data,
        test_config,
        k_murray_factor=test_config["vascular_properties"]["k_murray_scaling_factor"],
        murray_exponent=test_config["vascular_properties"]["murray_law_exponent"]
    )

    if bifurcation_result:
        c1p, c1r, c1q_new, c2p, c2r, c2q_new, loss = bifurcation_result
        logger.info(f"Optimal Bifurcation Found for {parent_term.id} to supply new region:")
        logger.info(f"  Child 1: Pos={np.round(c1p,3)}, Radius={c1r:.4f}, Flow_new={c1q_new:.3e}")
        logger.info(f"  Child 2: Pos={np.round(c2p,3)}, Radius={c2r:.4f}, Flow_new={c2q_new:.3e}")
        logger.info(f"  Minimized Loss (for new segments): {loss:.3e}")
        # Verify total new flow captured
        total_new_demand_calc = np.sum(mock_demand_q_met_for_new_voxels * mock_voxel_vol)
        logger.info(f"  Sum of child flows from new region: {(c1q_new + c2q_new):.3e}")
        logger.info(f"  Total demand of new region: {total_new_demand_calc:.3e}")
        assert np.isclose(c1q_new + c2q_new, total_new_demand_calc), "Child flows do not sum to total new demand"
    else:
        logger.info(f"No optimal bifurcation found for {parent_term.id} in this test.")