# src/constants.py
import numpy as np

# Physical Constants
PI = np.pi
# Blood viscosity (e.g., in Pa.s). Typical human blood viscosity is 3-4 cP (0.003-0.004 Pa.s).
# Ensure consistency with pressure (Pa) and length (m or mm) units.
# If length in mm, pressure in Pa, Q in mm^3/s, then viscosity in Pa.s.
# Example: 0.0035 Pa.s
DEFAULT_BLOOD_VISCOSITY = 3.5e-3  # Pa.s

# Default metabolic rates
# These are example values. Units should be carefully considered for flow calculation.
# Let's assume units of: ml_blood / s / ml_tissue (volume flow rate per unit volume of tissue)
# Conversion from common literature values (e.g., ml_blood / min / 100g_tissue):
# Assume tissue density ~ 1 g/ml. So 100g_tissue ~ 100ml_tissue.
# Example: GM 60 ml/min/100g = 60 ml/min/100ml = 1 ml/min/ml_tissue = (1/60) ml/s/ml_tissue ~= 0.0167 s^-1
Q_MET_GM_PER_ML = 0.0167  # 1/s (equivalent to ml_blood / s / ml_tissue)
Q_MET_WM_PER_ML = 0.0056  # 1/s (approx 20-25 ml/min/100g)
Q_MET_TUMOR_RIM_PER_ML = 0.0434 # 1/s (matches config.yaml tumor_rim rate)
Q_MET_TUMOR_CORE_PER_ML = 0.003 # 1/s (matches config.yaml tumor_core rate)
Q_MET_CSF_PER_ML = 0.0 # No metabolic demand for CSF

# GBO Parameters
MURRAY_LAW_EXPONENT = 3.0 # For r_parent^gamma = sum(r_child_i^gamma)
# C_met for metabolic maintenance cost: E_metabolic = C_met * PI * r^2 * L
# Units must be consistent with E_flow. E_flow is in Joules if Q is m^3/s, P in Pa, L in m, r in m.
# E_flow = (Pressure_drop * Q) * time_implicit = ( (8 * mu * L * Q) / (PI * r^4) ) * Q
# So E_flow has units of Power (Energy/time). For the sum, it's more like total power dissipation.
# E_metabolic should also be in units of Power.
# C_met * PI * r^2 * L => C_met units = Power / Length^3 = Power / Volume
# (e.g., W/m^3). This represents volumetric metabolic power density of vessel wall.
DEFAULT_C_MET_VESSEL_WALL = 1.0e-1 # mW/mm^3 (matches config.yaml energy_coefficient_C_met_vessel_wall)

# Initial small flow for new terminals to seed Voronoi calculation (e.g., mm^3/s or m^3/s)
# Must be > 0. Let's use mm^3/s if coordinates are in mm.
INITIAL_TERMINAL_FLOW_Q = 1.0e-4 # mm^3/s (matches config.yaml initial_terminal_flow)

# Perfusion Model Parameters
# Tissue permeability K (e.g., in mm^2 or m^2). This is for Darcy's Law: v = -(K/mu) * grad(P)
# For brain tissue, K is very low. Values can range from 10^-12 to 10^-18 m^2.
# Let's use mm, so if K is in mm^2:
DEFAULT_TISSUE_PERMEABILITY_GM = 1e-7 # mm^2 (Placeholder, highly dependent on tissue type)
DEFAULT_TISSUE_PERMEABILITY_WM = 5e-8  # mm^2 (Placeholder)
# Coupling coefficient beta for Q_terminal = beta * (P_vessel_terminal - P_tissue_at_terminal)
# Units: (mm^3/s) / Pa if P is in Pa and Q in mm^3/s.
DEFAULT_COUPLING_BETA = 1e-7 # mm^3 / (s * Pa) (Placeholder)

# Simulation control
DEFAULT_VOXEL_SIZE_MM = 1.0 # Default isotropic voxel size in mm, if not from NIfTI

# Small epsilon for numerical stability
EPSILON = 1e-9


MIN_VESSEL_RADIUS_MM = 0.004 # mm (4 microns, matches config.yaml min_radius)