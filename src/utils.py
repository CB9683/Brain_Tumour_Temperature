# src/utils.py
import numpy as np
import random
import logging
import os
import shutil
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def set_rng_seed(seed: int):
    """Sets the random seed for Python's random, NumPy, and potentially other libraries."""
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random Number Generator seed set to: {seed}")
    # Add other libraries like TensorFlow/PyTorch if used:
    # tf.random.set_seed(seed)
    # torch.manual_seed(seed)

def get_voxel_volume_from_affine(affine: np.ndarray) -> float:
    """
    Calculates the volume of a single voxel from the NIfTI affine matrix.
    Assumes the affine matrix maps voxel coordinates to physical coordinates.
    The volume is the absolute value of the determinant of the first 3x3 submatrix.

    Args:
        affine (np.ndarray): The 4x4 affine matrix.

    Returns:
        float: The volume of a single voxel.
    """
    return abs(np.linalg.det(affine[:3, :3]))

def voxel_to_world(voxel_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Converts voxel coordinates to world (physical) coordinates.

    Args:
        voxel_coords (np.ndarray): A (N, 3) array of voxel coordinates (i, j, k).
        affine (np.ndarray): The 4x4 NIfTI affine matrix.

    Returns:
        np.ndarray: A (N, 3) array of world coordinates (x, y, z).
    """
    voxel_coords = np.asarray(voxel_coords)
    if voxel_coords.ndim == 1:
        voxel_coords = voxel_coords.reshape(1, -1)
    
    # Add homogeneous coordinate
    homogeneous_coords = np.hstack((voxel_coords, np.ones((voxel_coords.shape[0], 1))))
    
    # Apply affine transformation
    world_coords_homogeneous = homogeneous_coords @ affine.T
    
    return world_coords_homogeneous[:, :3]

def world_to_voxel(world_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Converts world (physical) coordinates to voxel coordinates.
    Uses the inverse of the affine matrix. Resulting voxel coordinates might be fractional.

    Args:
        world_coords (np.ndarray): A (N, 3) array of world coordinates (x, y, z).
        affine (np.ndarray): The 4x4 NIfTI affine matrix.

    Returns:
        np.ndarray: A (N, 3) array of voxel coordinates (i, j, k).
    """
    world_coords = np.asarray(world_coords)
    if world_coords.ndim == 1:
        world_coords = world_coords.reshape(1, -1)

    # Add homogeneous coordinate
    homogeneous_coords = np.hstack((world_coords, np.ones((world_coords.shape[0], 1))))
    
    # Invert affine matrix
    inv_affine = np.linalg.inv(affine)
    
    # Apply inverse affine transformation
    voxel_coords_homogeneous = homogeneous_coords @ inv_affine.T
    
    return voxel_coords_homogeneous[:, :3]

def distance_squared(p1: np.ndarray, p2: np.ndarray) -> float:
    """Computes the squared Euclidean distance between two 3D points."""
    return np.sum((p1 - p2)**2)

def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Computes the Euclidean distance between two 3D points."""
    return np.sqrt(np.sum((p1 - p2)**2))

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalizes a vector."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def create_output_directory(base_dir: str, sim_name: str = "gbo_sim", timestamp: bool = True) -> str:
    """
    Creates a unique output directory.
    Example: base_dir/YYYYMMDD_HHMMSS_sim_name or base_dir/sim_name
    """
    from datetime import datetime
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{ts}_{sim_name}"
    else:
        dir_name = sim_name
    
    full_path = os.path.join(base_dir, dir_name)
    
    if os.path.exists(full_path):
        # Option 1: Overwrite (dangerous)
        # shutil.rmtree(full_path) 
        # Option 2: Add a suffix
        count = 1
        new_full_path = f"{full_path}_{count}"
        while os.path.exists(new_full_path):
            count += 1
            new_full_path = f"{full_path}_{count}"
        full_path = new_full_path
        logger.warning(f"Output directory {os.path.join(base_dir, dir_name)} existed. Using {full_path} instead.")

    os.makedirs(full_path, exist_ok=True)
    logger.info(f"Created output directory: {full_path}")
    return full_path

def is_voxel_in_bounds(voxel_coord: np.ndarray, shape: Tuple[int, ...]) -> bool:
    """Checks if a voxel coordinate is within the bounds of a given shape."""
    voxel_coord = np.asarray(voxel_coord) # Ensure it's a numpy array
    if voxel_coord.ndim == 0 or voxel_coord.shape[0] != len(shape): # Check for scalar or mismatched dimensions
        return False
    return all(0 <= voxel_coord[d] < shape[d] for d in range(len(shape)))



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Test RNG seed
    set_rng_seed(123)
    print(f"Random float after seed 123: {random.random()}")
    set_rng_seed(123)
    print(f"Random float after seed 123 again: {random.random()}")
    print(f"Numpy random array after seed 123: {np.random.rand(3)}")
    set_rng_seed(123)
    print(f"Numpy random array after seed 123 again: {np.random.rand(3)}")

    # Test affine transformations
    # A typical NIfTI affine for 1mm isotropic voxels, origin at corner
    dummy_affine = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    # A more realistic affine (e.g. -1mm x, 1mm y, 1mm z, with an offset)
    # RAS orientation: X points Left to Right, Y Posterior to Anterior, Z Inferior to Superior
    # If voxel (0,0,0) is at world (-90, 90, -120) and voxels are 1mm:
    # This means i maps to -X, j maps to +Y, k maps to +Z (LPI orientation for data array)
    # If data array is stored radiological (i from R->L), then first column of affine is positive.
    # Assuming standard interpretation (voxel index increases, world coordinate increases along axis basis vector)
    # Let's use a simple affine for testing:
    test_affine = np.array([
        [-1.0, 0.0, 0.0, 100.0],  # Voxel i -> World -x direction; (0,0,0) maps to x=100
        [0.0, 1.0, 0.0, -50.0],  # Voxel j -> World +y direction; (0,0,0) maps to y=-50
        [0.0, 0.0, 2.0, -20.0],  # Voxel k -> World +z direction, 2mm thick; (0,0,0) maps to z=-20
        [0.0, 0.0, 0.0, 1.0]
    ])

    print(f"Voxel volume for test_affine: {get_voxel_volume_from_affine(test_affine)} mm^3 (expected 2)")

    voxel_pts = np.array([[0,0,0], [10,20,5]])
    world_pts = voxel_to_world(voxel_pts, test_affine)
    print(f"Voxel points:\n{voxel_pts}")
    print(f"Converted to World points:\n{world_pts}")
    # Expected for [0,0,0]: [100, -50, -20]
    # Expected for [10,20,5]: [-1*10+100, 1*20-50, 2*5-20] = [90, -30, -10]

    reconverted_voxel_pts = world_to_voxel(world_pts, test_affine)
    print(f"Reconverted to Voxel points:\n{reconverted_voxel_pts}")
    assert np.allclose(voxel_pts, reconverted_voxel_pts), "Voxel-World-Voxel conversion failed"

    # Test distance
    p1 = np.array([0,0,0])
    p2 = np.array([3,4,0])
    print(f"Distance squared between {p1} and {p2}: {distance_squared(p1, p2)} (expected 25)")
    print(f"Distance between {p1} and {p2}: {distance(p1, p2)} (expected 5)")

    # Test output directory
    base_output = "temp_test_output"
    os.makedirs(base_output, exist_ok=True)
    path1 = create_output_directory(base_output, "my_sim")
    path2 = create_output_directory(base_output, "my_sim") # Should create my_sim_1
    print(f"Path1: {path1}")
    print(f"Path2: {path2}")
    shutil.rmtree(base_output)
    print("Cleaned up temp_test_output directory.")

def get_random_point_in_mask(mask: np.ndarray, affine: np.ndarray) -> Optional[np.ndarray]:
    """
    Gets a random point within a boolean mask in world coordinates.
    
    Args:
        mask: Boolean 3D array
        affine: 4x4 affine matrix
        
    Returns:
        Random world coordinate point within mask, or None if mask is empty
    """
    indices = np.array(np.where(mask)).T
    if indices.shape[0] == 0:
        return None
    
    random_idx = np.random.choice(indices.shape[0])
    voxel_coord = indices[random_idx]
    world_coord = voxel_to_world(voxel_coord, affine)[0]
    return world_coord