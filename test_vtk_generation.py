#!/usr/bin/env python3
"""
Test VTK generation with simplified approach
"""
import os
import sys
import numpy as np
sys.path.append('/Users/c3495249/Coding/Gemini_Pro_Vasculature')

from src import io_utils, config_manager

def test_vtk_generation():
    """Test generating VTK files with current approach"""
    
    # Load config to get real data
    config = config_manager.load_config('/Users/c3495249/Coding/Gemini_Pro_Vasculature/config.yaml')
    
    # Load tissue data (simplified)
    paths = config_manager.get_param(config, "paths", {})
    gm_data, affine_gm, _ = io_utils.load_nifti_image(paths.get("gm_nifti",""))
    
    if gm_data is None or affine_gm is None:
        print("ERROR: Could not load GM data")
        return
    
    print(f"Loaded GM data: shape={gm_data.shape}, affine=\n{affine_gm}")
    
    # Create tissue data dict
    tissue_data = {
        'GM': gm_data.astype(bool),
        'affine': affine_gm
    }
    
    # Create a simple perfused mask (subset of GM)
    perfused_mask = np.zeros_like(gm_data, dtype=bool)
    # Make a small central region "perfused" for testing
    center = [s//2 for s in gm_data.shape]
    region_size = 20
    perfused_mask[
        center[0]-region_size:center[0]+region_size,
        center[1]-region_size:center[1]+region_size,
        center[2]-region_size:center[2]+region_size
    ] = gm_data[
        center[0]-region_size:center[0]+region_size,
        center[1]-region_size:center[1]+region_size,
        center[2]-region_size:center[2]+region_size
    ].astype(bool)
    
    print(f"Created perfused mask with {np.sum(perfused_mask)} voxels")
    
    # Create output directory
    test_output_dir = "/Users/c3495249/Coding/Gemini_Pro_Vasculature/test_vtk_output"
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Export VTK files
    print("Generating VTK files...")
    io_utils.export_tissue_masks_to_vtk(tissue_data, perfused_mask, test_output_dir, iteration=999)
    
    print(f"VTK files saved to: {test_output_dir}/tissue_vtk/")
    print("Files generated:")
    vtk_dir = os.path.join(test_output_dir, "tissue_vtk")
    if os.path.exists(vtk_dir):
        for f in os.listdir(vtk_dir):
            if f.endswith('.vti'):
                filepath = os.path.join(vtk_dir, f)
                size = os.path.getsize(filepath)
                print(f"  {f}: {size} bytes")

if __name__ == "__main__":
    test_vtk_generation()