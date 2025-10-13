#!/usr/bin/env python3
"""
Test coordinate transformation in VTK export
"""
import os
import sys
import numpy as np
sys.path.append('/Users/c3495249/Coding/Gemini_Pro_Vasculature')

from src import io_utils, config_manager

def test_with_and_without_transform():
    """Test VTK export with and without coordinate transformation"""
    
    # Load config to get real data
    config = config_manager.load_config('/Users/c3495249/Coding/Gemini_Pro_Vasculature/config.yaml')
    
    # Load tissue data (simplified - just GM for testing)
    paths = config_manager.get_param(config, "paths", {})
    gm_data, affine_gm, _ = io_utils.load_nifti_image(paths.get("gm_nifti",""))
    
    if gm_data is None or affine_gm is None:
        print("ERROR: Could not load GM data")
        return
    
    print(f"Original GM data shape: {gm_data.shape}")
    print(f"Original affine:\n{affine_gm}")
    
    # Create tissue data dict
    tissue_data = {
        'GM': gm_data.astype(bool),
        'affine': affine_gm
    }
    
    # Create a small test perfused mask 
    perfused_mask = np.zeros_like(gm_data, dtype=bool)
    center = [s//2 for s in gm_data.shape]
    region_size = 10
    perfused_mask[
        center[0]-region_size:center[0]+region_size,
        center[1]-region_size:center[1]+region_size,
        center[2]-region_size:center[2]+region_size
    ] = gm_data[
        center[0]-region_size:center[0]+region_size,
        center[1]-region_size:center[1]+region_size,
        center[2]-region_size:center[2]+region_size
    ].astype(bool)
    
    print(f"Created test perfused mask with {np.sum(perfused_mask)} voxels")
    
    # Create output directories
    base_dir = "/Users/c3495249/Coding/Gemini_Pro_Vasculature/transform_test"
    os.makedirs(base_dir, exist_ok=True)
    
    # Test WITHOUT transformation
    no_transform_dir = os.path.join(base_dir, "no_transform")
    os.makedirs(no_transform_dir, exist_ok=True)
    print("\n=== EXPORTING WITHOUT TRANSFORMATION ===")
    io_utils.export_tissue_masks_to_vtk(tissue_data, perfused_mask, no_transform_dir, 
                                       iteration=1, apply_coordinate_transform=False)
    
    # Test WITH transformation  
    with_transform_dir = os.path.join(base_dir, "with_transform")
    os.makedirs(with_transform_dir, exist_ok=True)
    print("\n=== EXPORTING WITH TRANSFORMATION ===")
    io_utils.export_tissue_masks_to_vtk(tissue_data, perfused_mask, with_transform_dir, 
                                       iteration=1, apply_coordinate_transform=True)
    
    # Analyze the results
    print(f"\n{'='*80}")
    print("RESULTS:")
    print(f"Files without transformation: {no_transform_dir}/tissue_vtk/")
    print(f"Files with transformation: {with_transform_dir}/tissue_vtk/")
    print()
    print("COORDINATE TRANSFORMATION APPLIED:")
    print("- X stays X (Right -> Right)")  
    print("- Y becomes -Z (Anterior -> Down)")
    print("- Z becomes Y (Superior -> Forward)")
    print()
    print("This matches the transformation used in vessel VTP exports")
    print("and should align tissue masks with vessel data in ParaView.")
    print(f"{'='*80}")
    
    return with_transform_dir

def analyze_transform_results(transform_dir):
    """Analyze the transformed VTK files"""
    vtk_dir = os.path.join(transform_dir, "tissue_vtk")
    if not os.path.exists(vtk_dir):
        print("Transform directory not found")
        return
    
    print(f"\n=== ANALYZING TRANSFORMED FILES ===")
    
    import vtk
    from vtk.util import numpy_support
    
    gm_file = os.path.join(vtk_dir, "GM_iter_1.vti")
    if os.path.exists(gm_file):
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(gm_file)
        reader.Update()
        
        image_data = reader.GetOutput()
        print(f"Transformed dimensions: {image_data.GetDimensions()}")
        print(f"Transformed spacing: {image_data.GetSpacing()}")
        print(f"Transformed origin: {image_data.GetOrigin()}")
        print(f"Transformed bounds: {image_data.GetBounds()}")
        
        # Check that data is present
        scalars = image_data.GetPointData().GetScalars()
        if scalars:
            numpy_data = numpy_support.vtk_to_numpy(scalars)
            print(f"Non-zero voxels: {np.count_nonzero(numpy_data)}")
            print("✅ Transformation applied successfully")
        else:
            print("❌ No scalar data found")

def main():
    print("TESTING COORDINATE TRANSFORMATION IN VTK EXPORT")
    print("="*80)
    
    # Test both versions
    transform_dir = test_with_and_without_transform()
    
    # Analyze results
    analyze_transform_results(transform_dir)
    
    print(f"\n{'='*80}")
    print("NEXT STEPS:")
    print("1. Load the transformed VTK files in ParaView")
    print("2. Load your vessel VTP files in the same ParaView session")
    print("3. They should now be properly aligned!")
    print("4. Use Contour filter (value 0.5) for surface rendering of masks")
    print("="*80)

if __name__ == "__main__":
    main()