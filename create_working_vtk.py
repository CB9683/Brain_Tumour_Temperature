#!/usr/bin/env python3
"""
Create a working VTK file using the exact same approach as our tissue export
but with controlled test data
"""
import os
import sys
import numpy as np
sys.path.append('/Users/c3495249/Coding/Gemini_Pro_Vasculature')

from src import io_utils

def create_test_data_and_export():
    """Create test data and export using our exact method"""
    
    print("Creating controlled test data...")
    
    # Create a simple 3D volume with known structure
    dims = [100, 120, 80]  # Different dimensions to test orientation
    
    # Create test masks
    gm_mask = np.zeros(dims, dtype=bool)
    perfused_mask = np.zeros(dims, dtype=bool)
    
    # Create GM as a large central region
    center = [dims[0]//2, dims[1]//2, dims[2]//2]
    for z in range(dims[2]):
        for y in range(dims[1]):
            for x in range(dims[0]):
                # Large spherical GM region
                dist = np.sqrt((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2)
                if dist <= 30:
                    gm_mask[x, y, z] = True
                
                # Smaller perfused region at center
                if dist <= 10:
                    perfused_mask[x, y, z] = True
    
    print(f"Created GM mask with {np.sum(gm_mask)} voxels")
    print(f"Created perfused mask with {np.sum(perfused_mask)} voxels")
    
    # Create tissue data dict with simple affine
    affine = np.eye(4)
    affine[0,0] = affine[1,1] = affine[2,2] = 1.0  # 1mm spacing
    
    tissue_data = {
        'GM': gm_mask,
        'WM': np.zeros(dims, dtype=bool),  # Empty WM for simplicity
        'affine': affine
    }
    
    # Export using our method
    output_dir = "/Users/c3495249/Coding/Gemini_Pro_Vasculature/controlled_test_vtk"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Exporting VTK files using our method...")
    io_utils.export_tissue_masks_to_vtk(tissue_data, perfused_mask, output_dir, iteration=0)
    
    # Verify the files were created
    vtk_dir = os.path.join(output_dir, "tissue_vtk")
    if os.path.exists(vtk_dir):
        files = [f for f in os.listdir(vtk_dir) if f.endswith('.vti')]
        print(f"Created files: {files}")
        
        # Return path to GM file for testing
        gm_file = os.path.join(vtk_dir, "GM_iter_0.vti")
        if os.path.exists(gm_file):
            return gm_file
    
    return None

def test_controlled_file(filepath):
    """Test the controlled VTK file"""
    if not filepath or not os.path.exists(filepath):
        print("File not found!")
        return
    
    print(f"\nTesting controlled file: {os.path.basename(filepath)}")
    
    try:
        import vtk
        from vtk.util import numpy_support
        
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(filepath)
        reader.Update()
        
        image_data = reader.GetOutput()
        dims = image_data.GetDimensions()
        
        print(f"Dimensions: {dims}")
        
        scalars = image_data.GetPointData().GetScalars()
        if scalars:
            numpy_data = numpy_support.vtk_to_numpy(scalars)
            print(f"Total voxels: {len(numpy_data)}")
            print(f"Non-zero count: {np.count_nonzero(numpy_data)}")
            
            # Check center point - we know this should be 1
            center_x, center_y, center_z = dims[0]//2, dims[1]//2, dims[2]//2
            center_idx = center_z * dims[1] * dims[0] + center_y * dims[0] + center_x
            
            if center_idx < len(numpy_data):
                center_value = numpy_data[center_idx]
                print(f"Center point ({center_x}, {center_y}, {center_z}) value: {center_value}")
                
                if center_value > 0:
                    print("✅ SUCCESS: Center point has correct value!")
                    print("This file should display properly in ParaView")
                    return True
                else:
                    print("❌ PROBLEM: Center point is 0, should be 1")
                    
                    # Find where the actual data is
                    nonzero_indices = np.nonzero(numpy_data)[0]
                    if len(nonzero_indices) > 0:
                        first_nonzero_idx = nonzero_indices[0]
                        # Convert back to 3D coordinates
                        z_coord = first_nonzero_idx // (dims[1] * dims[0])
                        y_coord = (first_nonzero_idx % (dims[1] * dims[0])) // dims[0]
                        x_coord = first_nonzero_idx % dims[0]
                        print(f"First non-zero at linear index {first_nonzero_idx}")
                        print(f"Which corresponds to 3D coordinates: ({x_coord}, {y_coord}, {z_coord})")
                        print("Data layout appears to be incorrect")
                    
                    return False
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("CREATING CONTROLLED TEST FOR VTK EXPORT")
    print("="*80)
    
    # Create and export controlled test data
    test_file = create_test_data_and_export()
    
    # Test the result
    if test_file:
        success = test_controlled_file(test_file)
        
        print(f"\n{'='*80}")
        if success:
            print("🎉 VTK export is working correctly!")
            print(f"Test this file in ParaView: {test_file}")
        else:
            print("❌ VTK export still has data layout issues")
            print("Need to investigate the memory ordering further")
        print("="*80)

if __name__ == "__main__":
    main()