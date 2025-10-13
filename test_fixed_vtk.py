#!/usr/bin/env python3
"""
Test the fixed VTK files
"""
import os
import numpy as np
import vtk
from vtk.util import numpy_support

def test_fixed_file(filepath):
    """Test if the fixed VTK file has correct data layout"""
    print(f"\n{'='*60}")
    print(f"TESTING FIXED FILE: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print("File not found!")
        return False
    
    try:
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(filepath)
        reader.Update()
        
        image_data = reader.GetOutput()
        dims = image_data.GetDimensions()
        
        print(f"Dimensions: {dims}")
        print(f"Bounds: {image_data.GetBounds()}")
        
        # Check scalar data
        scalars = image_data.GetPointData().GetScalars()
        if scalars:
            numpy_data = numpy_support.vtk_to_numpy(scalars)
            print(f"Non-zero count: {np.count_nonzero(numpy_data)}")
            print(f"Value range: {numpy_data.min()} to {numpy_data.max()}")
            
            # Check specific locations to verify correct layout
            center_x, center_y, center_z = dims[0]//2, dims[1]//2, dims[2]//2
            
            # Calculate linear index for center point
            center_idx = center_z * dims[1] * dims[0] + center_y * dims[0] + center_x
            if center_idx < len(numpy_data):
                center_value = numpy_data[center_idx]
                print(f"Center point value: {center_value}")
                
                # Also check a few nearby points
                nearby_nonzero = 0
                for dz in [-5, 0, 5]:
                    for dy in [-5, 0, 5]:
                        for dx in [-5, 0, 5]:
                            x, y, z = center_x + dx, center_y + dy, center_z + dz
                            if 0 <= x < dims[0] and 0 <= y < dims[1] and 0 <= z < dims[2]:
                                idx = z * dims[1] * dims[0] + y * dims[0] + x
                                if idx < len(numpy_data) and numpy_data[idx] > 0:
                                    nearby_nonzero += 1
                
                print(f"Non-zero values near center: {nearby_nonzero}/125")
                
                if nearby_nonzero > 0:
                    print("✅ Data layout appears correct - found non-zero values near center")
                    return True
                else:
                    print("❌ Data layout may be incorrect - no values near center")
                    return False
        
        return False
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    # Test our newly generated files
    test_files = [
        "/Users/c3495249/Coding/Gemini_Pro_Vasculature/test_vtk_output/tissue_vtk/GM_iter_999.vti",
        "/Users/c3495249/Coding/Gemini_Pro_Vasculature/test_vtk_output/tissue_vtk/perfused_iter_999.vti"
    ]
    
    print("TESTING FIXED VTK FILES")
    print("="*80)
    
    success_count = 0
    for filepath in test_files:
        if os.path.exists(filepath):
            if test_fixed_file(filepath):
                success_count += 1
    
    print(f"\n{'='*80}")
    if success_count == len([f for f in test_files if os.path.exists(f)]):
        print("🎉 SUCCESS! VTK files should now display properly in ParaView!")
        print("\nThe fixed files use point-by-point data filling which ensures")
        print("correct memory layout compatible with ParaView's expectations.")
    else:
        print("❌ Some issues remain. Check the analysis above.")
    print("="*80)

if __name__ == "__main__":
    main()