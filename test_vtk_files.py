#!/usr/bin/env python3
"""
Test script to verify VTK files are properly readable
"""
import os
import sys
import vtk
from vtk.util import numpy_support

def test_vti_file(filepath):
    """Test if a VTI file is properly readable and contains actual data"""
    print(f"\n=== Testing {os.path.basename(filepath)} ===")
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return False
    
    try:
        # Read VTI file
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(filepath)
        reader.Update()
        
        image_data = reader.GetOutput()
        
        # Get basic information
        dimensions = image_data.GetDimensions()
        spacing = image_data.GetSpacing()
        origin = image_data.GetOrigin()
        bounds = image_data.GetBounds()
        
        print(f"Dimensions: {dimensions}")
        print(f"Spacing: {spacing}")
        print(f"Origin: {origin}")
        print(f"Bounds: {bounds}")
        
        # Get scalar data
        point_data = image_data.GetPointData()
        if point_data.GetNumberOfArrays() > 0:
            scalar_array = point_data.GetArray(0)
            array_name = scalar_array.GetName()
            data_range = scalar_array.GetRange()
            num_points = scalar_array.GetNumberOfTuples()
            
            # Convert to numpy to check for actual data
            numpy_array = numpy_support.vtk_to_numpy(scalar_array)
            non_zero_count = (numpy_array > 0).sum()
            
            print(f"Scalar array name: {array_name}")
            print(f"Data range: {data_range}")
            print(f"Number of points: {num_points}")
            print(f"Non-zero values: {non_zero_count} ({100*non_zero_count/len(numpy_array):.1f}%)")
            
            if non_zero_count > 0:
                print("✅ SUCCESS: File contains actual data")
                return True
            else:
                print("⚠️  WARNING: File contains only zeros")
                return False
        else:
            print("❌ ERROR: No scalar data found")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: Failed to read VTI file: {e}")
        return False

def main():
    # Test the most recent VTK files
    base_dir = "/Users/c3495249/Coding/Gemini_Pro_Vasculature/output/simulation_results/20250730_160324_mida_all_89_terminals_gbo/tissue_vtk"
    
    if not os.path.exists(base_dir):
        print(f"ERROR: Directory not found: {base_dir}")
        return
    
    test_files = [
        "GM_iter_0.vti",
        "GM_iter_2.vti", 
        "WM_iter_0.vti",
        "WM_iter_2.vti",
        "perfused_iter_2.vti"
    ]
    
    print("Testing VTK files for ParaView compatibility...")
    print("=" * 60)
    
    success_count = 0
    for filename in test_files:
        filepath = os.path.join(base_dir, filename)
        if test_vti_file(filepath):
            success_count += 1
    
    print(f"\n" + "=" * 60)
    print(f"SUMMARY: {success_count}/{len(test_files)} files passed tests")
    
    if success_count == len(test_files):
        print("🎉 All VTK files should now display properly in ParaView!")
        print("\nInstructions for ParaView:")
        print("1. Open ParaView")
        print("2. File -> Open -> Navigate to tissue_vtk folder")
        print("3. Select multiple .vti files and click OK")
        print("4. Click 'Apply' in the Properties panel")
        print("5. Use the 'eye' icon to toggle visibility")
        print("6. For masks: Set representation to 'Volume' or 'Surface'")
    else:
        print("❌ Some files may still have issues")

if __name__ == "__main__":
    main()