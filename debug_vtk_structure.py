#!/usr/bin/env python3
"""
Debug VTK file structure to understand why ParaView shows only cubes
"""
import os
import numpy as np
import vtk
from vtk.util import numpy_support

def analyze_vti_file(filepath):
    """Detailed analysis of VTI file structure"""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found")
        return
    
    try:
        # Read the file
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(filepath)
        reader.Update()
        
        image_data = reader.GetOutput()
        
        # Basic structure info
        print(f"VTK Object Type: {type(image_data)}")
        print(f"Dimensions: {image_data.GetDimensions()}")
        print(f"Number of Points: {image_data.GetNumberOfPoints()}")
        print(f"Number of Cells: {image_data.GetNumberOfCells()}")
        print(f"Spacing: {image_data.GetSpacing()}")
        print(f"Origin: {image_data.GetOrigin()}")
        print(f"Bounds: {image_data.GetBounds()}")
        print(f"Extent: {image_data.GetExtent()}")
        
        # Point data analysis
        point_data = image_data.GetPointData()
        print(f"\nPoint Data Arrays: {point_data.GetNumberOfArrays()}")
        
        for i in range(point_data.GetNumberOfArrays()):
            array = point_data.GetArray(i)
            array_name = array.GetName()
            data_type = array.GetDataType()
            num_tuples = array.GetNumberOfTuples()
            num_components = array.GetNumberOfComponents()
            data_range = array.GetRange()
            
            print(f"  Array {i}: '{array_name}'")
            print(f"    Type: {data_type} ({array.GetDataTypeAsString()})")
            print(f"    Tuples: {num_tuples}, Components: {num_components}")
            print(f"    Range: {data_range}")
            
            # Convert to numpy for analysis
            numpy_array = numpy_support.vtk_to_numpy(array)
            print(f"    Numpy shape: {numpy_array.shape}")
            print(f"    Numpy dtype: {numpy_array.dtype}")
            print(f"    Non-zero count: {np.count_nonzero(numpy_array)}")
            print(f"    Unique values: {len(np.unique(numpy_array))}")
            
            # Sample some values
            if len(numpy_array) > 0:
                print(f"    First 10 values: {numpy_array.flat[:10]}")
                if np.any(numpy_array):
                    nonzero_indices = np.nonzero(numpy_array)
                    if len(nonzero_indices[0]) > 0:
                        print(f"    First nonzero at index: {nonzero_indices[0][0]}")
                        print(f"    First nonzero value: {numpy_array.flat[nonzero_indices[0][0]]}")
        
        # Cell data analysis
        cell_data = image_data.GetCellData()
        print(f"\nCell Data Arrays: {cell_data.GetNumberOfArrays()}")
        
        # Check if scalars are properly set
        scalars = point_data.GetScalars()
        if scalars:
            print(f"\nActive Scalars: '{scalars.GetName()}'")
        else:
            print(f"\nWARNING: No active scalars set!")
            
    except Exception as e:
        print(f"ERROR analyzing file: {e}")
        import traceback
        traceback.print_exc()

def create_test_vti():
    """Create a simple test VTI file that should work in ParaView"""
    print(f"\n{'='*60}")
    print("CREATING TEST VTI FILE")
    print(f"{'='*60}")
    
    # Create simple test data - a sphere
    dims = [50, 50, 50]
    center = [25, 25, 25]
    radius = 15
    
    # Create image data
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(dims)
    image_data.SetSpacing([1.0, 1.0, 1.0])
    image_data.SetOrigin([0.0, 0.0, 0.0])
    
    # Create scalar data
    num_points = dims[0] * dims[1] * dims[2]
    scalars = vtk.vtkFloatArray()
    scalars.SetName("test_data")
    scalars.SetNumberOfTuples(num_points)
    
    # Fill with sphere data
    for k in range(dims[2]):
        for j in range(dims[1]):
            for i in range(dims[0]):
                idx = k * dims[1] * dims[0] + j * dims[0] + i
                dist = np.sqrt((i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2)
                value = 1.0 if dist <= radius else 0.0
                scalars.SetValue(idx, value)
    
    # Set as point data
    image_data.GetPointData().SetScalars(scalars)
    
    # Write test file
    test_file = "/Users/c3495249/Coding/Gemini_Pro_Vasculature/test_sphere.vti"
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(test_file)
    writer.SetInputData(image_data)
    writer.Write()
    
    print(f"Created test file: {test_file}")
    print("This should display as a sphere in ParaView")
    
    return test_file

def main():
    # Analyze our problematic files
    base_dir = "/Users/c3495249/Coding/Gemini_Pro_Vasculature/output/simulation_results/20250730_160324_mida_all_89_terminals_gbo/tissue_vtk"
    
    problem_files = [
        "GM_iter_2.vti",
        "perfused_iter_2.vti"
    ]
    
    print("DEBUGGING VTK FILE STRUCTURE ISSUES")
    print("="*80)
    
    for filename in problem_files:
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            analyze_vti_file(filepath)
    
    # Create a test file that should work
    test_file = create_test_vti()
    analyze_vti_file(test_file)
    
    print(f"\n{'='*80}")
    print("DIAGNOSIS COMPLETE")
    print("="*80)
    print("1. Compare the test_sphere.vti with your problematic files")
    print("2. The test file should show a sphere in ParaView")
    print("3. If test file works but yours don't, the issue is in data structure")
    print("4. Check the analysis above for differences in data layout")

if __name__ == "__main__":
    main()