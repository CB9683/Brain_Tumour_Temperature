#!/usr/bin/env python3
"""
Deep debugging of VTK ImageData issues
"""
import os
import numpy as np
import vtk
from vtk.util import numpy_support

def create_working_reference():
    """Create a VTK file using the exact same method as VTK examples"""
    print("Creating reference VTK file using standard VTK approach...")
    
    # Create image data the "official" way
    image_data = vtk.vtkImageData()
    
    # Set dimensions - this is critical
    dims = [50, 60, 70]  # Different dimensions to see orientation
    image_data.SetDimensions(dims[0], dims[1], dims[2])
    image_data.SetSpacing(1.0, 1.0, 1.0)
    image_data.SetOrigin(0.0, 0.0, 0.0)
    
    # Allocate memory for the scalar data
    image_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    
    # Fill the data using VTK's approach
    for z in range(dims[2]):
        for y in range(dims[1]):
            for x in range(dims[0]):
                # Create a pattern: sphere + some structure
                center_x, center_y, center_z = dims[0]//2, dims[1]//2, dims[2]//2
                dist = np.sqrt((x-center_x)**2 + (y-center_y)**2 + (z-center_z)**2)
                
                if dist < 15:  # Sphere
                    value = 255
                elif x < 10 or y < 10 or z < 10:  # Borders
                    value = 128
                else:
                    value = 0
                
                image_data.SetScalarComponentFromDouble(x, y, z, 0, value)
    
    # Save reference file
    ref_file = "/Users/c3495249/Coding/Gemini_Pro_Vasculature/reference_working.vti"
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(ref_file)
    writer.SetInputData(image_data)
    writer.Write()
    
    print(f"Reference file created: {ref_file}")
    return ref_file

def create_our_method():
    """Create VTK file using our current method"""
    print("Creating VTK file using our current method...")
    
    # Create data the way we do it
    dims = [50, 60, 70]
    data = np.zeros(dims, dtype=np.uint8)
    
    # Fill with same pattern
    center_x, center_y, center_z = dims[0]//2, dims[1]//2, dims[2]//2
    for z in range(dims[2]):
        for y in range(dims[1]):
            for x in range(dims[0]):
                dist = np.sqrt((x-center_x)**2 + (y-center_y)**2 + (z-center_z)**2)
                if dist < 15:
                    data[x, y, z] = 255
                elif x < 10 or y < 10 or z < 10:
                    data[x, y, z] = 128
    
    # Create VTK ImageData our way
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(dims)
    image_data.SetSpacing([1.0, 1.0, 1.0])
    image_data.SetOrigin([0.0, 0.0, 0.0])
    
    # This is where the problem might be - how we set the scalar data
    vtk_data = numpy_support.numpy_to_vtk(
        data.ravel(order='C'),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR
    )
    vtk_data.SetName("our_data")
    image_data.GetPointData().SetScalars(vtk_data)
    
    # Save our file
    our_file = "/Users/c3495249/Coding/Gemini_Pro_Vasculature/our_method.vti"
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(our_file)
    writer.SetInputData(image_data)
    writer.Write()
    
    print(f"Our method file created: {our_file}")
    return our_file

def analyze_vtk_internals(filepath):
    """Deep analysis of VTK file internals"""
    print(f"\n{'='*60}")
    print(f"DEEP ANALYSIS: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print("File not found!")
        return
    
    try:
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(filepath)
        reader.Update()
        
        image_data = reader.GetOutput()
        
        print(f"Basic Properties:")
        print(f"  Dimensions: {image_data.GetDimensions()}")
        print(f"  Spacing: {image_data.GetSpacing()}")
        print(f"  Origin: {image_data.GetOrigin()}")
        print(f"  Extent: {image_data.GetExtent()}")
        print(f"  Bounds: {image_data.GetBounds()}")
        print(f"  Number of Points: {image_data.GetNumberOfPoints()}")
        print(f"  Number of Cells: {image_data.GetNumberOfCells()}")
        
        # Check data type and memory layout
        scalars = image_data.GetPointData().GetScalars()
        if scalars:
            print(f"\nScalar Data:")
            print(f"  Name: {scalars.GetName()}")
            print(f"  Data Type: {scalars.GetDataType()} ({scalars.GetDataTypeAsString()})")
            print(f"  Number of Tuples: {scalars.GetNumberOfTuples()}")
            print(f"  Number of Components: {scalars.GetNumberOfComponents()}")
            print(f"  Size: {scalars.GetSize()}")
            
            # Convert to numpy and analyze
            numpy_data = numpy_support.vtk_to_numpy(scalars)
            print(f"  Numpy shape: {numpy_data.shape}")
            print(f"  Numpy dtype: {numpy_data.dtype}")
            print(f"  Min/Max: {numpy_data.min()}/{numpy_data.max()}")
            print(f"  Non-zero count: {np.count_nonzero(numpy_data)}")
            
            # Check if the data makes sense geometrically
            expected_size = image_data.GetDimensions()[0] * image_data.GetDimensions()[1] * image_data.GetDimensions()[2]
            print(f"  Expected size: {expected_size}")
            print(f"  Actual size: {len(numpy_data)}")
            print(f"  Size match: {expected_size == len(numpy_data)}")
            
            # Sample some specific locations
            dims = image_data.GetDimensions()
            if len(numpy_data) == expected_size:
                # Check center point (should be non-zero in our sphere)
                center_idx = (dims[2]//2) * dims[1] * dims[0] + (dims[1]//2) * dims[0] + (dims[0]//2)
                if center_idx < len(numpy_data):
                    print(f"  Center value: {numpy_data[center_idx]} (should be 255)")
                
                # Check corner (should be 0 or 128)
                corner_idx = 0
                print(f"  Corner value: {numpy_data[corner_idx]} (should be 0 or 128)")
            
        else:
            print("  ERROR: No scalar data found!")
            
        # Check if there are any attributes that might affect visualization
        field_data = image_data.GetFieldData()
        if field_data.GetNumberOfArrays() > 0:
            print(f"\nField Data: {field_data.GetNumberOfArrays()} arrays")
            for i in range(field_data.GetNumberOfArrays()):
                array = field_data.GetArray(i)
                print(f"  Array {i}: {array.GetName()}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def check_vtk_xml_content(filepath):
    """Check the raw XML content of the VTK file"""
    print(f"\n{'='*60}")
    print(f"XML CONTENT CHECK: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print("File not found!")
        return
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Look for key XML elements
        lines = content.split('\n')
        for i, line in enumerate(lines[:20]):  # First 20 lines
            if any(keyword in line.lower() for keyword in ['imagedata', 'extent', 'spacing', 'origin', 'dataarray']):
                print(f"Line {i+1}: {line.strip()}")
                
        # Look for DataArray content
        print("\nDataArray sections:")
        in_data_array = False
        for i, line in enumerate(lines):
            if '<DataArray' in line:
                print(f"Line {i+1}: {line.strip()}")
                in_data_array = True
            elif '</DataArray>' in line and in_data_array:
                print(f"Line {i+1}: {line.strip()}")
                in_data_array = False
                break
            elif in_data_array and i < 1000:  # Don't print too much data
                print(f"Line {i+1}: {line.strip()[:100]}...")
                if i > 10:  # Just a few lines of actual data
                    break
                    
    except Exception as e:
        print(f"Error reading XML: {e}")

def main():
    print("DEEP VTK DEBUGGING - Finding the Real Problem")
    print("="*80)
    
    # Create both reference and our method files
    ref_file = create_working_reference()
    our_file = create_our_method()
    
    # Analyze both files in detail
    analyze_vtk_internals(ref_file)
    analyze_vtk_internals(our_file)
    
    # Check XML content
    check_vtk_xml_content(ref_file)
    check_vtk_xml_content(our_file)
    
    # Also analyze one of our problematic real files
    real_file = "/Users/c3495249/Coding/Gemini_Pro_Vasculature/test_vtk_output/tissue_vtk/GM_iter_999.vti"
    if os.path.exists(real_file):
        analyze_vtk_internals(real_file)
    
    print(f"\n{'='*80}")
    print("TEST IN PARAVIEW:")
    print(f"1. Try the reference file: {ref_file}")
    print(f"2. Try our method file: {our_file}")
    print(f"3. Compare how they display - this will tell us what's wrong")
    print("="*80)

if __name__ == "__main__":
    main()