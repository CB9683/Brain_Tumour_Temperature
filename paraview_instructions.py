#!/usr/bin/env python3
"""
Instructions and test for proper ParaView visualization of VTK files
"""
import os
import numpy as np
import vtk
from vtk.util import numpy_support

def create_simple_test_volume():
    """Create a simple test volume that should definitely work in ParaView"""
    
    print("Creating simple test volume...")
    
    # Create a 3D array with a clear structure
    dims = [100, 100, 100]
    data = np.zeros(dims, dtype=np.float32)
    
    # Create multiple structures that should be clearly visible:
    # 1. A large sphere in the center
    center = [50, 50, 50]
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                dist = np.sqrt((i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2)
                if dist <= 30:  # Large sphere
                    data[i, j, k] = 1.0
                elif 20 <= i <= 80 and 20 <= j <= 80 and k == 50:  # A slice
                    data[i, j, k] = 0.5
    
    print(f"Created volume with {np.count_nonzero(data)} non-zero voxels")
    
    # Create VTK ImageData
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(dims)
    image_data.SetSpacing([1.0, 1.0, 1.0])
    image_data.SetOrigin([0.0, 0.0, 0.0])
    
    # Convert numpy array to VTK array
    vtk_data = numpy_support.numpy_to_vtk(
        data.ravel(order='C'),  # Use C order
        deep=True,
        array_type=vtk.VTK_FLOAT
    )
    vtk_data.SetName("test_volume")
    
    # Set as scalar data
    image_data.GetPointData().SetScalars(vtk_data)
    
    # Write to file
    output_file = "/Users/c3495249/Coding/Gemini_Pro_Vasculature/test_volume_simple.vti"
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(image_data)
    writer.Write()
    
    print(f"Test volume written to: {output_file}")
    return output_file

def analyze_mask_file(filepath):
    """Analyze a mask file and suggest ParaView settings"""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print("File not found!")
        return
    
    try:
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(filepath)
        reader.Update()
        
        image_data = reader.GetOutput()
        
        # Get scalar data
        point_data = image_data.GetPointData()
        if point_data.GetNumberOfArrays() > 0:
            scalar_array = point_data.GetArray(0)
            numpy_array = numpy_support.vtk_to_numpy(scalar_array)
            
            unique_vals = np.unique(numpy_array)
            nonzero_count = np.count_nonzero(numpy_array)
            total_count = len(numpy_array)
            
            print(f"Data summary:")
            print(f"  Total voxels: {total_count:,}")
            print(f"  Non-zero voxels: {nonzero_count:,} ({100*nonzero_count/total_count:.1f}%)")
            print(f"  Unique values: {unique_vals}")
            print(f"  Data range: {numpy_array.min()} to {numpy_array.max()}")
            
            # ParaView recommendations
            print(f"\nParaView visualization recommendations:")
            print(f"  1. Set Representation to 'Volume' (not 'Outline' or 'Surface')")
            print(f"  2. If using Volume rendering:")
            print(f"     - Click 'Edit' next to 'Volume Rendering Mode'")
            print(f"     - Adjust opacity transfer function")
            print(f"     - Set opacity for value 1.0 to be visible (e.g., 0.8)")
            print(f"  3. Alternative: Set Representation to 'Surface'")
            print(f"     - This will show the boundary surface of the mask")
            print(f"  4. Color by: '{scalar_array.GetName()}'")
            
            if len(unique_vals) == 2 and 0 in unique_vals and 1 in unique_vals:
                print(f"  5. This is a binary mask - perfect for Surface rendering")
            else:
                print(f"  5. This has multiple values - Volume rendering might be better")
                
    except Exception as e:
        print(f"Error analyzing file: {e}")

def print_paraview_instructions():
    """Print detailed ParaView instructions"""
    print(f"\n{'='*80}")
    print("DETAILED PARAVIEW INSTRUCTIONS")
    print(f"{'='*80}")
    
    instructions = """
STEP-BY-STEP PARAVIEW USAGE:

1. OPEN FILES:
   - Launch ParaView
   - File -> Open
   - Navigate to your tissue_vtk folder
   - Select one or more .vti files
   - Click OK
   - Click "Apply" in the Properties panel

2. IF YOU SEE ONLY A WIREFRAME CUBE:
   This means ParaView is showing the "Outline" representation.
   
   FIX: In the Properties panel:
   - Find "Representation" dropdown (usually shows "Outline")
   - Change it to "Surface" or "Volume"
   - Click "Apply"

3. FOR BINARY MASKS (like GM, WM, perfused):
   - Use "Surface" representation
   - This shows the 3D boundary of the mask region
   - Should look like brain tissue shapes

4. FOR VOLUME RENDERING:
   - Use "Volume" representation  
   - Click "Edit" button next to "Volume Rendering Mode"
   - In the transfer function editor:
     * Set opacity for your data values (usually 0 and 1)
     * Make sure value 1.0 has visible opacity (try 0.5-0.8)
     * Value 0.0 should have opacity 0.0 (transparent)

5. COLORING:
   - Use the "Color by" dropdown to select your scalar field
   - Choose appropriate color map (e.g., "Cool to Warm", "Viridis")

6. MULTIPLE FILES:
   - Load each .vti file separately
   - Each will appear as a separate item in Pipeline Browser
   - Toggle visibility with the eye icon
   - Different representations for each

COMMON ISSUES:
- "Only seeing wireframe": Change Representation from "Outline" to "Surface"
- "Volume is invisible": Adjust opacity in Volume Rendering transfer function
- "Wrong colors": Check "Color by" setting and color map
- "Too transparent": Increase opacity values in transfer function
"""
    
    print(instructions)

def main():
    print("PARAVIEW VTK TROUBLESHOOTING TOOL")
    print("="*80)
    
    # Create a simple test file that should definitely work
    test_file = create_simple_test_volume()
    
    # Analyze the test file
    analyze_mask_file(test_file)
    
    # Analyze some of our actual mask files
    mask_files = [
        "/Users/c3495249/Coding/Gemini_Pro_Vasculature/test_vtk_output/tissue_vtk/GM_iter_999.vti",
        "/Users/c3495249/Coding/Gemini_Pro_Vasculature/test_vtk_output/tissue_vtk/perfused_iter_999.vti"
    ]
    
    for filepath in mask_files:
        if os.path.exists(filepath):
            analyze_mask_file(filepath)
    
    # Print detailed instructions
    print_paraview_instructions()
    
    print(f"\n{'='*80}")
    print("TESTING PRIORITY:")
    print("1. First test with test_volume_simple.vti")
    print("2. If that works, then test with GM_iter_999.vti") 
    print("3. Make sure to change Representation from 'Outline' to 'Surface'")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()