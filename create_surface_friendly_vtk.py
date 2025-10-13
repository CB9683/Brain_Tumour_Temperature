#!/usr/bin/env python3
"""
Create VTK files that work better with surface rendering
"""
import os
import sys
import numpy as np
sys.path.append('/Users/c3495249/Coding/Gemini_Pro_Vasculature')

def create_surface_friendly_vtk():
    """Create VTK file optimized for surface rendering"""
    import vtk
    
    print("Creating surface-friendly VTK file...")
    
    # Create test data
    dims = [100, 120, 80]
    center = [dims[0]//2, dims[1]//2, dims[2]//2]
    
    # Create VTK ImageData
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(dims[0], dims[1], dims[2])
    image_data.SetSpacing(1.0, 1.0, 1.0)
    image_data.SetOrigin(0.0, 0.0, 0.0)
    image_data.AllocateScalars(vtk.VTK_FLOAT, 1)  # Use float for better surface generation
    
    # Fill with distance field (this creates smooth surfaces)
    for z in range(dims[2]):
        for y in range(dims[1]):
            for x in range(dims[0]):
                dist = np.sqrt((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2)
                
                # Create a smooth distance field
                if dist <= 30:
                    # Inside sphere: positive values
                    value = 30.0 - dist  # Ranges from 30 (center) to 0 (edge)
                else:
                    # Outside sphere: negative values
                    value = -(dist - 30.0)  # Negative values outside
                
                image_data.SetScalarComponentFromDouble(x, y, z, 0, value)
    
    # Save the file
    output_file = "/Users/c3495249/Coding/Gemini_Pro_Vasculature/surface_friendly_sphere.vti"
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(image_data)
    writer.Write()
    
    print(f"Created surface-friendly file: {output_file}")
    print("This should work with both Volume and Surface representations")
    print("For Surface mode, the isosurface will be at value 0.0")
    
    return output_file

def create_binary_with_proper_contour():
    """Create binary VTK but with proper contour hints"""
    import vtk
    
    print("Creating binary VTK with contour optimization...")
    
    dims = [100, 120, 80]
    center = [dims[0]//2, dims[1]//2, dims[2]//2]
    
    # Create VTK ImageData
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(dims[0], dims[1], dims[2])
    image_data.SetSpacing(1.0, 1.0, 1.0)
    image_data.SetOrigin(0.0, 0.0, 0.0)
    image_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    
    # Fill with binary data but add some boundary smoothing
    for z in range(dims[2]):
        for y in range(dims[1]):
            for x in range(dims[0]):
                dist = np.sqrt((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2)
                
                if dist <= 28:
                    value = 255  # Solid inside
                elif dist <= 32:
                    # Create a transition zone for better surface generation
                    blend = (32 - dist) / 4.0  # Linear blend over 4 voxels
                    value = int(255 * blend)
                else:
                    value = 0  # Outside
                
                image_data.SetScalarComponentFromDouble(x, y, z, 0, value)
    
    # Save the file
    output_file = "/Users/c3495249/Coding/Gemini_Pro_Vasculature/binary_smooth_sphere.vti"
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(image_data)
    writer.Write()
    
    print(f"Created binary smooth file: {output_file}")
    print("For Surface mode, try contour values around 127-128")
    
    return output_file

def main():
    print("CREATING SURFACE-RENDERING FRIENDLY VTK FILES")
    print("="*80)
    
    # Create different versions
    surface_file = create_surface_friendly_vtk()
    binary_file = create_binary_with_proper_contour()
    
    print(f"\n{'='*80}")
    print("TESTING INSTRUCTIONS:")
    print("1. Test surface_friendly_sphere.vti:")
    print("   - Volume mode: Should show smooth sphere")
    print("   - Surface mode: Should automatically show surface at value 0")
    print()
    print("2. Test binary_smooth_sphere.vti:")
    print("   - Volume mode: Should show sphere")
    print("   - Surface mode: Use Contour filter with value 127.5")
    print()
    print("3. For your brain data:")
    print("   - Always use Contour filter with value 0.5 for binary masks")
    print("   - Or modify export to create smooth distance fields")
    print("="*80)

if __name__ == "__main__":
    main()