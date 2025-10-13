#!/usr/bin/env python3
"""
Script to extract terminal coordinates from terminals_world.txt and format them as YAML seed points.
"""

import re

def parse_terminals_file(file_path):
    """
    Parse the terminals_world.txt file and extract outlet coordinates and radii.
    
    Args:
        file_path (str): Path to the terminals_world.txt file
        
    Returns:
        list: List of dictionaries containing outlet data
    """
    outlets = []
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Skip comments, empty lines, and inlet lines
            if line.startswith('#') or not line or line.startswith('Inlet:'):
                continue
            
            # Parse outlet lines
            if line.startswith('Outlet_'):
                # Extract outlet number and coordinates using regex
                match = re.match(r'Outlet_(\d+):\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*radius:\s*([-\d.]+)', line)
                
                if match:
                    outlet_num = int(match.group(1))
                    x = float(match.group(2))
                    y = float(match.group(3))
                    z = float(match.group(4))
                    radius = float(match.group(5))
                    
                    outlet_data = {
                        'id': f'outlet_{outlet_num}',
                        'position': [x, y, z],
                        'initial_radius': radius
                    }
                    
                    outlets.append(outlet_data)
    
    return outlets

def main():
    """Main function to extract terminals and generate YAML output."""
    file_path = '/Users/c3495249/Coding/Gemini_Pro_Vasculature/data/mida_reoriented_processing_v2/mida_arteries_mask_centerlines_TARGET_RAS/terminals_world.txt'
    
    try:
        # Parse the terminals file
        outlets = parse_terminals_file(file_path)
        
        print(f"# Extracted {len(outlets)} outlets from terminals_world.txt")
        print("# Copy the following YAML entries to your config.yaml file:")
        print()
        
        # Generate YAML output
        for outlet in outlets:
            print(f"- id: {outlet['id']}")
            print(f"  position: {outlet['position']}")
            print(f"  initial_radius: {outlet['initial_radius']}")
            print()
        
        print(f"# Total outlets processed: {len(outlets)}")
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()