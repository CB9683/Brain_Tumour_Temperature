# extract_mida_masks.py
import nibabel as nib
import numpy as np
import os

MIDA_LABEL_FILE = "/Users/c3495249/Coding/Gemini_Pro_Vasculature/data/MIDA_v1.nii" # Path to your MIDA label map
OUTPUT_DIR = "mida_processed_masks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define label values from your text file
LABELS = {
    "GM_Brain": 10,
    "WM_Brain": 12,
    "GM_Cerebellum": 2,
    "WM_Cerebellum": 9,
    "CSF_General": 32,
    "CSF_Ventricles": 6,
    "Arteries": 24
}

# --- Function to create and save a binary mask ---
def create_save_mask(label_data, affine, header, label_value, output_filename_base):
    mask_data = (label_data == label_value).astype(np.uint8)
    if np.sum(mask_data) == 0:
        print(f"Warning: No voxels found for label {label_value} ({output_filename_base}). Mask will be empty.")
    
    mask_img = nib.Nifti1Image(mask_data, affine, header)
    nib.save(mask_img, os.path.join(OUTPUT_DIR, f"{output_filename_base}.nii.gz"))
    print(f"Saved {output_filename_base}.nii.gz ({np.sum(mask_data)} voxels)")

# --- Function to combine masks (e.g., GM_Brain + GM_Cerebellum) ---
def combine_save_masks(label_data, affine, header, label_values_list, output_filename_base):
    combined_mask_data = np.zeros(label_data.shape, dtype=np.uint8)
    for label_value in label_values_list:
        combined_mask_data[label_data == label_value] = 1
    
    if np.sum(combined_mask_data) == 0:
        print(f"Warning: No voxels found for combined labels {label_values_list} ({output_filename_base}). Mask will be empty.")

    mask_img = nib.Nifti1Image(combined_mask_data, affine, header)
    nib.save(mask_img, os.path.join(OUTPUT_DIR, f"{output_filename_base}.nii.gz"))
    print(f"Saved combined {output_filename_base}.nii.gz ({np.sum(combined_mask_data)} voxels)")

if __name__ == "__main__":
    print(f"Loading MIDA label file: {MIDA_LABEL_FILE}")
    try:
        img = nib.load(MIDA_LABEL_FILE)
        # Ensure data is integer for label comparison
        # MIDA data type was float64, but values are integer labels.
        label_data_float = img.get_fdata()
        if not np.all(np.mod(label_data_float, 1) == 0):
            print("Warning: Loaded data contains non-integer values, but expecting a label map.")
        label_data_int = label_data_float.astype(np.int32) # Use a robust integer type

        affine = img.affine
        header = img.header # Preserve header for consistent geometry

        print(f"Successfully loaded. Shape: {label_data_int.shape}, Affine:\n{affine}")

        # Create individual masks
        create_save_mask(label_data_int, affine, header, LABELS["Arteries"], "mida_arteries_mask")
        
        # Create combined masks for GM and WM
        # GM = Brain GM + Cerebellum GM
        combine_save_masks(label_data_int, affine, header, 
                           [LABELS["GM_Brain"], LABELS["GM_Cerebellum"]], 
                           "mida_total_gm_mask")
        
        # WM = Brain WM + Cerebellum WM
        combine_save_masks(label_data_int, affine, header, 
                           [LABELS["WM_Brain"], LABELS["WM_Cerebellum"]], 
                           "mida_total_wm_mask")

        # CSF = General CSF + Ventricles CSF
        combine_save_masks(label_data_int, affine, header,
                           [LABELS["CSF_General"], LABELS["CSF_Ventricles"]],
                           "mida_total_csf_mask")

        print(f"Mask extraction complete. Files saved in {OUTPUT_DIR}")

    except FileNotFoundError:
        print(f"Error: MIDA file not found at {MIDA_LABEL_FILE}")
    except Exception as e:
        print(f"An error occurred during mask extraction: {e}")