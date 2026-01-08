import nibabel as nib
import numpy as np
import os

# === User paths ===
atlas_path = "/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.1_sym/NMT_v2.1_sym_05mm/supplemental_SARM/SARM_5_in_NMT_v2.1_sym_05mm.nii.gz"
output_dir = "/Users/similovesyou/Desktop/qts/simian-brain/masks"

os.makedirs(output_dir, exist_ok=True)

# VTA indices from SARM_key_5.txt
# SL_VTA = 1207 (left), SR_VTA = 1707 (right)
label_ids = {
    "VTA_bilateral": [1207, 1707],  # Both left and right VTA
    "VTA_left": [1207],             # Left VTA only
    "VTA_right": [1707]             # Right VTA only
}

# === Load atlas ===
print("Loading atlas...")
atlas_img = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata()
print(f"Atlas shape: {atlas_data.shape}")
print(f"Atlas data type: {atlas_data.dtype}")

# === Generate and save masks ===
print("\nGenerating VTA masks...")
for region, ids in label_ids.items():
    mask = np.isin(atlas_data, ids).astype(np.uint8)
    mask_img = nib.Nifti1Image(mask, affine=atlas_img.affine, header=atlas_img.header)
    out_path = os.path.join(output_dir, f"{region}_seed.nii.gz")
    nib.save(mask_img, out_path)

    # Print some info about the mask
    voxel_count = int(np.sum(mask))
    print(f"{region} (IDs: {ids}): {voxel_count} voxels → saved to {out_path}")

print(f"\nVTA seed masks saved to: {output_dir}")

# Optional: Check if the expected indices are present in the atlas
print("\nVerifying atlas contains VTA indices...")
unique_values = np.unique(atlas_data)
vta_indices = [1207, 1707]
for idx in vta_indices:
    if idx in unique_values:
        count = np.sum(atlas_data == idx)
        print(f"Index {idx}: Found ({count} voxels)")
    else:
        print(f"Index {idx}: NOT FOUND in atlas")