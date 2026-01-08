import pandas as pd
import nibabel as nib
import numpy as np
import os

# Paths - UPDATED to use key_5.txt files
sarm_table = "/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.1_sym/tables_SARM/SARM_key_5.txt"
charm_table = "/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.1_sym/tables_CHARM/CHARM_key_5.txt"
sarm_atlas = "/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.1_sym/NMT_v2.1_sym_05mm/SARM_in_NMT_v2.1_sym_05mm.nii.gz"
charm_atlas = "/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.1_sym/NMT_v2.1_sym_05mm/CHARM_in_NMT_v2.1_sym_05mm.nii.gz"
save_dir = "/Users/similovesyou/Desktop/qts/simian-brain/masks/motor-impulsivity"

os.makedirs(save_dir, exist_ok=True)

# Load atlases
sarm_img = nib.load(sarm_atlas)
sarm_data = sarm_img.get_fdata()
charm_img = nib.load(charm_atlas)
charm_data = charm_img.get_fdata()

# Load tables - tab-separated .txt files
sarm_df = pd.read_csv(sarm_table, sep='\t')
charm_df = pd.read_csv(charm_table, sep='\t')

print(f"SARM atlas shape: {sarm_data.shape}, range: [{sarm_data.min()}, {sarm_data.max()}]")
print(f"CHARM atlas shape: {charm_data.shape}, range: [{charm_data.min()}, {charm_data.max()}]")
print(f"\nSARM table columns: {list(sarm_df.columns)}")
print(f"CHARM table columns: {list(charm_df.columns)}")

# MOTOR IMPULSIVITY NETWORK SEEDS
cortical_regions = {
    'mPFC': ['area_32'],
    'ACC': ['area_24a/b', 'area_24c'],
    'MCC': ['area_24a/b_prime', 'area_24c_prime'],
    'OFC': ['area_11', 'area_13', 'area_12m/o', 'area_12r/l', 'area_14'],
    'preSMA': ['preSMA'],
}

subcortical_regions = {
    'striatum': ['_Cd', '_Pu', '_Acb'],
    'MD_thalamus': ['_MD'],
}

def create_mask_from_charm(df, atlas_data, abbr_list, region_name):
    """Create binary mask from abbreviations in CHARM atlas"""
    mask = np.zeros_like(atlas_data)
    
    for abbr in abbr_list:
        matching_rows = df[df['Abbreviation'].astype(str).str.contains(abbr, na=False, regex=False)]
        
        if len(matching_rows) == 0:
            print(f"  Warning: No match for '{abbr}'")
            continue
        
        for idx, row in matching_rows.iterrows():
            atlas_index = row['Index']
            region_indices = atlas_data == atlas_index
            mask[region_indices] = 1
            print(f"  Added '{row['Abbreviation']}' (Index {atlas_index})")
        
        print(f"  '{abbr}': {len(matching_rows)} matches")
    
    nvoxels = np.sum(mask > 0)
    print(f"  Total voxels in {region_name}: {nvoxels}")
    if nvoxels == 0:
        print(f"  WARNING: {region_name} mask is EMPTY")
    
    return mask

def create_subcortical_mask(df, atlas_data, abbr_list, region_name):
    """Create subcortical masks using SARM indices"""
    mask = np.zeros_like(atlas_data)
    
    for abbr in abbr_list:
        matching = df[df['Abbreviation'].astype(str).str.contains(abbr, na=False, regex=False)]
        
        if len(matching) == 0:
            print(f"  Warning: No match for '{abbr}'")
            continue
        
        for idx, row in matching.iterrows():
            atlas_index = row['Index']
            region_indices = atlas_data == atlas_index
            mask[region_indices] = 1
            print(f"  Added '{row['Abbreviation']}' (Index {atlas_index})")
        
        print(f"  '{abbr}': {len(matching)} matches")
    
    nvoxels = np.sum(mask > 0)
    print(f"  Total voxels in {region_name}: {nvoxels}")
    if nvoxels == 0:
        print(f"  WARNING: {region_name} mask is EMPTY")
    
    return mask

# Create cortical masks
print("\n" + "="*80)
print("CREATING CORTICAL MASKS")
print("="*80)
for region_name, abbr_list in cortical_regions.items():
    print(f"\n{region_name}:")
    mask = create_mask_from_charm(charm_df, charm_data, abbr_list, region_name)
    
    mask_img = nib.Nifti1Image(mask.astype(np.int16), charm_img.affine, charm_img.header)
    output_path = os.path.join(save_dir, f"{region_name}_seed.nii.gz")
    nib.save(mask_img, output_path)
    print(f"  ✓ Saved: {region_name}_seed.nii.gz")

# Create subcortical masks
print("\n" + "="*80)
print("CREATING SUBCORTICAL MASKS")
print("="*80)
for region_name, abbr_list in subcortical_regions.items():
    print(f"\n{region_name}:")
    mask = create_subcortical_mask(sarm_df, sarm_data, abbr_list, region_name)
    
    mask_img = nib.Nifti1Image(mask.astype(np.int16), sarm_img.affine, sarm_img.header)
    output_path = os.path.join(save_dir, f"{region_name}_seed.nii.gz")
    nib.save(mask_img, output_path)
    print(f"  ✓ Saved: {region_name}_seed.nii.gz")

print(f"\n{'='*80}")
print(f"All masks saved to: {save_dir}")
print("="*80)