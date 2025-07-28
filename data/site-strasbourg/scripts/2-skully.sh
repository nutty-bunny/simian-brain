#!/bin/bash
# BAD BAD BAD 
# Input/output directories
output_dir="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/baal/"
registered_T1="${output_dir}T1-registered.nii.gz"
NMT_mask="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.1_sym/NMT_v2.1_sym_05mm/NMT_v2.1_sym_05mm_brainmask.nii.gz"

# Step 0: Dilate the NMT brain mask to make it less aggressive
dilated_mask="${output_dir}NMT-mask-dilated.nii.gz"
fslmaths "$NMT_mask" -dilM "$dilated_mask"

# Step 1: Multiply registered T1 by dilated mask to skull-strip
fslmaths "$registered_T1" -mul "$dilated_mask" "${output_dir}T1-brain.nii.gz"

# Step 2: Optional - Create a mask of the skull-stripped brain for quality check
fslmaths "${output_dir}T1-brain.nii.gz" -bin "${output_dir}T1-brain-mask.nii.gz"

echo "Skull stripping complete (using dilated mask):"
echo "Brain extracted image: ${output_dir}T1-brain.nii.gz"
echo "Brain mask: ${output_dir}T1-brain-mask.nii.gz"