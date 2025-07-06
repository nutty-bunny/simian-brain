#!/bin/bash

# Input/output directories
output_dir="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/patchoulli/"
registered_T1="${output_dir}T1_registered.nii.gz"
NMT_mask="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.0_sym/NMT_v2.0_sym_05mm/NMT_v2.0_sym_05mm_brainmask.nii.gz"

# Step 1: Multiply registered T1 by NMT mask to skull-strip
fslmaths "$registered_T1" -mul "$NMT_mask" "${output_dir}T1_brain.nii.gz"

# Step 2: Optional - Create a mask of the skull-stripped brain for quality check
fslmaths "${output_dir}T1_brain.nii.gz" -bin "${output_dir}T1_brain_mask.nii.gz"

# Step 3: Optional - Create a skull image (for visualization/QC)
fslmaths "$registered_T1" -sub "${output_dir}T1_brain.nii.gz" "${output_dir}T1_skull.nii.gz"

echo "Skull stripping complete:"
echo "Brain extracted image: ${output_dir}T1_brain.nii.gz"
echo "Brain mask: ${output_dir}T1_brain_mask.nii.gz"
echo "Skull image: ${output_dir}T1_skull.nii.gz"