#!/bin/bash

# Define paths
base_dir="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/patchoulli/"
func_img="${base_dir}func/topupped_mc_mean.nii.gz"
ref_nmt="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.0_sym/NMT_v2.0_sym_05mm/NMT_v2.0_sym_05mm.nii.gz"

# Transform files (from previous steps)
func2t1_mat="${base_dir}func/func2T1_rigid_ants_refined0GenericAffine.mat"
t1_to_nmt_affine="${base_dir}aligned_0GenericAffine.mat"
t1_to_nmt_warp="${base_dir}registered_1Warp.nii.gz"

# Output
output_func_nmt="${base_dir}func/single_func_in_NMT.nii.gz"

# Apply transforms: func → T1 → NMT
antsApplyTransforms \
  -d 3 \
  -i "$func_img" \
  -r "$ref_nmt" \
  -o "$output_func_nmt" \
  -n Linear \
  -t "$t1_to_nmt_warp" \
  -t "$t1_to_nmt_affine" \
  -t "$func2t1_mat"

echo "Functional image transformed to NMT space."
echo "Output saved as: $output_func_nmt"
echo ""
echo "Check result:"
echo "fsleyes $ref_nmt $output_func_nmt &"
