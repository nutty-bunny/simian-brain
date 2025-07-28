#!/bin/bash

# Define paths
base_dir="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/iron/"
func_img="${base_dir}func/topupped-mc-mean.nii.gz"
ref_nmt="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.1_sym/NMT_v2.1_sym_05mm/NMT_v2.1_sym_05mm.nii.gz"
#ref_nmt="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.1_sym/NMT_v2.1_sym_05mm/NMT-padded.nii.gz" # use if you padded NMT to avoid edge cropping


# Transform files (from previous steps)
func2t1_mat="${base_dir}func/func2T1-rigid0GenericAffine.mat"
t1_to_nmt_affine="${base_dir}aligned-0GenericAffine.mat"

t1_to_nmt_warp="${base_dir}registered-1Warp.nii.gz"

# Output
output_func_nmt="${base_dir}func/single-func-in-NMT.nii.gz"

# Apply transforms: func → T1 → NMT
antsApplyTransforms \
  -d 3 \
  -i "$func_img" \
  -r "$ref_nmt" \
  -o "$output_func_nmt" \
  -n Linear \
  -t "$t1_to_nmt_warp" \
  -t "$t1_to_nmt_affine"
# if want to do native func-t1, add -t "$func2t1_mat" after the -t "$t1_to_nmt_affine"

echo "Functional image transformed to NMT space."
echo "Output saved as: $output_func_nmt"
echo ""
echo "Check result:"
echo "fsleyes $ref_nmt $output_func_nmt &"