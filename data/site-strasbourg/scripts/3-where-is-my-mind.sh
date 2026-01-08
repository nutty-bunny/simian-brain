#!/bin/bash

# Single functional image transformation to NMT space
# Matching logic to anat-to-NMT pipeline
# Keeps unmasked output and masked output separately
# Stores results in func2nmt directory

set -e  # Exit on error

# Configurable parameters
USE_FUNC2T1_TRANSFORM=0  # Set to 1 to use func2t1 transform
if [[ $# -ge 1 ]]; then
    USE_FUNC2T1_TRANSFORM=$1
    echo "Command line override - USE_FUNC2T1_TRANSFORM: $USE_FUNC2T1_TRANSFORM"
fi

CREATE_QC_OVERLAYS=0
VERBOSE=1

# Paths
base_dir="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/final-derivatives/yin/"
func_img="${base_dir}moco_mean.nii.gz"
func_mask="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/final-derivatives/yin/T1-in-func-mask.nii.gz"

ref_nmt_padded="${base_dir}anat2nmt/temp-anat2nmt/NMT-padded.nii.gz"
ref_nmt_orig="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.1_sym/NMT_v2.1_sym_05mm/NMT_v2.1_sym_05mm.nii.gz"
nmt_mask="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.1_sym/NMT_v2.1_sym_05mm/NMT_v2.1_sym_05mm_brainmask.nii.gz"
nmt_mask_padded="${base_dir}anat2nmt/temp-anat2nmt/NMT-mask-padded.nii.gz"

func2anat="${base_dir}func2anat/func2anat-0GenericAffine.mat"
t1_to_nmt_initial="${base_dir}anat2nmt/T1-to-NMT-0GenericAffine.mat"
t1_to_nmt_affine="${base_dir}anat2nmt/T1-to-NMT-1GenericAffine.mat"
t1_to_nmt_warp="${base_dir}anat2nmt/T1-to-NMT-1Warp.nii.gz"

# Output directory
output_dir="${base_dir}func2nmt/"
mkdir -p "$output_dir"
temp_dir="${output_dir}temp/"
mkdir -p "$temp_dir"

# Strategy
if [[ $USE_FUNC2T1_TRANSFORM -eq 1 ]]; then
    transform_strategy="func → T1 → NMT"
    transforms=("$t1_to_nmt_warp" "$t1_to_nmt_affine" "$t1_to_nmt_initial" "$func2anat")
else
    transform_strategy="T1 → NMT only (func/T1 aligned)"
    transforms=("$t1_to_nmt_warp" "$t1_to_nmt_affine" "$t1_to_nmt_initial")
fi

echo "Strategy: $transform_strategy"

# Check files
for f in "${transforms[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "Missing transform: $f"
        exit 1
    fi
done
[[ -f "$func_img" ]] || { echo "Missing functional image: $func_img"; exit 1; }

# Transform functional image to padded NMT
antsApplyTransforms \
  -d 3 \
  -i "$func_img" \
  -r "$ref_nmt_padded" \
  -o "${temp_dir}single-func-in-NMT-padded.nii.gz" \
  -n Linear \
  $(printf -- '-t %s ' "${transforms[@]}")

# Transform mask (if exists)
if [[ -f "$func_mask" ]]; then
    antsApplyTransforms \
      -d 3 \
      -i "$func_mask" \
      -r "$ref_nmt_padded" \
      -o "${temp_dir}single-func-mask-in-NMT-padded.nii.gz" \
      -n NearestNeighbor \
      $(printf -- '-t %s ' "${transforms[@]}")
fi

# Resample to standard NMT (unmasked output)
antsApplyTransforms \
  -d 3 \
  -i "${temp_dir}single-func-in-NMT-padded.nii.gz" \
  -r "$ref_nmt_orig" \
  -o "${output_dir}single-func-in-NMT.nii.gz" \
  -n Linear

# Create masked version
ImageMath 3 "${output_dir}single-func-in-NMT-masked.nii.gz" m "${output_dir}single-func-in-NMT.nii.gz" "$nmt_mask"

# Resample mask (if exists)
if [[ -f "${temp_dir}single-func-mask-in-NMT-padded.nii.gz" ]]; then
    antsApplyTransforms \
      -d 3 \
      -i "${temp_dir}single-func-mask-in-NMT-padded.nii.gz" \
      -r "$ref_nmt_orig" \
      -o "${output_dir}single-func-mask-in-NMT.nii.gz" \
      -n NearestNeighbor
fi

# QC overlay
if [[ $CREATE_QC_OVERLAYS -eq 1 ]]; then
    CreateTiledMosaic \
      -i "${output_dir}single-func-in-NMT.nii.gz" \
      -r "$ref_nmt_orig" \
      -o "${output_dir}QC-single-func-in-NMT-overlay.png" \
      -t -1x-1 -d 2 -p mask \
      -s [5,mask,mask] \
      -x "$nmt_mask" \
      -a 0.7
fi

# Summary
cat > "${output_dir}transform-summary.txt" << EOF
Functional to NMT Transform Summary
Date: $(date)
Functional input: $func_img
NMT reference: $ref_nmt_orig
Transform strategy: $transform_strategy
Transforms used:
$(printf '  %s\n' "${transforms[@]}")
Outputs:
  single-func-in-NMT.nii.gz (unmasked single functional mean image)
  single-func-in-NMT-masked.nii.gz (masked)
  single-func-mask-in-NMT.nii.gz (mask in NMT space)
EOF

echo "Done. Outputs in: $output_dir"