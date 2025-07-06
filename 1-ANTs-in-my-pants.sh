#!/bin/bash

output_dir="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/patchoulli/"
input_T1="${output_dir}T1-reoriented-cropped.nii.gz"

# NMTs
NMT_brain="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.0_sym/NMT_v2.0_sym_05mm/NMT_v2.0_sym_05mm.nii.gz"
NMT_mask="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.0_sym/NMT_v2.0_sym_05mm/NMT_v2.0_sym_05mm_brainmask.nii.gz"

# Denoise
DenoiseImage -d 3 -i "$input_T1" -o "${output_dir}T1_denoised.nii.gz"

# Affine align T1 to NMT template
antsRegistrationSyNQuick.sh \
  -d 3 \
  -f "$NMT_brain" \
  -m "${output_dir}T1_denoised.nii.gz" \
  -o "${output_dir}aligned_" \
  -t a \
  -n 8 \
  -p f

# Apply transform to reorient T1
antsApplyTransforms \
  -d 3 \
  -i "${output_dir}T1_denoised.nii.gz" \
  -r "$NMT_brain" \
  -t "${output_dir}aligned_0GenericAffine.mat" \
  -o "${output_dir}T1_aligned.nii.gz"

# Nonlinear registration (rigid + affine + SyN)
FIXED="$NMT_brain"
MOVING="${output_dir}T1_aligned.nii.gz"
OUTPREFIX="${output_dir}registered_"

antsRegistration \
--dimensionality 3 \
--float 0 \
--collapse-output-transforms 1 \
--interpolation Linear \
--winsorize-image-intensities [0.005,0.995] \
--use-histogram-matching 1 \
--initial-moving-transform [${FIXED},${MOVING},1] \
--transform Rigid[0.1] \
--metric MI[${FIXED},${MOVING},1,32,Regular,0.25] \
--convergence [1000x500x250x100,1e-6,10] \
--shrink-factors 8x4x2x1 \
--smoothing-sigmas 3x2x1x0vox \
--transform Affine[0.1] \
--metric MI[${FIXED},${MOVING},1,32,Regular,0.25] \
--convergence [1000x500x250x100,1e-6,10] \
--shrink-factors 8x4x2x1 \
--smoothing-sigmas 3x2x1x0vox \
--transform SyN[0.1,3,0] \
--metric CC[${FIXED},${MOVING},1,4] \
--convergence [100x70x50x20,1e-6,10] \
--shrink-factors 8x4x2x1 \
--smoothing-sigmas 3x2x1x0vox \
--output [${OUTPREFIX},${OUTPREFIX}Warped.nii.gz,${OUTPREFIX}InverseWarped.nii.gz]

# Rename the output files
mv "${output_dir}registered_Warped.nii.gz" "${output_dir}T1_registered.nii.gz"