#!/bin/bash

output_dir="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/iron/"
input_T1="${output_dir}T1-reoriented-cropped.nii.gz"

# NMT paths
NMT_brain_orig="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.1_sym/NMT_v2.1_sym_05mm/NMT_v2.1_sym_05mm.nii.gz"
NMT_mask="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.1_sym/NMT_v2.1_sym_05mm/NMT_v2.1_sym_05mm_brainmask.nii.gz"

# Pad T1 image to ensure overlapping FOV with NMT
ImageMath 3 "${output_dir}T1-padded.nii.gz" PadImage "$input_T1" 8

# Denoise padded T1
DenoiseImage -d 3 -i "${output_dir}T1-padded.nii.gz" -o "${output_dir}T1-denoised.nii.gz"

# Pad NMT to avoid edge cropping
ImageMath 3 "${output_dir}NMT-padded.nii.gz" PadImage "$NMT_brain_orig" 5
NMT_brain="${output_dir}NMT-padded.nii.gz"

# Affine align T1 to NMT template
antsRegistrationSyNQuick.sh \
  -d 3 \
  -f "$NMT_brain" \
  -m "${output_dir}T1-denoised.nii.gz" \
  -o "${output_dir}aligned-" \
  -t a \
  -n 8 \
  -p f

# Apply transform to reorient T1
antsApplyTransforms \
  -d 3 \
  -i "${output_dir}T1-denoised.nii.gz" \
  -r "$NMT_brain" \
  -t "${output_dir}aligned-0GenericAffine.mat" \
  -o "${output_dir}T1-aligned.nii.gz"

# Nonlinear registration (rigid + affine + SyN)
FIXED="$NMT_brain"
MOVING="${output_dir}T1-aligned.nii.gz"
OUTPREFIX="${output_dir}registered-"

antsRegistration \
--dimensionality 3 \
--float 0 \
--collapse-output-transforms 1 \
--interpolation BSpline \
--winsorize-image-intensities [0.005,0.995] \
--use-histogram-matching 1 \
--initial-moving-transform [${FIXED},${MOVING},1] \
--transform Rigid[0.1] \
--metric MI[${FIXED},${MOVING},1,64,Regular,0.2] \
--convergence [1000x500x250x100,1e-6,10] \
--shrink-factors 8x4x2x1 \
--smoothing-sigmas 3x2x1x0vox \
--transform Affine[0.1] \
--metric MI[${FIXED},${MOVING},1,64,Regular,0.2] \
--convergence [1000x500x250x100,1e-6,10] \
--shrink-factors 8x4x2x1 \
--smoothing-sigmas 3x2x1x0vox \
--transform SyN[0.05,2,0.5] \
--metric CC[${FIXED},${MOVING},1,4] \
--convergence [300x200x100x50,1e-7,10] \
--shrink-factors 6x4x2x1 \
--smoothing-sigmas 2x1x0.5x0vox \
--output [${OUTPREFIX},${OUTPREFIX}Warped.nii.gz,${OUTPREFIX}InverseWarped.nii.gz]

# Rename the output files
mv "${output_dir}registered-Warped.nii.gz" "${output_dir}T1-registered.nii.gz"