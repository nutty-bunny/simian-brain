#!/bin/bash
set -e  # Exit on error

# ------------ CONFIGURABLE PARAMETERS ------------
T1_PADDING=15   # Set default padding to 0 for skull-stripped input
NMT_PADDING=15

if [[ $# -ge 1 ]]; then
    T1_PADDING=$1
    echo "Using T1 padding: $T1_PADDING voxels"
fi

if [[ $# -ge 2 ]]; then
    NMT_PADDING=$2
    echo "Using NMT padding: $NMT_PADDING voxels"
fi

DILATION_SIZE=5
VERBOSE=1
# --------------------------------------------------

# Input directory
input_dir="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/final-derivatives/yin/"

# Output directory for anat2nmt
output_dir="${input_dir}anat2nmt/"
mkdir -p "$output_dir"

# Temp directory inside output folder
temp_dir="${output_dir}temp-anat2nmt/"
mkdir -p "$temp_dir"

# *** Changed input T1 to skull-stripped brain ***
input_T1="${input_dir}T1-brain.nii.gz"
input_mask="${input_dir}T1-mask.nii.gz"

# NMT paths
NMT_brain_orig="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.1_sym/NMT_v2.1_sym_05mm/NMT_v2.1_sym_05mm.nii.gz"
NMT_mask="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.1_sym/NMT_v2.1_sym_05mm/NMT_v2.1_sym_05mm_brainmask.nii.gz"

echo "Starting robust T1 to NMT registration..."
echo "Configuration:"
echo "  T1 padding: $T1_PADDING voxels"
echo "  NMT padding: $NMT_PADDING voxels"
echo "  Dilation size: $DILATION_SIZE voxels"
echo ""

# Check files
if [[ ! -f "$input_T1" ]]; then
    echo "ERROR: Input T1 file not found: $input_T1"
    exit 1
fi
if [[ ! -f "$input_mask" ]]; then
    echo "ERROR: Input mask file not found: $input_mask"
    exit 1
fi
if [[ ! -f "$NMT_brain_orig" ]]; then
    echo "ERROR: NMT template not found: $NMT_brain_orig"
    exit 1
fi
if [[ ! -f "$NMT_mask" ]]; then
    echo "ERROR: NMT mask not found: $NMT_mask"
    exit 1
fi

# Log file
log_file="${output_dir}registration.log"
echo "Registration log - $(date)" > "$log_file"
echo "T1 padding: $T1_PADDING, NMT padding: $NMT_PADDING" >> "$log_file"

# Preprocessing
echo "Preprocessing..."

# *** Skip padding on skull-stripped T1 if T1_PADDING=0 ***
if [[ $T1_PADDING -gt 0 ]]; then
    ImageMath 3 "${temp_dir}T1-padded.nii.gz" PadImage "$input_T1" $T1_PADDING
    ImageMath 3 "${temp_dir}T1-mask-padded.nii.gz" PadImage "$input_mask" $T1_PADDING
else
    cp "$input_T1" "${temp_dir}T1-padded.nii.gz"
    cp "$input_mask" "${temp_dir}T1-mask-padded.nii.gz"
fi

# Denoise skull-stripped brain (optional but recommended)
DenoiseImage -d 3 -i "${temp_dir}T1-padded.nii.gz" -o "${temp_dir}T1-denoised.nii.gz"

# Pad NMT template and mask
if [[ $NMT_PADDING -gt 0 ]]; then
    ImageMath 3 "${temp_dir}NMT-padded.nii.gz" PadImage "$NMT_brain_orig" $NMT_PADDING
    ImageMath 3 "${temp_dir}NMT-mask-padded.nii.gz" PadImage "$NMT_mask" $NMT_PADDING
else
    cp "$NMT_brain_orig" "${temp_dir}NMT-padded.nii.gz"
    cp "$NMT_mask" "${temp_dir}NMT-mask-padded.nii.gz"
fi
NMT_brain="${temp_dir}NMT-padded.nii.gz"

# Initial affine registration (skull-stripped)
antsRegistrationSyNQuick.sh \
  -d 3 \
  -f "$NMT_brain" \
  -m "${temp_dir}T1-denoised.nii.gz" \
  -o "${temp_dir}initial-" \
  -t s \
  -n 8 \
  -p f

antsApplyTransforms \
  -d 3 \
  -i "${temp_dir}T1-denoised.nii.gz" \
  -r "$NMT_brain" \
  -t "${temp_dir}initial-0GenericAffine.mat" \
  -o "${temp_dir}T1-initial-aligned.nii.gz"

antsApplyTransforms \
  -d 3 \
  -i "${temp_dir}T1-mask-padded.nii.gz" \
  -r "$NMT_brain" \
  -t "${temp_dir}initial-0GenericAffine.mat" \
  -o "${temp_dir}T1-mask-initial-aligned.nii.gz" \
  -n NearestNeighbor

# Constrain T1 to avoid overhang using dilated NMT mask
ImageMath 3 "${temp_dir}NMT-dilated-mask.nii.gz" MD "${temp_dir}NMT-mask-padded.nii.gz" $DILATION_SIZE
ImageMath 3 "${temp_dir}T1-constrained.nii.gz" m "${temp_dir}T1-initial-aligned.nii.gz" "${temp_dir}NMT-dilated-mask.nii.gz"
ImageMath 3 "${temp_dir}T1-mask-constrained.nii.gz" m "${temp_dir}T1-mask-initial-aligned.nii.gz" "${temp_dir}NMT-dilated-mask.nii.gz"

# Final nonlinear registration
FIXED="$NMT_brain"
MOVING="${temp_dir}T1-constrained.nii.gz"
MOVING_MASK="${temp_dir}T1-mask-constrained.nii.gz"
FIXED_MASK="${temp_dir}NMT-mask-padded.nii.gz"
OUTPREFIX="${temp_dir}registered-"

antsRegistration \
--verbose $VERBOSE \
--dimensionality 3 \
--collapse-output-transforms 1 \
--interpolation Linear \
--winsorize-image-intensities [0.005,0.995] \
--use-histogram-matching 1 \
--initial-moving-transform [${FIXED},${MOVING},1] \
--masks [${FIXED_MASK},${MOVING_MASK}] \
--transform Rigid[0.1] \
--metric MI[${FIXED},${MOVING},1,32,Regular,0.25] \
--convergence [1000x500x250,1e-6,10] \
--shrink-factors 4x2x1 \
--smoothing-sigmas 2x1x0vox \
--transform Affine[0.1] \
--metric MI[${FIXED},${MOVING},1,32,Regular,0.25] \
--metric CC[${FIXED},${MOVING},0.5,4] \
--convergence [1000x500x250,1e-6,10] \
--shrink-factors 4x2x1 \
--smoothing-sigmas 2x1x0vox \
--transform SyN[0.1,3,0] \
--metric CC[${FIXED},${MOVING},1,4] \
--metric MI[${FIXED},${MOVING},0.5,32] \
--convergence [100x70x50x25,1e-6,10] \
--shrink-factors 6x4x2x1 \
--smoothing-sigmas 3x2x1x0vox \
--output [${OUTPREFIX},${OUTPREFIX}Warped.nii.gz,${OUTPREFIX}InverseWarped.nii.gz]

# Save unmasked final image
cp "${temp_dir}registered-Warped.nii.gz" "${output_dir}T1-registered.nii.gz"

# QC outputs
CreateTiledMosaic -i "${output_dir}T1-registered.nii.gz" -r "$NMT_brain" -o "${output_dir}QC-overlay.png" -t -1x-1 -d 2 -p mask -s [5,mask,mask] -x "${temp_dir}NMT-mask-padded.nii.gz" -a 0.5
LabelOverlapMeasures 3 "${temp_dir}NMT-mask-padded.nii.gz" "${temp_dir}T1-mask-constrained.nii.gz" "${output_dir}overlap-metrics.csv"

# Save transforms
cp "${temp_dir}initial-0GenericAffine.mat" "${output_dir}T1-to-NMT-0GenericAffine.mat"
cp "${temp_dir}registered-0GenericAffine.mat" "${output_dir}T1-to-NMT-1GenericAffine.mat"
cp "${temp_dir}registered-1Warp.nii.gz" "${output_dir}T1-to-NMT-1Warp.nii.gz"

# Settings summary
cat > "${output_dir}registration-settings.txt" << EOF
ANTs Registration Settings Summary
=================================
Date: $(date)
T1 input: $input_T1
NMT template: $NMT_brain_orig
Padding:
- T1: $T1_PADDING
- NMT: $NMT_PADDING
EOF

echo ""
echo "Robust registration completed!"
echo "All outputs saved to: ${output_dir}"