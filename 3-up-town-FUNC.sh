#!/bin/bash

FUNC_IMG="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/patchoulli/func/topupped_mc_mean.nii.gz"
T1_IMG="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/patchoulli/T1_denoised.nii.gz"
OUTPUT_DIR="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/patchoulli/func"
OUTPUT_PREFIX="${OUTPUT_DIR}/func2T1_rigid_ants_refined"

antsRegistration --dimensionality 3 \
                 --float 0 \
                 --output [${OUTPUT_PREFIX},${OUTPUT_PREFIX}.nii.gz] \
                 --interpolation Linear \
                 --winsorize-image-intensities [0.005,0.995] \
                 --use-histogram-matching 1 \
                 --initial-moving-transform [${T1_IMG},${FUNC_IMG},1] \
                 --transform Rigid[0.1] \
                 --metric MI[${T1_IMG},${FUNC_IMG},1,32,Random,0.5] \
                 --convergence [1000x500x250x100,1e-7,15] \
                 --shrink-factors 8x4x2x1 \
                 --smoothing-sigmas 3x2x1x0vox

echo "ANTs rigid-body registration done."
echo "Transformed func saved as: ${OUTPUT_PREFIX}.nii.gz"
echo ""
echo "Inspect alignment:"
echo "fsleyes ${T1_IMG} ${OUTPUT_PREFIX}.nii.gz &"
