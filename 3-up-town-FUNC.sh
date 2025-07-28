#!/bin/bash

FUNC_IMG="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/yin/func/topupped-mc-mean.nii.gz"
T1_IMG="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/yin/T1-denoised.nii.gz"
OUTPUT_DIR="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/yin/func"
OUTPUT_PREFIX="${OUTPUT_DIR}/func2T1-rigid"

if [[ ! -f "$FUNC_IMG" ]]; then
  echo "Error: Functional image not found: $FUNC_IMG"
  exit 1
fi
if [[ ! -f "$T1_IMG" ]]; then
  echo "Error: T1 image not found: $T1_IMG"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

antsRegistration --dimensionality 3 \
                 --float 0 \
                 --output [${OUTPUT_PREFIX},] \
                 --interpolation Linear \
                 --winsorize-image-intensities [0.01,0.99] \
                 --use-histogram-matching 1 \
                 --initial-moving-transform [${T1_IMG},${FUNC_IMG},1] \
                 --transform Rigid[0.1] \
                 --metric MI[${T1_IMG},${FUNC_IMG},1,256,Regular,0.3] \
                 --convergence [2000x1500x1000x500,1e-8,50] \
                 --shrink-factors 12x8x4x2 \
                 --smoothing-sigmas 4x3x2x1vox

if [[ $? -ne 0 ]]; then
  echo "Error: antsRegistration failed."
  exit 1
fi

antsApplyTransforms -d 3 \
                   -i "${FUNC_IMG}" \
                   -r "${T1_IMG}" \
                   -o "${OUTPUT_PREFIX}.nii.gz" \
                   -t "${OUTPUT_PREFIX}0GenericAffine.mat" \
                   -n Linear

if [[ $? -ne 0 ]]; then
  echo "Error: antsApplyTransforms failed."
  exit 1
fi

echo "ANTs rigid-body registration done."
echo "Transformed functional image saved as: ${OUTPUT_PREFIX}.nii.gz"

if command -v fsleyes >/dev/null 2>&1; then
  echo "Launching fsleyes for visual inspection..."
  fsleyes "${T1_IMG}" "${OUTPUT_PREFIX}.nii.gz" &
else
  echo "fsleyes not found. Please install fsleyes to inspect images visually."
fi
