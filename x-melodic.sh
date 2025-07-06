#!/bin/bash

# Paths
FUNC="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/patchoulli/func/preproc/func_clean.nii.gz"
MASK="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.0_sym/NMT_v2.0_sym_05mm/NMT_v2.0_sym_05mm_brainmask.nii.gz"
OUTDIR="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/patchoulli/melodic"
BGIMG="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.0_sym/NMT_v2.0_sym_05mm/NMT_v2.0_sym_05mm.nii.gz"

# Create output directory
mkdir -p "$OUTDIR"

# Run MELODIC
melodic \
  -i "$FUNC" \
  -o "$OUTDIR" \
  --mask="$MASK" \
  --tr=1.45 \
  --nobet \
  --mmthresh=0.5 \
  --numICs=30 \
  --bgimage="$BGIMG" \
  --report