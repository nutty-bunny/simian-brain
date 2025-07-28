#!/bin/bash

# Paths
FUNC="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives//Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/arwen/T1-registered.nii.gz/func/preproc/func-clean.nii.gz"
MASK="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.1_sym/NMT_v2.1_sym_05mm/NMT_v2.1_sym_05mm_brainmask.nii.gz"
OUTDIR="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives//Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/arwen/T1-registered.nii.gz/melodic"
BGIMG="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/arwen/T1-registered.nii.gz"

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
  --bgimage="$BGIMG" \
  --report