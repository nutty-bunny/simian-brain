#!/bin/bash
set -e

# ------------ CONFIG-ME ----------------
FUNC="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/iron/func/func-in-NMT.nii.gz"
MC_PAR="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/iron/func/topupped-mc.par"
SEGMENT="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.1_sym/NMT_v2.1_sym_05mm/NMT_v2.1_sym_05mm_segmentation.nii.gz"
MASK="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.1_sym/NMT_v2.1_sym_05mm/NMT_v2.1_sym_05mm_brainmask.nii.gz"
TR=1.45
SCRIPT_DIR="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/scripts"
OUTDIR=$(dirname "$FUNC")/func-a-licious
# ---------------------------------------

mkdir -p "$OUTDIR"

echo "Resample brain mask to func space"
3dresample -master "$FUNC" -inset "$MASK" -prefix "$OUTDIR/brainmask-resampled.nii.gz"

echo "Despiking using brain mask"
3dcalc -a "$FUNC" -b "$OUTDIR/brainmask-resampled.nii.gz" -expr 'a*b' -prefix "$OUTDIR/func-masked.nii.gz"
3dDespike -prefix "$OUTDIR/func-despike.nii.gz" "$OUTDIR/func-masked.nii.gz"
 
echo "Create combined confounds (no Volterra)" # If want to include Volterra expansions, add --volterra flag
python3 "$SCRIPT_DIR/6-confounds.py" \
  --mc "$MC_PAR" \
  --output "$OUTDIR/all-confounds.txt"

1d_tool.py -overwrite \
           -infile "$MC_PAR" \
           -set_nruns 1 \
           -show_censor_count \
           -censor_motion 0.2 censor_temp

mv censor_temp_censor.1D "$OUTDIR/censor.1D"

echo "Bandpass filtering + confound regression using AFNI 3dTproject"
3dTproject -input "$OUTDIR/func-despike.nii.gz" \
          -mask "$OUTDIR/brainmask-resampled.nii.gz" \
          -ort "$OUTDIR/all-confounds.txt" \
          -prefix "$OUTDIR/func-clean.nii.gz" \
          -censor "$OUTDIR/censor.1D" \
          -bandpass 0.005 0.1 \
          -polort 0

echo "Compute mean functional image"
3dTstat -mean -prefix "$OUTDIR/mean-func.nii.gz" "$OUTDIR/func-clean.nii.gz"

echo "DONE. Output: $OUTDIR/func-clean.nii.gz"