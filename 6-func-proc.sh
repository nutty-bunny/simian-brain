#!/bin/bash
set -e

# ------------ FILLED-IN USER CONFIG ----------------
FUNC="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/patchoulli/func/func_in_NMT.nii.gz"
MC_PAR="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/patchoulli/func/topupped_mc.par"
SEGMENT="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.0_sym/NMT_v2.0_sym_05mm/NMT_v2.0_sym_05mm_segmentation.nii.gz"
MASK="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.0_sym/NMT_v2.0_sym_05mm/NMT_v2.0_sym_05mm_brainmask.nii.gz"
TR=1.45
SCRIPT_DIR="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/scripts"
OUTDIR=$(dirname "$FUNC")/preproc
# ----------------------------------------------------

mkdir -p "$OUTDIR"
3dresample \
  -master "$FUNC" \
  -inset "$MASK" \
  -prefix "$OUTDIR/brainmask_resampled.nii.gz"

echo "Step 1: Despiking using brain mask"
# Apply brain mask first to reduce data size
3dcalc -a "$FUNC" -b "$OUTDIR/brainmask_resampled.nii.gz" -expr 'a*b' -prefix "$OUTDIR/func_masked.nii.gz"

# Now despike the masked version
3dDespike -prefix "$OUTDIR/func_despike.nii.gz" "$OUTDIR/func_masked.nii.gz"

echo "Step 2: High-pass filtering (cutoff ~0.01Hz)"
SIGMA=$(echo "2000 / (2 * $TR)" | bc -l)
fslmaths "$OUTDIR/func_despike.nii.gz" -bptf $SIGMA -1 "$OUTDIR/func_filtered.nii.gz"

echo "Step 3: WM & CSF masks"
fslmaths "$SEGMENT" -thr 4 -uthr 4 -bin "$OUTDIR/wm_mask.nii.gz"
fslmaths "$SEGMENT" -thr 1 -uthr 1 -bin "$OUTDIR/csf_mask.nii.gz"

echo "Step 4: Extract WM & CSF time series"
fslmeants -i "$OUTDIR/func_filtered.nii.gz" -m "$OUTDIR/wm_mask.nii.gz" -o "$OUTDIR/wm.txt"
fslmeants -i "$OUTDIR/func_filtered.nii.gz" -m "$OUTDIR/csf_mask.nii.gz" -o "$OUTDIR/csf.txt"

echo "Step 5: Create combined confounds"
python3 "$SCRIPT_DIR/6-confounds.py" \
  --wm "$OUTDIR/wm.txt" \
  --csf "$OUTDIR/csf.txt" \
  --mc "$MC_PAR" \
  --volterra \
  --output "$OUTDIR/all_confounds.txt"

echo "Step 6: Regress confounds"
N_CONF=$(awk '{print NF}' "$OUTDIR/all_confounds.txt" | head -n 1)
fsl_regfilt -i "$OUTDIR/func_filtered.nii.gz" \
            -d "$OUTDIR/all_confounds.txt" \
            -f $(seq -s "," 1 $N_CONF) \
            -o "$OUTDIR/func_clean.nii.gz"

echo "Step 7: Compute mean functional image"
fslmaths "$OUTDIR/func_clean.nii.gz" -Tmean "$OUTDIR/mean_func.nii.gz"

echo "DONE. Output: $OUTDIR/func_clean.nii.gz"
