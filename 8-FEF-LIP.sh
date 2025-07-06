#!/bin/bash

set -euo pipefail

# Paths
FUNC="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/patchoulli/func/preproc/func_clean.nii.gz"
ROI_DIR="/Users/similovesyou/Desktop/qts/simian-brain/NMT-ROIs"
OUTDIR="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/patchoulli/func/connectivity"
mkdir -p "$OUTDIR"

# ROI masks
FEF_ROI="$ROI_DIR/FEF.nii.gz"
LIP_ROI="$ROI_DIR/LIP_joint.nii.gz"

# Resample ROIs to functional space if needed
for roi in "$FEF_ROI" "$LIP_ROI"; do
    resampled="$OUTDIR/$(basename $roi)"
    if ! fslstats "$roi" -R | awk '{exit ($1!=0 || $2!=$(NF))}'; then
        # Basic check failed: resample
        echo "Resampling $(basename $roi) to functional space..."
        flirt -in "$roi" -ref "$FUNC" -out "$resampled" -applyxfm -usesqform -interp nearestneighbour
    else
        echo "$(basename $roi) looks aligned. Copying to output."
        cp "$roi" "$resampled"
    fi
done

FEF_RESAMP="$OUTDIR/FEF.nii.gz"
LIP_RESAMP="$OUTDIR/LIP_joint.nii.gz"

# Extract mean timeseries
echo "Extracting mean time series..."
fslmeants -i "$FUNC" -m "$FEF_RESAMP" > "$OUTDIR/FEF_ts.txt"
fslmeants -i "$FUNC" -m "$LIP_RESAMP" > "$OUTDIR/LIP_ts.txt"

# Compute Pearson correlation in Python
python3 << EOF
import numpy as np

fe_ts = np.loadtxt("$OUTDIR/FEF_ts.txt")
lip_ts = np.loadtxt("$OUTDIR/LIP_ts.txt")

corr = np.corrcoef(fe_ts, lip_ts)[0,1]
print(f"Pearson correlation between FEF and LIP time series: {corr:.3f}")

with open("$OUTDIR/fc_result.txt", "w") as f:
    f.write(f"Pearson correlation: {corr:.3f}\n")
EOF

echo "Done! Results saved in $OUTDIR"
