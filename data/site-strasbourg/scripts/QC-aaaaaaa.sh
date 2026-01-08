#!/bin/bash
set -e

# ---------------- CONFIG ----------------
#BASE_DIR="/Volumes/simians/simian-brain/data/site-strasbourg/final-derivatives-no-spatial-smoothing"
BASE_DIR="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/final-derivatives"
NMT_MASK="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.1_sym/NMT_v2.1_sym_05mm/NMT_v2.1_sym_05mm_brainmask.nii.gz"
OUTPUT_FILE="$BASE_DIR/tsnr_qc_summary_all.txt"
# ---------------------------------------

# Initialize summary file only if it doesn’t already exist
if [[ ! -f "$OUTPUT_FILE" ]]; then
    echo "Monkey fMRI Quality Control Summary" > "$OUTPUT_FILE"
    echo "===================================" >> "$OUTPUT_FILE"
    echo "Generated: $(date)" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "Subject      Mean_tSNR  Median_tSNR  Min_tSNR   Max_tSNR   N_Voxels" >> "$OUTPUT_FILE"
    echo "--------------------------------------------------------------------------------------" >> "$OUTPUT_FILE"
fi

# Loop over monkeys
for monkey_dir in "$BASE_DIR"/*/; do
    monkey=$(basename "$monkey_dir")
    FUNC_CLEAN="$monkey_dir/func-a-licious/func-despike.nii.gz"
    
    # Skip if already processed
    if grep -q "^$monkey " "$OUTPUT_FILE"; then
        echo "Skipping $monkey: already in summary"
        continue
    fi
    
    if [[ ! -f "$FUNC_CLEAN" ]]; then
        echo "Skipping $monkey: func-despike.nii.gz not found"
        continue
    fi
    
    echo "Processing $monkey..."
    cd "$monkey_dir/func-a-licious/"
    
    # Resample NMT brain mask to functional resolution
    RESAMP_MASK="nmt_mask_resampled.nii.gz"
    3dresample -overwrite -master "$FUNC_CLEAN" -inset "$NMT_MASK" -prefix "$RESAMP_MASK" 2>/dev/null
    
    # Compute voxel-wise mean and std inside brain mask
    MEAN_FILE="tsnr_mean_tmp.nii.gz"
    STD_FILE="tsnr_std_tmp.nii.gz"
    TMAP="tsnr-map.nii.gz"
    
    3dTstat -overwrite -mean -mask "$RESAMP_MASK" -prefix "$MEAN_FILE" "$FUNC_CLEAN" 2>/dev/null
    3dTstat -overwrite -stdev -mask "$RESAMP_MASK" -prefix "$STD_FILE" "$FUNC_CLEAN" 2>/dev/null
    
    # Compute tSNR: mean / std (voxel-wise) with better division by zero protection
    3dcalc -overwrite -a "$MEAN_FILE" -b "$STD_FILE" -expr 'step(b-0.01)*a/max(b,0.01)' -prefix "$TMAP" 2>/dev/null
    
    # Get statistics BEFORE deleting temp files
    MEAN_TSNR=$(3dROIstats -mask "$RESAMP_MASK" "$TMAP" 2>/dev/null | tail -n1 | awk '{print $3}')
    MEDIAN_TSNR=$(3dBrickStat -mask "$RESAMP_MASK" -percentile 50 50 50 "$TMAP" 2>/dev/null | awk '{print $2}')
    TSNR_MIN=$(3dBrickStat -mask "$RESAMP_MASK" -min "$TMAP" 2>/dev/null | awk '{print $1}')
    TSNR_MAX=$(3dBrickStat -mask "$RESAMP_MASK" -max "$TMAP" 2>/dev/null | awk '{print $1}')
    
    # Count voxels
    N_VOXELS=$(3dmaskave -mask "$RESAMP_MASK" -quiet "$RESAMP_MASK" 2>&1 | grep "voxels survive" | awk '{print $2}')
    
    # Clean up numeric values and handle empty cases
    MEAN_TSNR=${MEAN_TSNR:-0}
    MEDIAN_TSNR=${MEDIAN_TSNR:-0}
    TSNR_MIN=${TSNR_MIN:-0}
    TSNR_MAX=${TSNR_MAX:-0}
    N_VOXELS=${N_VOXELS:-0}
    
    # Format and output results
    printf "%-12s %-10.2f %-12.2f %-10.2f %-10.2f %-8s\n" \
        "$monkey" "$MEAN_TSNR" "$MEDIAN_TSNR" "$TSNR_MIN" "$TSNR_MAX" "$N_VOXELS" | tee -a "$OUTPUT_FILE"
    
    # QC warnings
    if (( $(echo "$MEAN_TSNR < 20" | bc -l 2>/dev/null || echo "0") )); then
        echo "  ⚠️  WARNING: Low tSNR for $monkey (< 20)" | tee -a "$OUTPUT_FILE"
    fi
    
    # Clean up temp files
    rm -f "$MEAN_FILE" "$STD_FILE"
done
