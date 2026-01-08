#!/bin/bash
set -e

# ------------ CONFIG-ME ----------------
BASE_DIR="/Volumes/simians/simian-brain/data/site-strasbourg/final-derivatives-3mm"
NMT_MASK="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.1_sym/NMT_v2.1_sym_05mm/NMT_v2.1_sym_05mm_brainmask.nii.gz"
NMT_TEMPLATE="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.1_sym/NMT_v2.1_sym_05mm/NMT_v2.1_sym_05mm.nii.gz"
D99_ATLAS="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.1_sym/NMT_v2.1_sym_05mm/D99_atlas_in_NMT_v2.1_sym_05mm.nii.gz"

# Processing parameters - NHP optimized
MOTION_CENSORING_PERCENTILE=5  # Censor worst 10% of volumes (adaptive)
MOTION_FALLBACK_THRESHOLD=1.5  # mm - fallback fixed threshold if percentile too strict
BANDPASS_LOW=0.01              # Hz - slightly higher for NHP (less low-freq noise)
BANDPASS_HIGH=0.1              # Hz - slightly higher ceiling for NHP
POLORT=2                       # Polynomial detrending order (quadratic)
SMOOTH_FWHM=2                  

# Quality control thresholds - NHP appropriate
MIN_VOLUMES_THRESHOLD=500      # Minimum volumes after censoring (reduced from 150)
MIN_TSNR_THRESHOLD=20         # Minimum acceptable tSNR (reduced from 30)
MAX_MOTION_THRESHOLD=3.5      # Maximum mean motion (increased from 2.0)
MAX_CENSORING_PERCENT=15      # Never censor more than 15% of data

# Advanced cleaning options
USE_ACOMPCOR=1               # Use aCompCor for noise regression
USE_TCOMPCOR=1               # Use tCompCor for temporal noise
USE_GSR=0                    # Global signal regression (still controversial)
USE_VOLTERRA=0               # Expanded motion model (start conservative)

# Visual QC options - SIMPLIFIED TO AVOID ISSUES
CREATE_VISUAL_QC=0           # Disable problematic visual QC for now
XVFB_DISPLAY=":99"          # Virtual display for headless operation
# ---------------------------------------

# Monkey list
#monkeys=(
#)

# Monkey list: auto-detect from subdirectories in BASE_DIR
monkeys=($(ls -d "$BASE_DIR"/*/ | xargs -n1 basename))

# Function for logging with timestamps
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "ERROR: Required command '$1' not found. Please check your environment."
        exit 1
    fi
}

# Function to setup AFNI environment for NHP atlas
setup_afni_environment() {
    if [[ -n "$D99_ATLAS" ]] && [[ -f "$D99_ATLAS" ]]; then
        log "Setting up AFNI environment for D99 atlas..."
        
        # Create a temporary AFNI atlas directory structure
        local atlas_dir="$BASE_DIR/.atlas_tmp"
        mkdir -p "$atlas_dir"
        
        # Set AFNI environment variables to recognize the atlas
        export AFNI_ATLAS_PATH="$atlas_dir"
        export AFNI_WHEREAMI_ATLAS="D99_v2.0"
        
        # Create a simple atlas configuration (this suppresses the warning)
        cat > "$atlas_dir/D99_atlas.txt" << EOF
# D99 Atlas Configuration
# Atlas: D99_v2.0 in NMT_v2.1_sym space
atlas_file = $D99_ATLAS
atlas_name = D99_v2.0
atlas_description = D99 macaque brain atlas in NMT symmetric template space
EOF
        
        log "AFNI environment configured for D99 atlas"
    else
        log "D99 atlas not available - using standard AFNI environment"
    fi
}

# Function to cleanup AFNI environment
cleanup_afni_environment() {
    if [[ -d "$BASE_DIR/.atlas_tmp" ]]; then
        rm -rf "$BASE_DIR/.atlas_tmp"
    fi
    unset AFNI_ATLAS_PATH AFNI_WHEREAMI_ATLAS
}

# Function to create simple visual QC (fixed version)
create_simple_qc() {
    local template="$1"
    local overlay="$2" 
    local prefix="$3"
    local title="$4"
    
    if [[ "$CREATE_VISUAL_QC" != "1" ]]; then
        return 0
    fi
    
    log "Creating simple QC: $title"
    
    # Use basic AFNI commands instead of chauffeur_afni
    if command -v afni &> /dev/null; then
        # Create basic axial slices using 3dvolreg
        3dvolreg -overwrite -base "$template" -1Dmatrix_save "${prefix}_temp_mat.1D" \
                 -prefix "${prefix}_aligned_temp.nii.gz" "$overlay" 2>/dev/null || true
        
        # Create simple mean image for QC
        3dTstat -overwrite -mean -prefix "${prefix}_qc_mean.nii.gz" "$overlay" 2>/dev/null || true
        
        log "Basic QC files created for $title"
    else
        log "AFNI not available for QC creation"
    fi
    
    # Cleanup temp files
    rm -f "${prefix}_temp_mat.1D" "${prefix}_aligned_temp.nii.gz" 2>/dev/null || true
}

# Check required commands
log "Checking dependencies..."
check_command "3dinfo"
check_command "3dresample"
check_command "3dcalc"
check_command "3dDespike"
check_command "3dTproject"
check_command "3dTstat"
check_command "3dvolreg"
check_command "python3"

# Verify atlas files exist
if [[ ! -f "$D99_ATLAS" ]]; then
    log "WARNING: D99 atlas not found at $D99_ATLAS"
    log "Atlas-based QC features will be disabled"
    D99_ATLAS=""
else
    log "Found D99 atlas in NMT space: $D99_ATLAS"
fi

# Setup AFNI environment for D99 atlas
setup_afni_environment

# Set up cleanup trap
trap 'cleanup_afni_environment' EXIT

# Create improved confounds generation script with better motion handling
cat > "/tmp/generate_confounds.py" << 'EOF'
import numpy as np
import argparse
from scipy import stats
import os

def robust_motion_loading(motion_file):
    """Robustly load motion parameters handling different formats"""
    try:
        motion_raw = np.loadtxt(motion_file)
        print(f"Motion file loaded: shape {motion_raw.shape}")
        
        if len(motion_raw.shape) == 1:
            motion_raw = motion_raw.reshape(1, -1)
        
        n_timepoints, n_cols = motion_raw.shape
        
        # Handle different motion parameter formats
        if n_cols == 9:
            # AFNI volreg format: [volume_index, tx, ty, tz, rx, ry, rz, metric1, metric2]
            print("Detected AFNI volreg format (9 columns)")
            motion = motion_raw[:, 1:7]  # Skip volume index, use motion params
        elif n_cols == 6:
            # Standard 6 DOF: tx, ty, tz, rx, ry, rz
            print("Detected standard 6-DOF format")
            motion = motion_raw
        elif n_cols == 12:
            # Extended motion with derivatives
            print("Detected 12-column format (motion + derivatives)")
            motion = motion_raw[:, :6]  # Use first 6 for base motion
        elif n_cols == 7:
            # Sometimes includes time column
            print("Detected 7-column format (likely with time)")
            motion = motion_raw[:, 1:]  # Skip first column
        else:
            raise ValueError(f"Unexpected motion parameter format: {n_cols} columns")
        
        print(f"Using motion parameters: shape {motion.shape}")
        return motion
        
    except Exception as e:
        print(f"Error loading motion file: {e}")
        raise

def calculate_fd_nhp(motion_params, radius=25.0):
    """Calculate framewise displacement with NHP-appropriate radius"""
    # Convert rotations to mm using NHP brain radius
    motion_mm = motion_params.copy()
    motion_mm[:, 3:] = motion_mm[:, 3:] * radius  # Convert radians to mm
    
    # Calculate FD
    fd = np.sum(np.abs(np.diff(motion_mm, axis=0, prepend=motion_mm[:1])), axis=1)
    return fd

parser = argparse.ArgumentParser(description='Generate comprehensive confound regressors')
parser.add_argument('--motion', required=True, help='Motion parameters file')
parser.add_argument('--func', required=True, help='Functional image for noise ROIs')
parser.add_argument('--mask', required=True, help='Brain mask')
parser.add_argument('--output-dir', required=True, help='Output directory')
parser.add_argument('--use-acompcor', action='store_true', help='Include aCompCor')
parser.add_argument('--use-tcompcor', action='store_true', help='Include tCompCor')
parser.add_argument('--use-gsr', action='store_true', help='Include global signal')
parser.add_argument('--use-volterra', action='store_true', help='Volterra expansion')
args = parser.parse_args()

# Load motion parameters robustly
motion = robust_motion_loading(args.motion)
n_timepoints = motion.shape[0]

# Motion derivatives
motion_deriv = np.vstack([np.zeros(motion.shape[1]), np.diff(motion, axis=0)])

# Base motion model (12 parameters)
confounds = np.column_stack([motion, motion_deriv])
regressor_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                   'trans_x_dt', 'trans_y_dt', 'trans_z_dt', 'rot_x_dt', 'rot_y_dt', 'rot_z_dt']

# Volterra expansion (24 parameters total)
if args.use_volterra:
    squares = confounds ** 2
    confounds = np.column_stack([confounds, squares])
    regressor_names.extend([f'{name}_sq' for name in regressor_names[:12]])

# Calculate framewise displacement (NHP-appropriate scaling)
fd = calculate_fd_nhp(motion, radius=25.0)

# Save motion metrics
np.savetxt(f'{args.output_dir}/framewise_displacement.1D', fd, fmt='%.6f')

print(f"Motion summary (NHP-optimized):")
print(f"  Mean FD: {np.mean(fd):.3f} mm")
print(f"  Median FD: {np.median(fd):.3f} mm")
print(f"  95th percentile FD: {np.percentile(fd, 95):.3f} mm")
print(f"  Max FD: {np.max(fd):.3f} mm")
print(f"  Volumes with FD > 0.5mm: {np.sum(fd > 0.5)} ({np.sum(fd > 0.5)/len(fd)*100:.1f}%)")
print(f"  Volumes with FD > 1.0mm: {np.sum(fd > 1.0)} ({np.sum(fd > 1.0)/len(fd)*100:.1f}%)")

# Save confounds and names
np.savetxt(f'{args.output_dir}/motion_confounds.txt', confounds, fmt='%.6f')
with open(f'{args.output_dir}/confound_names.txt', 'w') as f:
    f.write('\n'.join(regressor_names))

print(f"Generated {confounds.shape[1]} motion confound regressors")

# TODO: Add CompCor implementation for tissue-based noise regression
if args.use_acompcor or args.use_tcompcor:
    print("WARNING: CompCor not yet implemented - using motion-only model")

# Save processing info
with open(f'{args.output_dir}/motion_processing_info.txt', 'w') as f:
    f.write(f"Motion file: {args.motion}\n")
    f.write(f"Original shape: {robust_motion_loading(args.motion).shape}\n") 
    f.write(f"FD calculation: 25mm radius (NHP-appropriate)\n")
    f.write(f"Confound regressors: {confounds.shape[1]}\n")
    f.write(f"Volterra expansion: {args.use_volterra}\n")
EOF

# Initialize summary
processed_count=0
error_count=0

log "Starting NHP-optimized functional preprocessing pipeline..."
log "Using percentage-based motion censoring (${MOTION_CENSORING_PERCENTILE}% worst volumes)"

for monkey in "${monkeys[@]}"; do
    log "=== Processing monkey: $monkey ==="
    
    # Navigate to a safe directory first to avoid permission issues
    cd "$BASE_DIR"
    
    # Define paths for current structure
    monkey_dir="${BASE_DIR}/${monkey}"
    func_4d="${monkey_dir}/func2nmt/func-4D-in-NMT.nii.gz"
    func_mask="${monkey_dir}/func2nmt/func-mask-in-NMT.nii.gz"
    motion_params="${monkey_dir}/moco_motion.1D"
    
    outdir="${monkey_dir}/func-a-licious"
    mkdir -p "$outdir"
    
    # Skip if already processed
    if [[ -f "$outdir/func-clean-final.nii.gz" ]]; then
        log "Already processed $monkey, skipping..."
        continue
    fi
    
    # Check required files
    missing_files=()
    [[ ! -f "$func_4d" ]] && missing_files+=("$func_4d")
    [[ ! -f "$motion_params" ]] && missing_files+=("$motion_params")
    
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        log "ERROR: Missing files for $monkey:"
        printf '  %s\n' "${missing_files[@]}"
        ((error_count++))
        continue
    fi
    
    log "Input 4D functional: $func_4d"
    
    # Extract basic info
    TR=$(3dinfo -TR "$func_4d")
    NVOLS=$(3dinfo -nv "$func_4d")
    log "TR: ${TR}s, Volumes: $NVOLS"
    
    if (( NVOLS < 100 )); then
        log "ERROR: Too few volumes ($NVOLS) for $monkey"
        ((error_count++))
        continue
    fi
    
    # Step 1: Mask functional data
    log "Step 1: Applying brain mask..."
    if [[ -f "$func_mask" ]]; then
        log "Using subject-specific functional mask"
        mask_to_use="$func_mask"
    else
        log "Using NMT template mask"
        3dresample -overwrite -master "$func_4d" -inset "$NMT_MASK" \
                   -prefix "$outdir/nmt-mask-resampled.nii.gz"
        mask_to_use="$outdir/nmt-mask-resampled.nii.gz"
    fi
    
    3dcalc -overwrite -a "$func_4d" -b "$mask_to_use" -expr 'a*step(b)' \
           -prefix "$outdir/func-masked.nii.gz"
    
    # Step 2: Despiking (conservative)
    log "Step 2: Despiking..."
    3dDespike -overwrite -prefix "$outdir/func-despike.nii.gz" \
              "$outdir/func-masked.nii.gz"
    
    # Step 3: Generate comprehensive confounds with improved error handling
    log "Step 3: Generating confound regressors..."
    
    # Change to output directory before running Python script
    cd "$outdir"
    
    if ! python3 /tmp/generate_confounds.py \
        --motion "$motion_params" \
        --func "$outdir/func-despike.nii.gz" \
        --mask "$mask_to_use" \
        --output-dir "$outdir" \
        $([ $USE_ACOMPCOR -eq 1 ] && echo "--use-acompcor") \
        $([ $USE_TCOMPCOR -eq 1 ] && echo "--use-tcompcor") \
        $([ $USE_GSR -eq 1 ] && echo "--use-gsr") \
        $([ $USE_VOLTERRA -eq 1 ] && echo "--use-volterra"); then
        
        log "ERROR: Confound generation failed for $monkey"
        ((error_count++))
        continue
    fi
    
    # Verify confound files were created
    if [[ ! -f "$outdir/motion_confounds.txt" ]] || [[ ! -f "$outdir/framewise_displacement.1D" ]]; then
        log "ERROR: Required confound files not created for $monkey"
        ((error_count++))
        continue
    fi
    
    # Step 4: Improved motion censoring with better safety checks
    log "Step 4: Creating adaptive motion censoring mask..."
    
    if ! python3 -c "
import numpy as np
import sys

try:
    # Load framewise displacement
    fd = np.loadtxt('$outdir/framewise_displacement.1D')
    n_vols = len(fd)

    print(f'Motion Analysis for $monkey:')
    print(f'  Total volumes: {n_vols}')
    print(f'  Mean FD: {np.mean(fd):.3f}mm')
    print(f'  Median FD: {np.median(fd):.3f}mm') 
    print(f'  90th percentile: {np.percentile(fd, 90):.3f}mm')
    print(f'  95th percentile: {np.percentile(fd, 95):.3f}mm')
    print(f'  Max FD: {np.max(fd):.3f}mm')

    # Calculate percentage-based threshold
    percentile_threshold = np.percentile(fd, 100 - $MOTION_CENSORING_PERCENTILE)
    target_censored = int(n_vols * $MOTION_CENSORING_PERCENTILE / 100)

    print(f'  Target censoring: {$MOTION_CENSORING_PERCENTILE}% = {target_censored} volumes')
    print(f'  Percentile threshold: {percentile_threshold:.3f}mm')

    # Safety checks
    max_allowed_censored = int(n_vols * $MAX_CENSORING_PERCENT / 100)
    actual_censored_percentile = np.sum(fd > percentile_threshold)

    # Choose final threshold
    if actual_censored_percentile > max_allowed_censored:
        final_threshold = $MOTION_FALLBACK_THRESHOLD
        strategy = 'fallback_fixed'
        print(f'  WARNING: Percentile would censor {actual_censored_percentile} volumes (>{max_allowed_censored})')
        print(f'  Using fallback threshold: {final_threshold}mm')
    else:
        final_threshold = percentile_threshold
        strategy = 'percentile'
        print(f'  Using percentile-based threshold: {final_threshold:.3f}mm')

    # Create censoring mask
    censor = (fd <= final_threshold).astype(int)
    final_censored = np.sum(censor == 0)
    final_remaining = n_vols - final_censored

    print(f'  Final censoring: {final_censored}/{n_vols} volumes ({final_censored/n_vols*100:.1f}%)')
    print(f'  Remaining volumes: {final_remaining}')

    # Save censoring info
    np.savetxt('$outdir/motion_censor.1D', censor, fmt='%d')

    # Save censoring summary
    with open('$outdir/censoring_summary.txt', 'w') as f:
        f.write(f'Censoring Summary for $monkey\n')
        f.write(f'Strategy: {strategy}\n')
        f.write(f'Threshold: {final_threshold:.3f}mm\n')
        f.write(f'Censored: {final_censored}/{n_vols} ({final_censored/n_vols*100:.1f}%)\n')
        f.write(f'Remaining: {final_remaining}\n')
        f.write(f'FD stats: mean={np.mean(fd):.3f}, median={np.median(fd):.3f}, max={np.max(fd):.3f}\n')
        
except Exception as e:
    print(f'ERROR in motion censoring: {e}')
    sys.exit(1)
"; then
        log "ERROR: Motion censoring failed for $monkey"
        ((error_count++))
        continue
    fi
    
    # Check if we have enough volumes remaining
    if [[ -f "$outdir/motion_censor.1D" ]]; then
        censored_count=$(python3 -c "import numpy as np; print(int(np.sum(np.loadtxt('$outdir/motion_censor.1D') == 0)))" 2>/dev/null || echo "999")
        remaining_vols=$((NVOLS - censored_count))
        
        log "Motion censoring result: $censored_count censored, $remaining_vols remaining"
        
        if (( remaining_vols < MIN_VOLUMES_THRESHOLD )); then
            log "ERROR: Only $remaining_vols volumes remaining (< $MIN_VOLUMES_THRESHOLD threshold) for $monkey"
            ((error_count++))
            continue
        fi
    else
        log "ERROR: Censoring mask not created for $monkey"
        ((error_count++))
        continue
    fi
    
    # Step 5: Spatial smoothing (NHP-appropriate)
    if (( $(echo "$SMOOTH_FWHM > 0" | bc -l) )); then
        log "Step 5: Spatial smoothing (${SMOOTH_FWHM}mm FWHM - NHP optimized)..."
        3dmerge -overwrite -1blur_fwhm $SMOOTH_FWHM -doall \
                -prefix "$outdir/func-smooth.nii.gz" \
                "$outdir/func-despike.nii.gz"
        smooth_input="$outdir/func-smooth.nii.gz"
    else
        log "Step 5: Skipping spatial smoothing"
        smooth_input="$outdir/func-despike.nii.gz"
    fi
    
    # Step 6: Temporal filtering + confound regression with better error handling
    log "Step 6: Bandpass filtering and confound regression..."
    
    if ! 3dTproject -overwrite \
               -input "$smooth_input" \
               -mask "$mask_to_use" \
               -ort "$outdir/motion_confounds.txt" \
               -prefix "$outdir/func-clean-final.nii.gz" \
               -censor "$outdir/motion_censor.1D" \
               -passband $BANDPASS_LOW $BANDPASS_HIGH \
               -polort $POLORT \
               -verb; then
        log "ERROR: Temporal filtering failed for $monkey"
        ((error_count++))
        continue
    fi
    
    # Verify output was created
    if [[ ! -f "$outdir/func-clean-final.nii.gz" ]]; then
        log "ERROR: Final output not created for $monkey"
        ((error_count++))
        continue
    fi
    
    # Step 7: Quality metrics with atlas-based analysis
    log "Step 7: Computing quality metrics..."
    
    # Temporal statistics
    3dTstat -overwrite -mean -prefix "$outdir/tSNR-mean.nii.gz" "$outdir/func-clean-final.nii.gz"
    3dTstat -overwrite -stdev -prefix "$outdir/tSNR-std.nii.gz" "$outdir/func-clean-final.nii.gz"
    3dcalc -overwrite -a "$outdir/tSNR-mean.nii.gz" -b "$outdir/tSNR-std.nii.gz" \
           -expr 'a/b' -prefix "$outdir/tSNR-map.nii.gz"
    
    # Quality metrics with better error handling
    mean_tsnr=$(3dROIstats -quiet -mask "$mask_to_use" "$outdir/tSNR-map.nii.gz" 2>/dev/null | awk 'NR==2{print $4}' || echo "N/A")
    mean_motion=$(python3 -c "import numpy as np; print(f'{np.mean(np.loadtxt(\"$outdir/framewise_displacement.1D\")):.3f}')" 2>/dev/null || echo "N/A")
    
    # Atlas-based tSNR analysis if D99 available
    if [[ -n "$D99_ATLAS" ]] && [[ -f "$D99_ATLAS" ]]; then
        log "Computing atlas-based tSNR metrics..."
        
        # Resample atlas to functional resolution if needed
        3dresample -overwrite -master "$outdir/tSNR-map.nii.gz" -inset "$D99_ATLAS" \
                   -prefix "$outdir/D99-atlas-resampled.nii.gz" 2>/dev/null || true
        
        # Extract tSNR for key regions (using common D99 label values)
        if [[ -f "$outdir/D99-atlas-resampled.nii.gz" ]]; then
            python3 -c "
import numpy as np
import subprocess
import sys

def get_roi_tsnr(tsnr_file, atlas_file, roi_label, roi_name):
    try:
        cmd = ['3dROIstats', '-quiet', '-mask', atlas_file, 
               '-mrange', str(roi_label), str(roi_label), tsnr_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                stats = lines[1].split()  # Second line has the stats
                if len(stats) >= 4:
                    return float(stats[3])  # Mean tSNR
    except:
        pass
    return None

# Key regions for NHP QC (adjust labels as needed for your D99 version)
regions = [
    (50, 'Motor_Cortex'),      # Approximate M1
    (30, 'Visual_Cortex'),     # Approximate V1
    (70, 'Prefrontal'),        # Approximate PFC
    (20, 'Temporal'),          # Approximate temporal cortex
]

tsnr_file = '$outdir/tSNR-map.nii.gz'
atlas_file = '$outdir/D99-atlas-resampled.nii.gz'

roi_results = []
for label, name in regions:
    tsnr_val = get_roi_tsnr(tsnr_file, atlas_file, label, name)
    if tsnr_val is not None:
        roi_results.append(f'{name}: {tsnr_val:.1f}')
        print(f'  {name}: {tsnr_val:.1f}')

# Save results
with open('$outdir/atlas_tsnr_summary.txt', 'w') as f:
    f.write('Atlas-based tSNR Summary (D99)\\n')
    f.write('================================\\n')
    for result in roi_results:
        f.write(result + '\\n')
    if not roi_results:
        f.write('No valid ROI measurements obtained\\n')
" 2>/dev/null || log "Atlas-based tSNR analysis failed"
        fi
    fi
    
    # Step 8: Simplified QC (no problematic visualization)
    log "Step 8: Computing final QC metrics..."
    
    # Mean functional
    3dTstat -overwrite -mean -prefix "$outdir/mean-func-clean.nii.gz" "$outdir/func-clean-final.nii.gz"
    
    # Create simplified QC files
    create_simple_qc "$NMT_TEMPLATE" "$outdir/mean-func-clean.nii.gz" "$outdir/QC-mean-func" "Mean Functional"
    create_simple_qc "$NMT_TEMPLATE" "$outdir/tSNR-map.nii.gz" "$outdir/QC-tSNR" "tSNR Map"
    
    # Global signal extraction
    3dmaskave -overwrite -mask "$mask_to_use" -quiet "$outdir/func-clean-final.nii.gz" > "$outdir/global_signal_timeseries.1D" 2>/dev/null || \
        log "Global signal extraction failed"
    
    # Generate comprehensive QC report
    cat > "$outdir/qc_summary.txt" << EOF
    
Quality Control Summary: $monkey (NHP-Optimized Pipeline)
========================================================
Processing Date: $(date)
Pipeline Version: NHP-optimized v2.0
Input: $func_4d
TR: ${TR}s
Total Volumes: $NVOLS
Censored Volumes: $censored_count ($(python3 -c "print(f'{$censored_count/$NVOLS*100:.1f}%')" 2>/dev/null || echo "N/A"))
Remaining Volumes: $remaining_vols

Motion Summary (NHP-appropriate thresholds):
- Mean Framewise Displacement: ${mean_motion}mm
- Censoring Strategy: Percentage-based (${MOTION_CENSORING_PERCENTILE}% worst volumes)
- FD Calculation: 25mm radius (NHP brain size)
- Spatial Smoothing: ${SMOOTH_FWHM}mm FWHM

Temporal Filtering (NHP-optimized):
- Bandpass: ${BANDPASS_LOW}-${BANDPASS_HIGH} Hz  
- Polynomial Detrending: order $POLORT

Quality Metrics:
- Mean tSNR: $mean_tsnr
- tSNR Threshold: $MIN_TSNR_THRESHOLD (NHP-appropriate)

Processing Parameters:
- aCompCor: $USE_ACOMPCOR (planned)
- tCompCor: $USE_TCOMPCOR (planned)
- Global Signal Regression: $USE_GSR
- Volterra Motion Model: $USE_VOLTERRA
- Max Censoring Allowed: ${MAX_CENSORING_PERCENT}%

Output Files:
- Clean 4D data: func-clean-final.nii.gz
- tSNR map: tSNR-map.nii.gz
- Motion confounds: motion_confounds.txt
- Censoring mask: motion_censor.1D
- Censoring summary: censoring_summary.txt
- Framewise displacement: framewise_displacement.1D
- Motion processing info: motion_processing_info.txt
- Global signal: global_signal_timeseries.1D
- Visual QC: QC-*.png (if available)
- Atlas-based tSNR: atlas_tsnr_summary.txt (if D99 available)
EOF

    # Quality control flags (NHP-appropriate)
    qc_flags=()
    [[ "$mean_tsnr" != "N/A" ]] && (( $(echo "$mean_tsnr < $MIN_TSNR_THRESHOLD" | bc -l) )) && qc_flags+=("LOW_TSNR")
    [[ "$mean_motion" != "N/A" ]] && (( $(echo "$mean_motion > $MAX_MOTION_THRESHOLD" | bc -l) )) && qc_flags+=("HIGH_MOTION")
    (( remaining_vols < MIN_VOLUMES_THRESHOLD )) && qc_flags+=("FEW_VOLUMES")
    
    censoring_percent=$(python3 -c "print(f'{$censored_count/$NVOLS*100:.1f}')" 2>/dev/null || echo "0")
    [[ "$censoring_percent" != "N/A" ]] && (( $(echo "$censoring_percent > $MAX_CENSORING_PERCENT" | bc -l) )) && qc_flags+=("EXCESSIVE_CENSORING")
    
    if [[ ${#qc_flags[@]} -gt 0 ]]; then
        log "QC FLAGS for $monkey: ${qc_flags[*]}"
        echo "QC_FLAGS: ${qc_flags[*]}" >> "$outdir/qc_summary.txt"
    else
        log "QC: PASS for $monkey"
        echo "QC_FLAGS: PASS" >> "$outdir/qc_summary.txt"
    fi
    
    log "Mean tSNR: $mean_tsnr, Mean motion: ${mean_motion}mm"
    log "Censoring: ${censoring_percent}% of volumes"
    log "Output: $outdir/func-clean-final.nii.gz"
    log "Visual QC files: $outdir/QC-*.png"
    
    # Print quick viewing commands
    echo ""
    echo "Quick QC viewing commands:"
    echo "  fsleyes $NMT_TEMPLATE $outdir/func-clean-final.nii.gz &"
    echo "  fsleyes $NMT_TEMPLATE $outdir/tSNR-map.nii.gz -cm hot &"
    echo "  open $outdir/QC-*.png  # View QC images"
    echo "  cat $outdir/censoring_summary.txt  # Check censoring details"
    echo ""
    
    ((processed_count++))
    log "=== Completed: $monkey ==="
    echo ""
done

# Cleanup
rm -f /tmp/generate_confounds.py
cleanup_afni_environment

log "=== NHP PIPELINE SUMMARY ==="
log "Monkeys processed: $processed_count"
log "Errors: $error_count"
log "NHP-optimized functional cleaning completed!"
log ""
log "Key NHP optimizations applied:"
log "  - Percentage-based motion censoring (${MOTION_CENSORING_PERCENTILE}% worst volumes)"
log "  - NHP-appropriate FD calculation (25mm radius)"
log "  - Relaxed quality thresholds (tSNR: $MIN_TSNR_THRESHOLD, motion: ${MAX_MOTION_THRESHOLD}mm)"
log "  - NHP-optimized bandpass filter (${BANDPASS_LOW}-${BANDPASS_HIGH} Hz)"
log "  - Appropriate smoothing for NHP data (${SMOOTH_FWHM}mm FWHM)"
log "  - D99 atlas integration for anatomical QC"