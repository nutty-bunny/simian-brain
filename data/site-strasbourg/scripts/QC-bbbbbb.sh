#!/bin/bash
set -e

# ------------ CONFIG ----------------
BASE_DIR="/Volumes/simians/simian-brain/data/site-strasbourg/final-derivatives-no-spatial-smoothing"
TSNR_QC_FILE="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/final-derivatives/tsnr_qc_summary.txt"
OUTPUT_FILE="$BASE_DIR/preprocessing_summary_report.txt"
CSV_OUTPUT="$BASE_DIR/preprocessing_summary.csv"
# ------------------------------------

# Function to extract tSNR data from the centralized tSNR QC file
get_tsnr_data() {
    local monkey="$1"
    local tsnr_file="$2"
    
    if [[ ! -f "$tsnr_file" ]]; then
        echo "N/A N/A N/A N/A N/A"
        return
    fi
    
    # Extract line for this monkey (skip header and warning lines)
    local tsnr_line=$(grep "^$monkey " "$tsnr_file" 2>/dev/null || echo "")
    
    if [[ -n "$tsnr_line" ]]; then
        # Parse the line: Subject Mean_tSNR Median_tSNR Min_tSNR Max_tSNR N_Voxels
        local mean_tsnr=$(echo "$tsnr_line" | awk '{print $2}')
        local median_tsnr=$(echo "$tsnr_line" | awk '{print $3}')
        local min_tsnr=$(echo "$tsnr_line" | awk '{print $4}')
        local max_tsnr=$(echo "$tsnr_line" | awk '{print $5}')
        local n_voxels=$(echo "$tsnr_line" | awk '{print $6}')
        
        echo "$mean_tsnr $median_tsnr $min_tsnr $max_tsnr $n_voxels"
    else
        echo "N/A N/A N/A N/A N/A"
    fi
}

# Function to determine tSNR quality category (NHP-appropriate thresholds)
get_tsnr_quality() {
    local mean_tsnr="$1"
    
    if [[ "$mean_tsnr" == "N/A" ]]; then
        echo "UNKNOWN"
    elif (( $(echo "$mean_tsnr >= 25" | bc -l 2>/dev/null || echo "0") )); then
        echo "GOOD"
    elif (( $(echo "$mean_tsnr >= 20" | bc -l 2>/dev/null || echo "0") )); then
        echo "ACCEPTABLE"
    else
        echo "POOR"
    fi
}

# Initialize summary files
echo "NHP fMRI Preprocessing Results Summary" > "$OUTPUT_FILE"
echo "=====================================" >> "$OUTPUT_FILE"
echo "Generated: $(date)" >> "$OUTPUT_FILE"
echo "Pipeline: NHP-optimized v2.0" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# CSV header
echo "Monkey,Total_Volumes,Censored_Volumes,Censoring_Percent,Remaining_Volumes,Mean_FD,Median_FD,Max_FD,Censoring_Threshold,Mean_tSNR,Median_tSNR,Min_tSNR,Max_tSNR,tSNR_Quality,QC_Status,Processing_Status" > "$CSV_OUTPUT"

# Track summary statistics
total_monkeys=0
processed_monkeys=0
passed_qc=0
failed_qc=0
total_volumes=0
total_censored=0
fd_values=()
tsnr_values=()
good_tsnr=0
acceptable_tsnr=0
poor_tsnr=0

echo "INDIVIDUAL MONKEY RESULTS" >> "$OUTPUT_FILE"
echo "=========================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Loop through all monkey directories
for monkey_dir in "$BASE_DIR"/*/; do
    if [[ ! -d "$monkey_dir" ]]; then continue; fi
    
    monkey=$(basename "$monkey_dir")
    func_dir="$monkey_dir/func-a-licious"
    
    echo "Processing $monkey..."
    ((total_monkeys++))
    
    # Check if preprocessing was completed
    if [[ ! -d "$func_dir" ]] || [[ ! -f "$func_dir/func-clean-final.nii.gz" ]]; then
        echo "$monkey: PREPROCESSING INCOMPLETE" >> "$OUTPUT_FILE"
        echo "$monkey,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,UNKNOWN,INCOMPLETE,FAILED" >> "$CSV_OUTPUT"
        echo "" >> "$OUTPUT_FILE"
        continue
    fi
    
    ((processed_monkeys++))
    
    # Extract data from files
    qc_file="$func_dir/qc_summary.txt"
    censoring_file="$func_dir/censoring_summary.txt"
    
    # Extract tSNR data from centralized file
    tsnr_data=($(get_tsnr_data "$monkey" "$TSNR_QC_FILE"))
    mean_tsnr="${tsnr_data[0]}"
    median_tsnr="${tsnr_data[1]}"
    min_tsnr="${tsnr_data[2]}"
    max_tsnr="${tsnr_data[3]}"
    n_voxels="${tsnr_data[4]}"
    tsnr_quality=$(get_tsnr_quality "$mean_tsnr")
    
    # Basic info from QC file
    if [[ -f "$qc_file" ]]; then
        total_vols=$(grep "Total Volumes:" "$qc_file" | cut -d':' -f2 | xargs 2>/dev/null || echo "N/A")
        tr_value=$(grep "TR:" "$qc_file" | cut -d':' -f2 | sed 's/s//' | xargs 2>/dev/null || echo "N/A")
        remaining_vols=$(grep "Remaining Volumes:" "$qc_file" | cut -d':' -f2 | xargs 2>/dev/null || echo "N/A")
        censored_vols=$(grep "Censored Volumes:" "$qc_file" | cut -d':' -f2 | cut -d'(' -f1 | xargs 2>/dev/null || echo "N/A")
        mean_fd=$(grep "Mean Framewise Displacement:" "$qc_file" | cut -d':' -f2 | sed 's/mm//' | xargs 2>/dev/null || echo "N/A")
        qc_flags=$(grep "QC_FLAGS:" "$qc_file" | cut -d':' -f2 | xargs 2>/dev/null || echo "UNKNOWN")
    else
        total_vols="N/A"; tr_value="N/A"; remaining_vols="N/A"
        censored_vols="N/A"; mean_fd="N/A"; qc_flags="UNKNOWN"
    fi
    
    # Censoring details
    if [[ -f "$censoring_file" ]]; then
        threshold=$(grep "Threshold:" "$censoring_file" | cut -d':' -f2 | sed 's/mm//' | xargs 2>/dev/null || echo "N/A")
        # Extract FD stats from censoring file
        fd_mean=$(grep "FD stats:" "$censoring_file" | sed 's/.*mean=\([0-9.]*\).*/\1/' || echo "N/A")
        fd_median=$(grep "FD stats:" "$censoring_file" | sed 's/.*median=\([0-9.]*\).*/\1/' || echo "N/A")
        fd_max=$(grep "FD stats:" "$censoring_file" | sed 's/.*max=\([0-9.]*\).*/\1/' || echo "N/A")
        
        # Extract censoring percentage
        censoring_line=$(grep "Censored:" "$censoring_file" || echo "")
        if [[ -n "$censoring_line" ]]; then
            censoring_percent=$(echo "$censoring_line" | sed 's/.*(\([0-9.]*\)%).*/\1/' || echo "N/A")
        else
            censoring_percent="N/A"
        fi
    else
        threshold="N/A"; fd_mean="N/A"; fd_median="N/A"; fd_max="N/A"; censoring_percent="N/A"
    fi
    
    # Determine processing status with tSNR flagging
    if [[ "$qc_flags" == "PASS" ]]; then
        if [[ "$tsnr_quality" == "POOR" ]]; then
            status="PASS_BUT_POOR_TSNR"
            ((failed_qc++))  # Count as failed for exclusion purposes
        else
            status="PASS"
            ((passed_qc++))
        fi
    elif [[ "$qc_flags" == "UNKNOWN" ]]; then
        status="UNKNOWN"
    else
        status="FLAGGED"
        ((failed_qc++))
    fi
    
    # Add to summary statistics
    if [[ "$total_vols" != "N/A" ]]; then
        total_volumes=$((total_volumes + total_vols))
    fi
    if [[ "$censored_vols" != "N/A" ]]; then
        total_censored=$((total_censored + censored_vols))
    fi
    if [[ "$fd_mean" != "N/A" ]]; then
        fd_values+=("$fd_mean")
    fi
    if [[ "$mean_tsnr" != "N/A" ]]; then
        tsnr_values+=("$mean_tsnr")
        case "$tsnr_quality" in
            "GOOD") ((good_tsnr++)) ;;
            "ACCEPTABLE") ((acceptable_tsnr++)) ;;
            "POOR") ((poor_tsnr++)) ;;
        esac
    fi
    
    # Write individual results with tSNR flagging
    tsnr_flag=""
    if [[ "$tsnr_quality" == "POOR" ]]; then
        tsnr_flag=" ⚠️ RECOMMENDED FOR EXCLUSION (tSNR < 20)"
    fi
    
    cat >> "$OUTPUT_FILE" << EOF
$monkey (Status: $status, tSNR: $tsnr_quality)$tsnr_flag
$(printf '=%.0s' {1..50})
- Total Volumes: $total_vols (TR: ${tr_value}s)
- Censored: $censored_vols volumes (${censoring_percent}%)
- Remaining: $remaining_vols volumes
- Motion Summary:
  * Mean FD: ${fd_mean}mm
  * Median FD: ${fd_median}mm  
  * Max FD: ${fd_max}mm
  * Censoring threshold: ${threshold}mm
- tSNR Quality Metrics:
  * Mean tSNR: $mean_tsnr ($tsnr_quality)
  * Median tSNR: $median_tsnr
  * Min tSNR: $min_tsnr
  * Max tSNR: $max_tsnr
  * Brain voxels: $n_voxels
- QC Status: $qc_flags

EOF

    # Write to CSV
    echo "$monkey,$total_vols,$censored_vols,$censoring_percent,$remaining_vols,$fd_mean,$fd_median,$fd_max,$threshold,$mean_tsnr,$median_tsnr,$min_tsnr,$max_tsnr,$tsnr_quality,$status,$status" >> "$CSV_OUTPUT"
done

# Calculate group statistics
if [[ ${#fd_values[@]} -gt 0 ]]; then
    # Calculate mean FD across all subjects
    fd_sum=0
    for fd in "${fd_values[@]}"; do
        fd_sum=$(echo "$fd_sum + $fd" | bc -l)
    done
    mean_group_fd=$(echo "scale=3; $fd_sum / ${#fd_values[@]}" | bc -l)
else
    mean_group_fd="N/A"
fi

if [[ ${#tsnr_values[@]} -gt 0 ]]; then
    # Calculate mean tSNR across all subjects
    tsnr_sum=0
    for tsnr in "${tsnr_values[@]}"; do
        tsnr_sum=$(echo "$tsnr_sum + $tsnr" | bc -l)
    done
    mean_group_tsnr=$(echo "scale=2; $tsnr_sum / ${#tsnr_values[@]}" | bc -l)
else
    mean_group_tsnr="N/A"
fi

avg_volumes_per_monkey=$((total_monkeys > 0 ? total_volumes / processed_monkeys : 0))
avg_censored_per_monkey=$((processed_monkeys > 0 ? total_censored / processed_monkeys : 0))
avg_censoring_percent=$(echo "scale=1; $total_censored * 100 / $total_volumes" | bc -l 2>/dev/null || echo "N/A")

# Write group summary
cat >> "$OUTPUT_FILE" << EOF

GROUP SUMMARY
=============
Total Monkeys Found: $total_monkeys
Successfully Processed: $processed_monkeys
QC Status:
- PASS (good quality): $passed_qc
- FLAGGED/FAILED/POOR_tSNR: $failed_qc
- RECOMMENDED FOR ANALYSIS: $passed_qc subjects
- RECOMMENDED FOR EXCLUSION: $failed_qc subjects (poor tSNR)

Volume Statistics:
- Total volumes across all monkeys: $total_volumes
- Total censored volumes: $total_censored
- Average volumes per monkey: $avg_volumes_per_monkey
- Average censored per monkey: $avg_censored_per_monkey
- Group censoring rate: ${avg_censoring_percent}%

Motion Statistics:
- Group mean FD: ${mean_group_fd}mm

tSNR Quality Distribution:
- Good (≥25): $good_tsnr subjects
- Acceptable (20-24.9): $acceptable_tsnr subjects  
- Poor (<20): $poor_tsnr subjects [EXCLUSION RECOMMENDED]
- Group mean tSNR: ${mean_group_tsnr}

QC Guidelines Applied (NHP-appropriate):
- Mean tSNR: Good ≥25, Acceptable 20-24.9, Poor <20 (exclusion threshold)
- Motion threshold: 3.5mm mean FD
- Max censoring allowed: 15%

Processing Parameters Used:
- Censoring strategy: Percentage-based (5% worst volumes)
- FD calculation: 25mm radius (NHP-appropriate)
- Bandpass filter: 0.01-0.1 Hz
- tSNR threshold: 20 (NHP-appropriate)
- Motion threshold: 3.5mm
- Max censoring allowed: 15%

FILES GENERATED:
- Detailed report: $(basename "$OUTPUT_FILE")
- CSV summary: $(basename "$CSV_OUTPUT")

EOF

echo ""
echo "=== PREPROCESSING SUMMARY COMPLETE ==="
echo "Processed: $processed_monkeys/$total_monkeys monkeys"
echo "QC Pass rate: $passed_qc/$processed_monkeys ($(echo "scale=1; $passed_qc * 100 / $processed_monkeys" | bc -l 2>/dev/null || echo "N/A")%) - RECOMMENDED FOR ANALYSIS"
echo "Exclusion recommended: $failed_qc/$processed_monkeys ($(echo "scale=1; $failed_qc * 100 / $processed_monkeys" | bc -l 2>/dev/null || echo "N/A")%) - POOR tSNR"
echo "Group mean FD: ${mean_group_fd}mm"
echo "Group mean tSNR: ${mean_group_tsnr}"
echo "Group censoring rate: ${avg_censoring_percent}%"
echo "tSNR quality: Good=$good_tsnr, Acceptable=$acceptable_tsnr, Poor=$poor_tsnr"
echo ""
echo "SUBJECTS RECOMMENDED FOR EXCLUSION (tSNR < 20):"
if [[ $poor_tsnr -gt 0 ]]; then
    echo "Check the detailed report for subjects marked with: ⚠️ RECOMMENDED FOR EXCLUSION"
else
    echo "None"
fi
echo ""
echo "Reports saved to:"
echo "  - $OUTPUT_FILE"
echo "  - $CSV_OUTPUT"
echo ""
echo "Quick viewing commands:"
echo "  cat '$OUTPUT_FILE' | less"
echo "  open '$CSV_OUTPUT'  # View in spreadsheet"