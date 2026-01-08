#!/bin/bash

# Functional to anatomical registration for multiple monkeys
# Aligns moco_mean.nii.gz to T1-reoriented-cropped.nii.gz
# Creates func2anat directory with transforms and QC outputs

set -e  # Exit on error

# Configurable parameters
VERBOSE=1
CREATE_QC=1
USE_SKULL_STRIPPED=1  # Use skull-stripped for better registration

# Monkey list
monkeys=(
    "yin"
)

for monkey in "${monkeys[@]}"; do
    echo "Processing monkey: $monkey"
    
    # Paths for this monkey
    base_dir="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/final-derivatives/${monkey}/"
    
    # Input files
    func_img="${base_dir}moco_mean.nii.gz"
    func_mask="${base_dir}T1-in-func-mask.nii.gz"
    
    # Anatomical reference options
    if [[ $USE_SKULL_STRIPPED -eq 1 ]]; then
        anat_ref="${base_dir}T1-brain.nii.gz"  # Skull-stripped for better registration
        anat_mask="${base_dir}T1-mask.nii.gz"
        echo "Using skull-stripped T1 as reference"
    else
        anat_ref="${base_dir}T1-reoriented-cropped.nii.gz"  # Full head
        anat_mask=""
        echo "Using full-head T1 as reference"
    fi
    
    # Output directory
    output_dir="${base_dir}func2anat/"
    mkdir -p "$output_dir"
    temp_dir="${output_dir}temp/"
    mkdir -p "$temp_dir"
    
    echo "=== Functional to Anatomical Registration ==="
    echo "Functional image: $func_img"
    echo "Functional mask: $func_mask"
    echo "Anatomical reference: $anat_ref"
    echo "Output directory: $output_dir"
    echo ""
    
    # Check input files
    if [[ ! -f "$func_img" ]]; then
        echo "ERROR: Functional image not found: $func_img"
        continue
    fi
    if [[ ! -f "$anat_ref" ]]; then
        echo "ERROR: Anatomical reference not found: $anat_ref"
        continue
    fi
    if [[ $USE_SKULL_STRIPPED -eq 1 && ! -f "$anat_mask" ]]; then
        echo "ERROR: Anatomical mask not found: $anat_mask"
        continue
    fi
    
    # Check if functional mask exists, if not create it from T1-mask
    if [[ ! -f "$func_mask" ]]; then
        echo "WARNING: Functional mask not found: $func_mask"
        if [[ -f "$anat_mask" ]]; then
            echo "Creating T1-in-func-mask from T1-mask..."
            
            # Extract a single volume from functional data for preliminary registration
            func_dims=$(fslval "$func_img" dim4 2>/dev/null || echo "1")
            if [[ $func_dims -gt 1 ]]; then
                middle_vol=$((func_dims / 2))
                fslroi "$func_img" "${temp_dir}func_single_vol_for_mask.nii.gz" $middle_vol 1
                func_for_mask="${temp_dir}func_single_vol_for_mask.nii.gz"
            else
                func_for_mask="$func_img"
            fi
            
            # Quick rigid registration from anatomical to functional space
            echo "Performing quick registration to create functional mask..."
            antsRegistrationSyNQuick.sh \
              -d 3 \
              -f "$func_for_mask" \
              -m "$anat_ref" \
              -o "${temp_dir}anat2func-" \
              -t r \
              -n 4 \
              -p f
            
            # Transform T1-mask to functional space
            antsApplyTransforms \
              -d 3 \
              -i "$anat_mask" \
              -r "$func_for_mask" \
              -t "${temp_dir}anat2func-0GenericAffine.mat" \
              -o "$func_mask" \
              -n NearestNeighbor
            
            echo "Created T1-in-func-mask: $func_mask"
        else
            echo "ERROR: Cannot create functional mask - T1-mask not found: $anat_mask"
            continue
        fi
    fi
    
    # Extract a single volume from functional data if it's 4D
    echo "Checking functional image dimensions..."
    func_dims=$(fslval "$func_img" dim4 2>/dev/null || echo "1")
    if [[ $func_dims -gt 1 ]]; then
        echo "4D functional detected, extracting middle volume for registration..."
        middle_vol=$((func_dims / 2))
        fslroi "$func_img" "${temp_dir}func_single_vol.nii.gz" $middle_vol 1
        moving_img="${temp_dir}func_single_vol.nii.gz"
    else
        echo "3D functional image detected"
        moving_img="$func_img"
    fi
    
    # Use existing or newly created T1-in-func mask
    echo "Using T1-in-func mask for registration..."
    
    # Registration strategy: Rigid + Affine
    # Rigid first (6 DOF) - good for initial alignment
    echo "Step 1: Rigid registration (6 DOF)..."
    antsRegistrationSyNQuick.sh \
      -d 3 \
      -f "$anat_ref" \
      -m "$moving_img" \
      -o "${temp_dir}rigid-" \
      -t r \
      -n 8 \
      -p f
    
    # Apply rigid transform to check alignment
    antsApplyTransforms \
      -d 3 \
      -i "$moving_img" \
      -r "$anat_ref" \
      -t "${temp_dir}rigid-0GenericAffine.mat" \
      -o "${temp_dir}func-rigid-aligned.nii.gz"
    
    # Full affine registration (12 DOF) for final alignment
    echo "Step 2: Affine registration (12 DOF)..."
    if [[ $USE_SKULL_STRIPPED -eq 1 ]]; then
        # Use masks for skull-stripped registration
        antsRegistration \
          --verbose $VERBOSE \
          --dimensionality 3 \
          --float 0 \
          --collapse-output-transforms 1 \
          --output [${temp_dir}func2anat-,${temp_dir}func2anat-Warped.nii.gz] \
          --interpolation Linear \
          --winsorize-image-intensities [0.005,0.995] \
          --use-histogram-matching 1 \
          --initial-moving-transform "${temp_dir}rigid-0GenericAffine.mat" \
          --masks [$anat_mask,$func_mask] \
          --transform Rigid[0.1] \
          --metric MI[$anat_ref,$moving_img,1,32,Regular,0.25] \
          --convergence [1000x500x250,1e-6,10] \
          --shrink-factors 4x2x1 \
          --smoothing-sigmas 2x1x0vox \
          --transform Affine[0.1] \
          --metric MI[$anat_ref,$moving_img,1,32,Regular,0.25] \
          --metric CC[$anat_ref,$moving_img,0.5,4] \
          --convergence [1000x500x250,1e-6,10] \
          --shrink-factors 4x2x1 \
          --smoothing-sigmas 2x1x0vox
    else
        # No masks for full-head registration
        antsRegistration \
          --verbose $VERBOSE \
          --dimensionality 3 \
          --float 0 \
          --collapse-output-transforms 1 \
          --output [${temp_dir}func2anat-,${temp_dir}func2anat-Warped.nii.gz] \
          --interpolation Linear \
          --winsorize-image-intensities [0.005,0.995] \
          --use-histogram-matching 1 \
          --initial-moving-transform "${temp_dir}rigid-0GenericAffine.mat" \
          --transform Rigid[0.1] \
          --metric MI[$anat_ref,$moving_img,1,32,Regular,0.25] \
          --convergence [1000x500x250,1e-6,10] \
          --shrink-factors 4x2x1 \
          --smoothing-sigmas 2x1x0vox \
          --transform Affine[0.1] \
          --metric MI[$anat_ref,$moving_img,1,32,Regular,0.25] \
          --metric CC[$anat_ref,$moving_img,0.5,4] \
          --convergence [1000x500x250,1e-6,10] \
          --shrink-factors 4x2x1 \
          --smoothing-sigmas 2x1x0vox
    fi
    
    # Copy final transform to output directory
    cp "${temp_dir}func2anat-0GenericAffine.mat" "${output_dir}func2anat-0GenericAffine.mat"
    
    # Apply transform to full functional mean image
    echo "Applying transform to functional mean image..."
    antsApplyTransforms \
      -d 3 \
      -i "$func_img" \
      -r "$anat_ref" \
      -t "${output_dir}func2anat-0GenericAffine.mat" \
      -o "${output_dir}moco_mean_in_T1.nii.gz"
    
    # Transform the functional mask to anatomical space
    echo "Transforming functional mask to anatomical space..."
    antsApplyTransforms \
      -d 3 \
      -i "$func_mask" \
      -r "$anat_ref" \
      -t "${output_dir}func2anat-0GenericAffine.mat" \
      -o "${output_dir}func_mask_in_T1.nii.gz" \
      -n NearestNeighbor
    
    # Create QC outputs
    if [[ $CREATE_QC -eq 1 ]]; then
        echo "Creating QC outputs..."
        
        # Overlay visualization
        CreateTiledMosaic \
          -i "${output_dir}moco_mean_in_T1.nii.gz" \
          -r "$anat_ref" \
          -o "${output_dir}QC-func2anat-overlay.png" \
          -t -1x-1 \
          -d 2 \
          -p mask \
          -s [5,mask,mask] \
          -a 0.7
    
        # Edge overlay for alignment checking
        ImageMath 3 "${temp_dir}func_edges.nii.gz" Laplacian "${output_dir}moco_mean_in_T1.nii.gz" 1.5 1
        ImageMath 3 "${temp_dir}anat_edges.nii.gz" Laplacian "$anat_ref" 1.5 1
        
        CreateTiledMosaic \
          -i "${temp_dir}func_edges.nii.gz" \
          -r "${temp_dir}anat_edges.nii.gz" \
          -o "${output_dir}QC-func2anat-edges.png" \
          -t -1x-1 \
          -d 2 \
          -p mask \
          -s [5,mask,mask] \
          -a 0.5
    fi
    
    # Calculate overlap metrics if masks exist
    if [[ -f "$anat_mask" && -f "${output_dir}func_mask_in_T1.nii.gz" ]]; then
        echo "Computing overlap metrics..."
        LabelOverlapMeasures 3 "$anat_mask" "${output_dir}func_mask_in_T1.nii.gz" "${output_dir}mask_overlap_metrics.csv"
    fi
    
    # Create summary
    cat > "${output_dir}registration_summary.txt" << EOF
Functional to Anatomical Registration Summary
==========================================
Monkey: $monkey
Date: $(date)
Functional input: $func_img
Anatomical reference: $anat_ref
Registration type: Rigid + Affine (linear only)
Skull-stripped registration: $USE_SKULL_STRIPPED

Outputs:
- func2anat-0GenericAffine.mat (transform matrix)
- moco_mean_in_T1.nii.gz (functional mean in anatomical space)
- func_mask_in_T1.nii.gz (functional mask in anatomical space)
- QC-func2anat-overlay.png (quality control overlay)
- QC-func2anat-edges.png (edge-based alignment check)

Notes:
- Linear registration only (no nonlinear deformation)
- Transform can be used for func → T1 → template pipeline
- Check QC images to verify alignment quality
- T1-in-func-mask was automatically created from T1-mask if missing
EOF
    
    # Cleanup temp files (optional)
    # rm -rf "$temp_dir"
    
    echo ""
    echo "=== Registration completed successfully for $monkey! ==="
    echo "Transform: ${output_dir}func2anat-0GenericAffine.mat"
    echo "Aligned functional: ${output_dir}moco_mean_in_T1.nii.gz"
    echo "QC images: ${output_dir}QC-func2anat-*.png"
    echo ""
    echo "Check QC images to verify alignment quality before proceeding."
    
done

echo "All monkeys processed!"