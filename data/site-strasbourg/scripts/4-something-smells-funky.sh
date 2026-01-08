#!/bin/bash
set -e

# Configuration flags
USE_FUNC2T1_TRANSFORM=0  # Set to 0 to skip func2t1 transformation
CREATE_QC_OVERLAYS=0
VERBOSE=1

# Monkey list
monkeys=(
  "yin"
)

# consider patsy yannick w/o func2anat transform

# NMT reference paths
ref_nmt_orig="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.1_sym/NMT_v2.1_sym_05mm/NMT_v2.1_sym_05mm.nii.gz"
nmt_mask="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.1_sym/NMT_v2.1_sym_05mm/NMT_v2.1_sym_05mm_brainmask.nii.gz"

for monkey in "${monkeys[@]}"; do
  echo "Processing monkey: $monkey"
  
  # Updated paths to match your working script structure
  base_dir="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/final-derivatives/${monkey}/"
  func_4d="${base_dir}moco.nii.gz"  # Use full 4D functional data
  func_mask="${base_dir}T1-in-func-mask.nii.gz"
  
  # Transform paths matching your working script
  func2anat="${base_dir}func2anat/func2anat-0GenericAffine.mat"
  t1_to_nmt_initial="${base_dir}anat2nmt/T1-to-NMT-0GenericAffine.mat"
  t1_to_nmt_affine="${base_dir}anat2nmt/T1-to-NMT-1GenericAffine.mat"
  t1_to_nmt_warp="${base_dir}anat2nmt/T1-to-NMT-1Warp.nii.gz"
  
  # Use padded NMT reference (matching your working script)
  ref_nmt_padded="${base_dir}anat2nmt/temp-anat2nmt/NMT-padded.nii.gz"
  nmt_mask_padded="${base_dir}anat2nmt/temp-anat2nmt/NMT-mask-padded.nii.gz"
  
  # Output directory
  output_dir="${base_dir}func2nmt/"
  mkdir -p "$output_dir"
  temp_dir="${output_dir}temp/"
  mkdir -p "$temp_dir"
  
  # Strategy
  if [[ $USE_FUNC2T1_TRANSFORM -eq 1 ]]; then
      transform_strategy="func → T1 → NMT"
      transforms=("$t1_to_nmt_warp" "$t1_to_nmt_affine" "$t1_to_nmt_initial" "$func2anat")
      echo "Mode: Applying func2anat + anat2nmt transformations"
  else
      transform_strategy="T1 → NMT only (func/T1 aligned)"
      transforms=("$t1_to_nmt_warp" "$t1_to_nmt_affine" "$t1_to_nmt_initial")
      echo "Mode: Applying anat2nmt transformations only"
  fi
  
  echo "Strategy: $transform_strategy"
  
  # Check files
  for f in "${transforms[@]}"; do
      if [[ ! -f "$f" ]]; then
          echo "Missing transform for $monkey: $f"
          continue 2  # Skip to next monkey
      fi
  done
  
  if [[ ! -f "$func_4d" ]]; then
      echo "Missing functional image for $monkey: $func_4d"
      continue
  fi
  
  if [[ ! -f "$ref_nmt_padded" ]]; then
      echo "Missing padded NMT reference for $monkey: $ref_nmt_padded"
      continue
  fi
  
  echo "Splitting 4D functional image into volumes..."
  # Clear any existing split files
  rm -f "${temp_dir}"/vol_*.nii.gz "${temp_dir}"/nmt_vol_*.nii.gz
  
  fslsplit "$func_4d" "${temp_dir}/vol_" -t
  
  echo "Applying transform chain to each volume..."
  vol_count=0
  for vol in "${temp_dir}"/vol_*.nii.gz; do
      vol_base=$(basename "$vol")
      out_file_padded="${temp_dir}/nmt_padded_${vol_base}"
      out_file_final="${temp_dir}/nmt_${vol_base}"
      
      if [[ -f "$out_file_final" ]]; then
          ((vol_count++))
          continue
      fi
      
      # Transform to padded NMT space first
      antsApplyTransforms \
        -d 3 \
        -i "$vol" \
        -r "$ref_nmt_padded" \
        -o "$out_file_padded" \
        -n Linear \
        $(printf -- '-t %s ' "${transforms[@]}")
      
      # Then resample to standard NMT space
      antsApplyTransforms \
        -d 3 \
        -i "$out_file_padded" \
        -r "$ref_nmt_orig" \
        -o "$out_file_final" \
        -n Linear
      
      ((vol_count++))
      if (( vol_count % 10 == 0 )); then
          echo "  Processed $vol_count volumes..."
      fi
  done
  
  echo "Merging transformed volumes into final 4D NMT-aligned image..."
  find "$temp_dir" -name 'nmt_vol_*.nii.gz' | sort -V > "${temp_dir}/filelist.txt"
  
  if [[ ! -s "${temp_dir}/filelist.txt" ]]; then
      echo "No transformed volumes found for $monkey."
      continue
  fi
  
  # Create unmasked 4D output
  output_func_nmt="${output_dir}func-4D-in-NMT.nii.gz"
  fslmerge -t "$output_func_nmt" $(cat "${temp_dir}/filelist.txt")

  # Transform functional mask to NMT space
  if [[ -f "$func_mask" ]]; then
      echo "Transforming functional mask to NMT space..."
      antsApplyTransforms \
        -d 3 \
        -i "$func_mask" \
        -r "$ref_nmt_padded" \
        -o "${temp_dir}func-mask-in-NMT-padded.nii.gz" \
        -n NearestNeighbor \
        $(printf -- '-t %s ' "${transforms[@]}")
      
      antsApplyTransforms \
        -d 3 \
        -i "${temp_dir}func-mask-in-NMT-padded.nii.gz" \
        -r "$ref_nmt_orig" \
        -o "${output_dir}func-mask-in-NMT.nii.gz" \
        -n NearestNeighbor
  fi
  
  # QC overlay for mean functional image
  if [[ $CREATE_QC_OVERLAYS -eq 1 ]]; then
      echo "Creating QC overlays..."
      # Create mean image for QC
      fslmaths "$output_func_nmt" -Tmean "${temp_dir}func-mean-in-NMT.nii.gz"
      
      CreateTiledMosaic \
        -i "${temp_dir}func-mean-in-NMT.nii.gz" \
        -r "$ref_nmt_orig" \
        -o "${output_dir}QC-func-4D-in-NMT-overlay.png" \
        -t -1x-1 -d 2 -p mask \
        -s [5,mask,mask] \
        -x "$nmt_mask" \
        -a 0.7
  fi
  
  # Summary
  cat > "${output_dir}transform-summary.txt" << EOF
4D Functional to NMT Transform Summary
Date: $(date)
Subject: $monkey
Functional input: $func_4d
NMT reference: $ref_nmt_orig
Transform strategy: $transform_strategy
Transforms used:
$(printf '  %s\n' "${transforms[@]}")
Outputs:
  func-4D-in-NMT.nii.gz (unmasked 4D functional in NMT space)
  func-4D-in-NMT-masked.nii.gz (masked 4D functional)
  func-mask-in-NMT.nii.gz (functional mask in NMT space)
Volumes processed: $vol_count
EOF
  
  # Cleanup intermediate files (comment out for debugging)
  rm -f "${temp_dir}"/vol_*.nii.gz "${temp_dir}"/nmt_vol_*.nii.gz "${temp_dir}"/nmt_padded_*.nii.gz
  
  echo "✓ Done with $monkey"
  echo "Outputs saved to: ${output_dir}"
  echo "You can inspect the result using:"
  echo "fsleyes $ref_nmt_orig $output_func_nmt &"
  echo ""
done

echo "✓ All monkeys processed."