#!/bin/bash

set -e

# Monkey list
monkeys=(
  "iron"
)

# Reference NMT template
ref_nmt="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.1_sym/NMT_v2.1_sym_05mm/NMT_v2.1_sym_05mm.nii.gz"

# Loop over each monkey
for monkey in "${monkeys[@]}"; do
  echo "Processing monkey: $monkey"

  base_dir="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/${monkey}/"
  func_4d="${base_dir}func/topupped-mc.nii.gz"
  func2t1_mat="${base_dir}func/func2T1-rigid0GenericAffine.mat"
  t1_to_nmt_affine="${base_dir}aligned-0GenericAffine.mat"
  t1_to_nmt_warp="${base_dir}registered-1Warp.nii.gz"

  tmpdir="${base_dir}func/tmp-applytransforms"
  mkdir -p "$tmpdir"

  if ! compgen -G "${tmpdir}/vol_*.nii.gz" > /dev/null; then
    fslsplit "$func_4d" "${tmpdir}/vol_" -t
  fi

  for vol in "${tmpdir}"/vol_*.nii.gz; do
    vol_base=$(basename "$vol")
    out_file="${tmpdir}/nmt_${vol_base}"

    if [[ -f "$out_file" ]]; then
        continue
    fi

    antsApplyTransforms \
      -d 3 \
      -i "$vol" \
      -r "$ref_nmt" \
      -o "$out_file" \
      -n Linear \
      -t "$t1_to_nmt_warp" \
      -t "$t1_to_nmt_affine"
  done

  find "$tmpdir" -name 'nmt_vol_*.nii.gz' | sort > "${tmpdir}/filelist.txt"

  if [[ ! -s "${tmpdir}/filelist.txt" ]]; then
    echo "No transformed volumes found for $monkey."
    continue
  fi

  fslmerge -t "${base_dir}func/func-in-NMT.nii.gz" $(cat "${tmpdir}/filelist.txt")

  rm -rf "$tmpdir"

  echo "Done with $monkey"
  echo "fsleyes $ref_nmt ${base_dir}func/func-in-NMT.nii.gz &"
done

echo "All monkeys processed."