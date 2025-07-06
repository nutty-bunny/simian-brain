#!/bin/bash

set -e

base_dir="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/patchoulli/"
func_4d="${base_dir}func/topupped_mc.nii.gz"
ref_nmt="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.0_sym/NMT_v2.0_sym_05mm/NMT_v2.0_sym_05mm.nii.gz"

func2t1_mat="${base_dir}func/func2T1_rigid_ants_refined0GenericAffine.mat"
t1_to_nmt_affine="${base_dir}aligned_0GenericAffine.mat"
t1_to_nmt_warp="${base_dir}registered_1Warp.nii.gz"

tmpdir="${base_dir}func/tmp_applytransforms"
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
      -t "$t1_to_nmt_affine" \
      -t "$func2t1_mat"
done

find "$tmpdir" -name 'nmt_vol_*.nii.gz' | sort > "${tmpdir}/filelist.txt"

if [[ ! -s "${tmpdir}/filelist.txt" ]]; then
  echo "No transformed volumes found."
  exit 1
fi

fslmerge -t "${base_dir}func/func_in_NMT.nii.gz" $(cat "${tmpdir}/filelist.txt")

rm -rf "$tmpdir"

echo "Done"
echo "fsleyes $ref_nmt ${base_dir}func/func_in_NMT.nii.gz &"
