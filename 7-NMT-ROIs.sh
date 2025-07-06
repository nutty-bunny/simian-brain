#!/bin/bash

atlas="/Users/similovesyou/Desktop/qts/simian-brain/NMT_v2.0_sym/NMT_v2.0_sym_05mm/D99_atlas_in_NMT_v2.0_sym_05mm.nii.gz"
outdir="/Users/similovesyou/Desktop/qts/simian-brain/NMT-ROIs"
mkdir -p "$outdir"

fslmaths "$atlas" -thr 31 -uthr 31 -bin "$outdir/LIPv.nii.gz"   # LIPv

fslmaths "$atlas" -thr 130 -uthr 130 -bin "$outdir/LIPd.nii.gz" # LIPd

fslmaths "$outdir/LIPv.nii.gz" -add "$outdir/LIPd.nii.gz" -bin "$outdir/LIP_joint.nii.gz"   # Joint LIP

fslmaths "$atlas" -thr 125 -uthr 125 -bin "$outdir/TEO.nii.gz" # TEO

fslmaths "$atlas" -thr 148 -uthr 148 -bin "$outdir/FEF.nii.gz" # Core FEF (8Ad)
# Double check if FEF is correct -- maybe it's too frontal 


