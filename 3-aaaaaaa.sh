#!/bin/bash

infile="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/yin/fMRI/topupped.nii.gz"
trimmed="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/yin/func/topupped-trimmed.nii.gz"
mc_out="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/yin/func/topupped-mc.nii.gz"
mc_par="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/yin/func/topupped-mc.par"
mean_img="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/yin/func/topupped-mc-mean.nii.gz"

if [ ! -f "$infile" ]; then
  echo "ERROR: Input file not found: $infile"
  exit 1
fi

fslroi "$infile" "$trimmed" 5 -1

mcflirt -in "$trimmed" -out "${mc_out%.nii.gz}" -mats -plots -report

if [ -f "${mc_out%.nii.gz}.nii.gz" ]; then
  mv "${mc_out%.nii.gz}.nii.gz" "$mc_out"
fi

if [ -f "${mc_out%.nii.gz}.par" ]; then
  cp "${mc_out%.nii.gz}.par" "$mc_par"
else
  echo "WARNING: Motion parameters (.par) file not found!"
fi

# Create mean image
fslmaths "$mc_out" -Tmean "$mean_img"

echo "Preprocessing complete:"
echo "Trimmed: $trimmed"
echo "Motion-corrected: $mc_out"
echo "Mean image: $mean_img"