#!/bin/bash

infile="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/patchoulli/fMRI_Files/topupped.nii.gz"
trimmed="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/patchoulli/func/topupped_trimmed.nii.gz"
mc_out="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/patchoulli/func/topupped_mc.nii.gz"
mc_par="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/patchoulli/func/topupped_mc.par"
mean_img="/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/patchoulli/func/topupped_mc_mean.nii.gz"

fslroi "$infile" "$trimmed" 5 -1
mcflirt -in "$trimmed" -out "$mc_out" -mats -plots -report -reffile "$trimmed" -save_plots
cp "${mc_out%.nii.gz}.par" "$mc_par"
fslmaths "$mc_out" -Tmean "$mean_img"
