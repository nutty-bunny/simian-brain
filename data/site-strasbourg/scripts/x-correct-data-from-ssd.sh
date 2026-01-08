#!/bin/bash
set -e

# Source (your external drive)
SRC_BASE="/Volumes/simians/simian-brain/data/site-strasbourg/final-derivatives"

# Destination (new cleaned structure)
DST_BASE="/Volumes/simians/simian-brain/data/site-strasbourg/final-derivatives-3mm"

mkdir -p "$DST_BASE"

# Loop through all monkeys
for monkey in "$SRC_BASE"/*; do
    name=$(basename "$monkey")

    # Skip if monkey already has func-clean-final.nii.gz in destination
    if [[ -f "$DST_BASE/$name/func-a-licious/func-clean-final.nii.gz" ]]; then
        echo "Skipping $name: func-clean-final.nii.gz already exists in destination"
        continue
    fi

    if [[ -d "$monkey/func2nmt" ]]; then
        echo "Processing $name..."

        # Create destination folders
        mkdir -p "$DST_BASE/$name/func2nmt"

        # Copy required functional file
        if [[ -f "$monkey/func2nmt/func-4D-in-NMT.nii.gz" ]]; then
            cp "$monkey/func2nmt/func-4D-in-NMT.nii.gz" "$DST_BASE/$name/func2nmt/"
        else
            echo "WARNING: Missing func-4D-in-NMT.nii.gz for $name"
        fi

        # Copy and rename mask if exists
        if [[ -f "$monkey/func2nmt/single-func-mask-in-NMT.nii.gz" ]]; then
            cp "$monkey/func2nmt/single-func-mask-in-NMT.nii.gz" \
               "$DST_BASE/$name/func2nmt/func-mask-in-NMT.nii.gz"
        fi

        # Copy motion file if exists
        if [[ -f "$monkey/moco_motion.1D" ]]; then
            cp "$monkey/moco_motion.1D" "$DST_BASE/$name/"
        else
            echo "NOTE: No motion file found for $name"
        fi
    fi
done

echo "Done! Organized dataset is in: $DST_BASE"
