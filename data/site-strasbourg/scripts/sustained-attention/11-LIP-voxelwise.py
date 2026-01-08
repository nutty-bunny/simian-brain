# figure out how to actually display this. so far i am just processing all the subjects
import os
import numpy as np
import nibabel as nib
from scipy.stats import pearsonr
from tqdm import tqdm
import subprocess
import pandas as pd
from nilearn import image  # Needed for resampling

# Configuration
BASE_DIR = "/Users/similovesyou/Desktop/qts/simian-brain"
DATASET = "site-strasbourg"
DERIVATIVES_DIR = os.path.join(BASE_DIR, f"data/{DATASET}/final-derivatives")
MASK_DIR = os.path.join(BASE_DIR, "masks")
TARGET_SEED = "LIP"
NMT_BRAINMASK = os.path.join(BASE_DIR, "NMT_v2.1_sym/NMT_v2.1_sym_05mm/NMT_v2.1_sym_05mm_brainmask.nii.gz")
OUTPUT_DIR = os.path.join(BASE_DIR, f"functional-connectivity/{DATASET}/voxelwise-{TARGET_SEED}-connectivity")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_subjects_and_sessions():
    subjects = []
    for subject in os.listdir(DERIVATIVES_DIR):
        func_path = os.path.join(
            DERIVATIVES_DIR,
            subject,
            "func-a-licious",
            "func-clean-final.nii.gz"
        )
        if os.path.exists(func_path):
            subjects.append((subject, func_path))
        else:
            print(f"!! Skipping {subject}: func-clean-final.nii.gz not found")
    return subjects

def extract_seed_time_series(seed_mask_path, func_file, brain_mask_path=None):
    if not os.path.exists(seed_mask_path):
        print(f"!! ERROR: Seed mask not found at {seed_mask_path}")
        return None
    
    temp_mask = None
    if brain_mask_path and os.path.exists(brain_mask_path):
        temp_mask = os.path.join(OUTPUT_DIR, "temp_combined_mask.nii.gz")
        cmd = f"fslmaths {seed_mask_path} -mul {brain_mask_path} {temp_mask}"
        subprocess.run(cmd, shell=True, check=True)
        mask_to_use = temp_mask
    else:
        mask_to_use = seed_mask_path
    
    ts_file = os.path.join(OUTPUT_DIR, "temp_seed_ts.txt")
    cmd = f"fslmeants -i {func_file} -m {mask_to_use} --usemm -o {ts_file}"
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        ts = np.loadtxt(ts_file)
        os.remove(ts_file)
        if temp_mask and os.path.exists(temp_mask):
            os.remove(temp_mask)
        return ts
    except Exception as e:
        print(f"!! ERROR extracting time series: {e}")
        if os.path.exists(ts_file):
            os.remove(ts_file)
        if temp_mask and os.path.exists(temp_mask):
            os.remove(temp_mask)
        return None

def compute_voxelwise_fc(subject, func_file, seed_ts, brain_mask_path=None):
    try:
        img = nib.load(func_file)
        data = img.get_fdata()

        if brain_mask_path and os.path.exists(brain_mask_path):
            mask_img = nib.load(brain_mask_path)
            if mask_img.shape != data.shape[:3]:
                print("++ Resampling brain mask to match functional data")
                mask_img = image.resample_to_img(mask_img, img, interpolation='nearest')
            brain_mask = mask_img.get_fdata() > 0
        else:
            brain_mask = (data.std(axis=-1) > 0)

        fc_map = np.zeros(data.shape[:3])

        print("++ Computing voxel-wise correlations...")
        mask_indices = np.argwhere(brain_mask)

        for i, (x, y, z) in enumerate(tqdm(mask_indices, desc="Voxels")):
            voxel_ts = data[x, y, z, :]
            if np.std(voxel_ts) > 0:
                r, _ = pearsonr(seed_ts, voxel_ts)
                fc_map[x, y, z] = r
            else:
                fc_map[x, y, z] = 0  # optional

        output_file = os.path.join(OUTPUT_DIR, f"{subject}_{TARGET_SEED}_voxelwise_fc.nii.gz")
        nib.save(nib.Nifti1Image(fc_map, img.affine), output_file)

        # Optional: print value range for QC
        print(f"Saved FC map range for {subject}: min={fc_map.min():.3f}, max={fc_map.max():.3f}")
        
        return output_file
    except Exception as e:
        print(f"!! ERROR in voxel-wise computation: {e}")
        return None

def run_analysis():
    subjects_files = get_subjects_and_sessions()
    if not subjects_files:
        print("!! ERROR: No valid subjects found.")
        return
    
    seed_mask_path = os.path.join(MASK_DIR, f"{TARGET_SEED}_seed.nii.gz")
    if not os.path.exists(seed_mask_path):
        print(f"!! ERROR: Seed mask not found at {seed_mask_path}")
        return
    
    if not os.path.exists(NMT_BRAINMASK):
        print(f"!! WARNING: NMT brain mask not found at {NMT_BRAINMASK}")
        print("++ Proceeding without brain mask (using intensity-based mask)")
        brain_mask_path = None
    else:
        brain_mask_path = NMT_BRAINMASK
    
    results = []
    for subject, func_file in subjects_files:
        print(f"\n++ Processing Subject: {subject}")
        
        seed_ts = extract_seed_time_series(seed_mask_path, func_file, brain_mask_path)
        if seed_ts is None:
            results.append({'subject': subject, 'status': 'failed (seed ts extraction)'})
            continue
        
        fc_file = compute_voxelwise_fc(subject, func_file, seed_ts, brain_mask_path)
        if fc_file:
            results.append({
                'subject': subject,
                'fc_map': fc_file,
                'status': 'completed'
            })
        else:
            results.append({
                'subject': subject,
                'fc_map': None,
                'status': 'failed (voxelwise computation)'
            })
    
    summary_file = os.path.join(OUTPUT_DIR, "analysis-summary.csv")
    pd.DataFrame(results).to_csv(summary_file, index=False)
    print(f"\n++ Analysis complete. Summary saved to {summary_file}")

if __name__ == "__main__":
    print("++ Voxel-wise Functional Connectivity Analysis ++")
    print(f"++ Target Seed: {TARGET_SEED}")
    print(f"++ Using NMT Brain Mask: {NMT_BRAINMASK if os.path.exists(NMT_BRAINMASK) else 'NOT FOUND'}")
    print(f"++ Output Directory: {OUTPUT_DIR}")
    
    run_analysis()
