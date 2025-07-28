import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import subprocess

# ++ CONFIGURATION ++
BASE_DIR = "/Users/similovesyou/Desktop/qts/simian-brain"
DATASET = "site-strasbourg"
FC_DIR = f"{BASE_DIR}/functional-connectivity/{DATASET}"
MASK_DIR = f"{BASE_DIR}/masks"
SEED_NAMES = ["V1", "V2", "V3", "V4", "V4t", "MT", "MST", "FST", "LIP"]
TARGET_SEED = "FEF"
OUTPUT_DIR = os.path.join(FC_DIR, f"seed-to-{TARGET_SEED}-connectivity")
DERIVATIVES_DIR = os.path.join(BASE_DIR, f"data/{DATASET}/derivatives")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ++ Function to find subject + func-clean.nii.gz ++
def get_subjects_and_sessions():
    subjects = []
    for subject in os.listdir(DERIVATIVES_DIR):
        func_path = os.path.join(
            DERIVATIVES_DIR,
            subject,
            "func",
            "func-a-licious",
            "func-clean.nii.gz"
        )
        if os.path.exists(func_path):
            subjects.append((subject, func_path))
        else:
            print(f"!! Skipping {subject}: func-clean.nii.gz not found")
    return subjects

# ++ Function to extract time series ++
def extract_seed_ts(seed_name, target_seed, func_file, subject):
    ts_file = os.path.join(
        OUTPUT_DIR,
        f"{subject}_{seed_name}_{target_seed}_ts.txt"
    )

    seed_mask = os.path.join(MASK_DIR, f"{seed_name}_seed.nii.gz")
    target_mask = os.path.join(MASK_DIR, f"{target_seed}_seed.nii.gz")

    print(f"++ {seed_name} → {target_seed}", end=' ')

    for f in [func_file, seed_mask, target_mask]:
        if not os.path.exists(f):
            print(f"!! ERROR: Missing file: {f}")
            return None

    env = os.environ.copy()
    env["FSLDIR"] = "/usr/local/fsl"
    env["PATH"] = f"{env['FSLDIR']}/bin:" + env["PATH"]

    if not os.path.exists(ts_file):
        cmd = f"fslmeants -i {func_file} -m {seed_mask} --usemm -o {ts_file}"
        try:
            subprocess.run(cmd, shell=True, env=env, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"!! fslmeants failed: {e.stderr.strip()}")
            return None
        except Exception as e:
            print(f"!! Unexpected error: {e}")
            return None

    if not os.path.exists(ts_file):
        print("!! ERROR: Output time series file missing.")
        return None

    ts = np.loadtxt(ts_file)
    if len(ts) < 2:
        print(f"!! ERROR: Too few time points ({len(ts)})")
        return None

    print(f"[{len(ts)} timepoints extracted]")
    return ts

# ++ Function to compute correlations ++
def compute_correlations(subject, func_file):
    output_file = os.path.join(OUTPUT_DIR, f"{subject}_correlation_results.csv")
    if os.path.exists(output_file):
        print(f"++ Skipping {subject} — output already exists.")
        return None

    print(f"++ Computing correlations for {subject}")

    # Step 1: Extract reference seed (e.g., FEF)
    ref_ts = extract_seed_ts(TARGET_SEED, TARGET_SEED, func_file, subject)
    if ref_ts is None:
        print("!! CRITICAL: Failed to extract reference time series")
        return None

    # Step 2: Compute correlations
    results = {}
    for seed in SEED_NAMES:
        seed_ts = extract_seed_ts(seed, TARGET_SEED, func_file, subject)
        if seed_ts is None:
            results[seed] = None
            continue

        try:
            min_len = min(len(ref_ts), len(seed_ts))
            r, _ = pearsonr(ref_ts[:min_len], seed_ts[:min_len])
            results[seed] = r
            print(f"++ {seed}-{TARGET_SEED} correlation: {r:.3f}")
        except Exception as e:
            print(f"!! Correlation failed for {seed}: {e}")
            results[seed] = None

    pd.DataFrame.from_dict(results, orient='index', columns=['Correlation']).to_csv(output_file)
    print(f"++ Saved results to {output_file}")
    return results

# ++ MAIN FUNCTION ++
if __name__ == "__main__":
    print("++ Seed-to-Seed Connectivity Analysis ++")

    subjects_files = get_subjects_and_sessions()
    if not subjects_files:
        print("!! ERROR: No valid subjects found.")
        exit(1)

    for subject, func_file in subjects_files:
        print(f"\n++ Processing Subject: {subject}")
        results = compute_correlations(subject, func_file)

        if results:
            print("++ Final Correlations:")
            for seed, r in results.items():
                print(f"++ {seed}: {r:.3f}" if r is not None else f"++ {seed}: Failed")

    print("\n++ Analysis Complete ++")
