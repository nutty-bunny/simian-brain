import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_1samp
import subprocess

plt.rcParams['svg.fonttype'] = 'none'

# === CONFIGURATION ===
BASE_DIR = "/Users/similovesyou/Desktop/qts/simian-brain"
DATASET = "site-strasbourg"
FC_DIR = f"{BASE_DIR}/functional-connectivity/{DATASET}"
MASK_DIR = f"{BASE_DIR}/masks"
SEED_NAMES = ["V1", "V2", "V3", "V4", "V4t", "MT", "MST", "FST", "LIP", "FEF"]
OUTPUT_DIR = os.path.join(FC_DIR, "seed-2-seed-connectivity")
CLEAN_DIR = os.path.join(BASE_DIR, "data", DATASET, "derivatives")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BOLD_CACHE = {}

def find_bold_file(subject):
    if subject in BOLD_CACHE:
        return BOLD_CACHE[subject]
    
    bold_file = os.path.join(CLEAN_DIR, subject, "func", "func-a-licious", "func-clean.nii.gz")
    if os.path.exists(bold_file):
        BOLD_CACHE[subject] = bold_file
        return bold_file
    else:
        print(f"++ WARNING: No BOLD file for subject {subject}")
        return None

def extract_seed_ts(subject, seed_name):
    seed_mask = os.path.join(MASK_DIR, f"{seed_name}_seed.nii.gz")
    seed_dir = os.path.join(FC_DIR, f"seed-{seed_name}")
    os.makedirs(seed_dir, exist_ok=True)
    ts_file = os.path.join(seed_dir, f"{subject}_{seed_name}_seed_ts.txt")
    bold_file = find_bold_file(subject)

    if not os.path.exists(ts_file):
        if bold_file and os.path.exists(seed_mask):
            cmd = f"/usr/local/fsl/bin/fslmeants -i {bold_file} -m {seed_mask} --usemm -o {ts_file}"
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"++ ERROR extracting {seed_name} for {subject}: {e}")
                return None
        else:
            print(f"++ Missing input for subject {subject}, seed {seed_name}")
            return None
    return np.loadtxt(ts_file) if os.path.exists(ts_file) else None

def compute_connectivity_matrix(subject):
    seed_ts = {}
    available = []
    lengths = set()

    for seed in SEED_NAMES:
        ts = extract_seed_ts(subject, seed)
        if ts is not None:
            seed_ts[seed] = ts
            available.append(seed)
            lengths.add(len(ts))

    if len(lengths) != 1:
        print(f"++ WARNING: Inconsistent time series lengths for {subject}")
        return None
    if len(available) < 2:
        print(f"++ WARNING: Not enough valid seeds for {subject}")
        return None

    mat = pd.DataFrame(index=available, columns=available, dtype=float)
    for i in available:
        for j in available:
            r, _ = pearsonr(seed_ts[i], seed_ts[j])
            mat.loc[i, j] = r
    return mat

def plot_connectivity_matrix(matrix, subject):
    path = os.path.join(OUTPUT_DIR, f"{subject}_seed_connectivity.svg")
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2, style="white")

    annot = matrix.copy().astype(str)
    for i in matrix.index:
        for j in matrix.columns:
            val = matrix.loc[i, j]
            annot.loc[i, j] = f"{val:.2f}" if abs(val) >= 0.2 else ""

    ax = sns.heatmap(
        matrix, annot=annot, cmap="PuBu", vmin=-1, vmax=1, square=True,
        linewidths=0.5, cbar_kws={"pad": 0.05, "label": "Pearson's r"}, fmt=""
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel("Correlation (r)", rotation=270, labelpad=20, fontsize=12, weight="bold")
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.title("Simian Signals: Seed-to-Seed Functional Connectivity", fontsize=14, weight='bold', pad=20)
    for b in [5, 8]: ax.axhline(b, color='lightgray'); ax.axvline(b, color='lightgray')
    plt.tight_layout()
    plt.savefig(path, format="svg", dpi=300)
    plt.savefig(path.replace(".svg", ".png"), format="png", dpi=300)

    print(f"++ Saved: {path}")

def process_all_subjects():
    subject_dirs = sorted([d for d in os.listdir(CLEAN_DIR) if os.path.isdir(os.path.join(CLEAN_DIR, d))])
    for subject in subject_dirs:
        csv = os.path.join(OUTPUT_DIR, f"{subject}_seed_connectivity.csv")
        svg = os.path.join(OUTPUT_DIR, f"{subject}_seed_connectivity.svg")
        if os.path.exists(csv) and os.path.exists(svg):
            print(f"++ SKIPPING {subject}")
            continue
        print(f"++ Processing: {subject}")
        mat = compute_connectivity_matrix(subject)
        if mat is not None:
            mat.to_csv(csv)
            print(f"++ Saved matrix: {csv}")
            plot_connectivity_matrix(mat, subject)

def generate_group_matrix():
    print("++ Generating group-level matrix...")
    csvs = glob.glob(os.path.join(OUTPUT_DIR, "*_seed_connectivity.csv"))
    matrices = [pd.read_csv(f, index_col=0) for f in csvs]

    if not matrices:
        print("++ No subject matrices found.")
        return

    ref = matrices[0].index
    data = np.stack([m.loc[ref, ref].values.astype(float) for m in matrices])
    mean = np.nanmean(data, axis=0)
    tvals, pvals = ttest_1samp(data, popmean=0, axis=0, nan_policy="omit")
    sig_mask = pvals < 0.05

    mean_df = pd.DataFrame(mean, index=ref, columns=ref)
    mean_df.to_csv(os.path.join(OUTPUT_DIR, "group_mean_connectivity_matrix.csv"))
    print(f"++ Group matrix saved: group_mean_connectivity_matrix.csv")
    print("\n++ Group connectivity matrix (mean r values):")
    print(mean_df.round(2))
    plot_group_matrix(mean_df, sig_mask)

def plot_group_matrix(matrix, significance_mask):
    path = os.path.join(OUTPUT_DIR, "group_mean_connectivity_matrix.svg")
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2, style="white")

    annot = matrix.copy().astype(str)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if significance_mask[i, j]:
                annot.iloc[i, j] = f"{matrix.iloc[i, j]:.2f}"
            else:
                annot.iloc[i, j] = ""

    ax = sns.heatmap(
        matrix, annot=annot, cmap="PuBu", vmin=-1, vmax=1, square=True,
        linewidths=0.5, cbar_kws={"pad": 0.05, "label": "Mean Pearson's r"}, fmt=""
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel("Group Correlation (r)", rotation=270, labelpad=20, fontsize=12, weight="bold")
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.title("Group Mean Seed-to-Seed Connectivity", fontsize=14, weight='bold', pad=20)
    for b in [5, 8]: ax.axhline(b, color='lightgray'); ax.axvline(b, color='lightgray')
    plt.tight_layout()
    plt.savefig(path, format="svg", dpi=300)
    plt.savefig(path.replace(".svg", ".png"), format="png", dpi=300)

    print(f"++ Group matrix figure saved: {path}")

if __name__ == "__main__":
    print("++ Starting connectivity pipeline...")
    process_all_subjects()
    generate_group_matrix()
    print("++ All done!")