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
BASE_DIR = "/Volumes/simians/qts/simian-brain"
DATASET = "site-strasbourg"
FC_DIR = f"/Users/similovesyou/Desktop/qts/simian-brain/final-functional-connectivity-2mm/{DATASET}"
MASK_DIR = "/Users/similovesyou/Desktop/qts/simian-brain/masks"
SEED_NAMES = ["V1", "V2", "V3", "V4", "V4t", "MT", "MST", "FST", "LIP", "FEF"]
OUTPUT_DIR = os.path.join(FC_DIR, "attention-seed-2-seed-connectivity")
CLEAN_DIR = os.path.join(BASE_DIR, "data", DATASET, "final-derivatives-2mm")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BOLD_CACHE = {}

def find_bold_file(subject):
    if subject in BOLD_CACHE:
        return BOLD_CACHE[subject]
    
    bold_file = os.path.join(CLEAN_DIR, subject, "func-a-licious", "func-clean-final.nii.gz")
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

from statsmodels.stats.multitest import multipletests

def generate_group_matrix():
    print("++ Generating group-level matrix (FDR corrected)...")
    csvs = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_seed_connectivity.csv")))
    matrices = [pd.read_csv(f, index_col=0) for f in csvs]

    if not matrices:
        print("++ No subject matrices found.")
        return

    ref = matrices[0].index
    k = len(ref)
    nsub = len(matrices)
    print(f"++ Found {nsub} subject matrices, {k} ROIs -> {k*(k-1)//2} unique tests.")

    # stack subject matrices (subjects, k, k)
    data_r = np.stack([m.loc[ref, ref].values.astype(float) for m in matrices], axis=0)
    data_r = np.clip(data_r, -0.999999, 0.999999)  # avoid inf in atanh

    # Fisher z-transform
    data_z = np.arctanh(data_r)

    # run tests only on upper triangle
    iu, ju = np.triu_indices(k, k=1)
    pvals = []
    for i, j in zip(iu, ju):
        _, p = ttest_1samp(data_z[:, i, j], popmean=0, nan_policy="omit")
        pvals.append(p)
    pvals = np.array(pvals)

    # FDR correction
    reject, pvals_corr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    sig_mask = np.zeros((k, k), dtype=bool)
    sig_mask[iu, ju] = reject
    sig_mask[ju, iu] = reject

    print(f"++ FDR significant unique connections: {reject.sum()} / {len(pvals)}")

    # compute group mean (z → r)
    mean_z = np.nanmean(data_z, axis=0)
    mean_r = np.tanh(mean_z)
    mean_df = pd.DataFrame(mean_r, index=ref, columns=ref)
    mean_df.to_csv(os.path.join(OUTPUT_DIR, "group_mean_connectivity_matrix.csv"))
    print("++ Group mean (back-transformed r) saved: group_mean_connectivity_matrix.csv")
    print("\n++ Group connectivity matrix (mean r values):")
    print(mean_df.round(2))

    # save corrected p-values
    pvals_full = np.full((k, k), np.nan)
    pvals_full[iu, ju] = pvals_corr
    pvals_full[ju, iu] = pvals_corr
    pd.DataFrame(pvals_full, index=ref, columns=ref).to_csv(
        os.path.join(OUTPUT_DIR, "group_pvals_fdr_corrected.csv")
    )

    # plot (only FDR significant connections)
    plot_group_matrix(mean_df, sig_mask)

    return {
        "mean_r": mean_df,
        "mean_z": pd.DataFrame(mean_z, index=ref, columns=ref),
        "sig_mask_fdr": sig_mask,
        "pvals_fdr": pd.DataFrame(pvals_full, index=ref, columns=ref),
        "n_subjects": nsub,
        "n_rois": k,
    }

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

# the below is ugly but functional
def plot_group_matrix(matrix, significance_mask):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import numpy as np

    colormaps = [
        'Blues', 'Greens', 'BuPu'
    ]

    # Calculate min/max values ignoring diagonal for vmax
    vmin = matrix.values[np.triu_indices_from(matrix, k=1)].min()
    vmax = matrix.values[np.triu_indices_from(matrix, k=1)].max()

    annot = matrix.copy().astype(str)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if significance_mask[i, j]:
                annot.iloc[i, j] = f"{matrix.iloc[i, j]:.2f}"
            else:
                annot.iloc[i, j] = ""

    for cmap in colormaps:
        plt.figure(figsize=(10, 8))

        # Set Times New Roman font
        plt.rcParams['font.family'] = 'Times New Roman'
        sns.set(font_scale=1.2, style="white")

        ax = sns.heatmap(
            matrix, annot=annot, cmap=cmap, vmin=vmin, vmax=vmax, square=True,
            linewidths=0.5, cbar_kws={"pad": 0.05, "label": "Mean Pearson's r"}, fmt="",
            annot_kws={'fontfamily': 'Times New Roman'}
        )

        # Customize colorbar with explicit min and max ticks
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
        cbar.ax.set_ylabel("Group Correlation (r)", rotation=270, labelpad=20,
                          fontsize=12, weight="bold", fontfamily='Times New Roman')
        # Set colorbar tick label font
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontname("Times New Roman")
            tick.set_fontsize(12)

        # Increase font size for region labels
        plt.xticks(rotation=0, fontsize=12, fontfamily='Times New Roman')
        plt.yticks(rotation=0, fontsize=12, fontfamily='Times New Roman')

        # Increase title font size and remove color name
        plt.title("Group Mean Seed-to-Seed Connectivity", fontsize=16,
                  weight='bold', pad=20, fontfamily='Times New Roman')

        for b in [5, 8]:
            ax.axhline(b, color='lightgray')
            ax.axvline(b, color='lightgray')

        plt.tight_layout()

        # Save each figure
        path_svg = os.path.join(OUTPUT_DIR, f"group_mean_connectivity_matrix_{cmap}.svg")
        path_png = path_svg.replace(".svg", ".png")
        plt.savefig(path_svg, format="svg", dpi=300)
        plt.savefig(path_png, format="png", dpi=300)

        plt.show()  # Display each plot

        print(f"++ Group matrix figure saved: {path_svg}")
        print(f"++ Colormap range (ignoring diagonal): {vmin:.3f} to {vmax:.3f}")


if __name__ == "__main__":
    print("++ Starting connectivity pipeline...")
    process_all_subjects()
    generate_group_matrix()
    print("++ All done!")
