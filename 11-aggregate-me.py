import os
import pandas as pd
import glob
import numpy as np
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import fdrcorrection

# === CONFIGURATION ===
BASE_DIR = "/Users/similovesyou/Desktop/qts/simian-brain"
DATASET = "site-strasbourg"
DERIVATIVES_DIR = os.path.join(BASE_DIR, "data", DATASET, "derivatives")
FC_DIR = os.path.join(BASE_DIR, "functional-connectivity", DATASET)
TARGET_SEEDS = ["LIP", "FEF"]

def run_regionwise_ttests(combined_df):
    results = []
    for region in combined_df.index:
        values = combined_df.loc[region].dropna().values
        if len(values) < 2:
            continue
        t_stat, p_val = ttest_1samp(values, popmean=0)
        cohen_d = np.mean(values) / np.std(values, ddof=1)
        results.append({
            "Region": region,
            "t-stat": t_stat,
            "p-value": p_val,
            "Cohen's d": cohen_d,
            "n": len(values)
        })
    return pd.DataFrame(results)

def apply_fdr_correction(stats_df, alpha=0.05):
    pvals = stats_df["p-value"].values
    rejected, corrected_pvals = fdrcorrection(pvals, alpha=alpha)
    stats_df["FDR-corrected p"] = corrected_pvals
    stats_df["Significant (FDR < 0.05)"] = rejected
    return stats_df

def aggregate_results_for_seed(target_seed):
    input_dir = os.path.join(FC_DIR, f"seed-to-{target_seed}-connectivity")
    if not os.path.exists(input_dir):
        print(f"!! Directory does not exist: {input_dir}")
        return None

    csv_files = glob.glob(os.path.join(input_dir, "*_correlation_results.csv"))
    if not csv_files:
        print(f"!! No CSVs found for seed: {target_seed}")
        return None

    all_dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, index_col=0)
            filename = os.path.basename(file)
            subj_id = filename.split("_")[0]  # e.g., arwen or patchoulli
            df = df.rename(columns={"Correlation": subj_id})
            all_dfs.append(df)
        except Exception as e:
            print(f"!! Failed to load {file}: {e}")

    if not all_dfs:
        print(f"!! No valid CSVs to process for seed {target_seed}")
        return None

    combined = pd.concat(all_dfs, axis=1)

    # Save group mean
    mean_df = pd.DataFrame()
    mean_df["Correlation"] = combined.mean(axis=1)
    N = len(csv_files)
    mean_output_name = f"group_mean_correlation_subs-{N}_seed-to-{target_seed}.csv"
    mean_output_path = os.path.join(input_dir, mean_output_name)
    mean_df.to_csv(mean_output_path)
    print(f"++ Saved mean correlations for {target_seed} to {mean_output_path}")

    # Run t-tests + FDR
    stats_df = run_regionwise_ttests(combined)
    stats_df = apply_fdr_correction(stats_df)

    # Save full stats
    stats_output_name = f"regionwise_ttests_seed-to-{target_seed}.csv"
    stats_output_path = os.path.join(input_dir, stats_output_name)
    stats_df.to_csv(stats_output_path, index=False)
    print(f"++ Saved region-wise t-tests for {target_seed} to {stats_output_path}")

    # Save only significant ones
    sig_df = stats_df[stats_df["Significant (FDR < 0.05)"] == True]
    sig_output_name = f"significant_regions_FDR_seed-to-{target_seed}.csv"
    sig_output_path = os.path.join(input_dir, sig_output_name)
    sig_df.to_csv(sig_output_path, index=False)
    print(f"++ Significant regions saved to {sig_output_path}")
    print(sig_df.head())

    return sig_output_path

# === RUN FOR ALL TARGET SEEDS ===
if __name__ == "__main__":
    group_csvs = []
    for seed in TARGET_SEEDS:
        path = aggregate_results_for_seed(seed)
        if path:
            group_csvs.append(path)

    print("\n=== Group-Level Summary Complete ===")
    print("You can now use these files in your radar plot or significance reporting:")
    for path in group_csvs:
        print(path)