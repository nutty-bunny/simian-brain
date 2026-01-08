#!/usr/bin/env python3
"""
Preprocessing Quality Analysis for NHP fMRI Data
Generates descriptive statistics and visualizations for methods reporting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Configuration
CSV_FILE = "/Volumes/simians/simian-brain/data/site-strasbourg/final-derivatives-no-spatial-smoothing/preprocessing_summary.csv"
OUTPUT_DIR = "/Volumes/simians/simian-brain/data/site-strasbourg/final-derivatives-no-spatial-smoothing/qc_analysis"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print("Loading preprocessing summary data...")
df = pd.read_csv(CSV_FILE)

# Clean numeric columns
numeric_cols = ['Total_Volumes', 'Censored_Volumes', 'Censoring_Percent', 'Remaining_Volumes', 
                'Mean_FD', 'Median_FD', 'Max_FD', 'Censoring_Threshold', 'Mean_tSNR', 
                'Median_tSNR', 'Min_tSNR', 'Max_tSNR']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove subjects with missing data
df_clean = df.dropna(subset=['Mean_FD', 'Mean_tSNR', 'Total_Volumes'])

print(f"Loaded data for {len(df)} subjects")
print(f"Complete data available for {len(df_clean)} subjects")

# ===== DESCRIPTIVE STATISTICS =====
print("\n" + "="*50)
print("PREPROCESSING QUALITY ANALYSIS")
print("="*50)

# Basic sample characteristics
n_subjects = len(df_clean)
print(f"\nSample Size: {n_subjects} subjects")

# Volume statistics
vol_stats = {
    'Total volumes per subject': df_clean['Total_Volumes'].describe(),
    'Remaining volumes after censoring': df_clean['Remaining_Volumes'].describe(),
    'Censoring percentage': df_clean['Censoring_Percent'].describe()
}

print("\nVOLUME STATISTICS:")
print("-" * 30)
for key, stats_obj in vol_stats.items():
    print(f"{key}:")
    print(f"  Mean ± SD: {stats_obj['mean']:.1f} ± {stats_obj['std']:.1f}")
    print(f"  Range: {stats_obj['min']:.1f} - {stats_obj['max']:.1f}")
    print(f"  Median [IQR]: {stats_obj['50%']:.1f} [{stats_obj['25%']:.1f}-{stats_obj['75%']:.1f}]")
    print()

# Motion statistics  
print("MOTION PARAMETERS:")
print("-" * 30)
fd_stats = df_clean['Mean_FD'].describe()
print(f"Mean Framewise Displacement (mm):")
print(f"  Mean ± SD: {fd_stats['mean']:.3f} ± {fd_stats['std']:.3f}")
print(f"  Range: {fd_stats['min']:.3f} - {fd_stats['max']:.3f}")
print(f"  Median [IQR]: {fd_stats['50%']:.3f} [{fd_stats['25%']:.3f}-{fd_stats['75%']:.3f}]")

# Motion thresholds
motion_1mm = (df_clean['Mean_FD'] > 1.0).sum()
motion_15mm = (df_clean['Mean_FD'] > 1.5).sum() 
motion_2mm = (df_clean['Mean_FD'] > 2.0).sum()

print(f"\nMotion threshold analysis:")
print(f"  Subjects with mean FD > 1.0mm: {motion_1mm} ({motion_1mm/n_subjects*100:.1f}%)")
print(f"  Subjects with mean FD > 1.5mm: {motion_15mm} ({motion_15mm/n_subjects*100:.1f}%)")
print(f"  Subjects with mean FD > 2.0mm: {motion_2mm} ({motion_2mm/n_subjects*100:.1f}%)")

# tSNR statistics
print("\ntSNR QUALITY METRICS:")
print("-" * 30)
tsnr_stats = df_clean['Mean_tSNR'].describe()
print(f"Mean tSNR:")
print(f"  Mean ± SD: {tsnr_stats['mean']:.2f} ± {tsnr_stats['std']:.2f}")
print(f"  Range: {tsnr_stats['min']:.2f} - {tsnr_stats['max']:.2f}")
print(f"  Median [IQR]: {tsnr_stats['50%']:.2f} [{tsnr_stats['25%']:.2f}-{tsnr_stats['75%']:.2f}]")

# tSNR quality categories
tsnr_good = (df_clean['tSNR_Quality'] == 'GOOD').sum()
tsnr_acceptable = (df_clean['tSNR_Quality'] == 'ACCEPTABLE').sum()
tsnr_poor = (df_clean['tSNR_Quality'] == 'POOR').sum()

print(f"\ntSNR Quality Distribution:")
print(f"  Good (≥25): {tsnr_good} ({tsnr_good/n_subjects*100:.1f}%)")
print(f"  Acceptable (20-24.9): {tsnr_acceptable} ({tsnr_acceptable/n_subjects*100:.1f}%)")
print(f"  Poor (<20): {tsnr_poor} ({tsnr_poor/n_subjects*100:.1f}%)")

# Exclusion analysis
excluded_subjects = df_clean[df_clean['tSNR_Quality'] == 'POOR']['Monkey'].tolist()
final_n = n_subjects - tsnr_poor

print(f"\nEXCLUSION ANALYSIS:")
print("-" * 30)
print(f"Subjects recommended for exclusion (tSNR < 20): {tsnr_poor}")
if excluded_subjects:
    print(f"Excluded subjects: {', '.join(excluded_subjects)}")
print(f"Final sample size: {final_n} subjects")

# ===== CORRELATION ANALYSIS =====
print(f"\nCORRELATION ANALYSIS:")
print("-" * 30)

# Motion vs tSNR correlation
corr_motion_tsnr, p_motion_tsnr = stats.pearsonr(df_clean['Mean_FD'], df_clean['Mean_tSNR'])
print(f"Motion (FD) vs tSNR correlation: r = {corr_motion_tsnr:.3f}, p = {p_motion_tsnr:.3f}")

# Censoring vs tSNR correlation  
corr_censor_tsnr, p_censor_tsnr = stats.pearsonr(df_clean['Censoring_Percent'], df_clean['Mean_tSNR'])
print(f"Censoring % vs tSNR correlation: r = {corr_censor_tsnr:.3f}, p = {p_censor_tsnr:.3f}")

# ===== VISUALIZATION =====
print(f"\nGenerating visualizations...")

# Set up plotting parameters
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('NHP fMRI Preprocessing Quality Control', fontsize=16, y=0.98)

# 1. tSNR distribution
axes[0,0].hist(df_clean['Mean_tSNR'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].axvline(20, color='red', linestyle='--', label='Exclusion threshold')
axes[0,0].axvline(25, color='orange', linestyle='--', label='Good quality threshold')
axes[0,0].set_xlabel('Mean tSNR')
axes[0,0].set_ylabel('Number of Subjects')
axes[0,0].set_title('tSNR Distribution')
axes[0,0].legend()

# 2. Motion distribution
axes[0,1].hist(df_clean['Mean_FD'], bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
axes[0,1].axvline(1.0, color='orange', linestyle='--', label='1.0mm threshold')
axes[0,1].axvline(2.0, color='red', linestyle='--', label='2.0mm threshold')
axes[0,1].set_xlabel('Mean Framewise Displacement (mm)')
axes[0,1].set_ylabel('Number of Subjects')
axes[0,1].set_title('Motion Distribution')
axes[0,1].legend()

# 3. Censoring distribution
axes[0,2].hist(df_clean['Censoring_Percent'], bins=15, alpha=0.7, color='salmon', edgecolor='black')
axes[0,2].axvline(15, color='red', linestyle='--', label='Max allowed (15%)')
axes[0,2].set_xlabel('Censoring Percentage (%)')
axes[0,2].set_ylabel('Number of Subjects')
axes[0,2].set_title('Volume Censoring Distribution')
axes[0,2].legend()

# 4. Motion vs tSNR scatter
colors = {'GOOD': 'green', 'ACCEPTABLE': 'orange', 'POOR': 'red'}
for quality in colors.keys():
    subset = df_clean[df_clean['tSNR_Quality'] == quality]
    if len(subset) > 0:
        axes[1,0].scatter(subset['Mean_FD'], subset['Mean_tSNR'], 
                         c=colors[quality], label=quality, alpha=0.7)

axes[1,0].set_xlabel('Mean Framewise Displacement (mm)')
axes[1,0].set_ylabel('Mean tSNR')
axes[1,0].set_title(f'Motion vs tSNR (r = {corr_motion_tsnr:.3f})')
axes[1,0].legend()

# 5. tSNR quality pie chart
quality_counts = df_clean['tSNR_Quality'].value_counts()
axes[1,1].pie(quality_counts.values, labels=quality_counts.index, autopct='%1.1f%%',
              colors=['green', 'orange', 'red'])
axes[1,1].set_title('tSNR Quality Distribution')

# 6. Remaining volumes after censoring
axes[1,2].hist(df_clean['Remaining_Volumes'], bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
axes[1,2].set_xlabel('Remaining Volumes')
axes[1,2].set_ylabel('Number of Subjects')
axes[1,2].set_title('Usable Data After Censoring')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/preprocessing_qc_summary.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/preprocessing_qc_summary.pdf', bbox_inches='tight')

# Save detailed statistics to file
with open(f'{OUTPUT_DIR}/preprocessing_statistics.txt', 'w') as f:
    f.write("NHP fMRI Preprocessing Quality Control Analysis\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Sample Size: {n_subjects} subjects\n")
    f.write(f"Excluded (tSNR < 20): {tsnr_poor} subjects\n") 
    f.write(f"Final Sample: {final_n} subjects\n\n")
    
    f.write("VOLUME STATISTICS:\n")
    f.write(f"Total volumes: {df_clean['Total_Volumes'].mean():.1f} ± {df_clean['Total_Volumes'].std():.1f} (range: {df_clean['Total_Volumes'].min():.0f}-{df_clean['Total_Volumes'].max():.0f})\n")
    f.write(f"Censored volumes: {df_clean['Censored_Volumes'].mean():.1f} ± {df_clean['Censored_Volumes'].std():.1f}\n")
    f.write(f"Censoring percentage: {df_clean['Censoring_Percent'].mean():.1f} ± {df_clean['Censoring_Percent'].std():.1f}%\n")
    f.write(f"Remaining volumes: {df_clean['Remaining_Volumes'].mean():.1f} ± {df_clean['Remaining_Volumes'].std():.1f}\n\n")
    
    f.write("MOTION PARAMETERS:\n")
    f.write(f"Mean FD: {df_clean['Mean_FD'].mean():.3f} ± {df_clean['Mean_FD'].std():.3f} mm\n")
    f.write(f"Subjects with FD > 1.0mm: {motion_1mm} ({motion_1mm/n_subjects*100:.1f}%)\n")
    f.write(f"Subjects with FD > 2.0mm: {motion_2mm} ({motion_2mm/n_subjects*100:.1f}%)\n\n")
    
    f.write("tSNR QUALITY:\n")
    f.write(f"Mean tSNR: {df_clean['Mean_tSNR'].mean():.2f} ± {df_clean['Mean_tSNR'].std():.2f}\n")
    f.write(f"Good quality (≥25): {tsnr_good} ({tsnr_good/n_subjects*100:.1f}%)\n")
    f.write(f"Acceptable quality (20-24.9): {tsnr_acceptable} ({tsnr_acceptable/n_subjects*100:.1f}%)\n")
    f.write(f"Poor quality (<20): {tsnr_poor} ({tsnr_poor/n_subjects*100:.1f}%)\n\n")
    
    f.write("CORRELATIONS:\n")
    f.write(f"Motion vs tSNR: r = {corr_motion_tsnr:.3f}, p = {p_motion_tsnr:.3f}\n")
    f.write(f"Censoring vs tSNR: r = {corr_censor_tsnr:.3f}, p = {p_censor_tsnr:.3f}\n\n")
    
    if excluded_subjects:
        f.write("EXCLUDED SUBJECTS:\n")
        for subj in excluded_subjects:
            tsnr_val = df_clean[df_clean['Monkey'] == subj]['Mean_tSNR'].iloc[0]
            f.write(f"{subj}: tSNR = {tsnr_val:.2f}\n")

# Save summary table for included subjects
df_included = df_clean[df_clean['tSNR_Quality'] != 'POOR'].copy()
summary_table = df_included[['Monkey', 'Total_Volumes', 'Censoring_Percent', 'Remaining_Volumes', 
                           'Mean_FD', 'Mean_tSNR', 'tSNR_Quality']].copy()
summary_table.to_csv(f'{OUTPUT_DIR}/included_subjects_summary.csv', index=False)

print(f"\nAnalysis complete!")
print(f"Results saved to: {OUTPUT_DIR}/")
print(f"- preprocessing_qc_summary.png/pdf (visualizations)")
print(f"- preprocessing_statistics.txt (detailed statistics)")
print(f"- included_subjects_summary.csv (final sample)")

print(f"\nSUMMARY FOR METHODS SECTION:")
print("-" * 40)
print(f"Of {n_subjects} subjects, {tsnr_poor} were excluded due to poor tSNR (< 20), ")
print(f"leaving {final_n} subjects for analysis. The final sample showed mean ")
print(f"framewise displacement of {df_clean['Mean_FD'].mean():.3f} ± {df_clean['Mean_FD'].std():.3f} mm ")
print(f"and mean tSNR of {df_clean['Mean_tSNR'].mean():.2f} ± {df_clean['Mean_tSNR'].std():.2f}. ")
print(f"On average, {df_clean['Censoring_Percent'].mean():.1f}% ± {df_clean['Censoring_Percent'].std():.1f}% ")
print(f"of volumes were censored due to motion, leaving {df_clean['Remaining_Volumes'].mean():.0f} ± {df_clean['Remaining_Volumes'].std():.0f} ")
print(f"usable volumes per subject.")