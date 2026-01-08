import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

import os
import pickle
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data paths - UPDATE THESE FOR YOUR SYSTEM
BASE_DIR = "/Users/similovesyou/Desktop/qts/simian-behavior"
DATA_DIR = os.path.join(BASE_DIR, "data/py")
BRAIN_DIR = "/Users/similovesyou/Desktop/qts/simian-brain"
FC_DIR = os.path.join(BRAIN_DIR, "final-functional-connectivity-2mm/site-strasbourg/attention-seed-2-seed-connectivity")

# Demographics file with birthdates
DEMOGRAPHICS_FILE = "/Users/similovesyou/Desktop/qts/simian-brain/data/demographics/IRM-birthdates.xlsx"

# Files
PICKLE_FILE = os.path.join(DATA_DIR, "data.pickle")

# Brain scan dates
SCAN_DATES = {
    "amidala": "2024-10-17", "arwen": "2024-03-08", "baal": "2024-05-24",
    "berenice": "2025-02-14", "dory": "2024-09-27", "ficelle": "2025-01-17",
    "gabie": "2024-09-05", "gandhi": "2025-02-28", "horus": "2025-02-07",
    "indigo": "2024-03-15", "iron": "2024-10-31", "isis": "2025-03-14",
    "jazz": "2025-01-31", "jipsy": "2024-10-18", "joy": "2025-03-13",
    "karma": "2024-11-01", "kenobi": "2025-02-21", "kenya": "2024-11-29",
    "marouchka": "2024-10-25", "natasha": "2024-04-19", "nema": "2025-01-24",
    "radja": "2024-10-24", "samael": "2024-09-26", "volga": "2024-05-17",
    "yannick": "2025-01-30", "yin": "2024-11-08", "patsy": "2024-11-22"
}

# Analysis parameters to test
WINDOW_SIZES = [1, 3, 6, 12]  # months
MIN_ATTEMPTS_OPTIONS = [50, 100, 200]  # different thresholds to test
CHOSEN_WINDOW = 6  # Final choice based on optimization
CHOSEN_MIN_ATTEMPTS = 100

print("="*80)
print("PARAMETER-OPTIMIZED TEMPORAL BRAIN-BEHAVIOR ANALYSIS")
print("="*80)

# =============================================================================
# LOAD DEMOGRAPHICS DATA WITH REAL AGES
# =============================================================================

print("Loading demographics data with birthdates...")
demographics = pd.read_excel(DEMOGRAPHICS_FILE)

# Convert column names to lowercase for easier matching
demographics.columns = demographics.columns.str.lower()
print(f"Demographics columns: {list(demographics.columns)}")

# Convert dates
demographics['date of birth'] = pd.to_datetime(demographics['date of birth'])
demographics['mri date'] = pd.to_datetime(demographics['mri date'])

# Calculate age at scan
demographics['age_at_scan'] = (demographics['mri date'] - demographics['date of birth']).dt.days / 365.25

# Convert individual names to lowercase for matching
demographics['individual'] = demographics['individual'].str.lower()

print(f"Loaded demographics for {len(demographics)} subjects")
print(f"Age range: {demographics['age_at_scan'].min():.1f} - {demographics['age_at_scan'].max():.1f} years")

# Create age lookup dictionary
age_lookup = dict(zip(demographics['individual'], demographics['age_at_scan']))
species_lookup = dict(zip(demographics['individual'], demographics['species']))

print("Age lookup sample:", dict(list(age_lookup.items())[:5]))

# =============================================================================
# DATA LOADING
# =============================================================================

print("\nLoading behavioral data...")
with open(PICKLE_FILE, "rb") as handle:
    data = pickle.load(handle)

print(f"Found behavioral data for {len(data.get('rhesus', {}))} rhesus and {len(data.get('tonkean', {}))} tonkean subjects")

print("Loading brain connectivity data...")
def load_brain_connectivity():
    """Load brain connectivity for subjects with scan dates"""
    connectivity_data = {}
    
    for name in SCAN_DATES.keys():
        conn_path = os.path.join(FC_DIR, f"{name}_seed_connectivity.csv")
        if os.path.exists(conn_path):
            try:
                mat = pd.read_csv(conn_path, index_col=0)
                # Extract V1-MT connectivity
                if 'V1' in mat.index and 'MT' in mat.columns:
                    connectivity_data[name] = {'fc_V1_MT': mat.at['V1', 'MT']}
            except Exception as e:
                print(f"    Error loading {name}: {e}")
    
    print(f"  Loaded connectivity for {len(connectivity_data)} subjects")
    return connectivity_data

brain_data = load_brain_connectivity()

# =============================================================================
# TEMPORAL WINDOW OPTIMIZATION FUNCTIONS
# =============================================================================

def analyze_temporal_window(data, scan_dates, window_months=6, min_attempts=100):
    """
    Analyze behavioral data within a specified window before scan date
    """
    results = []
    
    for species, species_data in data.items():
        if species in ['hierarchy', 'plots']:  # Skip non-behavioral data
            continue
            
        for name, monkey_data in species_data.items():
            if name not in scan_dates or 'attempts' not in monkey_data:
                continue
                
            scan_date = pd.to_datetime(scan_dates[name])
            attempts = monkey_data['attempts'].copy()
            
            # Convert timestamps
            attempts['instant_begin'] = pd.to_datetime(attempts['instant_begin'], unit='ms', errors='coerce')
            attempts = attempts.dropna(subset=['instant_begin'])
            
            if len(attempts) == 0:
                continue
            
            # Define window: X months before scan date
            window_start = scan_date - pd.DateOffset(months=window_months)
            window_end = scan_date
            
            # Filter to window
            window_attempts = attempts[
                (attempts['instant_begin'] >= window_start) & 
                (attempts['instant_begin'] <= window_end)
            ]
            
            if len(window_attempts) >= min_attempts:
                # Calculate ALL behavioral metrics
                outcomes = window_attempts['result'].value_counts(normalize=True) if 'result' in window_attempts.columns else pd.Series()
                
                # Calculate temporal characteristics
                days_span = (window_attempts['instant_begin'].max() - window_attempts['instant_begin'].min()).days
                days_to_scan = (scan_date - window_attempts['instant_begin'].max()).days
                
                results.append({
                    'name': name,
                    'species': species,
                    'n_attempts': len(window_attempts),
                    'p_success': outcomes.get('success', 0.0),
                    'p_premature': outcomes.get('prematured', 0.0),  # Note: uses 'prematured'
                    'p_omission': outcomes.get('stepomission', 0.0),  # Note: uses 'stepomission'
                    'p_error': outcomes.get('error', 0.0),
                    'window_months': window_months,
                    'days_span': days_span,
                    'days_to_scan': days_to_scan,
                    'scan_date': scan_date.date()
                })
    
    return pd.DataFrame(results)

# =============================================================================
# PARAMETER OPTIMIZATION
# =============================================================================

print("\nTesting different temporal windows:")
print(f"{'Window':<8} {'N Subjects':<12} {'Mean Attempts':<15} {'Mean Success':<12} {'Mean Days to Scan':<18}")
print("-" * 75)

window_results = {}

for months in WINDOW_SIZES:
    window_data = analyze_temporal_window(data, SCAN_DATES, window_months=months, min_attempts=CHOSEN_MIN_ATTEMPTS)
    
    if len(window_data) > 0:
        mean_attempts = window_data['n_attempts'].mean()
        mean_success = window_data['p_success'].mean()
        mean_days_to_scan = window_data['days_to_scan'].mean()
        
        window_results[months] = window_data
        
        print(f"{months:>7} {len(window_data):>11} {mean_attempts:>14.0f} {mean_success:>11.3f} {mean_days_to_scan:>17.1f}")
    else:
        print(f"{months:>7} {0:>11} {'--':>14} {'--':>11} {'--':>17}")

print(f"\nTesting different minimum attempt thresholds (using {CHOSEN_WINDOW}-month window):")
print(f"{'Min Attempts':<12} {'N Subjects':<12} {'Mean Attempts':<15} {'Range':<15}")
print("-" * 60)

for threshold in MIN_ATTEMPTS_OPTIONS:
    thresh_data = analyze_temporal_window(data, SCAN_DATES, window_months=CHOSEN_WINDOW, min_attempts=threshold)
    
    if len(thresh_data) > 0:
        mean_attempts = thresh_data['n_attempts'].mean()
        min_attempts = thresh_data['n_attempts'].min()
        max_attempts = thresh_data['n_attempts'].max()
        print(f"{threshold:>11} {len(thresh_data):>11} {mean_attempts:>14.0f} {min_attempts:.0f}-{max_attempts:.0f}")
    else:
        print(f"{threshold:>11} {0:>11} {'--':>14} {'--':<15}")

# =============================================================================
# GENERATE FINAL DATA WITH OPTIMAL PARAMETERS AND REAL AGES
# =============================================================================

print(f"\n{'='*80}")
print(f"GENERATING FINAL DATA WITH OPTIMAL PARAMETERS AND REAL AGES")
print(f"Window: {CHOSEN_WINDOW} months, Min attempts: {CHOSEN_MIN_ATTEMPTS}")
print("="*80)

# Get behavioral data with chosen parameters
behavioral_data = analyze_temporal_window(data, SCAN_DATES, window_months=CHOSEN_WINDOW, min_attempts=CHOSEN_MIN_ATTEMPTS)

if len(behavioral_data) > 0:
    # Add brain connectivity data
    behavioral_data['fc_V1_MT'] = behavioral_data['name'].map(lambda x: brain_data.get(x, {}).get('fc_V1_MT', np.nan))
    
    # Remove subjects without brain data
    final_data = behavioral_data.dropna(subset=['fc_V1_MT']).copy()
    
    # Add REAL age data from demographics file
    def get_real_age(name):
        """Get real age from demographics data"""
        name_lower = name.lower()
        if name_lower in age_lookup:
            return age_lookup[name_lower]
        else:
            print(f"Warning: No age data found for {name}")
            return np.nan
    
    final_data['age_at_scan'] = final_data['name'].apply(get_real_age)
    
    # Remove subjects without age data
    subjects_before = len(final_data)
    final_data = final_data.dropna(subset=['age_at_scan']).copy()
    subjects_after = len(final_data)
    
    if subjects_before > subjects_after:
        print(f"Removed {subjects_before - subjects_after} subjects without age data")
    
    # Add subject identifier for visualization script
    final_data['subject'] = final_data['name']
    
    # Update species from demographics (more accurate than behavioral data)
    def get_species_from_demographics(name):
        """Get species from demographics data"""
        name_lower = name.lower()
        if name_lower in species_lookup:
            species_full = species_lookup[name_lower]
            if 'mulatta' in species_full.lower():
                return 'rhesus'
            elif 'tonkeana' in species_full.lower():
                return 'tonkean'
        return final_data[final_data['name'] == name]['species'].iloc[0] if len(final_data[final_data['name'] == name]) > 0 else 'unknown'
    
    final_data['species'] = final_data['name'].apply(get_species_from_demographics)
    
    # Reorder columns to match expected format
    column_order = ['subject', 'name', 'species', 'age_at_scan', 'fc_V1_MT', 
                   'p_success', 'p_premature', 'p_omission', 'p_error', 
                   'n_attempts', 'window_months', 'days_span', 'days_to_scan', 'scan_date']
    
    final_data = final_data.reindex(columns=[col for col in column_order if col in final_data.columns])
    
    print(f"\nFINAL DATA SUMMARY:")
    print(f"Total subjects: {len(final_data)}")
    print(f"Species distribution: {final_data['species'].value_counts().to_dict()}")
    print(f"Age range: {final_data['age_at_scan'].min():.1f} - {final_data['age_at_scan'].max():.1f} years")
    print(f"Brain connectivity range: {final_data['fc_V1_MT'].min():.3f} - {final_data['fc_V1_MT'].max():.3f}")
    
    print(f"\nBehavioral measures summary:")
    behavioral_cols = ['p_success', 'p_premature', 'p_omission', 'p_error']
    for col in behavioral_cols:
        if col in final_data.columns:
            mean_val = final_data[col].mean()
            std_val = final_data[col].std()
            print(f"  {col}: {mean_val:.3f} ± {std_val:.3f}")
    
    print(f"\nAttempts per subject: {final_data['n_attempts'].mean():.0f} ± {final_data['n_attempts'].std():.0f}")
    print(f"Temporal distance to scan: {final_data['days_to_scan'].mean():.1f} ± {final_data['days_to_scan'].std():.1f} days")
    
    # Show individual subject data
    print(f"\nIndividual subject data:")
    print(f"{'Name':<12} {'Species':<8} {'Age':<6} {'Success':<8} {'Brain FC':<10}")
    print("-" * 50)
    for _, row in final_data.iterrows():
        print(f"{row['name']:<12} {row['species']:<8} {row['age_at_scan']:>5.1f} {row['p_success']:>7.3f} {row['fc_V1_MT']:>9.3f}")
    
    # Quick correlation check with REAL ages
    print(f"\nCorrelations with REAL ages:")
    for behavioral_col in behavioral_cols:
        if behavioral_col in final_data.columns:
            r, p = stats.pearsonr(final_data[behavioral_col], final_data['fc_V1_MT'])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"{behavioral_col} vs fc_V1_MT: r = {r:.3f}, p = {p:.3f} {sig}")
    
    # Age correlations
    print(f"\nAge correlations:")
    for behavioral_col in behavioral_cols:
        if behavioral_col in final_data.columns:
            r, p = stats.pearsonr(final_data[behavioral_col], final_data['age_at_scan'])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"{behavioral_col} vs age: r = {r:.3f}, p = {p:.3f} {sig}")
    
    r_brain_age, p_brain_age = stats.pearsonr(final_data['fc_V1_MT'], final_data['age_at_scan'])
    sig_brain_age = "***" if p_brain_age < 0.001 else "**" if p_brain_age < 0.01 else "*" if p_brain_age < 0.05 else "ns"
    print(f"fc_V1_MT vs age: r = {r_brain_age:.3f}, p = {p_brain_age:.3f} {sig_brain_age}")
    
    print(f"\n{'='*80}")
    print("DATA READY FOR VISUALIZATION WITH REAL AGES!")
    print(f"Run your visualization script now - all 4 behavioral measures are available with actual ages.")
    print("="*80)
    
else:
    print("No data could be matched with chosen parameters!")
    final_data = pd.DataFrame()

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

def create_single_behavior_plot(final_data, behavior_measure, behavior_label, plot_title):
    """
    Create a single brain-behavior-age plot for a specific behavioral measure
    """
    
    # Use original viridis-style colormap
    colors_age = ['#440154', '#31688e', '#35b779']
    cmap_age = LinearSegmentedColormap.from_list('custom_age', colors_age)
    
    # Species markers and colors - square for rhesus, circle for tonkean
    # Updated with scientific names in italics
    species_markers = {'rhesus': 's', 'tonkean': 'o'}  # square vs circle
    species_colors = {'rhesus': 'steelblue', 'tonkean': 'purple'}
    species_labels = {'rhesus': r'$\mathit{M.\ mulatta}$', 'tonkean': r'$\mathit{M.\ tonkeana}$'}
    
    # Calculate correlations for annotations
    r_sb, p_sb = stats.pearsonr(final_data[behavior_measure], final_data['fc_V1_MT'])
    r_sa, p_sa = stats.pearsonr(final_data[behavior_measure], final_data['age_at_scan'])
    r_ba, p_ba = stats.pearsonr(final_data['fc_V1_MT'], final_data['age_at_scan'])
    
    # Create figure with modified layout
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
    
    # =============================================================================
    # 1. MAIN 3D PLOT - Center stage
    # =============================================================================
    ax_3d = fig.add_subplot(gs[0:2, 1:3], projection='3d')
    
    # Create 3D scatter with species differentiation
    for species in final_data['species'].unique():
        species_data = final_data[final_data['species'] == species]
        
        scatter = ax_3d.scatter(
            species_data[behavior_measure], 
            species_data['fc_V1_MT'],
            species_data['age_at_scan'],
            c=species_data['age_at_scan'],
            s=120,
            alpha=0.8,
            cmap=cmap_age,
            marker=species_markers[species],
            edgecolors='white',
            linewidth=1.5,
            label=species_labels[species]  # Updated to use scientific names
        )
    
    ax_3d.set_xlabel(behavior_label, fontweight='bold', fontsize=12)
    ax_3d.set_ylabel('V1-MT Connectivity', fontweight='bold', fontsize=12)
    ax_3d.set_zlabel('Age (years)', fontweight='bold', fontsize=12)
    ax_3d.set_title(f'{plot_title}\nAcross Macaque Lifespan', 
                    fontweight='bold', fontsize=16, pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax_3d, shrink=0.6, pad=0.1)
    cbar.set_label('Age (years)', fontweight='bold')
    
    # Create custom legend for species with markers only
    legend_elements = []
    for species in sorted(final_data['species'].unique()):
        legend_elements.append(
            plt.Line2D([0], [0], marker=species_markers[species], color='w', 
                      markerfacecolor=species_colors[species], markersize=10,
                      label=species_labels[species], linestyle='None')  # Updated to use scientific names
        )
    
    ax_3d.legend(handles=legend_elements, loc='upper left', fontsize=12, frameon=True, 
                fancybox=True, shadow=True)
    
    # =============================================================================
    # 2. BEHAVIOR vs BRAIN CONNECTIVITY (main relationship)
    # =============================================================================
    ax_sb = fig.add_subplot(gs[0, 0])
    
    # Scatter plot with age coloring and species markers
    for species in final_data['species'].unique():
        species_data = final_data[final_data['species'] == species]
        ax_sb.scatter(
            species_data[behavior_measure], 
            species_data['fc_V1_MT'],
            c=species_data['age_at_scan'],
            s=100,
            alpha=0.8,
            cmap=cmap_age,
            marker=species_markers[species],
            edgecolors='white',
            linewidth=1.5
        )
    
    # Add regression line in dark blue (only if significant)
    if p_sb < 0.05:
        z = np.polyfit(final_data[behavior_measure], final_data['fc_V1_MT'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(final_data[behavior_measure].min(), final_data[behavior_measure].max(), 100)
        ax_sb.plot(x_line, p(x_line), color='#1f4e79', linestyle='--', alpha=0.8, linewidth=2)
    
    # Formatting
    ax_sb.set_xlabel(behavior_label, fontweight='bold')
    ax_sb.set_ylabel('V1-MT Connectivity', fontweight='bold')
    ax_sb.set_title('Core Brain-Behavior Relationship', fontweight='bold', fontsize=14)
    ax_sb.grid(True, alpha=0.3)
    
    # Add correlation info
    sig_text = "***" if p_sb < 0.001 else "**" if p_sb < 0.01 else "*" if p_sb < 0.05 else "ns"
    ax_sb.text(0.05, 0.95, f'r = {r_sb:.3f}\np = {p_sb:.3f} {sig_text}', 
               transform=ax_sb.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
               fontsize=11, fontweight='bold')
    
    # =============================================================================
    # 3. AGE vs BEHAVIOR
    # =============================================================================
    ax_as = fig.add_subplot(gs[0, 3])
    
    # Species-specific colors and markers
    for species in final_data['species'].unique():
        species_data = final_data[final_data['species'] == species]
        ax_as.scatter(
            species_data['age_at_scan'],
            species_data[behavior_measure],
            c=species_colors[species],
            s=100,
            alpha=0.8,
            marker=species_markers[species],
            edgecolors='white',
            linewidth=1.5
        )
    
    # Regression line in dark blue (only if significant)
    if p_sa < 0.05:
        z_as = np.polyfit(final_data['age_at_scan'], final_data[behavior_measure], 1)
        p_as = np.poly1d(z_as)
        x_line_as = np.linspace(final_data['age_at_scan'].min(), final_data['age_at_scan'].max(), 100)
        ax_as.plot(x_line_as, p_as(x_line_as), color='#1f4e79', linestyle='--', alpha=0.8, linewidth=2)
    
    ax_as.set_xlabel('Age (years)', fontweight='bold')
    ax_as.set_ylabel(behavior_label, fontweight='bold')
    ax_as.set_title('Age vs Performance', fontweight='bold', fontsize=14)
    ax_as.grid(True, alpha=0.3)
    
    # Add correlation
    sig_text_as = "***" if p_sa < 0.001 else "**" if p_sa < 0.01 else "*" if p_sa < 0.05 else "ns"
    ax_as.text(0.05, 0.95, f'r = {r_sa:.3f}\np = {p_sa:.3f} {sig_text_as}', 
               transform=ax_as.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
               fontsize=11)
    
    # =============================================================================
    # 4. AGE vs BRAIN CONNECTIVITY
    # =============================================================================
    ax_ab = fig.add_subplot(gs[1, 0])
    
    # Species-specific plot
    for species in final_data['species'].unique():
        species_data = final_data[final_data['species'] == species]
        ax_ab.scatter(
            species_data['age_at_scan'],
            species_data['fc_V1_MT'],
            c=species_colors[species],
            s=100,
            alpha=0.8,
            marker=species_markers[species],
            edgecolors='white',
            linewidth=1.5
        )
    
    # Regression line in dark blue (only if significant)
    if p_ba < 0.05:
        z_ab = np.polyfit(final_data['age_at_scan'], final_data['fc_V1_MT'], 1)
        p_ab = np.poly1d(z_ab)
        x_line_ab = np.linspace(final_data['age_at_scan'].min(), final_data['age_at_scan'].max(), 100)
        ax_ab.plot(x_line_ab, p_ab(x_line_ab), color='#1f4e79', linestyle='--', alpha=0.8, linewidth=2)
    
    ax_ab.set_xlabel('Age (years)', fontweight='bold')
    ax_ab.set_ylabel('V1-MT Connectivity', fontweight='bold')
    ax_ab.set_title('Age vs Brain Connectivity', fontweight='bold', fontsize=14)
    ax_ab.grid(True, alpha=0.3)
    
    # Add correlation
    sig_text_ab = "***" if p_ba < 0.001 else "**" if p_ba < 0.01 else "*" if p_ba < 0.05 else "ns"
    ax_ab.text(0.05, 0.95, f'r = {r_ba:.3f}\np = {p_ba:.3f} {sig_text_ab}', 
               transform=ax_ab.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
               fontsize=11)
    
    # =============================================================================
    # 5. RESIDUALS PLOT (Age-adjusted relationship)
    # =============================================================================
    ax_resid = fig.add_subplot(gs[1, 3])
    
    # Calculate residuals
    age_matrix = final_data['age_at_scan'].values.reshape(-1, 1)
    
    from sklearn.linear_model import LinearRegression
    behavior_model = LinearRegression().fit(age_matrix, final_data[behavior_measure])
    behavior_residuals = final_data[behavior_measure] - behavior_model.predict(age_matrix)
    
    brain_model = LinearRegression().fit(age_matrix, final_data['fc_V1_MT'])
    brain_residuals = final_data['fc_V1_MT'] - brain_model.predict(age_matrix)
    
    partial_r, partial_p = stats.pearsonr(behavior_residuals, brain_residuals)
    
    # Plot residuals with species markers
    for species in final_data['species'].unique():
        species_mask = final_data['species'] == species
        ax_resid.scatter(
            behavior_residuals[species_mask],
            brain_residuals[species_mask],
            c=species_colors[species],
            s=100,
            alpha=0.8,
            marker=species_markers[species],
            edgecolors='white',
            linewidth=1.5
        )
    
    # Regression line for residuals in dark blue (only if significant)
    if partial_p < 0.05:
        z_resid = np.polyfit(behavior_residuals, brain_residuals, 1)
        p_resid = np.poly1d(z_resid)
        x_line_resid = np.linspace(behavior_residuals.min(), behavior_residuals.max(), 100)
        ax_resid.plot(x_line_resid, p_resid(x_line_resid), color='#1f4e79', linestyle='--', alpha=0.8, linewidth=2)
    
    ax_resid.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax_resid.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    ax_resid.set_xlabel(f'{behavior_label} (age-adjusted)', fontweight='bold')
    ax_resid.set_ylabel('V1-MT (age-adjusted)', fontweight='bold')
    ax_resid.set_title('Age-Adjusted Relationship', fontweight='bold', fontsize=14)
    ax_resid.grid(True, alpha=0.3)
    
    # Add partial correlation with highlighting
    sig_text_partial = "***" if partial_p < 0.001 else "**" if partial_p < 0.01 else "*" if partial_p < 0.05 else "ns"
    ax_resid.text(0.05, 0.95, f'partial r = {partial_r:.3f}\np = {partial_p:.3f} {sig_text_partial}', 
                  transform=ax_resid.transAxes, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='navy'),
                  fontsize=11, fontweight='bold')
    
    # Overall title
    fig.suptitle(f'{plot_title} in Macaque Attention Networks', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save as SVG with text as paths for Illustrator compatibility
    import matplotlib
    matplotlib.rcParams['svg.fonttype'] = 'none'  # Keep fonts as text (editable in Illustrator)
    
    svg_filename = f"/Users/similovesyou/Desktop/qts/simian-behavior/plots/brain-behavior/MT-V1/{behavior_measure}_plot.svg"
    if not os.path.exists(os.path.dirname(svg_filename)):
        os.makedirs(os.path.dirname(svg_filename))
    plt.savefig(svg_filename, format='svg', bbox_inches='tight')
    print(f"  Saved plot to {svg_filename}")
    plt.show()
    
    return fig

# Create all four plots
def create_all_behavior_plots(final_data):
    """
    Create separate plots for all four behavioral measures
    """
    
    behavioral_measures = [
        ('p_success', 'Success Proportion', 'Brain-Success-Age Relationships'),
        ('p_premature', 'Premature Proportion', 'Brain-Premature-Age Relationships'),
        ('p_omission', 'Omission Proportion', 'Brain-Omission-Age Relationships'),
        ('p_error', 'Error Proportion', 'Brain-Error-Age Relationships')
    ]
    
    figures = []
    
    for measure, label, title in behavioral_measures:
        if measure in final_data.columns:
            print(f"Creating plot for {label}...")
            fig = create_single_behavior_plot(final_data, measure, label, title)
            figures.append((measure, fig))
        else:
            print(f"Warning: {measure} not found in data")
    
    return figures

# Create the visualizations
if 'final_data' in locals() and len(final_data) > 0:
    all_figures = create_all_behavior_plots(final_data)
    print(f"Created {len(all_figures)} comprehensive brain-behavior-age visualizations!")
else:
    print("Run the temporal analysis script first to generate 'final_data'")


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np

def create_subject_timeline_plot(data, scan_dates, window_months=6, min_attempts=100):
    """
    Create a timeline plot showing when each subject played during the window
    Each row is a subject, dots show days with gameplay
    """
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Match colors from main plots
    species_colors = {'rhesus': 'steelblue', 'tonkean': 'purple'}
    species_labels = {'rhesus': r'$\mathit{M.\ mulatta}$', 'tonkean': r'$\mathit{M.\ tonkeana}$'}
    
    subjects_plotted = []
    y_positions = []
    y_pos = 0
    
    for species, species_data in data.items():
        if species in ['hierarchy', 'plots']:
            continue
            
        for name, monkey_data in species_data.items():
            if name not in scan_dates or 'attempts' not in monkey_data:
                continue
                
            scan_date = pd.to_datetime(scan_dates[name])
            attempts = monkey_data['attempts'].copy()
            
            # Convert timestamps
            attempts['instant_begin'] = pd.to_datetime(attempts['instant_begin'], unit='ms', errors='coerce')
            attempts = attempts.dropna(subset=['instant_begin'])
            
            if len(attempts) == 0:
                continue
            
            # Define window
            window_start = scan_date - pd.DateOffset(months=window_months)
            window_end = scan_date
            
            # Filter to window
            window_attempts = attempts[
                (attempts['instant_begin'] >= window_start) & 
                (attempts['instant_begin'] <= window_end)
            ]
            
            if len(window_attempts) >= min_attempts:
                # Get unique days with gameplay
                window_attempts['date_only'] = window_attempts['instant_begin'].dt.date
                unique_days = window_attempts['date_only'].unique()
                n_attempts = len(window_attempts)
                n_days = len(unique_days)
                
                # Plot each day as a dot
                color = species_colors.get(species, 'gray')
                ax.scatter([pd.to_datetime(d) for d in unique_days], 
                          [y_pos] * len(unique_days),
                          c=color, s=40, alpha=0.8, zorder=3, edgecolors='white', linewidths=0.5)
                
                # Add scan date marker
                ax.scatter([scan_date], [y_pos], 
                          marker='*', s=400, c='#f39c12',  # Gold star
                          edgecolors='#d68910', linewidth=1.5,
                          zorder=4, label='Scan date' if y_pos == 0 else '')
                
                # Add window shading
                ax.add_patch(Rectangle((mdates.date2num(window_start), y_pos - 0.4),
                                      mdates.date2num(window_end) - mdates.date2num(window_start),
                                      0.8, alpha=0.15, color=color, zorder=1))
                
                # Add subject label with attempt info
                subjects_plotted.append(f"{name}  (n={n_attempts}, {n_days}d)")
                y_positions.append(y_pos)
                y_pos += 1
    
    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(subjects_plotted, fontsize=10, fontweight='500')
    ax.set_xlabel('Date', fontweight='bold', fontsize=13)
    ax.set_ylabel('Subject', fontweight='bold', fontsize=13)
    
    # Better title
    ax.set_title(f'Behavioral Testing Timeline: {window_months}-Month Pre-Scan Window\nEach dot represents a testing day', 
                 fontweight='bold', fontsize=15, pad=15)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    if window_months <= 6:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    elif window_months <= 12:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Grid
    ax.grid(True, axis='x', alpha=0.3, zorder=0, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Add subtle background color
    ax.set_facecolor('#fafafa')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', 
               markersize=10, label=species_labels['rhesus'], markeredgecolor='white', markeredgewidth=0.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
               markersize=10, label=species_labels['tonkean'], markeredgecolor='white', markeredgewidth=0.5),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#f39c12', 
               markersize=14, markeredgecolor='#d68910', markeredgewidth=1.5, label='Brain scan date')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, frameon=True, 
             fancybox=True, shadow=True, framealpha=0.95)
    
    # Add summary text
    total_attempts = sum([int(s.split('n=')[1].split(',')[0]) for s in subjects_plotted])
    total_days = sum([int(s.split(', ')[1].split('d')[0]) for s in subjects_plotted])
    summary_text = f'Total: {len(subjects_plotted)} subjects, {total_attempts} attempts, {total_days} testing days'
    ax.text(0.5, -0.08, summary_text, transform=ax.transAxes, 
            ha='center', fontsize=11, style='italic', color='#555')
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Create timeline plots for different windows
if 'data' in locals() and 'SCAN_DATES' in locals():
    print("Creating 6-month timeline...")
    timeline_6m = create_subject_timeline_plot(data, SCAN_DATES, 
                                               window_months=6, 
                                               min_attempts=CHOSEN_MIN_ATTEMPTS)
    
    print("\nCreating 12-month timeline...")
    timeline_12m = create_subject_timeline_plot(data, SCAN_DATES, 
                                                window_months=12, 
                                                min_attempts=CHOSEN_MIN_ATTEMPTS)
    
    print("\nCreating 24-month timeline...")
    timeline_24m = create_subject_timeline_plot(data, SCAN_DATES, 
                                                window_months=24, 
                                                min_attempts=CHOSEN_MIN_ATTEMPTS)
    
    print("\nAll timeline visualizations complete!")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

def create_individual_performance_plots(data, scan_dates, window_months=6, min_attempts=100):
    """
    Create individual plots for each subject showing daily success rate
    """
    
    # Match colors from main plots
    species_colors = {'rhesus': 'steelblue', 'tonkean': 'purple'}
    species_labels = {'rhesus': r'$\mathit{M.\ mulatta}$', 'tonkean': r'$\mathit{M.\ tonkeana}$'}
    
    subjects_to_plot = []
    
    # First, collect all subjects that meet criteria
    for species, species_data in data.items():
        if species in ['hierarchy', 'plots']:
            continue
            
        for name, monkey_data in species_data.items():
            if name not in scan_dates or 'attempts' not in monkey_data:
                continue
                
            scan_date = pd.to_datetime(scan_dates[name])
            attempts = monkey_data['attempts'].copy()
            
            # Convert timestamps
            attempts['instant_begin'] = pd.to_datetime(attempts['instant_begin'], unit='ms', errors='coerce')
            attempts = attempts.dropna(subset=['instant_begin'])
            
            if len(attempts) == 0 or 'result' not in attempts.columns:
                continue
            
            # Define window
            window_start = scan_date - pd.DateOffset(months=window_months)
            window_end = scan_date
            
            # Filter to window
            window_attempts = attempts[
                (attempts['instant_begin'] >= window_start) & 
                (attempts['instant_begin'] <= window_end)
            ].copy()
            
            if len(window_attempts) >= min_attempts:
                subjects_to_plot.append((name, species, window_attempts, scan_date, window_start, window_end))
    
    # Create subplot grid
    n_subjects = len(subjects_to_plot)
    n_cols = 4
    n_rows = int(np.ceil(n_subjects / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 4*n_rows))
    axes = axes.flatten() if n_subjects > 1 else [axes]
    
    for idx, (name, species, window_attempts, scan_date, window_start, window_end) in enumerate(subjects_to_plot):
        ax = axes[idx]
        
        # Calculate daily success rate
        window_attempts['date'] = window_attempts['instant_begin'].dt.date
        daily_stats = window_attempts.groupby('date').agg({
            'result': lambda x: (x == 'success').sum() / len(x)
        }).reset_index()
        daily_stats.columns = ['date', 'success_rate']
        daily_stats['date'] = pd.to_datetime(daily_stats['date'])
        
        # Calculate number of trials per day
        daily_counts = window_attempts.groupby('date').size().reset_index()
        daily_counts.columns = ['date', 'n_trials']
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        
        # Merge
        daily_stats = daily_stats.merge(daily_counts, on='date')
        
        color = species_colors.get(species, 'gray')
        
        # Plot daily success rate
        ax.plot(daily_stats['date'], daily_stats['success_rate'], 
               color=color, alpha=0.7, linewidth=2, marker='o', markersize=4)
        
        # Add scan date marker
        ax.axvline(scan_date, color='#f39c12', linestyle='--', linewidth=2, alpha=0.8, label='Scan date')
        
        # Add mean line
        mean_success = daily_stats['success_rate'].mean()
        ax.axhline(mean_success, color='black', linestyle=':', linewidth=1, alpha=0.5, label=f'Mean: {mean_success:.2f}')
        
        # Formatting
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Date', fontsize=9)
        ax.set_ylabel('Success Rate', fontsize=9)
        ax.set_title(f'{name} ({species_labels[species]})\nn={len(window_attempts)} trials, {len(daily_stats)} days', 
                    fontweight='bold', fontsize=10)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        if window_months <= 6:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')
        ax.legend(fontsize=8, loc='best')
    
    # Hide unused subplots
    for idx in range(n_subjects, len(axes)):
        axes[idx].axis('off')
    
    # Overall title
    fig.suptitle(f'Daily Success Rate Per Subject: {window_months}-Month Pre-Scan Window', 
                fontweight='bold', fontsize=16, y=0.995)
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Create individual performance plots
if 'data' in locals() and 'SCAN_DATES' in locals():
    
    print("Creating 6-month individual performance plots...")
    fig_6m = create_individual_performance_plots(data, SCAN_DATES, 
                                                 window_months=6, 
                                                 min_attempts=CHOSEN_MIN_ATTEMPTS)
    
    print("\nCreating 12-month individual performance plots...")
    fig_12m = create_individual_performance_plots(data, SCAN_DATES, 
                                                  window_months=12, 
                                                  min_attempts=CHOSEN_MIN_ATTEMPTS)
    
    print("\nAll individual performance plots complete!")

from scipy.stats import gaussian_kde
import os

def create_subject_timeline_plot(data, scan_dates, window_months=6, min_attempts=100, save_path=None):
    """
    Create a timeline plot showing when each subject played during the window
    X-axis shows days relative to scan date (scan = day 0)
    """
    fig, ax = plt.subplots(figsize=(24, 14))  # MUCH BIGGER
    
    # Match colors from main plots
    species_colors = {'rhesus': 'steelblue', 'tonkean': 'purple'}
    species_labels = {'rhesus': r'$\mathit{M.\ mulatta}$', 'tonkean': r'$\mathit{M.\ tonkeana}$'}
    
    subjects_plotted = []
    y_positions = []
    y_pos = 0
    
    for species, species_data in data.items():
        if species in ['hierarchy', 'plots']:
            continue
            
        for name, monkey_data in species_data.items():
            if name not in scan_dates or 'attempts' not in monkey_data:
                continue
                
            scan_date = pd.to_datetime(scan_dates[name])
            attempts = monkey_data['attempts'].copy()
            
            # Convert timestamps
            attempts['instant_begin'] = pd.to_datetime(attempts['instant_begin'], unit='ms', errors='coerce')
            attempts = attempts.dropna(subset=['instant_begin'])
            
            if len(attempts) == 0:
                continue
            
            # Define window
            window_start = scan_date - pd.DateOffset(months=window_months)
            window_end = scan_date
            
            # Filter to window
            window_attempts = attempts[
                (attempts['instant_begin'] >= window_start) & 
                (attempts['instant_begin'] <= window_end)
            ]
            
            if len(window_attempts) >= min_attempts:
                # Calculate days relative to scan
                window_attempts['days_to_scan'] = (scan_date - window_attempts['instant_begin']).dt.days
                
                # Get unique days with gameplay
                window_attempts['date_only'] = window_attempts['instant_begin'].dt.date
                unique_days_relative = window_attempts.groupby('date_only')['days_to_scan'].first().values
                
                n_attempts = len(window_attempts)
                n_days = len(unique_days_relative)
                
                # Calculate daily attempt counts
                daily_counts = window_attempts.groupby('days_to_scan').size().reset_index()
                daily_counts.columns = ['days_to_scan', 'count']
                
                color = species_colors.get(species, 'gray')
                
                # Create violin/wave shape based on activity density
                if len(daily_counts) > 3:
                    days_numeric = daily_counts['days_to_scan'].values
                    weights = daily_counts['count'].values
                    
                    # Create weighted points for KDE
                    weighted_points = np.repeat(days_numeric, weights.astype(int))
                    
                    if len(weighted_points) > 1:
                        # Compute KDE
                        kde = gaussian_kde(weighted_points, bw_method=0.15)
                        
                        # Evaluate KDE on a grid
                        max_days = window_months * 30
                        x_grid = np.linspace(0, max_days, 500)
                        density = kde(x_grid)
                        
                        # Normalize density for plotting (violin width)
                        density = density / density.max() * 0.35  # Scale to ±0.35
                        
                        # Plot violin shape (horizontal fill_between)
                        ax.fill_between(-x_grid,  # negative for backwards time
                                      y_pos - density,
                                      y_pos + density,
                                      alpha=0.3,
                                      color=color,
                                      zorder=1,
                                      linewidth=0)
                
                # Plot dots for each testing day (negative for backwards time)
                ax.scatter([-d for d in unique_days_relative],
                          [y_pos] * len(unique_days_relative),
                          c=color,
                          s=200,
                          alpha=0.85,
                          zorder=3,
                          edgecolors='white',
                          linewidths=2)
                
                # Capitalize name - TWO ROW FORMAT
                name_cap = name.capitalize()
                subjects_plotted.append(f"{name_cap}\n{n_days}d, n={n_attempts}")
                y_positions.append(y_pos)
                y_pos += 1
    
    # Formatting - MUCH BIGGER LABELS (changed fontweight to 'bold')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(subjects_plotted, fontsize=20, fontweight='bold')
    ax.set_xlabel('Days Before Scan', fontweight='bold', fontsize=26)
    ax.set_ylabel('Subject', fontweight='bold', fontsize=26)
    
    # Simpler title - BIGGER
    ax.set_title(f'{window_months}-Month Pre-Scan Activity', 
                fontweight='bold', fontsize=30, pad=30)
    
    # Set x-axis limits
    max_days = window_months * 30
    ax.set_xlim(-max_days, 10)
    
    # Format x-axis - MUCH BIGGER TICKS
    ax.tick_params(axis='x', labelsize=20, width=2, length=8)
    ax.tick_params(axis='y', width=2, length=8)
    
    # Add vertical line at scan date (x=0) - THICKER
    ax.axvline(0, color='#f39c12', linestyle='-', linewidth=4, alpha=0.6, zorder=0)
    
    # Grid - MORE STYLISTIC
    ax.grid(True, axis='x', alpha=0.3, zorder=0, linestyle=':', linewidth=2, color='gray')
    ax.set_axisbelow(True)
    
    # FULLY WHITE BACKGROUND
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# Create timeline plots for different windows
if 'data' in locals() and 'SCAN_DATES' in locals():
    # Define save directory
    save_dir = '/Users/similovesyou/Desktop/qts/simian-behavior/plots/brain-behavior'
    
    print("Creating 6-month timeline...")
    timeline_6m = create_subject_timeline_plot(
        data, SCAN_DATES, 
        window_months=6, 
        min_attempts=CHOSEN_MIN_ATTEMPTS,
        save_path=os.path.join(save_dir, 'timeline_6month.svg')
    )
    
    print("\nCreating 12-month timeline...")
    timeline_12m = create_subject_timeline_plot(
        data, SCAN_DATES, 
        window_months=12, 
        min_attempts=CHOSEN_MIN_ATTEMPTS,
        save_path=os.path.join(save_dir, 'timeline_12month.svg')
    )
    
    print("\nCreating 24-month timeline...")
    timeline_24m = create_subject_timeline_plot(
        data, SCAN_DATES, 
        window_months=24, 
        min_attempts=CHOSEN_MIN_ATTEMPTS
    )
    
    print("\nAll timeline visualizations complete!")