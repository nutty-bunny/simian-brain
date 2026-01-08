import os
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
from scipy.optimize import curve_fit
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from scipy import stats
from scipy.stats import f

warnings.filterwarnings('ignore')
plt.rcParams["svg.fonttype"] = "none"

directory = "/Users/similovesyou/Desktop/qts/simian-behavior/data/py"
figure_me_out = "/Users/similovesyou/Desktop/qts/simian-brain/plots"
Path(figure_me_out).mkdir(parents=True, exist_ok=True)

rhesus_table = pd.read_csv("/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/derivatives/rhesus-elo-min-attempts.csv")
tonkean_table = pd.read_csv("/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/derivatives/tonkean-elo-min-attempts.csv")

fc_dir_attention = "/Users/similovesyou/Desktop/qts/simian-brain/final-functional-connectivity-2mm/site-strasbourg/attention-seed-2-seed-connectivity"
fc_dir_impulsivity = "/Users/similovesyou/Desktop/qts/simian-brain/final-functional-connectivity-2mm/site-strasbourg/impulsivity-seed-2-seed-connectivity"

def get_roi_pairs(fc_directory, subject_list):
    rois = []
    for name in subject_list:
        conn_path = os.path.join(fc_directory, f"{name}_seed_connectivity.csv")
        if os.path.exists(conn_path):
            mat = pd.read_csv(conn_path, index_col=0)
            rois = list(mat.index)
            break
    if not rois:
        print(f"No FC matrices found in {fc_directory}")
        return []
    pairs = [(a, b) for i, a in enumerate(rois) for j, b in enumerate(rois) if i < j]
    return pairs

attention_pairs = get_roi_pairs(fc_dir_attention, rhesus_table["name"])
impulsivity_pairs = get_roi_pairs(fc_dir_impulsivity, rhesus_table["name"])

for a, b in attention_pairs:
    col = f"fc_{a}_{b}"
    rhesus_table[col] = pd.NA
    tonkean_table[col] = pd.NA

for a, b in impulsivity_pairs:
    col = f"fc_{a}_{b}"
    rhesus_table[col] = pd.NA
    tonkean_table[col] = pd.NA

def fill_fc(df, fc_directory, roi_pairs):
    for i, row in df.iterrows():
        name = row["name"]
        conn_path = os.path.join(fc_directory, f"{name}_seed_connectivity.csv")
        if not os.path.isfile(conn_path):
            print(f"-- Skipping {name}: no file in {fc_directory}")
            continue
        try:
            mat = pd.read_csv(conn_path, index_col=0)
            for a, b in roi_pairs:
                if a in mat.index and b in mat.columns:
                    df.at[i, f"fc_{a}_{b}"] = mat.at[a, b]
        except Exception as e:
            print(f"!! Error processing {name} in {fc_directory}: {e}")

fill_fc(rhesus_table, fc_dir_attention, attention_pairs)
fill_fc(tonkean_table, fc_dir_attention, attention_pairs)
fill_fc(rhesus_table, fc_dir_impulsivity, impulsivity_pairs)
fill_fc(tonkean_table, fc_dir_impulsivity, impulsivity_pairs)

scan_dates = {
    "amidala": "2024-10-17",
    "arwen": "2024-03-08",
    "baal": "2024-05-24",
    "berenice": "2025-02-14",
    "dory": "2024-09-27",
    "ficelle": "2025-01-17",
    "gabie": "2024-09-05",
    "gandhi": "2025-02-28",
    "horus": "2025-02-07",
    "indigo": "2024-03-15",
    "iron": "2024-10-31",
    "isis": "2025-03-14",
    "jazz": "2025-01-31",
    "jipsy": "2024-10-18",
    "joy": "2025-03-13",
    "karma": "2024-11-01",
    "kenobi": "2025-02-21",
    "kenya": "2024-11-29",
    "marouchka": "2024-10-25",
    "natasha": "2024-04-19",
    "nema": "2025-01-24",
    "radja": "2024-10-24",
    "samael": "2024-09-26",
    "volga": "2024-05-17",
    "yannick": "2025-01-30",
    "yin": "2024-11-08",
    "patsy": "2024-11-22"
}

rhesus_table["scan_date"] = pd.NaT
rhesus_table["age_at_scan"] = pd.NA
rhesus_table["start_date"] = pd.to_datetime(rhesus_table["start_date"])

tonkean_table["scan_date"] = pd.NaT
tonkean_table["age_at_scan"] = pd.NA
tonkean_table["start_date"] = pd.to_datetime(tonkean_table["start_date"])

for i, row in rhesus_table.iterrows():
    name = row["name"]
    if name in scan_dates:
        scan_dt = pd.to_datetime(scan_dates[name])
        rhesus_table.at[i, "scan_date"] = scan_dt
        
        if pd.notna(row["age"]) and pd.notna(row["start_date"]):
            days_diff = (scan_dt - row["start_date"]).days
            years_added = days_diff / 365.25
            rhesus_table.at[i, "age_at_scan"] = round(row["age"] + years_added, 2)

for i, row in tonkean_table.iterrows():
    name = row["name"]
    if name in scan_dates:
        scan_dt = pd.to_datetime(scan_dates[name])
        tonkean_table.at[i, "scan_date"] = scan_dt
        
        if pd.notna(row["age"]) and pd.notna(row["start_date"]):
            days_diff = (scan_dt - row["start_date"]).days
            years_added = days_diff / 365.25
            tonkean_table.at[i, "age_at_scan"] = round(row["age"] + years_added, 2)

rhesus_table["start_date"] = pd.to_datetime(rhesus_table["start_date"])
rhesus_table["end_date"] = pd.to_datetime(rhesus_table["end_date"])
tonkean_table["start_date"] = pd.to_datetime(tonkean_table["start_date"])
tonkean_table["end_date"] = pd.to_datetime(tonkean_table["end_date"])

rhesus_table["task_midpoint"] = rhesus_table["start_date"] + (rhesus_table["end_date"] - rhesus_table["start_date"]) / 2
tonkean_table["task_midpoint"] = tonkean_table["start_date"] + (tonkean_table["end_date"] - tonkean_table["start_date"]) / 2

rhesus_table["age_at_task"] = rhesus_table.apply(
    lambda row: round(row["age"] + (row["task_midpoint"] - row["start_date"]).days / 365.25, 2)
    if pd.notna(row["age"]) and pd.notna(row["task_midpoint"]) else pd.NA,
    axis=1
)
tonkean_table["age_at_task"] = tonkean_table.apply(
    lambda row: round(row["age"] + (row["task_midpoint"] - row["start_date"]).days / 365.25, 2)
    if pd.notna(row["age"]) and pd.notna(row["task_midpoint"]) else pd.NA,
    axis=1
)

summary_cols = ["name", "age", "age_at_task", "age_at_scan"]
summary_rhesus_table = rhesus_table[summary_cols].copy()
summary_tonkean_table = tonkean_table[summary_cols].copy()

summary_rhesus_table.sort_values(by="name", inplace=True)
summary_tonkean_table.sort_values(by="name", inplace=True)

print(summary_rhesus_table.to_string(index=False))
print(summary_tonkean_table.to_string(index=False))

for table in [rhesus_table, tonkean_table]:
    table['age_at_scan'] = pd.to_numeric(table['age_at_scan'], errors='coerce')
    
    fc_cols = [col for col in table.columns if col.startswith('fc_')]
    for col in fc_cols:
        table[col] = pd.to_numeric(table[col], errors='coerce')

combined_df = pd.concat(
    [
        rhesus_table.assign(species="Rhesus"),
        tonkean_table.assign(species="Tonkean"),
    ]
).drop_duplicates(subset="name", ignore_index=True)

def summarize_demographics(df, age_col, species_col="species", gender_col="gender"):
    summary = []
    for species in df[species_col].unique():
        subset = df[df[species_col] == species]
        females = subset[subset[gender_col] == 2]
        males = subset[subset[gender_col] != 2]

        age_vals = subset[age_col].dropna()
        if not age_vals.empty:
            age_mean = round(age_vals.mean(), 2)
            age_min = round(age_vals.min(), 2)
            age_max = round(age_vals.max(), 2)
        else:
            age_mean, age_min, age_max = [None] * 3

        female_pct = round(len(females) / len(subset) * 100, 1) if len(subset) > 0 else 0

        summary.append({
            "Species": species,
            "N": len(subset),
            "Mean Age": age_mean,
            "Min Age": age_min,
            "Max Age": age_max,
            "Female (%)": female_pct,
            "Male (%)": 100 - female_pct,
        })

    return pd.DataFrame(summary)

behavior_summary = summarize_demographics(combined_df, "age_at_task")
print("\n--- Behavioral Cohort Summary ---")
print(behavior_summary.to_string(index=False))

neuro_df = combined_df[combined_df["age_at_scan"].notna()]
neuro_summary = summarize_demographics(neuro_df, "age_at_scan")
print("\n--- Neuroimaging Cohort Summary ---")
print(neuro_summary.to_string(index=False))

demographics_path = "/Users/similovesyou/Desktop/qts/simian-brain/data/demographics/IRM-demographics.xlsx"
birthdates_path = "/Users/similovesyou/Desktop/qts/simian-brain/data/demographics/IRM-birthdates.xlsx"

IRM_demographics_df = pd.read_excel(demographics_path)
IRM_birthdates_df = pd.read_excel(birthdates_path)

behavioral_subjects = set(rhesus_table['name']).union(set(tonkean_table['name']))
non_behavioral_subjects = [s for s in scan_dates.keys() if s not in behavioral_subjects]

non_behavioral_df = pd.DataFrame({
    'name': non_behavioral_subjects,
    'scan_date': pd.to_datetime([scan_dates[name] for name in non_behavioral_subjects])
})

IRM_demographics_df['name_lower'] = IRM_demographics_df['name'].str.lower()
IRM_birthdates_df['name_lower'] = IRM_birthdates_df['Individual'].str.lower()
non_behavioral_df['name_lower'] = non_behavioral_df['name'].str.lower()

non_behavioral_df = non_behavioral_df.merge(
    IRM_demographics_df[['name_lower', 'sex']],
    on='name_lower',
    how='left'
)

non_behavioral_df = non_behavioral_df.merge(
    IRM_birthdates_df[['name_lower', 'Date of birth']],
    on='name_lower',
    how='left'
)

non_behavioral_df.rename(columns={
    "sex": "gender", 
    "Date of birth": "birth_date"
}, inplace=True)

non_behavioral_df['birth_date'] = pd.to_datetime(non_behavioral_df['birth_date'])
non_behavioral_df['scan_date'] = pd.to_datetime(non_behavioral_df['scan_date'])

non_behavioral_df = non_behavioral_df.merge(
    IRM_demographics_df[['name_lower', 'group']],
    on='name_lower',
    how='left'
)

non_behavioral_df.rename(columns={'group': 'species'}, inplace=True)

def calculate_decimal_age(birth_date, scan_date):
    if pd.isna(birth_date) or pd.isna(scan_date):
        return None
    
    days_diff = (scan_date - birth_date).days
    age_years = days_diff / 365.25
    
    return round(age_years, 2)

non_behavioral_df['age_at_scan'] = non_behavioral_df.apply(
    lambda row: calculate_decimal_age(row['birth_date'], row['scan_date']), 
    axis=1
)

non_behavioral_df = non_behavioral_df.drop(columns=['name_lower', 'birth_date'])

gender_map = {"M": 1, "F": 2}
non_behavioral_df["gender"] = non_behavioral_df["gender"].map(gender_map)

roi_pairs_attention = attention_pairs
roi_pairs_motor = impulsivity_pairs

for a, b in roi_pairs_attention:
    non_behavioral_df[f"fc_{a}_{b}"] = pd.NA

for a, b in roi_pairs_motor:
    non_behavioral_df[f"fc_{a}_{b}"] = pd.NA

for i, row in non_behavioral_df.iterrows():
    name = row["name"]
    
    conn_path_attention = os.path.join(fc_dir_attention, f"{name}_seed_connectivity.csv")
    if os.path.isfile(conn_path_attention):
        try:
            mat = pd.read_csv(conn_path_attention, index_col=0)
            for a, b in roi_pairs_attention:
                if a in mat.index and b in mat.columns:
                    non_behavioral_df.at[i, f"fc_{a}_{b}"] = mat.at[a, b]
        except Exception as e:
            print(f"Error processing {name} attention: {e}")
    
    conn_path_motor = os.path.join(fc_dir_impulsivity, f"{name}_seed_connectivity.csv")
    if os.path.isfile(conn_path_motor):
        try:
            mat = pd.read_csv(conn_path_motor, index_col=0)
            for a, b in roi_pairs_motor:
                if a in mat.index and b in mat.columns:
                    non_behavioral_df.at[i, f"fc_{a}_{b}"] = mat.at[a, b]
        except Exception as e:
            print(f"Error processing {name} motor: {e}")

print("\nSuccessfully processed subjects:")
print(non_behavioral_df[['name', 'gender', 'age_at_scan']].dropna())

def find_fc_column(a, b, table):
    direct = f"fc_{a}_{b}"
    flipped = f"fc_{b}_{a}"
    if direct in table.columns:
        return direct
    elif flipped in table.columns:
        return flipped
    else:
        raise ValueError(f"No FC column found for pair {a}–{b}")

def calculate_model_significance(y_true, y_pred, n_params, n_obs):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    df_model = n_params - 1
    df_error = n_obs - n_params
    
    if df_error <= 0:
        return None
    
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    ss_explained = ss_total - ss_residual
    
    if ss_residual == 0:
        f_stat = np.inf
        p_value_f = 0.0
    else:
        f_stat = (ss_explained / df_model) / (ss_residual / df_error)
        p_value_f = 1 - f.cdf(f_stat, df_model, df_error)
    
    aic = n_obs * np.log(mse) + 2 * n_params
    bic = n_obs * np.log(mse) + np.log(n_obs) * n_params
    
    return {
        'r2': r2,
        'mse': mse,
        'aic': aic,
        'bic': bic,
        'f_stat': f_stat,
        'p_value': p_value_f,
        'df_model': df_model,
        'df_error': df_error,
        'significant': p_value_f < 0.05
    }

def likelihood_ratio_test(y_true, y_pred_simple, y_pred_complex, n_params_simple, n_params_complex, n_obs):
    mse_simple = mean_squared_error(y_true, y_pred_simple)
    mse_complex = mean_squared_error(y_true, y_pred_complex)
    
    if mse_simple <= 0 or mse_complex <= 0:
        return np.nan
    
    log_likelihood_simple = -n_obs/2 * (1 + np.log(2*np.pi) + np.log(mse_simple))
    log_likelihood_complex = -n_obs/2 * (1 + np.log(2*np.pi) + np.log(mse_complex))
    
    lr_stat = 2 * (log_likelihood_complex - log_likelihood_simple)
    df_diff = n_params_complex - n_params_simple
    
    if df_diff <= 0:
        return np.nan
    
    p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)
    
    return p_value

def calculate_peak_age(model_results, age_range=(0, 25)):
    """
    Calculate peak age for quadratic models, with strict validation.
    Only returns peak if:
    1. Quadratic model is inverted-U (negative coefficient)
    2. Peak falls within biologically plausible age range
    3. Quadratic is meaningfully better than linear (ΔAIC < -2)
    
    Args:
        model_results: Results from linear_vs_quadratic_testing
        age_range: Tuple of (min_age, max_age) for valid peaks
    """
    if model_results is None:
        return None
    
    # Only report peaks for quadratic models that are CLEARLY better by AIC
    best_model = model_results.get('best_aic')
    delta_aic = model_results.get('delta_aic', np.nan)
    
    # Strict criterion: quadratic must be better AND ΔAIC < -2 (moderate evidence)
    if best_model != 'quadratic':
        return None
    
    if np.isnan(delta_aic) or delta_aic > -2:
        return None
    
    quad_results = model_results.get('quadratic')
    if quad_results is None:
        return None
    
    params = quad_results.get('params')
    if params is None or len(params) < 3:
        return None
    
    # params = [intercept, linear_coef, quadratic_coef]
    intercept = params[0]
    linear_coef = params[1]
    quadratic_coef = params[2]
    
    # Only inverted-U shapes (peaks, not troughs)
    if quadratic_coef >= 0:
        return None
    
    # Calculate peak: x = -b/(2a)
    peak_age = -linear_coef / (2 * quadratic_coef)
    
    # Validate peak is within biologically plausible range
    min_age, max_age = age_range
    if peak_age < min_age or peak_age > max_age:
        return None
    
    return round(peak_age, 2)

def linear_vs_quadratic_testing(df, x_col, y_col, group_name=""):
    valid_data = df[[x_col, y_col]].dropna()
    if len(valid_data) < 5:
        return None
    
    X = pd.to_numeric(valid_data[x_col], errors='coerce').values
    y = pd.to_numeric(valid_data[y_col], errors='coerce').values

    mask = np.isfinite(X) & np.isfinite(y)
    X = X[mask]
    y = y[mask]
    
    if len(X) < 5:
        return None
    
    n = len(y)
    results = {}
    
    X_reshaped = X.reshape(-1, 1)
    linear_model = LinearRegression()
    linear_model.fit(X_reshaped, y)
    y_pred_linear = linear_model.predict(X_reshaped)
    
    linear_sig = calculate_model_significance(y, y_pred_linear, 2, n)
    if linear_sig:
        linear_sig.update({
            'params': [linear_model.intercept_, linear_model.coef_[0]],
            'model_type': 'linear',
            'equation': f"y = {linear_model.coef_[0]:.4f}x + {linear_model.intercept_:.4f}",
            'vs_linear_p': np.nan
        })
        results['linear'] = linear_sig
    
    try:
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly_features.fit_transform(X_reshaped)
        quad_model = LinearRegression()
        quad_model.fit(X_poly, y)
        y_pred_quad = quad_model.predict(X_poly)
        
        quad_sig = calculate_model_significance(y, y_pred_quad, 3, n)
        if quad_sig:
            vs_linear_p = likelihood_ratio_test(y, y_pred_linear, y_pred_quad, 2, 3, n)
            
            quad_sig.update({
                'params': [quad_model.intercept_] + list(quad_model.coef_),
                'model_type': 'quadratic',
                'equation': f"y = {quad_model.coef_[1]:.6f}x² + {quad_model.coef_[0]:.4f}x + {quad_model.intercept_:.4f}",
                'vs_linear_p': vs_linear_p,
                'better_than_linear': vs_linear_p < 0.05 if not np.isnan(vs_linear_p) else False
            })
            results['quadratic'] = quad_sig
    except Exception as e:
        results['quadratic'] = None

    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if valid_results:
        best_aic = min(valid_results.keys(), key=lambda k: valid_results[k]['aic'])
        best_r2 = max(valid_results.keys(), key=lambda k: valid_results[k]['r2'])
        significant_models = {k: v for k, v in valid_results.items() if v.get('significant', False)}
        best_significant_r2 = max(significant_models.keys(), key=lambda k: significant_models[k]['r2']) if significant_models else None
        
        results['best_aic'] = best_aic
        results['best_r2'] = best_r2
        results['best_significant_r2'] = best_significant_r2
        results['n'] = n
        results['group'] = group_name
        results['n_significant'] = len(significant_models)
        
        # Calculate delta AIC if both models exist
        if 'linear' in valid_results and 'quadratic' in valid_results:
            results['delta_aic'] = valid_results['quadratic']['aic'] - valid_results['linear']['aic']
        else:
            results['delta_aic'] = np.nan
    
    return results

def print_linear_vs_quadratic_results(results_dict, metric_name):
    """
    Enhanced reporting with AIC as primary model selection criterion.
    
    Interpretation guide:
    - ΔAIC < 2: Models essentially equivalent
    - ΔAIC 2-4: Moderate evidence for better model  
    - ΔAIC > 4: Strong evidence for better model
    - ΔAIC > 10: Very strong evidence
    """
    print(f"\n{'='*80}")
    print(f"{metric_name}")
    print('='*80)
    
    for group_name, results in results_dict.items():
        if results is None:
            print(f"{group_name}: Insufficient data")
            continue
        
        print(f"\n{group_name} (n={results['n']})")
        print("-" * 60)
        
        for model_name in ['linear', 'quadratic']:
            if model_name in results and results[model_name] is not None:
                model_info = results[model_name]
                p_val = model_info.get('p_value', np.nan)
                aic = model_info.get('aic', np.nan)
                
                if p_val < 0.001:
                    p_str = "p<0.001***"
                elif p_val < 0.01:
                    p_str = f"p={p_val:.3f}**"
                elif p_val < 0.05:
                    p_str = f"p={p_val:.3f}*"
                else:
                    p_str = f"p={p_val:.3f}"
                
                print(f"  {model_name.upper():10} | R²={model_info['r2']:.3f} | {p_str:12} | AIC={aic:.2f}")
                
                if model_name == 'quadratic':
                    vs_linear_p = model_info.get('vs_linear_p', np.nan)
                    if not np.isnan(vs_linear_p):
                        if vs_linear_p < 0.05:
                            print(f"             └─ Quadratic vs. Linear: p={vs_linear_p:.3f}*")
                        else:
                            print(f"             └─ Quadratic vs. Linear: p={vs_linear_p:.3f} (n.s.)")
        
        # Report best model by AIC with interpretation
        if 'best_aic' in results:
            best_model = results['best_aic']
            delta_aic = results.get('delta_aic', np.nan)
            
            print(f"\n  {'─'*56}")
            print(f"  MODEL SELECTION (AIC-based):")
            print(f"  Best model: {best_model.upper()}")
            
            if not np.isnan(delta_aic):
                # Note: positive ΔAIC means quadratic has HIGHER (worse) AIC
                abs_delta = abs(delta_aic)
                
                if abs_delta < 2:
                    evidence = "negligible"
                    interpretation = "Models essentially equivalent"
                elif abs_delta < 4:
                    evidence = "moderate"
                    interpretation = f"Moderate support for {best_model}"
                elif abs_delta < 10:
                    evidence = "strong"
                    interpretation = f"Strong support for {best_model}"
                else:
                    evidence = "very strong"
                    interpretation = f"Very strong support for {best_model}"
                
                print(f"  ΔAIC = {delta_aic:.2f} (quadratic - linear)")
                print(f"  Evidence: {evidence.upper()} ({interpretation})")
                
                # Add note about significance vs AIC
                linear_sig = results.get('linear', {}).get('significant', False)
                quad_sig = results.get('quadratic', {}).get('significant', False)
                
                if best_model == 'quadratic' and not quad_sig:
                    print(f"  ⚠️  Note: Quadratic preferred by AIC but p>{0.05}")
                elif best_model == 'linear' and quad_sig:
                    print(f"  ⚠️  Note: Linear preferred by AIC despite quadratic p<0.05")


def p_lifespan_all(
    ax, metric, title, rhesus_table, tonkean_table, non_behavioral_df, colors, show_xlabel, show_ylabel
):
    sns.scatterplot(
        ax=ax,
        data=rhesus_table[rhesus_table["gender"] == 1],
        x="age_at_scan",
        y=metric,
        facecolors="none",
        edgecolor="steelblue",
        s=80,
        linewidth=1,
    )
    sns.scatterplot(
        ax=ax,
        data=rhesus_table[rhesus_table["gender"] == 2],
        x="age_at_scan",
        y=metric,
        color=colors["rhesus"],
        s=80,
        alpha=0.8,
        edgecolor="steelblue",
        linewidth=1,
    )

    sns.scatterplot(
        ax=ax,
        data=tonkean_table[tonkean_table["gender"] == 1],
        x="age_at_scan",
        y=metric,
        facecolors="none",
        edgecolor="purple",
        s=80,
        linewidth=1,
    )
    sns.scatterplot(
        ax=ax,
        data=tonkean_table[tonkean_table["gender"] == 2],
        x="age_at_scan",
        y=metric,
        color=colors["tonkean"],
        s=80,
        alpha=0.8,
        edgecolor="purple",
        linewidth=1,
    )

    if metric in non_behavioral_df.columns:
        valid_df = non_behavioral_df.dropna(subset=["age_at_scan", metric])
        rhesus_males = valid_df[(valid_df["gender"] == 1) & (valid_df["species"] == "rhesus")]
        rhesus_females = valid_df[(valid_df["gender"] == 2) & (valid_df["species"] == "rhesus")]
        tonkean_males = valid_df[(valid_df["gender"] == 1) & (valid_df["species"] == "tonkean")]
        tonkean_females = valid_df[(valid_df["gender"] == 2) & (valid_df["species"] == "tonkean")]

        sns.scatterplot(
            ax=ax,
            data=rhesus_males,
            x="age_at_scan",
            y=metric,
            facecolors="none",
            edgecolor=colors["rhesus"],
            s=80,
            linewidth=1,
            alpha=0.3,
            legend=False,
        )
        sns.scatterplot(
            ax=ax,
            data=rhesus_females,
            x="age_at_scan",
            y=metric,
            color=colors["rhesus"],
            s=80,
            alpha=0.3,
            edgecolor=colors["rhesus"],
            linewidth=1,
            legend=False,
        )
        sns.scatterplot(
            ax=ax,
            data=tonkean_males,
            x="age_at_scan",
            y=metric,
            facecolors="none",
            edgecolor=colors["tonkean"],
            s=80,
            linewidth=1,
            alpha=0.3,
            legend=False,
        )
        sns.scatterplot(
            ax=ax,
            data=tonkean_females,
            x="age_at_scan",
            y=metric,
            color=colors["tonkean"],
            s=80,
            alpha=0.3,
            edgecolor=colors["tonkean"],
            linewidth=1,
            legend=False,
        )

    all_subjects = pd.concat([rhesus_table, tonkean_table, non_behavioral_df], ignore_index=True)

    if metric not in all_subjects.columns:
        all_subjects[metric] = np.nan

    all_subjects["age_at_scan"] = pd.to_numeric(all_subjects["age_at_scan"], errors="coerce")
    all_subjects[metric] = pd.to_numeric(all_subjects[metric], errors="coerce")

    all_subjects = all_subjects.dropna(subset=["age_at_scan", metric])

    ax.set_title(title, fontsize=22, fontweight="bold", fontname="DejaVu Sans", color="black")
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, 25)
    ax.set_xticks(range(0, 26, 5))
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(["-1", "", "0", "", "1"])
    ax.grid(True, which="both", linestyle="--")
    ax.grid(True, which="major", axis="x")

    ax.spines["bottom"].set_linewidth(1)
    ax.spines["left"].set_linewidth(1)
    ax.tick_params(axis="both", which="major", width=2.5, length=7, labelsize=18)

    if show_xlabel:
        ax.set_xlabel("Age", fontsize=18)
    else:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel("Functional Connectivity (r)", fontsize=18)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")

sns.set(style="whitegrid", font_scale=1.5, rc={"font.family": "DejaVu Sans"})

fig2, axes2 = plt.subplots(2, 2, figsize=(13, 14))
colors2 = {"tonkean": "purple", "rhesus": "steelblue"}

metrics2 = [
    find_fc_column("LIP", "FEF", rhesus_table),
    find_fc_column("V1", "MT", rhesus_table),
    find_fc_column("OFC", "nucleus_accumbens", rhesus_table),
    find_fc_column("thalamus", "nucleus_accumbens", rhesus_table)
]

titles2 = ["LIP-FEF", "V1-MT", "OFC-NAcc", "Thalamus-NAcc"]

for m in metrics2:
    if m not in non_behavioral_df.columns:
        non_behavioral_df[m] = np.nan

axes_flat = axes2.flatten()
show_xlabel_flags = [False, False, True, True]
show_ylabel_flags = [True, False, True, False]

for i, (ax, metric, title, show_xlabel, show_ylabel) in enumerate(zip(
    axes_flat, metrics2, titles2, show_xlabel_flags, show_ylabel_flags
)):
    p_lifespan_all(
        ax, metric, title, rhesus_table, tonkean_table, non_behavioral_df,
        colors2, show_xlabel, show_ylabel
    )
    
    ax.tick_params(labelsize=14)
    
    all_data_frames = [rhesus_table, tonkean_table]
    if (metric in non_behavioral_df.columns and 'age_at_scan' in non_behavioral_df.columns):
        non_behavioral_subset = non_behavioral_df[['age_at_scan', metric]].dropna()
        if not non_behavioral_subset.empty:
            all_data_frames.append(non_behavioral_subset)
    
    all_data = pd.concat(all_data_frames, ignore_index=True)
    
    # Run statistical test to get actual values
    test_results = linear_vs_quadratic_testing(all_data, 'age_at_scan', metric, 'All Subjects')
    
    # Use AIC as primary criterion for model selection
    if test_results is not None:
        # Primary criterion: AIC
        best_model = test_results.get('best_aic', 'linear')
        delta_aic = test_results.get('delta_aic', np.nan)
        
        model_info = test_results.get(best_model)
        
        if model_info:
            fit_type = best_model
            r2_val = model_info['r2']
            p_val = model_info['p_value']
            
            # Determine if we should plot the line based on AIC difference
            # Plot if: ΔAIC > 2 (moderate evidence) OR model is significant
            plot_line = False
            if not np.isnan(delta_aic):
                if abs(delta_aic) > 2:  # Moderate or stronger evidence
                    plot_line = True
            if model_info['significant']:  # Or if model itself is significant
                plot_line = True
            
            # Build statistics text for plot
            if fit_type == "quadratic":
                quad_coef = model_info['params'][2]
                direction = "inverted U" if quad_coef < 0 else "U-shaped"
                
                if not np.isnan(delta_aic):
                    stats_text = f"R² = {r2_val:.3f}, {direction}\np = {p_val:.4f}, ΔAIC = {delta_aic:.1f}"
                else:
                    stats_text = f"R² = {r2_val:.3f}, {direction}\np = {p_val:.4f}"
                    
                # Add significance markers
                if p_val < 0.001:
                    stats_text = stats_text.replace(f"p = {p_val:.4f}", "p < 0.001***")
                elif p_val < 0.01:
                    stats_text += "**"
                elif p_val < 0.05:
                    stats_text += "*"
                    
            else:  # linear
                slope = model_info['params'][1]
                r_val = np.sqrt(r2_val) * (1 if slope > 0 else -1)
                
                if p_val < 0.001:
                    p_text = "p < 0.001***"
                elif p_val < 0.01:
                    p_text = f"p = {p_val:.4f}**"
                elif p_val < 0.05:
                    p_text = f"p = {p_val:.4f}*"
                else:
                    p_text = f"p = {p_val:.4f}"
                    
                if not np.isnan(delta_aic):
                    stats_text = f"r = {r_val:.3f}, R² = {r2_val:.3f}\n{p_text}, ΔAIC = {delta_aic:.1f}"
                else:
                    stats_text = f"r = {r_val:.3f}, R² = {r2_val:.3f}\n{p_text}"
            
            # Display statistics
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9,
                             edgecolor='lightgray', linewidth=0.5))
            
            # Plot regression line based on criteria
            if plot_line:
                clean_data = all_data[[metric, 'age_at_scan']].dropna().copy()
                clean_data['age_at_scan'] = pd.to_numeric(clean_data['age_at_scan'], errors='coerce')
                clean_data[metric] = pd.to_numeric(clean_data[metric], errors='coerce')
                clean_data = clean_data.dropna()
                
                if len(clean_data) > 3:
                    # Use dashed line if model not significant but preferred by AIC
                    if not model_info['significant'] and abs(delta_aic) > 2:
                        line_kws = {"linewidth": 1.5, "alpha": 0.8, "linestyle": "--", "color": "grey"}
                    else:
                        line_kws = {"linewidth": 1.5, "alpha": 0.8, "linestyle": "-", "color": "black"}
                    
                    if fit_type == "quadratic":
                        sns.regplot(ax=ax, data=clean_data, x="age_at_scan", y=metric,
                                   order=2, scatter=False, ci=None, line_kws=line_kws)
                    else:  # linear
                        sns.regplot(ax=ax, data=clean_data, x="age_at_scan", y=metric,
                                   order=1, scatter=False, ci=None, line_kws=line_kws)
    
    ax.set_title(title, color='black', fontsize=24, fontweight='bold')

for i, ax in enumerate(axes_flat):
    if i < 2:  # Top row
        ax.set_xlabel('')
        ax.tick_params(labelbottom=False)
    else:  # Bottom row
        ax.set_xlabel('Age', fontsize=20)
    
    if i == 0 or i == 2:  # Left column
        ax.set_ylabel('rs-fc (r)', fontsize=20)
    else:  # Right column
        ax.set_ylabel('')

handles2 = [
    plt.Line2D([], [], color=colors2["rhesus"], marker="o", linestyle="None", markersize=10, label=r"$\it{M.\ mulatta}$ (f, behavioral)"),
    plt.Line2D([], [], color="none", marker="o", markeredgecolor="steelblue", linestyle="None", markersize=10, label=r"$\it{M.\ mulatta}$ (m, behavioral)"),
    plt.Line2D([], [], color=colors2["tonkean"], marker="o", linestyle="None", markersize=10, label=r"$\it{M.\ tonkeana}$ (f, behavioral)"),
    plt.Line2D([], [], color="none", marker="o", markeredgecolor="purple", linestyle="None", markersize=10, label=r"$\it{M.\ tonkeana}$ (m, behavioral)"),
    plt.Line2D([], [], color=colors2["rhesus"], marker="o", linestyle="None", markersize=10, alpha=0.3, label=r"$\it{M.\ mulatta}$ (f, imaging only)"),
    plt.Line2D([], [], color="none", marker="o", markeredgecolor="steelblue", linestyle="None", markersize=10, alpha=0.3, label=r"$\it{M.\ mulatta}$ (m, imaging only)"),
    plt.Line2D([], [], color=colors2["tonkean"], marker="o", linestyle="None", markersize=10, alpha=0.3, label=r"$\it{M.\ tonkeana}$ (f, imaging only)"),
    plt.Line2D([], [], color="none", marker="o", markeredgecolor="purple", linestyle="None", markersize=10, alpha=0.3, label=r"$\it{M.\ tonkeana}$ (m, imaging only)"),
]

legend = fig2.legend(handles=handles2, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=14)
fig2.suptitle("Resting State FC Across the Lifespan in Macaque Species", fontsize=28, fontweight="bold", color="black")
plt.tight_layout(rect=[0, 0.02, 1, 0.93])

filename_base = "final-functional-connectivity-AIC"
save_path_svg = os.path.join(figure_me_out, f"{filename_base}.svg")
save_path_png = os.path.join(figure_me_out, f"{filename_base}.png")
plt.savefig(save_path_svg, format="svg")
plt.savefig(save_path_png, format="png", dpi=300)
plt.show()

def describe_neuroimaging_sample(
    df, age_col="age_at_scan", gender_col="gender", species_col="species", by_gender=True
):
    df = df[df[age_col].notna()].copy()
    summary = []

    if by_gender:
        groupings = [
            (species, gender_code, gender_label)
            for species in sorted(df[species_col].dropna().unique())
            for gender_code, gender_label in [(1, "Male"), (2, "Female")]
        ]
    else:
        groupings = [(species, None, None) for species in sorted(df[species_col].dropna().unique())]

    for species, gender_code, gender_label in groupings:
        if by_gender:
            subset = df[(df[species_col] == species) & (df[gender_col] == gender_code)]
        else:
            subset = df[df[species_col] == species]

        count = len(subset)
        ages = subset[age_col].dropna()

        if not ages.empty:
            mean_age = round(ages.mean(), 2)
            std_age = round(ages.std(), 2)
            min_age = round(ages.min(), 2)
            max_age = round(ages.max(), 2)
        else:
            mean_age = std_age = min_age = max_age = None

        summary.append({
            "Species": species,
            "Gender": gender_label if by_gender else "All",
            "N": count,
            "Mean Age": mean_age,
            "SD": std_age,
            "Min Age": min_age,
            "Max Age": max_age,
        })

    return pd.DataFrame(summary)

if "species" not in non_behavioral_df.columns:
    if "group" in IRM_demographics_df.columns:
        group_map = dict(zip(IRM_demographics_df['name_lower'], IRM_demographics_df['group']))
        non_behavioral_df['species'] = non_behavioral_df['name'].str.lower().map(group_map)
    else:
        non_behavioral_df['species'] = pd.NA

combined_all_df = pd.concat([combined_df, non_behavioral_df], ignore_index=True)

def clean_species_column(df):
    df["species"] = df["species"].str.strip().str.capitalize()
    return df

combined_df = clean_species_column(combined_df)
non_behavioral_df = clean_species_column(non_behavioral_df)

combined_all_df = pd.concat([combined_df, non_behavioral_df], ignore_index=True)

neuro_summary_detailed = describe_neuroimaging_sample(combined_all_df, by_gender=True)
neuro_summary_species = describe_neuroimaging_sample(combined_all_df, by_gender=False)

print("\n--- Neuroimaging Cohort (Non-Behaviorals Incl.) Detailed Summary ---")
print(neuro_summary_detailed.to_string(index=False))

print("\n--- Neuroimaging Cohort (Species-Level Only) ---")
print(neuro_summary_species.to_string(index=False))

non_behavioral_df = non_behavioral_df[non_behavioral_df["age_at_scan"].notna()]
non_behavioral_df["age_at_scan"] = pd.to_numeric(non_behavioral_df["age_at_scan"], errors='coerce')

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [2, 2]}
    )

    jitter = 0.15
    colors = {"Rhesus": "steelblue", "Tonkean": "purple", "mulatta": "steelblue"}

    for species in ["Rhesus", "Tonkean"]:
        males = combined_df[
            (combined_df["species"] == species) & (combined_df["gender"] != 2)
        ]
        females = combined_df[
            (combined_df["species"] == species) & (combined_df["gender"] == 2)
        ]

        sns.stripplot(
            x="species",
            y="age_at_task",
            data=females,
            jitter=jitter,
            marker="o",
            color=colors[species],
            alpha=0.7,
            size=10,
            ax=ax1,
        )
        sns.stripplot(
            x="species",
            y="age_at_task",
            data=males,
            jitter=jitter,
            marker="o",
            facecolors="none",
            edgecolor=colors[species],
            size=10,
            linewidth=1.5,
            ax=ax1,
        )

    for species in ["Rhesus", "Tonkean"]:
        subset = combined_df[combined_df["species"] == species]
        mean_age = subset["age_at_task"].mean()
        sem_age = subset["age_at_task"].std() / np.sqrt(len(subset))
        ci_95 = 1.96 * sem_age
        ax1.errorbar(
            x=[species],
            y=[mean_age],
            yerr=[ci_95],
            fmt="none",
            ecolor=colors[species],
            capsize=5,
            elinewidth=2,
            zorder=10,
        )

    ax1.set_ylabel("Age", fontsize=14)
    ax1.set_xticks(ticks=[0, 1])
    ax1.set_xticklabels(
        [
            f'$\\it{{Macaca\\ mulatta}}$\n(n={combined_df[combined_df["species"] == "Rhesus"].shape[0]})',
            f'$\\it{{Macaca\\ tonkeana}}$\n(n={combined_df[combined_df["species"] == "Tonkean"].shape[0]})',
        ],
        fontsize=14,
    )

    ax1.text(
        0.5,
        1.05,
        "Behavioral Data",
        transform=ax1.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    for species in ["Rhesus", "Tonkean"]:
        males = combined_df[
            (combined_df["species"] == species) & (combined_df["gender"] != 2) & (combined_df["age_at_scan"].notna())
        ]
        females = combined_df[
            (combined_df["species"] == species) & (combined_df["gender"] == 2) & (combined_df["age_at_scan"].notna())
        ]

        sns.stripplot(
            x="species",
            y="age_at_scan",
            data=females,
            jitter=jitter,
            marker="o",
            color=colors[species],
            alpha=0.7,
            size=10,
            ax=ax2,
        )

        sns.stripplot(
            x="species",
            y="age_at_scan",
            data=males,
            jitter=jitter,
            marker="o",
            facecolors="none",
            edgecolor=colors[species],
            size=10,
            linewidth=1.5,
            alpha=0.7,
            ax=ax2,
        )

    non_behavioral_valid = non_behavioral_df[non_behavioral_df["age_at_scan"].notna()].copy()
    
    if not non_behavioral_valid.empty:
        if "species" not in non_behavioral_valid.columns or non_behavioral_valid["species"].isna().all():
            non_behavioral_valid["species"] = "Rhesus"
        
        males_non_behav = non_behavioral_valid[non_behavioral_valid["gender"] == 1]
        females_non_behav = non_behavioral_valid[non_behavioral_valid["gender"] == 2]
        
        for species in ["Rhesus", "Tonkean"]:
            females_species = females_non_behav[females_non_behav["species"] == species]
            males_species = males_non_behav[males_non_behav["species"] == species]
            
            if not females_species.empty:
                sns.stripplot(
                    x="species",
                    y="age_at_scan",
                    data=females_species,
                    jitter=jitter,
                    marker="o",
                    color=colors[species],
                    alpha=0.3,
                    size=10,
                    ax=ax2,
                )
            
            if not males_species.empty:
                sns.stripplot(
                    x="species",
                    y="age_at_scan",
                    data=males_species,
                    jitter=jitter,
                    marker="o",
                    facecolors="none",
                    edgecolor=colors[species],
                    size=10,
                    linewidth=1.5,
                    alpha=0.3,
                    ax=ax2,
                )

    for species in ["Rhesus", "Tonkean"]:
        behavioral_subset = combined_df[(combined_df["species"] == species) & (combined_df["age_at_scan"].notna())]
        
        if not behavioral_subset.empty:
            behavioral_ages = behavioral_subset["age_at_scan"].dropna().tolist()
            if behavioral_ages:
                mean_behavioral = np.mean(behavioral_ages)
                sem_behavioral = np.std(behavioral_ages, ddof=1) / np.sqrt(len(behavioral_ages)) if len(behavioral_ages) > 1 else 0
                ci_95_behavioral = 1.96 * sem_behavioral
                ax2.errorbar(
                    x=[species],
                    y=[mean_behavioral],
                    yerr=[ci_95_behavioral],
                    fmt="none",
                    ecolor=colors[species],
                    capsize=5,
                    elinewidth=2,
                    zorder=10,
                )
        
        non_behavioral_subset = non_behavioral_valid[non_behavioral_valid["species"] == species] if not non_behavioral_valid.empty else pd.DataFrame()
        
        all_ages = []
        if not behavioral_subset.empty:
            all_ages.extend(behavioral_subset["age_at_scan"].dropna().tolist())
        if not non_behavioral_subset.empty:
            all_ages.extend(non_behavioral_subset["age_at_scan"].dropna().tolist())
        
        if all_ages and len(all_ages) > len(behavioral_ages):
            mean_all = np.mean(all_ages)
            sem_all = np.std(all_ages, ddof=1) / np.sqrt(len(all_ages)) if len(all_ages) > 1 else 0
            ci_95_all = 1.96 * sem_all
            offset = 0.05
            ax2.errorbar(
                x=[0 + offset if species == "Rhesus" else 1 + offset],
                y=[mean_all],
                yerr=[ci_95_all],
                fmt="none",
                ecolor=colors[species],
                capsize=5,
                elinewidth=2,
                zorder=9,
                alpha=0.3,
            )

    ax2.set_ylabel("")
    ax2.set_yticklabels([])
    ax2.set_xticks(ticks=[0, 1])
    
    rhesus_total = len(combined_df[(combined_df["species"] == "Rhesus") & (combined_df["age_at_scan"].notna())])
    tonkean_total = len(combined_df[(combined_df["species"] == "Tonkean") & (combined_df["age_at_scan"].notna())])
    
    if not non_behavioral_valid.empty:
        rhesus_total += len(non_behavioral_valid[non_behavioral_valid["species"] == "Rhesus"])
        tonkean_total += len(non_behavioral_valid[non_behavioral_valid["species"] == "Tonkean"])
    
    ax2.set_xticklabels(
        [
            f'$\\it{{Macaca\\ mulatta}}$\n(n={rhesus_total})',
            f'$\\it{{Macaca\\ tonkeana}}$\n(n={tonkean_total})',
        ],
        fontsize=14,
    )

    ax2.text(
        0.5,
        1.05,
        "Neuroimaging Data (All)",
        transform=ax2.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.set_ylabel("")
    ax2.set_yticklabels([])
    ax2.tick_params(axis='y', labelleft=False)
    
    for ax in [ax1, ax2]:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.set_ylim(0, 25)
        ax.set_xlabel("")

    ax1.set_xlim(-0.6, 1.6)
    ax2.set_xlim(-0.6, 1.6)

    fig.suptitle(
        "Demographics (Including Non-Behavioral Subjects)",
        fontsize=25,
        fontweight="bold",
        fontname="DejaVu Sans",
        color="purple",
        y=1.02,
    )

    plt.tight_layout()

    filename_base = "demographics-neuroimaging-all-AIC"
    save_path_svg = os.path.join(figure_me_out, f"{filename_base}.svg")
    save_path_png = os.path.join(figure_me_out, f"{filename_base}.png")

    plt.savefig(save_path_svg, format="svg")
    plt.savefig(save_path_png, format="png", dpi=300)
    plt.show()

# Run tests for BOTH attention and motor impulsivity networks
print("\n" + "="*80)
print("RUNNING COMPREHENSIVE STATISTICAL TESTING WITH AIC")
print("="*80)
print("\nAIC Interpretation Guide:")
print("  ΔAIC < 2:    Models essentially equivalent")
print("  ΔAIC 2-4:    Moderate evidence for better model")
print("  ΔAIC 4-10:   Strong evidence for better model")
print("  ΔAIC > 10:   Very strong evidence for better model")
print("="*80)

# ATTENTION NETWORK CONNECTIONS
print("\nANALYZING VISUAL-ATTENTION CONNECTIVITY PATTERNS")
print()

core_attention = [
    ("LIP", "FEF"),
    ("LIP", "MT"),
    ("FEF", "V4"),
    ("LIP", "V4"),
]

visual_hierarchy = [
    ("V1", "V2"),
    ("V2", "V4"),
    ("V4", "V4t"),
    ("MT", "MST"),
    ("V4", "MT"),
]

top_down = [
    ("FEF", "V1"),
    ("FEF", "V2"),
    ("FEF", "MT"),
    ("LIP", "V1"),
    ("LIP", "MST"),
]

integration = [
    ("FST", "LIP"),
    ("FST", "FEF"),
    ("V4t", "LIP"),
]

attention_connections = core_attention + visual_hierarchy + top_down + integration

attention_metrics = []
attention_names = []

for area1, area2 in attention_connections:
    try:
        metric = find_fc_column(area1, area2, rhesus_table)
        if metric:
            attention_metrics.append(metric)
            attention_names.append(f"{area1}-{area2} rs-fc")
    except ValueError:
        print(f"Warning: Connection {area1}-{area2} not found")

for m in attention_metrics:
    if m not in non_behavioral_df.columns:
        non_behavioral_df[m] = np.nan

peak_ages_attention = {}

for metric, metric_name in zip(attention_metrics, attention_names):
    comp_results = {}
    
    all_data = pd.concat([
        rhesus_table[['age_at_scan', metric]],
        tonkean_table[['age_at_scan', metric]],
        non_behavioral_df[['age_at_scan', metric]] if metric in non_behavioral_df.columns else pd.DataFrame()
    ], ignore_index=True)
    
    comp_results['All Subjects'] = linear_vs_quadratic_testing(all_data, 'age_at_scan', metric, 'All Subjects')
    
    # Calculate peak age if quadratic model is best
    peak_age = calculate_peak_age(comp_results['All Subjects'])
    if peak_age is not None:
        peak_ages_attention[metric_name] = peak_age
    
    print_linear_vs_quadratic_results(comp_results, metric_name)

print("\n" + "="*80)
print("PEAK AGES FOR ATTENTION NETWORK CONNECTIONS (Validated)")
print("="*80)
if peak_ages_attention:
    for connection, peak in sorted(peak_ages_attention.items(), key=lambda x: x[1]):
        print(f"{connection}: Peak at {peak} years")
else:
    print("No significant inverted-U trajectories with valid peak ages detected.")
print("="*80)

# Core impulsivity circuit - test all pairwise
core_impulsivity = ["thalamus", "dorsal_striatum", "nucleus_accumbens", "OFC"]

impulsivity_connections = []

# All core circuit combinations
impulsivity_connections.extend([(a, b) for i, a in enumerate(core_impulsivity) 
                                for j, b in enumerate(core_impulsivity) if i < j])

impulsivity_metrics = []
impulsivity_names = []

for area1, area2 in impulsivity_connections:
    try:
        metric = find_fc_column(area1, area2, rhesus_table)
        if metric:
            impulsivity_metrics.append(metric)
            impulsivity_names.append(f"{area1}-{area2} rs-fc")
    except ValueError:
        print(f"Warning: Connection {area1}-{area2} not found")

for m in impulsivity_metrics:
    if m not in non_behavioral_df.columns:
        non_behavioral_df[m] = np.nan

peak_ages_impulsivity = {}

for metric, metric_name in zip(impulsivity_metrics, impulsivity_names):
    comp_results = {}
    
    all_data = pd.concat([
        rhesus_table[['age_at_scan', metric]],
        tonkean_table[['age_at_scan', metric]],
        non_behavioral_df[['age_at_scan', metric]] if metric in non_behavioral_df.columns else pd.DataFrame()
    ], ignore_index=True)
    
    comp_results['All Subjects'] = linear_vs_quadratic_testing(all_data, 'age_at_scan', metric, 'All Subjects')
    
    # Calculate peak age if quadratic model is best
    peak_age = calculate_peak_age(comp_results['All Subjects'])
    if peak_age is not None:
        peak_ages_impulsivity[metric_name] = peak_age
    
    print_linear_vs_quadratic_results(comp_results, metric_name)

# Print summary of peak ages
print("\n" + "="*80)
print("PEAK AGES FOR IMPULSIVITY NETWORK CONNECTIONS (Validated)")
print("="*80)
if peak_ages_impulsivity:
    for connection, peak in sorted(peak_ages_impulsivity.items(), key=lambda x: x[1]):
        print(f"{connection}: Peak at {peak} years")
else:
    print("No significant inverted-U trajectories with valid peak ages detected.")
print("="*80)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    jitter = 0.15
    colors = {"Rhesus": "steelblue", "Tonkean": "purple"}

    # LEFT POSITION (0) - BEHAVIORAL COHORT
    for species in ["Rhesus", "Tonkean"]:
        males = combined_df[
            (combined_df["species"] == species) & (combined_df["gender"] != 2)
        ]
        females = combined_df[
            (combined_df["species"] == species) & (combined_df["gender"] == 2)
        ]

        sns.stripplot(
            x=[0] * len(females),
            y="age_at_task",
            data=females,
            jitter=jitter,
            marker="o",
            color=colors[species],
            alpha=0.7,
            size=12,
            ax=ax,
        )
        sns.stripplot(
            x=[0] * len(males),
            y="age_at_task",
            data=males,
            jitter=jitter,
            marker="o",
            facecolors="none",
            edgecolor=colors[species],
            size=12,
            linewidth=1.5,
            ax=ax,
        )

    # RIGHT POSITION (1) - NEUROIMAGING COHORT
    for species in ["Rhesus", "Tonkean"]:
        # Behavioral subjects with imaging
        males = combined_df[
            (combined_df["species"] == species) & 
            (combined_df["gender"] != 2) & 
            (combined_df["age_at_scan"].notna())
        ]
        females = combined_df[
            (combined_df["species"] == species) & 
            (combined_df["gender"] == 2) & 
            (combined_df["age_at_scan"].notna())
        ]

        sns.stripplot(
            x=[1] * len(females),
            y="age_at_scan",
            data=females,
            jitter=jitter,
            marker="o",
            color=colors[species],
            alpha=0.7,
            size=12,
            ax=ax,
        )

        sns.stripplot(
            x=[1] * len(males),
            y="age_at_scan",
            data=males,
            jitter=jitter,
            marker="o",
            facecolors="none",
            edgecolor=colors[species],
            size=12,
            linewidth=1.5,
            alpha=0.7,
            ax=ax,
        )

        # Non-behavioral subjects (same opacity)
        non_behavioral_valid = non_behavioral_df[non_behavioral_df["age_at_scan"].notna()].copy()
        
        if not non_behavioral_valid.empty:
            if "species" not in non_behavioral_valid.columns or non_behavioral_valid["species"].isna().all():
                non_behavioral_valid["species"] = "Rhesus"
            
            males_non_behav = non_behavioral_valid[
                (non_behavioral_valid["species"] == species) & 
                (non_behavioral_valid["gender"] == 1)
            ]
            females_non_behav = non_behavioral_valid[
                (non_behavioral_valid["species"] == species) & 
                (non_behavioral_valid["gender"] == 2)
            ]
            
            if not females_non_behav.empty:
                sns.stripplot(
                    x=[1] * len(females_non_behav),
                    y="age_at_scan",
                    data=females_non_behav,
                    jitter=jitter,
                    marker="o",
                    color=colors[species],
                    alpha=0.7,
                    size=12,
                    ax=ax,
                )
            
            if not males_non_behav.empty:
                sns.stripplot(
                    x=[1] * len(males_non_behav),
                    y="age_at_scan",
                    data=males_non_behav,
                    jitter=jitter,
                    marker="o",
                    facecolors="none",
                    edgecolor=colors[species],
                    size=12,
                    linewidth=1.5,
                    alpha=0.7,
                    ax=ax,
                )

    # Calculate sample sizes
    n_behavioral = len(combined_df)
    
    rhesus_neuro = len(combined_df[(combined_df["species"] == "Rhesus") & (combined_df["age_at_scan"].notna())])
    tonkean_neuro = len(combined_df[(combined_df["species"] == "Tonkean") & (combined_df["age_at_scan"].notna())])
    
    if not non_behavioral_valid.empty:
        rhesus_neuro += len(non_behavioral_valid[non_behavioral_valid["species"] == "Rhesus"])
        tonkean_neuro += len(non_behavioral_valid[non_behavioral_valid["species"] == "Tonkean"])
    
    n_neuroimaging = rhesus_neuro + tonkean_neuro

    # Styling
    ax.set_ylabel("Age", fontsize=28, fontname='Times New Roman')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [f'Behavior\n(n = {n_behavioral})', f'Brain\n(n = {n_neuroimaging})'],
        fontsize=26,
        fontname='Times New Roman',
    )
    ax.set_xlim(-0.6, 1.6)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Set y-axis ticks and labels - only show 5, 15, 25
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.set_yticklabels(['', '5', '', '15', '', '25'], fontsize=26, fontname='Times New Roman')
    ax.set_ylim(0, 25)
    ax.set_xlabel("")
    
    # Darker grid
    ax.grid(True, which="both", linestyle="--", alpha=0.5, linewidth=0.8)

    plt.tight_layout()

    filename_base = "demographics-merged-AIC"
    save_path_svg = os.path.join(figure_me_out, f"{filename_base}.svg")
    save_path_png = os.path.join(figure_me_out, f"{filename_base}.png")

    plt.savefig(save_path_svg, format="svg")
    plt.savefig(save_path_png, format="png", dpi=300)
    plt.show()



with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    jitter = 0.15
    colors = {"Rhesus": "steelblue", "Tonkean": "purple"}

    # LEFT POSITION (0) - BEHAVIORAL COHORT
    for species in ["Rhesus", "Tonkean"]:
        males = combined_df[
            (combined_df["species"] == species) & (combined_df["gender"] != 2)
        ]
        females = combined_df[
            (combined_df["species"] == species) & (combined_df["gender"] == 2)
        ]

        sns.stripplot(
            x=[0] * len(females),
            y="age_at_task",
            data=females,
            jitter=jitter,
            marker="o",
            color=colors[species],
            alpha=0.7,
            size=12,
            ax=ax,
        )
        sns.stripplot(
            x=[0] * len(males),
            y="age_at_task",
            data=males,
            jitter=jitter,
            marker="o",
            facecolors="none",
            edgecolor=colors[species],
            size=12,
            linewidth=1.5,
            ax=ax,
        )

    # RIGHT POSITION (1) - NEUROIMAGING COHORT
    for species in ["Rhesus", "Tonkean"]:
        # Behavioral subjects with imaging
        males = combined_df[
            (combined_df["species"] == species) & 
            (combined_df["gender"] != 2) & 
            (combined_df["age_at_scan"].notna())
        ]
        females = combined_df[
            (combined_df["species"] == species) & 
            (combined_df["gender"] == 2) & 
            (combined_df["age_at_scan"].notna())
        ]

        sns.stripplot(
            x=[1] * len(females),
            y="age_at_scan",
            data=females,
            jitter=jitter,
            marker="o",
            color=colors[species],
            alpha=0.7,
            size=12,
            ax=ax,
        )

        sns.stripplot(
            x=[1] * len(males),
            y="age_at_scan",
            data=males,
            jitter=jitter,
            marker="o",
            facecolors="none",
            edgecolor=colors[species],
            size=12,
            linewidth=1.5,
            alpha=0.7,
            ax=ax,
        )

        # Non-behavioral subjects (same opacity)
        non_behavioral_valid = non_behavioral_df[non_behavioral_df["age_at_scan"].notna()].copy()
        
        if not non_behavioral_valid.empty:
            if "species" not in non_behavioral_valid.columns or non_behavioral_valid["species"].isna().all():
                non_behavioral_valid["species"] = "Rhesus"
            
            males_non_behav = non_behavioral_valid[
                (non_behavioral_valid["species"] == species) & 
                (non_behavioral_valid["gender"] == 1)
            ]
            females_non_behav = non_behavioral_valid[
                (non_behavioral_valid["species"] == species) & 
                (non_behavioral_valid["gender"] == 2)
            ]
            
            if not females_non_behav.empty:
                sns.stripplot(
                    x=[1] * len(females_non_behav),
                    y="age_at_scan",
                    data=females_non_behav,
                    jitter=jitter,
                    marker="o",
                    color=colors[species],
                    alpha=0.7,
                    size=12,
                    ax=ax,
                )
            
            if not males_non_behav.empty:
                sns.stripplot(
                    x=[1] * len(males_non_behav),
                    y="age_at_scan",
                    data=males_non_behav,
                    jitter=jitter,
                    marker="o",
                    facecolors="none",
                    edgecolor=colors[species],
                    size=12,
                    linewidth=1.5,
                    alpha=0.7,
                    ax=ax,
                )

    # Calculate sample sizes
    n_behavioral = len(combined_df)
    
    rhesus_neuro = len(combined_df[(combined_df["species"] == "Rhesus") & (combined_df["age_at_scan"].notna())])
    tonkean_neuro = len(combined_df[(combined_df["species"] == "Tonkean") & (combined_df["age_at_scan"].notna())])
    
    if not non_behavioral_valid.empty:
        rhesus_neuro += len(non_behavioral_valid[non_behavioral_valid["species"] == "Rhesus"])
        tonkean_neuro += len(non_behavioral_valid[non_behavioral_valid["species"] == "Tonkean"])
    
    n_neuroimaging = rhesus_neuro + tonkean_neuro

    # Styling
    ax.set_ylabel("Age", fontsize=28, fontname='Times New Roman')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [f'Behavior\n(n = {n_behavioral})', f'Brain\n(n = {n_neuroimaging})'],
        fontsize=26,
        fontname='Times New Roman',
    )
    ax.set_xlim(-0.6, 1.6)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Set y-axis ticks and labels - only show 5, 15, 25
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.set_yticklabels(['', '5', '', '15', '', '25'], fontsize=26, fontname='Times New Roman')
    ax.set_ylim(0, 25)
    ax.set_xlabel("")
    
    # Darker grid
    ax.grid(True, which="both", linestyle="--", alpha=0.5, linewidth=0.8)

    # Add title - bold and italic, MUCH BIGGER
    fig.suptitle(
        "Demographics",
        fontsize=36,
        fontweight="bold",
        fontstyle="italic",
        fontname="Times New Roman",
        color="black",
        y=0.98,
    )

    plt.tight_layout()

    filename_base = "demographics-merged-AIC"
    save_path_svg = os.path.join(figure_me_out, f"{filename_base}.svg")
    save_path_png = os.path.join(figure_me_out, f"{filename_base}.png")

    plt.savefig(save_path_svg, format="svg")
    plt.savefig(save_path_png, format="png", dpi=300)
    plt.show()