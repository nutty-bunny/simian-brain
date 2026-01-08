"""
Generate comprehensive statistical table for all ROI connectivity pairs
Compares linear vs quadratic models across the lifespan
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

# Use the functions from your existing script
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
        p_value_f = 1 - stats.f.cdf(f_stat, df_model, df_error)
    
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

def linear_vs_quadratic_testing(df, x_col, y_col):
    """Run linear vs quadratic comparison"""
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
    
    # LINEAR MODEL
    X_reshaped = X.reshape(-1, 1)
    linear_model = LinearRegression()
    linear_model.fit(X_reshaped, y)
    y_pred_linear = linear_model.predict(X_reshaped)
    
    linear_sig = calculate_model_significance(y, y_pred_linear, 2, n)
    if linear_sig:
        linear_sig.update({
            'params': [linear_model.intercept_, linear_model.coef_[0]],
            'model_type': 'linear',
            'vs_linear_p': np.nan
        })
        results['linear'] = linear_sig
    
    # QUADRATIC MODEL
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
                'vs_linear_p': vs_linear_p,
                'better_than_linear': vs_linear_p < 0.05 if not np.isnan(vs_linear_p) else False
            })
            results['quadratic'] = quad_sig
    except Exception as e:
        results['quadratic'] = None

    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if valid_results:
        best_aic = min(valid_results.keys(), key=lambda k: valid_results[k]['aic'])
        delta_aic = (results['linear']['aic'] - results['quadratic']['aic']) if 'quadratic' in results and results['quadratic'] else np.nan
        
        results['best_aic'] = best_aic
        results['delta_aic'] = delta_aic
        results['n'] = n
    
    return results

def find_fc_column(a, b, table):
    """Find FC column name for a pair of ROIs"""
    direct = f"fc_{a}_{b}"
    flipped = f"fc_{b}_{a}"
    if direct in table.columns:
        return direct
    elif flipped in table.columns:
        return flipped
    else:
        raise ValueError(f"No FC column found for pair {a}–{b}")

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("="*80)
print("GENERATING COMPREHENSIVE CONNECTIVITY STATISTICS TABLE")
print("="*80)

# Combine all subjects (behavioral + non-behavioral)
all_data = pd.concat([
    rhesus_table[['name', 'age_at_scan', 'gender', 'species'] + [col for col in rhesus_table.columns if col.startswith('fc_')]],
    tonkean_table[['name', 'age_at_scan', 'gender', 'species'] + [col for col in tonkean_table.columns if col.startswith('fc_')]],
    non_behavioral_df[['name', 'age_at_scan', 'gender', 'species'] + [col for col in non_behavioral_df.columns if col.startswith('fc_')]]
], ignore_index=True)

# Define all connections
print("\nDefining connectivity pairs...")

# ATTENTION NETWORK
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

# IMPULSIVITY NETWORK
core_impulsivity = ["thalamus", "dorsal_striatum", "nucleus_accumbens", "OFC"]
impulsivity_connections = [(a, b) for i, a in enumerate(core_impulsivity) 
                           for j, b in enumerate(core_impulsivity) if i < j]

# ALL CONNECTIONS
all_connections = attention_connections + impulsivity_connections

print(f"Total attention connections: {len(attention_connections)}")
print(f"Total impulsivity connections: {len(impulsivity_connections)}")
print(f"Total connections to analyze: {len(all_connections)}")

# Build results table
results_list = []

print("\nProcessing connections...")
for i, (area1, area2) in enumerate(all_connections, 1):
    try:
        metric = find_fc_column(area1, area2, rhesus_table)
        connection_name = f"{area1}-{area2}"
        
        print(f"  [{i}/{len(all_connections)}] {connection_name}...", end="")
        
        # Determine network
        if (area1, area2) in attention_connections:
            network = "Attention"
        else:
            network = "Impulsivity"
        
        # Run statistical tests
        test_results = linear_vs_quadratic_testing(all_data, 'age_at_scan', metric)
        
        if test_results is None:
            print(" INSUFFICIENT DATA")
            continue
        
        # Extract results
        n = test_results.get('n', 0)
        
        # Linear results
        lin_r2 = test_results.get('linear', {}).get('r2', np.nan)
        lin_p = test_results.get('linear', {}).get('p_value', np.nan)
        lin_aic = test_results.get('linear', {}).get('aic', np.nan)
        lin_sig = test_results.get('linear', {}).get('significant', False)
        
        # Quadratic results
        quad_r2 = test_results.get('quadratic', {}).get('r2', np.nan)
        quad_p = test_results.get('quadratic', {}).get('p_value', np.nan)
        quad_aic = test_results.get('quadratic', {}).get('aic', np.nan)
        quad_sig = test_results.get('quadratic', {}).get('significant', False)
        quad_vs_linear_p = test_results.get('quadratic', {}).get('vs_linear_p', np.nan)
        quad_better = test_results.get('quadratic', {}).get('better_than_linear', False)
        
        # Model comparison
        delta_aic = test_results.get('delta_aic', np.nan)
        best_aic = test_results.get('best_aic', 'linear')
        
        # Determine pattern
        if quad_better and quad_sig:
            quad_coef = test_results.get('quadratic', {}).get('params', [0, 0, 0])[2]
            if quad_coef < 0:
                pattern = "Inverted-U"
            else:
                pattern = "U-shaped"
        elif lin_sig:
            lin_coef = test_results.get('linear', {}).get('params', [0, 0])[1]
            if lin_coef > 0:
                pattern = "Linear ↑"
            else:
                pattern = "Linear ↓"
        else:
            pattern = "None"
        
        results_list.append({
            'Network': network,
            'Connection': connection_name,
            'N': n,
            'Linear_R2': lin_r2,
            'Linear_p': lin_p,
            'Linear_AIC': lin_aic,
            'Linear_sig': '✓' if lin_sig else '',
            'Quad_R2': quad_r2,
            'Quad_p': quad_p,
            'Quad_AIC': quad_aic,
            'Quad_sig': '✓' if quad_sig else '',
            'Quad_vs_Lin_p': quad_vs_linear_p,
            'Quad_better': '✓' if quad_better else '',
            'Delta_AIC': delta_aic,
            'Best_AIC': best_aic,
            'Pattern': pattern
        })
        
        print(f" ✓ (Pattern: {pattern})")
        
    except ValueError as e:
        print(f" SKIPPED ({e})")
        continue
    except Exception as e:
        print(f" ERROR ({e})")
        continue

# Create DataFrame
results_df = pd.DataFrame(results_list)

# Save to CSV
output_path = os.path.join(figure_me_out, "connectivity_statistics_table.csv")
results_df.to_csv(output_path, index=False)

print("\n" + "="*80)
print("TABLE GENERATION COMPLETE")
print("="*80)
print(f"\nSaved to: {output_path}")
print(f"Total connections analyzed: {len(results_df)}")

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\nBy Network:")
print(results_df.groupby('Network').size())

print("\nSignificant Patterns:")
print(results_df[results_df['Pattern'] != 'None']['Pattern'].value_counts())

print("\nQuadratic Better than Linear:")
print(f"  {results_df['Quad_better'].sum()} out of {len(results_df)} connections")

print("\nLinear Significant:")
print(f"  {results_df['Linear_sig'].sum()} out of {len(results_df)} connections")

print("\nQuadratic Significant:")
print(f"  {results_df['Quad_sig'].sum()} out of {len(results_df)} connections")

# Print formatted table
print("\n" + "="*120)
print("FULL RESULTS TABLE")
print("="*120)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
pd.set_option('display.max_rows', None)

# Format for display
display_df = results_df.copy()
display_df['Linear_R2'] = display_df['Linear_R2'].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "—")
display_df['Linear_p'] = display_df['Linear_p'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "—")
display_df['Quad_R2'] = display_df['Quad_R2'].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "—")
display_df['Quad_p'] = display_df['Quad_p'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "—")
display_df['Quad_vs_Lin_p'] = display_df['Quad_vs_Lin_p'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "—")
display_df['Delta_AIC'] = display_df['Delta_AIC'].apply(lambda x: f"{x:.1f}" if not pd.isna(x) else "—")

print(display_df.to_string(index=False))

print("\n" + "="*120)
print("LEGEND")
print("="*120)
print("Linear_sig / Quad_sig: ✓ = p < 0.05")
print("Quad_better: ✓ = Quadratic significantly better than linear (LRT p < 0.05)")
print("Delta_AIC: Linear_AIC - Quad_AIC (positive = quadratic better)")
print("Pattern: Developmental trajectory pattern")
print("  Inverted-U = Peak in middle age")
print("  U-shaped = Valley in middle age")
print("  Linear ↑ = Increases with age")
print("  Linear ↓ = Decreases with age")
print("  None = No significant pattern")
print("="*120)