import os
import pandas as pd
import numpy as np
import pickle
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, LeaveOneOut
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def pure_behavioral_cross_task_analysis():
    """Clean cross-task behavioral analysis without brain or hierarchy data"""
    print("="*70)
    print("PURE CROSS-TASK BEHAVIORAL ANALYSIS")
    print("="*70)
    
    # Load data
    data_dir = "/Users/similovesyou/Desktop/qts/simian-behavior/data/py"
    base_dir = "/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt"
    
    # Load behavioral datasets
    datasets = {}
    for task in ['csrt', 'dms', 'tav']:
        filename = 'data.pickle' if task == 'csrt' else f'{task}.pickle'
        with open(os.path.join(data_dir, filename), 'rb') as f:
            data = pickle.load(f)
            # Remove problematic subjects
            for species in ['rhesus', 'tonkean']:
                if species in data:
                    for name in ['joy', 'jipsy']:
                        if name in data[species]:
                            del data[species][name]
            datasets[task] = data
    
    # Load ELO data for subject info
    rhesus_elo = pd.read_csv(os.path.join(base_dir, "derivatives/rhesus-elo.csv"))
    tonkean_elo = pd.read_csv(os.path.join(base_dir, "derivatives/tonkean-elo.csv"))
    rhesus_elo['species'] = 'Rhesus'
    tonkean_elo['species'] = 'Tonkean'
    all_elo = pd.concat([rhesus_elo, tonkean_elo], ignore_index=True)
    
    def extract_comprehensive_behavioral_metrics(attempts_df, task_name):
        """Extract comprehensive behavioral metrics"""
        if attempts_df.empty:
            return {}
        
        # Basic performance metrics
        total = len(attempts_df)
        success_rate = (attempts_df['result'] == 'success').mean()
        error_rate = (attempts_df['result'] == 'error').mean()
        premature_rate = (attempts_df['result'].isin(['premature', 'prematured'])).mean()
        omission_rate = (attempts_df['result'] == 'stepomission').mean()
        
        metrics = {
            f'{task_name}_total_attempts': total,
            f'{task_name}_success_rate': success_rate,
            f'{task_name}_error_rate': error_rate,
            f'{task_name}_premature_rate': premature_rate,
            f'{task_name}_omission_rate': omission_rate,
            f'{task_name}_n_sessions': attempts_df['session'].nunique() if 'session' in attempts_df.columns else 1
        }
        
        # Reaction time metrics
        success_trials = attempts_df[attempts_df['result'] == 'success']
        if not success_trials.empty and 'reaction_time' in success_trials.columns:
            rt_data = success_trials['reaction_time'].dropna()
            if not rt_data.empty:
                metrics[f'{task_name}_mean_rt'] = rt_data.mean()
                metrics[f'{task_name}_median_rt'] = rt_data.median()
                metrics[f'{task_name}_cv_rt'] = rt_data.std() / rt_data.mean()
                metrics[f'{task_name}_rt_p25'] = rt_data.quantile(0.25)
                metrics[f'{task_name}_rt_p75'] = rt_data.quantile(0.75)
                metrics[f'{task_name}_rt_range'] = rt_data.max() - rt_data.min()
        
        # Performance stability metrics
        if 'session' in attempts_df.columns and attempts_df['session'].nunique() >= 3:
            session_performance = attempts_df.groupby('session')['result'].apply(
                lambda x: (x == 'success').mean()
            )
            if len(session_performance) >= 3:
                metrics[f'{task_name}_session_consistency'] = 1.0 - session_performance.std()
                metrics[f'{task_name}_best_session_performance'] = session_performance.max()
                metrics[f'{task_name}_worst_session_performance'] = session_performance.min()
                metrics[f'{task_name}_performance_range'] = session_performance.max() - session_performance.min()
        
        # Learning analysis
        if total >= 30:
            # Sort by progression (or time if no progression)
            if 'progression' in attempts_df.columns:
                attempts_sorted = attempts_df.sort_values('progression')
            else:
                attempts_sorted = attempts_df.sort_values('instant_begin')
            
            # Divide into thirds for learning analysis
            third = total // 3
            early_trials = attempts_sorted.iloc[:third]
            middle_trials = attempts_sorted.iloc[third:2*third]
            late_trials = attempts_sorted.iloc[-third:]
            
            early_success = (early_trials['result'] == 'success').mean()
            middle_success = (middle_trials['result'] == 'success').mean()
            late_success = (late_trials['result'] == 'success').mean()
            
            metrics[f'{task_name}_early_success'] = early_success
            metrics[f'{task_name}_middle_success'] = middle_success
            metrics[f'{task_name}_late_success'] = late_success
            metrics[f'{task_name}_learning_slope'] = late_success - early_success
            metrics[f'{task_name}_learning_acceleration'] = (late_success - middle_success) - (middle_success - early_success)
            
            # Learning stability
            if third >= 10:
                early_premature = (early_trials['result'].isin(['premature', 'prematured'])).mean()
                late_premature = (late_trials['result'].isin(['premature', 'prematured'])).mean()
                metrics[f'{task_name}_premature_learning'] = early_premature - late_premature
        
        # Task-specific metrics
        if task_name == 'csrt' and 'variable_delay' in attempts_df.columns:
            delay_data = attempts_df['variable_delay'].dropna()
            if not delay_data.empty and len(delay_data) >= 20:
                metrics[f'{task_name}_mean_delay'] = delay_data.mean()
                
                # Performance by delay quartiles
                delay_quartiles = pd.qcut(delay_data, q=4, labels=['q1', 'q2', 'q3', 'q4'], duplicates='drop')
                for quartile in ['q1', 'q2', 'q3', 'q4']:
                    if quartile in delay_quartiles.values:
                        quartile_trials = attempts_df.loc[delay_data.index[delay_quartiles == quartile]]
                        if len(quartile_trials) >= 5:
                            metrics[f'{task_name}_success_{quartile}_delay'] = (
                                quartile_trials['result'] == 'success'
                            ).mean()
                
                # Delay sensitivity
                if 'q1' in [f'{task_name}_success_{q}_delay' for q in ['q1', 'q4'] if f'{task_name}_success_{q}_delay' in metrics]:
                    short_delay_perf = metrics.get(f'{task_name}_success_q1_delay', 0)
                    long_delay_perf = metrics.get(f'{task_name}_success_q4_delay', 0)
                    metrics[f'{task_name}_delay_sensitivity'] = short_delay_perf - long_delay_perf
        
        # Error pattern analysis
        if total >= 20:
            error_types = ['error', 'premature', 'prematured', 'stepomission']
            for error_type in error_types:
                if error_type in ['premature', 'prematured']:
                    error_mask = attempts_df['result'].isin(['premature', 'prematured'])
                else:
                    error_mask = attempts_df['result'] == error_type
                
                if error_mask.sum() >= 5:
                    error_trials = attempts_df[error_mask]
                    if 'session' in error_trials.columns:
                        error_sessions = error_trials['session'].nunique()
                        total_sessions = attempts_df['session'].nunique()
                        metrics[f'{task_name}_{error_type}_session_spread'] = error_sessions / total_sessions
        
        return metrics
    
    # Build comprehensive dataset
    print("\nBuilding comprehensive behavioral dataset...")
    unified_data = all_elo.copy()
    
    # Extract metrics for all tasks
    for task_name, dataset in datasets.items():
        print(f"  Processing {task_name.upper()}...")
        
        for species in ['rhesus', 'tonkean']:
            if species in dataset:
                for subject_name, subject_data in dataset[species].items():
                    if 'attempts' in subject_data:
                        metrics = extract_comprehensive_behavioral_metrics(
                            subject_data['attempts'], task_name
                        )
                        
                        # Add to unified dataset
                        subject_idx = unified_data[unified_data['name'] == subject_name].index
                        if len(subject_idx) > 0:
                            idx = subject_idx[0]
                            for metric_name, value in metrics.items():
                                if metric_name not in unified_data.columns:
                                    unified_data[metric_name] = np.nan
                                unified_data.at[idx, metric_name] = value
    
    # Convert to numeric
    behavioral_cols = [col for col in unified_data.columns 
                      if any(task in col for task in ['csrt_', 'dms_', 'tav_'])]
    for col in behavioral_cols:
        unified_data[col] = pd.to_numeric(unified_data[col], errors='coerce')
    
    print(f"\nFinal dataset shape: {unified_data.shape}")
    print(f"Behavioral features: {len(behavioral_cols)}")
    
    # Data availability
    print(f"\nData availability:")
    for task in ['csrt', 'dms', 'tav']:
        col = f'{task}_total_attempts'
        if col in unified_data.columns:
            available = unified_data[col].notna().sum()
            print(f"  {task.upper()}: {available}/{len(unified_data)} subjects")
    
    # Enhanced correlation analysis
    print(f"\n" + "="*60)
    print("COMPREHENSIVE CROSS-TASK CORRELATION ANALYSIS")
    print("="*60)
    
    # CSRT outcomes (dependent variables)
    csrt_outcomes = [
        'csrt_success_rate', 'csrt_premature_rate', 'csrt_error_rate', 'csrt_omission_rate',
        'csrt_mean_rt', 'csrt_cv_rt', 'csrt_learning_slope'
    ]
    
    # Predictors from other tasks
    predictor_categories = {
        'performance': ['success_rate', 'error_rate', 'premature_rate', 'omission_rate'],
        'timing': ['mean_rt', 'median_rt', 'cv_rt'],
        'learning': ['learning_slope', 'early_success', 'late_success'],
        'stability': ['session_consistency', 'performance_range']
    }
    
    predictors = []
    for task in ['dms', 'tav']:
        for category, metrics in predictor_categories.items():
            for metric in metrics:
                col = f'{task}_{metric}'
                if col in unified_data.columns:
                    predictors.append(col)
    
    # Add age
    predictors.append('age')
    
    # Calculate correlations by category
    correlation_results = {}
    
    for outcome in csrt_outcomes:
        if outcome not in unified_data.columns:
            continue
            
        outcome_data = unified_data[outcome].dropna()
        if len(outcome_data) < 8:
            continue
        
        print(f"\n{outcome.upper()} correlations (n={len(outcome_data)}):")
        outcome_corrs = {}
        
        # Group correlations by category
        categorized_corrs = {cat: [] for cat in predictor_categories.keys()}
        categorized_corrs['demographic'] = []
        
        for predictor in predictors:
            if predictor not in unified_data.columns:
                continue
                
            valid_idx = outcome_data.index.intersection(unified_data[predictor].dropna().index)
            
            if len(valid_idx) >= 8:
                y = outcome_data.loc[valid_idx].astype(float)
                x = unified_data.loc[valid_idx, predictor].astype(float)
                
                if x.std() > 1e-10 and y.std() > 1e-10:
                    try:
                        r, p = pearsonr(x, y)
                        outcome_corrs[predictor] = {'r': r, 'p': p, 'n': len(valid_idx)}
                        
                        # Categorize correlation
                        if predictor == 'age':
                            categorized_corrs['demographic'].append((predictor, r, p, len(valid_idx)))
                        else:
                            for category, metrics in predictor_categories.items():
                                if any(metric in predictor for metric in metrics):
                                    categorized_corrs[category].append((predictor, r, p, len(valid_idx)))
                                    break
                    except:
                        continue
        
        # Display correlations by category
        for category, corr_list in categorized_corrs.items():
            if corr_list:
                print(f"  {category.upper()}:")
                for predictor, r, p, n in sorted(corr_list, key=lambda x: abs(x[1]), reverse=True):
                    if p < 0.1:
                        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "†"
                        print(f"    {predictor}: r={r:.3f}, p={p:.3f} {sig} (n={n})")
        
        correlation_results[outcome] = outcome_corrs
    
    # Simple, robust prediction models
    print(f"\n" + "="*60)
    print("SIMPLE PREDICTION MODELS (SINGLE PREDICTORS)")
    print("="*60)
    
    for outcome in ['csrt_success_rate', 'csrt_premature_rate']:
        if outcome not in correlation_results:
            continue
            
        outcome_data = unified_data[outcome].dropna()
        if len(outcome_data) < 10:
            continue
        
        print(f"\n{outcome.upper()} prediction:")
        
        # Find best single predictor
        corrs = correlation_results[outcome]
        significant_predictors = [(pred, stats) for pred, stats in corrs.items() 
                                if stats['p'] < 0.05 and stats['n'] >= 10]
        
        if significant_predictors:
            # Sort by correlation strength
            best_predictor, best_stats = max(significant_predictors, 
                                          key=lambda x: abs(x[1]['r']))
            
            print(f"  Best predictor: {best_predictor}")
            print(f"  Correlation: r={best_stats['r']:.3f}, p={best_stats['p']:.3f}, n={best_stats['n']}")
            
            # Simple linear model
            valid_idx = outcome_data.index.intersection(
                unified_data[best_predictor].dropna().index
            )
            
            if len(valid_idx) >= 10:
                y = outcome_data.loc[valid_idx].astype(float)
                x = unified_data.loc[valid_idx, best_predictor].astype(float).values.reshape(-1, 1)
                
                # Cross-validated R²
                model = LinearRegression()
                if len(y) >= 15:
                    cv_scores = cross_val_score(model, x, y, cv=5, scoring='r2')
                else:
                    cv_scores = cross_val_score(model, x, y, cv=LeaveOneOut(), scoring='r2')
                
                cv_r2 = cv_scores.mean()
                cv_r2_std = cv_scores.std()
                
                print(f"  Cross-validated R² = {cv_r2:.3f} ± {cv_r2_std:.3f}")
                
                # Fit final model
                model.fit(x, y)
                slope = model.coef_[0]
                intercept = model.intercept_
                
                print(f"  Model: {outcome} = {intercept:.3f} + {slope:.3f} * {best_predictor}")
                print(f"  Interpretation: 1 unit increase in {best_predictor} → {slope:.3f} change in {outcome}")
    
    # Cross-task consistency analysis
    print(f"\n" + "="*60)
    print("CROSS-TASK CONSISTENCY ANALYSIS")
    print("="*60)
    
    consistency_metrics = ['success_rate', 'premature_rate', 'error_rate', 'mean_rt', 'learning_slope']
    
    print("Cross-task correlations for key metrics:")
    for metric in consistency_metrics:
        csrt_col = f'csrt_{metric}'
        dms_col = f'dms_{metric}'
        tav_col = f'tav_{metric}'
        
        correlations_found = []
        
        # CSRT vs DMS
        if csrt_col in unified_data.columns and dms_col in unified_data.columns:
            valid_data = unified_data[[csrt_col, dms_col]].dropna()
            if len(valid_data) >= 8:
                r, p = pearsonr(valid_data[csrt_col], valid_data[dms_col])
                correlations_found.append(f"CSRT-DMS: r={r:.3f}, p={p:.3f}, n={len(valid_data)}")
        
        # CSRT vs TAV
        if csrt_col in unified_data.columns and tav_col in unified_data.columns:
            valid_data = unified_data[[csrt_col, tav_col]].dropna()
            if len(valid_data) >= 8:
                r, p = pearsonr(valid_data[csrt_col], valid_data[tav_col])
                correlations_found.append(f"CSRT-TAV: r={r:.3f}, p={p:.3f}, n={len(valid_data)}")
        
        # DMS vs TAV
        if dms_col in unified_data.columns and tav_col in unified_data.columns:
            valid_data = unified_data[[dms_col, tav_col]].dropna()
            if len(valid_data) >= 8:
                r, p = pearsonr(valid_data[dms_col], valid_data[tav_col])
                correlations_found.append(f"DMS-TAV: r={r:.3f}, p={p:.3f}, n={len(valid_data)}")
        
        if correlations_found:
            print(f"\n{metric.upper()}:")
            for corr_str in correlations_found:
                print(f"  {corr_str}")
    
    # Summary
    print(f"\n" + "="*60)
    print("KEY FINDINGS SUMMARY")
    print("="*60)
    
    print(f"\nDataset: {len(unified_data)} subjects")
    print(f"  Species: Rhesus={sum(unified_data['species']=='Rhesus')}, Tonkean={sum(unified_data['species']=='Tonkean')}")
    
    print(f"\nStrongest cross-task relationships:")
    all_strong_corrs = []
    for outcome, corrs in correlation_results.items():
        for pred, stats in corrs.items():
            if stats['p'] < 0.01 and stats['n'] >= 15 and 'csrt' not in pred:
                all_strong_corrs.append((abs(stats['r']), outcome, pred, stats))
    
    all_strong_corrs.sort(reverse=True)
    for abs_r, outcome, pred, stats in all_strong_corrs[:15]:
        print(f"  {outcome} ~ {pred}: r={stats['r']:.3f}, p={stats['p']:.3f}, n={stats['n']}")
 
    return unified_data, correlation_results

# Run the analysis
if __name__ == "__main__":
    data, correlations = pure_behavioral_cross_task_analysis()