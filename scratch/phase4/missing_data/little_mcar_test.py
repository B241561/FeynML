"""
Little's MCAR Test Module
=========================
Implementation of Little's (1988) test for Missing Completely At Random (MCAR).

Theory:
-------
The null hypothesis is that the data is MCAR. The test partitions the data 
into groups with identical missingness patterns. For each group, it compares 
the observed means to the global ML estimates of the means.

Statistic:
d^2 = Σ n_j * (y_j_obs - μ_j_obs)^T * Σ_j_obs^-1 * (y_j_obs - μ_j_obs)
where:
- n_j is the number of samples in pattern group j
- y_j_obs is the mean of observed variables in group j
- μ_j_obs is the global mean of those same variables
- Σ_j_obs is the global covariance of those same variables

The statistic follows a chi-square distribution with df = Σ k_j - k, 
where k_j is the number of observed variables in group j and k is the 
total number of variables.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2

def missingness_pattern_groups(X):
    """
    Group rows by identical missingness patterns.
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    else:
        X_arr = X
        
    mask = np.isnan(X_arr)
    # Convert mask to tuples to make them hashable
    patterns = [tuple(row) for row in mask]
    
    unique_patterns = sorted(list(set(patterns)))
    groups = {p: [] for p in unique_patterns}
    
    for i, p in enumerate(patterns):
        groups[p].append(i)
        
    return groups

def little_mcar_test(X, alpha=0.05):
    """
    Run Little's MCAR test.
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    else:
        X_arr = X
        
    N, K = X_arr.shape
    
    # 1. Global estimates (ignoring NaNs for simplicity in this scratch version)
    # A more robust version would use EM algorithm to estimate global mean/cov
    global_mean = np.nanmean(X_arr, axis=0)
    global_cov = np.cov(X_arr, rowvar=False, ddof=1)
    # Handle NaNs in covariance calculation if np.cov doesn't
    if np.any(np.isnan(global_cov)):
        # Fallback to pairwise covariance
        df_tmp = pd.DataFrame(X_arr)
        global_cov = df_tmp.cov().values
    
    groups = missingness_pattern_groups(X_arr)
    
    d2 = 0
    df = 0
    
    for pattern, indices in groups.items():
        n_j = len(indices)
        if n_j == 0: continue
        
        # Identify observed variables in this pattern
        obs_vars = [k for k, is_nan in enumerate(pattern) if not is_nan]
        if not obs_vars: continue # Skip all-missing rows
        
        k_j = len(obs_vars)
        df += k_j
        
        # Sub-data for this group
        X_j = X_arr[indices][:, obs_vars]
        y_j_obs = np.nanmean(X_j, axis=0)
        
        # Global estimates for these variables
        mu_j_obs = global_mean[obs_vars]
        sigma_j_obs = global_cov[np.ix_(obs_vars, obs_vars)]
        
        # Difference
        diff = y_j_obs - mu_j_obs
        
        try:
            # Inv sigma
            inv_sigma = np.linalg.pinv(sigma_j_obs)
            d2 += n_j * diff.T @ inv_sigma @ diff
        except np.linalg.LinAlgError:
            continue
            
    df -= K # Total df = sum(k_j) - k
    
    if df <= 0:
        return {
            "statistic": 0.0,
            "pvalue": 1.0,
            "mcar_likely": True,
            "df": 0,
            "error": "Insufficient degrees of freedom"
        }
        
    p_value = 1 - chi2.cdf(d2, df)
    
    return {
        "statistic": float(d2),
        "pvalue": float(p_value),
        "mcar_likely": bool(p_value > alpha),
        "df": int(df)
    }

def run_verification():
    """Run module verification with synthetic data."""
    print("--- Little's MCAR Test Verification ---")
    np.random.seed(42)
    
    N, K = 500, 3
    X = np.random.randn(N, K)
    
    # 1. Case: MCAR
    X_mcar = X.copy()
    mask = np.random.random((N, K)) < 0.1
    X_mcar[mask] = np.nan
    
    res_mcar = little_mcar_test(X_mcar)
    print(f"MCAR Data - p-value: {res_mcar['pvalue']:.4f}, Likely MCAR: {res_mcar['mcar_likely']}")
    
    # 2. Case: MNAR (Missingness depends on the value itself)
    X_mnar = X.copy()
    # If X_0 is large, make it more likely to be missing
    mask_mnar = X_mnar[:, 0] > 1.0
    X_mnar[mask_mnar, 0] = np.nan
    
    res_mnar = little_mcar_test(X_mnar)
    print(f"MNAR Data - p-value: {res_mnar['pvalue']:.4f}, Likely MCAR: {res_mnar['mcar_likely']}")

if __name__ == "__main__":
    run_verification()
