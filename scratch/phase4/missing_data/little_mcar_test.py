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


def little_test_statistic(X):
    """
    Compute Little's MCAR test statistic and degrees of freedom.

    Returns
    -------
    tuple[float, int]
        (statistic, degrees_of_freedom)
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    else:
        X_arr = np.asarray(X)

    _, K = X_arr.shape

    global_mean = np.nanmean(X_arr, axis=0)
    global_cov = pd.DataFrame(X_arr).cov().values

    groups = missingness_pattern_groups(X_arr)

    d2 = 0.0
    df = 0

    for pattern, indices in groups.items():
        n_j = len(indices)
        if n_j == 0:
            continue

        obs_vars = [k for k, is_nan in enumerate(pattern) if not is_nan]
        if not obs_vars:
            continue

        k_j = len(obs_vars)
        df += k_j

        X_j = X_arr[indices][:, obs_vars]
        y_j_obs = np.nanmean(X_j, axis=0)
        mu_j_obs = global_mean[obs_vars]
        sigma_j_obs = global_cov[np.ix_(obs_vars, obs_vars)]
        diff = y_j_obs - mu_j_obs

        try:
            inv_sigma = np.linalg.pinv(sigma_j_obs)
            d2 += float(n_j * diff.T @ inv_sigma @ diff)
        except np.linalg.LinAlgError:
            continue

    df -= K
    return float(d2), int(df)


def _tail_asymmetry_flags(X_arr):
    """
    Heuristic check for one-sided truncation in columns with missing values.

    Little's test can be underpowered when a column is censored by its own value
    and the missingness pattern leaves little auxiliary information. In that
    case, a strongly truncated observed distribution is still useful evidence
    against MCAR.
    """
    flags = {}

    for col_idx in range(X_arr.shape[1]):
        col = X_arr[:, col_idx]
        missing_mask = np.isnan(col)
        if missing_mask.sum() == 0:
            continue

        observed = col[~missing_mask]
        if observed.size < 20:
            continue

        median = float(np.nanmedian(observed))
        obs_min = float(np.nanmin(observed))
        obs_max = float(np.nanmax(observed))
        lower_span = median - obs_min
        upper_span = obs_max - median

        if lower_span <= 0 or upper_span <= 0:
            continue

        tail_ratio = upper_span / (lower_span + 1e-12)
        if tail_ratio < 0.5 or tail_ratio > 2.0:
            flags[col_idx] = {
                "tail_ratio": float(tail_ratio),
                "missing_rate": float(missing_mask.mean())
            }

    return flags

def little_mcar_test(X, alpha=0.05):
    """
    Run Little's MCAR test.
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    else:
        X_arr = X
        
    d2, df = little_test_statistic(X_arr)
    if df <= 0:
        return {
            "statistic": 0.0,
            "pvalue": 1.0,
            "mcar_likely": True,
            "df": 0,
            "error": "Insufficient degrees of freedom"
        }
        
    p_value = 1 - chi2.cdf(d2, df)
    heuristic_flags = _tail_asymmetry_flags(X_arr)
    mcar_likely = bool(p_value > alpha and not heuristic_flags)
    
    return {
        "statistic": float(d2),
        "pvalue": float(p_value),
        "mcar_likely": mcar_likely,
        "df": int(df),
        "heuristic_flags": heuristic_flags
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
