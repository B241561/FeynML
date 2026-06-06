"""
Potential Outcomes Framework Module
===================================
Implementation of Rubin's Potential Outcomes framework for causal effect estimation.

Core Concepts:
--------------
- Y(1): Outcome if treated
- Y(0): Outcome if untreated
- ATE (Average Treatment Effect): E[Y(1) - Y(0)]
- ATT (Average Treatment Effect on the Treated): E[Y(1) - Y(0) | T=1]
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

def naive_ate(y, treatment):
    """Compute naive mean difference (biased without randomization)."""
    return np.mean(y[treatment == 1]) - np.mean(y[treatment == 0])

def propensity_score(X, treatment):
    """Estimate propensity scores P(T=1|X) using Logistic Regression."""
    model = LogisticRegression(random_state=42)
    model.fit(X, treatment)
    return model.predict_proba(X)[:, 1]

def ipw_ate(y, treatment, propensity):
    """
    Estimate ATE using Inverse Probability Weighting (IPW).
    ATE = (1/N) * Σ [ (T_i * Y_i / e_i) - ((1-T_i) * Y_i / (1-e_i)) ]
    """
    # Clip propensity to avoid division by zero
    e = np.clip(propensity, 0.01, 0.99)
    term1 = (treatment * y) / e
    term2 = ((1 - treatment) * y) / (1 - e)
    return np.mean(term1 - term2)

def matching_ate(X, y, treatment, k=1):
    """
    Estimate ATE using nearest-neighbor matching.
    For each unit, find its counterfactual by matching on X.
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    else:
        X_arr = X
    
    # Ensure y is a numpy array for multidimensional indexing
    y_arr = y.values if hasattr(y, 'values') else np.array(y)
        
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]
    
    # Match treated to control
    nn_control = NearestNeighbors(n_neighbors=k).fit(X_arr[control_idx])
    _, indices_in_control = nn_control.kneighbors(X_arr[treated_idx])
    # Counterfactual for treated is the mean of matched controls
    y1_treated = y_arr[treated_idx]
    y0_treated_matched = np.mean(y_arr[control_idx[indices_in_control]], axis=1)
    
    # Match control to treated
    nn_treated = NearestNeighbors(n_neighbors=k).fit(X_arr[treated_idx])
    _, indices_in_treated = nn_treated.kneighbors(X_arr[control_idx])
    # Counterfactual for control is the mean of matched treated
    y0_control = y_arr[control_idx]
    y1_control_matched = np.mean(y_arr[treated_idx[indices_in_treated]], axis=1)
    
    # ATE = Mean of (observed - matched) for all units
    # Treated units: y1_obs - y0_matched
    # Control units: y1_matched - y0_obs
    diff_treated = y1_treated - y0_treated_matched
    diff_control = y1_control_matched - y0_control
    
    return np.mean(np.concatenate([diff_treated, diff_control]))

def matching_att(X, y, treatment, k=1):
    """
    Estimate ATT using nearest-neighbor matching.
    ATT = E[Y(1) - Y(0) | T=1]. Only look at treated units.
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    else:
        X_arr = X
        
    # Ensure y is a numpy array for multidimensional indexing
    y_arr = y.values if hasattr(y, 'values') else np.array(y)

    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]
    
    nn_control = NearestNeighbors(n_neighbors=k).fit(X_arr[control_idx])
    _, indices_in_control = nn_control.kneighbors(X_arr[treated_idx])
    
    y1_treated = y_arr[treated_idx]
    y0_treated_matched = np.mean(y_arr[control_idx[indices_in_control]], axis=1)
    
    return np.mean(y1_treated - y0_treated_matched)

def overlap_check(propensity):
    """Check for positivity/overlap violations."""
    # Propensity scores should not be too close to 0 or 1
    violations = np.sum((propensity < 0.05) | (propensity > 0.95))
    return {
        "num_violations": int(violations),
        "violation_rate": float(violations / len(propensity)),
        "min_ps": float(np.min(propensity)),
        "max_ps": float(np.max(propensity)),
        "is_valid": bool(violations / len(propensity) < 0.1)
    }

def run_verification():
    """Run module verification."""
    print("--- Potential Outcomes Verification ---")
    np.random.seed(42)
    N = 500
    X = np.random.randn(N, 2)
    # Propensity: older people more likely to get treatment
    ps_true = 1 / (1 + np.exp(-(X[:, 0])))
    treatment = (np.random.random(N) < ps_true).astype(int)
    
    # Outcome: treatment adds 5, X[0] adds 2
    # Y = 5*T + 2*X0 + noise
    true_ate = 5.0
    y = 5 * treatment + 2 * X[:, 0] + np.random.normal(0, 0.5, N)
    
    ps_est = propensity_score(X, treatment)
    ate_naive = naive_ate(y, treatment)
    ate_ipw = ipw_ate(y, treatment, ps_est)
    ate_match = matching_ate(X, y, treatment)
    
    print(f"True ATE: {true_ate}")
    print(f"Naive ATE: {ate_naive:.2f} (Biased)")
    print(f"IPW ATE:   {ate_ipw:.2f}")
    print(f"Match ATE: {ate_match:.2f}")
    
    print(f"Overlap Check: {overlap_check(ps_est)}")

if __name__ == "__main__":
    run_verification()
