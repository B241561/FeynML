"""
Missingness Mechanism Classifier
================================
Classifies features into MCAR, MAR, or MNAR mechanisms.

Definitions (Rubin 1976):
-------------------------
1. MCAR (Missing Completely At Random):
   P(M | X_obs, X_miss) = P(M)
   Missingness is independent of any data.

2. MAR (Missing At Random):
   P(M | X_obs, X_miss) = P(M | X_obs)
   Missingness depends only on observed data.

3. MNAR (Missing Not At Random):
   P(M | X_obs, X_miss) depends on X_miss.
   Missingness depends on the missing values themselves.
"""

import numpy as np
import pandas as pd
import os
import sys

# Import Little's test
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from little_mcar_test import little_mcar_test

def compute_missingness_rate(X):
    """
    Compute per-feature missingness rate.
    """
    if isinstance(X, pd.DataFrame):
        return X.isnull().mean().to_dict()
    return {f"feat_{i}": np.isnan(X[:, i]).mean() for i in range(X.shape[1])}

def missingness_indicator_matrix(X):
    """
    Generate binary indicator matrix M (1 if missing, 0 if observed).
    """
    if isinstance(X, pd.DataFrame):
        return X.isnull().astype(int).values
    return np.isnan(X).astype(int)

def correlate_missingness_with_observed(X):
    """
    Heuristic for MAR: correlate missingness indicators of feature i 
    with observed values of other features j.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
        
    M = X.isnull().astype(int)
    # Fill NaNs with mean for correlation calculation
    X_filled = X.fillna(X.mean())
    
    mar_evidence = {}
    for col in X.columns:
        if X[col].isnull().sum() == 0: continue
        
        m_col = M[col]
        corrs = []
        for other_col in X.columns:
            if col == other_col: continue
            c = np.abs(m_col.corr(X_filled[other_col]))
            if not np.isnan(c):
                corrs.append(c)
        
        mar_evidence[col] = np.max(corrs) if corrs else 0.0
        
    return mar_evidence

def correlate_missingness_with_target(X, y):
    """
    Heuristic for MNAR: correlate missingness indicators with target variable.
    If the fact that data is missing predicts the target, it suggests selection bias.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
        
    M = X.isnull().astype(int)
    mnar_evidence = {}
    
    y_series = pd.Series(y)
    
    for col in X.columns:
        if X[col].isnull().sum() == 0: continue
        c = np.abs(M[col].corr(y_series))
        mnar_evidence[col] = c if not np.isnan(c) else 0.0
        
    return mnar_evidence

def classify_mechanism(X, y=None, alpha=0.05):
    """
    Classify missingness mechanism per feature.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
        
    # 1. Check Global MCAR
    global_mcar = little_mcar_test(X, alpha=alpha)
    
    # 2. Check MAR and MNAR evidence
    mar_scores = correlate_missingness_with_observed(X)
    mnar_scores = correlate_missingness_with_target(X, y) if y is not None else {}
    
    classifications = {}
    for col in X.columns:
        if X[col].isnull().sum() == 0:
            classifications[col] = "NONE"
            continue
            
        # Decision Logic
        if global_mcar["mcar_likely"]:
            classifications[col] = "MCAR"
        else:
            mar_val = mar_scores.get(col, 0)
            mnar_val = mnar_scores.get(col, 0)
            
            if mnar_val > 0.3: # Heuristic threshold
                classifications[col] = "MNAR"
            elif mar_val > 0.2: # Heuristic threshold
                classifications[col] = "MAR"
            else:
                # Default to MNAR if not MCAR and not clearly MAR (more conservative)
                classifications[col] = "MNAR (Suspected)"
                
    return {
        "classifications": classifications,
        "global_mcar": global_mcar,
        "mar_evidence": mar_scores,
        "mnar_evidence": mnar_scores
    }

def run_verification():
    """Run module verification."""
    print("--- Missingness Classifier Verification ---")
    np.random.seed(42)
    N = 1000
    
    X = np.random.randn(N, 3)
    # Target correlated with X_0
    y = X[:, 0] + np.random.normal(0, 0.5, N)
    
    df = pd.DataFrame(X, columns=['f0', 'f1', 'f2'])
    
    # MCAR: random
    df.loc[np.random.choice(N, 100), 'f1'] = np.nan
    
    # MAR: missingness in f2 depends on f0
    mask_mar = df['f0'] > 0.5
    df.loc[mask_mar, 'f2'] = np.nan
    
    # MNAR: missingness in f0 depends on f0 itself
    # (Simulated via target correlation since we can't see the missing values)
    mask_mnar = df['f0'] > 1.0
    df.loc[mask_mnar, 'f0'] = np.nan
    
    results = classify_mechanism(df, y)
    print("Classifications:")
    for col, mech in results['classifications'].items():
        print(f"  {col}: {mech}")

if __name__ == "__main__":
    run_verification()
