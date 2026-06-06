"""
Missingness-as-Signal Module
============================
Treats the pattern of missingness itself as a predictive feature.

Key Insight:
-----------
The fact that data is missing is often informative. This module 
quantifies that signal by correlating missingness indicators 
with the target and clustering missingness patterns.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def create_indicator_features(X, feature_names=None):
    """
    Augment X with binary missingness indicators.
    """
    if not isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
        
    M = X.isnull().astype(int)
    # Only keep indicators for columns that actually have missingness
    M = M.loc[:, M.any()]
    M.columns = [f"is_missing_{c}" for c in M.columns]
    
    return pd.concat([X, M], axis=1)

def missingness_pattern_clusters(X, n_clusters=3, random_state=42):
    """
    Identify clusters of missingness patterns using KMeans on indicator matrix.
    """
    if isinstance(X, pd.DataFrame):
        M = X.isnull().astype(int).values
    else:
        M = np.isnan(X).astype(int)
        
    if not M.any():
        return np.zeros(M.shape[0])
        
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(M)
    
    return clusters

def indicator_target_correlation(X, y):
    """
    Rank correlations between missingness indicators and the target.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
        
    M = X.isnull().astype(int)
    y_series = pd.Series(y)
    
    corrs = {}
    for col in M.columns:
        if M[col].any():
            c = M[col].corr(y_series)
            corrs[col] = c if not np.isnan(c) else 0.0
            
    # Sort by absolute correlation
    sorted_corrs = dict(sorted(corrs.items(), key=lambda item: abs(item[1]), reverse=True))
    return sorted_corrs

def should_include_indicators(X, y, threshold=0.1):
    """
    Decide which indicator features to include based on target correlation.
    """
    corrs = indicator_target_correlation(X, y)
    include = {feat: abs(val) >= threshold for feat, val in corrs.items()}
    return include

def run_verification():
    """Run module verification."""
    print("--- Missingness-as-Signal Verification ---")
    np.random.seed(42)
    N = 200
    X = np.random.randn(N, 2)
    # Missingness in f1 depends on target (MNAR)
    y = X[:, 0] + np.random.normal(0, 0.5, N)
    
    df = pd.DataFrame(X, columns=['f0', 'f1'])
    # Inject signal-heavy missingness
    mask = y > 0.5
    df.loc[mask, 'f1'] = np.nan
    
    corrs = indicator_target_correlation(df, y)
    print(f"Indicator-Target Correlations: {corrs}")
    
    clusters = missingness_pattern_clusters(df, n_clusters=2)
    print(f"Pattern Clusters (first 10): {clusters[:10]}")
    
    aug_df = create_indicator_features(df)
    print(f"Augmented Columns: {aug_df.columns.tolist()}")

if __name__ == "__main__":
    run_verification()
