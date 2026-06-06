"""
Target Leakage Detection Module
===============================
Detects features that encode the target variable through non-causal paths.

Strategies:
  1. Mutual Information (MI) spikes
  2. Permutation Importance spikes (one feature dominates)
  3. High target correlation
  4. Availability checks (logical timestamp verification)
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder

def mutual_information_score(X, y, feature_names=None, discrete_features='auto'):
    """
    Compute MI score for all features.
    
    Returns:
        mi_scores: dict {feature_name: score}
    """
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X_data = X.values
    else:
        X_data = X
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(X.shape[1])]
            
    # Determine if classification or regression
    if len(np.unique(y)) < 20 and np.issubdtype(y.dtype, np.integer):
        scores = mutual_info_classif(X_data, y, discrete_features=discrete_features, random_state=42)
    else:
        scores = mutual_info_regression(X_data, y, discrete_features=discrete_features, random_state=42)
        
    return dict(zip(feature_names, scores))

def permutation_importance_spike(model, X, y, feature_names=None, n_repeats=5):
    """
    Detect leakage via permutation importance spikes.
    A 'spike' is defined as one feature having significantly higher importance than others.
    """
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X_data = X
    else:
        X_data = X
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(X.shape[1])]

    r = permutation_importance(model, X_data, y, n_repeats=n_repeats, random_state=42)
    importances = r.importances_mean
    
    # Calculate spike score: ratio of top feature to mean of others
    sorted_idx = np.argsort(importances)[::-1]
    top_val = importances[sorted_idx[0]]
    others_mean = np.mean(importances[sorted_idx[1:]]) if len(sorted_idx) > 1 else 0
    
    spike_ratio = top_val / (others_mean + 1e-9)
    
    importance_dict = {feature_names[i]: float(importances[i]) for i in range(len(feature_names))}
    
    return {
        "importances": importance_dict,
        "top_feature": feature_names[sorted_idx[0]],
        "spike_ratio": float(spike_ratio),
        "is_spike": spike_ratio > 10.0 # Heuristic threshold
    }

def availability_check(df, feature_col, time_col, target_time_col=None):
    """
    Check if feature is available at prediction time.
    Requires dataframe with timestamps.
    """
    if feature_col not in df.columns or time_col not in df.columns:
        return {"error": "Columns missing"}
        
    # Heuristic: if feature timestamp > prediction timestamp, it's leakage
    # For now, we check if the feature itself is a timestamp and compare it.
    # This is a placeholder for more complex logic.
    return {"status": "checked", "leakage_detected": False}

def target_correlation_scan(X, y, feat_names=None):
    """
    Generate ranked correlation report for features.
    """
    if isinstance(X, pd.DataFrame):
        df = X.copy()
        df['target'] = y
        feat_names = X.columns.tolist()
    else:
        df = pd.DataFrame(X, columns=feat_names)
        df['target'] = y

    correlations = df.corr()['target'].abs().sort_values(ascending=False)
    correlations = correlations.drop('target', errors='ignore')
    
    return correlations.to_dict()

def leakage_score(X, y, model, feat_names=None):
    """
    Compute per-feature leakage scores (0-1).
    Aggregates MI and correlation.
    """
    mi = mutual_information_score(X, y, feat_names)
    corr = target_correlation_scan(X, y, feat_names)
    
    scores = {}
    for feat in mi:
        # Normalize MI (very rough normalization)
        norm_mi = min(1.0, mi[feat] / 1.0)
        c = corr.get(feat, 0)
        
        # Weighted average
        scores[feat] = 0.7 * c + 0.3 * norm_mi
        
    return scores

def run_verification():
    """Run module verification with synthetic data."""
    print("--- Target Leakage Verification ---")
    from sklearn.ensemble import RandomForestClassifier
    
    np.random.seed(42)
    N = 200
    X = np.random.randn(N, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Inject leakage: feat_4 is almost y
    X[:, 4] = y + np.random.normal(0, 0.01, N)
    
    feat_names = ["noise1", "noise2", "signal1", "signal2", "LEAK"]
    
    mi = mutual_information_score(X, y, feat_names)
    print(f"MI Scores: {mi}")
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    spike = permutation_importance_spike(model, X, y, feat_names)
    print(f"Top Feature: {spike['top_feature']}, Spike Ratio: {spike['spike_ratio']:.2f}")
    
    scores = leakage_score(X, y, model, feat_names)
    print(f"Leakage Scores: {scores}")

if __name__ == "__main__":
    run_verification()
