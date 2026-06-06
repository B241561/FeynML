"""
Unified Leakage Scanner Module
==============================
Orchestrates target and temporal leakage detection.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add local directory to path for imports
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from target_leakage import leakage_score, permutation_importance_spike
from temporal_leakage import temporal_leakage_report

def scan(X, y, df, model=None, feature_names=None, time_col=None, train_idx=None, test_idx=None):
    """
    Run unified leakage report scanner.
    """
    report = {
        "target_leakage": {},
        "temporal_leakage": {}
    }
    
    # Preprocessing: Factorize categorical columns for numeric routines (Correlation, MI)
    # We use 'df' if available, otherwise we inspect 'X'
    if df is not None:
        X_proc = df.drop(columns=[time_col] if time_col else [], errors='ignore')
        # If 'target' is in df, drop it to avoid self-correlation
        X_proc = X_proc.drop(columns=['target'], errors='ignore')
    elif isinstance(X, pd.DataFrame):
        X_proc = X.copy()
    else:
        # Convert numpy to dataframe for easier factorisation
        X_proc = pd.DataFrame(X, columns=feature_names)
    
    for col in X_proc.columns:
        if not pd.api.types.is_numeric_dtype(X_proc[col].dtype):
            # Factorize preserving NaNs
            mask = X_proc[col].isnull()
            codes, _ = pd.factorize(X_proc[col])
            X_proc[col] = pd.Series(codes, index=X_proc.index, dtype=float)
            X_proc.loc[mask, col] = np.nan
            
    X_data = X_proc.values
    feature_names = X_proc.columns.tolist()

    # 1. Target Leakage
    # Mutual Information and Correlation do not require a model
    report["target_leakage"]["scores"] = leakage_score(X_proc, y, model, feature_names)
    
    if model is not None:
        report["target_leakage"]["spike"] = permutation_importance_spike(model, X_data, y, feature_names)
    
    # 2. Temporal Leakage
    if df is not None and time_col is not None:
        target_col = 'target' # Default placeholder
        report["temporal_leakage"] = temporal_leakage_report(df, time_col, target_col, train_idx, test_idx)
        
    return report

def rank_leakage_suspects(report):
    """
    Prioritize leakage suspects from report.
    """
    suspects = []
    
    # Check target leakage scores
    target_scores = report.get("target_leakage", {}).get("scores", {})
    for feat, score in target_scores.items():
        if score > 0.8:
            suspects.append({"feature": feat, "score": score, "type": "target_leakage", "severity": "HIGH"})
        elif score > 0.5:
            suspects.append({"feature": feat, "score": score, "type": "target_leakage", "severity": "MEDIUM"})
            
    # Check spike
    spike = report.get("target_leakage", {}).get("spike", {})
    if spike.get("is_spike"):
        suspects.append({
            "feature": spike["top_feature"], 
            "score": spike["spike_ratio"], 
            "type": "importance_spike", 
            "severity": "HIGH"
        })
        
    # Check future features
    future_feats = report.get("temporal_leakage", {}).get("future_features", [])
    for f in future_feats:
        suspects.append({
            "feature": f["feature"], 
            "score": f["violations"], 
            "type": "future_feature", 
            "severity": "CRITICAL"
        })
        
    return suspects

def leakage_summary(report):
    """
    Generate plain-English summary of leakage findings.
    """
    suspects = rank_leakage_suspects(report)
    if not suspects:
        return "No significant leakage detected."
        
    summary = [f"Detected {len(suspects)} potential leakage issues:"]
    for s in suspects:
        summary.append(f"- [{s['severity']}] {s['feature']}: {s['type']} (Score: {s['score']:.2f})")
        
    return "\n".join(summary)

def run_verification():
    """Run module verification."""
    print("--- Leakage Scanner Verification ---")
    from sklearn.ensemble import RandomForestClassifier
    
    np.random.seed(42)
    N = 100
    dates = pd.date_range('2023-01-01', periods=N, freq='D')
    X = np.random.randn(N, 3)
    y = (X[:, 0] > 0).astype(int)
    
    # Inject leakage
    df = pd.DataFrame(X, columns=['f1', 'f2', 'f3'])
    df['timestamp'] = dates
    df['leak'] = y + np.random.normal(0, 0.01, N)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(df.drop('timestamp', axis=1), y)
    
    report = scan(df.drop('timestamp', axis=1), y, df, model, time_col='timestamp')
    print(leakage_summary(report))

if __name__ == "__main__":
    run_verification()
