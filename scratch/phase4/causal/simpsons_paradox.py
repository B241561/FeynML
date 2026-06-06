"""
Simpson's Paradox Detection Module
==================================
Detects and demonstrates Simpson's Paradox — when a trend present in 
subgroups reverses when groups are combined.
"""

import numpy as np
import pandas as pd

def aggregate_correlation(X, y):
    """Compute aggregate correlation between X and y."""
    return np.corrcoef(X, y)[0, 1]

def stratum_correlations(X, y, Z):
    """Compute correlations within each stratum of Z."""
    df = pd.DataFrame({'X': X, 'y': y, 'Z': Z})
    corrs = {}
    for val in df['Z'].unique():
        subset = df[df['Z'] == val]
        if len(subset) > 1:
            c = subset['X'].corr(subset['y'])
            corrs[val] = c
    return corrs

def detect_simpsons_paradox(X, y, Z):
    """
    Detect if Simpson's Paradox is present.
    Paradox detected if sign(aggregate correlation) != sign(all stratum correlations).
    """
    agg_corr = aggregate_correlation(X, y)
    strata = stratum_correlations(X, y, Z)
    
    agg_sign = np.sign(agg_corr)
    strata_signs = [np.sign(c) for c in strata.values() if not np.isnan(c)]
    
    # Paradox if all strata have different sign than aggregate
    detected = False
    if strata_signs:
        if all(s != agg_sign and s != 0 for s in strata_signs):
            detected = True
            
    return {
        "detected": detected,
        "aggregate_correlation": float(agg_corr),
        "stratum_correlations": {str(k): float(v) for k, v in strata.items()}
    }

def generate_paradox_example(seed=42):
    """
    Generate synthetic data demonstrating Simpson's Paradox.
    Example: Exercise vs Cholesterol, grouped by Age.
    In each age group, exercise reduces cholesterol (negative corr).
    But older people exercise more and have higher cholesterol (positive aggregate corr).
    """
    np.random.seed(seed)
    N_per_group = 50
    
    # Group 0: Young
    X0 = np.random.normal(2, 1, N_per_group)
    y0 = 10 - 0.5 * X0 + np.random.normal(0, 0.5, N_per_group)
    Z0 = np.zeros(N_per_group)
    
    # Group 1: Old
    X1 = np.random.normal(8, 1, N_per_group)
    y1 = 20 - 0.5 * X1 + np.random.normal(0, 0.5, N_per_group)
    Z1 = np.ones(N_per_group)
    
    X = np.concatenate([X0, X1])
    y = np.concatenate([y0, y1])
    Z = np.concatenate([Z0, Z1])
    
    return X, y, Z

def paradox_explanation(result):
    """Generate plain-English explanation of paradox findings."""
    if not result['detected']:
        return "No Simpson's Paradox detected. Trends are consistent across groups."
    
    agg_trend = "positive" if result['aggregate_correlation'] > 0 else "negative"
    explanation = [
        f"Simpson's Paradox DETECTED!",
        f"The aggregate trend is {agg_trend} (corr: {result['aggregate_correlation']:.2f}),",
        "but within every subgroup, the trend is reversed."
    ]
    return " ".join(explanation)

def run_verification():
    """Run module verification."""
    print("--- Simpson's Paradox Verification ---")
    X, y, Z = generate_paradox_example()
    
    result = detect_simpsons_paradox(X, y, Z)
    print(f"Result: {result}")
    print(paradox_explanation(result))

if __name__ == "__main__":
    run_verification()
