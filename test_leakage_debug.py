"""Quick debug script for leakage detector."""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from engine.leakage_detector import LeakageDetector

# Generate data
n_samples = 200
n_features = 10
X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
df['target'] = y
df['date'] = pd.date_range('2020-01-01', periods=len(df))

# Add leakage
df['leaky'] = df['target'] * 5

print("DataFrame shape:", df.shape)
print("DataFrame columns:", df.columns.tolist())

# Test detector
try:
    detector = LeakageDetector(verbose=True)
    result = detector.run(df, 'target', date_col='date')
    print("\n=== RESULT ===")
    print("Result type:", type(result))
    print("Result keys:", list(result.keys()) if isinstance(result, dict) else "NOT A DICT")
    if isinstance(result, dict):
        for key in result.keys():
            print(f"  - {key}: {type(result[key])}")
        if 'findings' in result:
            print("\nFindings keys:", list(result['findings'].keys()))
except Exception as e:
    import traceback
    print("\n=== ERROR ===")
    print(f"Exception: {e}")
    traceback.print_exc()
