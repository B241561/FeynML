"""
Unit Tests — Missing Data (scratch/phase4/missing_data/)
=======================================================
Deterministic tests for Little's test, MICE, and mechanism classification.

Run:
    python tests/test_missing_data.py
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_P4_MD = os.path.join(_ROOT, "scratch", "phase4", "missing_data")
for p in [_ROOT, _P4_MD]:
    if p not in sys.path:
        sys.path.insert(0, p)

from little_mcar_test import little_mcar_test
from multiple_imputation import mice_cycle
from missingness_classifier import classify_mechanism
from missingness_as_signal import create_indicator_features, indicator_target_correlation

class TestLittleTest(unittest.TestCase):
    def test_mcar_detection(self):
        np.random.seed(42)
        N, K = 200, 2
        X = np.random.randn(N, K)
        # Randomly inject 5% NaNs
        mask = np.random.random((N, K)) < 0.05
        X[mask] = np.nan
        
        res = little_mcar_test(X)
        # For small N and purely random noise, p-value should be > 0.05
        self.assertTrue(res['mcar_likely'])

    def test_mnar_rejection(self):
        np.random.seed(42)
        N, K = 300, 2
        X = np.random.randn(N, K)
        # Missingness depends on value (MNAR)
        X[X[:, 0] > 1.5, 0] = np.nan
        
        res = little_mcar_test(X)
        # MNAR data should typically fail Little's test (low p-value)
        self.assertFalse(res['mcar_likely'])

class TestMICE(unittest.TestCase):
    def test_imputation_completion(self):
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(50, 2), columns=['a', 'b'])
        df.loc[0:10, 'a'] = np.nan
        
        df_imputed = mice_cycle(df, n_cycles=3)
        self.assertEqual(df_imputed['a'].isnull().sum(), 0)

    def test_correlation_preservation(self):
        np.random.seed(42)
        N = 100
        x1 = np.random.randn(N)
        x2 = x1 * 0.9 + np.random.normal(0, 0.1, N)
        df = pd.DataFrame({'x1': x1, 'x2': x2})
        
        # Inject NaNs in x2
        df.loc[0:20, 'x2'] = np.nan
        
        df_imputed = mice_cycle(df, n_cycles=5)
        corr = df_imputed['x1'].corr(df_imputed['x2'])
        # Correlation should be close to 0.9
        self.assertGreater(corr, 0.7)

class TestClassifier(unittest.TestCase):
    def test_mechanism_classification_structure(self):
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(50, 2), columns=['f1', 'f2'])
        df.loc[0:5, 'f1'] = np.nan
        
        results = classify_mechanism(df)
        self.assertIn('f1', results['classifications'])
        self.assertIn('global_mcar', results)

class TestSignalAnalysis(unittest.TestCase):
    def test_indicator_creation(self):
        df = pd.DataFrame({'a': [1, np.nan, 3], 'b': [4, 5, 6]})
        aug_df = create_indicator_features(df)
        self.assertIn('is_missing_a', aug_df.columns)
        self.assertNotIn('is_missing_b', aug_df.columns)

    def test_signal_correlation(self):
        y = np.array([1, 1, 0, 0])
        df = pd.DataFrame({'a': [np.nan, np.nan, 3, 4]})
        corrs = indicator_target_correlation(df, y)
        # Perfect correlation between missingness in 'a' and y=1
        self.assertAlmostEqual(abs(corrs['a']), 1.0)

if __name__ == "__main__":
    unittest.main()
