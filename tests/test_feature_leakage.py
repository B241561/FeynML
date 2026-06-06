"""
Unit Tests — Feature Leakage (scratch/phase4/feature_leakage/)
============================================================
Deterministic tests for Target and Temporal Leakage detection.

Run:
    python tests/test_feature_leakage.py
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_P4_FL = os.path.join(_ROOT, "scratch", "phase4", "feature_leakage")
for p in [_ROOT, _P4_FL]:
    if p not in sys.path:
        sys.path.insert(0, p)

from target_leakage import mutual_information_score, permutation_importance_spike, leakage_score
from temporal_leakage import check_split_ordering, detect_future_features
from leakage_scanner import scan, rank_leakage_suspects

class TestTargetLeakage(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.N = 100
        self.X = np.random.randn(self.N, 3)
        self.y = (self.X[:, 0] > 0).astype(int)
        
        # Inject leakage in X[:, 2]
        self.X[:, 2] = self.y + np.random.normal(0, 0.001, self.N)
        self.feat_names = ["noise1", "signal1", "LEAK"]
        
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X, self.y)

    def test_mi_spike(self):
        mi = mutual_information_score(self.X, self.y, self.feat_names)
        self.assertGreater(mi["LEAK"], mi["noise1"])
        self.assertGreater(mi["LEAK"], 0.5)

    def test_permutation_importance_spike(self):
        spike = permutation_importance_spike(self.model, self.X, self.y, self.feat_names)
        self.assertEqual(spike["top_feature"], "LEAK")
        self.assertTrue(spike["is_spike"])
        self.assertGreater(spike["spike_ratio"], 10.0)

    def test_leakage_score_aggregation(self):
        scores = leakage_score(self.X, self.y, self.model, self.feat_names)
        self.assertGreater(scores["LEAK"], 0.9)
        self.assertLess(scores["noise1"], 0.3)

class TestTemporalLeakage(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range('2023-01-01', periods=10, freq='D')
        self.df = pd.DataFrame({
            'timestamp': self.dates,
            'feat': np.random.randn(10)
        })

    def test_valid_split_ordering(self):
        train_idx = list(range(7))
        test_idx = list(range(7, 10))
        res = check_split_ordering(self.df['timestamp'], train_idx, test_idx)
        self.assertTrue(res['is_valid'])
        self.assertEqual(res['num_violations'], 0)

    def test_invalid_split_ordering(self):
        train_idx = list(range(7))
        test_idx = list(range(5, 10)) # Overlap
        res = check_split_ordering(self.df['timestamp'], train_idx, test_idx)
        self.assertFalse(res['is_valid'])
        self.assertGreater(res['num_violations'], 0)

    def test_future_feature_detection(self):
        df_future = self.df.copy()
        # feat_future is timestamp + 2 days
        df_future['feat_future'] = df_future['timestamp'] + pd.TimedOffset(days=2)
        
        future_feats = detect_future_features(df_future, ['feat', 'feat_future'], 'timestamp')
        self.assertEqual(len(future_feats), 1)
        self.assertEqual(future_feats[0]['feature'], 'feat_future')

class TestLeakageScanner(unittest.TestCase):
    def test_full_scan(self):
        np.random.seed(42)
        N = 50
        dates = pd.date_range('2023-01-01', periods=N, freq='D')
        X = np.random.randn(N, 2)
        y = (X[:, 0] > 0).astype(int)
        
        df = pd.DataFrame(X, columns=['f1', 'f2'])
        df['timestamp'] = dates
        # Inject leakage
        df['leak'] = y + np.random.normal(0, 0.001, N)
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(df.drop('timestamp', axis=1), y)
        
        report = scan(df.drop('timestamp', axis=1), y, df, model, time_col='timestamp')
        suspects = rank_leakage_suspects(report)
        
        feat_suspects = [s['feature'] for s in suspects]
        self.assertIn('leak', feat_suspects)

if __name__ == "__main__":
    unittest.main()
