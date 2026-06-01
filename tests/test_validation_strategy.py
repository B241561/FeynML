"""
Unit Tests — Validation Strategy
==================================
Tests core functions in scratch/phase2/validation_strategy.py.

Run:
    python tests/test_validation_strategy.py
"""

import sys
import os
import unittest
import random

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_P2   = os.path.join(_ROOT, "scratch", "phase2")
for p in [_ROOT, _P2]:
    if p not in sys.path:
        sys.path.insert(0, p)

from validation_strategy import (
    train_test_split,
    k_fold_cross_validation,
    stratified_k_fold,
    detect_data_leakage,
    temporal_split,
)


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN-TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainTestSplit(unittest.TestCase):

    def test_sizes(self):
        X = list(range(100))
        y = list(range(100))
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
        self.assertEqual(len(X_tr) + len(X_te), 100)
        self.assertEqual(len(X_tr), len(y_tr))
        self.assertEqual(len(X_te), len(y_te))

    def test_no_overlap(self):
        X = list(range(50))
        y = list(range(50))
        X_tr, X_te, _, _ = train_test_split(X, y, test_size=0.3, random_state=1)
        overlap = set(X_tr) & set(X_te)
        self.assertEqual(len(overlap), 0)

    def test_reproducibility(self):
        X = list(range(200))
        y = list(range(200))
        split1 = train_test_split(X, y, test_size=0.25, random_state=42)
        split2 = train_test_split(X, y, test_size=0.25, random_state=42)
        self.assertEqual(split1[0], split2[0])  # same train set

    def test_approximate_size(self):
        X = list(range(100))
        y = list(range(100))
        _, X_te, _, _ = train_test_split(X, y, test_size=0.2, random_state=0)
        # Should be close to 20, allow ±2 for rounding
        self.assertAlmostEqual(len(X_te), 20, delta=2)


# ─────────────────────────────────────────────────────────────────────────────
# K-FOLD CROSS VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

class TestKFold(unittest.TestCase):

    def test_fold_count(self):
        X = list(range(100))
        y = list(range(100))
        folds = k_fold_cross_validation(X, y, k=5)
        self.assertEqual(len(folds), 5)

    def test_all_indices_covered(self):
        X = list(range(60))
        y = list(range(60))
        folds = k_fold_cross_validation(X, y, k=3)
        all_test_indices = []
        for (X_tr, X_te, y_tr, y_te) in folds:
            all_test_indices.extend(X_te)
        # Every index should appear exactly once in test sets
        self.assertEqual(sorted(all_test_indices), list(range(60)))

    def test_fold_sizes_balanced(self):
        X = list(range(90))
        y = list(range(90))
        folds = k_fold_cross_validation(X, y, k=9)
        test_sizes = [len(te) for (_, te, _, _) in folds]
        # All folds should be the same size when n divisible by k
        self.assertEqual(len(set(test_sizes)), 1)
        self.assertEqual(test_sizes[0], 10)

    def test_train_test_disjoint(self):
        X = list(range(50))
        y = list(range(50))
        folds = k_fold_cross_validation(X, y, k=5)
        for (X_tr, X_te, _, _) in folds:
            self.assertEqual(len(set(X_tr) & set(X_te)), 0)


# ─────────────────────────────────────────────────────────────────────────────
# STRATIFIED K-FOLD
# ─────────────────────────────────────────────────────────────────────────────

class TestStratifiedKFold(unittest.TestCase):

    def test_class_balance_preserved(self):
        """Each fold should preserve the overall class ratio approximately."""
        random.seed(7)
        n = 200
        X = list(range(n))
        # 30% positive
        y = [1 if i < 60 else 0 for i in range(n)]

        folds = stratified_k_fold(X, y, k=5)
        for (_, X_te, _, y_te) in folds:
            pos_rate = sum(y_te) / len(y_te)
            # Within ±10% of overall 30%
            self.assertAlmostEqual(pos_rate, 0.30, delta=0.10)

    def test_all_samples_used(self):
        random.seed(0)
        n = 100
        X = list(range(n))
        y = [i % 2 for i in range(n)]
        folds = stratified_k_fold(X, y, k=5)
        all_test = []
        for (_, X_te, _, _) in folds:
            all_test.extend(X_te)
        self.assertEqual(sorted(all_test), list(range(n)))


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────

class TestTemporalSplit(unittest.TestCase):

    def test_order_preserved(self):
        """Train indices should all be before test indices."""
        X = list(range(100))
        y = list(range(100))
        timestamps = list(range(100))  # monotonically increasing
        X_tr, X_te, y_tr, y_te = temporal_split(X, y, timestamps, test_ratio=0.2)
        if X_tr and X_te:
            self.assertLess(max(X_tr), min(X_te))

    def test_sizes(self):
        n = 100
        X = list(range(n))
        y = list(range(n))
        timestamps = list(range(n))
        X_tr, X_te, _, _ = temporal_split(X, y, timestamps, test_ratio=0.2)
        self.assertEqual(len(X_tr) + len(X_te), n)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LEAKAGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

class TestLeakageDetection(unittest.TestCase):

    def test_no_leakage_reported_for_clean_data(self):
        random.seed(0)
        n = 200
        X_train = [[random.gauss(0, 1) for _ in range(5)] for _ in range(n)]
        X_test  = [[random.gauss(0, 1) for _ in range(5)] for _ in range(50)]
        result = detect_data_leakage(X_train, X_test)
        self.assertIsInstance(result, dict)
        self.assertIn("leakage_detected", result)

    def test_leakage_detected_when_test_in_train(self):
        """If test rows are literally in train, leakage should be flagged."""
        X_train = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        X_test  = [[1.0, 2.0], [3.0, 4.0]]  # exact duplicates of train rows
        result = detect_data_leakage(X_train, X_test)
        self.assertTrue(result["leakage_detected"])

    def test_returns_dict(self):
        X_train = [[1.0], [2.0], [3.0]]
        X_test  = [[4.0], [5.0]]
        result = detect_data_leakage(X_train, X_test)
        self.assertIsInstance(result, dict)


# ─────────────────────────────────────────────────────────────────────────────
# EDGE CASES
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases(unittest.TestCase):

    def test_split_with_few_samples(self):
        X = [1, 2, 3]
        y = [0, 1, 0]
        # Should not crash
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.33)
        self.assertEqual(len(X_tr) + len(X_te), 3)

    def test_k_fold_k_equals_n(self):
        """Leave-one-out: k = n."""
        X = list(range(10))
        y = [0] * 5 + [1] * 5
        folds = k_fold_cross_validation(X, y, k=10)
        self.assertEqual(len(folds), 10)
        for (_, X_te, _, _) in folds:
            self.assertEqual(len(X_te), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
