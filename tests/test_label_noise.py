"""
Unit Tests — Label Noise (scratch/phase4/label_noise/)
======================================================
Deterministic tests for Confident Learning, Cleanlab Integration, and Asymmetric Noise.

Run:
    python tests/test_label_noise.py
"""

import sys
import os
import unittest
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_P4_LN = os.path.join(_ROOT, "scratch", "phase4", "label_noise")
for p in [_ROOT, _P4_LN]:
    if p not in sys.path:
        sys.path.insert(0, p)

from confident_learning import (
    compute_thresholds,
    identify_label_errors,
    noise_transition_matrix
)
from asymmetric_noise import inject_asymmetric_noise, detect_asymmetric_noise
from cleanlab_integration import find_label_issues_wrapper, compare_implementations

class TestConfidentLearning(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.N, self.K = 100, 2
        # Simple binary case
        self.y_true = np.random.randint(0, self.K, self.N)
        self.y_noisy = self.y_true.copy()
        # Flip 10%
        self.noise_indices = np.random.choice(self.N, 10, replace=False)
        self.y_noisy[self.noise_indices] = 1 - self.y_noisy[self.noise_indices]
        
        # Simulated probabilities
        self.y_proba = np.zeros((self.N, self.K))
        for i in range(self.N):
            self.y_proba[i, self.y_true[i]] = 0.8 + 0.1 * np.random.random()
            self.y_proba[i, 1 - self.y_true[i]] = 1.0 - self.y_proba[i, self.y_true[i]]

    def test_threshold_calculation(self):
        thresholds = compute_thresholds(self.y_proba, self.y_noisy)
        self.assertEqual(len(thresholds), self.K)
        self.assertTrue(all(0.4 < t < 0.9 for t in thresholds))

    def test_error_identification_recall(self):
        errors = identify_label_errors(self.y_proba, self.y_noisy)
        intersection = np.intersect1d(errors, self.noise_indices)
        recall = len(intersection) / len(self.noise_indices)
        # With 80-90% confidence, we should catch most injected errors
        self.assertGreaterEqual(recall, 0.7)

    def test_noise_matrix_estimation(self):
        T = noise_transition_matrix(self.y_proba, self.y_noisy)
        self.assertEqual(T.shape, (self.K, self.K))
        self.assertAlmostEqual(np.sum(T, axis=1).all(), 1.0)
        # Diagonals should be higher than off-diagonals for low noise
        self.assertTrue(np.diag(T).all() > 0.7)

class TestAsymmetricNoise(unittest.TestCase):
    def test_injection_determinism(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        T = np.array([[0.5, 0.5], [0.1, 0.9]])
        y_n1 = inject_asymmetric_noise(y, T, seed=42)
        y_n2 = inject_asymmetric_noise(y, T, seed=42)
        np.testing.assert_array_equal(y_n1, y_n2)

    def test_detection_asymmetry(self):
        np.random.seed(42)
        # Highly asymmetric matrix
        T_asym = np.array([[0.9, 0.1], [0.4, 0.6]])
        y_true = np.random.randint(0, 2, 500)
        y_noisy = inject_asymmetric_noise(y_true, T_asym, seed=42)
        
        # Perfect probabilities for detection
        y_proba = np.zeros((500, 2))
        for i in range(500):
            y_proba[i, y_true[i]] = 0.99
            y_proba[i, 1-y_true[i]] = 0.01
            
        report = detect_asymmetric_noise(y_proba, y_noisy)
        self.assertTrue(report['is_asymmetric'])
        self.assertGreater(report['off_diag_std'], 0.1)

class TestIntegration(unittest.TestCase):
    def test_cleanlab_fallback(self):
        # Even if cleanlab is missing, wrapper should return results from scratch
        np.random.seed(42)
        y_proba = np.random.random((10, 2))
        y_noisy = np.random.randint(0, 2, 10)
        
        # This should not raise error regardless of cleanlab availability
        issues = find_label_issues_wrapper(y_proba, y_noisy)
        self.assertIsInstance(issues, np.ndarray)

    def test_comparison_structure(self):
        np.random.seed(42)
        y_proba = np.random.random((20, 2))
        y_noisy = np.random.randint(0, 2, 20)
        
        res = compare_implementations(y_proba, y_noisy)
        self.assertIn("status", res)
        self.assertIn("scratch_count", res)

if __name__ == "__main__":
    unittest.main()
