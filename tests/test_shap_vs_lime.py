"""
Unit Tests — SHAP vs LIME agreement (scratch/phase3/shap_vs_lime.py)

Run:
    python tests/test_shap_vs_lime.py
"""

import sys
import os
import unittest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_P3 = os.path.join(_ROOT, "scratch", "phase3")
for p in [_ROOT, _P3]:
    if p not in sys.path:
        sys.path.insert(0, p)

from shap_vs_lime import (
    compare_explanations,
    batch_agreement,
    stability_analysis,
    spearman_rho,
    DECISION_GUIDE,
)


class TestSpearmanRho(unittest.TestCase):

    def test_perfect_positive(self):
        x = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(spearman_rho(x, x), 1.0, places=4)

    def test_perfect_negative(self):
        x = [1.0, 2.0, 3.0, 4.0]
        y = [-1.0, -2.0, -3.0, -4.0]
        self.assertAlmostEqual(spearman_rho(x, y), -1.0, places=4)


class TestCompareExplanations(unittest.TestCase):

    def test_output_keys(self):
        r = compare_explanations([0.5, -0.2, 0.1], [0.4, -0.15, 0.05], ["a", "b", "c"])
        for key in ("spearman_rho", "top3_overlap", "verdict", "top_shap", "top_lime"):
            self.assertIn(key, r)

    def test_strong_agreement_similar_vectors(self):
        shap = [0.8, -0.5, 0.1, 0.02]
        lime = [0.75, -0.48, 0.12, 0.01]
        r = compare_explanations(shap, lime, ["f0", "f1", "f2", "f3"])
        self.assertGreaterEqual(r["spearman_rho"], 0.75)
        self.assertIn(r["verdict"], ("STRONG_AGREEMENT", "MODERATE_AGREEMENT"))

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            compare_explanations([1.0, 2.0], [1.0], ["a", "b"])


class TestBatchAgreement(unittest.TestCase):

    def test_batch_mean_rho(self):
        names = ["a", "b"]
        shap_batch = [[1.0, 0.0], [0.0, 1.0]]
        lime_batch = [[0.9, 0.0], [0.0, 0.9]]
        r = batch_agreement(shap_batch, lime_batch, names)
        self.assertEqual(r["n_instances"], 2)
        self.assertIsNotNone(r["mean_spearman_rho"])
        self.assertEqual(len(r["per_instance"]), 2)


class TestDecisionGuide(unittest.TestCase):

    def test_guide_keys(self):
        self.assertIn("n_features_lte_8", DECISION_GUIDE)
        self.assertIn("recommendation", DECISION_GUIDE["n_features_lte_8"])


class TestStabilityAnalysis(unittest.TestCase):

    def test_lime_stability_runs(self):
        def model(x):
            return float(x[0]) * 0.7 + float(x[1]) * 0.3

        x = [1.0, 2.0]
        bg = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        r = stability_analysis(
            model, x, bg, feature_names=["a", "b"], n_seeds=3, n_samples=40, K=2
        )
        self.assertIn("overall_stable", r)
        self.assertIn("stability_report", r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
