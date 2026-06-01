"""
Unit Tests — Calibration
==========================
Tests scratch/phase2/calibration.py

Run:
    python tests/test_calibration.py
"""

import sys
import os
import unittest
import random
import math

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_P2   = os.path.join(_ROOT, "scratch", "phase2")
for p in [_ROOT, _P2]:
    if p not in sys.path:
        sys.path.insert(0, p)

from calibration import (
    reliability_curve,
    expected_calibration_error,
    maximum_calibration_error,
    brier_score,
    brier_skill_score,
    platt_scaling,
    temperature_scaling,
    isotonic_regression_calibration,
    compare_calibration,
    calibration_by_group,
    calibration_summary,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_overconfident(n=400, seed=0):
    """Simulate overconfident model: probs near 0/1 but true labels mixed."""
    random.seed(seed)
    y_true = [random.randint(0, 1) for _ in range(n)]
    y_prob = []
    for y in y_true:
        if y == 1:
            p = random.uniform(0.75, 0.99)
        else:
            p = random.uniform(0.01, 0.25)
        y_prob.append(p)
    return y_true, y_prob

def _make_perfect_calibration(n=200, seed=1):
    """Probabilities perfectly match empirical rates by construction."""
    random.seed(seed)
    y_prob = [i / (n - 1) for i in range(n)]
    y_true = [1 if random.random() < p else 0 for p in y_prob]
    return y_true, y_prob


# ─────────────────────────────────────────────────────────────────────────────
# RELIABILITY CURVE
# ─────────────────────────────────────────────────────────────────────────────

class TestReliabilityCurve(unittest.TestCase):

    def test_output_keys(self):
        y_true, y_prob = _make_overconfident()
        curve = reliability_curve(y_true, y_prob, n_bins=10)
        for key in ["bin_edges", "mean_predicted", "fraction_pos", "counts"]:
            self.assertIn(key, curve)

    def test_lengths_match(self):
        y_true, y_prob = _make_overconfident()
        curve = reliability_curve(y_true, y_prob, n_bins=10)
        n = len(curve["bin_edges"])
        self.assertEqual(len(curve["mean_predicted"]), n)
        self.assertEqual(len(curve["fraction_pos"]), n)
        self.assertEqual(len(curve["counts"]), n)

    def test_counts_sum_to_n(self):
        y_true, y_prob = _make_overconfident(n=300)
        curve = reliability_curve(y_true, y_prob, n_bins=10)
        self.assertEqual(sum(curve["counts"]), 300)

    def test_mean_predicted_in_range(self):
        y_true, y_prob = _make_overconfident()
        curve = reliability_curve(y_true, y_prob)
        for mp in curve["mean_predicted"]:
            self.assertGreaterEqual(mp, 0.0)
            self.assertLessEqual(mp, 1.0)

    def test_fraction_pos_in_range(self):
        y_true, y_prob = _make_overconfident()
        curve = reliability_curve(y_true, y_prob)
        for fp in curve["fraction_pos"]:
            self.assertGreaterEqual(fp, 0.0)
            self.assertLessEqual(fp, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# ECE
# ─────────────────────────────────────────────────────────────────────────────

class TestECE(unittest.TestCase):

    def test_ece_range(self):
        y_true, y_prob = _make_overconfident()
        ece = expected_calibration_error(y_true, y_prob)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)

    def test_ece_non_negative(self):
        y_true = [0, 1, 0, 1]
        y_prob  = [0.1, 0.9, 0.2, 0.8]
        self.assertGreaterEqual(expected_calibration_error(y_true, y_prob), 0.0)

    def test_ece_uniform_probs(self):
        """All predicting 0.5 → ECE = |0.5 - base_rate|."""
        n = 100
        y_true = [1] * 30 + [0] * 70
        y_prob  = [0.5] * n
        ece = expected_calibration_error(y_true, y_prob)
        expected = abs(0.5 - 0.30)
        self.assertAlmostEqual(ece, expected, delta=0.05)


# ─────────────────────────────────────────────────────────────────────────────
# MCE
# ─────────────────────────────────────────────────────────────────────────────

class TestMCE(unittest.TestCase):

    def test_mce_geq_ece(self):
        """MCE must be ≥ ECE (worst bin ≥ weighted average)."""
        y_true, y_prob = _make_overconfident()
        ece = expected_calibration_error(y_true, y_prob)
        mce = maximum_calibration_error(y_true, y_prob)
        self.assertGreaterEqual(mce, ece - 1e-9)

    def test_mce_range(self):
        y_true, y_prob = _make_overconfident()
        mce = maximum_calibration_error(y_true, y_prob)
        self.assertGreaterEqual(mce, 0.0)
        self.assertLessEqual(mce, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# BRIER SCORE
# ─────────────────────────────────────────────────────────────────────────────

class TestBrierScore(unittest.TestCase):

    def test_perfect_brier_zero(self):
        y_true = [1, 0, 1, 0]
        y_prob  = [1.0, 0.0, 1.0, 0.0]
        self.assertAlmostEqual(brier_score(y_true, y_prob), 0.0, places=6)

    def test_worst_brier_one(self):
        y_true = [1, 0, 1, 0]
        y_prob  = [0.0, 1.0, 0.0, 1.0]
        self.assertAlmostEqual(brier_score(y_true, y_prob), 1.0, places=6)

    def test_random_brier_half(self):
        """Predicting 0.5 → BS = 0.25."""
        y_true = [1, 1, 0, 0]
        y_prob  = [0.5] * 4
        self.assertAlmostEqual(brier_score(y_true, y_prob), 0.25, places=6)

    def test_brier_range(self):
        y_true, y_prob = _make_overconfident()
        bs = brier_score(y_true, y_prob)
        self.assertGreaterEqual(bs, 0.0)
        self.assertLessEqual(bs, 1.0)

    def test_bss_better_than_climatology(self):
        y_true, y_prob = _make_overconfident()
        bss = brier_skill_score(y_true, y_prob)
        # A model correlated with labels should beat climatology
        self.assertGreater(bss, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# PLATT SCALING
# ─────────────────────────────────────────────────────────────────────────────

class TestPlattScaling(unittest.TestCase):

    def setUp(self):
        random.seed(0)
        n = 300
        self.y_true = [random.randint(0, 1) for _ in range(n)]
        self.scores = [0.1 + 0.8 * y + random.gauss(0, 0.1) for y in self.y_true]

    def test_output_keys(self):
        result = platt_scaling(self.y_true, self.scores)
        for key in ["A", "B", "calibrate", "train_loss"]:
            self.assertIn(key, result)

    def test_calibrate_is_callable(self):
        result = platt_scaling(self.y_true, self.scores)
        probs = result["calibrate"](self.scores[:10])
        self.assertEqual(len(probs), 10)

    def test_output_probs_in_range(self):
        result = platt_scaling(self.y_true, self.scores)
        probs = result["calibrate"](self.scores)
        for p in probs:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_calibration_improves_ece(self):
        half = len(self.y_true) // 2
        model = platt_scaling(self.y_true[:half], self.scores[:half])
        cal_probs = model["calibrate"](self.scores[half:])
        # Convert raw scores to "probabilities" via sigmoid for fair comparison
        raw_probs = [1 / (1 + math.exp(-s)) for s in self.scores[half:]]
        ece_before = expected_calibration_error(self.y_true[half:], raw_probs)
        ece_after  = expected_calibration_error(self.y_true[half:], cal_probs)
        # Platt scaling should not drastically worsen calibration
        self.assertLess(ece_after, ece_before + 0.05)


# ─────────────────────────────────────────────────────────────────────────────
# TEMPERATURE SCALING
# ─────────────────────────────────────────────────────────────────────────────

class TestTemperatureScaling(unittest.TestCase):

    def setUp(self):
        random.seed(2)
        n = 200
        self.y_true = [random.randint(0, 1) for _ in range(n)]
        self.scores = [random.gauss(1.5 * y - 0.75, 0.5) for y in self.y_true]

    def test_output_keys(self):
        result = temperature_scaling(self.y_true, self.scores)
        for key in ["temperature", "calibrate", "train_nll"]:
            self.assertIn(key, result)

    def test_temperature_positive(self):
        result = temperature_scaling(self.y_true, self.scores)
        self.assertGreater(result["temperature"], 0.0)

    def test_calibrate_callable(self):
        result = temperature_scaling(self.y_true, self.scores)
        out = result["calibrate"](self.scores[:5])
        self.assertEqual(len(out), 5)

    def test_output_probs_in_range(self):
        result = temperature_scaling(self.y_true, self.scores)
        probs = result["calibrate"](self.scores)
        for p in probs:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# ISOTONIC REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

class TestIsotonicRegression(unittest.TestCase):

    def setUp(self):
        random.seed(3)
        n = 300
        self.y_true = [random.randint(0, 1) for _ in range(n)]
        self.scores = [0.2 + 0.6 * y + random.gauss(0, 0.1) for y in self.y_true]

    def test_output_keys(self):
        result = isotonic_regression_calibration(self.y_true, self.scores)
        for key in ["breakpoints", "calibrate", "method"]:
            self.assertIn(key, result)

    def test_method_label(self):
        result = isotonic_regression_calibration(self.y_true, self.scores)
        self.assertEqual(result["method"], "isotonic")

    def test_output_probs_in_range(self):
        result = isotonic_regression_calibration(self.y_true, self.scores)
        probs = result["calibrate"](self.scores[:50])
        for p in probs:
            self.assertGreaterEqual(p, -1e-6)
            self.assertLessEqual(p, 1 + 1e-6)

    def test_monotone_outputs(self):
        """Higher input scores should produce ≥ output probs (monotone)."""
        result = isotonic_regression_calibration(self.y_true, self.scores)
        test_x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        probs = result["calibrate"](test_x)
        for i in range(len(probs) - 1):
            self.assertLessEqual(probs[i] - probs[i + 1], 0.05)


# ─────────────────────────────────────────────────────────────────────────────
# COMPARE CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

class TestCompareCalibration(unittest.TestCase):

    def setUp(self):
        random.seed(4)
        n = 300
        self.y_true = [random.randint(0, 1) for _ in range(n)]
        self.raw    = [0.1 + 0.8 * y + random.gauss(0, 0.05) for y in self.y_true]
        # Mock calibrated: nudge toward 0.5
        self.cal    = [0.3 + 0.4 * p for p in self.raw]

    def test_output_keys(self):
        result = compare_calibration(self.y_true, self.raw, self.cal)
        for key in ["raw", "calibrated", "delta_ece", "delta_brier", "method", "improved"]:
            self.assertIn(key, result)

    def test_raw_ece_computed(self):
        result = compare_calibration(self.y_true, self.raw, self.cal)
        self.assertIn("ece", result["raw"])
        self.assertGreaterEqual(result["raw"]["ece"], 0.0)

    def test_improved_flag_consistent(self):
        result = compare_calibration(self.y_true, self.raw, self.cal)
        if result["delta_ece"] > 0:
            self.assertTrue(result["improved"])
        else:
            self.assertFalse(result["improved"])


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION BY GROUP
# ─────────────────────────────────────────────────────────────────────────────

class TestCalibrationByGroup(unittest.TestCase):

    def setUp(self):
        random.seed(5)
        n = 400
        self.y_true = [random.randint(0, 1) for _ in range(n)]
        self.y_prob = [0.3 + 0.5 * y + random.gauss(0, 0.05) for y in self.y_true]
        self.groups = ["A"] * 200 + ["B"] * 200

    def test_both_groups_present(self):
        result = calibration_by_group(self.y_true, self.y_prob, self.groups)
        self.assertIn("A", result)
        self.assertIn("B", result)

    def test_summary_present(self):
        result = calibration_by_group(self.y_true, self.y_prob, self.groups)
        self.assertIn("_summary", result)

    def test_ece_gap_non_negative(self):
        result = calibration_by_group(self.y_true, self.y_prob, self.groups)
        self.assertGreaterEqual(result["_summary"]["ece_gap"], 0.0)

    def test_group_ece_in_range(self):
        result = calibration_by_group(self.y_true, self.y_prob, self.groups)
        for g in ["A", "B"]:
            self.assertGreaterEqual(result[g]["ece"], 0.0)
            self.assertLessEqual(result[g]["ece"], 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

class TestCalibrationSummary(unittest.TestCase):

    def test_output_keys(self):
        y_true, y_prob = _make_overconfident()
        s = calibration_summary(y_true, y_prob, label="test_model")
        for key in ["label", "n", "ece", "mce", "brier", "bss", "severity",
                    "curve", "interpretation"]:
            self.assertIn(key, s)

    def test_label_preserved(self):
        y_true, y_prob = _make_overconfident()
        s = calibration_summary(y_true, y_prob, label="my_model")
        self.assertEqual(s["label"], "my_model")

    def test_severity_valid_value(self):
        y_true, y_prob = _make_overconfident()
        s = calibration_summary(y_true, y_prob)
        self.assertIn(s["severity"], ["EXCELLENT", "GOOD", "MODERATE", "POOR"])

    def test_n_correct(self):
        y_true, y_prob = _make_overconfident(n=250)
        s = calibration_summary(y_true, y_prob)
        self.assertEqual(s["n"], 250)

    def test_perfect_model_excellent(self):
        """A perfectly calibrated model should score EXCELLENT or GOOD."""
        y_true, y_prob = _make_perfect_calibration(n=500)
        s = calibration_summary(y_true, y_prob)
        self.assertIn(s["severity"], ["EXCELLENT", "GOOD"])


# ─────────────────────────────────────────────────────────────────────────────
# EDGE CASES
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases(unittest.TestCase):

    def test_all_positive_labels(self):
        y_true = [1] * 20
        y_prob  = [0.8] * 20
        ece = expected_calibration_error(y_true, y_prob)
        self.assertAlmostEqual(ece, abs(1.0 - 0.8), delta=0.05)

    def test_all_zeros(self):
        y_true = [0] * 20
        y_prob  = [0.1] * 20
        ece = expected_calibration_error(y_true, y_prob)
        self.assertAlmostEqual(ece, 0.1, delta=0.05)

    def test_single_sample(self):
        y_true = [1]
        y_prob  = [0.9]
        bs = brier_score(y_true, y_prob)
        self.assertAlmostEqual(bs, 0.01, places=4)

    def test_platt_single_parameter(self):
        """Platt should converge even on tiny datasets."""
        y_true = [1, 0, 1, 0, 1, 0]
        scores = [2.0, -2.0, 1.5, -1.5, 1.0, -1.0]
        result = platt_scaling(y_true, scores, epochs=500)
        self.assertIn("A", result)

    def test_isotonic_extrapolation(self):
        """Scores outside training range should not crash."""
        y_true = [0, 0, 1, 1]
        scores = [0.3, 0.4, 0.6, 0.7]
        iso = isotonic_regression_calibration(y_true, scores)
        # Extrapolate below and above range
        out = iso["calibrate"]([-1.0, 2.0])
        self.assertEqual(len(out), 2)
        for p in out:
            self.assertGreaterEqual(p, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
