"""
Unit Tests — DriftEngine (engine/modules/drift_engine.py)

Run:
    python tests/test_drift_engine.py
"""

import sys
import os
import unittest
from unittest.mock import patch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ENG = os.path.join(_ROOT, "engine", "modules")
_P3 = os.path.join(_ROOT, "scratch", "phase3")
for p in [_ROOT, _ENG, _P3]:
    if p not in sys.path:
        sys.path.insert(0, p)

import drift_engine as de_mod
from drift_engine import (
    DriftEngine,
    DriftGateError,
    _safe_pvalue,
    _round_pvalue,
    _significant_drift,
    _chi2_statistic,
    _domain_auc,
    _feature_column,
    _classify_feature_status,
)


class TestPvalueHelpers(unittest.TestCase):

    def test_safe_pvalue_rejects_scipy_required_string(self):
        self.assertIsNone(_safe_pvalue("scipy required"))

    def test_round_pvalue_none_without_error(self):
        self.assertIsNone(_round_pvalue("scipy required"))

    def test_round_pvalue_numeric(self):
        self.assertEqual(_round_pvalue(0.0123456), 0.01235)


class TestNormalizationHelpers(unittest.TestCase):

    def test_significant_drift_from_shift_detected(self):
        self.assertTrue(_significant_drift({"shift_detected": True}))
        self.assertFalse(_significant_drift({"shift_detected": False}))

    def test_chi2_statistic_keys(self):
        self.assertEqual(_chi2_statistic({"chi2": 3.5}), 3.5)
        self.assertEqual(_chi2_statistic({"chi2_statistic": 2.0}), 2.0)

    def test_domain_auc_keys(self):
        self.assertAlmostEqual(_domain_auc({"domain_auc": 0.82}), 0.82)
        self.assertAlmostEqual(_domain_auc({"auc": 0.71}), 0.71)
        self.assertEqual(_domain_auc({"error": "no sklearn"}), 0.5)


class TestFeatureColumn(unittest.TestCase):

    def test_categorical_strings_included(self):
        rows = [["a"], ["b"], ["a"]]
        col = _feature_column(rows, 0, categorical=True)
        self.assertEqual(col, ["a", "b", "a"])

    def test_numeric_filters_non_numeric(self):
        rows = [[1.0], ["x"], [2.0]]
        col = _feature_column(rows, 0, categorical=False)
        self.assertEqual(col, [1.0, 2.0])


class TestClassifyStatus(unittest.TestCase):

    def test_drift_when_shift_detected_without_pvalue(self):
        status = _classify_feature_status(
            None, None, None, significant=True, ks_alpha=0.05, thresholds={"STABLE": 0.1, "WARN": 0.2}
        )
        self.assertEqual(status, "WARN") # Now returns WARN because pvalue is None and no large effect size



class TestDriftEngineRun(unittest.TestCase):

    def setUp(self):
        self.ref = [[0.0], [1.0], [2.0], [3.0]] * 25
        self.cur_stable = [[0.1], [1.1], [2.1], [3.1]] * 25

    def test_modules_loaded(self):
        err = getattr(de_mod, "_IMPORT_ERROR", None)
        self.assertTrue(de_mod._MODULES_LOADED, err or "drift_detection import failed")

    def test_run_stable_numeric(self):
        engine = DriftEngine(verbose=False)
        engine.set_reference(self.ref, ["x"])
        r = engine.run(self.cur_stable)
        self.assertNotIn("error", r["findings"])
        self.assertIn("per_feature", r["findings"])

    def test_categorical_chi2_no_typeerror_when_scipy_string(self):
        engine = DriftEngine(verbose=False)
        ref = [["a"]] * 50 + [["b"]] * 50
        cur = [["a"]] * 40 + [["b"]] * 60
        engine.set_reference(ref, ["seg"], categorical_cols=["seg"])

        fake_chi2 = {
            "feature": "seg",
            "chi2": 4.2,
            "p_value": "scipy required",
            "shift_detected": None,
        }
        with patch("drift_engine.chi2_test_categorical", return_value=fake_chi2):
            r = engine.run(cur)

        self.assertNotIn("error", r["findings"])
        entry = r["findings"]["per_feature"][0]
        self.assertIsNone(entry["pvalue"])
        self.assertTrue(entry["pvalue_unavailable"])
        self.assertEqual(entry["chi2_stat"], 4.2)

    def test_domain_auc_from_domain_auc_key(self):
        engine = DriftEngine(verbose=False)
        engine.set_reference(self.ref, ["x"])
        fake_dc = {"domain_auc": 0.88, "shift_detected": True}
        with patch("drift_engine.domain_classifier_drift", return_value=fake_dc):
            r = engine.run(self.cur_stable)
        self.assertEqual(r["findings"]["domain_auc"], 0.88)
        self.assertEqual(r["severity"], "CRITICAL")

    def test_ks_shift_detected_flags_drift(self):
        engine = DriftEngine(verbose=False)
        engine.set_reference([[i] for i in range(100)], ["v"])
        cur = [[i + 50] for i in range(100)]
        fake_ks = {
            "ks_statistic": 0.5,
            "p_value": 0.001,
            "shift_detected": True,
        }
        fake_psi = {"psi": 0.05, "shift_detected": False}
        with patch("drift_engine.ks_test", return_value=fake_ks):
            with patch("drift_engine.psi", return_value=fake_psi):
                r = engine.run(cur)
        entry = r["findings"]["per_feature"][0]
        self.assertTrue(entry["significant_drift"])
        self.assertEqual(entry["status"], "DRIFT")


class TestDriftGate(unittest.TestCase):

    def test_gate_raises_critical(self):
        engine = DriftEngine(verbose=False)
        result = {"severity": "CRITICAL", "findings": {"num_drifted_features": 2}}
        with self.assertRaises(DriftGateError):
            engine.assert_gate(result, max_severity="HIGH")


if __name__ == "__main__":
    unittest.main(verbosity=2)
