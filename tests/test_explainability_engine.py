"""
Unit Tests — ExplainabilityEngine (engine/modules/explainability_engine.py)

Run:
    python tests/test_explainability_engine.py
"""

import sys
import os
import unittest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ENG = os.path.join(_ROOT, "engine", "modules")
_P3 = os.path.join(_ROOT, "scratch", "phase3")
for p in [_ROOT, _ENG, _P3]:
    if p not in sys.path:
        sys.path.insert(0, p)

import explainability_engine as ee_mod
from explainability_engine import (
    ExplainabilityEngine,
    ExplainabilityGateError,
    _lime_feature_importances,
    _lime_local_r2,
)


class TestModuleLoad(unittest.TestCase):

    def test_modules_loaded(self):
        err = getattr(ee_mod, "_IMPORT_ERROR", None)
        self.assertTrue(ee_mod._MODULES_LOADED, err or "Phase 3 imports failed")


class TestLimeAdapters(unittest.TestCase):

    def test_importances_from_explanation(self):
        result = {
            "explanation": [
                {"feature": "a", "weight": 0.5},
                {"feature": "c", "weight": -0.2},
            ],
            "local_fidelity_r2": 0.85,
        }
        imp = _lime_feature_importances(result, ["a", "b", "c"])
        self.assertEqual(imp["a"], 0.5)
        self.assertEqual(imp["b"], 0.0)
        self.assertEqual(imp["c"], -0.2)
        self.assertEqual(_lime_local_r2(result), 0.85)

    def test_local_r2_legacy_key(self):
        self.assertEqual(_lime_local_r2({"local_r2": 0.42}), 0.42)


class TestExplainabilityEngineRun(unittest.TestCase):

    def setUp(self):
        self.bg = [[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [0.5, 2.0]]
        self.names = ["x0", "x1"]
        self.x = [1.0, 2.0]

    def _linear_model(self, x):
        return float(x[0]) * 0.6 + float(x[1]) * 0.4

    def test_shap_method(self):
        engine = ExplainabilityEngine(method="shap", verbose=False)
        r = engine.run(self._linear_model, self.x, self.bg, self.names, n_samples=80)
        self.assertNotEqual(r["severity"], "CRITICAL")
        self.assertIn("shap_values", r["findings"])
        self.assertNotIn("error", r["findings"])

    def test_lime_method(self):
        engine = ExplainabilityEngine(method="lime", verbose=False)
        r = engine.run(self._linear_model, self.x, self.bg, self.names, n_samples=80)
        self.assertNotEqual(r["severity"], "CRITICAL")
        self.assertIn("lime_values", r["findings"])
        self.assertIn("lime_r2", r["findings"])
        self.assertIsNotNone(r["findings"]["lime_r2"])
        self.assertEqual(set(r["findings"]["lime_values"].keys()), set(self.names))

    def test_both_method_agreement(self):
        engine = ExplainabilityEngine(method="both", verbose=False)
        r = engine.run(self._linear_model, self.x, self.bg, self.names, n_samples=100)
        self.assertNotEqual(r["severity"], "CRITICAL")
        f = r["findings"]
        self.assertIn("shap_values", f)
        self.assertIn("lime_values", f)
        self.assertIn("agreement", f)
        self.assertIn("spearman_rho", f["agreement"])
        self.assertIn("verdict", f["agreement"])

    def test_top5_features_populated(self):
        engine = ExplainabilityEngine(method="shap", verbose=False)
        r = engine.run(self._linear_model, self.x, self.bg, self.names, n_samples=50)
        top5 = r["findings"].get("top5_features", [])
        self.assertGreater(len(top5), 0)
        self.assertIn("feature", top5[0])


class TestExplainabilityGate(unittest.TestCase):

    def test_gate_passes_high_agreement(self):
        engine = ExplainabilityEngine(verbose=False)
        result = {
            "severity": "NONE",
            "findings": {"agreement": {"spearman_rho": 0.95}},
        }
        engine.assert_gate(result, min_spearman=0.6)

    def test_gate_raises_low_agreement(self):
        engine = ExplainabilityEngine(verbose=False)
        result = {
            "severity": "NONE",
            "findings": {"agreement": {"spearman_rho": 0.1}},
        }
        with self.assertRaises(ExplainabilityGateError):
            engine.assert_gate(result, min_spearman=0.6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
