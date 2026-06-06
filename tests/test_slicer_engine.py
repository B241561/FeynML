"""
Unit Tests — SlicerEngine (engine/modules/slicer_engine.py)

Run:
    python tests/test_slicer_engine.py
"""

import sys
import os
import random
import unittest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ENG = os.path.join(_ROOT, "engine", "modules")
_P3 = os.path.join(_ROOT, "scratch", "phase3")
for p in [_ROOT, _ENG, _P3]:
    if p not in sys.path:
        sys.path.insert(0, p)

import slicer_engine as se_mod
from slicer_engine import (
    SlicerEngine,
    SlicerGateError,
    _matrix_to_data_rows,
    _prepare_slice_rows,
    _format_slice_entry,
    _counterpart_mean_loss,
)
from slice_finder import Slice


class TestModuleLoad(unittest.TestCase):

    def test_modules_loaded(self):
        err = getattr(se_mod, "_IMPORT_ERROR", None)
        self.assertTrue(se_mod._MODULES_LOADED, err or "slice_finder import failed")


class TestMatrixHelpers(unittest.TestCase):

    def test_matrix_to_data_rows(self):
        X = [[1.0, "a"], [2.0, "b"]]
        rows = _matrix_to_data_rows(X, ["x0", "cat"])
        self.assertEqual(rows[0], {"x0": 1.0, "cat": "a"})
        self.assertEqual(len(rows), 2)

    def test_prepare_slice_rows_numeric(self):
        X = [[0.0], [1.0], [2.0], [10.0]]
        rows, cols = _prepare_slice_rows(X, ["age"], n_bins=2)
        self.assertEqual(len(rows), 4)
        self.assertIn("age_binned", cols)
        self.assertTrue(all(isinstance(rows[i]["age_binned"], str) for i in range(4)))

    def test_format_slice_entry(self):
        sl = Slice([("seg", "bad")])
        raw = {
            "slice": sl,
            "description": "seg=bad",
            "slice_size": 10,
            "effect_size": 0.9,
            "p_value": 0.01,
            "slice_loss": 0.8,
            "overall_loss": 0.3,
            "loss_gap": 0.5,
        }
        out = _format_slice_entry(raw, n_samples=100)
        self.assertEqual(out["predicate"], {"seg": "bad"})
        self.assertEqual(out["size"], 10)
        self.assertGreater(out["rest_loss"], 0.0)


class TestCounterpartLoss(unittest.TestCase):

    def test_rest_loss_below_slice_mean_when_slice_worse(self):
        # overall=0.5, slice 10 samples at 1.0 -> rest lower
        rest = _counterpart_mean_loss(1.0, 10, 0.5, 100)
        self.assertLess(rest, 1.0)
        self.assertGreater(rest, 0.4)


class TestSlicerEngineRun(unittest.TestCase):

    def _bad_subgroup_data(self, n_per_group=50, seed=0):
        """
        Segment 'bad' has ~90% error rate; 'good' ~10%.
        Non-zero within-group variance is required for Cohen's effect size.
        """
        random.seed(seed)
        X = [["bad"]] * n_per_group + [["good"]] * n_per_group
        y_true = [1] * (2 * n_per_group)
        y_pred = []
        for i in range(n_per_group):
            y_pred.append(0 if random.random() < 0.9 else 1)
        for i in range(n_per_group):
            y_pred.append(1 if random.random() < 0.9 else 0)
        return y_true, y_pred, X, ["segment"]

    def test_run_no_exception(self):
        y_true, y_pred, X, names = self._bad_subgroup_data()
        engine = SlicerEngine(k=3, effect_size_threshold=0.15, verbose=False)
        r = engine.run(y_true, y_pred, X, names)
        self.assertNotIn("error", r["findings"])
        self.assertIn("n_slices_found", r["findings"])

    def test_finds_problematic_slice(self):
        y_true, y_pred, X, names = self._bad_subgroup_data(n_per_group=50)
        engine = SlicerEngine(
            k=5, effect_size_threshold=0.2, alpha=0.05, verbose=False
        )
        r = engine.run(y_true, y_pred, X, names)
        f = r["findings"]
        self.assertGreaterEqual(f["n_slices_found"], 1)
        worst = f["worst_slice"]
        self.assertIsNotNone(worst)
        self.assertGreater(worst["effect_size"], 0.2)
        self.assertIn("segment", worst.get("predicate", {}) or str(worst["description"]))

    def test_numeric_matrix_discretization_path(self):
        random.seed(1)
        n = 80
        X = [[0.0, float(i % 10)] for i in range(40)] + [
            [1.0, float(i % 10)] for i in range(40, 80)
        ]
        y_true = [1] * n
        y_pred = []
        for row in X:
            err_rate = 0.85 if row[0] == 0.0 else 0.15
            y_pred.append(0 if random.random() < err_rate else 1)
        engine = SlicerEngine(k=3, effect_size_threshold=0.15, verbose=False)
        r = engine.run(y_true, y_pred, X, ["group", "idx"])
        self.assertNotIn("error", r["findings"])
        self.assertIn("slices", r["findings"])

    def test_slice_output_schema(self):
        y_true, y_pred, X, names = self._bad_subgroup_data()
        engine = SlicerEngine(k=2, effect_size_threshold=0.1, verbose=False)
        r = engine.run(y_true, y_pred, X, names)
        if not r["findings"]["slices"]:
            self.skipTest("no slices at this threshold")
        s = r["findings"]["slices"][0]
        for key in (
            "predicate", "description", "size", "effect_size",
            "slice_loss", "rest_loss", "p_value", "severity",
        ):
            self.assertIn(key, s)


class TestSlicerGate(unittest.TestCase):

    def test_gate_raises_on_critical(self):
        engine = SlicerEngine(verbose=False)
        result = {
            "severity": "CRITICAL",
            "findings": {
                "n_slices_found": 1,
                "worst_slice": {"description": "x=bad", "effect_size": 1.5},
            },
        }
        with self.assertRaises(SlicerGateError):
            engine.assert_gate(result, max_severity="HIGH")


if __name__ == "__main__":
    unittest.main(verbosity=2)
