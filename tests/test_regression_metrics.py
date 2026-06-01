"""
Unit Tests — Regression Metrics
=================================
Tests every function in scratch/phase2/regression_metrics.py.
Pure stdlib (unittest) — no pytest needed.

Run:
    python tests/test_regression_metrics.py
"""

import sys
import os
import unittest
import math

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_P2   = os.path.join(_ROOT, "scratch", "phase2")
for p in [_ROOT, _P2]:
    if p not in sys.path:
        sys.path.insert(0, p)

from regression_metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r_squared,
    mean_absolute_percentage_error,
    median_absolute_error,
    max_error,
    explained_variance,
)


class TestPerfectPredictions(unittest.TestCase):
    """Perfect predictions → MSE=0, R²=1, MAE=0."""

    def setUp(self):
        self.y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.y_pred = [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_mse_zero(self):
        self.assertAlmostEqual(mean_squared_error(self.y_true, self.y_pred), 0.0)

    def test_rmse_zero(self):
        self.assertAlmostEqual(root_mean_squared_error(self.y_true, self.y_pred), 0.0)

    def test_mae_zero(self):
        self.assertAlmostEqual(mean_absolute_error(self.y_true, self.y_pred), 0.0)

    def test_r2_one(self):
        self.assertAlmostEqual(r_squared(self.y_true, self.y_pred), 1.0, places=5)

    def test_mape_zero(self):
        self.assertAlmostEqual(
            mean_absolute_percentage_error(self.y_true, self.y_pred), 0.0
        )


class TestKnownValues(unittest.TestCase):

    def test_mse_known(self):
        # errors: [1, -1, 2] → squared: [1, 1, 4] → mean: 2.0
        y_true = [3.0, 2.0, 5.0]
        y_pred = [2.0, 3.0, 3.0]
        self.assertAlmostEqual(mean_squared_error(y_true, y_pred), 2.0, places=6)

    def test_rmse_known(self):
        y_true = [3.0, 2.0, 5.0]
        y_pred = [2.0, 3.0, 3.0]
        self.assertAlmostEqual(
            root_mean_squared_error(y_true, y_pred), math.sqrt(2.0), places=5
        )

    def test_mae_known(self):
        # |errors|: [1, 1, 2] → mean: 4/3
        y_true = [3.0, 2.0, 5.0]
        y_pred = [2.0, 3.0, 3.0]
        self.assertAlmostEqual(mean_absolute_error(y_true, y_pred), 4/3, places=5)

    def test_r2_negative(self):
        """Constant prediction can produce R² < 0."""
        y_true = [1.0, 2.0, 3.0]
        y_pred = [0.0, 0.0, 0.0]
        r2 = r_squared(y_true, y_pred)
        self.assertLess(r2, 0.0)

    def test_r2_constant_true(self):
        """All true values the same → R² is undefined, should not crash."""
        y_true = [5.0, 5.0, 5.0]
        y_pred = [5.0, 5.0, 5.0]
        # Should return 1.0 or handle gracefully
        result = r_squared(y_true, y_pred)
        self.assertIsInstance(result, float)


class TestSymmetry(unittest.TestCase):
    """MAE and RMSE should be symmetric w.r.t. sign of error."""

    def test_mae_symmetric(self):
        y_true = [5.0, 3.0]
        y_over = [7.0, 1.0]   # predict high
        y_under = [3.0, 5.0]  # predict low (same absolute error)
        self.assertAlmostEqual(
            mean_absolute_error(y_true, y_over),
            mean_absolute_error(y_true, y_under),
            places=6
        )

    def test_mse_symmetric(self):
        y_true = [5.0, 3.0]
        y_over = [7.0, 1.0]
        y_under = [3.0, 5.0]
        self.assertAlmostEqual(
            mean_squared_error(y_true, y_over),
            mean_squared_error(y_true, y_under),
            places=6
        )


class TestMaxError(unittest.TestCase):

    def test_max_error_positive(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 10.0]
        self.assertAlmostEqual(max_error(y_true, y_pred), 7.0, places=5)

    def test_max_error_negative_diff(self):
        y_true = [10.0, 2.0]
        y_pred = [1.0, 2.0]
        self.assertAlmostEqual(max_error(y_true, y_pred), 9.0, places=5)


class TestMedianAbsoluteError(unittest.TestCase):

    def test_median_odd(self):
        # Sorted abs errors: [0, 1, 2, 3, 4] → median = 2
        y_true = [0.0, 1.0, 2.0, 3.0, 4.0]
        y_pred = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertAlmostEqual(median_absolute_error(y_true, y_pred), 2.0, places=5)

    def test_median_even(self):
        # Abs errors: [1, 1, 3, 3] → median = (1+3)/2 = 2
        y_true = [1.0, 2.0, 4.0, 5.0]
        y_pred = [0.0, 3.0, 1.0, 8.0]
        mae = median_absolute_error(y_true, y_pred)
        self.assertAlmostEqual(mae, 2.0, places=5)


class TestExplainedVariance(unittest.TestCase):

    def test_perfect(self):
        y = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(explained_variance(y, y), 1.0, places=5)

    def test_zero(self):
        """Constant offset → explained_variance = 0."""
        y_true = [1.0, 2.0, 3.0]
        y_pred = [2.0, 3.0, 4.0]   # systematic bias, no residual variance
        ev = explained_variance(y_true, y_pred)
        self.assertAlmostEqual(ev, 1.0, places=4)   # bias corrected → still 1


class TestEdgeCases(unittest.TestCase):

    def test_single_element(self):
        y_true = [5.0]
        y_pred = [3.0]
        self.assertAlmostEqual(mean_absolute_error(y_true, y_pred), 2.0)
        self.assertAlmostEqual(mean_squared_error(y_true, y_pred), 4.0)

    def test_large_values(self):
        y_true = [1e6, 2e6, 3e6]
        y_pred = [1e6, 2e6, 3e6]
        self.assertAlmostEqual(mean_squared_error(y_true, y_pred), 0.0)

    def test_mape_avoids_division_by_zero(self):
        """y_true = 0 should be handled gracefully."""
        y_true = [0.0, 1.0, 2.0]
        y_pred = [0.0, 1.0, 2.0]
        # Should not raise ZeroDivisionError
        result = mean_absolute_percentage_error(y_true, y_pred)
        self.assertIsInstance(result, float)


if __name__ == "__main__":
    unittest.main(verbosity=2)
