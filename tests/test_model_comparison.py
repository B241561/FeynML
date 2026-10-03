"""
Unit Tests — Model Comparison Engine
====================================

Tests for Model Comparison capability.

Run:
    pytest tests/test_model_comparison.py -q
"""

import sys
import os
import unittest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ENG = os.path.join(_ROOT, "engine", "modules")
for p in [_ROOT, _ENG]:
    if p not in sys.path:
        sys.path.insert(0, p)


class TestModelComparisonEngine(unittest.TestCase):
    """Test Model Comparison Engine."""

    def setUp(self):
        from model_comparison_engine import ModelComparisonEngine
        self.engine = ModelComparisonEngine(verbose=False)

    def test_can_run_with_valid_models(self):
        """Test can_run returns True with valid model reports."""
        model_reports = [
            {"model_name": "ModelA", "task": "classification", "accuracy": 0.85},
            {"model_name": "ModelB", "task": "classification", "accuracy": 0.88}
        ]
        can_run, reason = self.engine.can_run(model_reports=model_reports)
        self.assertTrue(can_run)
        self.assertEqual(reason, "Model comparison can run")

    def test_can_run_with_none_reports(self):
        """Test can_run returns False with None reports."""
        can_run, reason = self.engine.can_run(model_reports=None)
        self.assertFalse(can_run)
        self.assertIn("requires model evaluation reports", reason)

    def test_can_run_with_empty_list(self):
        """Test can_run returns False with empty list."""
        can_run, reason = self.engine.can_run(model_reports=[])
        self.assertFalse(can_run)
        self.assertIn("No model evaluation reports available", reason)

    def test_can_run_with_single_model(self):
        """Test can_run returns False with single model."""
        model_reports = [
            {"model_name": "ModelA", "task": "classification", "accuracy": 0.85}
        ]
        can_run, reason = self.engine.can_run(model_reports=model_reports)
        self.assertFalse(can_run)
        self.assertIn("requires at least two compatible models", reason)

    def test_can_run_with_incompatible_task_types(self):
        """Test can_run returns False with incompatible task types."""
        model_reports = [
            {"model_name": "ModelA", "task": "classification", "accuracy": 0.85},
            {"model_name": "ModelB", "task": "regression", "rmse": 0.10}
        ]
        can_run, reason = self.engine.can_run(model_reports=model_reports)
        self.assertFalse(can_run)
        self.assertIn("Incompatible task types", reason)

    def test_can_run_with_non_list_input(self):
        """Test can_run returns False with non-list input."""
        can_run, reason = self.engine.can_run(model_reports="not a list")
        self.assertFalse(can_run)
        self.assertIn("must be a list", reason)

    def test_can_run_with_invalid_report_type(self):
        """Test can_run returns False with invalid report type."""
        model_reports = [
            {"model_name": "ModelA", "task": "classification", "accuracy": 0.85},
            "not a dict"
        ]
        can_run, reason = self.engine.can_run(model_reports=model_reports)
        self.assertFalse(can_run)
        self.assertIn("not a dictionary", reason)

    def test_basic_classification_comparison(self):
        """Test basic classification model comparison."""
        model_reports = [
            {
                "model_name": "RandomForest",
                "task": "classification",
                "accuracy": 0.84,
                "precision": 0.83,
                "recall": 0.85,
                "f1_score": 0.84,
                "roc_auc": 0.89
            },
            {
                "model_name": "XGBoost",
                "task": "classification",
                "accuracy": 0.88,
                "precision": 0.87,
                "recall": 0.89,
                "f1_score": 0.88,
                "roc_auc": 0.93
            }
        ]
        
        result = self.engine.run(model_reports)
        
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["findings"]["n_models"], 2)
        self.assertEqual(result["findings"]["task_type"], "classification")
        self.assertIn("comparison_table", result["findings"])
        self.assertIn("primary_metric", result["findings"])
        self.assertIn("best_model", result["findings"])

    def test_multiple_classification_models(self):
        """Test comparison with 3+ classification models."""
        model_reports = [
            {
                "model_name": "RandomForest",
                "task": "classification",
                "accuracy": 0.84,
                "precision": 0.83,
                "recall": 0.85,
                "f1_score": 0.84,
                "roc_auc": 0.89
            },
            {
                "model_name": "XGBoost",
                "task": "classification",
                "accuracy": 0.88,
                "precision": 0.87,
                "recall": 0.89,
                "f1_score": 0.88,
                "roc_auc": 0.93
            },
            {
                "model_name": "LogisticRegression",
                "task": "classification",
                "accuracy": 0.81,
                "precision": 0.80,
                "recall": 0.82,
                "f1_score": 0.81,
                "roc_auc": 0.86
            }
        ]
        
        result = self.engine.run(model_reports)
        
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["findings"]["n_models"], 3)
        self.assertEqual(len(result["findings"]["comparison_table"]), 3)

    def test_regression_comparison(self):
        """Test regression model comparison."""
        model_reports = [
            {
                "model_name": "LinearRegression",
                "task": "regression",
                "rmse": 0.15,
                "mae": 0.12,
                "r2": 0.85,
                "mape": 0.10
            },
            {
                "model_name": "RandomForestRegressor",
                "task": "regression",
                "rmse": 0.10,
                "mae": 0.08,
                "r2": 0.92,
                "mape": 0.07
            }
        ]
        
        result = self.engine.run(model_reports)
        
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["findings"]["task_type"], "regression")
        self.assertEqual(result["findings"]["primary_metric"], "r2")
        self.assertIn("comparison_table", result["findings"])

    def test_zero_models_skipped(self):
        """Test that zero models returns SKIPPED."""
        result = self.engine.run([])
        
        self.assertEqual(result["status"], "SKIPPED")
        self.assertIn("No model evaluation reports available", result["findings"]["skip_reason"])

    def test_single_model_skipped(self):
        """Test that single model returns SKIPPED."""
        model_reports = [
            {"model_name": "ModelA", "task": "classification", "accuracy": 0.85}
        ]
        
        result = self.engine.run(model_reports)
        
        self.assertEqual(result["status"], "SKIPPED")
        self.assertIn("requires at least two compatible models", result["findings"]["skip_reason"])

    def test_incompatible_task_types_skipped(self):
        """Test that incompatible task types return SKIPPED."""
        model_reports = [
            {"model_name": "ModelA", "task": "classification", "accuracy": 0.85},
            {"model_name": "ModelB", "task": "regression", "rmse": 0.10}
        ]
        
        result = self.engine.run(model_reports)
        
        self.assertEqual(result["status"], "SKIPPED")
        self.assertIn("Incompatible task types", result["findings"]["skip_reason"])

    def test_missing_model_name_skipped(self):
        """Test that missing model_name returns SKIPPED."""
        model_reports = [
            {"task": "classification", "accuracy": 0.85},
            {"model_name": "ModelB", "task": "classification", "accuracy": 0.88}
        ]
        
        result = self.engine.run(model_reports)
        
        self.assertEqual(result["status"], "SKIPPED")
        self.assertIn("missing 'model_name'", result["findings"]["skip_reason"])

    def test_missing_optional_metrics(self):
        """Test that missing optional metrics don't crash comparison."""
        model_reports = [
            {
                "model_name": "ModelA",
                "task": "classification",
                "accuracy": 0.85,
                "precision": 0.83
            },
            {
                "model_name": "ModelB",
                "task": "classification",
                "accuracy": 0.88,
                "precision": 0.87
            }
        ]
        
        result = self.engine.run(model_reports)
        
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["findings"]["n_models"], 2)

    def test_none_values_handled(self):
        """Test that None values are handled gracefully."""
        model_reports = [
            {
                "model_name": "ModelA",
                "task": "classification",
                "accuracy": 0.85,
                "roc_auc": None
            },
            {
                "model_name": "ModelB",
                "task": "classification",
                "accuracy": 0.88,
                "roc_auc": 0.93
            }
        ]
        
        result = self.engine.run(model_reports)
        
        self.assertEqual(result["status"], "SUCCESS")

    def test_metric_differences_calculated(self):
        """Test that metric differences are calculated."""
        model_reports = [
            {
                "model_name": "ModelA",
                "task": "classification",
                "accuracy": 0.80,
                "precision": 0.75,
                "f1_score": 0.78
            },
            {
                "model_name": "ModelB",
                "task": "classification",
                "accuracy": 0.90,
                "precision": 0.88,
                "f1_score": 0.89
            }
        ]
        
        result = self.engine.run(model_reports)
        
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("metric_differences", result["findings"])
        metric_diffs = result["findings"]["metric_differences"]
        self.assertIn("accuracy", metric_diffs)
        self.assertIn("precision", metric_diffs)
        self.assertIn("f1_score", metric_diffs)

    def test_comparison_table_structure(self):
        """Test that comparison table has correct structure."""
        model_reports = [
            {
                "model_name": "ModelA",
                "task": "classification",
                "accuracy": 0.85,
                "precision": 0.83
            },
            {
                "model_name": "ModelB",
                "task": "classification",
                "accuracy": 0.88,
                "precision": 0.87
            }
        ]
        
        result = self.engine.run(model_reports)
        
        self.assertEqual(result["status"], "SUCCESS")
        table = result["findings"]["comparison_table"]
        self.assertEqual(len(table), 2)
        self.assertIn("model_name", table[0])
        self.assertIn("accuracy", table[0])

    def test_best_model_identification(self):
        """Test that best model is correctly identified."""
        model_reports = [
            {
                "model_name": "ModelA",
                "task": "classification",
                "accuracy": 0.85,
                "roc_auc": 0.89
            },
            {
                "model_name": "ModelB",
                "task": "classification",
                "accuracy": 0.88,
                "roc_auc": 0.93
            }
        ]
        
        result = self.engine.run(model_reports)
        
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["findings"]["best_model"], "ModelB")


if __name__ == "__main__":
    unittest.main()
