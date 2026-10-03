"""
Unit Tests — Phase 2 Investigation Engines
==========================================

Tests for Phase 2 investigation capabilities:
- Prediction Drift Engine
- Error Taxonomy Engine
- Temporal Analysis Engine
- Threshold Simulator Engine

Run:
    pytest tests/test_phase2_investigation.py -q
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ENG = os.path.join(_ROOT, "engine", "modules")
for p in [_ROOT, _ENG]:
    if p not in sys.path:
        sys.path.insert(0, p)


class TestPredictionDriftEngine(unittest.TestCase):
    """Test Prediction Drift Engine."""

    def setUp(self):
        from prediction_drift_engine import PredictionDriftEngine
        self.engine = PredictionDriftEngine(verbose=False)

    def test_can_run_with_valid_data(self):
        """Test can_run returns True with valid data."""
        can_run, reason = self.engine.can_run(
            y_pred_reference=[0, 1, 0, 1],
            y_pred_current=[0, 1, 0, 1]
        )
        self.assertTrue(can_run)
        self.assertEqual(reason, "Prediction drift analysis can run")

    def test_can_run_missing_reference(self):
        """Test can_run returns False without reference predictions."""
        can_run, reason = self.engine.can_run(
            y_pred_reference=None,
            y_pred_current=[0, 1, 0, 1]
        )
        self.assertFalse(can_run)
        self.assertIn("requires both reference and current predictions", reason)

    def test_can_run_empty_arrays(self):
        """Test can_run returns False with empty arrays."""
        can_run, reason = self.engine.can_run(
            y_pred_reference=[],
            y_pred_current=[]
        )
        self.assertFalse(can_run)
        self.assertIn("non-empty", reason)

    def test_classification_drift_no_drift(self):
        """Test classification drift detection with no drift."""
        result = self.engine.run(
            y_pred_reference=[0, 1, 0, 1, 0, 1],
            y_pred_current=[0, 1, 0, 1, 0, 1],
            task_type="classification"
        )
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("drift_metrics", result["findings"])
        self.assertIn("class_distribution", result["findings"])

    def test_classification_drift_with_drift(self):
        """Test classification drift detection with actual drift."""
        result = self.engine.run(
            y_pred_reference=[0, 0, 0, 0, 0, 0],
            y_pred_current=[1, 1, 1, 1, 1, 1],
            task_type="classification"
        )
        # HIGH severity triggers FAILED status in BaseModule
        self.assertIn(result["status"], ["FAILED", "SUCCESS"])
        self.assertIn("drift_metrics", result["findings"])
        # Should detect significant drift via KS statistic
        self.assertGreater(result["findings"]["drift_metrics"]["ks_statistic"], 0.10)

    def test_regression_drift(self):
        """Test regression drift detection."""
        result = self.engine.run(
            y_pred_reference=[1.0, 2.0, 3.0, 4.0],
            y_pred_current=[1.1, 2.1, 3.1, 4.1],
            task_type="regression"
        )
        self.assertIn(result["status"], ["FAILED", "SUCCESS"])
        self.assertIn("score_distribution", result["findings"])


class TestErrorTaxonomyEngine(unittest.TestCase):
    """Test Error Taxonomy Engine."""

    def setUp(self):
        from error_taxonomy_engine import ErrorTaxonomyEngine
        self.engine = ErrorTaxonomyEngine(verbose=False)

    def test_can_run_with_valid_data(self):
        """Test can_run returns True with valid data."""
        can_run, reason = self.engine.can_run(
            y_true=[0, 1, 0, 1],
            y_pred=[0, 1, 0, 1]
        )
        self.assertTrue(can_run)
        self.assertEqual(reason, "Error taxonomy analysis can run")

    def test_can_run_missing_true_labels(self):
        """Test can_run returns False without true labels."""
        can_run, reason = self.engine.can_run(
            y_true=None,
            y_pred=[0, 1, 0, 1]
        )
        self.assertFalse(can_run)
        self.assertIn("requires both true labels and predictions", reason)

    def test_can_run_mismatched_lengths(self):
        """Test can_run returns False with mismatched lengths."""
        can_run, reason = self.engine.can_run(
            y_true=[0, 1],
            y_pred=[0, 1, 0, 1]
        )
        self.assertFalse(can_run)
        self.assertIn("same length", reason)

    def test_classification_error_taxonomy(self):
        """Test classification error taxonomy."""
        result = self.engine.run(
            y_true=[0, 1, 0, 1, 0, 1],
            y_pred=[0, 1, 1, 1, 0, 0],
            y_prob=np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.1, 0.9], [0.8, 0.2], [0.7, 0.3]]),
            task_type="classification"
        )
        self.assertIn(result["status"], ["FAILED", "SUCCESS"])
        self.assertIn("overall_metrics", result["findings"])
        self.assertIn("error_categories", result["findings"])
        self.assertIn("TP", result["findings"]["overall_metrics"])
        self.assertIn("FP", result["findings"]["overall_metrics"])

    def test_regression_error_taxonomy(self):
        """Test regression error taxonomy."""
        result = self.engine.run(
            y_true=[1.0, 2.0, 3.0, 4.0],
            y_pred=[1.1, 2.2, 2.8, 4.3],
            task_type="regression"
        )
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("overall_metrics", result["findings"])
        self.assertIn("mae", result["findings"]["overall_metrics"])
        self.assertIn("rmse", result["findings"]["overall_metrics"])

    def test_segment_errors(self):
        """Test segment-level error analysis."""
        X = np.array([
            [0, 10],
            [0, 20],
            [1, 10],
            [1, 20],
            [0, 15],
            [1, 25]
        ])
        result = self.engine.run(
            y_true=[0, 1, 0, 1, 0, 1],
            y_pred=[0, 1, 1, 1, 0, 0],
            X=X,
            feature_names=["feature_0", "feature_1"],
            task_type="classification"
        )
        self.assertIn(result["status"], ["FAILED", "SUCCESS"])
        self.assertIn("segment_errors", result["findings"])


class TestTemporalAnalysisEngine(unittest.TestCase):
    """Test Temporal Analysis Engine."""

    def setUp(self):
        from temporal_analysis_engine import TemporalAnalysisEngine
        self.engine = TemporalAnalysisEngine(verbose=False)

    def test_can_run_with_valid_timestamps(self):
        """Test can_run returns True with valid timestamps."""
        can_run, reason = self.engine.can_run(
            timestamps=pd.date_range('2024-01-01', periods=10)
        )
        self.assertTrue(can_run)
        self.assertEqual(reason, "Temporal analysis can run")

    def test_can_run_missing_timestamps(self):
        """Test can_run returns False without timestamps."""
        can_run, reason = self.engine.can_run(timestamps=None)
        self.assertFalse(can_run)
        self.assertIn("requires timestamp data", reason)

    def test_can_run_empty_timestamps(self):
        """Test can_run returns False with empty timestamps."""
        can_run, reason = self.engine.can_run(timestamps=[])
        self.assertFalse(can_run)
        self.assertIn("non-empty", reason)

    def test_temporal_analysis_classification(self):
        """Test temporal analysis for classification."""
        timestamps = pd.date_range('2024-01-01', periods=30, freq='D')
        result = self.engine.run(
            y_true=[0, 1] * 15,
            y_pred=[0, 1] * 15,
            timestamps=timestamps,
            task_type="classification"
        )
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("period_metrics", result["findings"])
        self.assertIn("trends", result["findings"])

    def test_temporal_analysis_regression(self):
        """Test temporal analysis for regression."""
        timestamps = pd.date_range('2024-01-01', periods=30, freq='D')
        result = self.engine.run(
            y_true=np.arange(30),
            y_pred=np.arange(30) + 0.1,
            timestamps=timestamps,
            task_type="regression"
        )
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("period_metrics", result["findings"])

    def test_temporal_analysis_insufficient_periods(self):
        """Test temporal analysis with insufficient time periods."""
        timestamps = pd.date_range('2024-01-01', periods=5, freq='D')
        result = self.engine.run(
            y_true=[0, 1, 0, 1, 0],
            y_pred=[0, 1, 0, 1, 0],
            timestamps=timestamps,
            task_type="classification"
        )
        # Should skip due to insufficient periods
        self.assertIn("status", result["findings"])


class TestThresholdSimulatorEngine(unittest.TestCase):
    """Test Threshold Simulator Engine."""

    def setUp(self):
        from threshold_simulator_engine import ThresholdSimulatorEngine
        self.engine = ThresholdSimulatorEngine(verbose=False)

    def test_can_run_with_valid_data(self):
        """Test can_run returns True with valid data."""
        can_run, reason = self.engine.can_run(
            y_true=[0, 1, 0, 1],
            y_prob=[0.1, 0.9, 0.2, 0.8]
        )
        self.assertTrue(can_run)
        self.assertEqual(reason, "Threshold simulation can run")

    def test_can_run_missing_true_labels(self):
        """Test can_run returns False without true labels."""
        can_run, reason = self.engine.can_run(
            y_true=None,
            y_prob=[0.1, 0.9, 0.2, 0.8]
        )
        self.assertFalse(can_run)
        self.assertIn("requires both true labels and prediction probabilities", reason)

    def test_can_run_missing_probabilities(self):
        """Test can_run returns False without probabilities."""
        can_run, reason = self.engine.can_run(
            y_true=[0, 1, 0, 1],
            y_prob=None
        )
        self.assertFalse(can_run)
        self.assertIn("requires both true labels and prediction probabilities", reason)

    def test_threshold_simulation(self):
        """Test threshold simulation."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.15, 0.85])
        
        result = self.engine.run(y_true, y_prob, thresholds=[0.3, 0.5, 0.7])
        
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("threshold_results", result["findings"])
        self.assertEqual(len(result["findings"]["threshold_results"]), 3)
        self.assertIn("optimal_threshold", result["findings"])

    def test_threshold_simulation_default_thresholds(self):
        """Test threshold simulation with default thresholds."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.15, 0.85])
        
        result = self.engine.run(y_true, y_prob)
        
        self.assertEqual(result["status"], "SUCCESS")
        # Should use 5 default thresholds
        self.assertEqual(len(result["findings"]["threshold_results"]), 5)

    def test_threshold_metrics(self):
        """Test that threshold metrics are calculated correctly."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8])
        
        result = self.engine.run(y_true, y_prob, thresholds=[0.5])
        
        threshold_result = result["findings"]["threshold_results"][0]
        self.assertEqual(threshold_result["threshold"], 0.5)
        self.assertIn("precision", threshold_result)
        self.assertIn("recall", threshold_result)
        self.assertIn("f1", threshold_result)
        self.assertIn("TP", threshold_result)
        self.assertIn("TN", threshold_result)
        self.assertIn("FP", threshold_result)
        self.assertIn("FN", threshold_result)


if __name__ == "__main__":
    unittest.main()
