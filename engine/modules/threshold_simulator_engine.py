"""
Engine Module — Threshold Simulator Engine
==========================================
Phase 2: Threshold analysis for binary classification models.

Simulates different decision thresholds to help understand the trade-offs
between precision, recall, and other metrics.

This is an analysis/simulation tool only - it does NOT automatically change
the production model threshold.

Responsibilities:
  • Calculate metrics at multiple thresholds
  • Show precision-recall trade-offs
  • Help answer "What changes if the threshold changes?"
  • Provide evidence for threshold selection

Usage:
    from engine.modules.threshold_simulator_engine import ThresholdSimulatorEngine

    tse = ThresholdSimulatorEngine()
    result = tse.run(y_true, y_prob, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7])
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

try:
    from base_module import BaseModule
except ImportError:
    from engine.base_module import BaseModule

try:
    from evidence_registry import EvidenceRegistry
except ImportError:
    EvidenceRegistry = None


class ThresholdSimulatorEngine(BaseModule):
    """
    Threshold Simulator Engine for Binary Classification.
    
    Simulates different decision thresholds to analyze the trade-offs
    between precision, recall, F1, and other metrics.
    """
    
    def __init__(self, default_thresholds: Optional[List[float]] = None, verbose: bool = True):
        """
        Initialize Threshold Simulator Engine.
        
        Args:
            default_thresholds: Default thresholds to analyze if none provided
            verbose: Enable logging
        """
        super().__init__(verbose=verbose)
        self.default_thresholds = default_thresholds or [0.3, 0.4, 0.5, 0.6, 0.7]
        self._evidence_registry = EvidenceRegistry() if EvidenceRegistry is not None else None
    
    def can_run(self, y_true=None, y_prob=None, **kwargs) -> Tuple[bool, str]:
        """
        Check if threshold simulation can run.
        
        Returns:
            Tuple of (can_run: bool, reason: str)
        """
        if y_true is None or y_prob is None:
            return False, "Threshold simulation requires both true labels and prediction probabilities"
        
        if len(y_true) == 0 or len(y_prob) == 0:
            return False, "Threshold simulation requires non-empty arrays"
        
        if len(y_true) != len(y_prob):
            return False, "Threshold simulation requires y_true and y_prob to have the same length"
        
        return True, "Threshold simulation can run"
    
    def _run(self, y_true, y_prob, thresholds=None) -> Dict[str, Any]:
        """
        Run threshold simulation analysis.
        
        Args:
            y_true: True labels (binary: 0 or 1)
            y_prob: Prediction probabilities (positive class)
            thresholds: List of thresholds to analyze (optional)
            
        Returns:
            Structured result envelope with threshold analysis findings
        """
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        
        n_samples = len(y_true)
        
        # Use default thresholds if none provided
        if thresholds is None:
            thresholds = self.default_thresholds
        
        self._log(f"Threshold simulation: n={n_samples}, thresholds={thresholds}")
        
        findings = {
            "n_samples": n_samples,
            "thresholds_analyzed": thresholds,
            "threshold_results": [],
            "optimal_threshold": {},
            "severity": "NONE",
            "status": "SUCCESS"
        }
        
        # Calculate metrics at each threshold
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            metrics = self._calculate_metrics_at_threshold(y_true, y_pred, y_prob, threshold)
            findings["threshold_results"].append(metrics)
        
        # Find optimal threshold by F1 score
        f1_scores = [r["f1"] for r in findings["threshold_results"]]
        if f1_scores:
            best_idx = int(np.argmax(f1_scores))
            best_threshold = findings["threshold_results"][best_idx]
            findings["optimal_threshold"] = {
                "threshold": best_threshold["threshold"],
                "f1": best_threshold["f1"],
                "precision": best_threshold["precision"],
                "recall": best_threshold["recall"],
                "reason": "Optimal by F1 score"
            }
        
        return self._result(findings, severity="NONE")
    
    def _calculate_metrics_at_threshold(self, y_true: np.ndarray, y_pred: np.ndarray,
                                       y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
        """Calculate classification metrics at a specific threshold."""
        # Confusion matrix
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        
        total = len(y_true)
        
        # Metrics
        accuracy = (TP + TN) / total if total > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Additional metrics
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
        
        return {
            "threshold": float(threshold),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "specificity": float(specificity),
            "false_positive_rate": float(fpr),
            "false_negative_rate": float(fnr),
            "TP": int(TP),
            "TN": int(TN),
            "FP": int(FP),
            "FN": int(FN)
        }


if __name__ == "__main__":
    # Quick verification
    engine = ThresholdSimulatorEngine()
    print("ThresholdSimulatorEngine loaded successfully.")
