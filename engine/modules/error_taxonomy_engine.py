"""
Engine Module — Error Taxonomy Engine
====================================
Phase 2: Structured error analysis and classification.

Instead of only reporting overall accuracy, classifies failures into
meaningful categories with supporting evidence.

Responsibilities:
  • Classify errors (FP, FN, high/low confidence, boundary, etc.)
  • Analyze error patterns by segment
  • Identify systematic under/over-prediction (regression)
  • Track error concentration
  • Provide evidence for each error pattern

Usage:
    from engine.modules.error_taxonomy_engine import ErrorTaxonomyEngine

    ete = ErrorTaxonomyEngine()
    result = ete.run(y_true, y_pred, y_prob=None, X=None, feature_names=None, task_type="classification")
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict

try:
    from base_module import BaseModule
except ImportError:
    from engine.base_module import BaseModule

try:
    from evidence_registry import EvidenceRegistry
except ImportError:
    EvidenceRegistry = None


class ErrorTaxonomyEngine(BaseModule):
    """
    Error Taxonomy Analysis Engine.
    
    Classifies model errors into meaningful categories and provides
    supporting evidence for each pattern.
    """
    
    def __init__(self, confidence_threshold: float = 0.5, min_samples: int = 10, verbose: bool = True):
        """
        Initialize Error Taxonomy Engine.
        
        Args:
            confidence_threshold: Threshold for high/low confidence classification
            min_samples: Minimum samples for segment-level analysis
            verbose: Enable logging
        """
        super().__init__(verbose=verbose)
        self.confidence_threshold = confidence_threshold
        self.min_samples = min_samples
        self._evidence_registry = EvidenceRegistry() if EvidenceRegistry is not None else None
    
    def can_run(self, y_true=None, y_pred=None, **kwargs) -> Tuple[bool, str]:
        """
        Check if error taxonomy analysis can run.
        
        Returns:
            Tuple of (can_run: bool, reason: str)
        """
        if y_true is None or y_pred is None:
            return False, "Error taxonomy requires both true labels and predictions"
        
        if len(y_true) == 0 or len(y_pred) == 0:
            return False, "Error taxonomy requires non-empty arrays"
        
        if len(y_true) != len(y_pred):
            return False, "Error taxonomy requires y_true and y_pred to have the same length"
        
        return True, "Error taxonomy analysis can run"
    
    def _run(self, y_true, y_pred, y_prob=None, X=None, feature_names=None, 
             task_type="classification") -> Dict[str, Any]:
        """
        Run error taxonomy analysis.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            X: Feature matrix (optional, for segment analysis)
            feature_names: Feature names (optional)
            task_type: "classification" or "regression"
            
        Returns:
            Structured result envelope with error taxonomy findings
        """
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        n_samples = len(y_true)
        
        self._log(f"Error taxonomy analysis: n={n_samples}, task={task_type}")
        
        findings = {
            "task_type": task_type,
            "n_samples": n_samples,
            "error_categories": {},
            "error_patterns": [],
            "segment_errors": {},
            "severity": "NONE",
            "status": "SUCCESS"
        }
        
        if task_type == "classification":
            findings.update(self._analyze_classification_errors(y_true, y_pred, y_prob, X, feature_names))
        else:
            findings.update(self._analyze_regression_errors(y_true, y_pred, X, feature_names))
        
        # Determine severity based on error rate
        error_rate = findings.get("overall_metrics", {}).get("error_rate", 0.0)
        if error_rate > 0.30:
            findings["severity"] = "HIGH"
        elif error_rate > 0.20:
            findings["severity"] = "MEDIUM"
        else:
            findings["severity"] = "NONE"
        
        # Register evidence if significant errors detected
        if self._evidence_registry and findings["severity"] != "NONE":
            try:
                self._evidence_registry.register(
                    claim=f"Error rate of {error_rate:.2%} detected",
                    category="error_taxonomy",
                    evidence=[findings.get("error_categories", {})],
                    source_module='error_taxonomy_engine',
                    confidence=min(100, int(error_rate * 100))
                )
            except Exception:
                pass
        
        return self._result(findings, severity=findings["severity"])
    
    def _analyze_classification_errors(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                        y_prob: Optional[np.ndarray] = None,
                                        X: Optional[np.ndarray] = None,
                                        feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze classification error patterns."""
        findings = {}
        
        # Confusion matrix components
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        
        total = len(y_true)
        
        # Overall metrics
        accuracy = (TP + TN) / total if total > 0 else 0.0
        error_rate = 1.0 - accuracy
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        findings["overall_metrics"] = {
            "accuracy": float(accuracy),
            "error_rate": float(error_rate),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "TP": int(TP),
            "TN": int(TN),
            "FP": int(FP),
            "FN": int(FN)
        }
        
        # Error categories
        findings["error_categories"] = {
            "false_positives": {
                "count": int(FP),
                "rate": float(FP / (FP + TN) if (FP + TN) > 0 else 0.0),
                "description": "Predicted positive when actual was negative"
            },
            "false_negatives": {
                "count": int(FN),
                "rate": float(FN / (FN + TP) if (FN + TP) > 0 else 0.0),
                "description": "Predicted negative when actual was positive"
            }
        }
        
        # Confidence-based error analysis if probabilities available
        error_patterns = []
        
        if y_prob is not None:
            y_prob = np.array(y_prob)
            if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
                pos_probs = y_prob[:, 1]
            else:
                pos_probs = y_prob.flatten()
            
            # High-confidence errors
            high_conf_mask = pos_probs > self.confidence_threshold
            low_conf_mask = pos_probs <= self.confidence_threshold
            
            # High-confidence false positives
            hc_fp = np.sum((y_true == 0) & (y_pred == 1) & high_conf_mask)
            # High-confidence false negatives
            hc_fn = np.sum((y_true == 1) & (y_pred == 0) & low_conf_mask)
            # Low-confidence errors
            lc_errors = np.sum((y_true != y_pred) & ((pos_probs > 0.4) & (pos_probs < 0.6)))
            
            error_patterns.append({
                "type": "high_confidence_false_positives",
                "count": int(hc_fp),
                "description": "False positives made with high model confidence"
            })
            error_patterns.append({
                "type": "high_confidence_false_negatives",
                "count": int(hc_fn),
                "description": "False negatives made with low model confidence"
            })
            error_patterns.append({
                "type": "boundary_errors",
                "count": int(lc_errors),
                "description": "Errors near decision boundary (confidence ~0.5)"
            })
        
        # Boundary errors (predictions near threshold)
        if y_prob is not None:
            boundary_mask = (pos_probs > 0.45) & (pos_probs < 0.55)
            boundary_errors = np.sum((y_true != y_pred) & boundary_mask)
            error_patterns.append({
                "type": "threshold_boundary_errors",
                "count": int(boundary_errors),
                "description": "Errors occurring near the decision threshold"
            })
        
        findings["error_patterns"] = error_patterns
        
        # Segment-level error analysis if features provided
        if X is not None and feature_names is not None:
            findings["segment_errors"] = self._analyze_segment_errors(
                y_true, y_pred, X, feature_names
            )
        
        return findings
    
    def _analyze_regression_errors(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    X: Optional[np.ndarray] = None,
                                    feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze regression error patterns."""
        findings = {}
        
        # Calculate errors
        errors = y_pred - y_true
        abs_errors = np.abs(errors)
        
        # Overall metrics
        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        
        # Systematic bias
        mean_error = float(np.mean(errors))
        underprediction_rate = float(np.mean(errors < 0))
        overprediction_rate = float(np.mean(errors > 0))
        
        findings["overall_metrics"] = {
            "mae": mae,
            "rmse": rmse,
            "mean_error": mean_error,
            "underprediction_rate": underprediction_rate,
            "overprediction_rate": overprediction_rate
        }
        
        # Error categories
        large_error_threshold = np.percentile(abs_errors, 75) + 1.5 * (np.percentile(abs_errors, 75) - np.percentile(abs_errors, 25))
        large_errors = np.sum(abs_errors > large_error_threshold)
        
        findings["error_categories"] = {
            "large_errors": {
                "count": int(large_errors),
                "threshold": float(large_error_threshold),
                "description": f"Errors exceeding {large_error_threshold:.2f}"
            },
            "systematic_underprediction": {
                "count": int(np.sum(errors < -mae)),
                "description": "Systematic underprediction (error < -MAE)"
            },
            "systematic_overprediction": {
                "count": int(np.sum(errors > mae)),
                "description": "Systematic overprediction (error > MAE)"
            }
        }
        
        # Error patterns
        error_patterns = [
            {
                "type": "systematic_underprediction",
                "count": int(np.sum(errors < 0)),
                "description": "Model consistently predicts lower than actual"
            },
            {
                "type": "systematic_overprediction",
                "count": int(np.sum(errors > 0)),
                "description": "Model consistently predicts higher than actual"
            },
            {
                "type": "outlier_errors",
                "count": int(large_errors),
                "description": "Errors significantly larger than typical"
            }
        ]
        
        findings["error_patterns"] = error_patterns
        
        # Segment-level error analysis if features provided
        if X is not None and feature_names is not None:
            findings["segment_errors"] = self._analyze_segment_errors(
                y_true, y_pred, X, feature_names, task_type="regression"
            )
        
        return findings
    
    def _analyze_segment_errors(self, y_true: np.ndarray, y_pred: np.ndarray,
                                X: np.ndarray, feature_names: List[str],
                                task_type: str = "classification") -> Dict[str, Any]:
        """Analyze errors by feature segments."""
        segment_errors = {}
        
        X = np.array(X)
        
        for i, feature in enumerate(feature_names):
            if i >= X.shape[1]:
                continue
            
            feature_values = X[:, i]
            
            # Skip if too many unique values or all same
            unique_vals = np.unique(feature_values[~np.isnan(feature_values)])
            if len(unique_vals) > 10 or len(unique_vals) < 2:
                continue
            
            # Analyze errors per segment
            feature_segments = {}
            
            for val in unique_vals:
                mask = feature_values == val
                segment_size = np.sum(mask)
                
                if segment_size < self.min_samples:
                    feature_segments[str(val)] = {
                        "status": "SKIPPED",
                        "reason": f"Insufficient sample size ({segment_size} < {self.min_samples})"
                    }
                    continue
                
                y_true_seg = y_true[mask]
                y_pred_seg = y_pred[mask]
                
                if task_type == "classification":
                    error_rate = 1.0 - np.mean(y_true_seg == y_pred_seg)
                else:
                    error_rate = float(np.mean(np.abs(y_pred_seg - y_true_seg)))
                
                feature_segments[str(val)] = {
                    "status": "ANALYZED",
                    "sample_count": int(segment_size),
                    "error_rate": float(error_rate)
                }
            
            if feature_segments:
                segment_errors[feature] = feature_segments
        
        return segment_errors


if __name__ == "__main__":
    # Quick verification
    engine = ErrorTaxonomyEngine()
    print("ErrorTaxonomyEngine loaded successfully.")
