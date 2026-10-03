"""
Engine Module — Temporal Performance Analysis Engine
=====================================================
Phase 2: Analyzes model behavior over time.

If a valid timestamp is available, analyzes model performance trends
across time periods (daily, weekly, monthly depending on dataset size).

Responsibilities:
  • Track metrics over time (accuracy, F1, precision, recall, etc.)
  • Identify performance degradation patterns
  • Detect prediction distribution shifts over time
  • Surface temporal anomalies
  • Provide evidence for temporal findings

Usage:
    from engine.modules.temporal_analysis_engine import TemporalAnalysisEngine

    tae = TemporalAnalysisEngine()
    result = tae.run(y_true, y_pred, timestamps, y_prob=None, task_type="classification")
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from datetime import datetime

try:
    from base_module import BaseModule
except ImportError:
    from engine.base_module import BaseModule

try:
    from evidence_registry import EvidenceRegistry
except ImportError:
    EvidenceRegistry = None


class TemporalAnalysisEngine(BaseModule):
    """
    Temporal Performance Analysis Engine.
    
    Analyzes model behavior over time to identify degradation patterns,
    performance shifts, and temporal anomalies.
    """
    
    def __init__(self, min_samples_per_period: int = 10, verbose: bool = True):
        """
        Initialize Temporal Analysis Engine.
        
        Args:
            min_samples_per_period: Minimum samples required per time period
            verbose: Enable logging
        """
        super().__init__(verbose=verbose)
        self.min_samples_per_period = min_samples_per_period
        self._evidence_registry = EvidenceRegistry() if EvidenceRegistry is not None else None
    
    def can_run(self, timestamps=None, **kwargs) -> Tuple[bool, str]:
        """
        Check if temporal analysis can run.
        
        Returns:
            Tuple of (can_run: bool, reason: str)
        """
        if timestamps is None:
            return False, "Temporal analysis requires timestamp data"
        
        if len(timestamps) == 0:
            return False, "Temporal analysis requires non-empty timestamp array"
        
        return True, "Temporal analysis can run"
    
    def _run(self, y_true, y_pred, timestamps, y_prob=None, 
             task_type="classification", granularity="auto") -> Dict[str, Any]:
        """
        Run temporal performance analysis.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            timestamps: Timestamp values (datetime objects or strings)
            y_prob: Prediction probabilities (optional)
            task_type: "classification" or "regression"
            granularity: "auto", "daily", "weekly", "monthly"
            
        Returns:
            Structured result envelope with temporal findings
        """
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Convert timestamps to datetime
        timestamps = self._parse_timestamps(timestamps)
        
        n_samples = len(y_true)
        
        self._log(f"Temporal analysis: n={n_samples}, task={task_type}")
        
        findings = {
            "task_type": task_type,
            "n_samples": n_samples,
            "time_range": {
                "start": str(timestamps.min()),
                "end": str(timestamps.max())
            },
            "granularity": granularity,
            "period_metrics": [],
            "trends": {},
            "anomalies": [],
            "severity": "NONE",
            "status": "SUCCESS"
        }
        
        # Determine granularity if auto
        if granularity == "auto":
            granularity = self._determine_granularity(timestamps)
            findings["granularity"] = granularity
        
        # Group by time period
        period_groups = self._group_by_period(timestamps, granularity)
        
        if not period_groups:
            findings["status"] = "SKIPPED"
            findings["skip_reason"] = "Unable to group data by time period"
            return self._result(findings, severity="NONE")
        
        # Calculate metrics per period
        period_metrics = []
        for period_name, indices in period_groups.items():
            if len(indices) < self.min_samples_per_period:
                continue
            
            y_true_period = y_true[indices]
            y_pred_period = y_pred[indices]
            y_prob_period = y_prob[indices] if y_prob is not None else None
            
            metrics = self._calculate_period_metrics(
                y_true_period, y_pred_period, y_prob_period, task_type
            )
            metrics["period"] = period_name
            metrics["sample_count"] = len(indices)
            period_metrics.append(metrics)
        
        findings["period_metrics"] = period_metrics
        
        if len(period_metrics) < 2:
            findings["status"] = "SKIPPED"
            findings["skip_reason"] = f"Insufficient periods with minimum samples ({len(period_metrics)} < 2)"
            return self._result(findings, severity="NONE")
        
        # Analyze trends
        findings["trends"] = self._analyze_trends(period_metrics, task_type)
        
        # Detect anomalies
        findings["anomalies"] = self._detect_anomalies(period_metrics, task_type)
        
        # Determine severity
        if findings["anomalies"]:
            findings["severity"] = "HIGH"
        elif findings["trends"].get("degradation_detected", False):
            findings["severity"] = "MEDIUM"
        else:
            findings["severity"] = "NONE"
        
        # Register evidence if temporal issues detected
        if self._evidence_registry and findings["severity"] != "NONE":
            try:
                evidence_list = []
                if findings["anomalies"]:
                    evidence_list.extend(findings["anomalies"])
                if findings["trends"].get("degradation_detected", False):
                    evidence_list.append(findings["trends"])
                
                self._evidence_registry.register(
                    claim=f"Temporal performance degradation detected",
                    category="temporal_analysis",
                    evidence=evidence_list,
                    source_module='temporal_analysis_engine',
                    confidence=80 if findings["severity"] == "HIGH" else 60
                )
            except Exception:
                pass
        
        return self._result(findings, severity=findings["severity"])
    
    def _parse_timestamps(self, timestamps) -> pd.Series:
        """Parse timestamps to datetime objects."""
        if isinstance(timestamps, pd.Series):
            return pd.to_datetime(timestamps, errors='coerce')
        
        timestamps = np.array(timestamps)
        if timestamps.dtype.kind in ['M', 'm']:
            return pd.to_datetime(timestamps)
        
        return pd.to_datetime(timestamps, errors='coerce')
    
    def _determine_granularity(self, timestamps: pd.Series) -> str:
        """Determine appropriate time granularity based on data span."""
        time_span = (timestamps.max() - timestamps.min()).days
        
        if time_span <= 7:
            return "daily"
        elif time_span <= 30:
            return "daily"
        elif time_span <= 90:
            return "weekly"
        else:
            return "monthly"
    
    def _group_by_period(self, timestamps: pd.Series, granularity: str) -> Dict[str, List[int]]:
        """Group indices by time period."""
        df = pd.DataFrame({"timestamp": timestamps, "index": range(len(timestamps))})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        if granularity == "daily":
            df["period"] = df["timestamp"].dt.strftime("%Y-%m-%d")
        elif granularity == "weekly":
            df["period"] = df["timestamp"].dt.strftime("%Y-W%W")
        elif granularity == "monthly":
            df["period"] = df["timestamp"].dt.strftime("%Y-%m")
        else:
            df["period"] = df["timestamp"].dt.strftime("%Y-%m-%d")
        
        groups = defaultdict(list)
        for period, idx in zip(df["period"], df["index"]):
            groups[period].append(idx)
        
        return dict(groups)
    
    def _calculate_period_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   y_prob: Optional[np.ndarray], task_type: str) -> Dict[str, Any]:
        """Calculate metrics for a single time period."""
        metrics = {}
        
        if task_type == "classification":
            TP = np.sum((y_true == 1) & (y_pred == 1))
            TN = np.sum((y_true == 0) & (y_pred == 0))
            FP = np.sum((y_true == 0) & (y_pred == 1))
            FN = np.sum((y_true == 1) & (y_pred == 0))
            
            total = len(y_true)
            accuracy = (TP + TN) / total if total > 0 else 0.0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics.update({
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "error_rate": float(1.0 - accuracy)
            })
            
            # Prediction distribution
            pred_dist = {
                "class_0_ratio": float(np.mean(y_pred == 0)),
                "class_1_ratio": float(np.mean(y_pred == 1))
            }
            metrics["prediction_distribution"] = pred_dist
            
        else:  # regression
            errors = y_pred - y_true
            abs_errors = np.abs(errors)
            
            metrics.update({
                "mae": float(np.mean(abs_errors)),
                "rmse": float(np.sqrt(np.mean(errors ** 2))),
                "mean_prediction": float(np.mean(y_pred)),
                "mean_actual": float(np.mean(y_true))
            })
        
        return metrics
    
    def _analyze_trends(self, period_metrics: List[Dict], task_type: str) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        trends = {
            "degradation_detected": False,
            "improvement_detected": False,
            "stable": False,
            "trend_direction": "unknown",
            "trend_magnitude": 0.0
        }
        
        if len(period_metrics) < 2:
            return trends
        
        # Extract primary metric
        if task_type == "classification":
            metric_values = [m.get("accuracy", 0.0) for m in period_metrics]
        else:
            metric_values = [1.0 / (m.get("mae", 1.0) + 1e-6) for m in period_metrics]
        
        # Simple trend detection
        first_half = metric_values[:len(metric_metrics)//2]
        second_half = metric_values[len(metric_metrics)//2:]
        
        first_avg = np.mean(first_half)
        second_avg = np.mean(second_half)
        
        trend_magnitude = second_avg - first_avg
        trends["trend_magnitude"] = float(trend_magnitude)
        
        # Determine trend direction
        if abs(trend_magnitude) < 0.02:  # Small threshold for stability
            trends["stable"] = True
            trends["trend_direction"] = "stable"
        elif trend_magnitude < -0.05:
            trends["degradation_detected"] = True
            trends["trend_direction"] = "degrading"
        elif trend_magnitude > 0.05:
            trends["improvement_detected"] = True
            trends["trend_direction"] = "improving"
        else:
            trends["trend_direction"] = "slight_" + ("degradation" if trend_magnitude < 0 else "improvement")
        
        return trends
    
    def _detect_anomalies(self, period_metrics: List[Dict], task_type: str) -> List[Dict]:
        """Detect temporal anomalies in performance."""
        anomalies = []
        
        if len(period_metrics) < 3:
            return anomalies
        
        # Extract primary metric
        if task_type == "classification":
            metric_values = [m.get("accuracy", 0.0) for m in period_metrics]
        else:
            metric_values = [m.get("mae", 0.0) for m in period_metrics]
        
        # Calculate statistics
        mean_val = np.mean(metric_values)
        std_val = np.std(metric_values)
        
        # Detect outliers (2 standard deviations)
        for i, (metrics, val) in enumerate(zip(period_metrics, metric_values)):
            z_score = (val - mean_val) / (std_val + 1e-6)
            
            if abs(z_score) > 2:
                anomaly_type = "performance_spike" if z_score > 0 else "performance_drop"
                anomalies.append({
                    "period": metrics.get("period", f"period_{i}"),
                    "type": anomaly_type,
                    "metric_value": float(val),
                    "z_score": float(z_score),
                    "sample_count": metrics.get("sample_count", 0)
                })
        
        return anomalies


if __name__ == "__main__":
    # Quick verification
    engine = TemporalAnalysisEngine()
    print("TemporalAnalysisEngine loaded successfully.")
