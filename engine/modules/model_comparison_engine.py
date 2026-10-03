"""
Model Comparison Engine
========================

Compares multiple model evaluation results to identify performance differences,
trade-offs, and the best model according to primary metrics.

This engine wraps the existing Evaluator.compare_models() functionality
and integrates it into the FeynML investigation workflow.
"""

try:
    from base_module import BaseModule
except ImportError:
    from engine.base_module import BaseModule

try:
    from evidence_registry import EvidenceRegistry
except ImportError:
    EvidenceRegistry = None

try:
    from evaluator import Evaluator
except ImportError:
    Evaluator = None

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime


class ModelComparisonEngine(BaseModule):
    """
    Model Comparison Engine.
    
    Compares multiple model evaluation results to identify performance differences
    and trade-offs between models.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize Model Comparison Engine.
        
        Args:
            verbose: Enable logging
        """
        super().__init__(verbose=verbose)
        self._evidence_registry = EvidenceRegistry() if EvidenceRegistry is not None else None
        self._evaluator = Evaluator() if Evaluator is not None else None
    
    def can_run(self, model_reports=None, **kwargs) -> Tuple[bool, str]:
        """
        Check if model comparison can run.
        
        Args:
            model_reports: List of model evaluation report dictionaries
        
        Returns:
            Tuple of (can_run: bool, reason: str)
        """
        if model_reports is None:
            return False, "Model comparison requires model evaluation reports"
        
        if not isinstance(model_reports, list):
            return False, "Model reports must be a list"
        
        if len(model_reports) == 0:
            return False, "No model evaluation reports available"
        
        if len(model_reports) == 1:
            return False, "Model comparison requires at least two compatible models"
        
        # Validate that all reports are dictionaries
        for i, report in enumerate(model_reports):
            if not isinstance(report, dict):
                return False, f"Model report at index {i} is not a dictionary"
        
        # Check that all reports have model_name
        for i, report in enumerate(model_reports):
            if "model_name" not in report:
                return False, f"Model report at index {i} missing 'model_name'"
        
        # Check task type compatibility
        task_types = set()
        for report in model_reports:
            task = report.get("task", "classification")
            task_types.add(task)
        
        if len(task_types) > 1:
            return False, f"Incompatible task types: {task_types}. All models must have the same task type."
        
        return True, "Model comparison can run"
    
    def _validate_compatibility(self, model_reports: List[Dict]) -> Tuple[bool, str]:
        """
        Validate that model reports are compatible for comparison.
        
        Args:
            model_reports: List of model evaluation report dictionaries
        
        Returns:
            Tuple of (is_compatible: bool, reason: str)
        """
        if not model_reports:
            return False, "No model reports provided"
        
        # Check that all reports have model_name
        for i, report in enumerate(model_reports):
            if "model_name" not in report:
                return False, f"Model report at index {i} missing 'model_name'"
        
        # Check task type consistency
        task_types = set()
        for report in model_reports:
            task = report.get("task", "classification")
            task_types.add(task)
        
        if len(task_types) > 1:
            return False, f"Incompatible task types: {task_types}"
        
        return True, "Models are compatible for comparison"
    
    def _run(self, model_reports: List[Dict]) -> Dict[str, Any]:
        """
        Run model comparison.
        
        Args:
            model_reports: List of model evaluation report dictionaries
        
        Returns:
            Dictionary with comparison results
        """
        self._log("Starting Model Comparison Engine...")
        
        # Validate compatibility
        is_compatible, compatibility_reason = self._validate_compatibility(model_reports)
        if not is_compatible:
            findings = {
                "status": "SKIPPED",
                "skip_reason": compatibility_reason,
                "severity": "NONE",
                "n_models": len(model_reports),
                "models": [r.get("model_name", "unknown") for r in model_reports]
            }
            return self._result(findings, severity="NONE")
        
        # Use existing Evaluator.compare_models() if available
        if self._evaluator is not None:
            try:
                comparison_result = self._evaluator.compare_models(model_reports)
                
                # Build structured findings
                findings = {
                    "status": "SUCCESS",
                    "severity": "NONE",
                    "task_type": comparison_result.get("task", "classification"),
                    "n_models": comparison_result.get("n_models", len(model_reports)),
                    "models": [r.get("model_name", "unknown") for r in model_reports],
                    "primary_metric": comparison_result.get("primary_metric"),
                    "best_model": comparison_result.get("best_model"),
                    "best_score": comparison_result.get("best_score"),
                    "comparison_table": comparison_result.get("comparison", []),
                    "metric_differences": self._calculate_metric_differences(model_reports, comparison_result)
                }
                
                # Register evidence if meaningful differences found
                if self._evidence_registry and findings["n_models"] >= 2:
                    try:
                        self._register_comparison_evidence(findings, model_reports)
                    except Exception:
                        pass
                
                self._log(f"Done — compared {findings['n_models']} models")
                return self._result(findings, severity="NONE")
                
            except Exception as e:
                self._log(f"Error using Evaluator.compare_models(): {e}")
                # Fall back to manual comparison
        else:
            self._log("Evaluator not available, using manual comparison")
        
        # Manual comparison fallback
        return self._manual_comparison(model_reports)
    
    def _manual_comparison(self, model_reports: List[Dict]) -> Dict[str, Any]:
        """
        Manual model comparison when Evaluator is not available.
        
        Args:
            model_reports: List of model evaluation report dictionaries
        
        Returns:
            Dictionary with comparison results
        """
        task = model_reports[0].get("task", "classification")
        
        # Determine metrics based on task type
        if task == "classification":
            metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "mcc", "brier_score"]
            primary = "roc_auc" if any("roc_auc" in r for r in model_reports) else "f1_score"
        else:
            metric_keys = ["rmse", "mae", "r2", "mape"]
            primary = "r2"
        
        # Build comparison table
        comparison_table = []
        for report in model_reports:
            row = {"model_name": report.get("model_name", "unknown")}
            for key in metric_keys:
                row[key] = report.get(key, "N/A")
            comparison_table.append(row)
        
        # Find best model by primary metric
        best_model = None
        best_score = None
        for report in model_reports:
            score = report.get(primary)
            if isinstance(score, (int, float)):
                if best_score is None or score > best_score:
                    best_score = score
                    best_model = report.get("model_name", "unknown")
        
        findings = {
            "status": "SUCCESS",
            "severity": "NONE",
            "task_type": task,
            "n_models": len(model_reports),
            "models": [r.get("model_name", "unknown") for r in model_reports],
            "primary_metric": primary,
            "best_model": best_model,
            "best_score": best_score,
            "comparison_table": comparison_table,
            "metric_differences": self._calculate_metric_differences(model_reports, {"primary_metric": primary})
        }
        
        # Register evidence
        if self._evidence_registry:
            try:
                self._register_comparison_evidence(findings, model_reports)
            except Exception:
                pass
        
        self._log(f"Done — compared {findings['n_models']} models (manual)")
        return self._result(findings, severity="NONE")
    
    def _calculate_metric_differences(self, model_reports: List[Dict], comparison_result: Dict) -> Dict[str, Any]:
        """
        Calculate metric differences between models.
        
        Args:
            model_reports: List of model evaluation reports
            comparison_result: Result from compare_models
        
        Returns:
            Dictionary with metric differences
        """
        if len(model_reports) < 2:
            return {}
        
        primary = comparison_result.get("primary_metric", "f1_score")
        task = model_reports[0].get("task", "classification")
        
        # Determine metrics to compare
        if task == "classification":
            metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
        else:
            metrics = ["rmse", "mae", "r2"]
        
        differences = {}
        for metric in metrics:
            values = []
            for report in model_reports:
                val = report.get(metric)
                if isinstance(val, (int, float)):
                    values.append((report.get("model_name", "unknown"), val))
            
            if len(values) >= 2:
                # Calculate range
                numeric_values = [v for _, v in values]
                min_val = min(numeric_values)
                max_val = max(numeric_values)
                range_val = max_val - min_val
                
                differences[metric] = {
                    "values": values,
                    "min": min_val,
                    "max": max_val,
                    "range": range_val,
                    "best_model": max(values, key=lambda x: x[1])[0] if metric != "rmse" and metric != "mae" and metric != "mape" else min(values, key=lambda x: x[1])[0]
                }
        
        return differences
    
    def _register_comparison_evidence(self, findings: Dict, model_reports: List[Dict]):
        """
        Register comparison evidence in EvidenceRegistry.
        
        Args:
            findings: Comparison findings
            model_reports: Original model reports
        """
        if not self._evidence_registry:
            return
        
        # Register evidence for each metric with significant differences
        metric_diffs = findings.get("metric_differences", {})
        for metric, diff_info in metric_diffs.items():
            if diff_info.get("range", 0) > 0.05:  # Only register if difference is meaningful
                try:
                    claim = f"Model performance difference in {metric}: range={diff_info['range']:.4f}"
                    self._evidence_registry.register(
                        claim=claim,
                        category="model_comparison",
                        evidence=[diff_info],
                        source_module='model_comparison_engine',
                        confidence=min(100, int(diff_info['range'] * 100))
                    )
                except Exception:
                    pass
        
        # Register best model evidence
        if findings.get("best_model") and findings.get("best_score") is not None:
            try:
                claim = f"Best model by {findings['primary_metric']}: {findings['best_model']} (score={findings['best_score']})"
                self._evidence_registry.register(
                    claim=claim,
                    category="model_comparison",
                    evidence=[{
                        "best_model": findings["best_model"],
                        "primary_metric": findings["primary_metric"],
                        "best_score": findings["best_score"],
                        "n_models": findings["n_models"]
                    }],
                    source_module='model_comparison_engine',
                    confidence=70
                )
            except Exception:
                pass
    
    def run(self, model_reports: List[Dict]) -> Dict:
        """
        Public run method for model comparison.
        
        Args:
            model_reports: List of model evaluation report dictionaries
        
        Returns:
            Dictionary with comparison results
        """
        can_run, reason = self.can_run(model_reports=model_reports)
        if not can_run:
            findings = {
                "status": "SKIPPED",
                "skip_reason": reason,
                "severity": "NONE",
                "n_models": len(model_reports) if model_reports else 0,
                "models": [r.get("model_name", "unknown") for r in model_reports] if model_reports else []
            }
            # Use _result but override status to SKIPPED
            result = self._result(findings, severity="NONE")
            result["status"] = "SKIPPED"
            result["findings"]["status"] = "SKIPPED"
            return result
        
        return self._run(model_reports)
