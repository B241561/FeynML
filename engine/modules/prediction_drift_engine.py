"""
Engine Module — Prediction Drift Engine
========================================
Phase 2: Analyzes changes in prediction distributions over time.

Distinguishes between:
- Data drift (feature distribution changes)
- Prediction drift (prediction output changes)
- Performance degradation (metric changes)

Responsibilities:
  • Analyze prediction distribution changes
  • Track class proportion shifts (classification)
  • Track prediction score distribution changes
  • Compare reference vs current predictions
  • Return drift metrics with evidence

Usage:
    from engine.modules.prediction_drift_engine import PredictionDriftEngine

    pde = PredictionDriftEngine()
    result = pde.run(y_pred_reference, y_pred_current, y_prob_current=None, task_type="classification")
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter

try:
    from base_module import BaseModule
except ImportError:
    from engine.base_module import BaseModule

try:
    from evidence_registry import EvidenceRegistry
except ImportError:
    EvidenceRegistry = None


def _psi(expected: List[float], actual: List[float], n_bins: int = 10, eps: float = 1e-4) -> float:
    """
    Population Stability Index.
    PSI < 0.10  → stable
    PSI 0.10–0.25 → slight shift
    PSI > 0.25  → significant shift
    """
    if not expected or not actual:
        return 0.0
    
    lo = min(min(expected), min(actual))
    hi = max(max(expected), max(actual))
    step = (hi - lo) / n_bins if hi > lo else 1.0

    exp_bins = [0] * n_bins
    act_bins = [0] * n_bins
    for v in expected:
        b = min(int((v - lo) / step), n_bins - 1)
        exp_bins[b] += 1
    for v in actual:
        b = min(int((v - lo) / step), n_bins - 1)
        act_bins[b] += 1

    ne, na = len(expected), len(actual)
    psi = 0.0
    for e, a in zip(exp_bins, act_bins):
        pe = max(e / ne, eps)
        pa = max(a / na, eps)
        psi += (pa - pe) * np.log(pa / pe)
    return round(psi, 6)


def _ks_statistic(a: List[float], b: List[float]) -> float:
    """Kolmogorov–Smirnov statistic between two 1-D samples."""
    if not a or not b:
        return 0.0
    
    combined = sorted(set(a + b))
    na, nb = len(a), len(b)
    ca, cb = Counter(a), Counter(b)
    cum_a = cum_b = 0.0
    max_diff = 0.0
    for val in combined:
        cum_a += ca.get(val, 0) / na
        cum_b += cb.get(val, 0) / nb
        max_diff = max(max_diff, abs(cum_a - cum_b))
    return round(max_diff, 6)


def _kl_divergence(p: List[float], q: List[float], eps: float = 1e-10) -> float:
    """Kullback-Leibler divergence between two probability distributions."""
    if len(p) != len(q):
        return 0.0
    
    p = np.array(p) + eps
    q = np.array(q) + eps
    p = p / p.sum()
    q = q / q.sum()
    
    return float(np.sum(p * np.log(p / q)))


class PredictionDriftEngine(BaseModule):
    """
    Prediction Drift Analysis Engine.
    
    Analyzes changes in prediction outputs over time, distinguishing
    prediction drift from data drift and performance degradation.
    """
    
    def __init__(self, psi_threshold: float = 0.10, ks_threshold: float = 0.10, verbose: bool = True):
        """
        Initialize Prediction Drift Engine.
        
        Args:
            psi_threshold: PSI threshold for drift detection
            ks_threshold: KS statistic threshold for drift detection
            verbose: Enable logging
        """
        super().__init__(verbose=verbose)
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self._evidence_registry = EvidenceRegistry() if EvidenceRegistry is not None else None
    
    def can_run(self, y_pred_reference=None, y_pred_current=None, y_prob_current=None, **kwargs) -> Tuple[bool, str]:
        """
        Check if prediction drift analysis can run.
        
        Returns:
            Tuple of (can_run: bool, reason: str)
        """
        if y_pred_reference is None or y_pred_current is None:
            return False, "Prediction drift requires both reference and current predictions"
        
        if len(y_pred_reference) == 0 or len(y_pred_current) == 0:
            return False, "Prediction drift requires non-empty prediction arrays"
        
        return True, "Prediction drift analysis can run"
    
    def _run(self, y_pred_reference, y_pred_current, y_prob_current=None, 
             task_type="classification", reference_name="reference", current_name="current") -> Dict[str, Any]:
        """
        Run prediction drift analysis.
        
        Args:
            y_pred_reference: Reference predictions (baseline)
            y_pred_current: Current predictions (to compare)
            y_prob_current: Current prediction probabilities (optional)
            task_type: "classification" or "regression"
            reference_name: Name for reference dataset
            current_name: Name for current dataset
            
        Returns:
            Structured result envelope with drift findings
        """
        # Convert to numpy arrays if needed
        y_pred_ref = np.array(y_pred_reference) if not isinstance(y_pred_reference, np.ndarray) else y_pred_reference
        y_pred_cur = np.array(y_pred_current) if not isinstance(y_pred_current, np.ndarray) else y_pred_current
        
        n_ref = len(y_pred_ref)
        n_cur = len(y_pred_cur)
        
        self._log(f"Prediction drift analysis: {reference_name} (n={n_ref}) vs {current_name} (n={n_cur})")
        
        findings = {
            "task_type": task_type,
            "reference": {
                "name": reference_name,
                "n_samples": n_ref
            },
            "current": {
                "name": current_name,
                "n_samples": n_cur
            },
            "drift_metrics": {},
            "class_distribution": {},
            "score_distribution": {},
            "severity": "NONE",
            "status": "SUCCESS"
        }
        
        if task_type == "classification":
            findings.update(self._analyze_classification_drift(y_pred_ref, y_pred_cur, y_prob_current))
        else:
            findings.update(self._analyze_regression_drift(y_pred_ref, y_pred_cur))
        
        # Determine overall severity
        drift_metrics = findings.get("drift_metrics", {})
        psi_val = drift_metrics.get("psi", 0.0)
        ks_val = drift_metrics.get("ks_statistic", 0.0)
        
        if psi_val > 0.25 or ks_val > 0.20:
            findings["severity"] = "HIGH"
        elif psi_val > self.psi_threshold or ks_val > self.ks_threshold:
            findings["severity"] = "MEDIUM"
        else:
            findings["severity"] = "NONE"
        
        # Register evidence if drift detected
        if self._evidence_registry and findings["severity"] != "NONE":
            try:
                self._evidence_registry.register(
                    claim=f"Prediction drift detected (PSI={psi_val:.4f}, KS={ks_val:.4f})",
                    category="prediction_drift",
                    evidence=[{
                        "psi": psi_val,
                        "ks_statistic": ks_val,
                        "reference_samples": n_ref,
                        "current_samples": n_cur
                    }],
                    source_module='prediction_drift_engine',
                    confidence=min(100, int((psi_val / 0.25) * 100)) if psi_val > 0 else 0
                )
            except Exception:
                pass
        
        return self._result(findings, severity=findings["severity"])
    
    def _analyze_classification_drift(self, y_pred_ref: np.ndarray, y_pred_cur: np.ndarray, 
                                       y_prob_cur: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze classification prediction drift."""
        findings = {}
        
        # Class distribution analysis
        ref_counts = Counter(y_pred_ref)
        cur_counts = Counter(y_pred_cur)
        
        all_classes = sorted(set(ref_counts.keys()) | set(cur_counts.keys()))
        
        ref_dist = [ref_counts.get(c, 0) / len(y_pred_ref) for c in all_classes]
        cur_dist = [cur_counts.get(c, 0) / len(y_pred_cur) for c in all_classes]
        
        findings["class_distribution"] = {
            "reference": {str(c): float(ref_counts.get(c, 0)) for c in all_classes},
            "current": {str(c): float(cur_counts.get(c, 0)) for c in all_classes},
            "reference_proportions": {str(c): float(p) for c, p in zip(all_classes, ref_dist)},
            "current_proportions": {str(c): float(p) for c, p in zip(all_classes, cur_dist)},
            "classes": [str(c) for c in all_classes]
        }
        
        # Calculate drift metrics on class proportions
        psi_val = _psi(ref_dist, cur_dist)
        ks_val = _ks_statistic(y_pred_ref.tolist(), y_pred_cur.tolist())
        kl_val = _kl_divergence(ref_dist, cur_dist)
        
        findings["drift_metrics"] = {
            "psi": psi_val,
            "ks_statistic": ks_val,
            "kl_divergence": kl_val,
            "psi_threshold": self.psi_threshold,
            "ks_threshold": self.ks_threshold
        }
        
        # Score distribution analysis if probabilities available
        if y_prob_cur is not None:
            y_prob_cur = np.array(y_prob_cur)
            if y_prob_cur.ndim == 2 and y_prob_cur.shape[1] >= 2:
                # Use positive class probability
                pos_probs = y_prob_cur[:, 1]
            else:
                pos_probs = y_prob_cur.flatten()
            
            findings["score_distribution"] = {
                "mean": float(np.mean(pos_probs)),
                "std": float(np.std(pos_probs)),
                "min": float(np.min(pos_probs)),
                "max": float(np.max(pos_probs)),
                "median": float(np.median(pos_probs))
            }
        
        return findings
    
    def _analyze_regression_drift(self, y_pred_ref: np.ndarray, y_pred_cur: np.ndarray) -> Dict[str, Any]:
        """Analyze regression prediction drift."""
        findings = {}
        
        # Summary statistics
        ref_stats = {
            "mean": float(np.mean(y_pred_ref)),
            "std": float(np.std(y_pred_ref)),
            "min": float(np.min(y_pred_ref)),
            "max": float(np.max(y_pred_ref)),
            "median": float(np.median(y_pred_ref))
        }
        
        cur_stats = {
            "mean": float(np.mean(y_pred_cur)),
            "std": float(np.std(y_pred_cur)),
            "min": float(np.min(y_pred_cur)),
            "max": float(np.max(y_pred_cur)),
            "median": float(np.median(y_pred_cur))
        }
        
        findings["score_distribution"] = {
            "reference": ref_stats,
            "current": cur_stats,
            "mean_shift": float(cur_stats["mean"] - ref_stats["mean"]),
            "std_shift": float(cur_stats["std"] - ref_stats["std"])
        }
        
        # Calculate drift metrics
        psi_val = _psi(y_pred_ref.tolist(), y_pred_cur.tolist())
        ks_val = _ks_statistic(y_pred_ref.tolist(), y_pred_cur.tolist())
        
        findings["drift_metrics"] = {
            "psi": psi_val,
            "ks_statistic": ks_val,
            "psi_threshold": self.psi_threshold,
            "ks_threshold": self.ks_threshold
        }
        
        return findings


if __name__ == "__main__":
    # Quick verification
    engine = PredictionDriftEngine()
    print("PredictionDriftEngine loaded successfully.")
