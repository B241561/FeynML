"""
Engine Module — Label Noise Engine
==================================
Production wrapper around scratch/phase4/label_noise/ modules.

Responsibilities:
  • Identify potential label errors in a dataset
  • Estimate noise transition matrices (True -> Noisy)
  • Rank samples by label quality
  • Provide cleaned datasets for training
"""

import sys
import os
import numpy as np
from engine.base_module import BaseModule

_HERE = os.path.dirname(os.path.abspath(__file__))
_P4_LN = os.path.abspath(os.path.join(_HERE, "..", "..", "scratch", "phase4", "label_noise"))
if _P4_LN not in sys.path:
    sys.path.insert(0, _P4_LN)

try:
    from confident_learning import (
        identify_label_errors,
        noise_transition_matrix,
        label_error_fraction,
        rank_by_label_quality
    )
    from asymmetric_noise import detect_asymmetric_noise
    from cleanlab_integration import label_quality_scores, clean_dataset
    _MODULES_LOADED = True
except ImportError as e:
    _MODULES_LOADED = False
    _IMPORT_ERROR = str(e)

class LabelNoiseEngine(BaseModule):
    """
    Production-ready Label Noise audit engine.
    """
    
    def _run(self, y_proba, y_noisy, X=None):
        """
        Run a full label noise audit.
        
        Parameters:
            y_proba: np.ndarray - Predicted probabilities (N, K)
            y_noisy: np.ndarray - Noisy labels (N,)
            X: Optional data for cleaning
            
        Returns:
            dict - Structured results envelope
        """
        if not _MODULES_LOADED:
            self._error(f"Failed to load Label Noise modules: {_IMPORT_ERROR}")
            return self._result({"error": _IMPORT_ERROR}, severity="CRITICAL")

        self._log("Starting Label Noise audit...")
        
        # 1. Identify errors
        error_indices = identify_label_errors(y_proba, y_noisy)
        noise_fraction = label_error_fraction(y_proba, y_noisy)
        
        # 2. Analyze noise structure
        noise_report = detect_asymmetric_noise(y_proba, y_noisy)
        
        # 3. Quality scores
        scores = label_quality_scores(y_proba, y_noisy)
        
        # 4. Collect error details (top 10)
        error_details = []
        for idx in error_indices[:10]:
            s = int(y_noisy[idx])
            y_pred = int(np.argmax(y_proba[idx]))
            noise_prob = float(1 - y_proba[idx, s])  # 1 - confidence in noisy label
            error_details.append({
                "sample_index": int(idx),
                "true_label": int(s),
                "predicted_label": int(y_pred),
                "noise_probability": noise_prob
            })
            
        # 5. Per class noise counts
        num_classes = y_proba.shape[1]
        per_class_noise = []
        for c in range(num_classes):
            class_indices = np.where(y_noisy == c)[0]
            class_errors = np.intersect1d(class_indices, error_indices)
            per_class_noise.append({
                "class_label": int(c),
                "total_samples": len(class_indices),
                "error_samples": len(class_errors),
                "error_rate": len(class_errors) / len(class_indices) if len(class_indices) > 0 else 0.0
            })
        
        findings = {
            "num_errors_detected": len(error_indices),
            "total_samples": len(y_noisy),
            "estimated_noise_fraction": float(noise_fraction),
            "is_asymmetric": noise_report["is_asymmetric"],
            "noise_severity": noise_report["severity"],
            "avg_label_quality": float(np.mean(scores)),
            "error_indices": error_indices.tolist()[:100],
            "error_details": error_details,
            "per_class_noise": per_class_noise
        }
        
        # Determine severity
        severity = "NONE"
        if noise_fraction > 0.05: severity = "LOW"
        if noise_fraction > 0.15: severity = "MEDIUM"
        if noise_fraction > 0.30: severity = "HIGH"
        
        self._log(f"Audit complete. Detected {len(error_indices)} errors.")
        
        return self._result(findings, severity=severity)

    def get_clean_data(self, X, y_noisy, y_proba):
        """Utility to get cleaned dataset."""
        if not _MODULES_LOADED:
            return X, y_noisy
        return clean_dataset(X, y_noisy, y_proba)

def run_verification():
    """Run module verification."""
    np.random.seed(42)
    N, K = 100, 2
    y_true = np.random.randint(0, K, N)
    y_noisy = y_true.copy()
    y_noisy[0:10] = 1 - y_noisy[0:10] # 10% noise
    
    y_proba = np.zeros((N, K))
    for i in range(N):
        y_proba[i, y_true[i]] = 0.9
        y_proba[i, 1-y_true[i]] = 0.1
        
    engine = LabelNoiseEngine()
    result = engine.run(y_proba, y_noisy)
    import json
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    run_verification()
