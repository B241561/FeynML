"""
Cleanlab Integration Module
===========================
Wrapper around the cleanlab library for label noise detection.
Provides comparison with the scratch implementation in confident_learning.py.
"""

import numpy as np
import os
import sys

# Import scratch implementation for comparison
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from confident_learning import identify_label_errors as scratch_identify_errors

try:
    import cleanlab
    from cleanlab.filter import find_label_issues
    CLEANLAB_AVAILABLE = True
except ImportError:
    CLEANLAB_AVAILABLE = False

def find_label_issues_wrapper(y_proba, y_noisy):
    """
    Find label issue indices using cleanlab.
    
    Parameters:
        y_proba: np.ndarray (N, K) - Predicted probabilities
        y_noisy: np.ndarray (N,) - Noisy labels
        
    Returns:
        issue_indices: np.ndarray
    """
    if not CLEANLAB_AVAILABLE:
        print("[Warning] cleanlab not available. Falling back to scratch implementation.")
        return scratch_identify_errors(y_proba, y_noisy)
    
    # Cleanlab 2.x find_label_issues returns boolean mask or indices depending on parameters
    issues = find_label_issues(
        labels=y_noisy,
        pred_probs=y_proba,
        return_indices_ranked_by="self_confidence"
    )
    return issues

def compare_implementations(y_proba, y_noisy):
    """
    Compare scratch implementation with cleanlab.
    """
    scratch_errors = scratch_identify_errors(y_proba, y_noisy)
    
    if not CLEANLAB_AVAILABLE:
        return {
            "status": "cleanlab_not_available",
            "scratch_count": len(scratch_errors)
        }
        
    cleanlab_errors = find_label_issues_wrapper(y_proba, y_noisy)
    
    # Calculate intersection and Jaccard similarity
    set_scratch = set(scratch_errors)
    set_cleanlab = set(cleanlab_errors)
    
    intersection = set_scratch.intersection(set_cleanlab)
    union = set_scratch.union(set_cleanlab)
    
    jaccard = len(intersection) / len(union) if len(union) > 0 else 1.0
    
    return {
        "status": "success",
        "scratch_count": len(scratch_errors),
        "cleanlab_count": len(cleanlab_errors),
        "intersection_count": len(intersection),
        "jaccard_similarity": jaccard
    }

def label_quality_scores(y_proba, y_noisy):
    """
    Compute per-sample quality scores (0-1).
    Higher score = higher quality.
    """
    if CLEANLAB_AVAILABLE:
        from cleanlab.rank import get_label_quality_scores
        return get_label_quality_scores(y_noisy, y_proba)
    else:
        # Simple quality score: probability of the assigned label
        return np.array([y_proba[i, int(y_noisy[i])] for i in range(len(y_noisy))])

def clean_dataset(X, y_noisy, y_proba, threshold=0.5):
    """
    Return a filtered/cleaned dataset by removing suspected errors.
    """
    error_indices = find_label_issues_wrapper(y_proba, y_noisy)
    
    # If using scores, we could use a threshold. 
    # Here we just remove the identified issues.
    mask = np.ones(len(y_noisy), dtype=bool)
    mask[error_indices] = False
    
    if isinstance(X, np.ndarray):
        return X[mask], y_noisy[mask]
    else:
        # Assume pandas
        return X.iloc[mask], y_noisy[mask]

def run_verification():
    """Run module verification."""
    print("--- Cleanlab Integration Verification ---")
    np.random.seed(42)
    
    N, K = 200, 3
    y_true = np.random.randint(0, K, N)
    y_noisy = y_true.copy()
    noise_mask = np.random.random(N) < 0.2
    y_noisy[noise_mask] = (y_noisy[noise_mask] + 1) % K
    
    y_proba = np.zeros((N, K))
    for i in range(N):
        y_proba[i, y_true[i]] = 0.6 + 0.4 * np.random.random()
        others = [j for j in range(K) if j != y_true[i]]
        remaining = 1.0 - y_proba[i, y_true[i]]
        y_proba[i, others[0]] = remaining * 0.7
        y_proba[i, others[1]] = remaining * 0.3
        
    comparison = compare_implementations(y_proba, y_noisy)
    print(f"Comparison Result: {comparison}")
    
    if CLEANLAB_AVAILABLE:
        scores = label_quality_scores(y_proba, y_noisy)
        print(f"Average label quality score: {np.mean(scores):.4f}")
    else:
        print("Cleanlab not available for quality scores test.")

if __name__ == "__main__":
    run_verification()
