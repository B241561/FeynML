"""
Asymmetric Label Noise Module
=============================
Analysis and simulation of asymmetric label noise (class-dependent flipping).

Theory:
-------
Symmetric Noise: T[i,j] = ε / (K-1) for i != j, and T[i,i] = 1 - ε.
Asymmetric Noise: T[i,j] varies by (i, j).
"""

import numpy as np
import os
import sys

# Import from confident_learning for estimation
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from confident_learning import noise_transition_matrix as estimate_T

def inject_asymmetric_noise(y, T, seed=None):
    """
    Inject asymmetric noise into labels using transition matrix T.
    
    Parameters:
        y: np.ndarray (N,) - Clean labels
        T: np.ndarray (K, K) - Transition matrix T[i,j] = P(noisy=j | true=i)
        seed: int - Random seed
        
    Returns:
        y_noisy: np.ndarray (N,)
    """
    if seed is not None:
        np.random.seed(seed)
        
    y_noisy = y.copy()
    K = T.shape[0]
    
    for i in range(K):
        mask = (y == i)
        n_samples = np.sum(mask)
        if n_samples > 0:
            # Sample new labels based on the i-th row of T
            y_noisy[mask] = np.random.choice(K, size=n_samples, p=T[i])
            
    return y_noisy

def estimate_noise_matrix(y_proba, y_noisy):
    """
    Estimate noise transition matrix T from data.
    Wrapper around confident_learning.noise_transition_matrix.
    """
    return estimate_T(y_proba, y_noisy)

def detect_asymmetric_noise(y_proba, y_noisy, threshold=0.05):
    """
    Detect noise type and severity.
    
    Returns:
        report: dict
    """
    T = estimate_noise_matrix(y_proba, y_noisy)
    K = T.shape[0]
    
    # Calculate off-diagonal variance to check for asymmetry
    off_diagonals = []
    for i in range(K):
        for j in range(K):
            if i != j:
                off_diagonals.append(T[i, j])
                
    off_diag_std = np.std(off_diagonals)
    is_asymmetric = off_diag_std > threshold
    
    # Severity is average error rate
    severity = 1.0 - np.mean(np.diag(T))
    
    return {
        "is_asymmetric": bool(is_asymmetric),
        "off_diag_std": float(off_diag_std),
        "severity": float(severity),
        "transition_matrix": T.tolist()
    }

def noise_matrix_report(T):
    """
    Generate human-readable noise matrix report.
    """
    K = T.shape[0]
    lines = ["--- Noise Transition Matrix Report (True -> Noisy) ---"]
    
    header = "True \\ Noisy | " + " | ".join([f"C{j}" for j in range(K)])
    lines.append(header)
    lines.append("-" * len(header))
    
    for i in range(K):
        row = f"Class {i}     | " + " | ".join([f"{T[i,j]:.2f}" for j in range(K)])
        lines.append(row)
        
    # Analysis
    error_rates = 1.0 - np.diag(T)
    most_noisy_class = np.argmax(error_rates)
    lines.append(f"\nMost noisy class: {most_noisy_class} (Error rate: {error_rates[most_noisy_class]:.2f})")
    
    return "\n".join(lines)

def run_verification():
    """Run module verification."""
    print("--- Asymmetric Noise Verification ---")
    np.random.seed(42)
    
    K = 3
    # Define an asymmetric transition matrix
    # Class 0 often flipped to Class 1
    T_true = np.array([
        [0.8, 0.2, 0.0],
        [0.05, 0.9, 0.05],
        [0.0, 0.1, 0.9]
    ])
    
    N = 1000
    y_true = np.random.randint(0, K, N)
    y_noisy = inject_asymmetric_noise(y_true, T_true, seed=42)
    
    # Simulate perfect probabilities for estimation test
    y_proba = np.zeros((N, K))
    for i in range(N):
        y_proba[i, y_true[i]] = 0.95
        others = [j for j in range(K) if j != y_true[i]]
        y_proba[i, others[0]] = 0.03
        y_proba[i, others[1]] = 0.02
        
    report = detect_asymmetric_noise(y_proba, y_noisy)
    print(f"Is Asymmetric: {report['is_asymmetric']} (std: {report['off_diag_std']:.4f})")
    print(f"Severity: {report['severity']:.4f}")
    
    print("\n" + noise_matrix_report(np.array(report['transition_matrix'])))

if __name__ == "__main__":
    run_verification()
