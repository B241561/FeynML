"""
Confident Learning (CL) Implementation
======================================
Implements Northcutt et al. (2021) "Confident Learning: Estimating Errors in Dataset Labels".

Mathematical Explanation:
-------------------------
1. Thresholds (t_y):
   For each class y, we compute a threshold t_y as the average self-confidence:
   t_y = (1 / |X_y|) * Σ p(y | x) for all x where s_x = y
   where s_x is the noisy label.

2. Confident Joint (C_tilde):
   A matrix where entry (s, y) counts instances x where:
   - Noisy label s_x = s
   - Predicted probability p(y | x) >= t_y
   - y = argmax_j p(j | x) such that p(j | x) >= t_j

3. Joint Distribution (Q):
   Normalize C_tilde such that ΣΣ Q[s,y] = 1.0.
   Q[s,y] = C_tilde[s,y] / (ΣΣ C_tilde[s,y])

4. Label Error Identification:
   Instances are identified as errors if their noisy label s does not match the 
   predicted true label y that satisfies the confident joint criteria.
"""

import numpy as np

def compute_thresholds(y_proba, y_noisy):
    """
    Compute per-class thresholds t_y.
    
    Parameters:
        y_proba: np.ndarray (N, K) - Predicted probabilities
        y_noisy: np.ndarray (N,) - Noisy labels (integers 0 to K-1)
    
    Returns:
        thresholds: np.ndarray (K,)
    """
    num_classes = y_proba.shape[1]
    thresholds = np.zeros(num_classes)
    
    for k in range(num_classes):
        mask = (y_noisy == k)
        if np.any(mask):
            thresholds[k] = np.mean(y_proba[mask, k])
        else:
            thresholds[k] = 0.0 # Or some default
            
    return thresholds

def estimate_joint_distribution(y_proba, y_noisy):
    """
    Estimate the confident joint matrix C_tilde.
    
    Parameters:
        y_proba: np.ndarray (N, K) - Predicted probabilities
        y_noisy: np.ndarray (N,) - Noisy labels
        
    Returns:
        C_tilde: np.ndarray (K, K) - Confident joint matrix
    """
    num_classes = y_proba.shape[1]
    thresholds = compute_thresholds(y_proba, y_noisy)
    C_tilde = np.zeros((num_classes, num_classes))
    
    for i in range(len(y_noisy)):
        s = int(y_noisy[i])
        # Find all classes y where p(y|x) >= threshold t_y
        candidates = np.where(y_proba[i] >= thresholds)[0]
        
        if len(candidates) > 0:
            # Pick the class with highest probability among candidates
            y = candidates[np.argmax(y_proba[i, candidates])]
            C_tilde[s, y] += 1
            
    return C_tilde

def identify_label_errors(y_proba, y_noisy):
    """
    Identify indices of suspected label errors.
    
    Returns:
        error_indices: np.ndarray
    """
    num_classes = y_proba.shape[1]
    thresholds = compute_thresholds(y_proba, y_noisy)
    error_indices = []
    
    for i in range(len(y_noisy)):
        s = int(y_noisy[i])
        # A simple heuristic for identification:
        # If p(s|x) is significantly lower than the threshold t_s,
        # OR if p(y|x) for some y != s is very high.
        
        # CL specific: if p(y|x) >= t_y for some y != s AND y is the argmax
        candidates = np.where(y_proba[i] >= thresholds)[0]
        if len(candidates) > 0:
            y = candidates[np.argmax(y_proba[i, candidates])]
            if y != s:
                error_indices.append(i)
        elif y_proba[i, s] < thresholds[s]:
            # Optional: fallback if no class exceeds threshold
            # but noisy label is very weak
            error_indices.append(i)
                
    return np.array(error_indices)

def label_error_fraction(y_proba, y_noisy):
    """Estimate total noise rate ε."""
    errors = identify_label_errors(y_proba, y_noisy)
    return len(errors) / len(y_noisy)

def rank_by_label_quality(y_proba, y_noisy):
    """
    Rank instances by label quality/confidence.
    Lower score = more likely to be an error.
    """
    # Simple ranking: p(s|x) - max(p(y|x) for y != s)
    num_samples = len(y_noisy)
    scores = np.zeros(num_samples)
    
    for i in range(num_samples):
        s = int(y_noisy[i])
        p_s = y_proba[i, s]
        other_probs = np.delete(y_proba[i], s)
        scores[i] = p_s - np.max(other_probs)
        
    return np.argsort(scores) # Returns indices from worst to best

def noise_transition_matrix(y_proba, y_noisy):
    """
    Estimate noise transition matrix T[i,j] = P(s=j | y=i).
    Estimated from confident joint C_tilde.
    """
    C_tilde = estimate_joint_distribution(y_proba, y_noisy)
    # T[i, j] = C_tilde[j, i] / sum_j(C_tilde[j, i])
    # Note: C_tilde is (noisy, true). T is (true, noisy).
    # So T[true, noisy] = C_tilde[noisy, true] normalized over noisy.
    
    num_classes = C_tilde.shape[0]
    T = np.zeros((num_classes, num_classes))
    
    for i in range(num_classes):
        # sum over all noisy labels j for a fixed true label i
        col_sum = np.sum(C_tilde[:, i])
        if col_sum > 0:
            T[i, :] = C_tilde[:, i] / col_sum
        else:
            T[i, i] = 1.0
            
    return T

def run_verification():
    """Run module verification with synthetic data."""
    print("--- Confident Learning Verification ---")
    np.random.seed(42)
    
    # 100 samples, 3 classes
    N, K = 100, 3
    
    # Generate true labels
    y_true = np.random.randint(0, K, N)
    
    # Generate noisy labels (15% noise)
    y_noisy = y_true.copy()
    noise_mask = np.random.random(N) < 0.15
    y_noisy[noise_mask] = (y_noisy[noise_mask] + 1) % K
    
    # Generate probabilities (simulated model)
    y_proba = np.zeros((N, K))
    for i in range(N):
        # High prob for true class, lower for others
        y_proba[i, y_true[i]] = 0.7 + 0.3 * np.random.random()
        other_indices = [j for j in range(K) if j != y_true[i]]
        remaining = 1.0 - y_proba[i, y_true[i]]
        y_proba[i, other_indices[0]] = remaining * 0.6
        y_proba[i, other_indices[1]] = remaining * 0.4
        
    thresholds = compute_thresholds(y_proba, y_noisy)
    print(f"Thresholds: {thresholds}")
    
    errors = identify_label_errors(y_proba, y_noisy)
    actual_errors = np.where(y_true != y_noisy)[0]
    
    print(f"Detected {len(errors)} errors.")
    print(f"Actual errors injected: {len(actual_errors)}")
    
    intersection = np.intersect1d(errors, actual_errors)
    print(f"Recall: {len(intersection)/len(actual_errors):.2f}")
    
    T = noise_transition_matrix(y_proba, y_noisy)
    print("Estimated Noise Transition Matrix T (True -> Noisy):")
    print(np.round(T, 2))

if __name__ == "__main__":
    run_verification()
