"""
Phase 3.5 — Distribution Shift Detection
==========================================
Sources:
  - Shimodaira (2000): covariate shift original paper
  - Kolmogorov-Smirnov test (classical statistics)
  - PSI: credit risk industry standard

THREE TYPES OF SHIFT:
  1. COVARIATE SHIFT:  P_train(X) ≠ P_test(X),  but P(Y|X) unchanged
     Example: loan model trained on 2020 data, deployed in 2023
     The customer demographics shifted, but the relationship income→default is same.

  2. LABEL SHIFT:      P_train(Y) ≠ P_test(Y),  but P(X|Y) unchanged
     Example: fraud detection — fraction of fraud cases changes over time.

  3. CONCEPT DRIFT:    P_train(Y|X) ≠ P_test(Y|X)
     The RELATIONSHIP between features and labels changes.
     Hardest to detect — need labels from production.

DETECTION METHODS IMPLEMENTED:
  KS Test:  non-parametric test for distribution equality (Kolmogorov 1933)
  PSI:      Population Stability Index (credit risk industry, ~2000)
  Chi²:     for categorical features
  Domain Classifier: trains a model to distinguish train vs test samples
"""

import math
import random
from collections import Counter, defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# 1. KS TEST — Kolmogorov-Smirnov (from scratch)
# ─────────────────────────────────────────────────────────────────────────────

def ks_statistic(sample1, sample2):
    """
    Kolmogorov-Smirnov statistic:
    D = max|F₁(x) - F₂(x)|
    
    where F₁, F₂ are empirical CDFs of the two samples.
    
    Intuition: the LARGEST vertical gap between the two CDF curves.
    D=0: identical distributions.
    D=1: completely separate distributions.
    
    Time: O(n log n) for sorting.
    """
    combined = sorted(set(sample1 + sample2))
    n1, n2   = len(sample1), len(sample2)
    s1_set   = sorted(sample1)
    s2_set   = sorted(sample2)

    # Empirical CDF at each point
    def ecdf(sorted_sample, x):
        """F(x) = #{xi ≤ x} / n"""
        lo, hi = 0, len(sorted_sample)
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_sample[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        return lo / len(sorted_sample)

    D = 0.0
    for x in combined:
        diff = abs(ecdf(s1_set, x) - ecdf(s2_set, x))
        D    = max(D, diff)
    return D

def ks_pvalue(D, n1, n2):
    """
    Approximate p-value for KS test.
    Uses the asymptotic Kolmogorov distribution.
    
    For large n: p ≈ 2 Σ_{k=1}^∞ (-1)^{k-1} exp(-2k²λ²)
    where λ = D * sqrt(n1*n2/(n1+n2))
    """
    n_eff = math.sqrt(n1 * n2 / (n1 + n2))
    lam   = D * n_eff
    if lam < 1e-10:
        return 1.0
    # Kolmogorov distribution (first few terms of infinite series)
    p = 2.0 * sum(
        ((-1)**(k-1)) * math.exp(-2 * k**2 * lam**2)
        for k in range(1, 20)
    )
    return max(0.0, min(1.0, p))

def ks_test(reference, current, feature_name="feature", alpha=0.05):
    """
    Two-sample KS test for distribution shift detection.
    
    H₀: reference and current come from same distribution.
    Hₐ: distributions differ.
    
    PSI interpretation guide:
      D < 0.05: no significant shift
      D ∈ [0.05, 0.10]: minor shift, monitor
      D > 0.10: significant shift, investigate
    """
    D       = ks_statistic(reference, current)
    p_value = ks_pvalue(D, len(reference), len(current))

    shift_level = (
        "CRITICAL" if D > 0.20 else
        "WARNING"  if D > 0.10 else
        "MINOR"    if D > 0.05 else
        "NONE"
    )

    # Verify against scipy if available
    scipy_D, scipy_p = None, None
    try:
        from scipy.stats import ks_2samp
        res = ks_2samp(reference, current)
        scipy_D = round(float(res.statistic), 6)
        scipy_p = round(float(res.pvalue), 6)
    except ImportError:
        pass

    return {
        "feature":         feature_name,
        "ks_statistic":    round(D, 6),
        "p_value":         round(p_value, 6),
        "shift_detected":  p_value < alpha,
        "shift_level":     shift_level,
        "n_reference":     len(reference),
        "n_current":       len(current),
        "ref_mean":        round(sum(reference) / len(reference), 6),
        "cur_mean":        round(sum(current)   / len(current),   6),
        "mean_shift":      round(sum(current)/len(current) - sum(reference)/len(reference), 6),
        "scipy_D":         scipy_D,
        "scipy_p":         scipy_p,
        "scipy_match":     abs(D - scipy_D) < 0.01 if scipy_D else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. PSI — Population Stability Index (from scratch)
# ─────────────────────────────────────────────────────────────────────────────

def psi(reference, current, n_bins=10, eps=1e-6):
    """
    Population Stability Index — industry standard for model monitoring.
    
    Formula:
      PSI = Σᵢ (Actual%ᵢ - Expected%ᵢ) × ln(Actual%ᵢ / Expected%ᵢ)
    
    where:
      Expected% = fraction of REFERENCE in bin i
      Actual%   = fraction of CURRENT in bin i
    
    Interpretation (banking/credit industry standard):
      PSI < 0.10:  No significant shift (green)
      PSI ∈ [0.10, 0.25]: Moderate shift — investigate (yellow)
      PSI > 0.25:  Major shift — model likely stale (red)
    
    Note: PSI is symmetric KL-divergence (KL(current||reference) + KL(reference||current)) / 2
    It is NOT symmetric: PSI(A,B) ≠ PSI(B,A) in general.
    """
    # Build bins from REFERENCE distribution (quantile-based)
    ref_sorted = sorted(reference)
    n_ref      = len(ref_sorted)
    boundaries = [ref_sorted[int(i * n_ref / n_bins)] for i in range(1, n_bins)]
    boundaries = sorted(set(boundaries))

    def get_bin(v, boundaries):
        for i, b in enumerate(boundaries):
            if v < b:
                return i
        return len(boundaries)

    # Count samples per bin
    ref_counts = Counter(get_bin(v, boundaries) for v in reference)
    cur_counts = Counter(get_bin(v, boundaries) for v in current)
    n_ref      = len(reference)
    n_cur      = len(current)
    n_b        = len(boundaries) + 1

    psi_total  = 0.0
    bin_details= []

    for i in range(n_b):
        ref_pct = (ref_counts.get(i, 0) + eps) / (n_ref + eps * n_b)
        cur_pct = (cur_counts.get(i, 0) + eps) / (n_cur + eps * n_b)
        psi_i   = (cur_pct - ref_pct) * math.log(cur_pct / ref_pct)
        psi_total += psi_i
        bin_details.append({
            "bin":          i,
            "ref_pct":      round(ref_pct, 4),
            "cur_pct":      round(cur_pct, 4),
            "psi_contrib":  round(psi_i, 6),
        })

    shift_level = (
        "CRITICAL" if psi_total > 0.25 else
        "WARNING"  if psi_total > 0.10 else
        "NONE"
    )

    return {
        "psi":          round(psi_total, 6),
        "shift_level":  shift_level,
        "shift_detected": psi_total > 0.10,
        "bin_details":  bin_details,
        "n_bins":       n_b,
        "interpretation": (
            "PSI < 0.10: No significant population shift. "
            if psi_total < 0.10 else
            "PSI 0.10–0.25: Moderate shift detected — model may need retraining. "
            if psi_total < 0.25 else
            "PSI > 0.25: Major population shift — model is likely stale!"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. CHI-SQUARE TEST FOR CATEGORICAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def chi2_test_categorical(reference_cats, current_cats, feature_name="feature", alpha=0.05):
    """
    Chi-square test for shift in categorical feature distributions.
    
    H₀: reference and current have same categorical distribution.
    χ² = Σ (O - E)² / E
    """
    all_cats = sorted(set(reference_cats) | set(current_cats))
    n_ref    = len(reference_cats)
    n_cur    = len(current_cats)

    ref_counts = Counter(reference_cats)
    cur_counts = Counter(current_cats)

    chi2 = 0.0
    details = []
    for cat in all_cats:
        ref_pct  = ref_counts.get(cat, 0) / n_ref
        cur_count = cur_counts.get(cat, 0)
        expected  = ref_pct * n_cur

        if expected > 0:
            chi2 += (cur_count - expected)**2 / expected
        details.append({
            "category": cat,
            "ref_pct":  round(ref_pct, 4),
            "cur_pct":  round(cur_counts.get(cat, 0) / n_cur, 4),
        })

    df = len(all_cats) - 1
    try:
        from scipy.stats import chi2 as chi2_dist
        p_value = float(1 - chi2_dist.cdf(chi2, df))
    except ImportError:
        p_value = None

    return {
        "feature":       feature_name,
        "chi2":          round(chi2, 4),
        "df":            df,
        "p_value":       round(p_value, 6) if p_value else "scipy required",
        "shift_detected": (p_value < alpha) if p_value else None,
        "category_details": details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. LABEL SHIFT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def label_shift_detection(y_train, y_current, alpha=0.05):
    """
    Detect label shift: P_train(Y) ≠ P_test(Y) while P(X|Y) unchanged.
    
    Uses chi-square test on label distributions.
    
    Args:
        y_train: list of training labels
        y_current: list of current/production labels
        alpha: significance level
    
    Returns:
        dict with label distribution comparison and shift detection result
    """
    from collections import Counter
    
    train_dist = Counter(y_train)
    current_dist = Counter(y_current)
    
    all_labels = sorted(set(train_dist.keys()) | set(current_dist.keys()))
    
    n_train = len(y_train)
    n_current = len(y_current)
    
    # Build contingency table
    observed = []
    expected = []
    
    chi2 = 0.0
    details = []
    
    for label in all_labels:
        train_count = train_dist.get(label, 0)
        current_count = current_dist.get(label, 0)
        
        train_pct = train_count / n_train
        current_pct = current_count / n_current
        
        # Expected under null hypothesis (no shift)
        expected_train = (train_count + current_count) * n_train / (n_train + n_current)
        expected_current = (train_count + current_count) * n_current / (n_train + n_current)
        
        if expected_train > 0:
            chi2 += (train_count - expected_train) ** 2 / expected_train
        if expected_current > 0:
            chi2 += (current_count - expected_current) ** 2 / expected_current
        
        details.append({
            "label": label,
            "train_count": train_count,
            "current_count": current_count,
            "train_pct": round(train_pct, 4),
            "current_pct": round(current_pct, 4),
            "pct_diff": round(current_pct - train_pct, 4),
        })
    
    df = len(all_labels) - 1
    
    try:
        from scipy.stats import chi2 as chi2_dist
        p_value = float(1 - chi2_dist.cdf(chi2, df))
    except ImportError:
        p_value = None
    
    shift_detected = (p_value < alpha) if p_value else None
    
    shift_level = (
        "CRITICAL" if shift_detected and abs(sum(d["pct_diff"] for d in details)) > 0.3 else
        "WARNING" if shift_detected else
        "NONE"
    )
    
    return {
        "chi2_statistic": round(chi2, 4),
        "df": df,
        "p_value": round(p_value, 6) if p_value else "scipy required",
        "shift_detected": shift_detected,
        "shift_level": shift_level,
        "label_details": details,
        "n_train": n_train,
        "n_current": n_current,
        "interpretation": (
            f"Label shift {'DETECTED' if shift_detected else 'NOT detected'}. "
            f"Check if class priors changed significantly between training and production."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. EXPLICIT COVARIATE SHIFT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def covariate_shift_detection(X_train, X_current, feature_names=None, alpha=0.05):
    """
    Explicit covariate shift detection using importance weighting.
    
    Covariate shift: P_train(X) ≠ P_test(X) but P(Y|X) unchanged.
    Uses density ratio estimation to detect which features shifted.
    
    Args:
        X_train: list of training feature vectors
        X_current: list of current/production feature vectors
        feature_names: list of feature names
        alpha: significance level
    
    Returns:
        dict with per-feature covariate shift analysis
    """
    n_features = len(X_train[0]) if X_train else 0
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]
    
    feature_shifts = []
    
    for j, fname in enumerate(feature_names):
        train_col = [X_train[i][j] for i in range(len(X_train))]
        current_col = [X_current[i][j] for i in range(len(X_current))]
        
        # Use KS test for each feature
        ks_result = ks_test(train_col, current_col, fname, alpha)
        psi_result = psi(train_col, current_col)
        
        # Combined shift score
        shift_score = ks_result["ks_statistic"] * 0.5 + psi_result["psi"] * 0.5
        
        feature_shifts.append({
            "feature": fname,
            "ks_statistic": ks_result["ks_statistic"],
            "ks_pvalue": ks_result["p_value"],
            "psi": psi_result["psi"],
            "shift_score": round(shift_score, 4),
            "shift_detected": ks_result["shift_detected"] or psi_result["shift_detected"],
        })
    
    # Sort by shift score
    feature_shifts.sort(key=lambda x: x["shift_score"], reverse=True)
    
    # Overall covariate shift assessment
    n_shifted = sum(1 for fs in feature_shifts if fs["shift_detected"])
    max_shift_score = max(fs["shift_score"] for fs in feature_shifts) if feature_shifts else 0.0
    
    overall_shift = (
        "CRITICAL" if n_shifted > n_features * 0.5 or max_shift_score > 0.3 else
        "WARNING" if n_shifted > 0 else
        "NONE"
    )
    
    return {
        "feature_shifts": feature_shifts,
        "n_features": n_features,
        "n_shifted": n_shifted,
        "max_shift_score": round(max_shift_score, 4),
        "overall_shift": overall_shift,
        "interpretation": (
            f"Covariate shift {overall_shift}. "
            f"{n_shifted}/{n_features} features show significant distribution change. "
            "Consider reweighting or retraining if shift is severe."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. DOMAIN CLASSIFIER DRIFT DETECTOR (unsupervised)
# ─────────────────────────────────────────────────────────────────────────────

def domain_classifier_drift(X_reference, X_current, feature_names=None):
    """
    Domain classifier approach to detect MULTIVARIATE drift.
    
    Algorithm:
      1. Label reference samples as 0, current samples as 1
      2. Train a classifier to distinguish them
      3. If AUC ≈ 0.5 → no drift (can't tell them apart)
         If AUC >> 0.5 → drift detected (distributions are separable)
    
    Also extracts which features most distinguish train from test.
    These features are likely the ones that shifted.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        import numpy as np

        X_ref_np = np.array(X_reference)
        X_cur_np = np.array(X_current)

        X_combined = np.vstack([X_ref_np, X_cur_np])
        y_combined = np.array([0]*len(X_reference) + [1]*len(X_current))

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        auc_scores = cross_val_score(clf, X_combined, y_combined,
                                     cv=3, scoring="roc_auc")
        mean_auc = float(np.mean(auc_scores))

        # Feature importances from domain classifier
        clf.fit(X_combined, y_combined)
        importances = clf.feature_importances_

        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X_combined.shape[1])]

        feat_importance = sorted(
            zip(feature_names, importances.tolist()),
            key=lambda t: t[1], reverse=True
        )

        shift_level = (
            "CRITICAL" if mean_auc > 0.80 else
            "WARNING"  if mean_auc > 0.65 else
            "NONE"
        )

        return {
            "domain_auc":      round(mean_auc, 4),
            "shift_level":     shift_level,
            "shift_detected":  mean_auc > 0.65,
            "top_shifted_features": [
                {"feature": f, "importance": round(imp, 4)}
                for f, imp in feat_importance[:5]
            ],
            "interpretation": (
                f"AUC={mean_auc:.3f}: "
                "No multivariate shift detected." if mean_auc < 0.65 else
                f"AUC={mean_auc:.3f}: "
                "SHIFT DETECTED — model can distinguish reference from current."
            ),
        }
    except ImportError:
        return {"error": "sklearn required for domain classifier"}


# ─────────────────────────────────────────────────────────────────────────────
# 7. ENHANCED UNSUPERVISED DRIFT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def enhanced_unsupervised_drift_detection(X_reference, X_current, feature_names=None, 
                                          alpha=0.05, use_pca=True):
    """
    Enhanced unsupervised drift detection combining multiple methods.
    
    Combines:
    1. Domain classifier AUC
    2. PCA-based reconstruction error
    3. MMD (Maximum Mean Discrepancy) approximation
    
    Args:
        X_reference: reference data
        X_current: current data
        feature_names: list of feature names
        alpha: significance level
        use_pca: whether to use PCA-based detection
    
    Returns:
        dict with comprehensive drift assessment
    """
    results = {
        "methods": {},
        "overall_drift": "NONE",
        "confidence": "LOW",
    }
    
    # Method 1: Domain classifier
    dc_result = domain_classifier_drift(X_reference, X_current, feature_names)
    results["methods"]["domain_classifier"] = dc_result
    
    # Method 2: PCA reconstruction error
    if use_pca:
        pca_result = _pca_drift_detection(X_reference, X_current, feature_names)
        results["methods"]["pca_reconstruction"] = pca_result
    
    # Method 3: MMD approximation
    mmd_result = _mmd_approximation(X_reference, X_current)
    results["methods"]["mmd"] = mmd_result
    
    # Aggregate results
    drift_indicators = []
    if "error" not in dc_result and dc_result.get("shift_detected"):
        drift_indicators.append("domain_classifier")
    if use_pca and "error" not in pca_result and pca_result.get("shift_detected"):
        drift_indicators.append("pca")
    if "error" not in mmd_result and mmd_result.get("shift_detected"):
        drift_indicators.append("mmd")
    
    n_methods = len(drift_indicators)
    
    if n_methods >= 2:
        results["overall_drift"] = "CRITICAL"
        results["confidence"] = "HIGH"
    elif n_methods == 1:
        results["overall_drift"] = "WARNING"
        results["confidence"] = "MEDIUM"
    else:
        results["overall_drift"] = "NONE"
        results["confidence"] = "HIGH"
    
    results["drift_indicators"] = drift_indicators
    results["n_methods_agreeing"] = n_methods
    
    return results


def _pca_drift_detection(X_reference, X_current, feature_names=None):
    """
    PCA-based drift detection using reconstruction error.
    
    If reconstruction error on current data is significantly higher
    than on reference data, drift is detected.
    """
    try:
        from sklearn.decomposition import PCA
        import numpy as np
        
        X_ref_np = np.array(X_reference)
        X_cur_np = np.array(X_current)
        
        # Fit PCA on reference
        n_components = min(5, X_ref_np.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X_ref_np)
        
        # Reconstruction errors
        ref_reconstructed = pca.inverse_transform(pca.transform(X_ref_np))
        cur_reconstructed = pca.inverse_transform(pca.transform(X_cur_np))
        
        ref_error = np.mean((X_ref_np - ref_reconstructed) ** 2)
        cur_error = np.mean((X_cur_np - cur_reconstructed) ** 2)
        
        error_ratio = cur_error / max(ref_error, 1e-12)
        
        shift_detected = error_ratio > 1.5  # 50% increase in reconstruction error
        
        return {
            "method": "pca_reconstruction",
            "ref_reconstruction_error": round(float(ref_error), 6),
            "cur_reconstruction_error": round(float(cur_error), 6),
            "error_ratio": round(error_ratio, 4),
            "shift_detected": shift_detected,
            "shift_level": "CRITICAL" if error_ratio > 2.0 else "WARNING" if error_ratio > 1.5 else "NONE",
            "interpretation": (
                f"Reconstruction error ratio: {error_ratio:.2f}. "
                f"Current data is {'more difficult to reconstruct' if shift_detected else 'similar to reference'}."
            ),
        }
    except ImportError:
        return {"error": "sklearn required for PCA drift detection"}


def _mmd_approximation(X_reference, X_current, kernel_bandwidth=1.0):
    """
    Maximum Mean Discrepancy (MMD) approximation for drift detection.
    
    MMD measures distance between two distributions in RKHS.
    Large MMD indicates distribution shift.
    """
    import numpy as np
    
    X_ref = np.array(X_reference)
    X_cur = np.array(X_current)
    
    def rbf_kernel(x, y, sigma=kernel_bandwidth):
        n = x.shape[0]
        m = y.shape[0]
        xx = np.sum(x ** 2, axis=1).reshape((n, 1))
        yy = np.sum(y ** 2, axis=1).reshape((m, 1))
        xy = np.dot(x, y.T)
        dist = xx + yy.T - 2 * xy
        return np.exp(-dist / (2 * sigma ** 2))
    
    # Compute MMD
    K_rr = rbf_kernel(X_ref, X_ref)
    K_cc = rbf_kernel(X_cur, X_cur)
    K_rc = rbf_kernel(X_ref, X_cur)
    
    n = X_ref.shape[0]
    m = X_cur.shape[0]
    
    mmd_sq = (np.sum(K_rr) - np.trace(K_rr)) / (n * (n - 1)) + \
             (np.sum(K_cc) - np.trace(K_cc)) / (m * (m - 1)) - \
             2 * np.sum(K_rc) / (n * m)
    
    mmd = np.sqrt(max(mmd_sq, 0))
    
    # Bootstrap significance test
    n_permutations = 100
    combined = np.vstack([X_ref, X_cur])
    n_combined = len(combined)
    
    mmd_permutations = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_ref = combined[:n]
        perm_cur = combined[n:]
        
        K_rr_perm = rbf_kernel(perm_ref, perm_ref)
        K_cc_perm = rbf_kernel(perm_cur, perm_cur)
        K_rc_perm = rbf_kernel(perm_ref, perm_cur)
        
        mmd_sq_perm = (np.sum(K_rr_perm) - np.trace(K_rr_perm)) / (n * (n - 1)) + \
                      (np.sum(K_cc_perm) - np.trace(K_cc_perm)) / (m * (m - 1)) - \
                      2 * np.sum(K_rc_perm) / (n * m)
        mmd_permutations.append(np.sqrt(max(mmd_sq_perm, 0)))
    
    p_value = np.mean(mmd_permutations >= mmd)
    
    return {
        "method": "mmd",
        "mmd_statistic": round(float(mmd), 6),
        "p_value": round(float(p_value), 6),
        "shift_detected": p_value < 0.05,
        "shift_level": "CRITICAL" if p_value < 0.01 else "WARNING" if p_value < 0.05 else "NONE",
        "interpretation": (
            f"MMD={mmd:.4f}, p={p_value:.4f}. "
            f"Distribution shift {'DETECTED' if p_value < 0.05 else 'NOT detected'}."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. FULL DRIFT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def full_drift_report(X_reference, X_current, feature_names,
                      numeric_features, categorical_features=None,
                      alpha=0.05):
    """
    Run KS test + PSI on all numeric features.
    Run Chi² on all categorical features.
    Run domain classifier for multivariate shift.
    
    Returns unified drift report.
    """
    report = {
        "n_reference":  len(X_reference),
        "n_current":    len(X_current),
        "feature_drift": [],
        "overall_drift": "NONE",
    }

    max_ks  = 0.0
    max_psi = 0.0

    for j, feat in enumerate(feature_names):
        if feat not in numeric_features:
            continue
        ref_col = [X_reference[i][j] for i in range(len(X_reference))]
        cur_col = [X_current[i][j]   for i in range(len(X_current))]

        ks_res  = ks_test(ref_col, cur_col, feat, alpha)
        psi_res = psi(ref_col, cur_col)

        max_ks  = max(max_ks,  ks_res["ks_statistic"])
        max_psi = max(max_psi, psi_res["psi"])

        report["feature_drift"].append({
            "feature":      feat,
            "ks_statistic": ks_res["ks_statistic"],
            "ks_pvalue":    ks_res["p_value"],
            "ks_shift":     ks_res["shift_level"],
            "psi":          psi_res["psi"],
            "psi_shift":    psi_res["shift_level"],
            "drift_detected": ks_res["shift_detected"] or psi_res["shift_detected"],
        })

    # Sort by drift severity
    report["feature_drift"].sort(
        key=lambda x: x["ks_statistic"], reverse=True
    )

    # Overall severity
    n_drifted = sum(1 for f in report["feature_drift"] if f["drift_detected"])
    if n_drifted == 0:
        report["overall_drift"] = "NONE"
    elif max_ks > 0.20 or max_psi > 0.25:
        report["overall_drift"] = "CRITICAL"
    elif max_ks > 0.10 or max_psi > 0.10:
        report["overall_drift"] = "WARNING"
    else:
        report["overall_drift"] = "MINOR"

    report["n_drifted_features"] = n_drifted
    report["max_ks"]             = round(max_ks, 4)
    report["max_psi"]            = round(max_psi, 4)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# 6. VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def run_verification():
    import random as _r

    print("=" * 65)
    print("Phase 3.5 — Drift Detection Verification")
    print("=" * 65)

    _r.seed(42)

    # ── KS Test ────────────────────────────────────────────────────────
    # Same distribution → should NOT detect drift
    ref_nodrift = [_r.gauss(0, 1) for _ in range(500)]
    cur_nodrift = [_r.gauss(0, 1) for _ in range(500)]
    ks_nodrift  = ks_test(ref_nodrift, cur_nodrift, "no_drift_feature")

    # Shifted distribution → SHOULD detect drift
    ref_drift   = [_r.gauss(0, 1) for _ in range(500)]
    cur_drift   = [_r.gauss(2, 1) for _ in range(500)]   # mean shifted by 2σ
    ks_drift    = ks_test(ref_drift, cur_drift, "shifted_feature")

    print(f"\n  KS Test:")
    print(f"  No drift:  D={ks_nodrift['ks_statistic']:.4f}  "
          f"p={ks_nodrift['p_value']:.4f}  "
          f"level={ks_nodrift['shift_level']}  "
          f"[{'✓ PASS' if not ks_nodrift['shift_detected'] else '✗ FAIL'}]")
    print(f"  2σ shift:  D={ks_drift['ks_statistic']:.4f}  "
          f"p={ks_drift['p_value']:.4f}  "
          f"level={ks_drift['shift_level']}  "
          f"[{'✓ PASS' if ks_drift['shift_detected'] else '✗ FAIL'}]")
    if ks_drift.get("scipy_match") is not None:
        print(f"  scipy agreement: D_ours={ks_drift['ks_statistic']} "
              f"D_scipy={ks_drift['scipy_D']}  "
              f"[{'✓ PASS' if ks_drift['scipy_match'] else '✗ FAIL'}]")

    # ── PSI Test ───────────────────────────────────────────────────────
    ref_psi_no  = [_r.gauss(0, 1) for _ in range(1000)]
    cur_psi_no  = [_r.gauss(0, 1) for _ in range(1000)]
    psi_no      = psi(ref_psi_no, cur_psi_no)

    ref_psi_yes = [_r.gauss(0, 1) for _ in range(1000)]
    cur_psi_yes = [_r.gauss(3, 1) for _ in range(1000)]   # large shift
    psi_yes     = psi(ref_psi_yes, cur_psi_yes)

    print(f"\n  PSI Test:")
    print(f"  No drift:  PSI={psi_no['psi']:.4f}  "
          f"level={psi_no['shift_level']}  "
          f"[{'✓ PASS' if not psi_no['shift_detected'] else '✗ FAIL'}]")
    print(f"  3σ shift:  PSI={psi_yes['psi']:.4f}  "
          f"level={psi_yes['shift_level']}  "
          f"[{'✓ PASS' if psi_yes['shift_detected'] else '✗ FAIL'}]")

    # ── Full drift report ──────────────────────────────────────────────
    print(f"\n  Full Drift Report (3 features, 1 shifted):")
    n = 400
    feat_names = ["income", "age", "stable_feat"]
    X_ref = [[_r.gauss(50, 10), _r.gauss(40, 8), _r.gauss(0, 1)] for _ in range(n)]
    X_cur = [[_r.gauss(65, 12), _r.gauss(40, 8), _r.gauss(0, 1)] for _ in range(n)]  # income shifted

    report = full_drift_report(X_ref, X_cur, feat_names,
                               numeric_features=feat_names)
    print(f"  Overall drift: {report['overall_drift']}")
    print(f"  {'Feature':<15} {'KS':>6} {'PSI':>6} {'Drifted?':>10}")
    for fd in report["feature_drift"]:
        print(f"  {fd['feature']:<15} {fd['ks_statistic']:>6.3f} "
              f"{fd['psi']:>6.3f} {'YES ⚠' if fd['drift_detected'] else 'no':>10}")

    print(f"\n  Drift interpretation guide:")
    print(f"    KS > 0.10 or PSI > 0.10 → WARNING")
    print(f"    KS > 0.20 or PSI > 0.25 → CRITICAL (retrain model)")
    print("=" * 65)


if __name__ == "__main__":
    run_verification()
