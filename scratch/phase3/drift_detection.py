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
# 4. DOMAIN CLASSIFIER DRIFT DETECTOR (unsupervised)
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
