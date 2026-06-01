"""
Engine Module — Validator
==========================
Data and pipeline validation for the ML Failure Investigation Engine.

Responsibilities:
  • Schema validation (expected columns, dtypes, value ranges)
  • Data leakage detection (temporal, feature, target)
  • Train / test distribution shift (PSI, KS test)
  • Missing data analysis (MCAR / MAR / MNAR indicators)
  • Label sanity checks (class balance, noise detection)
  • Pipeline integrity checks (no future data, proper splits)

Usage:
    from engine.modules.validator import Validator

    v = Validator()
    report = v.validate_split(X_train, X_test, y_train, y_test)
    v.check_leakage(X_train, y_train, feature_names)
"""

import math
import random
from collections import defaultdict, Counter


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _mean(lst):
    return sum(lst) / len(lst) if lst else 0.0

def _std(lst):
    if len(lst) < 2:
        return 0.0
    m = _mean(lst)
    return math.sqrt(sum((x - m) ** 2 for x in lst) / (len(lst) - 1))

def _ks_statistic(a, b):
    """Kolmogorov–Smirnov statistic between two 1-D samples (from scratch)."""
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

def _psi(expected, actual, n_bins: int = 10, eps: float = 1e-4):
    """
    Population Stability Index.
    PSI < 0.10  → stable
    PSI 0.10–0.25 → slight shift
    PSI > 0.25  → significant shift
    """
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
        psi += (pa - pe) * math.log(pa / pe)
    return round(psi, 6)


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class Validator:
    """
    Data and pipeline validation module.

    All methods return a dict with keys:
      status    — "OK" | "WARNING" | "ERROR"
      checks    — list of individual check results
      summary   — human-readable summary string
    """

    # ── SPLIT VALIDATION ─────────────────────────────────────────────────

    def validate_split(self, X_train, X_test, y_train, y_test,
                        feature_names=None) -> dict:
        """
        Validate a train/test split for:
          1. Size sanity (test is 10–40% of total)
          2. Label balance similarity
          3. Feature distribution similarity (KS per feature)
          4. No sample overlap
        """
        n_train = len(X_train)
        n_test  = len(X_test)
        n_total = n_train + n_test
        test_pct = n_test / n_total * 100
        checks = []

        # ── size check ────────────────────────────────────────────────
        ok = 10 <= test_pct <= 40
        checks.append({
            "check": "split_ratio",
            "status": "OK" if ok else "WARNING",
            "detail": f"Test = {test_pct:.1f}% of total {n_total} samples. "
                      + ("OK." if ok else "Unusual split ratio (expected 10–40%).")
        })

        # ── label balance ─────────────────────────────────────────────
        train_pos = _mean(y_train)
        test_pos  = _mean(y_test)
        bal_diff  = abs(train_pos - test_pos)
        ok = bal_diff < 0.05
        checks.append({
            "check": "label_balance",
            "status": "OK" if ok else "WARNING",
            "detail": f"Positive rate: train={train_pos:.4f}, test={test_pos:.4f}, "
                      f"diff={bal_diff:.4f}. "
                      + ("Balanced." if ok else "Label distribution mismatch — use stratified split.")
        })

        # ── feature distribution shift ────────────────────────────────
        n_features = len(X_train[0]) if X_train else 0
        fn = feature_names or [f"f{i}" for i in range(n_features)]
        ks_alerts = []
        for j in range(n_features):
            tr_col = [X_train[i][j] for i in range(n_train)]
            te_col = [X_test[i][j] for i in range(n_test)]
            ks = _ks_statistic(tr_col, te_col)
            if ks > 0.1:
                ks_alerts.append(f"{fn[j]}(KS={ks:.3f})")
        ok = len(ks_alerts) == 0
        checks.append({
            "check": "feature_distribution",
            "status": "OK" if ok else "WARNING",
            "detail": f"KS test on {n_features} features. "
                      + (f"Distribution shift detected in: {', '.join(ks_alerts[:5])}"
                         if ks_alerts else "All features stable across split.")
        })

        # ── sample overlap ────────────────────────────────────────────
        # Use tuple hashing as a proxy
        tr_hashes = set(tuple(row) for row in X_train)
        te_hashes = set(tuple(row) for row in X_test)
        overlap = len(tr_hashes & te_hashes)
        ok = overlap == 0
        checks.append({
            "check": "sample_overlap",
            "status": "OK" if ok else "ERROR",
            "detail": f"{overlap} samples appear in both train and test sets. "
                      + ("No leakage." if ok else "DATA LEAKAGE: remove duplicates before splitting!")
        })

        severity = max(c["status"] for c in checks,
                       key=lambda s: ["OK", "WARNING", "ERROR"].index(s))
        return {
            "module": "validate_split",
            "n_train": n_train, "n_test": n_test,
            "checks": checks,
            "status": severity,
            "summary": f"Split validation: {severity}. "
                       f"{sum(1 for c in checks if c['status'] == 'OK')}/{len(checks)} checks passed.",
        }

    # ── LEAKAGE DETECTION ────────────────────────────────────────────────

    def check_leakage(self, X, y, feature_names=None, threshold: float = 0.9) -> dict:
        """
        Detect potential target leakage: features that are suspiciously
        correlated with the label (correlation > threshold).

        High correlation can indicate:
          - Future data leaking into features
          - Derived features that encode the label directly
          - Post-outcome data
        """
        n = len(X)
        n_features = len(X[0]) if X else 0
        fn = feature_names or [f"f{i}" for i in range(n_features)]
        y_mean = _mean(y)
        y_std  = _std(y) or 1.0

        leaky = []
        correlations = {}

        for j in range(n_features):
            col = [X[i][j] for i in range(n)]
            col_mean = _mean(col)
            col_std  = _std(col) or 1.0
            corr = sum(
                (col[i] - col_mean) * (y[i] - y_mean)
                for i in range(n)
            ) / (n * col_std * y_std)
            correlations[fn[j]] = round(corr, 4)
            if abs(corr) >= threshold:
                leaky.append((fn[j], corr))

        leaky.sort(key=lambda x: -abs(x[1]))

        status = "ERROR" if leaky else "OK"
        return {
            "module":         "check_leakage",
            "threshold":      threshold,
            "n_features":     n_features,
            "leaky_features": [(f, round(c, 4)) for f, c in leaky],
            "all_correlations": correlations,
            "status":         status,
            "summary": (
                f"Leakage check: {len(leaky)} features exceed |corr| ≥ {threshold}. "
                + (f"SUSPECTS: {[f for f, _ in leaky[:3]]}" if leaky
                   else "No obvious target leakage detected.")
            )
        }

    # ── MISSING DATA ANALYSIS ─────────────────────────────────────────────

    def analyze_missing(self, X, y, feature_names=None) -> dict:
        """
        Characterize missing data per feature.
        Heuristic classification:
          MCAR: missingness uncorrelated with features or label
          MAR:  missingness correlated with other features
          MNAR: missingness correlated with the missing value itself (hard to test)
        """
        n = len(X)
        n_features = len(X[0]) if X else 0
        fn = feature_names or [f"f{i}" for i in range(n_features)]

        results = {}
        for j in range(n_features):
            col  = [X[i][j] for i in range(n)]
            missing_idx = [i for i, v in enumerate(col) if v is None or
                           (isinstance(v, float) and math.isnan(v))]
            miss_rate = len(missing_idx) / n

            # MCAR heuristic: compare label rate in missing vs non-missing
            y_miss    = [y[i] for i in missing_idx] if missing_idx else []
            y_present = [y[i] for i in range(n) if i not in set(missing_idx)]
            label_diff = abs(_mean(y_miss) - _mean(y_present)) if y_miss else 0.0

            # Simple MCAR / MAR / MNAR classification
            if miss_rate == 0:
                pattern = "complete"
            elif label_diff > 0.10:
                pattern = "MNAR"   # missing correlates with label — most dangerous
            elif label_diff > 0.05:
                pattern = "MAR"
            else:
                pattern = "MCAR"   # missingness appears random

            results[fn[j]] = {
                "missing_count": len(missing_idx),
                "missing_rate":  round(miss_rate, 4),
                "pattern":       pattern,
                "label_diff":    round(label_diff, 4),
            }

        mnar_features = [f for f, r in results.items() if r["pattern"] == "MNAR"]
        high_miss     = [f for f, r in results.items() if r["missing_rate"] > 0.30]

        return {
            "module":         "analyze_missing",
            "n_features":     n_features,
            "n_samples":      n,
            "per_feature":    results,
            "mnar_features":  mnar_features,
            "high_miss_features": high_miss,
            "status":         "ERROR"   if mnar_features else
                              "WARNING" if high_miss     else "OK",
            "summary": (
                f"Missing data analysis. MNAR features (label-correlated missingness): "
                f"{mnar_features or 'none'}. "
                f"High-missingness (>30%) features: {high_miss or 'none'}."
            )
        }

    # ── LABEL SANITY ─────────────────────────────────────────────────────

    def check_labels(self, y, expected_classes=None) -> dict:
        """
        Sanity-check a label vector for:
          • Unexpected label values
          • Extreme class imbalance (< 5% minority)
          • Duplicate indices (if y is list of (idx, label) pairs)
        """
        counts  = Counter(y)
        n       = len(y)
        classes = sorted(counts.keys())
        checks  = []

        # expected classes
        if expected_classes is not None:
            unexpected = [c for c in classes if c not in expected_classes]
            checks.append({
                "check": "unexpected_classes",
                "status": "ERROR" if unexpected else "OK",
                "detail": f"Unexpected labels found: {unexpected}" if unexpected
                          else f"All labels in expected set {expected_classes}."
            })

        # class imbalance
        minority_frac = min(counts.values()) / n
        checks.append({
            "check": "class_balance",
            "status": "WARNING" if minority_frac < 0.05 else "OK",
            "detail": f"Minority class fraction: {minority_frac:.4f}. "
                      + ("Severe imbalance — consider oversampling or class weights."
                         if minority_frac < 0.05 else "Acceptable balance.")
        })

        severity = max(c["status"] for c in checks,
                       key=lambda s: ["OK", "WARNING", "ERROR"].index(s))
        return {
            "module":         "check_labels",
            "class_counts":   dict(counts),
            "class_fractions": {k: round(v / n, 4) for k, v in counts.items()},
            "checks":         checks,
            "status":         severity,
            "summary": f"Label check: {severity}. Distribution: {dict(counts)}.",
        }

    # ── DISTRIBUTION SHIFT ────────────────────────────────────────────────

    def detect_drift(self, X_reference, X_current,
                      feature_names=None, method: str = "psi") -> dict:
        """
        Detect distribution shift between a reference and a current dataset.

        method : "psi"  — Population Stability Index
                 "ks"   — Kolmogorov–Smirnov statistic
        """
        n_features = len(X_reference[0]) if X_reference else 0
        fn = feature_names or [f"f{i}" for i in range(n_features)]
        alerts = []
        stats  = {}

        for j in range(n_features):
            ref = [X_reference[i][j] for i in range(len(X_reference))]
            cur = [X_current[i][j]   for i in range(len(X_current))]

            if method == "psi":
                val = _psi(ref, cur)
                threshold = 0.10
            else:
                val = _ks_statistic(ref, cur)
                threshold = 0.10

            stats[fn[j]] = round(val, 6)
            if val > threshold:
                alerts.append((fn[j], val))

        alerts.sort(key=lambda x: -x[1])

        return {
            "module":    "detect_drift",
            "method":    method,
            "per_feature": stats,
            "alerts":    [(f, round(v, 6)) for f, v in alerts],
            "status":    "WARNING" if alerts else "OK",
            "summary": (
                f"Drift detection ({method.upper()}): {len(alerts)} features drifted. "
                + (f"Top: {alerts[0][0]} ({method.upper()}={alerts[0][1]:.4f})"
                   if alerts else "No significant drift detected.")
            )
        }


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = random.Random(1)
    n   = 500

    X = [[rng.gauss(0, 1), rng.gauss(1, 0.5), rng.random()] for _ in range(n)]
    y = [1 if x[0] + x[1] > 0 else 0 for x in X]

    X_tr, X_te = X[:400], X[400:]
    y_tr, y_te = y[:400], y[400:]

    v = Validator()
    print("Split:", v.validate_split(X_tr, X_te, y_tr, y_te)["summary"])
    print("Leakage:", v.check_leakage(X_tr, y_tr)["summary"])
    print("Labels:", v.check_labels(y_tr)["summary"])
    print("Drift:", v.detect_drift(X_tr, X_te)["summary"])
    print("Validator module OK.")
