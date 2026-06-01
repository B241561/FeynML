"""
Phase 3.1 — Feature Importance: Not All Methods Are Equal
===========================================================
Source: Interpretable ML Book (Molnar) Ch.8 + Hands-On ML (Géron) Ch.6

Three methods implemented from scratch, each with different reliability:

METHOD 1 — Gini / Impurity-based importance  [built into sklearn trees]
  Bias problem: HIGH-CARDINALITY features always look more important.
  Why: more splits possible → more opportunities to reduce impurity.
  Use: quick first look ONLY. Never trust for categorical features with many values.

METHOD 2 — Permutation Importance  [more reliable]
  Idea: shuffle one feature's values → measure accuracy drop.
  If accuracy drops a lot → feature was important.
  No cardinality bias. Works on ANY model (black box).
  Weakness: correlated features split importance between them.

METHOD 3 — Drop-Column Importance  [most reliable, rarely used]
  Idea: retrain model WITHOUT one feature → measure accuracy drop.
  The most honest estimate. Captures true marginal value.
  Weakness: O(n_features) retraining cycles → SLOW for large models.
"""

import math
import random
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _accuracy(y_true, y_pred):
    return sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)

def _mean(values):
    return sum(values) / len(values)

def _shuffle_column(X, col_idx, seed=None):
    """Return new X with column col_idx shuffled (in place copy)."""
    rng = random.Random(seed)
    col = [X[i][col_idx] for i in range(len(X))]
    rng.shuffle(col)
    X_new = [row[:] for row in X]
    for i in range(len(X_new)):
        X_new[i][col_idx] = col[i]
    return X_new

def _drop_column(X, col_idx):
    """Return X with column col_idx removed."""
    return [[row[j] for j in range(len(row)) if j != col_idx] for row in X]


# ─────────────────────────────────────────────────────────────────────────────
# 1. GINI / IMPURITY-BASED IMPORTANCE (explanation of sklearn internals)
# ─────────────────────────────────────────────────────────────────────────────

def gini_impurity(labels):
    """
    Gini impurity of a node: 1 - Σ p_i²
    = 0 for pure node (all same class)
    = 0.5 for maximally impure binary node (50/50 split)
    """
    if not labels:
        return 0.0
    n = len(labels)
    counts = defaultdict(int)
    for l in labels:
        counts[l] += 1
    return 1.0 - sum((c / n) ** 2 for c in counts.values())

def weighted_gini_reduction(parent_labels, left_labels, right_labels):
    """
    Impurity reduction from one split.
    ΔGini = Gini(parent) - (|left|/|parent|)*Gini(left) - (|right|/|parent|)*Gini(right)
    
    sklearn accumulates this across ALL splits for each feature
    → features used for many high-quality splits get high importance.
    """
    n  = len(parent_labels)
    nl = len(left_labels)
    nr = len(right_labels)
    return (gini_impurity(parent_labels)
            - (nl / n) * gini_impurity(left_labels)
            - (nr / n) * gini_impurity(right_labels))

def gini_importance_from_sklearn(sklearn_tree_model, feature_names=None):
    """
    Extract and explain sklearn's feature_importances_ attribute.
    Returns sorted list with cardinality bias warning.
    """
    importances = sklearn_tree_model.feature_importances_
    n = len(importances)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n)]

    result = sorted(
        [{"feature": feature_names[i], "importance": round(float(importances[i]), 6)}
         for i in range(n)],
        key=lambda x: x["importance"], reverse=True
    )

    return {
        "method":   "Gini / Impurity-based (sklearn feature_importances_)",
        "bias":     "HIGH-CARDINALITY features are systematically overrated. "
                    "Features with many unique values get more split opportunities.",
        "rankings": result,
        "warning":  "Do NOT use as the sole importance method for categorical features.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. PERMUTATION IMPORTANCE — from scratch
# ─────────────────────────────────────────────────────────────────────────────

def permutation_importance(
    model,
    X_val,
    y_val,
    feature_names=None,
    n_repeats=5,
    seed=42,
    score_fn=None,
):
    """
    Permutation Importance (Breiman 2001, popularised by sklearn).
    
    Algorithm:
      1. Compute baseline score on validation set.
      2. For each feature i:
           For each repeat r:
             a. Shuffle column i in X_val.
             b. Compute score on shuffled data.
             c. importance[i][r] = baseline_score - shuffled_score
      3. Report mean ± std of importance per feature.
    
    Key properties:
      ✓ Model-agnostic (black box)
      ✓ No cardinality bias
      ✓ Uses validation data → reflects true generalisation importance
      ✗ Correlated features: importance shared/diluted between them
      ✗ Extrapolation: shuffled data may be out-of-distribution
    
    Args:
      model: must have .predict(X) method
      X_val: list of feature vectors
      y_val: list of true labels
      n_repeats: number of times to shuffle per feature (reduces variance)
    """
    if score_fn is None:
        score_fn = _accuracy

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(X_val[0]))]

    baseline = score_fn(y_val, model.predict(X_val))
    n_features = len(X_val[0])
    results = []

    for j in range(n_features):
        drops = []
        for r in range(n_repeats):
            X_shuffled = _shuffle_column(X_val, j, seed=seed + r * 100 + j)
            shuffled_score = score_fn(y_val, model.predict(X_shuffled))
            drops.append(baseline - shuffled_score)

        mean_drop = _mean(drops)
        std_drop  = math.sqrt(_mean([(d - mean_drop)**2 for d in drops]))

        results.append({
            "feature":    feature_names[j],
            "importance": round(mean_drop, 6),
            "std":        round(std_drop, 6),
            "raw_drops":  [round(d, 6) for d in drops],
        })

    results.sort(key=lambda x: x["importance"], reverse=True)

    return {
        "method":          "Permutation Importance",
        "baseline_score":  round(baseline, 6),
        "n_repeats":       n_repeats,
        "rankings":        results,
        "interpretation":  (
            "importance = average drop in accuracy when feature is shuffled. "
            "Negative importance = shuffling IMPROVES score → feature hurts model."
        ),
        "correlated_warning": (
            "If two features are correlated, importance is SPLIT between them. "
            "Drop-column importance is more reliable in this case."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. DROP-COLUMN IMPORTANCE — most reliable, rarely used
# ─────────────────────────────────────────────────────────────────────────────

def drop_column_importance(
    model_fn,
    X_train,
    y_train,
    X_val,
    y_val,
    feature_names=None,
    score_fn=None,
):
    """
    Drop-Column Importance.
    
    Algorithm:
      1. Train full model on all features → baseline score.
      2. For each feature i:
           a. Remove column i from X_train AND X_val.
           b. Retrain model from scratch on reduced data.
           c. importance[i] = baseline_score - reduced_score
    
    Key properties:
      ✓ Most honest estimate of true feature value
      ✓ Handles correlated features correctly (no dilution)
      ✓ Captures non-linear interactions
      ✗ SLOW: requires (n_features + 1) full model retraining cycles
      ✗ Variance from retraining randomness (use fixed seed)
    
    Args:
      model_fn: callable() → returns a NEW unfitted model instance
                model must have .fit(X, y) and .predict(X)
    """
    if score_fn is None:
        score_fn = _accuracy
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(X_train[0]))]

    # Baseline: train on all features
    full_model = model_fn()
    full_model.fit(X_train, y_train)
    baseline = score_fn(y_val, full_model.predict(X_val))

    results = []
    n_features = len(X_train[0])

    for j in range(n_features):
        X_tr_reduced = _drop_column(X_train, j)
        X_vl_reduced = _drop_column(X_val,   j)

        reduced_model = model_fn()
        reduced_model.fit(X_tr_reduced, y_train)
        reduced_score = score_fn(y_val, reduced_model.predict(X_vl_reduced))

        results.append({
            "feature":    feature_names[j],
            "importance": round(baseline - reduced_score, 6),
            "full_score": round(baseline,       6),
            "drop_score": round(reduced_score,  6),
        })

    results.sort(key=lambda x: x["importance"], reverse=True)

    return {
        "method":         "Drop-Column Importance",
        "baseline_score": round(baseline, 6),
        "n_retrains":     n_features,
        "rankings":       results,
        "interpretation": (
            "importance = accuracy drop when feature is COMPLETELY removed and model retrained. "
            "This is the truest measure of a feature's marginal value."
        ),
        "cost_warning": (
            f"Required {n_features} full retraining cycles. "
            "For large models/datasets, consider permutation importance instead."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. COMPARISON UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def compare_importance_methods(gini_result, perm_result, drop_result=None):
    """
    Compare rankings from multiple methods.
    Highlights disagreements (potential cardinality bias or correlation issues).
    """
    def rank_dict(result_key):
        return {r["feature"]: i + 1 for i, r in enumerate(result_key["rankings"])}

    gini_ranks = rank_dict(gini_result)
    perm_ranks = rank_dict(perm_result)

    features = list(gini_ranks.keys())
    rows = []
    for f in features:
        row = {
            "feature":    f,
            "gini_rank":  gini_ranks.get(f, "—"),
            "perm_rank":  perm_ranks.get(f, "—"),
        }
        if drop_result:
            drop_ranks = rank_dict(drop_result)
            row["drop_rank"] = drop_ranks.get(f, "—")
            row["rank_spread"] = max(
                abs(row["gini_rank"] - row["perm_rank"]),
                abs(row["gini_rank"] - row.get("drop_rank", row["gini_rank"])),
            )
        else:
            row["rank_spread"] = abs(row["gini_rank"] - row["perm_rank"])

        row["suspicious"] = row["rank_spread"] >= 3
        rows.append(row)

    rows.sort(key=lambda x: x["rank_spread"], reverse=True)

    suspicious = [r["feature"] for r in rows if r["suspicious"]]

    return {
        "comparison_table": rows,
        "suspicious_features": suspicious,
        "interpretation": (
            "Features with large rank_spread disagree between methods. "
            "Likely cause: cardinality bias (many unique values) OR "
            "correlation with another feature. Investigate these features carefully."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def run_verification():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.inspection import permutation_importance as sklearn_perm
    import numpy as np

    print("=" * 65)
    print("Phase 3.1 — Feature Importance Verification")
    print("=" * 65)

    # Synthetic data: features 0,1 are informative; 2,3,4 are noise
    np.random.seed(42)
    X_np, y_np = make_classification(
        n_samples=600, n_features=5,
        n_informative=2, n_redundant=0,
        n_clusters_per_class=1, random_state=42
    )
    X_tr = X_np[:400].tolist(); y_tr = y_np[:400].tolist()
    X_vl = X_np[400:].tolist(); y_vl = y_np[400:].tolist()
    feat_names = ["info_A", "info_B", "noise_C", "noise_D", "noise_E"]

    # ── 1. Gini importance ────────────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_tr, y_tr)
    gini_res = gini_importance_from_sklearn(rf, feat_names)
    print(f"\n  Gini Importance (sklearn):")
    for r in gini_res["rankings"]:
        print(f"    {r['feature']:<12} {r['importance']:.4f}")

    # ── 2. Permutation importance ────────────────────────────────────
    perm_res = permutation_importance(rf, X_vl, y_vl, feat_names, n_repeats=5)
    skl_perm = sklearn_perm(rf, np.array(X_vl), np.array(y_vl),
                            n_repeats=5, random_state=42)
    print(f"\n  Permutation Importance (our impl vs sklearn):")
    for i, r in enumerate(perm_res["rankings"]):
        feat_idx = feat_names.index(r["feature"])
        sk_mean  = round(float(skl_perm.importances_mean[feat_idx]), 4)
        ok = abs(r["importance"] - sk_mean) < 0.05
        print(f"    {r['feature']:<12} ours={r['importance']:.4f} "
              f"sklearn≈{sk_mean:.4f}  [{'✓' if ok else '~'}]")

    # ── 3. Drop-column importance ─────────────────────────────────────
    def model_fn():
        return RandomForestClassifier(n_estimators=30, random_state=42)

    print(f"\n  Drop-Column Importance (retrains {len(feat_names)}x):")
    drop_res = drop_column_importance(model_fn, X_tr, y_tr, X_vl, y_vl, feat_names)
    print(f"  Baseline accuracy: {drop_res['baseline_score']:.4f}")
    for r in drop_res["rankings"]:
        bar = "█" * max(0, int(r["importance"] * 80))
        print(f"    {r['feature']:<12} {r['importance']:+.4f}  {bar}")

    # ── 4. Comparison ─────────────────────────────────────────────────
    comp = compare_importance_methods(gini_res, perm_res, drop_res)
    print(f"\n  Ranking Comparison:")
    print(f"    {'Feature':<12} {'Gini':>6} {'Perm':>6} {'Drop':>6} {'Spread':>8}")
    for row in comp["comparison_table"]:
        flag = " ⚠️ " if row["suspicious"] else ""
        print(f"    {row['feature']:<12} {row['gini_rank']:>6} "
              f"{row['perm_rank']:>6} {row.get('drop_rank','—'):>6} "
              f"{row['rank_spread']:>8}{flag}")

    if comp["suspicious_features"]:
        print(f"\n  Suspicious (rank spread ≥3): {comp['suspicious_features']}")

    # Informative features should rank top-2 in all methods
    top2_gini = {r["feature"] for r in gini_res["rankings"][:2]}
    top2_perm = {r["feature"] for r in perm_res["rankings"][:2]}
    ok = "info_A" in top2_perm and "info_B" in top2_perm
    print(f"\n  Informative features in top-2 (perm): {top2_perm}  [{'✓ PASS' if ok else '✗ FAIL'}]")
    print("=" * 65)


if __name__ == "__main__":
    run_verification()
