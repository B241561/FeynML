"""
Phase 3 — SHAP vs LIME: Agreement & Method Selection
=====================================================
Compares local explanations from SHAP and LIME on the same instance.

Used by engine/modules/explainability_engine.py when method="both".

DECISION_GUIDE summarizes when to prefer each method (from syllabus + engine defaults).
"""

import math
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# METHOD SELECTION GUIDE (consumed by ExplainabilityEngine docs / reports)
# ─────────────────────────────────────────────────────────────────────────────

DECISION_GUIDE = {
    "n_features_lte_8": {
        "recommendation": "both",
        "rationale": "Low dimensionality: exact/kernel SHAP is feasible; LIME cross-check is cheap.",
    },
    "n_features_9_to_20": {
        "recommendation": "shap",
        "rationale": "Kernel SHAP scales better than dense LIME when M is moderate.",
    },
    "n_features_gt_20": {
        "recommendation": "lime",
        "rationale": "LIME uses sparse top-K explanations; full SHAP sampling is expensive.",
    },
    "tree_model": {
        "recommendation": "shap_treeshap",
        "rationale": "Use shap_explainer.shap_with_library for exact polynomial-time TreeSHAP.",
    },
    "low_agreement": {
        "recommendation": "investigate",
        "rationale": "Spearman ρ below threshold: correlated features, insufficient LIME samples, or non-linear boundary.",
    },
}

# Agreement thresholds (aligned with ExplainabilityEngine.assert_gate default 0.6)
SPEARMAN_STRONG = 0.75
SPEARMAN_MODERATE = 0.50


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _rank_data(values):
    """Average ranks for ties (Spearman)."""
    sorted_idx = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_idx):
        j = i
        while j < len(sorted_idx) and values[sorted_idx[j]] == values[sorted_idx[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[sorted_idx[k]] = avg_rank
        i = j
    return ranks


def spearman_rho(x, y):
    """
    Spearman rank correlation in [-1, 1].
    Returns 0.0 if undefined (constant vectors or length < 2).
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    rx = _rank_data(x)
    ry = _rank_data(y)
    n = len(x)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    if n <= 1:
        return 0.0
    rho = 1.0 - (6.0 * d2) / (n * (n * n - 1))
    return max(-1.0, min(1.0, rho))


def _top_k_indices(values, k=3):
    """Indices of top-k features by absolute value."""
    ranked = sorted(range(len(values)), key=lambda i: abs(values[i]), reverse=True)
    return ranked[:k]


def _top_k_feature_names(values, feature_names, k=3):
    return [feature_names[i] for i in _top_k_indices(values, k)]


# ─────────────────────────────────────────────────────────────────────────────
# 1. COMPARE EXPLANATIONS
# ─────────────────────────────────────────────────────────────────────────────

def compare_explanations(shap_vals, lime_vals, feature_names=None):
    """
    Compare SHAP and LIME attribution vectors for one instance.

    Parameters
    ----------
    shap_vals, lime_vals : list[float]  same length, aligned feature order
    feature_names        : list[str]     optional labels

    Returns
    -------
    dict with spearman_rho, top3_overlap, verdict, top_shap, top_lime
    """
    if len(shap_vals) != len(lime_vals):
        raise ValueError(
            f"shap_vals length {len(shap_vals)} != lime_vals length {len(lime_vals)}"
        )
    n = len(shap_vals)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n)]

    rho = spearman_rho(shap_vals, lime_vals)

    top_shap = set(_top_k_feature_names(shap_vals, feature_names, k=min(3, n)))
    top_lime = set(_top_k_feature_names(lime_vals, feature_names, k=min(3, n)))
    overlap = len(top_shap & top_lime)
    top3_overlap = overlap / min(3, n) if n > 0 else 0.0

    if rho >= SPEARMAN_STRONG and overlap >= 2:
        verdict = "STRONG_AGREEMENT"
    elif rho >= SPEARMAN_MODERATE or overlap >= 1:
        verdict = "MODERATE_AGREEMENT"
    else:
        verdict = "DISAGREEMENT"

    return {
        "spearman_rho":   round(rho, 4),
        "top3_overlap":   round(top3_overlap, 4),
        "top3_count":     overlap,
        "verdict":        verdict,
        "top_shap":       list(top_shap),
        "top_lime":       list(top_lime),
        "interpretation": (
            f"Spearman ρ={rho:.3f}, top-3 overlap={overlap}/{min(3, n)}. "
            f"Verdict: {verdict}."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. STABILITY ANALYSIS (LIME-focused; SHAP optional second pass)
# ─────────────────────────────────────────────────────────────────────────────

def stability_analysis(
    model_fn,
    x,
    X_background,
    feature_names=None,
    n_seeds=5,
    n_samples=200,
    K=4,
):
    """
  Run LIME stability across seeds (delegates to lime_explainer).

  Returns lime_stability_report plus summary flags for the engine/report.
    """
    from lime_explainer import lime_stability_analysis

    report = lime_stability_analysis(
        model_fn,
        x,
        X_background,
        feature_names=feature_names,
        n_seeds=n_seeds,
        n_samples=n_samples,
        K=K,
    )
    return {
        "method":           "LIME multi-seed stability",
        "n_seeds":          report["n_seeds"],
        "overall_stable":   report["overall_stable"],
        "n_unstable_feats": report["n_unstable_feats"],
        "stability_report": report["stability_report"],
        "recommendation":   report["recommendation"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. BATCH AGREEMENT
# ─────────────────────────────────────────────────────────────────────────────

def batch_agreement(shap_batch, lime_batch, feature_names=None):
    """
    Compare SHAP/LIME vectors for multiple instances.

    Parameters
    ----------
    shap_batch, lime_batch : list[list[float]]  same outer length

    Returns
    -------
    dict with per_instance comparisons and mean spearman_rho
    """
    if len(shap_batch) != len(lime_batch):
        raise ValueError("shap_batch and lime_batch must have same length")
    if not shap_batch:
        return {"n_instances": 0, "per_instance": [], "mean_spearman_rho": None}

    per = []
    rhos = []
    for shap_v, lime_v in zip(shap_batch, lime_batch):
        cmp = compare_explanations(shap_v, lime_v, feature_names)
        per.append(cmp)
        rhos.append(cmp["spearman_rho"])

    mean_rho = sum(rhos) / len(rhos)
    n_disagree = sum(1 for c in per if c["verdict"] == "DISAGREEMENT")

    return {
        "n_instances":        len(per),
        "per_instance":       per,
        "mean_spearman_rho":  round(mean_rho, 4),
        "n_disagreement":     n_disagree,
        "batch_verdict":      (
            "CONSISTENT" if n_disagree == 0
            else "MIXED" if n_disagree < len(per)
            else "INCONSISTENT"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def run_verification():
    print("=" * 65)
    print("Phase 3 — SHAP vs LIME Agreement Verification")
    print("=" * 65)

    names = ["x0", "x1", "x2"]
    shap = [0.5, -0.3, 0.1]
    lime = [0.48, -0.25, 0.05]
    r = compare_explanations(shap, lime, names)
    assert r["spearman_rho"] > 0.5, r
    assert r["verdict"] in ("STRONG_AGREEMENT", "MODERATE_AGREEMENT"), r
    print(f"  compare_explanations: ρ={r['spearman_rho']} {r['verdict']} ✓")

    batch = batch_agreement([shap, [-0.5, 0.3, 0.0]], [lime, [0.5, -0.3, 0.0]], names)
    assert batch["n_instances"] == 2
    print(f"  batch_agreement: mean ρ={batch['mean_spearman_rho']} ✓")

    assert "n_features_lte_8" in DECISION_GUIDE
    print("  DECISION_GUIDE ✓")
    print("\n✓ shap_vs_lime OK")


if __name__ == "__main__":
    run_verification()
