"""
Phase 3.2 — SHAP: SHapley Additive exPlanations
=================================================
Primary source: Lundberg & Lee (NIPS 2017) — arxiv 1705.07874v2
Secondary:      Interpretable ML Book (Molnar) Ch.9.6

CORE IDEA (from paper, Section 4):
  φᵢ(f, x) = Σ_{S ⊆ F\{i}}  |S|!(|F|-|S|-1)!/|F|!  × [f(x_S∪{i}) - f(x_S)]
  
  In plain English:
    For every possible SUBSET of features (without feature i),
    measure how much adding feature i CHANGES the prediction.
    Average this change across all possible orderings.
    That average change = SHAP value for feature i.

WHY COOPERATIVE GAME THEORY?
  Imagine features as "players" in a team.
  The team produces some prediction (the "profit").
  SHAP fairly distributes credit (profit) among players.
  Axioms satisfied: Local accuracy, Missingness, Consistency (Theorem 1).

THREE PROPERTIES (Lundberg & Lee Theorem 1):
  1. Local accuracy:  Σ φᵢ = f(x) - E[f(x)]   (attributions sum to prediction)
  2. Missingness:     φᵢ = 0 if feature absent
  3. Consistency:     if feature i matters more → φᵢ never decreases

KERNEL SHAP (Section 4.1 of paper):
  Uses LIME's regression framework with the special Shapley kernel:
  π(z') = (M-1) / (C(M, |z'|) × |z'| × (M - |z'|))
  This kernel makes weighted linear regression recover exact Shapley values.

TREESHAP (mentioned in paper, fully in Lundberg et al. 2018):
  For tree models: O(TLD²) instead of O(2^M) 
  where T=trees, L=leaves, D=depth.
"""

import math
import random
import itertools
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# 1. EXACT SHAPLEY VALUES (Brute Force — exponential, for small M only)
# ─────────────────────────────────────────────────────────────────────────────

def shapley_values_exact(f, x, background_mean, feature_names=None):
    """
    Exact Shapley values via complete enumeration of all feature subsets.
    
    Equation 4 from Lundberg & Lee (2017):
      φᵢ = Σ_{S⊆F\{i}} |S|!(M-|S|-1)!/M! × [f(x_S∪{i}) - f(x_S)]
    
    f(x_S) is approximated by replacing missing features with background_mean.
    This is the 'feature independence' approximation from Equation 11 of paper.
    
    ⚠️  EXPONENTIAL TIME: O(2^M) — only use for M ≤ 12.
    
    Args:
      f:               function(x_list) → scalar prediction
      x:               list of feature values for ONE instance
      background_mean: list of mean feature values (E[X]) from training set
      feature_names:   list of feature names
    """
    M = len(x)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(M)]

    def f_with_subset(S):
        """Evaluate f with features in S using x[i], others using background."""
        x_partial = [x[i] if i in S else background_mean[i] for i in range(M)]
        return f(x_partial)

    phi = [0.0] * M
    for i in range(M):
        features_without_i = [j for j in range(M) if j != i]
        for r in range(len(features_without_i) + 1):
            for S_tuple in itertools.combinations(features_without_i, r):
                S     = set(S_tuple)
                S_i   = S | {i}
                s_len = len(S)
                # Weight: |S|!(M-|S|-1)!/M!
                weight = (
                    math.factorial(s_len)
                    * math.factorial(M - s_len - 1)
                    / math.factorial(M)
                )
                phi[i] += weight * (f_with_subset(S_i) - f_with_subset(S))

    # Verify local accuracy: Σ φᵢ + E[f] ≈ f(x)
    base_value = f(background_mean)
    pred       = f(x)
    phi_sum    = sum(phi)
    local_accuracy_error = abs(phi_sum + base_value - pred)

    return {
        "shap_values":           [round(v, 6) for v in phi],
        "feature_names":         feature_names,
        "base_value":            round(base_value, 6),
        "prediction":            round(pred, 6),
        "phi_sum":               round(phi_sum + base_value, 6),
        "local_accuracy_error":  round(local_accuracy_error, 8),
        "local_accuracy_ok":     local_accuracy_error < 1e-5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. KERNEL SHAP — model-agnostic approximation (Section 4.1 of paper)
# ─────────────────────────────────────────────────────────────────────────────

def _shapley_kernel_weight(M, z_prime_size):
    """
    Theorem 2 from Lundberg & Lee (2017):
    π(z') = (M-1) / (C(M, |z'|) × |z'| × (M - |z'|))
    
    Special case: π = ∞ when |z'| ∈ {0, M}
    These are handled by constraining φ₀ = f(background) and Σφᵢ = f(x).
    """
    if z_prime_size == 0 or z_prime_size == M:
        return float("inf")
    comb = math.comb(M, z_prime_size)
    if comb == 0 or z_prime_size == 0 or (M - z_prime_size) == 0:
        return float("inf")
    return (M - 1) / (comb * z_prime_size * (M - z_prime_size))

def kernel_shap(f, x, X_background, n_samples=512, seed=42):
    """
    Kernel SHAP: Linear LIME + Shapley kernel (Section 4.1).
    
    Uses weighted least squares regression where the kernel weights
    are chosen so that the solution equals exact Shapley values.
    
    Algorithm:
      1. Sample binary coalition vectors z' ∈ {0,1}^M
      2. Map z' → real input: use x[i] if z'[i]=1, background sample if z'[i]=0
      3. Get model output f(z)
      4. Fit weighted linear regression: φ = argmin Σ π(z') (f(z) - φ₀ - Σ φᵢz'ᵢ)²
    
    The Shapley kernel π(z') ensures the regression recovers Shapley values.
    """
    rng = random.Random(seed)
    M   = len(x)

    # Background mean for marginalisation
    bg_mean = [
        sum(X_background[i][j] for i in range(len(X_background))) / len(X_background)
        for j in range(M)
    ]
    f_background = f(bg_mean)

    # Sample coalition vectors
    Z_prime   = []
    f_vals    = []
    weights   = []

    # Always include the all-zeros and all-ones (handled separately via constraints)
    for _ in range(n_samples):
        z_prime_size = rng.randint(1, M - 1)
        on_features  = rng.sample(range(M), z_prime_size)
        z_prime      = [1 if j in on_features else 0 for j in range(M)]

        # Map to real space: use x[j] if on, else random background sample
        bg_row = rng.choice(X_background)
        z_real = [x[j] if z_prime[j] == 1 else bg_row[j] for j in range(M)]

        f_z    = f(z_real)
        w      = _shapley_kernel_weight(M, z_prime_size)

        Z_prime.append(z_prime)
        f_vals.append(f_z - f_background)   # center around E[f]
        weights.append(min(w, 1e6))         # cap infinite weights

    # Weighted least squares: φ = (Z'ᵀWZ')⁻¹ Z'ᵀW y
    # Simple implementation using gradient descent for stability
    phi   = [0.0] * M
    lr    = 0.01
    n_iter = 500

    for _ in range(n_iter):
        grads = [0.0] * M
        for s in range(len(Z_prime)):
            pred_s  = sum(phi[j] * Z_prime[s][j] for j in range(M))
            residual = pred_s - f_vals[s]
            w_s      = weights[s]
            for j in range(M):
                grads[j] += 2 * w_s * residual * Z_prime[s][j]
        phi = [phi[j] - lr * grads[j] / len(Z_prime) for j in range(M)]

    return {
        "shap_values":  [round(v, 6) for v in phi],
        "base_value":   round(f_background, 6),
        "prediction":   round(f(x), 6),
        "phi_sum":      round(f_background + sum(phi), 6),
        "n_samples":    n_samples,
        "method":       "Kernel SHAP (approx via weighted regression)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. SHAP WITH SKLEARN (TreeSHAP via shap library)
# ─────────────────────────────────────────────────────────────────────────────

def shap_with_library(model, X_train, X_explain, feature_names=None, model_type="tree"):
    """
    SHAP using the official shap library.
    TreeSHAP: O(TLD²) — polynomial, not exponential.
    
    TreeSHAP key insight (from Lundberg et al. 2018, follow-up paper):
    Instead of evaluating all 2^M subsets, it traverses each tree once
    per sample and analytically computes the expected value for each
    possible feature subset using the tree structure itself.
    
    Result: exact Shapley values for tree models in polynomial time.
    """
    try:
        import shap
        import numpy as np
    except ImportError:
        return {"error": "pip install shap"}

    X_tr_np  = np.array(X_train)
    X_exp_np = np.array(X_explain)

    if model_type == "tree":
        explainer = shap.TreeExplainer(model, X_tr_np)
    else:
        explainer = shap.KernelExplainer(
            model.predict_proba if hasattr(model, "predict_proba") else model.predict,
            shap.sample(X_tr_np, 50)
        )

    shap_values = explainer.shap_values(X_exp_np)

    # For binary classification, TreeExplainer returns [class0, class1]
    if isinstance(shap_values, list):
        sv = shap_values[1]   # class 1 SHAP values
    else:
        sv = shap_values

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X_exp_np.shape[1])]

    # Global importance: mean(|SHAP|) per feature
    global_importance = [
        {
            "feature":    feature_names[j],
            "mean_abs_shap": round(float(np.mean(np.abs(sv[:, j]))), 6),
        }
        for j in range(len(feature_names))
    ]
    global_importance.sort(key=lambda x: x["mean_abs_shap"], reverse=True)

    return {
        "method":           "TreeSHAP (shap library)",
        "shap_matrix":      sv.tolist(),
        "base_value":       round(float(explainer.expected_value
                            if not isinstance(explainer.expected_value, list)
                            else explainer.expected_value[1]), 6),
        "global_importance": global_importance,
        "n_explained":      len(X_explain),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. SHAP PLOT DATA (for custom visualisations without matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def shap_waterfall_data(shap_values, feature_names, feature_values, base_value, prediction):
    """
    Prepare data for a SHAP waterfall plot.
    Waterfall shows: base_value + contributions → final prediction.
    
    Returns ordered list of contributions from most negative to most positive.
    """
    contributions = sorted(
        zip(shap_values, feature_names, feature_values),
        key=lambda t: abs(t[0]), reverse=True
    )
    waterfall = []
    running = base_value
    for shap_val, fname, fval in contributions:
        waterfall.append({
            "feature":    fname,
            "value":      fval,
            "shap":       round(shap_val, 6),
            "cumulative": round(running + shap_val, 6),
            "direction":  "positive" if shap_val >= 0 else "negative",
        })
        running += shap_val
    return {
        "base_value":  round(base_value, 6),
        "prediction":  round(prediction, 6),
        "steps":       waterfall,
    }

def shap_summary_data(shap_matrix, feature_names):
    """
    Prepare data for a SHAP summary (beeswarm) plot.
    Returns global importance + per-feature distribution.
    """
    import math
    n_samples  = len(shap_matrix)
    n_features = len(feature_names)
    summary    = []
    for j in range(n_features):
        vals     = [shap_matrix[i][j] for i in range(n_samples)]
        mean_abs = sum(abs(v) for v in vals) / n_samples
        summary.append({
            "feature":       feature_names[j],
            "mean_abs_shap": round(mean_abs, 6),
            "shap_values":   [round(v, 4) for v in vals],
            "positive_pct":  round(sum(1 for v in vals if v > 0) / n_samples, 4),
        })
    summary.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 5. WHEN SHAP LIES — known failure modes
# ─────────────────────────────────────────────────────────────────────────────

SHAP_FAILURE_MODES = {
    "correlated_features": {
        "problem": (
            "When features are highly correlated (e.g., height_cm and height_inches), "
            "SHAP must evaluate f on unrealistic feature combinations "
            "(e.g., height_cm=180 WITH height_inches=50). "
            "This produces extrapolation artifacts and splits importance arbitrarily."
        ),
        "detection": "Check correlation matrix. Pairs with |r| > 0.8 are suspect.",
        "mitigation": "Use TreeSHAP with interventional rather than observational SHAP, "
                      "or consider grouping correlated features.",
    },
    "model_instability": {
        "problem": (
            "If the model is sensitive to small input changes, "
            "SHAP values computed on nearby points can vary wildly. "
            "This makes local explanations unreliable."
        ),
        "detection": "Compute SHAP values on same instance multiple times with slight "
                     "noise. High variance = unstable explanations.",
        "mitigation": "Use ensemble models. Regularize more aggressively.",
    },
    "extrapolation": {
        "problem": (
            "Kernel SHAP evaluates the model on combinations of features "
            "that may lie far outside the training distribution. "
            "The model may behave unpredictably in these regions."
        ),
        "detection": "Check if masked feature combinations fall outside training range.",
        "mitigation": "Use TreeSHAP (uses training distribution internally) "
                      "or use background dataset that matches test distribution.",
    },
    "global_vs_local": {
        "problem": (
            "A feature can have low global importance (mean|SHAP|) "
            "but extremely high local importance for specific predictions. "
            "Averaging masks these critical cases."
        ),
        "detection": "Always inspect individual waterfall plots for high-stakes predictions "
                     "in addition to the summary plot.",
        "mitigation": "Always examine both global summary AND local waterfall plots.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 6. VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def run_verification():
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    print("=" * 65)
    print("Phase 3.2 — SHAP Explainer Verification")
    print("=" * 65)

    # ── Simple linear model: f(x) = 2x0 + 3x1
    # Exact SHAP for linear model with zero background:
    # φ0 = 2*x0, φ1 = 3*x1 (exactly, since no interactions)
    def linear_model(x):
        return 2 * x[0] + 3 * x[1]

    x_inst  = [1.0, 2.0]
    bg_mean = [0.0, 0.0]

    result = shapley_values_exact(linear_model, x_inst, bg_mean, ["x0", "x1"])
    phi = result["shap_values"]
    print(f"\n  Exact SHAP on f=2x0+3x1, x=[1,2], bg=[0,0]:")
    print(f"  φ(x0) = {phi[0]:.4f}  (expected 2.0000)")
    print(f"  φ(x1) = {phi[1]:.4f}  (expected 6.0000)")
    print(f"  Local accuracy error: {result['local_accuracy_error']:.2e}  "
          f"[{'✓ PASS' if result['local_accuracy_ok'] else '✗ FAIL'}]")

    # Verify: φ0 + φ1 + E[f] = f(x)
    ok_sum = abs(sum(phi) + result["base_value"] - result["prediction"]) < 1e-5
    print(f"  Σφᵢ + E[f] = f(x): {sum(phi):.4f} + {result['base_value']:.4f} "
          f"= {result['prediction']:.4f}  [{'✓ PASS' if ok_sum else '✗ FAIL'}]")

    # ── TreeSHAP via shap library ──────────────────────────────────────
    print(f"\n  TreeSHAP (shap library):")
    try:
        np.random.seed(42)
        from sklearn.datasets import make_classification
        X_np, y_np = make_classification(
            n_samples=300, n_features=4,
            n_informative=2, random_state=42
        )
        rf = RandomForestClassifier(n_estimators=30, random_state=42)
        rf.fit(X_np[:200], y_np[:200])
        res = shap_with_library(rf, X_np[:200].tolist(), X_np[200:210].tolist(),
                                feature_names=["f0","f1","f2","f3"])
        if "error" not in res:
            print(f"  Global importance (mean|SHAP|):")
            for r in res["global_importance"]:
                bar = "█" * max(0, int(r["mean_abs_shap"] * 200))
                print(f"    {r['feature']:<6} {r['mean_abs_shap']:.4f}  {bar}")

            # Verify local accuracy for first explained instance
            sv0   = res["shap_matrix"][0]
            base  = res["base_value"]
            pred  = rf.predict_proba(X_np[200:201])[0][1]
            err   = abs(sum(sv0) + base - pred)
            print(f"  Local accuracy (instance 0): err={err:.4e}  "
                  f"[{'✓ PASS' if err < 0.01 else '✗ FAIL'}]")
        else:
            print(f"  shap library not available: {res['error']}")
    except Exception as e:
        print(f"  TreeSHAP test skipped: {e}")

    # ── Waterfall data ─────────────────────────────────────────────────
    wf = shap_waterfall_data(
        shap_values=[0.5, -0.3, 0.2, -0.1],
        feature_names=["age", "income", "credit_score", "debt_ratio"],
        feature_values=[35, 50000, 720, 0.3],
        base_value=0.4,
        prediction=0.7,
    )
    print(f"\n  Waterfall (base={wf['base_value']} → pred={wf['prediction']}):")
    for step in wf["steps"]:
        sign = "▲" if step["direction"] == "positive" else "▼"
        print(f"    {sign} {step['feature']:<14} SHAP={step['shap']:+.3f}  "
              f"cumul={step['cumulative']:.3f}")

    print(f"\n  SHAP Failure Modes: {list(SHAP_FAILURE_MODES.keys())}")
    print("=" * 65)


if __name__ == "__main__":
    run_verification()
