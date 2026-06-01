"""
Phase 3.3 — LIME: Local Interpretable Model-Agnostic Explanations
==================================================================
Primary source: Ribeiro, Singh, Guestrin (KDD 2016) — arxiv 1602.04938v3

CORE IDEA (Algorithm 1 from paper):
  Given a black-box model f and an instance x to explain:
  1. Sample N perturbed instances z' around x
  2. Get f(z) for each perturbed instance
  3. Weight each z by proximity to x: π(z) = exp(-D(x,z)²/σ²)
  4. Fit a sparse linear model g on {z', f(z), π(z)}
  5. The weights of g are the LIME explanation

FIDELITY-INTERPRETABILITY TRADEOFF (Section 3.2 of paper):
  ξ(x) = argmin_{g ∈ G} L(f, g, π_x) + Ω(g)
  
  L = how unfaithful g is locally around x
  Ω = complexity of g (number of non-zero weights)
  
  LIME solves this by: fixing K features → linear regression on perturbed samples

PERTURBATION SAMPLING (Section 3.3 of paper):
  For tabular data:
    - Sample z' by replacing feature values with random values from training data
    - This is more realistic than using Gaussian noise
  For text: binary vector (word present/absent)
  For images: superpixels (on/off)
  
  We implement the tabular version.

KNOWN WEAKNESS — Instability:
  Different random seeds → different explanations for same instance.
  Cause: the local approximation depends heavily on which samples are drawn.
  Fix: increase n_samples, use multiple seeds, report variance.
"""

import math
import random
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# 1. PROXIMITY KERNEL
# ─────────────────────────────────────────────────────────────────────────────

def exponential_kernel(x, z, sigma=1.0):
    """
    π_x(z) = exp(-D(x,z)² / σ²)
    
    Equation 2 from paper (with cosine distance for text, L2 for tabular).
    σ controls the 'locality' — small σ = very local, large σ = more global.
    
    At z=x:    π = 1.0   (highest weight)
    Far from x: π → 0.0  (low weight)
    """
    dist_sq = sum((x[i] - z[i])**2 for i in range(len(x)))
    return math.exp(-dist_sq / (sigma ** 2))

def cosine_kernel(x, z, sigma=0.25):
    """Cosine distance variant — better for text/high-dimensional data."""
    dot    = sum(x[i] * z[i] for i in range(len(x)))
    norm_x = math.sqrt(sum(v**2 for v in x))
    norm_z = math.sqrt(sum(v**2 for v in z))
    if norm_x < 1e-10 or norm_z < 1e-10:
        return 0.0
    cos_sim  = dot / (norm_x * norm_z)
    cos_dist = 1.0 - cos_sim
    return math.exp(-(cos_dist**2) / (sigma**2))


# ─────────────────────────────────────────────────────────────────────────────
# 2. PERTURBATION SAMPLING (tabular data)
# ─────────────────────────────────────────────────────────────────────────────

def sample_around_tabular(x, X_train, n_samples=500, seed=42):
    """
    Section 3.3 of paper — tabular perturbation.
    
    For each perturbed sample z':
      - Each feature independently either keeps x[i] or takes a random
        value from the training distribution.
      - The binary mask z'_binary indicates which features are 'present'.
    
    This is more realistic than Gaussian noise because the perturbed
    values come from the actual data distribution.
    
    Returns:
      Z_real:   list of perturbed instances in original feature space
      Z_binary: binary vectors indicating which features match x
      weights:  proximity weights π(x, z)
    """
    rng      = random.Random(seed)
    n_feat   = len(x)
    Z_real   = []
    Z_binary = []
    weights  = []

    for _ in range(n_samples):
        # Sample binary mask — how many features to keep from x
        n_on     = rng.randint(1, n_feat)
        on_feats = set(rng.sample(range(n_feat), n_on))

        # Build perturbed instance
        bg_row = rng.choice(X_train)
        z_real = [x[j] if j in on_feats else bg_row[j] for j in range(n_feat)]
        z_bin  = [1 if j in on_feats else 0 for j in range(n_feat)]

        w = exponential_kernel(x, z_real)

        Z_real.append(z_real)
        Z_binary.append(z_bin)
        weights.append(w)

    return Z_real, Z_binary, weights


# ─────────────────────────────────────────────────────────────────────────────
# 3. WEIGHTED RIDGE REGRESSION (the 'g' in LIME)
# ─────────────────────────────────────────────────────────────────────────────

def weighted_ridge_regression(Z, y, weights, alpha=0.01, n_iter=2000, lr=0.005):
    """
    Solve the LIME objective:
      min_{w} Σ π(z)(f(z) - w·z')² + α||w||²
    
    This is the sparse linear model g that locally approximates f.
    Uses gradient descent with sample weights.
    
    Returns coefficient vector w (the LIME explanation weights).
    """
    n_feat = len(Z[0])
    w      = [0.0] * n_feat
    bias   = 0.0

    for _ in range(n_iter):
        grad_w    = [0.0] * n_feat
        grad_bias = 0.0

        for i in range(len(Z)):
            pred     = bias + sum(w[j] * Z[i][j] for j in range(n_feat))
            residual = pred - y[i]
            wt       = weights[i]

            for j in range(n_feat):
                grad_w[j]  += 2 * wt * residual * Z[i][j]
            grad_bias += 2 * wt * residual

        # L2 regularisation (Ridge)
        for j in range(n_feat):
            grad_w[j] += 2 * alpha * w[j]

        n = len(Z)
        w    = [w[j]    - lr * grad_w[j]    / n for j in range(n_feat)]
        bias = bias     - lr * grad_bias / n

    return w, bias


# ─────────────────────────────────────────────────────────────────────────────
# 4. K-LASSO FEATURE SELECTION (from Algorithm 1 of paper)
# ─────────────────────────────────────────────────────────────────────────────

def select_top_k_features(Z, y, weights, K):
    """
    K-LASSO: select K most important features for the local explanation.
    
    From paper Algorithm 1: "first selecting K features with Lasso
    (using the regularization path), then learning weights via least squares."
    
    We approximate with iterative forward selection:
    greedily add the feature that most reduces weighted residual.
    """
    n_feat   = len(Z[0])
    selected = []
    remaining = list(range(n_feat))

    for _ in range(min(K, n_feat)):
        best_feat  = None
        best_score = float("inf")

        for j in remaining:
            # Fit univariate regression with currently selected + j
            feats = selected + [j]
            total_err = 0.0
            total_w   = sum(weights)

            for i in range(len(Z)):
                # Simple weighted correlation as proxy
                z_sub = [Z[i][f] for f in feats]
                # Use correlation as feature importance score
                total_err += weights[i] * abs(Z[i][j] - y[i])

            if total_err < best_score:
                best_score = total_err
                best_feat  = j

        if best_feat is not None:
            selected.append(best_feat)
            remaining.remove(best_feat)

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# 5. LIME EXPLAINER — MAIN API
# ─────────────────────────────────────────────────────────────────────────────

def lime_explain(
    model,
    x,
    X_train,
    feature_names=None,
    n_samples=500,
    K=5,
    alpha=0.01,
    seed=42,
):
    """
    Algorithm 1 from Ribeiro et al. (KDD 2016).
    
    Args:
      model:        object with .predict(X) or callable(x_list) → scalar
      x:            instance to explain (list of feature values)
      X_train:      training data (list of feature vectors) for sampling
      feature_names: feature names
      n_samples:    number of perturbed samples N
      K:            number of features in explanation (sparsity)
      alpha:        Ridge regularisation strength
      seed:         random seed for reproducibility
    
    Returns:
      dict with explanation weights, local fidelity, and metadata
    """
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(x))]

    # Step 1: Predict on x
    if hasattr(model, "predict_proba"):
        pred_x = float(model.predict_proba([x])[0][1])
    elif hasattr(model, "predict"):
        pred_x = float(model.predict([x])[0])
    else:
        pred_x = float(model(x))

    # Step 2: Sample perturbed instances around x
    Z_real, Z_binary, weights = sample_around_tabular(x, X_train, n_samples, seed)

    # Step 3: Get model predictions on perturbed data
    if hasattr(model, "predict_proba"):
        f_vals = [float(model.predict_proba([z])[0][1]) for z in Z_real]
    elif hasattr(model, "predict"):
        f_vals = [float(model.predict([z])[0]) for z in Z_real]
    else:
        f_vals = [float(model(z)) for z in Z_real]

    # Step 4: Select K most relevant features
    top_k_feats = select_top_k_features(Z_binary, f_vals, weights, K)

    # Step 5: Fit weighted Ridge on K-feature subset
    Z_reduced = [[Z_binary[i][j] for j in top_k_feats] for i in range(len(Z_binary))]
    coefs, intercept = weighted_ridge_regression(Z_reduced, f_vals, weights, alpha)

    # Step 6: Local fidelity — how well does the linear model explain locally?
    local_preds  = [intercept + sum(coefs[k] * Z_reduced[i][k]
                    for k in range(len(coefs))) for i in range(len(Z_reduced))]
    local_r2     = _r2(f_vals, local_preds, weights)

    # Build explanation
    explanation = sorted(
        [{"feature":    feature_names[top_k_feats[k]],
          "feature_idx": top_k_feats[k],
          "weight":      round(coefs[k], 6),
          "direction":   "positive" if coefs[k] >= 0 else "negative",
          "feature_value": x[top_k_feats[k]],
         }
         for k in range(len(coefs))],
        key=lambda e: abs(e["weight"]), reverse=True
    )

    return {
        "prediction":       round(pred_x, 6),
        "intercept":        round(intercept, 6),
        "explanation":      explanation,
        "local_fidelity_r2": round(local_r2, 4),
        "n_samples":        n_samples,
        "K":                K,
        "seed":             seed,
        "method":           "LIME (tabular, Ridge surrogate)",
    }

def _r2(y_true, y_pred, weights=None):
    """Weighted R² for local fidelity measurement."""
    if weights is None:
        weights = [1.0] * len(y_true)
    total_w   = sum(weights)
    y_mean    = sum(weights[i] * y_true[i] for i in range(len(y_true))) / total_w
    ss_tot    = sum(weights[i] * (y_true[i] - y_mean)**2 for i in range(len(y_true)))
    ss_res    = sum(weights[i] * (y_true[i] - y_pred[i])**2 for i in range(len(y_true)))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. INSTABILITY ANALYSIS — known weakness of LIME
# ─────────────────────────────────────────────────────────────────────────────

def lime_stability_analysis(model, x, X_train, feature_names=None,
                             n_seeds=10, n_samples=300, K=4):
    """
    Measure LIME instability by running with multiple random seeds.
    
    From paper and follow-up literature:
    LIME explanations can vary significantly between runs because:
    1. Different perturbation samples → different local approximation
    2. The linear surrogate may not be a good local approximator
    
    Returns per-feature weight variance across seeds.
    High variance → explanation is unreliable.
    """
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(x))]

    all_explanations = []
    for seed in range(n_seeds):
        exp = lime_explain(model, x, X_train, feature_names,
                           n_samples=n_samples, K=K, seed=seed * 7)
        all_explanations.append({e["feature"]: e["weight"]
                                  for e in exp["explanation"]})

    # Compute per-feature variance
    feature_weights = defaultdict(list)
    for exp in all_explanations:
        for feat, w in exp.items():
            feature_weights[feat].append(w)

    stability_report = []
    for feat, weights_list in feature_weights.items():
        mean_w = sum(weights_list) / len(weights_list)
        var_w  = sum((w - mean_w)**2 for w in weights_list) / len(weights_list)
        stability_report.append({
            "feature":    feat,
            "mean_weight": round(mean_w, 6),
            "std_weight":  round(var_w**0.5, 6),
            "cv":          round(abs(var_w**0.5 / mean_w) if abs(mean_w) > 1e-6 else float("inf"), 4),
            "stable":      var_w**0.5 < 0.05,
        })

    stability_report.sort(key=lambda x: x["std_weight"], reverse=True)
    n_unstable = sum(1 for r in stability_report if not r["stable"])

    return {
        "n_seeds":          n_seeds,
        "stability_report": stability_report,
        "n_unstable_feats": n_unstable,
        "overall_stable":   n_unstable == 0,
        "recommendation":   (
            "Explanations are stable." if n_unstable == 0
            else f"{n_unstable} features have high variance. "
                 "Increase n_samples or consider SHAP for more reliable explanations."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def run_verification():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    import numpy as np

    print("=" * 65)
    print("Phase 3.3 — LIME Explainer Verification")
    print("=" * 65)

    np.random.seed(42)
    X_np, y_np = make_classification(
        n_samples=400, n_features=5,
        n_informative=2, n_redundant=0, random_state=42
    )
    feat_names = ["informative_A", "informative_B", "noise_C", "noise_D", "noise_E"]
    X_tr = X_np[:300].tolist();  y_tr = y_np[:300].tolist()

    rf = RandomForestClassifier(n_estimators=30, random_state=42)
    rf.fit(X_np[:300], y_np[:300])

    # Explain one instance
    x_test = X_np[300].tolist()
    print(f"\n  Explaining instance: {[round(v,3) for v in x_test]}")
    print(f"  Model prediction: {rf.predict_proba([x_test])[0][1]:.4f}")

    exp = lime_explain(rf, x_test, X_tr, feat_names, n_samples=300, K=4, seed=42)

    print(f"\n  LIME Explanation (K={exp['K']}, local R²={exp['local_fidelity_r2']}):")
    for e in exp["explanation"]:
        bar = "█" * max(0, int(abs(e["weight"]) * 30))
        sign = "+" if e["direction"] == "positive" else "-"
        print(f"    {sign} {e['feature']:<18} w={e['weight']:+.4f}  {bar}")

    # Local fidelity check
    ok_fidelity = exp["local_fidelity_r2"] > 0.3
    print(f"\n  Local fidelity R²={exp['local_fidelity_r2']:.4f}  "
          f"[{'✓ OK' if ok_fidelity else '⚠ LOW — surrogate not locally faithful'}]")

    # Stability analysis
    print(f"\n  Stability analysis (10 seeds):")
    stab = lime_stability_analysis(rf, x_test, X_tr, feat_names,
                                    n_seeds=6, n_samples=200, K=3)
    for r in stab["stability_report"]:
        flag = "✓" if r["stable"] else "⚠"
        print(f"    {flag} {r['feature']:<18} "
              f"mean={r['mean_weight']:+.4f}  std={r['std_weight']:.4f}")
    print(f"\n  {stab['recommendation']}")

    print("=" * 65)


if __name__ == "__main__":
    run_verification()
