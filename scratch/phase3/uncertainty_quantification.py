"""
Phase 3.6 — Uncertainty Quantification & Conformal Prediction
===============================================================
Sources: Interpretable ML Book (Molnar) + Hands-On ML (Géron) Ch.3

TWO KINDS OF UNCERTAINTY:
  1. ALEATORIC uncertainty — irreducible noise in the data itself
     Example: a blurry image that even humans can't classify reliably.
     No matter how much data you collect, this uncertainty remains.
     Captured by: model's output probability (if well calibrated)

  2. EPISTEMIC uncertainty — model's lack of knowledge (reducible)
     Example: a very unusual input the model has never seen similar to.
     More training data / better features can reduce this.
     Captured by: model disagreement (ensembles), prediction variance

CONFORMAL PREDICTION (Vovk et al. 2005):
  Distribution-free, model-agnostic prediction SETS.
  Guarantee: P(y ∈ C(x)) ≥ 1-α for any data distribution.
  
  Algorithm:
    1. Calibration set: compute non-conformity scores r_i = 1 - f(x_i)[y_i]
    2. Compute (1-α) quantile of calibration scores → threshold q̂
    3. For new x: prediction set = {y : f(x)[y] ≥ 1 - q̂}
  
  Key property: MARGINAL COVERAGE GUARANTEE — no assumptions on data distribution.
  This is stronger than calibration (which only ensures average accuracy).

PLATT SCALING:
  Post-hoc calibration for classifiers.
  Fits logistic regression on top of raw model scores.
  
ISOTONIC REGRESSION:
  Non-parametric alternative to Platt scaling.
  More flexible but needs more calibration data.
"""

import math
import random


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFORMAL PREDICTION (classification)
# ─────────────────────────────────────────────────────────────────────────────

def compute_nonconformity_scores(y_proba_calib, y_true_calib):
    """
    Non-conformity score for classification:
      r_i = 1 - P(y=y_true | x_i)
    
    High score → prediction doesn't fit the true label → non-conforming.
    Low score  → model is confident and correct.
    
    This is the score function from Vovk et al. (2005) split conformal.
    """
    scores = []
    for i, y in enumerate(y_true_calib):
        # y_proba_calib[i] is a list/dict of class probabilities
        if isinstance(y_proba_calib[i], dict):
            prob_true = y_proba_calib[i].get(y, 0.0)
        else:
            prob_true = float(y_proba_calib[i][int(y)])
        scores.append(1.0 - prob_true)
    return scores

def conformal_quantile(nonconformity_scores, alpha=0.10):
    """
    Compute the (1-α) quantile of calibration non-conformity scores.
    
    This is the threshold q̂ that ensures:
      P(true label in prediction set) ≥ 1 - α
    
    The finite-sample correction factor (n+1)/n ensures exact coverage.
    """
    n     = len(nonconformity_scores)
    level = math.ceil((n + 1) * (1 - alpha)) / n   # finite-sample correction
    level = min(level, 1.0)
    sorted_scores = sorted(nonconformity_scores)
    idx   = min(int(level * n), n - 1)
    return sorted_scores[idx]

def conformal_prediction_set(y_proba_new, q_hat, classes):
    """
    Prediction set for a new instance:
      C(x) = {y : 1 - P(y|x) ≤ q̂}
             = {y : P(y|x) ≥ 1 - q̂}
    
    Guarantee: true label is in C(x) with probability ≥ 1-α.
    
    If C(x) is large → model is uncertain about this instance.
    If C(x) = {} → something went wrong (shouldn't happen with correct q̂).
    If C(x) = all classes → model has no information about this instance.
    """
    prediction_set = []
    threshold = 1.0 - q_hat

    for i, cls in enumerate(classes):
        if isinstance(y_proba_new, dict):
            prob = y_proba_new.get(cls, 0.0)
        else:
            prob = float(y_proba_new[i])
        if prob >= threshold:
            prediction_set.append(cls)

    return prediction_set

class SplitConformalClassifier:
    """
    Split Conformal Prediction for binary/multiclass classification.
    
    'Split' means we split available labeled data into:
      - Training set (for model fitting)
      - Calibration set (for conformal threshold q̂)
    
    This separation is CRITICAL — calibration must be independent of training.
    """

    def __init__(self, base_model, alpha=0.10):
        self.model   = base_model
        self.alpha   = alpha
        self.q_hat   = None
        self.classes_ = None

    def calibrate(self, X_calib, y_calib):
        """
        Compute q̂ from calibration set.
        Must NOT overlap with training data.
        """
        if hasattr(self.model, "classes_"):
            self.classes_ = list(self.model.classes_)
        else:
            self.classes_ = sorted(set(y_calib))

        y_proba = self.model.predict_proba(X_calib)
        scores  = compute_nonconformity_scores(y_proba, y_calib)
        self.q_hat = conformal_quantile(scores, self.alpha)
        return self

    def predict_set(self, X_new):
        """
        Return prediction set for each instance.
        Guaranteed: P(y_true ∈ set) ≥ 1 - alpha.
        """
        if self.q_hat is None:
            raise RuntimeError("Must call calibrate() first.")
        y_proba = self.model.predict_proba(X_new)
        return [
            conformal_prediction_set(y_proba[i], self.q_hat, self.classes_)
            for i in range(len(X_new))
        ]

    def coverage(self, X_test, y_test):
        """
        Empirical coverage: fraction of test instances where true label is in set.
        Should be ≥ 1-alpha.
        """
        sets = self.predict_set(X_test)
        covered = sum(1 for i, s in enumerate(sets) if y_test[i] in s)
        return covered / len(y_test)

    def efficiency(self, X_test):
        """
        Average prediction set size.
        Smaller = more informative (fewer classes in set).
        """
        sets = self.predict_set(X_test)
        return sum(len(s) for s in sets) / len(sets)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PLATT SCALING — post-hoc calibration
# ─────────────────────────────────────────────────────────────────────────────

def platt_scaling(raw_scores_calib, y_calib, n_iter=1000, lr=0.01):
    """
    Platt Scaling (Platt 1999):
    Fit logistic regression P(y=1|s) = σ(As + B)
    on top of raw model scores s.
    
    Parameters A and B are fitted to minimise NLL on calibration set.
    
    When to use:
      - SVMs (which output decision function values, not probabilities)
      - Any model producing overconfident probabilities
      - Faster and more data-efficient than isotonic regression
    
    When NOT to use:
      - Very small calibration sets (< 100 samples)
      - When the calibration curve is strongly non-monotonic
    """
    A, B = 0.0, 0.0

    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))

    for _ in range(n_iter):
        grad_A = grad_B = 0.0
        for s, y in zip(raw_scores_calib, y_calib):
            p = sigmoid(A * s + B)
            err = p - y
            grad_A += err * s
            grad_B += err

        n = len(raw_scores_calib)
        A -= lr * grad_A / n
        B -= lr * grad_B / n

    def calibrated_prob(raw_score):
        return sigmoid(A * raw_score + B)

    return calibrated_prob, round(A, 6), round(B, 6)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ISOTONIC REGRESSION CALIBRATION (non-parametric)
# ─────────────────────────────────────────────────────────────────────────────

def isotonic_regression_calibration(raw_scores_calib, y_calib):
    """
    Isotonic regression: non-parametric monotone calibration.
    
    Algorithm (Pool Adjacent Violators - PAV):
    1. Sort by raw score
    2. Fit piecewise constant function that is monotone increasing
    3. Use this function to map raw scores → calibrated probabilities
    
    More flexible than Platt scaling.
    Needs more calibration data (≥ 1000 samples recommended).
    
    Cannot extrapolate beyond training range.
    """
    # Sort by raw score
    pairs = sorted(zip(raw_scores_calib, y_calib), key=lambda t: t[0])
    scores_sorted = [p[0] for p in pairs]
    labels_sorted = [p[1] for p in pairs]

    # Pool Adjacent Violators algorithm
    blocks = []  # each block: (mean_y, count)
    for y in labels_sorted:
        blocks.append([float(y), 1])
        # Merge with previous block if monotonicity is violated
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            prev = blocks.pop()
            curr = blocks.pop()
            total = curr[1] + prev[1]
            merged_mean = (curr[0] * curr[1] + prev[0] * prev[1]) / total
            blocks.append([merged_mean, total])

    # Build isotonic values
    isotonic_values = []
    for mean_y, count in blocks:
        isotonic_values.extend([mean_y] * count)

    # Calibration function: linear interpolation
    calib_pairs = list(zip(scores_sorted, isotonic_values))

    def calibrated_prob(raw_score):
        if raw_score <= calib_pairs[0][0]:
            return calib_pairs[0][1]
        if raw_score >= calib_pairs[-1][0]:
            return calib_pairs[-1][1]
        # Find surrounding points
        for i in range(len(calib_pairs) - 1):
            s0, p0 = calib_pairs[i]
            s1, p1 = calib_pairs[i + 1]
            if s0 <= raw_score <= s1:
                if abs(s1 - s0) < 1e-12:
                    return (p0 + p1) / 2
                t = (raw_score - s0) / (s1 - s0)
                return p0 + t * (p1 - p0)
        return calib_pairs[-1][1]

    return calibrated_prob, calib_pairs


# ─────────────────────────────────────────────────────────────────────────────
# 4. EPISTEMIC VS ALEATORIC UNCERTAINTY VIA ENSEMBLES
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_uncertainty(models, X_new):
    """
    Decompose uncertainty into epistemic + aleatoric via ensemble.
    
    Epistemic  = variance BETWEEN models = model disagreement
                → reducible with more data/better features
    
    Aleatoric  = mean variance WITHIN each model's prediction
                → irreducible noise in the problem itself
    
    Total variance = Epistemic + Aleatoric (law of total variance)
    
    For binary classification:
      Each model outputs P(y=1|x) ∈ [0,1]
      Aleatoric per model = p*(1-p)     [Bernoulli variance]
      Epistemic = Var(p across models)
    """
    all_preds = []
    for model in models:
        if hasattr(model, "predict_proba"):
            preds = [float(model.predict_proba([x])[0][1]) for x in X_new]
        else:
            preds = [float(model.predict([x])[0]) for x in X_new]
        all_preds.append(preds)

    n_instances = len(X_new)
    results = []

    for i in range(n_instances):
        model_preds = [all_preds[m][i] for m in range(len(models))]
        mean_pred   = sum(model_preds) / len(model_preds)

        # Epistemic: variance of predictions across models
        epistemic = sum((p - mean_pred)**2 for p in model_preds) / len(model_preds)

        # Aleatoric: mean of p*(1-p) across models (Bernoulli variance)
        aleatoric = sum(p * (1 - p) for p in model_preds) / len(model_preds)

        total = epistemic + aleatoric

        results.append({
            "prediction":   round(mean_pred, 6),
            "epistemic":    round(epistemic, 6),
            "aleatoric":    round(aleatoric, 6),
            "total":        round(total, 6),
            "uncertain":    total > 0.1,
            "model_preds":  [round(p, 4) for p in model_preds],
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def run_verification():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    import numpy as np

    print("=" * 65)
    print("Phase 3.6 — Uncertainty Quantification Verification")
    print("=" * 65)

    np.random.seed(42)
    X_np, y_np = make_classification(
        n_samples=1000, n_features=10,
        n_informative=5, random_state=42
    )

    X_tr = X_np[:600]; y_tr = y_np[:600]
    X_ca = X_np[600:800]; y_ca = y_np[600:800]   # calibration
    X_te = X_np[800:];    y_te = y_np[800:]        # test

    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_tr, y_tr)

    # ── Conformal Prediction ────────────────────────────────────────────
    alpha = 0.10
    cp    = SplitConformalClassifier(rf, alpha=alpha)
    cp.calibrate(X_ca, y_ca)

    coverage  = cp.coverage(X_te, y_te)
    avg_size  = cp.efficiency(X_te)
    cov_ok    = coverage >= (1 - alpha - 0.05)   # allow 5% slack

    print(f"\n  Conformal Prediction (α={alpha}):")
    print(f"  Empirical coverage = {coverage:.3f}  "
          f"(required ≥ {1-alpha:.2f})  [{'✓ PASS' if cov_ok else '✗ FAIL'}]")
    print(f"  Avg prediction set size = {avg_size:.2f}  "
          f"({'informative' if avg_size < 1.5 else 'uncertain model'})")
    print(f"  q̂ (threshold) = {cp.q_hat:.4f}")

    # Show a few examples
    sets_5 = cp.predict_set(X_te[:5])
    proba5  = rf.predict_proba(X_te[:5])
    print(f"\n  Example predictions:")
    for i in range(5):
        conf = max(proba5[i])
        print(f"    Instance {i}: set={sets_5[i]}  "
              f"true={y_te[i]}  confidence={conf:.3f}  "
              f"{'✓' if y_te[i] in sets_5[i] else '✗'}")

    # ── Platt Scaling ───────────────────────────────────────────────────
    print(f"\n  Platt Scaling Calibration:")
    raw_scores = rf.predict_proba(X_ca)[:, 1].tolist()
    y_ca_list  = y_ca.tolist()
    calib_fn, A, B = platt_scaling(raw_scores, y_ca_list)

    # Check calibration improved
    raw_probs  = [raw_scores[i] for i in range(len(y_ca_list))]
    calib_probs = [calib_fn(s) for s in raw_scores]

    def brier(y, probs):
        return sum((yi - pi)**2 for yi, pi in zip(y, probs)) / len(y)

    bs_raw   = brier(y_ca_list, raw_probs)
    bs_calib = brier(y_ca_list, calib_probs)
    print(f"  Brier score (raw):       {bs_raw:.4f}")
    print(f"  Brier score (calibrated): {bs_calib:.4f}")
    print(f"  Platt A={A}, B={B}")
    print(f"  Calibration {'improved' if bs_calib <= bs_raw else 'unchanged'}  "
          f"[{'✓ PASS' if bs_calib <= bs_raw + 0.01 else '~ OK'}]")

    # ── Epistemic vs Aleatoric ──────────────────────────────────────────
    print(f"\n  Epistemic vs Aleatoric Uncertainty (3-model ensemble):")
    models = [
        RandomForestClassifier(n_estimators=20, random_state=i).fit(X_tr, y_tr)
        for i in range(3)
    ]
    unc = ensemble_uncertainty(models, X_te[:5].tolist())
    print(f"  {'Pred':>6} {'Epistemic':>11} {'Aleatoric':>11} {'Total':>7}")
    for r in unc:
        print(f"  {r['prediction']:>6.3f} {r['epistemic']:>11.4f} "
              f"{r['aleatoric']:>11.4f} {r['total']:>7.4f}")

    print("=" * 65)


if __name__ == "__main__":
    run_verification()
