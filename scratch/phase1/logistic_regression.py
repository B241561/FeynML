"""
Phase 1.2 — Logistic Regression from Scratch
==============================================
Topics:
  • Sigmoid function (log-odds derivation)
  • Binary Cross-Entropy loss (from MLE)
  • Gradient Descent for logistic regression
  • Softmax for multiclass
  • Decision boundary intuition
  • Verification against scikit-learn
"""

import math
import random


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def vec_add(a, b):
    return [x + y for x, y in zip(a, b)]

def vec_sub(a, b):
    return [x - y for x, y in zip(a, b)]

def vec_scale(v, s):
    return [x * s for x in v]

def mean(v):
    return sum(v) / len(v)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SIGMOID & SOFTMAX
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid(z):
    """
    σ(z) = 1 / (1 + e^{-z})
    
    Derivation from log-odds:
      log(p / (1-p)) = z  →  p = e^z / (1 + e^z) = 1 / (1 + e^{-z})
    
    Properties:
      σ(0) = 0.5   (decision boundary at z=0)
      σ(z) → 1 as z → +∞
      σ(z) → 0 as z → -∞
      σ'(z) = σ(z)(1 - σ(z))   ← used in backprop
    
    Numerical stability: clip z to avoid overflow.
    """
    z = max(-500, min(500, z))   # clip for numerical stability
    return 1.0 / (1.0 + math.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def softmax(z_vec):
    """
    Multiclass generalization of sigmoid.
    softmax(z)_k = e^{z_k} / Σ_j e^{z_j}
    
    Numerical trick: subtract max(z) before exp to prevent overflow.
    Output: probability vector summing to 1.
    """
    max_z = max(z_vec)
    exps = [math.exp(z - max_z) for z in z_vec]
    total = sum(exps)
    return [e / total for e in exps]


# ─────────────────────────────────────────────────────────────────────────────
# 2. BINARY LOGISTIC REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

class LogisticRegression:
    """
    Binary classification: P(y=1|x) = σ(wᵀx + b)
    
    Loss — Binary Cross-Entropy (derived from MLE):
      L = -(1/n) Σ [yi*log(ŷi) + (1-yi)*log(1-ŷi)]
    
    Gradient derivation:
      ∂L/∂w = (1/n) Xᵀ(ŷ - y)    ← remarkably clean!
      ∂L/∂b = (1/n) Σ(ŷi - yi)
    
    This is the same form as linear regression — only the activation differs.
    
    Decision boundary: wᵀx + b = 0  (linear hyperplane in feature space)
    """

    def __init__(self, lr=0.1, epochs=1000, tol=1e-6, verbose=False):
        self.lr = lr
        self.epochs = epochs
        self.tol = tol
        self.verbose = verbose
        self.w = []
        self.b = 0.0
        self.loss_history = []

    def _bce_loss(self, probs, y):
        eps = 1e-12
        n = len(y)
        return -sum(
            y[i] * math.log(probs[i] + eps) + (1 - y[i]) * math.log(1 - probs[i] + eps)
            for i in range(n)
        ) / n

    def fit(self, X, y):
        n, d = len(X), len(X[0])
        # Initialize weights near zero (important for symmetry breaking)
        rng = random.Random(42)
        self.w = [rng.gauss(0, 0.01) for _ in range(d)]
        self.b = 0.0

        for epoch in range(self.epochs):
            # Forward pass
            logits = [dot(X[i], self.w) + self.b for i in range(n)]
            probs  = [sigmoid(z) for z in logits]
            errors = [probs[i] - y[i] for i in range(n)]

            # Gradients
            grad_w = [sum(errors[i] * X[i][j] for i in range(n)) / n for j in range(d)]
            grad_b = sum(errors) / n

            # Update
            self.w = vec_sub(self.w, vec_scale(grad_w, self.lr))
            self.b -= self.lr * grad_b

            loss = self._bce_loss(probs, y)
            self.loss_history.append(loss)

            if self.verbose and epoch % 100 == 0:
                print(f"  Epoch {epoch:4d} | BCE Loss={loss:.6f}")

            # Early stopping
            if len(self.loss_history) > 1:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                    break

        return self

    def predict_proba(self, X):
        logits = [dot(X[i], self.w) + self.b for i in range(len(X))]
        return [sigmoid(z) for z in logits]

    def predict(self, X, threshold=0.5):
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(1 for i in range(len(y)) if preds[i] == y[i]) / len(y)


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOGISTIC REGRESSION WITH L2 REGULARIZATION
# ─────────────────────────────────────────────────────────────────────────────

class LogisticRegressionL2:
    """
    Adds L2 penalty: L_total = BCE + (λ/2n) Σ wj²
    
    Regularized gradient:
      ∂L/∂wj = (1/n) Σ(ŷi - yi)*xij + (λ/n)*wj
    
    Effect: prevents overfitting, especially with many features.
    λ=0: plain logistic regression
    λ→∞: weights → 0 (underfit)
    """

    def __init__(self, lr=0.1, epochs=1000, C=1.0):
        """C = 1/λ (sklearn convention — larger C = less regularization)."""
        self.lr = lr
        self.epochs = epochs
        self.lam = 1.0 / C
        self.w = []
        self.b = 0.0

    def fit(self, X, y):
        n, d = len(X), len(X[0])
        rng = random.Random(42)
        self.w = [rng.gauss(0, 0.01) for _ in range(d)]
        self.b = 0.0

        for _ in range(self.epochs):
            logits = [dot(X[i], self.w) + self.b for i in range(n)]
            probs  = [sigmoid(z) for z in logits]
            errors = [probs[i] - y[i] for i in range(n)]

            grad_w = [
                sum(errors[i] * X[i][j] for i in range(n)) / n + self.lam * self.w[j] / n
                for j in range(d)
            ]
            grad_b = sum(errors) / n

            self.w = vec_sub(self.w, vec_scale(grad_w, self.lr))
            self.b -= self.lr * grad_b

        return self

    def predict_proba(self, X):
        logits = [dot(X[i], self.w) + self.b for i in range(len(X))]
        return [sigmoid(z) for z in logits]

    def predict(self, X, threshold=0.5):
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(1 for i in range(len(y)) if preds[i] == y[i]) / len(y)


# ─────────────────────────────────────────────────────────────────────────────
# 4. MULTICLASS LOGISTIC REGRESSION (Softmax / One-vs-Rest)
# ─────────────────────────────────────────────────────────────────────────────

class SoftmaxRegression:
    """
    Generalization to K classes using softmax output.
    
    Model: P(y=k|x) = softmax(Wx + b)_k
    Loss : Categorical Cross-Entropy = -(1/n) Σ Σ y_ik * log(ŷ_ik)
    
    Gradient (w.r.t. Wk):
      ∂L/∂Wk = (1/n) Xᵀ(ŷk - yk)
    
    One-hot encode targets before training.
    """

    def __init__(self, lr=0.1, epochs=500, n_classes=3):
        self.lr = lr
        self.epochs = epochs
        self.n_classes = n_classes
        self.W = []   # shape: K × d
        self.b = []   # shape: K

    def _one_hot(self, y, K):
        n = len(y)
        oh = [[0.0] * K for _ in range(n)]
        for i in range(n):
            oh[i][int(y[i])] = 1.0
        return oh

    def fit(self, X, y):
        n, d = len(X), len(X[0])
        K = self.n_classes
        rng = random.Random(42)
        self.W = [[rng.gauss(0, 0.01) for _ in range(d)] for _ in range(K)]
        self.b = [0.0] * K
        Y_oh = self._one_hot(y, K)

        for _ in range(self.epochs):
            # Forward: compute logits and softmax probs
            probs = []
            for i in range(n):
                logits = [dot(self.W[k], X[i]) + self.b[k] for k in range(K)]
                probs.append(softmax(logits))

            # Gradients for each class k
            for k in range(K):
                errors = [probs[i][k] - Y_oh[i][k] for i in range(n)]
                for j in range(d):
                    self.W[k][j] -= self.lr * sum(errors[i] * X[i][j] for i in range(n)) / n
                self.b[k] -= self.lr * sum(errors) / n

        return self

    def predict_proba(self, X):
        result = []
        for i in range(len(X)):
            logits = [dot(self.W[k], X[i]) + self.b[k] for k in range(self.n_classes)]
            result.append(softmax(logits))
        return result

    def predict(self, X):
        probs = self.predict_proba(X)
        return [max(range(self.n_classes), key=lambda k: p[k]) for p in probs]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(1 for i in range(len(y)) if preds[i] == y[i]) / len(y)


# ─────────────────────────────────────────────────────────────────────────────
# 5. VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def _make_binary_data(n=300, d=3, seed=42):
    rng = random.Random(seed)
    true_w = [1.5, -2.0, 0.5]
    X = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n)]
    y = [1 if dot(X[i], true_w) + rng.gauss(0, 0.5) > 0 else 0 for i in range(n)]
    return X, y

def run_verification():
    import sklearn.linear_model as skl
    import numpy as np

    X, y = _make_binary_data()
    X_np, y_np = np.array(X), np.array(y)

    print("=" * 60)
    print("Phase 1.2 — Logistic Regression Verification")
    print("=" * 60)

    # --- Sigmoid ---
    for z, expected in [(0, 0.5), (100, 1.0), (-100, 0.0)]:
        s = sigmoid(z)
        ok = abs(s - expected) < 0.01
        print(f"  sigmoid({z:4d}) = {s:.4f}  (expected ~{expected})  [{'✓ PASS' if ok else '✗ FAIL'}]")

    # --- Binary Logistic Regression ---
    lr = LogisticRegression(lr=0.1, epochs=1000).fit(X, y)
    sk = skl.LogisticRegression(max_iter=1000, solver='lbfgs').fit(X_np, y_np)
    acc_ours = lr.score(X, y)
    acc_sk   = sk.score(X_np, y_np)
    ok = abs(acc_ours - acc_sk) < 0.05
    print(f"  BinaryLogistic   acc={acc_ours:.4f}  sk={acc_sk:.4f}  [{'✓ PASS' if ok else '✗ FAIL'}]")

    # --- L2 Regularized ---
    lr2 = LogisticRegressionL2(lr=0.1, epochs=1000, C=1.0).fit(X, y)
    sk2 = skl.LogisticRegression(C=1.0, max_iter=1000).fit(X_np, y_np)
    ok = abs(lr2.score(X, y) - sk2.score(X_np, y_np)) < 0.05
    print(f"  LogisticL2(C=1)  acc={lr2.score(X,y):.4f}  sk={sk2.score(X_np,y_np):.4f}  [{'✓ PASS' if ok else '✗ FAIL'}]")

    # --- Softmax ---
    from sklearn.datasets import make_classification
    Xm, ym = make_classification(n_samples=300, n_features=4, n_classes=3,
                                  n_informative=3, random_state=42)
    Xm_l = Xm.tolist()
    ym_l = ym.tolist()
    sm = SoftmaxRegression(lr=0.3, epochs=500, n_classes=3).fit(Xm_l, ym_l)
    sk_sm = skl.LogisticRegression(multi_class='multinomial', max_iter=1000).fit(Xm, ym)
    acc_sm    = sm.score(Xm_l, ym_l)
    acc_sk_sm = sk_sm.score(Xm, ym)
    ok = abs(acc_sm - acc_sk_sm) < 0.10
    print(f"  SoftmaxMulticlass acc={acc_sm:.4f}  sk={acc_sk_sm:.4f}  [{'✓ PASS' if ok else '✗ FAIL'}]")

    print()
    print("  Key intuitions:")
    print("  • Sigmoid maps any real number to (0,1) — models probability")
    print("  • BCE loss = negative log-likelihood → MLE for Bernoulli dist")
    print("  • Decision boundary is LINEAR in feature space")
    print("  • Softmax = sigmoid generalized to K classes")
    print("=" * 60)


if __name__ == "__main__":
    run_verification()
