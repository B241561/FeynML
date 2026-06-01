"""
Phase 1.1 — Linear Regression from Scratch
============================================
Topics:
  • Simple & Multiple Linear Regression
  • MSE derivation from MLE
  • Gradient Descent (Batch & Stochastic)
  • Normal Equation (closed-form solution)
  • Ridge (L2) & Lasso (L1) Regularization
  • Bias-Variance tradeoff
  • Verification against scikit-learn
"""

import math
import random
import copy


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def mat_mul(A, B):
    rA, cA = len(A), len(A[0])
    rB, cB = len(B), len(B[0])
    assert cA == rB
    C = [[0.0] * cB for _ in range(rA)]
    for i in range(rA):
        for j in range(cB):
            C[i][j] = sum(A[i][k] * B[k][j] for k in range(cA))
    return C

def mat_transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def mat_vec(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

def vec_add(a, b):
    return [x + y for x, y in zip(a, b)]

def vec_sub(a, b):
    return [x - y for x, y in zip(a, b)]

def vec_scale(v, s):
    return [x * s for x in v]

def mean(v):
    return sum(v) / len(v)

def mat_inverse_2x2(A):
    det = A[0][0]*A[1][1] - A[0][1]*A[1][0]
    return [[A[1][1]/det, -A[0][1]/det],
            [-A[1][0]/det, A[0][0]/det]]

def gaussian_solve(A, b):
    """Solve Ax=b via Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        max_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[max_row] = M[max_row], M[col]
        pivot = M[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Singular matrix")
        for row in range(col+1, n):
            f = M[row][col] / pivot
            for k in range(col, n+1):
                M[row][k] -= f * M[col][k]
    x = [0.0]*n
    for i in range(n-1, -1, -1):
        x[i] = M[i][n]
        for j in range(i+1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]
    return x


# ─────────────────────────────────────────────────────────────────────────────
# 1. SIMPLE LINEAR REGRESSION (closed-form)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleLinearRegression:
    """
    y = w*x + b
    
    Closed-form solution via Ordinary Least Squares (OLS):
      w = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
      b = ȳ - w * x̄
    
    Derived by setting dMSE/dw = 0 and dMSE/db = 0.
    """

    def __init__(self):
        self.w = 0.0
        self.b = 0.0

    def fit(self, X, y):
        """X: list of scalars, y: list of scalars."""
        n = len(X)
        x_bar = mean(X)
        y_bar = mean(y)

        num = sum((X[i] - x_bar) * (y[i] - y_bar) for i in range(n))
        den = sum((X[i] - x_bar)**2 for i in range(n))

        self.w = num / den if abs(den) > 1e-12 else 0.0
        self.b = y_bar - self.w * x_bar
        return self

    def predict(self, X):
        return [self.w * x + self.b for x in X]

    def score(self, X, y):
        preds = self.predict(X)
        ss_res = sum((y[i] - preds[i])**2 for i in range(len(y)))
        ss_tot = sum((y[i] - mean(y))**2 for i in range(len(y)))
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. MULTIPLE LINEAR REGRESSION — GRADIENT DESCENT
# ─────────────────────────────────────────────────────────────────────────────

class LinearRegressionGD:
    """
    Multiple Linear Regression via Batch Gradient Descent.
    
    Model: ŷ = Xw + b  (w is weight vector, b is bias scalar)
    Loss : MSE = (1/n) Σ (ŷi - yi)²
    
    Gradients:
      ∂L/∂w = (2/n) Xᵀ(ŷ - y)
      ∂L/∂b = (2/n) Σ(ŷi - yi)
    
    Update:
      w ← w - lr * ∂L/∂w
      b ← b - lr * ∂L/∂b
    """

    def __init__(self, lr=0.01, epochs=1000, verbose=False):
        self.lr = lr
        self.epochs = epochs
        self.verbose = verbose
        self.w = []
        self.b = 0.0
        self.loss_history = []

    def _mse(self, preds, y):
        return sum((preds[i] - y[i])**2 for i in range(len(y))) / len(y)

    def fit(self, X, y):
        """X: list of lists (n_samples × n_features), y: list."""
        n = len(X)
        d = len(X[0])
        self.w = [0.0] * d
        self.b = 0.0

        for epoch in range(self.epochs):
            # Forward pass
            preds = [dot(X[i], self.w) + self.b for i in range(n)]
            errors = [preds[i] - y[i] for i in range(n)]

            # Gradients
            grad_w = [2/n * sum(errors[i] * X[i][j] for i in range(n)) for j in range(d)]
            grad_b = 2/n * sum(errors)

            # Update
            self.w = vec_sub(self.w, vec_scale(grad_w, self.lr))
            self.b -= self.lr * grad_b

            loss = self._mse(preds, y)
            self.loss_history.append(loss)

            if self.verbose and epoch % 100 == 0:
                print(f"  Epoch {epoch:4d} | MSE={loss:.6f}")

        return self

    def predict(self, X):
        return [dot(X[i], self.w) + self.b for i in range(len(X))]

    def score(self, X, y):
        preds = self.predict(X)
        ss_res = sum((y[i] - preds[i])**2 for i in range(len(y)))
        ss_tot = sum((yi - mean(y))**2 for yi in y)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. NORMAL EQUATION (closed-form for multiple features)
# ─────────────────────────────────────────────────────────────────────────────

class LinearRegressionNormal:
    """
    Normal Equation: w = (XᵀX)⁻¹ Xᵀy
    
    Derivation:
      MSE = (1/n)||Xw - y||²
      Set gradient = 0: Xᵀ(Xw - y) = 0
      → XᵀXw = Xᵀy
      → w = (XᵀX)⁻¹ Xᵀy
    
    Adds bias column of 1s to X automatically.
    Time complexity: O(nd² + d³) — slow for large d.
    """

    def __init__(self):
        self.weights = []   # includes bias as weights[0]

    def _add_bias(self, X):
        return [[1.0] + row for row in X]

    def fit(self, X, y):
        Xb = self._add_bias(X)         # n × (d+1)
        Xt = mat_transpose(Xb)         # (d+1) × n
        XtX = mat_mul(Xt, Xb)          # (d+1) × (d+1)
        Xty = mat_vec(Xt, y)           # (d+1)
        self.weights = gaussian_solve(XtX, Xty)
        return self

    def predict(self, X):
        Xb = self._add_bias(X)
        return [dot(row, self.weights) for row in Xb]

    def score(self, X, y):
        preds = self.predict(X)
        ss_res = sum((y[i] - preds[i])**2 for i in range(len(y)))
        ss_tot = sum((yi - mean(y))**2 for yi in y)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. RIDGE REGRESSION (L2 Regularization)
# ─────────────────────────────────────────────────────────────────────────────

class RidgeRegression:
    """
    Ridge = OLS + L2 penalty: minimize ||Xw - y||² + α||w||²
    
    Closed form: w = (XᵀX + αI)⁻¹ Xᵀy
    
    Effect:
      • Shrinks all weights toward 0 but never to exactly 0
      • Always invertible (adds α to diagonal) → stable even when features correlated
      • Reduces variance at cost of slight bias → better generalization
    
    Use Ridge when: many small/medium effects, multicollinearity present.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = []

    def _add_bias(self, X):
        return [[1.0] + row for row in X]

    def fit(self, X, y):
        Xb = self._add_bias(X)
        Xt = mat_transpose(Xb)
        XtX = mat_mul(Xt, Xb)
        d = len(XtX)
        # Add α to diagonal (skip bias term at index 0)
        for i in range(1, d):
            XtX[i][i] += self.alpha
        Xty = mat_vec(Xt, y)
        self.weights = gaussian_solve(XtX, Xty)
        return self

    def predict(self, X):
        Xb = self._add_bias(X)
        return [dot(row, self.weights) for row in Xb]

    def score(self, X, y):
        preds = self.predict(X)
        ss_res = sum((y[i] - preds[i])**2 for i in range(len(y)))
        ss_tot = sum((yi - mean(y))**2 for yi in y)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. LASSO REGRESSION (L1 Regularization via coordinate descent)
# ─────────────────────────────────────────────────────────────────────────────

class LassoRegression:
    """
    Lasso = OLS + L1 penalty: minimize ||Xw - y||² + α Σ|wj|
    
    No closed form for L1. Uses coordinate descent:
      For each weight wj:
        ρj = Xjᵀ(y - Xw + wj*Xj)        # partial residual
        wj = soft_threshold(ρj / ||Xj||², α / ||Xj||²)
    
    Soft threshold: S(z, γ) = sign(z) * max(|z| - γ, 0)
    
    Key property: Lasso can set weights to EXACTLY 0 → feature selection.
    Use Lasso when: sparse solution expected, many irrelevant features.
    """

    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.w = []
        self.b = 0.0

    @staticmethod
    def _soft_threshold(z, gamma):
        if z > gamma:
            return z - gamma
        elif z < -gamma:
            return z + gamma
        return 0.0

    def fit(self, X, y):
        n, d = len(X), len(X[0])
        self.w = [0.0] * d
        self.b = mean(y)

        for _ in range(self.max_iter):
            w_old = self.w[:]

            # Update bias
            preds = [dot(X[i], self.w) + self.b for i in range(n)]
            self.b = self.b + mean([y[i] - preds[i] for i in range(n)])

            # Coordinate descent over each feature
            for j in range(d):
                preds = [dot(X[i], self.w) + self.b for i in range(n)]
                rho_j = sum(X[i][j] * (y[i] - preds[i] + self.w[j] * X[i][j])
                            for i in range(n))
                xj_sq = sum(X[i][j]**2 for i in range(n))
                if abs(xj_sq) < 1e-12:
                    self.w[j] = 0.0
                else:
                    self.w[j] = self._soft_threshold(rho_j / xj_sq,
                                                      self.alpha / xj_sq)

            # Check convergence
            delta = math.sqrt(sum((self.w[j] - w_old[j])**2 for j in range(d)))
            if delta < self.tol:
                break

        return self

    def predict(self, X):
        return [dot(X[i], self.w) + self.b for i in range(len(X))]

    def score(self, X, y):
        preds = self.predict(X)
        ss_res = sum((y[i] - preds[i])**2 for i in range(len(y)))
        ss_tot = sum((yi - mean(y))**2 for yi in y)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def run_verification():
    import sklearn.linear_model as skl
    import numpy as np

    rng = random.Random(42)
    n, d = 200, 3
    true_w = [2.5, -1.2, 0.8]
    true_b = 3.0
    X = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n)]
    y = [dot(X[i], true_w) + true_b + rng.gauss(0, 0.5) for i in range(n)]

    X_np = np.array(X)
    y_np = np.array(y)

    print("=" * 60)
    print("Phase 1.1 — Linear Regression Verification")
    print("=" * 60)

    # --- Simple Linear Regression ---
    X1 = [row[0] for row in X]
    slr = SimpleLinearRegression().fit(X1, y)
    sk_slr = skl.LinearRegression().fit(X_np[:, :1], y_np)
    ok = abs(slr.w - float(sk_slr.coef_[0])) < 1e-3
    print(f"  SimpleLinearReg  w={slr.w:.4f}  sk={float(sk_slr.coef_[0]):.4f}  [{'✓ PASS' if ok else '✗ FAIL'}]")

    # --- Normal Equation ---
    ne = LinearRegressionNormal().fit(X, y)
    sk_lr = skl.LinearRegression().fit(X_np, y_np)
    r2_ours = ne.score(X, y)
    r2_sk   = sk_lr.score(X_np, y_np)
    ok = abs(r2_ours - r2_sk) < 1e-3
    print(f"  NormalEquation   R²={r2_ours:.4f}  sk={r2_sk:.4f}  [{'✓ PASS' if ok else '✗ FAIL'}]")

    # --- Gradient Descent ---
    gd = LinearRegressionGD(lr=0.05, epochs=2000).fit(X, y)
    r2_gd = gd.score(X, y)
    ok = abs(r2_gd - r2_sk) < 0.01
    print(f"  GradientDescent  R²={r2_gd:.4f}  sk={r2_sk:.4f}  [{'✓ PASS' if ok else '✗ FAIL'}]")

    # --- Ridge ---
    ridge = RidgeRegression(alpha=1.0).fit(X, y)
    sk_ridge = skl.Ridge(alpha=1.0).fit(X_np, y_np)
    r2_ridge = ridge.score(X, y)
    r2_sk_r  = sk_ridge.score(X_np, y_np)
    ok = abs(r2_ridge - r2_sk_r) < 0.01
    print(f"  Ridge(α=1)       R²={r2_ridge:.4f}  sk={r2_sk_r:.4f}  [{'✓ PASS' if ok else '✗ FAIL'}]")

    # --- Lasso ---
    lasso = LassoRegression(alpha=0.1).fit(X, y)
    sk_lasso = skl.Lasso(alpha=0.1, max_iter=5000).fit(X_np, y_np)
    r2_lasso   = lasso.score(X, y)
    r2_sk_l    = sk_lasso.score(X_np, y_np)
    ok = abs(r2_lasso - r2_sk_l) < 0.02
    print(f"  Lasso(α=0.1)     R²={r2_lasso:.4f}  sk={r2_sk_l:.4f}  [{'✓ PASS' if ok else '✗ FAIL'}]")

    print()
    print("  Key intuitions:")
    print("  • Normal Equation exact but O(d³) — use for small d")
    print("  • GD scales well but needs tuned learning rate")
    print("  • Ridge shrinks weights → handles multicollinearity")
    print("  • Lasso zeros weights → automatic feature selection")
    print("=" * 60)


if __name__ == "__main__":
    run_verification()
