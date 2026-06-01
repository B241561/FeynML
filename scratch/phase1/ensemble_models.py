"""
Phase 1.4 — Ensemble Models from Scratch
==========================================
Topics:
  • Bagging (Bootstrap Aggregating) — math derivation
  • Random Forest (feature subsampling)
  • Gradient Boosting (additive model, residual fitting)
  • AdaBoost (adaptive weighting)
  • Bias-variance decomposition
  • Verification against scikit-learn
"""

import math
import random
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS FROM PHASE 1.3 (inline minimal tree)
# ─────────────────────────────────────────────────────────────────────────────

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def mean(v):
    return sum(v) / len(v)

def gini_impurity(y):
    if not y:
        return 0.0
    n = len(y)
    counts = Counter(y)
    return 1.0 - sum((c / n) ** 2 for c in counts.values())

def mse_impurity(y):
    if not y:
        return 0.0
    mu = mean(y)
    return sum((yi - mu) ** 2 for yi in y) / len(y)

class _Node:
    def __init__(self):
        self.feature_idx = self.threshold = self.left = self.right = self.value = None
    def is_leaf(self):
        return self.value is not None

def _best_split_cls(X, y, max_features=None, rng=None):
    n, d = len(X), len(X[0])
    features = list(range(d))
    if max_features:
        features = (rng or random).sample(features, min(max_features, d))
    best_ig, best = -1, None
    parent_g = gini_impurity(y)
    for f in features:
        vals = sorted(set(X[i][f] for i in range(n)))
        for t in ((vals[j]+vals[j+1])/2 for j in range(len(vals)-1)):
            lm = [i for i in range(n) if X[i][f] <= t]
            rm = [i for i in range(n) if X[i][f] > t]
            if not lm or not rm: continue
            y_l = [y[i] for i in lm]; y_r = [y[i] for i in rm]
            ig = parent_g - len(y_l)/n*gini_impurity(y_l) - len(y_r)/n*gini_impurity(y_r)
            if ig > best_ig:
                best_ig = ig
                best = (f, t, lm, rm)
    return best

def _best_split_reg(X, y, max_features=None, rng=None):
    n, d = len(X), len(X[0])
    features = list(range(d))
    if max_features:
        features = (rng or random).sample(features, min(max_features, d))
    best_gain, best = -1, None
    pm = mse_impurity(y)
    for f in features:
        vals = sorted(set(X[i][f] for i in range(n)))
        for t in ((vals[j]+vals[j+1])/2 for j in range(len(vals)-1)):
            lm = [i for i in range(n) if X[i][f] <= t]
            rm = [i for i in range(n) if X[i][f] > t]
            if not lm or not rm: continue
            y_l = [y[i] for i in lm]; y_r = [y[i] for i in rm]
            gain = pm - len(y_l)/n*mse_impurity(y_l) - len(y_r)/n*mse_impurity(y_r)
            if gain > best_gain:
                best_gain = gain
                best = (f, t, lm, rm)
    return best

def _build_cls(X, y, max_depth, min_split, max_features, depth, rng):
    node = _Node()
    if depth >= max_depth or len(y) < min_split or len(set(y)) == 1:
        node.value = Counter(y).most_common(1)[0][0]; return node
    split = _best_split_cls(X, y, max_features, rng)
    if split is None:
        node.value = Counter(y).most_common(1)[0][0]; return node
    f, t, lm, rm = split
    node.feature_idx = f; node.threshold = t
    X_l=[X[i] for i in lm]; y_l=[y[i] for i in lm]
    X_r=[X[i] for i in rm]; y_r=[y[i] for i in rm]
    node.left  = _build_cls(X_l, y_l, max_depth, min_split, max_features, depth+1, rng)
    node.right = _build_cls(X_r, y_r, max_depth, min_split, max_features, depth+1, rng)
    return node

def _build_reg(X, y, max_depth, min_split, max_features, depth, rng):
    node = _Node()
    if depth >= max_depth or len(y) < min_split:
        node.value = mean(y); return node
    split = _best_split_reg(X, y, max_features, rng)
    if split is None:
        node.value = mean(y); return node
    f, t, lm, rm = split
    node.feature_idx = f; node.threshold = t
    X_l=[X[i] for i in lm]; y_l=[y[i] for i in lm]
    X_r=[X[i] for i in rm]; y_r=[y[i] for i in rm]
    node.left  = _build_reg(X_l, y_l, max_depth, min_split, max_features, depth+1, rng)
    node.right = _build_reg(X_r, y_r, max_depth, min_split, max_features, depth+1, rng)
    return node

def _predict_node(node, x):
    if node.is_leaf(): return node.value
    return _predict_node(node.left, x) if x[node.feature_idx] <= node.threshold else _predict_node(node.right, x)


# ─────────────────────────────────────────────────────────────────────────────
# 1. RANDOM FOREST CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class RandomForestClassifier:
    """
    Bagging + Feature Subsampling.
    
    Bagging (Bootstrap Aggregating):
      1. Draw n bootstrap samples (with replacement) from training set
      2. Train a full decision tree on each sample
      3. Aggregate predictions by majority vote
    
    Why it reduces variance (bias-variance decomposition):
      Var[mean of T trees] = σ²/T + ρ * σ² * (1 - 1/T)
      ρ = correlation between trees
      → More trees AND lower correlation = lower variance
    
    Random feature subsampling (√d features per split):
      → Decorrelates trees → reduces ρ → bigger variance reduction
    
    Out-of-Bag (OOB) error: evaluate each tree on samples NOT in its bootstrap.
    Free internal validation without a separate test set.
    """

    def __init__(self, n_estimators=100, max_depth=10, max_features='sqrt',
                 min_samples_split=2, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []
        self.oob_score_ = None

    def fit(self, X, y):
        n, d = len(X), len(X[0])
        mf = int(math.sqrt(d)) if self.max_features == 'sqrt' else (self.max_features or d)
        rng = random.Random(self.random_state)

        oob_preds = [[] for _ in range(n)]   # for OOB scoring

        for t in range(self.n_estimators):
            # Bootstrap sample
            indices = [rng.randint(0, n-1) for _ in range(n)]
            X_boot = [X[i] for i in indices]
            y_boot = [y[i] for i in indices]
            oob_idx = set(range(n)) - set(indices)

            # Build tree
            tree = _build_cls(X_boot, y_boot, self.max_depth, self.min_samples_split,
                              mf, 0, rng)
            self.trees.append(tree)

            # OOB predictions
            for i in oob_idx:
                pred = _predict_node(tree, X[i])
                oob_preds[i].append(pred)

        # OOB score
        oob_correct = 0
        oob_total = 0
        for i in range(n):
            if oob_preds[i]:
                majority = Counter(oob_preds[i]).most_common(1)[0][0]
                if majority == y[i]:
                    oob_correct += 1
                oob_total += 1
        self.oob_score_ = oob_correct / oob_total if oob_total > 0 else 0.0
        return self

    def predict(self, X):
        all_preds = [[_predict_node(tree, x) for tree in self.trees] for x in X]
        return [Counter(row).most_common(1)[0][0] for row in all_preds]

    def predict_proba(self, X):
        """Vote-based probability estimate."""
        classes = None
        result = []
        for x in X:
            votes = [_predict_node(tree, x) for tree in self.trees]
            cnt = Counter(votes)
            if classes is None:
                classes = sorted(cnt.keys())
            total = len(self.trees)
            result.append([cnt.get(c, 0) / total for c in (classes or sorted(cnt.keys()))])
        return result

    def score(self, X, y):
        preds = self.predict(X)
        return sum(1 for i in range(len(y)) if preds[i] == y[i]) / len(y)


# ─────────────────────────────────────────────────────────────────────────────
# 2. GRADIENT BOOSTING CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class GradientBoostingClassifier:
    """
    Additive model: F_m(x) = F_{m-1}(x) + lr * h_m(x)
    
    For binary classification with log-loss:
      • Fit h_m to NEGATIVE GRADIENT of loss w.r.t. F_{m-1}
      • For log-loss: negative gradient = y - sigmoid(F)  (residuals)
    
    Algorithm:
      1. Initialize F_0 = log(p / (1-p))  (log-odds of base rate)
      2. For m = 1..M:
         a. Compute pseudo-residuals: r_i = y_i - sigmoid(F_{m-1}(x_i))
         b. Fit regression tree h_m to (X, r)
         c. F_m = F_{m-1} + lr * h_m
      3. Predict: sigmoid(F_M(x)) → threshold at 0.5
    
    Gradient boosting = functional gradient descent in function space.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, random_state=42):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []
        self.F0 = 0.0

    @staticmethod
    def _sigmoid(z):
        z = max(-500, min(500, z))
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, X, y):
        n = len(y)
        rng = random.Random(self.random_state)
        p0 = sum(y) / n
        eps = 1e-7
        p0 = max(eps, min(1 - eps, p0))
        self.F0 = math.log(p0 / (1 - p0))   # log-odds initialization

        F = [self.F0] * n   # current predictions (log-odds)

        for _ in range(self.n_estimators):
            # Pseudo-residuals (negative gradient of log-loss)
            probs = [self._sigmoid(f) for f in F]
            residuals = [y[i] - probs[i] for i in range(n)]

            # Fit regression tree to residuals
            tree = _build_reg(X, residuals, self.max_depth, self.min_samples_split,
                             None, 0, rng)
            self.trees.append(tree)

            # Update F
            updates = [_predict_node(tree, X[i]) for i in range(n)]
            F = [F[i] + self.lr * updates[i] for i in range(n)]

        return self

    def predict_proba(self, X):
        F = [self.F0] * len(X)
        for tree in self.trees:
            updates = [_predict_node(tree, X[i]) for i in range(len(X))]
            F = [F[i] + self.lr * updates[i] for i in range(len(X))]
        return [self._sigmoid(f) for f in F]

    def predict(self, X, threshold=0.5):
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(1 for i in range(len(y)) if preds[i] == y[i]) / len(y)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ADABOOST CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class AdaBoostClassifier:
    """
    Adaptive Boosting: reweight misclassified samples.
    Labels must be {-1, +1}.
    
    Algorithm (Freund & Schapire, 1997):
      1. Initialize weights w_i = 1/n
      2. For m = 1..M:
         a. Train weak learner h_m on weighted data
         b. Compute weighted error: ε_m = Σ w_i * I(h_m(x_i) ≠ y_i)
         c. Compute learner weight: α_m = 0.5 * log((1-ε)/ε)
         d. Update sample weights: w_i *= exp(-α_m * y_i * h_m(x_i))
         e. Normalize weights
      3. Final prediction: sign(Σ α_m * h_m(x))
    
    Intuition: each round focuses more on previously misclassified samples.
    """

    def __init__(self, n_estimators=50, max_depth=1, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.stumps = []       # (tree, alpha)
        self.classes_ = [-1, 1]

    def fit(self, X, y):
        """y must contain values convertible to {-1, 1}."""
        n = len(y)
        # Convert to {-1, 1}
        classes = sorted(set(y))
        self._class_map = {classes[0]: -1, classes[1]: 1}
        self._inv_map   = {-1: classes[0], 1: classes[1]}
        y_pm = [self._class_map[yi] for yi in y]

        weights = [1.0 / n] * n
        rng = random.Random(self.random_state)

        for _ in range(self.n_estimators):
            # Weighted bootstrap sample
            indices = rng.choices(range(n), weights=weights, k=n)
            X_w = [X[i] for i in indices]
            y_w = [y_pm[i] for i in indices]

            stump = _build_cls(X_w, y_w, self.max_depth, 1, None, 0, rng)

            # Predict on full training set
            preds = [1 if _predict_node(stump, X[i]) == 1 else -1 for i in range(n)]

            # Weighted error
            eps = sum(weights[i] for i in range(n) if preds[i] != y_pm[i])
            eps = max(1e-10, min(1 - 1e-10, eps))

            alpha = 0.5 * math.log((1 - eps) / eps)

            # Update weights
            new_weights = [
                weights[i] * math.exp(-alpha * y_pm[i] * preds[i])
                for i in range(n)
            ]
            total = sum(new_weights)
            weights = [w / total for w in new_weights]

            self.stumps.append((stump, alpha))

        return self

    def predict(self, X):
        scores = [0.0] * len(X)
        for stump, alpha in self.stumps:
            for i, x in enumerate(X):
                pred = 1 if _predict_node(stump, x) == 1 else -1
                scores[i] += alpha * pred
        pm_preds = [1 if s >= 0 else -1 for s in scores]
        return [self._inv_map[p] for p in pm_preds]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(1 for i in range(len(y)) if preds[i] == y[i]) / len(y)


# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def run_verification():
    from sklearn.ensemble import (RandomForestClassifier as SkRF,
                                   GradientBoostingClassifier as SkGB,
                                   AdaBoostClassifier as SkAB)
    from sklearn.datasets import make_classification
    import numpy as np

    print("=" * 60)
    print("Phase 1.4 — Ensemble Models Verification")
    print("=" * 60)

    X_np, y_np = make_classification(n_samples=300, n_features=6,
                                      n_informative=4, random_state=42)
    X = X_np.tolist(); y = y_np.tolist()

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42).fit(X, y)
    sk_rf = SkRF(n_estimators=50, max_depth=5, random_state=42).fit(X_np, y_np)
    acc_rf = rf.score(X, y)
    acc_sk_rf = sk_rf.score(X_np, y_np)
    ok = abs(acc_rf - acc_sk_rf) < 0.10
    print(f"  RandomForest     acc={acc_rf:.4f}  sk={acc_sk_rf:.4f}  OOB={rf.oob_score_:.4f}  [{'✓ PASS' if ok else '✗ FAIL'}]")

    # --- Gradient Boosting ---
    gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42).fit(X, y)
    sk_gb = SkGB(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42).fit(X_np, y_np)
    acc_gb = gb.score(X, y)
    acc_sk_gb = sk_gb.score(X_np, y_np)
    ok = abs(acc_gb - acc_sk_gb) < 0.10
    print(f"  GradientBoosting acc={acc_gb:.4f}  sk={acc_sk_gb:.4f}  [{'✓ PASS' if ok else '✗ FAIL'}]")

    # --- AdaBoost ---
    ab = AdaBoostClassifier(n_estimators=50, max_depth=1, random_state=42).fit(X, y)
    sk_ab = SkAB(n_estimators=50, random_state=42).fit(X_np, y_np)
    acc_ab = ab.score(X, y)
    acc_sk_ab = sk_ab.score(X_np, y_np)
    ok = abs(acc_ab - acc_sk_ab) < 0.10
    print(f"  AdaBoost         acc={acc_ab:.4f}  sk={acc_sk_ab:.4f}  [{'✓ PASS' if ok else '✗ FAIL'}]")

    print()
    print("  Key intuitions:")
    print("  • Bagging reduces variance; boosting reduces bias")
    print("  • RF: trees trained independently → parallelizable")
    print("  • GBM: trees trained sequentially on residuals → more powerful")
    print("  • AdaBoost: reweights samples → focuses on hard examples")
    print("  • RF OOB ≈ cross-validation score (free, no extra splits needed)")
    print("=" * 60)


if __name__ == "__main__":
    run_verification()
