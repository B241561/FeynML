"""
Phase 1.3 — Decision Tree from Scratch (CART)
===============================================
Topics:
  • Gini impurity & Entropy (information gain)
  • CART algorithm (Classification and Regression Trees)
  • Recursive binary splitting
  • Tree pruning (max_depth, min_samples_split)
  • Feature importance
  • Verification against scikit-learn
"""

import math
import random
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
# IMPURITY MEASURES
# ─────────────────────────────────────────────────────────────────────────────

def gini_impurity(y):
    """
    Gini = 1 - Σ p_k²
    
    Intuition: probability of misclassifying a randomly chosen element
    if it were labeled randomly according to the class distribution.
    
    Range: [0, 1 - 1/K] where K = number of classes
    Gini = 0 → perfectly pure node (all same class)
    Gini = 0.5 → maximally impure for binary classification
    
    Faster to compute than entropy (no log).
    """
    if not y:
        return 0.0
    n = len(y)
    counts = Counter(y)
    return 1.0 - sum((c / n) ** 2 for c in counts.values())

def entropy(y):
    """
    H = -Σ p_k * log2(p_k)
    
    Intuition: average bits of information needed to identify a label.
    H = 0 → pure node (no surprise)
    H = 1 → maximally uncertain (binary: 50/50 split)
    
    Used in ID3 / C4.5 algorithms.
    """
    if not y:
        return 0.0
    n = len(y)
    counts = Counter(y)
    return -sum(
        (c / n) * math.log2(c / n)
        for c in counts.values() if c > 0
    )

def information_gain(y_parent, y_left, y_right, criterion='gini'):
    """
    IG = impurity(parent) - weighted avg impurity(children)
    
    We want to maximize information gain → find the best split.
    """
    fn = gini_impurity if criterion == 'gini' else entropy
    n = len(y_parent)
    nl, nr = len(y_left), len(y_right)
    if nl == 0 or nr == 0:
        return 0.0
    return fn(y_parent) - (nl/n) * fn(y_left) - (nr/n) * fn(y_right)


# ─────────────────────────────────────────────────────────────────────────────
# TREE NODE
# ─────────────────────────────────────────────────────────────────────────────

class Node:
    def __init__(self):
        self.feature_idx = None    # which feature to split on
        self.threshold   = None    # split value (x[feature] <= threshold → left)
        self.left        = None    # left child Node
        self.right       = None    # right child Node
        self.value       = None    # leaf prediction (class label or mean)
        self.impurity    = 0.0     # node impurity (for feature importance)
        self.n_samples   = 0       # number of samples at this node

    def is_leaf(self):
        return self.value is not None


# ─────────────────────────────────────────────────────────────────────────────
# DECISION TREE CLASSIFIER (CART)
# ─────────────────────────────────────────────────────────────────────────────

class DecisionTreeClassifier:
    """
    Classification And Regression Tree (CART).
    
    Algorithm:
      1. At each node, find feature f and threshold t that maximizes info gain
      2. Split data: left = {x: x[f] <= t}, right = {x: x[f] > t}
      3. Recurse until stopping criteria met
      4. Leaf = majority class
    
    Stopping criteria:
      • max_depth reached
      • min_samples_split not met
      • node is already pure
    
    Time complexity: O(n * d * n log n) per level, O(depth) levels
    Space complexity: O(n) for the tree
    """

    def __init__(self, max_depth=None, min_samples_split=2,
                 criterion='gini', max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features   # None = use all features
        self.root = None
        self.n_features_ = 0
        self.feature_importances_ = []

    def fit(self, X, y):
        self.n_features_ = len(X[0])
        self.feature_importances_ = [0.0] * self.n_features_
        self.root = self._build(X, y, depth=0)
        # Normalize feature importances
        total = sum(self.feature_importances_)
        if total > 0:
            self.feature_importances_ = [fi / total for fi in self.feature_importances_]
        return self

    def _build(self, X, y, depth):
        node = Node()
        node.n_samples = len(y)
        fn = gini_impurity if self.criterion == 'gini' else entropy
        node.impurity = fn(y)

        # Stopping conditions
        if (self._should_stop(X, y, depth)):
            node.value = Counter(y).most_common(1)[0][0]
            return node

        # Find best split
        best = self._best_split(X, y)
        if best is None:
            node.value = Counter(y).most_common(1)[0][0]
            return node

        f_idx, thresh, X_l, y_l, X_r, y_r, ig = best

        # Accumulate feature importance (weighted impurity decrease)
        n = len(y)
        self.feature_importances_[f_idx] += n * ig

        node.feature_idx = f_idx
        node.threshold   = thresh
        node.left  = self._build(X_l, y_l, depth + 1)
        node.right = self._build(X_r, y_r, depth + 1)
        return node

    def _should_stop(self, X, y, depth):
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if len(y) < self.min_samples_split:
            return True
        if len(set(y)) == 1:
            return True
        return False

    def _best_split(self, X, y):
        n, d = len(X), len(X[0])
        best_ig = -1
        best = None

        # Feature subsampling (for Random Forest)
        feature_indices = list(range(d))
        if self.max_features is not None:
            k = min(self.max_features, d)
            feature_indices = random.sample(feature_indices, k)

        for f in feature_indices:
            values = sorted(set(X[i][f] for i in range(n)))
            # Try midpoints between consecutive unique values
            thresholds = [(values[j] + values[j+1]) / 2 for j in range(len(values)-1)]

            for thresh in thresholds:
                left_mask  = [i for i in range(n) if X[i][f] <= thresh]
                right_mask = [i for i in range(n) if X[i][f] > thresh]

                if not left_mask or not right_mask:
                    continue

                y_l = [y[i] for i in left_mask]
                y_r = [y[i] for i in right_mask]
                ig = information_gain(y, y_l, y_r, self.criterion)

                if ig > best_ig:
                    best_ig = ig
                    X_l = [X[i] for i in left_mask]
                    X_r = [X[i] for i in right_mask]
                    best = (f, thresh, X_l, y_l, X_r, y_r, ig)

        return best

    def _predict_one(self, node, x):
        if node.is_leaf():
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(node.left, x)
        else:
            return self._predict_one(node.right, x)

    def predict(self, X):
        return [self._predict_one(self.root, x) for x in X]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(1 for i in range(len(y)) if preds[i] == y[i]) / len(y)

    def get_depth(self):
        def _depth(node):
            if node is None or node.is_leaf():
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))
        return _depth(self.root)

    def get_n_leaves(self):
        def _count(node):
            if node is None:
                return 0
            if node.is_leaf():
                return 1
            return _count(node.left) + _count(node.right)
        return _count(self.root)


# ─────────────────────────────────────────────────────────────────────────────
# DECISION TREE REGRESSOR
# ─────────────────────────────────────────────────────────────────────────────

def mse_impurity(y):
    """Variance (MSE) used as impurity for regression trees."""
    if not y:
        return 0.0
    mu = sum(y) / len(y)
    return sum((yi - mu)**2 for yi in y) / len(y)

class DecisionTreeRegressor:
    """
    Regression tree: leaf predicts mean of samples.
    Splits minimize weighted MSE of children.
    """

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._build(X, y, 0)
        return self

    def _build(self, X, y, depth):
        node = Node()
        node.n_samples = len(y)

        if (self.max_depth is not None and depth >= self.max_depth
                or len(y) < self.min_samples_split):
            node.value = sum(y) / len(y)
            return node

        best = self._best_split(X, y)
        if best is None:
            node.value = sum(y) / len(y)
            return node

        f_idx, thresh, X_l, y_l, X_r, y_r = best
        node.feature_idx = f_idx
        node.threshold   = thresh
        node.left  = self._build(X_l, y_l, depth + 1)
        node.right = self._build(X_r, y_r, depth + 1)
        return node

    def _best_split(self, X, y):
        n, d = len(X), len(X[0])
        best_gain = -1
        best = None
        parent_mse = mse_impurity(y)

        for f in range(d):
            values = sorted(set(X[i][f] for i in range(n)))
            thresholds = [(values[j]+values[j+1])/2 for j in range(len(values)-1)]

            for thresh in thresholds:
                lm = [i for i in range(n) if X[i][f] <= thresh]
                rm = [i for i in range(n) if X[i][f] > thresh]
                if not lm or not rm:
                    continue
                y_l, y_r = [y[i] for i in lm], [y[i] for i in rm]
                gain = parent_mse - (len(y_l)/n)*mse_impurity(y_l) - (len(y_r)/n)*mse_impurity(y_r)
                if gain > best_gain:
                    best_gain = gain
                    best = (f, thresh, [X[i] for i in lm], y_l, [X[i] for i in rm], y_r)
        return best

    def _predict_one(self, node, x):
        if node.is_leaf():
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(node.left, x)
        return self._predict_one(node.right, x)

    def predict(self, X):
        return [self._predict_one(self.root, x) for x in X]

    def score(self, X, y):
        preds = self.predict(X)
        ss_res = sum((y[i] - preds[i])**2 for i in range(len(y)))
        mu = sum(y)/len(y)
        ss_tot = sum((yi - mu)**2 for yi in y)
        return 1 - ss_res/ss_tot if ss_tot > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def run_verification():
    from sklearn.tree import DecisionTreeClassifier as SkDTC
    from sklearn.tree import DecisionTreeRegressor as SkDTR
    from sklearn.datasets import make_classification, make_regression

    print("=" * 60)
    print("Phase 1.3 — Decision Tree Verification")
    print("=" * 60)

    # --- Impurity measures ---
    y_pure   = [0, 0, 0, 0]
    y_mixed  = [0, 0, 1, 1]
    ok1 = gini_impurity(y_pure)  == 0.0
    ok2 = abs(gini_impurity(y_mixed) - 0.5) < 1e-9
    ok3 = entropy(y_pure) == 0.0
    ok4 = abs(entropy(y_mixed) - 1.0) < 1e-9
    print(f"  Gini pure=0: {ok1}  mixed=0.5: {ok2}  [{'✓ PASS' if ok1 and ok2 else '✗ FAIL'}]")
    print(f"  Entropy pure=0: {ok3}  mixed=1.0: {ok4}  [{'✓ PASS' if ok3 and ok4 else '✗ FAIL'}]")

    # --- Classifier ---
    import numpy as np
    X_cls, y_cls = make_classification(n_samples=200, n_features=4,
                                        n_informative=3, random_state=42)
    X_l, y_l = X_cls.tolist(), y_cls.tolist()

    dt = DecisionTreeClassifier(max_depth=5, criterion='gini').fit(X_l, y_l)
    sk_dt = SkDTC(max_depth=5, criterion='gini', random_state=42).fit(X_cls, y_cls)
    acc_ours = dt.score(X_l, y_l)
    acc_sk   = sk_dt.score(X_cls, y_cls)
    ok = abs(acc_ours - acc_sk) < 0.05
    print(f"  DT Classifier    acc={acc_ours:.4f}  sk={acc_sk:.4f}  depth={dt.get_depth()}  [{'✓ PASS' if ok else '✗ FAIL'}]")

    # --- Regressor ---
    X_reg, y_reg = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)
    X_rl, y_rl = X_reg.tolist(), y_reg.tolist()

    dtr = DecisionTreeRegressor(max_depth=4).fit(X_rl, y_rl)
    sk_dtr = SkDTR(max_depth=4, random_state=42).fit(X_reg, y_reg)
    r2_ours = dtr.score(X_rl, y_rl)
    r2_sk   = sk_dtr.score(X_reg, y_reg)
    ok = abs(r2_ours - r2_sk) < 0.05
    print(f"  DT Regressor     R²={r2_ours:.4f}  sk={r2_sk:.4f}  [{'✓ PASS' if ok else '✗ FAIL'}]")

    print()
    print("  Key intuitions:")
    print("  • Gini & Entropy both measure impurity — Gini is faster")
    print("  • CART always makes binary splits (unlike ID3)")
    print("  • Deep trees overfit; use max_depth/min_samples_split")
    print("  • Feature importance = total weighted impurity decrease")
    print("=" * 60)


if __name__ == "__main__":
    run_verification()
