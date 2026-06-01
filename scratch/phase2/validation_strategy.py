"""
Phase 2.3 — Validation Strategy & Leakage Detection
=====================================================
Topics:
  - K-Fold cross-validation from scratch
  - Stratified K-Fold
  - Time-series split
  - Nested CV (correct hyperparameter tuning + evaluation)
  - Data leakage detection (preprocessing, target, temporal)
  - Pipeline-based safe cross-validation
"""

import math
import random
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# 1. CROSS-VALIDATION FROM SCRATCH
# ─────────────────────────────────────────────────────────────────────────────

def kfold_split(n, k=5, shuffle=True, seed=42):
    """
    Returns list of (train_indices, val_indices) for k-fold CV.
    """
    indices = list(range(n))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    fold_size = n // k
    splits = []
    for fold in range(k):
        start = fold * fold_size
        end   = (fold + 1) * fold_size if fold < k - 1 else n
        val_idx   = indices[start:end]
        train_idx = indices[:start] + indices[end:]
        splits.append((train_idx, val_idx))
    return splits

def stratified_kfold_split(y, k=5, shuffle=True, seed=42):
    """
    Stratified K-Fold: each fold has the same class distribution as the full dataset.
    Critical for imbalanced datasets — plain K-Fold can put all rare-class samples
    in one fold.
    """
    # Group indices by class
    class_indices = defaultdict(list)
    for i, label in enumerate(y):
        class_indices[label].append(i)

    if shuffle:
        rng = random.Random(seed)
        for cls in class_indices:
            rng.shuffle(class_indices[cls])

    # Assign each class's samples across folds proportionally
    fold_indices = [[] for _ in range(k)]
    for cls, idxs in class_indices.items():
        for i, idx in enumerate(idxs):
            fold_indices[i % k].append(idx)

    splits = []
    for fold in range(k):
        val_idx   = fold_indices[fold]
        train_idx = [idx for f in range(k) if f != fold for idx in fold_indices[f]]
        splits.append((train_idx, val_idx))
    return splits

def time_series_split(n, n_splits=5, gap=0):
    """
    Time-series aware split: validation always comes AFTER training.
    gap: number of samples to skip between train end and val start
         (prevents leakage in time-series with overlapping windows).
    """
    test_size = n // (n_splits + 1)
    splits = []
    for i in range(1, n_splits + 1):
        train_end = i * test_size
        val_start = train_end + gap
        val_end   = val_start + test_size
        if val_end > n:
            break
        splits.append((list(range(train_end)), list(range(val_start, val_end))))
    return splits

def cross_validate(model_fn, X, y, splits, score_fn, preprocess_fn=None):
    """
    Run cross-validation given pre-computed splits.
    model_fn()       → returns a new unfitted model
    score_fn(y, yp)  → scalar score
    preprocess_fn(X_train, X_val) → (X_train_processed, X_val_processed)
    NOTE: preprocess_fn is applied AFTER splitting to prevent leakage.
    """
    scores = []
    for fold_i, (train_idx, val_idx) in enumerate(splits):
        X_tr = [X[i] for i in train_idx]
        y_tr = [y[i] for i in train_idx]
        X_vl = [X[i] for i in val_idx]
        y_vl = [y[i] for i in val_idx]

        # Apply preprocessing fit on TRAIN, transform both — safe!
        if preprocess_fn is not None:
            X_tr, X_vl = preprocess_fn(X_tr, X_vl)

        model = model_fn()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_vl)
        score  = score_fn(y_vl, y_pred)
        scores.append(score)

    mean_score = sum(scores) / len(scores)
    variance   = sum((s - mean_score)**2 for s in scores) / (len(scores) - 1)
    return {
        "scores":   [round(s, 4) for s in scores],
        "mean":     round(mean_score, 4),
        "std":      round(math.sqrt(variance), 4),
        "ci_95_lo": round(mean_score - 1.96 * math.sqrt(variance / len(scores)), 4),
        "ci_95_hi": round(mean_score + 1.96 * math.sqrt(variance / len(scores)), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. STANDARDISER (fit on train only)
# ─────────────────────────────────────────────────────────────────────────────

class StandardScalerScratch:
    """
    Fit on training data ONLY. Transform both train and validation.
    Reason: if you fit on all data, test statistics "leak" into training.
    """
    def __init__(self):
        self.means = None
        self.stds  = None

    def fit(self, X):
        n = len(X)
        d = len(X[0]) if X else 0
        self.means = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
        self.stds  = [
            math.sqrt(sum((X[i][j] - self.means[j])**2 for i in range(n)) / (n - 1))
            for j in range(d)
        ]
        # Replace 0 std with 1 to avoid division by zero (constant feature)
        self.stds = [s if s > 1e-10 else 1.0 for s in self.stds]
        return self

    def transform(self, X):
        return [[(X[i][j] - self.means[j]) / self.stds[j]
                 for j in range(len(X[0]))] for i in range(len(X))]

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def safe_preprocess(self, X_train, X_val):
        """
        The safe cross-validation preprocessor.
        Fit on X_train, transform both. Never touch X_val during fit.
        """
        X_tr = self.fit(X_train).transform(X_train)
        X_vl = self.transform(X_val)
        return X_tr, X_vl


# ─────────────────────────────────────────────────────────────────────────────
# 3. LEAKAGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_preprocessing_leakage(X_train, X_test, feature_names=None, tol_z=3.0):
    """
    Check if test set statistics were accidentally used in preprocessing.
    Heuristic: if test feature means differ vastly from train means,
    someone probably scaled the full dataset before splitting.

    Actually detects the OPPOSITE: if means are suspiciously identical,
    the scaler was fit on the combined dataset (leakage!).

    Returns: leakage risk score and per-feature report.
    """
    if feature_names is None:
        feature_names = [f"f{j}" for j in range(len(X_train[0]))]

    d = len(X_train[0])
    report = []

    for j in range(d):
        train_col = [X_train[i][j] for i in range(len(X_train))]
        test_col  = [X_test[i][j]  for i in range(len(X_test))]
        train_m = sum(train_col) / len(train_col)
        test_m  = sum(test_col)  / len(test_col)
        train_s = math.sqrt(sum((x - train_m)**2 for x in train_col) / (len(train_col)-1))

        # Normalised difference in means (how many train SDs apart?)
        if train_s > 1e-10:
            z_diff = abs(train_m - test_m) / train_s
        else:
            z_diff = 0.0

        # Check if already zero-mean / unit-variance (sign of global scaling)
        already_scaled_train = abs(train_m) < 0.05 and abs(train_s - 1.0) < 0.05
        already_scaled_test  = abs(test_m)  < 0.05 and abs(
            math.sqrt(sum((x - test_m)**2 for x in test_col) / (len(test_col)-1)) - 1.0
        ) < 0.05

        leakage_suspect = already_scaled_train and already_scaled_test
        report.append({
            "feature":          feature_names[j],
            "train_mean":       round(train_m, 4),
            "test_mean":        round(test_m, 4),
            "train_std":        round(train_s, 4),
            "mean_diff_z":      round(z_diff, 4),
            "leakage_suspect":  leakage_suspect,
        })

    suspects = [r for r in report if r["leakage_suspect"]]
    return {
        "n_suspects":     len(suspects),
        "suspect_features": [r["feature"] for r in suspects],
        "feature_report": report,
        "leakage_risk":   "HIGH" if len(suspects) > len(report) * 0.5
                          else "MEDIUM" if len(suspects) > 0 else "LOW",
    }

def detect_target_leakage(X, y, feature_names, threshold_corr=0.95):
    """
    Detect features suspiciously correlated with the target.
    High correlation MIGHT indicate the feature encodes the label.
    """
    import math

    n = len(y)
    mean_y = sum(y) / n
    std_y  = math.sqrt(sum((yi - mean_y)**2 for yi in y) / (n - 1))

    suspects = []
    for j, fname in enumerate(feature_names):
        col = [X[i][j] for i in range(n)]
        mean_x = sum(col) / n
        std_x  = math.sqrt(sum((x - mean_x)**2 for x in col) / (n - 1))
        if std_x < 1e-10 or std_y < 1e-10:
            continue
        cov = sum((col[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / (n - 1)
        r   = cov / (std_x * std_y)
        if abs(r) >= threshold_corr:
            suspects.append({"feature": fname, "correlation": round(r, 4)})

    return {
        "n_suspects":     len(suspects),
        "suspects":       sorted(suspects, key=lambda x: abs(x["correlation"]), reverse=True),
        "leakage_risk":   "HIGH" if suspects else "LOW",
    }

def check_temporal_ordering(timestamps, train_idx, test_idx):
    """
    Verify no test timestamp is BEFORE any training timestamp.
    Temporal leakage: using future data to predict the past.
    """
    train_max = max(timestamps[i] for i in train_idx)
    test_min  = min(timestamps[i] for i in test_idx)
    leakage   = test_min < train_max
    overlap_count = sum(1 for i in test_idx if timestamps[i] <= train_max)
    return {
        "temporal_leakage": leakage,
        "train_max_time":   train_max,
        "test_min_time":    test_min,
        "test_samples_before_train_end": overlap_count,
        "recommendation": (
            "CRITICAL: Test data contains timestamps before/during training period! "
            "Use TimeSeriesSplit — never random split on time-series data."
            if leakage else
            "OK: Test data is fully after training data."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. NESTED CV
# ─────────────────────────────────────────────────────────────────────────────

def nested_cv_demo(model_fn, X, y, score_fn, param_grid,
                   outer_k=5, inner_k=3, seed=42):
    """
    Nested cross-validation: correct way to simultaneously tune AND evaluate.
    Outer loop:  performance estimation (not contaminated by tuning)
    Inner loop:  hyperparameter selection
    """
    outer_splits = stratified_kfold_split(y, k=outer_k, seed=seed)
    outer_scores = []
    best_params_per_fold = []

    for fold_i, (train_idx, test_idx) in enumerate(outer_splits):
        X_tr_outer = [X[i] for i in train_idx]
        y_tr_outer = [y[i] for i in train_idx]
        X_te_outer = [X[i] for i in test_idx]
        y_te_outer = [y[i] for i in test_idx]

        # Inner CV: find best hyperparameters on train_outer only
        best_inner_score = -math.inf
        best_params = None
        for params in param_grid:
            inner_splits = stratified_kfold_split(y_tr_outer, k=inner_k, seed=seed+fold_i)
            inner_scores = []
            for tr_in, vl_in in inner_splits:
                X_tr_in = [X_tr_outer[i] for i in tr_in]
                y_tr_in = [y_tr_outer[i] for i in tr_in]
                X_vl_in = [X_tr_outer[i] for i in vl_in]
                y_vl_in = [y_tr_outer[i] for i in vl_in]
                m = model_fn(**params)
                m.fit(X_tr_in, y_tr_in)
                inner_scores.append(score_fn(y_vl_in, m.predict(X_vl_in)))
            mean_inner = sum(inner_scores) / len(inner_scores)
            if mean_inner > best_inner_score:
                best_inner_score = mean_inner
                best_params = params

        # Refit on full outer train with best params, evaluate on outer test
        final_model = model_fn(**best_params)
        final_model.fit(X_tr_outer, y_tr_outer)
        outer_score = score_fn(y_te_outer, final_model.predict(X_te_outer))
        outer_scores.append(outer_score)
        best_params_per_fold.append(best_params)

    mean_score = sum(outer_scores) / len(outer_scores)
    return {
        "outer_scores":         [round(s, 4) for s in outer_scores],
        "mean_score":           round(mean_score, 4),
        "std_score":            round(math.sqrt(sum((s-mean_score)**2 for s in outer_scores)/len(outer_scores)), 4),
        "best_params_per_fold": best_params_per_fold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def run_verification():
    print("=" * 60)
    print("Phase 2.3 — Validation Strategy Verification")
    print("=" * 60)

    import random
    random.seed(42)
    n = 100
    X = [[random.random() for _ in range(4)] for _ in range(n)]
    y = [random.randint(0, 1) for _ in range(n)]

    # K-Fold splits: verify no overlap and full coverage
    splits = kfold_split(n, k=5)
    all_val = []
    for tr, vl in splits:
        all_val.extend(vl)
        overlap = set(tr) & set(vl)
        assert len(overlap) == 0, "Train/val overlap!"
    assert sorted(all_val) == list(range(n)), "Not full coverage!"
    print(f"  KFold(n={n}, k=5): ✓ no overlap, ✓ full coverage")

    # Stratified: check each fold has similar class balance
    splits_s = stratified_kfold_split(y, k=5)
    balances = []
    for tr, vl in splits_s:
        pos_rate = sum(y[i] for i in vl) / len(vl)
        balances.append(pos_rate)
    overall_rate = sum(y) / len(y)
    max_dev = max(abs(b - overall_rate) for b in balances)
    ok = max_dev < 0.1
    print(f"  StratifiedKFold: overall_pos={overall_rate:.3f}, "
          f"max_fold_deviation={max_dev:.3f}  [{'✓ PASS' if ok else '✗ FAIL'}]")

    # Time series split: verify ordering
    ts_splits = time_series_split(n, n_splits=4)
    for tr, vl in ts_splits:
        assert max(tr) < min(vl), "Temporal ordering violated!"
    print(f"  TimeSeriesSplit: ✓ all {len(ts_splits)} folds maintain temporal order")

    # Leakage detection demo
    print("\n  Leakage Detection Demo:")
    # Simulate target leakage: feature 0 = target + small noise
    X_leak = [[y[i] + random.gauss(0, 0.01)] + [random.random() for _ in range(3)]
              for i in range(n)]
    leak_report = detect_target_leakage(X_leak, y, ["leaky_f0", "f1", "f2", "f3"])
    print(f"    Target leakage suspects: {leak_report['suspects']}")
    print(f"    Risk: {leak_report['leakage_risk']}")
    ok_leak = leak_report['n_suspects'] >= 1 and leak_report['suspects'][0]['feature'] == 'leaky_f0'
    print(f"    Correctly detected leaky_f0  [{'✓ PASS' if ok_leak else '✗ FAIL'}]")

    # Temporal leakage demo
    timestamps = list(range(n))
    random.shuffle(timestamps)  # random split on ordered data = temporal leakage
    train_idx = list(range(60))
    test_idx  = list(range(60, 100))
    temporal  = check_temporal_ordering(timestamps, train_idx, test_idx)
    print(f"\n    Random split temporal check: leakage={temporal['temporal_leakage']}")
    print(f"    Samples before train end: {temporal['test_samples_before_train_end']}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    run_verification()
