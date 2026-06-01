"""
Phase 2.1 — Classification Metrics from Scratch
=================================================
Every metric derived from first principles.
Verified against sklearn at the bottom.

Topics:
  - Confusion matrix
  - Precision, Recall, F1, F-beta
  - ROC curve & AUC (trapezoidal rule)
  - Precision-Recall curve & Average Precision
  - Cohen's Kappa
  - Matthews Correlation Coefficient
  - Log-loss (cross-entropy)
  - Calibration curve
  - Multi-class extensions
"""

import math
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def confusion_matrix(y_true, y_pred, labels=None):
    """
    Returns dict with TN, FP, FN, TP for binary classification.
    For multi-class, returns the full n×n matrix.
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    n = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    cm = [[0] * n for _ in range(n)]
    for yt, yp in zip(y_true, y_pred):
        i = label_to_idx.get(yt)
        j = label_to_idx.get(yp)
        if i is not None and j is not None:
            cm[i][j] += 1
    return cm, labels

def binary_confusion(y_true, y_pred, pos_label=1):
    """Returns (TN, FP, FN, TP) for binary problem."""
    TN = FP = FN = TP = 0
    for yt, yp in zip(y_true, y_pred):
        if   yt == pos_label and yp == pos_label: TP += 1
        elif yt != pos_label and yp != pos_label: TN += 1
        elif yt != pos_label and yp == pos_label: FP += 1
        elif yt == pos_label and yp != pos_label: FN += 1
    return TN, FP, FN, TP

def print_confusion_matrix(cm, labels):
    col_w = max(len(str(l)) for l in labels) + 2
    header = " " * col_w + "".join(f"{str(l):>{col_w}}" for l in labels)
    print("  Predicted →")
    print(f"  {'':>{col_w}}{header}")
    for i, label in enumerate(labels):
        row_str = f"  {str(label):>{col_w}}" + "".join(f"{cm[i][j]:>{col_w}}" for j in range(len(labels)))
        print(row_str)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CORE BINARY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def accuracy(y_true, y_pred):
    return sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)

def precision(y_true, y_pred, pos_label=1):
    """TP / (TP + FP) — Of all predicted positive, how many truly are?"""
    _, _, _, TP = binary_confusion(y_true, y_pred, pos_label)
    _, FP, _, _ = binary_confusion(y_true, y_pred, pos_label)
    TN, FP, FN, TP = binary_confusion(y_true, y_pred, pos_label)
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0

def recall(y_true, y_pred, pos_label=1):
    """TP / (TP + FN) — Of all actually positive, how many did we catch?"""
    TN, FP, FN, TP = binary_confusion(y_true, y_pred, pos_label)
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0

def f_beta(y_true, y_pred, beta=1.0, pos_label=1):
    """
    F-beta score: weighted harmonic mean of precision and recall.
    beta=1: equal weight (F1)
    beta=2: recall weighted twice as much as precision (miss costs more)
    beta=0.5: precision weighted twice as much (false alarm costs more)
    """
    p = precision(y_true, y_pred, pos_label)
    r = recall(y_true, y_pred, pos_label)
    if p + r == 0:
        return 0.0
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r)

def f1_score(y_true, y_pred, pos_label=1):
    return f_beta(y_true, y_pred, beta=1.0, pos_label=pos_label)

def specificity(y_true, y_pred, pos_label=1):
    """TN / (TN + FP) — True Negative Rate."""
    TN, FP, FN, TP = binary_confusion(y_true, y_pred, pos_label)
    return TN / (TN + FP) if (TN + FP) > 0 else 0.0

def balanced_accuracy(y_true, y_pred, pos_label=1):
    """Average of recall and specificity. Better for imbalanced datasets."""
    return (recall(y_true, y_pred, pos_label) + specificity(y_true, y_pred, pos_label)) / 2

def matthews_corrcoef(y_true, y_pred, pos_label=1):
    """
    Matthews Correlation Coefficient (MCC) ∈ [-1, 1].
    The most informative single metric for imbalanced binary classification.
    Only high when TP, TN, FP, FN are all proportionally good.
    MCC = (TP*TN - FP*FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    """
    TN, FP, FN, TP = binary_confusion(y_true, y_pred, pos_label)
    denom = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    return (TP * TN - FP * FN) / denom if denom > 0 else 0.0

def cohen_kappa(y_true, y_pred):
    """
    Cohen's Kappa: agreement beyond chance.
    κ = (p_o - p_e) / (1 - p_e)
    κ=1: perfect agreement, κ=0: no better than random, κ<0: worse than random.
    """
    n = len(y_true)
    labels = sorted(set(y_true) | set(y_pred))
    cm, _ = confusion_matrix(y_true, y_pred, labels)
    k = len(labels)
    p_o = sum(cm[i][i] for i in range(k)) / n  # observed agreement
    row_sums = [sum(cm[i]) / n for i in range(k)]
    col_sums = [sum(cm[i][j] for i in range(k)) / n for j in range(k)]
    p_e = sum(row_sums[i] * col_sums[i] for i in range(k))  # expected agreement
    return (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. THRESHOLD-BASED ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def threshold_analysis(y_true, y_scores, thresholds=None, pos_label=1):
    """
    Compute metrics at every threshold. Helps choose optimal threshold
    for a given business objective.
    """
    if thresholds is None:
        thresholds = sorted(set(y_scores), reverse=True)
    results = []
    for t in thresholds:
        y_pred = [1 if s >= t else 0 for s in y_scores]
        TN, FP, FN, TP = binary_confusion(y_true, y_pred, pos_label)
        p = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        r = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        results.append({
            "threshold": round(t, 4), "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "accuracy": round((TP + TN) / (TP + TN + FP + FN), 4)
        })
    return results

def optimal_threshold_f1(y_true, y_scores):
    """Find threshold that maximises F1 score."""
    analysis = threshold_analysis(y_true, y_scores)
    best = max(analysis, key=lambda r: r["f1"])
    return best["threshold"], best["f1"]

def optimal_threshold_gmean(y_true, y_scores):
    """Find threshold that maximises geometric mean of sensitivity & specificity."""
    thresholds = sorted(set(y_scores), reverse=True)
    best_t, best_g = 0.5, 0.0
    for t in thresholds:
        y_pred = [1 if s >= t else 0 for s in y_scores]
        sens   = recall(y_true, y_pred)
        spec   = specificity(y_true, y_pred)
        g      = math.sqrt(sens * spec)
        if g > best_g:
            best_g, best_t = g, t
    return best_t, best_g


# ─────────────────────────────────────────────────────────────────────────────
# 4. ROC CURVE & AUC
# ─────────────────────────────────────────────────────────────────────────────

def roc_curve(y_true, y_scores, pos_label=1):
    """
    Compute ROC curve: (FPR, TPR) at each threshold.
    FPR = FP / (FP + TN) — False Positive Rate (x-axis)
    TPR = TP / (TP + FN) — True Positive Rate (y-axis)
    """
    thresholds = sorted(set(y_scores), reverse=True)
    fprs = [0.0]
    tprs = [0.0]
    for t in thresholds:
        y_pred = [1 if s >= t else 0 for s in y_scores]
        TN, FP, FN, TP = binary_confusion(y_true, y_pred, pos_label)
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        fprs.append(fpr)
        tprs.append(tpr)
    fprs.append(1.0)
    tprs.append(1.0)
    return fprs, tprs, [None] + list(thresholds) + [None]

def auc_trapezoidal(x, y):
    """
    Area Under Curve via trapezoidal rule.
    ∫ y dx ≈ Σ (x[i+1] - x[i]) * (y[i] + y[i+1]) / 2
    """
    assert len(x) == len(y)
    area = 0.0
    for i in range(len(x) - 1):
        area += abs(x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2
    return area

def roc_auc_score(y_true, y_scores, pos_label=1):
    fprs, tprs, _ = roc_curve(y_true, y_scores, pos_label)
    return auc_trapezoidal(fprs, tprs)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PRECISION-RECALL CURVE & AVERAGE PRECISION
# ─────────────────────────────────────────────────────────────────────────────

def precision_recall_curve(y_true, y_scores, pos_label=1):
    """
    PR curve: (recall, precision) at each threshold.
    More informative than ROC for highly imbalanced datasets.
    """
    thresholds = sorted(set(y_scores), reverse=True)
    precisions = [1.0]
    recalls    = [0.0]
    for t in thresholds:
        y_pred = [1 if s >= t else 0 for s in y_scores]
        p = precision(y_true, y_pred, pos_label)
        r = recall(y_true, y_pred, pos_label)
        precisions.append(p)
        recalls.append(r)
    return precisions, recalls

def average_precision_score(y_true, y_scores, pos_label=1):
    """
    Average Precision (AP) = area under PR curve.
    Uses step interpolation (not linear) matching sklearn.
    AP = Σ (R[n] - R[n-1]) * P[n]
    """
    precs, recs = precision_recall_curve(y_true, y_scores, pos_label)
    ap = 0.0
    for i in range(1, len(precs)):
        ap += (recs[i] - recs[i - 1]) * precs[i]
    return ap


# ─────────────────────────────────────────────────────────────────────────────
# 6. LOG-LOSS (CROSS-ENTROPY)
# ─────────────────────────────────────────────────────────────────────────────

def log_loss(y_true, y_probs, eps=1e-15):
    """
    Binary cross-entropy loss.
    Lower = better calibrated and more confident correct predictions.
    Penalises confident wrong predictions VERY heavily.
    """
    n = len(y_true)
    total = 0.0
    for yt, p in zip(y_true, y_probs):
        p     = max(eps, min(1 - eps, p))
        total -= yt * math.log(p) + (1 - yt) * math.log(1 - p)
    return total / n


# ─────────────────────────────────────────────────────────────────────────────
# 7. CALIBRATION CURVE
# ─────────────────────────────────────────────────────────────────────────────

def calibration_curve(y_true, y_probs, n_bins=10):
    """
    Compute calibration curve.
    Returns (mean_predicted_probs, fraction_positives_actual, bin_counts)
    Perfect calibration: fraction_positives ≈ mean_predicted_probs (diagonal).
    """
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    mean_pred = []
    frac_pos  = []
    counts    = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = [(p, y) for p, y in zip(y_probs, y_true) if lo <= p < hi]
        if not in_bin:
            continue
        probs, labels = zip(*in_bin)
        mean_pred.append(sum(probs) / len(probs))
        frac_pos.append(sum(labels) / len(labels))
        counts.append(len(in_bin))
    return mean_pred, frac_pos, counts

def expected_calibration_error(y_true, y_probs, n_bins=10):
    """
    ECE = weighted average of |mean_pred - frac_pos| across bins.
    Lower = better calibrated.
    """
    mean_pred, frac_pos, counts = calibration_curve(y_true, y_probs, n_bins)
    n = len(y_true)
    ece = sum(counts[i] / n * abs(mean_pred[i] - frac_pos[i])
              for i in range(len(counts)))
    return ece


# ─────────────────────────────────────────────────────────────────────────────
# 8. MULTI-CLASS METRICS
# ─────────────────────────────────────────────────────────────────────────────

def classification_report(y_true, y_pred):
    """
    Per-class precision, recall, F1 + macro/weighted averages.
    """
    labels = sorted(set(y_true) | set(y_pred))
    report = {}
    for label in labels:
        y_t = [1 if y == label else 0 for y in y_true]
        y_p = [1 if y == label else 0 for y in y_pred]
        p  = precision(y_t, y_p)
        r  = recall(y_t, y_p)
        f  = f1_score(y_t, y_p)
        n  = sum(y_t)
        report[label] = {"precision": p, "recall": r, "f1": f, "support": n}

    # Macro average (unweighted)
    macro_p = sum(report[l]["precision"] for l in labels) / len(labels)
    macro_r = sum(report[l]["recall"]    for l in labels) / len(labels)
    macro_f = sum(report[l]["f1"]        for l in labels) / len(labels)
    report["macro avg"] = {"precision": macro_p, "recall": macro_r, "f1": macro_f,
                            "support": len(y_true)}

    # Weighted average
    total = len(y_true)
    w_p = sum(report[l]["precision"] * report[l]["support"] / total for l in labels)
    w_r = sum(report[l]["recall"]    * report[l]["support"] / total for l in labels)
    w_f = sum(report[l]["f1"]        * report[l]["support"] / total for l in labels)
    report["weighted avg"] = {"precision": w_p, "recall": w_r, "f1": w_f, "support": total}

    return report

def print_classification_report(report):
    header = f"{'class':<15} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"
    print(f"\n{header}")
    print("-" * len(header))
    for label, metrics in report.items():
        if label in ("macro avg", "weighted avg"):
            continue
        print(f"{str(label):<15} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f} "
              f"{metrics['f1']:>10.3f} {metrics['support']:>10}")
    print()
    for avg in ("macro avg", "weighted avg"):
        m = report[avg]
        print(f"{avg:<15} {m['precision']:>10.3f} {m['recall']:>10.3f} "
              f"{m['f1']:>10.3f} {m['support']:>10}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. COMPLETE METRICS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def binary_metrics_summary(y_true, y_pred, y_scores=None):
    """
    Single function returning ALL binary classification metrics.
    """
    TN, FP, FN, TP = binary_confusion(y_true, y_pred)
    p  = precision(y_true, y_pred)
    r  = recall(y_true, y_pred)
    f  = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    kap = cohen_kappa(y_true, y_pred)
    ba  = balanced_accuracy(y_true, y_pred)

    summary = {
        "accuracy":          round(accuracy(y_true, y_pred), 4),
        "precision":         round(p, 4),
        "recall":            round(r, 4),
        "specificity":       round(specificity(y_true, y_pred), 4),
        "f1_score":          round(f, 4),
        "f2_score":          round(f_beta(y_true, y_pred, beta=2), 4),
        "balanced_accuracy": round(ba, 4),
        "mcc":               round(mcc, 4),
        "cohen_kappa":       round(kap, 4),
        "confusion": {"TP": TP, "TN": TN, "FP": FP, "FN": FN},
    }
    if y_scores is not None:
        summary["roc_auc"] = round(roc_auc_score(y_true, y_scores), 4)
        summary["avg_precision"] = round(average_precision_score(y_true, y_scores), 4)
        summary["log_loss"] = round(log_loss(y_true, y_scores), 4)
        summary["ece"] = round(expected_calibration_error(y_true, y_scores), 4)
        t_opt, f1_opt = optimal_threshold_f1(y_true, y_scores)
        summary["optimal_threshold_f1"] = round(t_opt, 4)
        summary["max_f1_at_opt_threshold"] = round(f1_opt, 4)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 10. VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def run_verification():
    import sklearn.metrics as skm
    import random

    print("=" * 60)
    print("Phase 2.1 — Classification Metrics Verification")
    print("=" * 60)

    random.seed(42)
    n = 500
    # Simulate imbalanced binary classification (10% positive rate)
    y_true   = [1 if random.random() < 0.1 else 0 for _ in range(n)]
    y_scores = [
        (random.random() * 0.4 + 0.5 if y else random.random() * 0.5)
        for y in y_true
    ]
    y_pred   = [1 if s >= 0.5 else 0 for s in y_scores]

    checks = [
        ("Accuracy",          accuracy(y_true, y_pred),
                               skm.accuracy_score(y_true, y_pred)),
        ("Precision",         precision(y_true, y_pred),
                               skm.precision_score(y_true, y_pred, zero_division=0)),
        ("Recall",            recall(y_true, y_pred),
                               skm.recall_score(y_true, y_pred, zero_division=0)),
        ("F1",                f1_score(y_true, y_pred),
                               skm.f1_score(y_true, y_pred, zero_division=0)),
        ("MCC",               matthews_corrcoef(y_true, y_pred),
                               skm.matthews_corrcoef(y_true, y_pred)),
        ("Cohen Kappa",       cohen_kappa(y_true, y_pred),
                               skm.cohen_kappa_score(y_true, y_pred)),
        ("ROC-AUC",           roc_auc_score(y_true, y_scores),
                               skm.roc_auc_score(y_true, y_scores)),
        ("Avg Precision",     average_precision_score(y_true, y_scores),
                               skm.average_precision_score(y_true, y_scores)),
        ("Log-loss",          log_loss(y_true, y_scores),
                               skm.log_loss(y_true, y_scores)),
    ]

    all_pass = True
    for name, ours, ref in checks:
        ok = abs(ours - ref) < 0.005
        status = "✓ PASS" if ok else "✗ FAIL"
        if not ok:
            all_pass = False
        print(f"  {name:<20} ours={ours:.4f}  sklearn={ref:.4f}  [{status}]")

    print()
    print("  Full Summary (imbalanced dataset — 10% positive):")
    summary = binary_metrics_summary(y_true, y_pred, y_scores)
    for k, v in summary.items():
        if k != "confusion":
            print(f"    {k:<35}: {v}")
    print(f"    confusion                          : {summary['confusion']}")

    print()
    print("  ⚠️  Accuracy Paradox Demo:")
    y_all_neg = [0] * n
    acc_neg = accuracy(y_true, y_all_neg)
    rec_neg = recall(y_true, y_all_neg)
    print(f"    Predicting all-negative on 10% pos data:")
    print(f"    Accuracy={acc_neg:.3f} (looks great!)  Recall={rec_neg:.3f} (useless!)")

    print()
    print(f"  Overall: {'✓ ALL METRICS VERIFIED' if all_pass else '✗ SOME METRICS FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    run_verification()
