"""
Engine Module — Evaluator
==========================
Production-grade model evaluation module for the ML Failure Investigation Engine.

Responsibilities:
  • Run classification and regression metric suites on any (y_true, y_pred) pair
  • Compare multiple models side-by-side
  • Detect performance regressions against a baseline
  • Return structured dicts suitable for the report engine

Usage:
    from engine.modules.evaluator import Evaluator

    ev = Evaluator()
    report = ev.evaluate_classifier(y_true, y_pred, y_prob, model_name="XGBoost v2")
    ev.compare_models([report_a, report_b, report_c])
"""

import math
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL METRIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _confusion_matrix(y_true, y_pred):
    TP = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    FP = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    TN = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    FN = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    return TP, FP, TN, FN


def _roc_auc(y_true, y_prob):
    """Trapezoidal AUC from scratch."""
    pairs = sorted(zip(y_prob, y_true), key=lambda x: -x[0])
    P = sum(y_true)
    N = len(y_true) - P
    if P == 0 or N == 0:
        return 0.5
    tp, fp = 0, 0
    prev_tp, prev_fp = 0, 0
    auc = 0.0
    prev_thr = float("inf")
    for prob, label in pairs:
        if prob != prev_thr:
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
            prev_tp, prev_fp, prev_thr = tp, fp, prob
        if label == 1:
            tp += 1
        else:
            fp += 1
    auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
    return round(auc / (P * N), 4)


def _pr_auc(y_true, y_prob):
    """Area under Precision-Recall curve (trapezoidal)."""
    pairs = sorted(zip(y_prob, y_true), key=lambda x: -x[0])
    P = sum(y_true)
    if P == 0:
        return 0.0
    tp, fp = 0, 0
    precisions, recalls = [1.0], [0.0]
    for prob, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        rec  = tp / P
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recalls.append(rec)
        precisions.append(prec)
    recalls.append(1.0)
    precisions.append(sum(y_true) / len(y_true))
    auc = 0.0
    for i in range(1, len(recalls)):
        auc += (recalls[i] - recalls[i - 1]) * (precisions[i] + precisions[i - 1]) / 2
    return round(auc, 4)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Model evaluation orchestrator.

    Methods
    -------
    evaluate_classifier(y_true, y_pred, y_prob, model_name) -> dict
    evaluate_regressor(y_true, y_pred, model_name)          -> dict
    compare_models(reports)                                  -> dict
    detect_regression(current_report, baseline_report)      -> dict
    per_class_report(y_true, y_pred, class_names)           -> dict
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._history  = []

    # ── CLASSIFICATION ───────────────────────────────────────────────────

    def evaluate_classifier(self, y_true, y_pred, y_prob=None,
                              model_name: str = "model") -> dict:
        """
        Compute full classification evaluation report.

        Parameters
        ----------
        y_true      : list[int]    — binary ground truth
        y_pred      : list[int]    — binary predictions
        y_prob      : list[float]  — probability for class 1 (optional)
        model_name  : str

        Returns structured dict with all standard metrics.
        """
        n = len(y_true)
        TP, FP, TN, FN = _confusion_matrix(y_true, y_pred)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy  = (TP + TN) / n
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        mcc_num   = (TP * TN - FP * FN)
        mcc_den   = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        mcc       = mcc_num / mcc_den if mcc_den > 0 else 0.0
        prevalence = (TP + FN) / n

        report = {
            "model_name":   model_name,
            "task":         "classification",
            "n_samples":    n,
            "prevalence":   round(prevalence, 4),
            "confusion_matrix": {"TP": TP, "FP": FP, "TN": TN, "FN": FN},
            "accuracy":     round(accuracy, 4),
            "precision":    round(precision, 4),
            "recall":       round(recall, 4),
            "f1_score":     round(f1, 4),
            "specificity":  round(specificity, 4),
            "mcc":          round(mcc, 4),
            "fpr":          round(FP / (FP + TN) if (FP + TN) > 0 else 0.0, 4),
            "fnr":          round(FN / (FN + TP) if (FN + TP) > 0 else 0.0, 4),
        }

        if y_prob is not None:
            report["roc_auc"]    = _roc_auc(y_true, y_prob)
            report["pr_auc"]     = _pr_auc(y_true, y_prob)
            report["brier_score"] = round(
                sum((y_prob[i] - y_true[i]) ** 2 for i in range(n)) / n, 6
            )

        report["grade"] = self._grade_classifier(report)
        self._history.append(report)
        return report

    def _grade_classifier(self, r: dict) -> str:
        score = r.get("roc_auc", r["f1_score"])
        if score >= 0.95:  return "A"
        elif score >= 0.90: return "B"
        elif score >= 0.80: return "C"
        elif score >= 0.70: return "D"
        else:               return "F"

    # ── REGRESSION ───────────────────────────────────────────────────────

    def evaluate_regressor(self, y_true, y_pred, model_name: str = "model") -> dict:
        """
        Compute full regression evaluation report.
        """
        n = len(y_true)
        residuals = [y_pred[i] - y_true[i] for i in range(n)]
        mse       = sum(r ** 2 for r in residuals) / n
        rmse      = math.sqrt(mse)
        mae       = sum(abs(r) for r in residuals) / n

        y_mean    = sum(y_true) / n
        ss_res    = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot    = sum((y_true[i] - y_mean) ** 2 for i in range(n))
        r2        = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        adj_r2    = 1 - (1 - r2) * (n - 1) / max(n - 2, 1)

        eps       = 1e-8
        mape      = sum(abs((y_true[i] - y_pred[i]) / (abs(y_true[i]) + eps))
                        for i in range(n)) / n * 100

        max_err   = max(abs(r) for r in residuals)

        report = {
            "model_name":  model_name,
            "task":        "regression",
            "n_samples":   n,
            "mse":         round(mse, 6),
            "rmse":        round(rmse, 6),
            "mae":         round(mae, 6),
            "mape":        round(mape, 4),
            "r2":          round(r2, 4),
            "adj_r2":      round(adj_r2, 4),
            "max_error":   round(max_err, 6),
            "residual_mean": round(sum(residuals) / n, 6),
            "residual_std":  round(
                math.sqrt(sum((r - sum(residuals)/n)**2 for r in residuals) / n), 6
            ),
        }
        report["grade"] = self._grade_regressor(r2)
        self._history.append(report)
        return report

    def _grade_regressor(self, r2: float) -> str:
        if r2 >= 0.95:  return "A"
        elif r2 >= 0.85: return "B"
        elif r2 >= 0.70: return "C"
        elif r2 >= 0.50: return "D"
        else:            return "F"

    # ── MODEL COMPARISON ─────────────────────────────────────────────────

    def compare_models(self, reports: list) -> dict:
        """
        Side-by-side comparison of multiple evaluation reports.

        Parameters
        ----------
        reports : list of dicts from evaluate_classifier or evaluate_regressor

        Returns
        -------
        dict with comparison table and recommended model
        """
        if not reports:
            raise ValueError("No reports provided.")

        task = reports[0].get("task", "classification")
        keys = (
            ["model_name", "accuracy", "precision", "recall", "f1_score",
             "roc_auc", "pr_auc", "mcc", "brier_score"]
            if task == "classification"
            else ["model_name", "rmse", "mae", "r2", "mape"]
        )

        table = []
        for r in reports:
            row = {k: r.get(k, "N/A") for k in keys}
            table.append(row)

        # Primary metric for ranking
        primary = "roc_auc" if task == "classification" and "roc_auc" in reports[0] \
                  else "f1_score" if task == "classification" else "r2"

        best_report = max(
            (r for r in reports if isinstance(r.get(primary), float)),
            key=lambda r: r[primary],
            default=reports[0]
        )

        return {
            "task":          task,
            "n_models":      len(reports),
            "comparison":    table,
            "primary_metric": primary,
            "best_model":    best_report["model_name"],
            "best_score":    best_report.get(primary, "N/A"),
        }

    # ── REGRESSION DETECTION ─────────────────────────────────────────────

    def detect_regression(self, current: dict, baseline: dict,
                           tolerance: float = 0.02) -> dict:
        """
        Compare a current evaluation against a stored baseline.
        Flag if primary metric drops by more than `tolerance`.

        Returns dict with status ("OK" | "WARNING" | "REGRESSION"), delta, and message.
        """
        task    = current.get("task", "classification")
        primary = "roc_auc" if task == "classification" and "roc_auc" in current \
                  else "f1_score" if task == "classification" else "r2"

        cur_val  = current.get(primary, 0.0)
        base_val = baseline.get(primary, 0.0)
        delta    = cur_val - base_val

        if delta < -tolerance * 2:
            status  = "REGRESSION"
            message = (f"REGRESSION DETECTED: {primary} dropped from "
                       f"{base_val:.4f} → {cur_val:.4f} (Δ={delta:.4f}). "
                       "Investigate data drift, label shifts, or code changes.")
        elif delta < -tolerance:
            status  = "WARNING"
            message = (f"WARNING: {primary} decreased slightly: "
                       f"{base_val:.4f} → {cur_val:.4f} (Δ={delta:.4f}). "
                       "Monitor closely.")
        else:
            status  = "OK"
            message = (f"OK: {primary} stable or improved: "
                       f"{base_val:.4f} → {cur_val:.4f} (Δ={delta:+.4f}).")

        return {
            "status":    status,
            "metric":    primary,
            "baseline":  round(base_val, 4),
            "current":   round(cur_val, 4),
            "delta":     round(delta, 4),
            "tolerance": tolerance,
            "message":   message,
        }

    # ── PER-CLASS REPORT ──────────────────────────────────────────────────

    def per_class_report(self, y_true, y_pred, class_names=None) -> dict:
        """
        Multi-class per-class precision, recall, F1, support.
        Works for both binary and multi-class classification.
        """
        classes = sorted(set(y_true) | set(y_pred))
        if class_names is None:
            class_names = {c: str(c) for c in classes}

        results = {}
        for cls in classes:
            tp = sum(1 for a, b in zip(y_true, y_pred) if a == cls and b == cls)
            fp = sum(1 for a, b in zip(y_true, y_pred) if a != cls and b == cls)
            fn = sum(1 for a, b in zip(y_true, y_pred) if a == cls and b != cls)
            support = tp + fn
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            results[class_names.get(cls, str(cls))] = {
                "precision": round(prec, 4),
                "recall":    round(rec, 4),
                "f1_score":  round(f1, 4),
                "support":   support,
            }

        n = len(y_true)
        macro_f1 = sum(v["f1_score"] for v in results.values()) / len(results)
        weighted_f1 = sum(
            v["f1_score"] * v["support"] / n for v in results.values()
        )

        return {
            "per_class":     results,
            "macro_f1":      round(macro_f1, 4),
            "weighted_f1":   round(weighted_f1, 4),
            "accuracy":      round(sum(a == b for a, b in zip(y_true, y_pred)) / n, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    rng = random.Random(0)
    n   = 400
    y_t = [rng.randint(0, 1) for _ in range(n)]
    y_p = [1 if rng.random() < (0.7 if yt else 0.3) else 0 for yt in y_t]
    y_s = [min(0.999, max(0.001, rng.gauss(0.7 if yt else 0.3, 0.1))) for yt in y_t]

    ev  = Evaluator()
    r   = ev.evaluate_classifier(y_t, y_p, y_s, "DemoModel")
    print("Evaluator report:", {k: r[k] for k in
          ["accuracy", "precision", "recall", "f1_score", "roc_auc", "grade"]})

    reg_y_t = [rng.gauss(0, 1) for _ in range(n)]
    reg_y_p = [v + rng.gauss(0, 0.3) for v in reg_y_t]
    rr = ev.evaluate_regressor(reg_y_t, reg_y_p, "DemoRegressor")
    print("Regressor report:", {k: rr[k] for k in ["rmse", "mae", "r2", "grade"]})
    print("Evaluator module OK.")
