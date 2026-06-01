"""
pipeline.py — ML Failure Investigation Engine
===============================================
Master orchestrator: accepts any sklearn-compatible model + labelled dataset,
runs all analysis modules, and returns a structured investigation report.

Usage:
    from pipeline import InvestigationPipeline
    import pickle, pandas as pd

    model = pickle.load(open("model.pkl", "rb"))
    df    = pd.read_csv("data.csv")
    X     = df.drop(columns=["label"]).values
    y     = df["label"].values
    feat_names = df.drop(columns=["label"]).columns.tolist()

    pipe   = InvestigationPipeline()
    report = pipe.run(model, X, y, feat_names)
    pipe.save_report(report, "output/")
"""

import sys
import os
import time
import json
import traceback
from datetime import datetime

_ROOT = os.path.abspath(os.path.dirname(__file__))
for path in [_ROOT, os.path.join(_ROOT, "scratch", "phase2")]:
    if path not in sys.path:
        sys.path.insert(0, path)


# ─────────────────────────────────────────────────────────────────────────────
# SAFE MODULE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

class ModuleResult:
    def __init__(self, name, status, summary, findings, severity, remediation):
        self.name        = name
        self.status      = status       # 'success' | 'error'
        self.summary     = summary      # one-sentence plain English
        self.findings    = findings     # list of dicts
        self.severity    = severity     # 'critical' | 'moderate' | 'minor' | 'ok'
        self.remediation = remediation  # list of action strings

    def to_dict(self):
        return {
            "module":      self.name,
            "status":      self.status,
            "severity":    self.severity,
            "summary":     self.summary,
            "findings":    self.findings,
            "remediation": self.remediation,
        }


def safe_run(name, fn, *args, **kwargs):
    """Run a module safely — one failure never blocks the rest."""
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = round(time.time() - t0, 2)
        print(f"  ✓  {name:<35} ({elapsed}s)")
        return result
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        print(f"  ✗  {name:<35} FAILED ({elapsed}s): {e}")
        return ModuleResult(
            name=name, status="error",
            summary=f"Module failed: {str(e)}",
            findings=[], severity="minor", remediation=[]
        )


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def run_metrics(model, X_test, y_test, model_name="model"):
    """Module 1: Classification / regression metrics."""
    from scratch.phase2.classification_metrics import binary_metrics_summary
    from scratch.phase2.regression_metrics import regression_summary

    findings    = []
    remediation = []
    severity    = "ok"

    # Auto-detect task
    import numpy as np
    unique = np.unique(y_test)
    is_clf = len(unique) <= 20 and set(unique.tolist()).issubset(set(range(20)))

    if is_clf:
        y_pred = model.predict(X_test)
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1].tolist()

        summary_dict = binary_metrics_summary(
            list(y_test), list(y_pred), y_prob
        )
        findings = [{"metric": k, "value": v}
                    for k, v in summary_dict.items() if k != "confusion"]
        findings.append({"metric": "confusion", "value": summary_dict["confusion"]})

        f1 = summary_dict.get("f1_score", 0)
        if f1 < 0.5:
            severity = "critical"
            remediation.append(f"F1={f1:.3f} is very low. Check class imbalance, feature quality, and model complexity.")
        elif f1 < 0.7:
            severity = "moderate"
            remediation.append(f"F1={f1:.3f} suggests room for improvement. Try feature engineering or threshold tuning.")

        summary = (f"Classification: F1={f1:.3f}, "
                   f"ROC-AUC={summary_dict.get('roc_auc', 'N/A')}, "
                   f"Precision={summary_dict.get('precision', 0):.3f}, "
                   f"Recall={summary_dict.get('recall', 0):.3f}")
    else:
        y_pred  = model.predict(X_test).tolist()
        summary_dict = regression_summary(list(y_test), y_pred)
        findings = [{"metric": k, "value": v} for k, v in summary_dict.items()]
        r2 = summary_dict.get("r2", 0)
        if r2 < 0.5:
            severity = "critical"
            remediation.append(f"R²={r2:.3f} is low. Model explains <50% of variance.")
        summary = f"Regression: RMSE={summary_dict['rmse']:.4f}, R²={summary_dict['r2']:.4f}"

    return ModuleResult("Metric Dashboard", "success", summary, findings, severity, remediation)


def run_validation_checks(X_train, X_test, y_train, y_test, feature_names):
    """Module 2: Data leakage & validation checks."""
    from scratch.phase2.validation_strategy import (
        detect_target_leakage, detect_preprocessing_leakage
    )

    findings    = []
    remediation = []
    severity    = "ok"

    # Target leakage
    tl = detect_target_leakage(X_train, list(y_train), feature_names, threshold_corr=0.9)
    if tl["n_suspects"] > 0:
        severity = "critical"
        for s in tl["suspects"]:
            findings.append({"type": "TARGET_LEAKAGE_SUSPECT",
                              "feature": s["feature"],
                              "correlation": s["correlation"]})
        remediation.append(
            f"HIGH LEAKAGE RISK: {tl['n_suspects']} feature(s) correlate >0.9 with target. "
            f"Investigate: {', '.join(s['feature'] for s in tl['suspects'][:3])}"
        )

    # Preprocessing leakage
    pl = detect_preprocessing_leakage(X_train, X_test, feature_names)
    if pl["n_suspects"] > 0:
        if severity == "ok":
            severity = "moderate"
        for feat in pl["suspect_features"][:5]:
            findings.append({"type": "PREPROCESSING_LEAKAGE_SUSPECT", "feature": feat})
        remediation.append(
            "Possible preprocessing leakage detected. Ensure StandardScaler/imputer "
            "is fit ONLY on training data and transform applied separately."
        )

    if not findings:
        findings.append({"type": "NO_LEAKAGE_DETECTED",
                          "detail": "No obvious leakage patterns found."})

    summary = (f"Leakage check: {tl['n_suspects']} target-leakage suspects, "
               f"{pl['n_suspects']} preprocessing suspects")
    return ModuleResult("Validation Checks", "success", summary, findings, severity, remediation)


def run_fairness_analysis(model, X_test, y_test, sensitive_attr, group_names=None):
    """Module 3: Fairness audit across sensitive attribute."""
    from scratch.phase2.fairness_metrics import fairness_report

    if sensitive_attr is None:
        return ModuleResult(
            "Fairness Audit", "skipped",
            "No sensitive attribute provided — skipped.",
            [], "minor", ["Provide a sensitive_attr array to enable fairness analysis."]
        )

    y_pred      = list(model.predict(X_test))
    group_names = group_names or ["group_0", "group_1"]
    report      = fairness_report(list(y_test), y_pred, list(sensitive_attr), group_names)

    findings    = []
    remediation = []
    severity    = "ok"

    di = report.get("disparate_impact", 1.0)
    dp_gap = abs(report.get("demographic_parity", {}).get("parity_gap", 0))
    eo_tpr = abs(report.get("equalized_odds", {}).get("TPR_gap", 0))

    if di < 0.8:
        severity = "critical"
        findings.append({"metric": "disparate_impact", "value": round(di, 4),
                          "threshold": 0.8, "status": "VIOLATION"})
        remediation.append(
            f"Disparate Impact = {di:.3f} < 0.8. This may violate US employment law "
            f"(EEOC four-fifths rule). Consider re-weighting or adversarial debiasing."
        )
    if dp_gap > 0.1:
        if severity == "ok":
            severity = "moderate"
        findings.append({"metric": "demographic_parity_gap", "value": round(dp_gap, 4)})
        remediation.append(f"Demographic parity gap = {dp_gap:.3f}. Positive rate differs across groups.")
    if eo_tpr > 0.1:
        if severity == "ok":
            severity = "moderate"
        findings.append({"metric": "equalized_odds_TPR_gap", "value": round(eo_tpr, 4)})
        remediation.append(f"TPR gap = {eo_tpr:.3f}. Model catches positive cases at different rates across groups.")

    if not findings:
        findings.append({"metric": "all_fairness_checks", "status": "PASS"})

    summary = (f"Fairness: DI={di:.3f}, DP-gap={dp_gap:.3f}, TPR-gap={eo_tpr:.3f}. "
               f"Status={'⚠️ CRITICAL' if severity=='critical' else '✓ OK'}")
    return ModuleResult("Fairness Audit", "success", summary, findings, severity, remediation)


def run_calibration_analysis(model, X_test, y_test):
    """Module 4: Probability calibration check."""
    if not hasattr(model, "predict_proba"):
        return ModuleResult(
            "Calibration", "skipped",
            "Model has no predict_proba — calibration skipped.",
            [], "minor", ["Use a probabilistic model to enable calibration analysis."]
        )
    from scratch.phase2.calibration import (
        expected_calibration_error, brier_score, reliability_curve
    )

    y_prob   = model.predict_proba(X_test)[:, 1].tolist()
    ece      = expected_calibration_error(list(y_test), y_prob)
    bs       = brier_score(list(y_test), y_prob)
    mp, fp, _ = reliability_curve(list(y_test), y_prob, n_bins=10)

    findings    = [{"metric": "ECE", "value": round(ece, 4)},
                   {"metric": "Brier_Score", "value": round(bs, 4)}]
    remediation = []
    severity    = "ok"

    if ece > 0.15:
        severity = "critical"
        remediation.append(
            f"ECE={ece:.3f} indicates poor calibration. "
            "Apply Platt Scaling or Isotonic Regression post-hoc."
        )
    elif ece > 0.05:
        severity = "moderate"
        remediation.append(f"ECE={ece:.3f} — moderate miscalibration. Consider Platt Scaling.")

    summary = f"Calibration: ECE={ece:.4f}, Brier={bs:.4f}"
    return ModuleResult("Calibration", "success", summary, findings, severity, remediation)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE CLASS
# ─────────────────────────────────────────────────────────────────────────────

SEVERITY_ORDER = {"critical": 0, "moderate": 1, "minor": 2, "ok": 3, "skipped": 4}


class InvestigationPipeline:
    """
    Orchestrates all analysis modules.
    Each module is run safely — one failure never blocks the rest.
    """

    def __init__(self, project_name="investigation"):
        self.project_name = project_name

    def run(self, model, X_test, y_test, feature_names,
            X_train=None, y_train=None,
            sensitive_attr=None, group_names=None):
        """
        Run the full investigation pipeline.

        Returns list of ModuleResult objects, sorted by severity.
        """
        import numpy as np
        X_test  = np.array(X_test)  if not hasattr(X_test,  "shape") else X_test
        y_test  = np.array(y_test)  if not hasattr(y_test,  "shape") else y_test
        X_train = X_test  if X_train is None else (np.array(X_train) if not hasattr(X_train, "shape") else X_train)
        y_train = y_test  if y_train is None else (np.array(y_train) if not hasattr(y_train, "shape") else y_train)

        print(f"\n{'='*55}")
        print(f"  ML Failure Investigation Engine")
        print(f"  Project: {self.project_name}")
        print(f"  Samples: {len(X_test)} | Features: {len(feature_names)}")
        print(f"{'='*55}")
        print("  Running modules:")

        results = []

        results.append(safe_run(
            "Metric Dashboard",
            run_metrics, model, X_test, y_test, model.__class__.__name__
        ))

        results.append(safe_run(
            "Validation Checks",
            run_validation_checks,
            X_train.tolist(), X_test.tolist(),
            y_train.tolist(), y_test.tolist(),
            feature_names
        ))

        results.append(safe_run(
            "Fairness Audit",
            run_fairness_analysis, model, X_test, y_test, sensitive_attr, group_names
        ))

        results.append(safe_run(
            "Calibration",
            run_calibration_analysis, model, X_test, y_test
        ))

        # Sort by severity
        results.sort(key=lambda r: SEVERITY_ORDER.get(r.severity, 99))

        print(f"\n  {'─'*50}")
        print("  FINDINGS SUMMARY:")
        for r in results:
            icon = {"critical":"🔴","moderate":"🟡","minor":"🔵","ok":"🟢","skipped":"⚪"}.get(r.severity, "○")
            print(f"  {icon} [{r.severity.upper():<10}] {r.name}: {r.summary[:60]}")

        critical_count = sum(1 for r in results if r.severity == "critical")
        moderate_count = sum(1 for r in results if r.severity == "moderate")

        print(f"\n  Critical: {critical_count} | Moderate: {moderate_count} | "
              f"Total modules: {len(results)}")
        print(f"{'='*55}\n")

        return results

    def save_report(self, results, output_dir="reports"):
        """Save JSON and plain-text reports."""
        os.makedirs(output_dir, exist_ok=True)
        ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # JSON report
        report_dict = {
            "project":       self.project_name,
            "generated_at":  datetime.utcnow().isoformat(),
            "executive_summary": self._executive_summary(results),
            "modules":       [r.to_dict() for r in results],
            "priority_actions": [
                action
                for r in results
                for action in r.remediation
            ][:10],
        }
        json_path = os.path.join(output_dir, f"{self.project_name}_{ts}.json")
        with open(json_path, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)

        # Text summary
        txt_path = os.path.join(output_dir, f"{self.project_name}_{ts}_summary.txt")
        with open(txt_path, "w") as f:
            f.write(f"ML FAILURE INVESTIGATION REPORT\n")
            f.write(f"Project: {self.project_name}\n")
            f.write(f"Generated: {datetime.utcnow().isoformat()}\n\n")
            f.write("EXECUTIVE SUMMARY\n" + "─"*40 + "\n")
            f.write(report_dict["executive_summary"] + "\n\n")
            f.write("MODULE FINDINGS\n" + "─"*40 + "\n")
            for r in results:
                f.write(f"\n[{r.severity.upper()}] {r.name}\n")
                f.write(f"  {r.summary}\n")
                if r.remediation:
                    f.write("  Actions:\n")
                    for action in r.remediation:
                        f.write(f"    • {action}\n")

        print(f"  Reports saved:")
        print(f"    JSON: {json_path}")
        print(f"    Text: {txt_path}")
        return json_path, txt_path

    def _executive_summary(self, results):
        critical = [r for r in results if r.severity == "critical"]
        moderate = [r for r in results if r.severity == "moderate"]
        if critical:
            return (f"CRITICAL issues found in {len(critical)} module(s): "
                    + "; ".join(r.summary[:80] for r in critical[:2])
                    + (". " if moderate else ""))
        if moderate:
            return (f"Moderate issues in {len(moderate)} module(s): "
                    + "; ".join(r.summary[:80] for r in moderate[:2]))
        return "No critical issues found. Model appears well-behaved on this test set."
