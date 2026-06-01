"""
Phase 2 — Failure Report Generator
=====================================
Converts raw metrics dictionaries into plain-English diagnostic reports.

This is the CAPSTONE output of Phase 2:
  Input : y_true, y_pred, fairness audit results, calibration results
  Output: readable HTML + text report explaining WHO the model fails and WHY

Roadmap milestone:
  "Write a plain-English Failure Report explaining who the model fails and why."
"""

import math
import json
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# 1. SEVERITY CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_severity(value, thresholds):
    """
    thresholds = {"critical": x, "warning": y}
    Returns "CRITICAL", "WARNING", or "OK"
    """
    if value >= thresholds.get("critical", float("inf")):
        return "CRITICAL"
    elif value >= thresholds.get("warning", float("inf")):
        return "WARNING"
    return "OK"


# ─────────────────────────────────────────────────────────────────────────────
# 2. OVERALL PERFORMANCE SECTION
# ─────────────────────────────────────────────────────────────────────────────

def performance_section(metrics: dict) -> dict:
    """
    Takes binary_metrics_summary() output.
    Returns human-readable performance narrative.
    """
    acc  = metrics.get("accuracy", None)
    prec = metrics.get("precision", None)
    rec  = metrics.get("recall", None)
    f1   = metrics.get("f1_score", None)
    auc  = metrics.get("roc_auc", None)
    mcc  = metrics.get("mcc", None)
    ba   = metrics.get("balanced_accuracy", None)
    conf = metrics.get("confusion", {})

    lines = []

    # Headline
    if acc is not None:
        lines.append(f"Overall accuracy: {acc*100:.1f}%")

    # Accuracy paradox warning
    if acc is not None and rec is not None and acc > 0.85 and rec < 0.4:
        lines.append(
            "⚠️  ACCURACY PARADOX DETECTED: High accuracy is misleading. "
            f"Recall is only {rec*100:.1f}% — the model misses most positive cases. "
            "This is a classic symptom of class imbalance."
        )

    if f1 is not None:
        severity = "strong" if f1 > 0.75 else ("moderate" if f1 > 0.5 else "weak")
        lines.append(f"F1 score is {f1:.3f} ({severity} overall balance between precision and recall).")

    if auc is not None:
        auc_desc = "excellent" if auc > 0.9 else ("good" if auc > 0.8 else ("fair" if auc > 0.7 else "poor"))
        lines.append(f"ROC-AUC = {auc:.3f} ({auc_desc} discrimination ability).")

    if mcc is not None:
        lines.append(
            f"Matthews Correlation Coefficient = {mcc:.3f} "
            f"({'reliable' if mcc > 0.5 else 'questionable'} given class imbalance)."
        )

    if conf:
        tp, tn, fp, fn = conf.get("TP",0), conf.get("TN",0), conf.get("FP",0), conf.get("FN",0)
        total = tp + tn + fp + fn
        if total > 0:
            lines.append(
                f"Confusion breakdown: {tp} true positives, {tn} true negatives, "
                f"{fp} false positives, {fn} false negatives out of {total} samples."
            )
            if fn > fp * 2:
                lines.append(
                    f"  → The model is biased toward PREDICTING NEGATIVE: "
                    f"it produces {fn} false negatives vs only {fp} false positives. "
                    "Consider lowering the decision threshold."
                )
            elif fp > fn * 2:
                lines.append(
                    f"  → The model is biased toward PREDICTING POSITIVE: "
                    f"it produces {fp} false positives vs only {fn} false negatives. "
                    "Consider raising the decision threshold."
                )

    return {
        "section": "Overall Performance",
        "lines": lines,
        "severity": "WARNING" if (rec is not None and rec < 0.4) else "OK"
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. FAIRNESS SECTION
# ─────────────────────────────────────────────────────────────────────────────

def fairness_section(fairness_results: dict) -> dict:
    """
    Takes output of fairness_audit() from fairness_audit.py.
    Returns plain-English fairness narrative.
    """
    lines    = []
    severity = "OK"

    group_metrics = fairness_results.get("group_metrics", {})
    gaps          = fairness_results.get("fairness_gaps", {})
    warnings      = fairness_results.get("warnings", [])
    sensitive_col = fairness_results.get("sensitive_feature", "group")

    if group_metrics:
        lines.append(f"Fairness audit performed across '{sensitive_col}' groups:")
        for grp, m in group_metrics.items():
            acc  = m.get("accuracy",  "N/A")
            rec  = m.get("recall",    "N/A")
            prec = m.get("precision", "N/A")
            n    = m.get("n",         "?")
            lines.append(
                f"  {grp} (n={n}): accuracy={acc:.3f}, recall={rec:.3f}, precision={prec:.3f}"
                if isinstance(acc, float) else f"  {grp}: insufficient data"
            )

    if gaps:
        lines.append("")
        lines.append("Key fairness gaps detected:")
        for metric_name, gap_val in gaps.items():
            if isinstance(gap_val, (int, float)):
                thresh = 0.1
                flag   = "⚠️ " if abs(gap_val) > thresh else "✓ "
                lines.append(f"  {flag}{metric_name} gap = {gap_val:.3f} (threshold={thresh})")
                if abs(gap_val) > thresh:
                    severity = "WARNING"
                if abs(gap_val) > 0.2:
                    severity = "CRITICAL"

    if warnings:
        lines.append("")
        lines.append("Fairness violations:")
        for w in warnings:
            lines.append(f"  ⚠️  {w}")

    # Disparate impact check
    di = fairness_results.get("disparate_impact")
    if di is not None:
        rule_pass = di >= 0.8
        lines.append(
            f"\nDisparate Impact (80% rule): {di:.3f} → "
            f"{'✓ PASSES (≥0.80)' if rule_pass else '✗ FAILS (<0.80) — legally significant disparity'}"
        )
        if not rule_pass:
            severity = "CRITICAL"

    # Impossibility theorem note
    demographic_parity = gaps.get("demographic_parity_gap")
    equalized_odds     = gaps.get("equalized_odds_gap")
    if demographic_parity is not None and equalized_odds is not None:
        if abs(demographic_parity) < 0.05 and abs(equalized_odds) > 0.1:
            lines.append(
                "\n📌 Fairness Impossibility Theorem in effect: "
                "The model achieves demographic parity but fails equalized odds. "
                "This is mathematically expected when base rates differ across groups "
                "(Chouldechova 2017). You must decide which fairness definition to prioritise."
            )

    return {
        "section": "Fairness Analysis",
        "lines":   lines,
        "severity": severity
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. CALIBRATION SECTION
# ─────────────────────────────────────────────────────────────────────────────

def calibration_section(calibration_results: dict) -> dict:
    """
    Takes output of calibration analysis.
    """
    lines    = []
    severity = "OK"

    ece     = calibration_results.get("ece")
    brier   = calibration_results.get("brier_score")
    method  = calibration_results.get("method", "raw model")
    n_bins  = calibration_results.get("n_bins", 10)

    if ece is not None:
        if ece < 0.05:
            quality = "well-calibrated (ECE < 0.05)"
        elif ece < 0.1:
            quality = "moderately calibrated (ECE 0.05–0.10)"
            severity = "WARNING"
        else:
            quality = "poorly calibrated (ECE > 0.10)"
            severity = "CRITICAL"
        lines.append(f"Calibration quality: {quality}.")
        lines.append(f"Expected Calibration Error (ECE) = {ece:.4f} ({n_bins}-bin estimate).")
        lines.append(
            "This means: when the model says 'I am X% confident', "
            f"the actual frequency of being correct is off by ~{ece*100:.1f}% on average."
        )

    if brier is not None:
        lines.append(f"Brier Score = {brier:.4f} (lower is better; 0=perfect, 0.25=random for 50/50).")

    if ece is not None and ece > 0.1:
        lines.append(
            "\nRecommendation: Apply post-hoc calibration. "
            "Options: Platt Scaling (works well for SVMs/logistic), "
            "Isotonic Regression (works for larger datasets), "
            "Temperature Scaling (works for neural networks)."
        )

    return {
        "section":  "Calibration Analysis",
        "lines":    lines,
        "severity": severity
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. REGRESSION SECTION
# ─────────────────────────────────────────────────────────────────────────────

def regression_section(reg_metrics: dict) -> dict:
    """
    Takes output of regression_summary() from regression_metrics.py.
    """
    lines    = []
    severity = "OK"

    r2      = reg_metrics.get("r2")
    rmse    = reg_metrics.get("rmse")
    mae     = reg_metrics.get("mae")
    mape    = reg_metrics.get("mape")
    bias    = reg_metrics.get("bias_detected")
    mean_r  = reg_metrics.get("mean_residual")

    if r2 is not None:
        quality = "strong" if r2 > 0.9 else ("moderate" if r2 > 0.7 else ("weak" if r2 > 0.5 else "poor"))
        lines.append(f"R² = {r2:.4f} ({quality} explanatory power — model explains {r2*100:.1f}% of variance).")
        if r2 < 0.5:
            severity = "WARNING"

    if rmse is not None and mae is not None:
        ratio = rmse / mae if mae > 1e-10 else 1.0
        lines.append(f"RMSE = {rmse:.4f}, MAE = {mae:.4f} (ratio = {ratio:.2f}).")
        if ratio > 1.5:
            lines.append(
                "  → RMSE/MAE ratio > 1.5 indicates outliers are inflating the error. "
                "Consider investigating high-residual samples."
            )

    if mape is not None:
        lines.append(f"MAPE = {mape:.2f}% (average percentage error).")

    if bias:
        lines.append(
            f"⚠️  SYSTEMATIC BIAS DETECTED: mean residual = {mean_r:.4f}. "
            "The model consistently over- or under-predicts. "
            "Check for missing features or target distribution mismatch."
        )
        severity = "WARNING"

    return {
        "section":  "Regression Diagnostics",
        "lines":    lines,
        "severity": severity
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. RECOMMENDATIONS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_recommendations(sections: list) -> list:
    """
    Look at all section severities and generate prioritised action list.
    """
    recs = []
    priority = 1

    for sec in sections:
        sev = sec.get("severity", "OK")
        name = sec.get("section", "")

        if sev == "CRITICAL" and "Fairness" in name:
            recs.append({
                "priority": priority,
                "action": "Fairness violation requires immediate attention.",
                "detail": (
                    "Run a full bias audit. Consider resampling training data, "
                    "applying reweighing (IBM AIF360), or using an in-processing "
                    "fairness constraint during training."
                )
            })
            priority += 1

        if sev in ("CRITICAL", "WARNING") and "Calibration" in name:
            recs.append({
                "priority": priority,
                "action": "Recalibrate model confidence scores.",
                "detail": (
                    "Apply Platt Scaling or Isotonic Regression on a held-out "
                    "calibration set. Never calibrate on the training set."
                )
            })
            priority += 1

        if sev == "WARNING" and "Performance" in name:
            recs.append({
                "priority": priority,
                "action": "Investigate low recall (many missed positives).",
                "detail": (
                    "Lower the decision threshold, or use class_weight='balanced' "
                    "in sklearn. Consider collecting more positive-class training examples."
                )
            })
            priority += 1

        if sev in ("CRITICAL", "WARNING") and "Regression" in name:
            recs.append({
                "priority": priority,
                "action": "Address systematic prediction bias.",
                "detail": (
                    "Examine residual plots. Look for non-linearity (consider adding "
                    "polynomial features), missing variables, or distribution shift "
                    "between train and test sets."
                )
            })
            priority += 1

    if not recs:
        recs.append({
            "priority": 1,
            "action": "No critical issues found.",
            "detail": "Model passes basic quality checks. Consider moving to Phase 3 explainability analysis."
        })

    return recs


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_failure_report(
    model_name: str = "Model",
    performance_metrics: dict = None,
    fairness_results:    dict = None,
    calibration_results: dict = None,
    regression_metrics:  dict = None,
    dataset_info:        dict = None,
) -> dict:
    """
    Master function: generate the complete plain-English failure report.

    Returns dict with:
      - text_report  : printable string
      - html_report  : HTML string for saving
      - json_report  : structured dict for downstream processing
      - severity     : overall "OK" / "WARNING" / "CRITICAL"
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections  = []

    if performance_metrics:
        sections.append(performance_section(performance_metrics))

    if fairness_results:
        sections.append(fairness_section(fairness_results))

    if calibration_results:
        sections.append(calibration_section(calibration_results))

    if regression_metrics:
        sections.append(regression_section(regression_metrics))

    recommendations = generate_recommendations(sections)

    # Overall severity = worst of all sections
    sev_order  = {"OK": 0, "WARNING": 1, "CRITICAL": 2}
    overall    = max(sections, key=lambda s: sev_order.get(s["severity"], 0))["severity"] if sections else "OK"

    # ── Text report ──────────────────────────────────────────────────
    sep      = "=" * 65
    thin_sep = "-" * 65
    lines    = [
        sep,
        f"  ML FAILURE INVESTIGATION REPORT",
        f"  Model: {model_name}",
        f"  Generated: {timestamp}",
        f"  Overall Status: {overall}",
        sep,
    ]

    if dataset_info:
        lines.append("\nDATASET")
        lines.append(thin_sep)
        for k, v in dataset_info.items():
            lines.append(f"  {k}: {v}")

    for sec in sections:
        lines.append(f"\n{sec['section'].upper()}  [{sec['severity']}]")
        lines.append(thin_sep)
        for line in sec["lines"]:
            lines.append(f"  {line}")

    lines.append(f"\nRECOMMENDATIONS")
    lines.append(thin_sep)
    for rec in recommendations:
        lines.append(f"  [{rec['priority']}] {rec['action']}")
        lines.append(f"      {rec['detail']}")

    lines.append(f"\n{sep}")
    text_report = "\n".join(lines)

    # ── HTML report ──────────────────────────────────────────────────
    sev_color = {"OK": "#28a745", "WARNING": "#fd7e14", "CRITICAL": "#dc3545"}

    html_sections = ""
    for sec in sections:
        color    = sev_color.get(sec["severity"], "#333")
        sec_html = "\n".join(f"<p>{ln}</p>" for ln in sec["lines"] if ln.strip())
        html_sections += f"""
        <div class="section">
          <h2>{sec['section']} <span style="color:{color}; font-size:0.8em;">[{sec['severity']}]</span></h2>
          {sec_html}
        </div>
        """

    rec_html = ""
    for rec in recommendations:
        rec_html += f"""
        <div class="rec">
          <strong>[{rec['priority']}] {rec['action']}</strong>
          <p>{rec['detail']}</p>
        </div>
        """

    html_report = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Failure Report — {model_name}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto;
          padding: 0 20px; color: #333; background: #fafafa; }}
  h1   {{ color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 10px; }}
  h2   {{ color: #16213e; margin-top: 30px; }}
  .section {{ background: white; border-left: 4px solid #e94560;
              padding: 15px 20px; margin: 20px 0; border-radius: 4px;
              box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
  .rec {{ background: #fff3cd; border-left: 4px solid #ffc107;
          padding: 10px 15px; margin: 10px 0; border-radius: 4px; }}
  .overall {{ font-size: 1.2em; font-weight: bold;
              color: {sev_color.get(overall, '#333')}; }}
  p    {{ line-height: 1.6; }}
</style>
</head><body>
<h1>ML Failure Investigation Report</h1>
<p><strong>Model:</strong> {model_name} &nbsp;|&nbsp;
   <strong>Generated:</strong> {timestamp} &nbsp;|&nbsp;
   <span class="overall">Status: {overall}</span></p>
{html_sections}
<div class="section">
<h2>Recommendations</h2>
{rec_html}
</div>
</body></html>"""

    return {
        "text_report":    text_report,
        "html_report":    html_report,
        "json_report":    {
            "model":           model_name,
            "timestamp":       timestamp,
            "overall_severity": overall,
            "sections":        sections,
            "recommendations": recommendations,
        },
        "severity": overall,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. SAVE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def save_text_report(report: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(report["text_report"])
    print(f"  ✓ Text report saved → {path}")

def save_html_report(report: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(report["html_report"])
    print(f"  ✓ HTML report saved → {path}")

def save_json_report(report: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report["json_report"], f, indent=2, default=str)
    print(f"  ✓ JSON report saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. DEMO / VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def run_verification():
    print("=" * 65)
    print("Phase 2 — Failure Report Generator Demo")
    print("=" * 65)

    # Simulate a biased, poorly calibrated model
    perf_metrics = {
        "accuracy": 0.87,
        "precision": 0.81,
        "recall": 0.34,          # Very low — accuracy paradox
        "f1_score": 0.48,
        "roc_auc": 0.76,
        "mcc": 0.41,
        "balanced_accuracy": 0.65,
        "confusion": {"TP": 68, "TN": 802, "FP": 16, "FN": 114},
    }

    fairness = {
        "sensitive_feature": "race",
        "group_metrics": {
            "Caucasian":       {"accuracy": 0.91, "recall": 0.45, "precision": 0.87, "n": 500},
            "African-American": {"accuracy": 0.81, "recall": 0.28, "precision": 0.72, "n": 300},
            "Hispanic":        {"accuracy": 0.84, "recall": 0.31, "precision": 0.78, "n": 200},
        },
        "fairness_gaps": {
            "demographic_parity_gap": 0.17,
            "equalized_odds_gap": 0.21,
            "recall_gap": 0.17,
        },
        "disparate_impact": 0.62,
        "warnings": [
            "Equalized odds gap 0.21 exceeds threshold 0.10",
            "Disparate impact 0.62 fails the 80% rule (threshold=0.80)",
        ],
    }

    calibration = {
        "ece": 0.13,
        "brier_score": 0.18,
        "method": "Random Forest (raw probabilities)",
        "n_bins": 10,
    }

    dataset_info = {
        "name": "COMPAS Recidivism (simulated)",
        "n_samples": 1000,
        "positive_rate": "18.2%",
        "features": 12,
        "train_test_split": "80/20 stratified",
    }

    report = generate_failure_report(
        model_name="Random Forest Classifier",
        performance_metrics=perf_metrics,
        fairness_results=fairness,
        calibration_results=calibration,
        dataset_info=dataset_info,
    )

    print(report["text_report"])

    # Save
    import os
    os.makedirs("reports", exist_ok=True)
    save_html_report(report, "reports/demo_failure_report.html")
    save_json_report(report, "reports/demo_failure_report.json")

    print(f"\n  Overall report severity: {report['severity']}")
    ok = report["severity"] in ("WARNING", "CRITICAL")  # expect non-OK due to biased sim
    print(f"  Severity detection  [{'✓ PASS' if ok else '✗ FAIL'}]")
    print("=" * 65)


if __name__ == "__main__":
    run_verification()
