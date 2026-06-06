"""
=============================================================================
engine/modules/failure_report_generator.py
ML Failure Investigation Engine — Automated Report Generator
=============================================================================
Purpose:
  Aggregates results from ALL Phase 2 analysis modules and generates
  a structured HTML investigation report with:
    • Executive summary
    • Model performance metrics
    • Fairness audit findings
    • Statistical significance results
    • Calibration analysis
    • Failure mode breakdown
    • Actionable recommendations

Usage:
    from engine.modules.failure_report_generator import FailureReportGenerator

    gen = FailureReportGenerator(model_name="COMPAS Risk Model")
    gen.add_metrics(classification_metrics_dict)
    gen.add_fairness(fairness_df)
    gen.add_significance(significance_results_dict)
    gen.add_calibration(calibration_df)
    report_path = gen.generate(output_dir="reports/")
=============================================================================
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: SEVERITY LEVELS & FINDING TYPES
# ─────────────────────────────────────────────────────────────────────────────

class Severity:
    CRITICAL = "CRITICAL"     # Must fix before deployment
    HIGH     = "HIGH"         # Should fix soon
    MEDIUM   = "MEDIUM"       # Worth investigating
    LOW      = "LOW"          # Minor concern
    OK       = "OK"           # Passing

SEVERITY_COLORS = {
    Severity.CRITICAL: "#e74c3c",
    Severity.HIGH:     "#e67e22",
    Severity.MEDIUM:   "#f1c40f",
    Severity.LOW:      "#3498db",
    Severity.OK:       "#2ecc71"
}

SEVERITY_ICONS = {
    Severity.CRITICAL: "🔴",
    Severity.HIGH:     "🟠",
    Severity.MEDIUM:   "🟡",
    Severity.LOW:      "🔵",
    Severity.OK:       "✅"
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: FINDING DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

class Finding:
    """Represents a single discovered issue in the model."""

    def __init__(self, category: str, title: str, description: str,
                 severity: str, metric_value: Optional[float] = None,
                 threshold: Optional[float] = None,
                 recommendation: str = ""):
        self.category = category
        self.title = title
        self.description = description
        self.severity = severity
        self.metric_value = metric_value
        self.threshold = threshold
        self.recommendation = recommendation
        self.timestamp = datetime.datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "severity": self.severity,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "recommendation": self.recommendation
        }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: CORE REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class FailureReportGenerator:
    """
    Aggregates analysis results and generates a complete HTML investigation report.

    Pipeline:
      1. Collect results from each module via add_*() methods
      2. Automatically detect findings and assign severities
      3. Call generate() to produce the HTML report
    """

    def __init__(self, model_name: str = "ML Model",
                 model_version: str = "1.0",
                 dataset_name: str = "Dataset",
                 analyst: str = "ML Failure Investigation Engine"):
        self.model_name    = model_name
        self.model_version = model_version
        self.dataset_name  = dataset_name
        self.analyst       = analyst
        self.report_date   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.findings: List[Finding] = []
        self.metrics: Dict[str, Any] = {}
        self.fairness_df: Optional[pd.DataFrame] = None
        self.significance_results: Dict[str, Any] = {}
        self.calibration_df: Optional[pd.DataFrame] = None
        self.explainability_results: Dict[str, Any] = {}
        self.root_cause_results: Dict[str, Any] = {}
        self.additional_sections: List[Dict] = []

    # ── Data Ingestion ────────────────────────────────────────────────────────

    def add_metrics(self, metrics: Dict[str, Any]) -> "FailureReportGenerator":
        """
        Add classification or regression metrics.
        Expected keys: accuracy, precision, recall, f1, roc_auc, etc.
        """
        self.metrics = metrics
        self._check_performance_findings()
        return self

    def add_fairness(self, fairness_df: pd.DataFrame) -> "FailureReportGenerator":
        """Add per-group fairness metrics DataFrame."""
        self.fairness_df = fairness_df
        self._check_fairness_findings()
        return self

    def add_significance(self, results: Dict[str, Any]) -> "FailureReportGenerator":
        """Add statistical significance test results."""
        self.significance_results = results
        self._check_significance_findings()
        return self

    def add_calibration(self, cal_df: pd.DataFrame) -> "FailureReportGenerator":
        """Add calibration analysis DataFrame."""
        self.calibration_df = cal_df
        self._check_calibration_findings()
        return self

    def add_explainability(self, results: Dict[str, Any]) -> "FailureReportGenerator":
        """Add SHAP/LIME explainability results."""
        self.explainability_results = results
        return self

    def add_root_cause(self, results: Dict[str, Any]) -> "FailureReportGenerator":
        """Add Root Cause Analysis results."""
        self.root_cause_results = results
        return self

    def add_finding(self, finding: Finding) -> "FailureReportGenerator":
        """Manually add a custom finding."""
        self.findings.append(finding)
        return self

    def add_section(self, title: str, content: str) -> "FailureReportGenerator":
        """Add a custom HTML section to the report."""
        self.additional_sections.append({"title": title, "content": content})
        return self

    # ── Auto-Finding Detection ────────────────────────────────────────────────

    def _check_performance_findings(self):
        m = self.metrics

        # Accuracy
        if "accuracy" in m:
            acc = m["accuracy"]
            if acc < 0.60:
                self.findings.append(Finding(
                    category="Performance",
                    title="Critical Low Accuracy",
                    description=f"Model accuracy {acc:.1%} is below 60%.",
                    severity=Severity.CRITICAL,
                    metric_value=acc, threshold=0.60,
                    recommendation="Revisit feature engineering, model architecture, or data quality."
                ))
            elif acc < 0.75:
                self.findings.append(Finding(
                    category="Performance",
                    title="Below-Target Accuracy",
                    description=f"Model accuracy {acc:.1%} is below 75% threshold.",
                    severity=Severity.HIGH,
                    metric_value=acc, threshold=0.75,
                    recommendation="Consider hyperparameter tuning, more training data, or ensembling."
                ))

        # ROC-AUC
        if "roc_auc" in m:
            auc = m["roc_auc"]
            if auc < 0.60:
                self.findings.append(Finding(
                    category="Performance",
                    title="Poor Discrimination (AUC < 0.60)",
                    description=f"AUC={auc:.3f}. Model barely better than random guessing.",
                    severity=Severity.CRITICAL,
                    metric_value=auc, threshold=0.60,
                    recommendation="Check for data leakage, label noise, or fundamental feature inadequacy."
                ))
            elif auc < 0.70:
                self.findings.append(Finding(
                    category="Performance",
                    title="Weak Discrimination (AUC < 0.70)",
                    description=f"AUC={auc:.3f} indicates weak predictive power.",
                    severity=Severity.HIGH,
                    metric_value=auc, threshold=0.70,
                    recommendation="Investigate feature importance, add informative features."
                ))

        # Class imbalance detection
        if "precision" in m and "recall" in m:
            prec, rec = m["precision"], m["recall"]
            if abs(prec - rec) > 0.25:
                self.findings.append(Finding(
                    category="Performance",
                    title="Precision-Recall Imbalance",
                    description=(
                        f"Large gap: Precision={prec:.3f}, Recall={rec:.3f}. "
                        f"Possible class imbalance or threshold mismatch."
                    ),
                    severity=Severity.MEDIUM,
                    recommendation="Check class distribution. Adjust decision threshold or use SMOTE/class weights."
                ))

        # F1 score
        if "f1" in m and m["f1"] < 0.65:
            self.findings.append(Finding(
                category="Performance",
                title="Low F1 Score",
                description=f"F1={m['f1']:.3f} indicates poor balance of precision and recall.",
                severity=Severity.HIGH,
                metric_value=m["f1"], threshold=0.65,
                recommendation="Tune decision threshold or re-examine class balance strategy."
            ))

    def _check_fairness_findings(self):
        if self.fairness_df is None or "FPR" not in self.fairness_df.columns:
            return

        fprs = self.fairness_df["FPR"]
        if len(fprs) >= 2:
            fpr_range = fprs.max() - fprs.min()
            if fpr_range > 0.15:
                groups = self.fairness_df.index.tolist()
                worst = fprs.idxmax()
                self.findings.append(Finding(
                    category="Fairness",
                    title="Disparate False Positive Rate",
                    description=(
                        f"FPR gap of {fpr_range:.1%} across groups {groups}. "
                        f"Group '{worst}' has highest FPR={fprs[worst]:.1%}. "
                        f"This group is disproportionately flagged incorrectly."
                    ),
                    severity=Severity.CRITICAL,
                    metric_value=fpr_range, threshold=0.10,
                    recommendation=(
                        "Apply post-processing threshold adjustment per group, "
                        "or use adversarial debiasing during training."
                    )
                ))
            elif fpr_range > 0.05:
                self.findings.append(Finding(
                    category="Fairness",
                    title="Moderate FPR Disparity",
                    description=f"FPR gap of {fpr_range:.1%} across groups.",
                    severity=Severity.MEDIUM,
                    metric_value=fpr_range, threshold=0.05,
                    recommendation="Monitor closely. Consider fairness constraints in next training run."
                ))

        # Check accuracy parity
        if "accuracy" in self.fairness_df.columns:
            accs = self.fairness_df["accuracy"]
            acc_range = accs.max() - accs.min()
            if acc_range > 0.10:
                self.findings.append(Finding(
                    category="Fairness",
                    title="Accuracy Disparity Across Groups",
                    description=f"Accuracy varies by {acc_range:.1%} across groups.",
                    severity=Severity.HIGH,
                    metric_value=acc_range, threshold=0.10,
                    recommendation="The model may be undertrained for minority groups. Check sample sizes."
                ))

    def _check_significance_findings(self):
        r = self.significance_results

        # Check if model comparison was significant
        if "test" in r and r.get("test") == "McNemar's Test":
            if not r.get("reject_H0", True):
                self.findings.append(Finding(
                    category="Statistical",
                    title="Model Improvement Not Statistically Significant",
                    description=(
                        f"McNemar's test p={r.get('p_value', '?'):.4f} — "
                        f"difference between models is NOT significant."
                    ),
                    severity=Severity.MEDIUM,
                    recommendation="Do not deploy new model based on this result alone. Collect more test data."
                ))

        # Check p-value directly if provided
        if "p_value" in r:
            if r["p_value"] > 0.05 and r.get("reject_H0") == False:
                self.findings.append(Finding(
                    category="Statistical",
                    title="Non-Significant Result",
                    description=f"p-value={r['p_value']:.4f} > 0.05, cannot reject null hypothesis.",
                    severity=Severity.LOW,
                    recommendation="Consider collecting more data or revisiting hypothesis formulation."
                ))

    def _check_calibration_findings(self):
        if self.calibration_df is None:
            return

        if "calibration_error" in self.calibration_df.columns:
            max_error = self.calibration_df["calibration_error"].max()
            if max_error > 0.20:
                self.findings.append(Finding(
                    category="Calibration",
                    title="Poor Probability Calibration",
                    description=f"Max calibration error={max_error:.3f} (>0.20). "
                                f"Predicted probabilities do not reflect true event rates.",
                    severity=Severity.HIGH,
                    metric_value=max_error, threshold=0.20,
                    recommendation="Apply Platt scaling or isotonic regression post-calibration."
                ))
            elif max_error > 0.10:
                self.findings.append(Finding(
                    category="Calibration",
                    title="Moderate Calibration Error",
                    description=f"Max calibration error={max_error:.3f}.",
                    severity=Severity.MEDIUM,
                    recommendation="Consider calibration; Platt scaling is low-cost."
                ))
            else:
                self.findings.append(Finding(
                    category="Calibration",
                    title="Calibration OK",
                    description=f"Max calibration error={max_error:.3f} (< 0.10).",
                    severity=Severity.OK,
                    recommendation="No calibration action needed."
                ))

    # ── Severity Summary ──────────────────────────────────────────────────────

    def get_severity_counts(self) -> Dict[str, int]:
        counts = {s: 0 for s in [Severity.CRITICAL, Severity.HIGH,
                                  Severity.MEDIUM, Severity.LOW, Severity.OK]}
        for f in self.findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1
        return counts

    def overall_status(self) -> str:
        counts = self.get_severity_counts()
        if counts[Severity.CRITICAL] > 0:
            return Severity.CRITICAL
        elif counts[Severity.HIGH] > 0:
            return Severity.HIGH
        elif counts[Severity.MEDIUM] > 0:
            return Severity.MEDIUM
        elif counts[Severity.LOW] > 0:
            return Severity.LOW
        return Severity.OK

    # ── JSON Export ───────────────────────────────────────────────────────────

    def to_json(self) -> str:
        return json.dumps({
            "model_name": self.model_name,
            "model_version": self.model_version,
            "dataset": self.dataset_name,
            "report_date": self.report_date,
            "overall_status": self.overall_status(),
            "severity_counts": self.get_severity_counts(),
            "metrics": self.metrics,
            "findings": [f.to_dict() for f in self.findings]
        }, indent=2, default=str)

    # ── HTML Report Generation ────────────────────────────────────────────────

    def _html_header(self) -> str:
        status = self.overall_status()
        status_color = SEVERITY_COLORS.get(status, "#95a5a6")
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ML Failure Report — {self.model_name}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f0f2f5; color: #2c3e50; line-height: 1.6; }}
  .page {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
  /* Header */
  .report-header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                   color: white; padding: 40px; border-radius: 12px;
                   margin-bottom: 24px; }}
  .report-header h1 {{ font-size: 28px; margin-bottom: 6px; }}
  .report-header .meta {{ opacity: 0.7; font-size: 14px; margin-top: 8px; }}
  .status-badge {{ display: inline-block; padding: 6px 18px; border-radius: 20px;
                  font-weight: 700; font-size: 14px; margin-top: 12px;
                  background: {status_color}; color: white; }}
  /* Cards */
  .card {{ background: white; border-radius: 10px; padding: 24px;
           margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .card h2 {{ font-size: 18px; color: #1a1a2e; margin-bottom: 16px;
             border-bottom: 2px solid #eee; padding-bottom: 8px; }}
  /* Severity summary */
  .severity-grid {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  .sev-card {{ flex: 1; min-width: 100px; text-align: center; padding: 16px;
              border-radius: 8px; color: white; }}
  .sev-card .count {{ font-size: 32px; font-weight: 900; }}
  .sev-card .label {{ font-size: 12px; opacity: 0.9; text-transform: uppercase; }}
  /* Metrics table */
  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
                  gap: 12px; }}
  .metric-box {{ background: #f8f9fa; padding: 14px; border-radius: 8px;
                border-left: 4px solid #3498db; }}
  .metric-box .val {{ font-size: 24px; font-weight: 700; color: #2c3e50; }}
  .metric-box .name {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; }}
  /* Findings */
  .finding {{ border-left: 4px solid; padding: 14px 18px; margin-bottom: 12px;
             border-radius: 0 8px 8px 0; background: #fafafa; }}
  .finding .finding-title {{ font-weight: 600; font-size: 15px; margin-bottom: 4px; }}
  .finding .finding-cat {{ font-size: 11px; text-transform: uppercase;
                          letter-spacing: 0.5px; opacity: 0.65; }}
  .finding .finding-desc {{ font-size: 14px; margin: 6px 0; color: #555; }}
  .finding .finding-rec {{ font-size: 13px; color: #27ae60; font-style: italic; }}
  /* Tables */
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th {{ background: #1a1a2e; color: white; padding: 10px 12px; text-align: left; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #eee; }}
  tr:nth-child(even) td {{ background: #f8f9fa; }}
  /* Footer */
  .footer {{ text-align: center; font-size: 12px; color: #999; padding: 20px; }}
</style>
</head>
<body><div class="page">
"""

    def _html_report_header(self) -> str:
        status = self.overall_status()
        icon = SEVERITY_ICONS.get(status, "⚪")
        return f"""
<div class="report-header">
  <h1>🔍 ML Failure Investigation Report</h1>
  <div style="font-size:20px; margin-top:8px; opacity:0.9">{self.model_name}
       <span style="opacity:0.6; font-size:14px">v{self.model_version}</span></div>
  <div class="meta">
    Dataset: {self.dataset_name} &nbsp;|&nbsp;
    Analyst: {self.analyst} &nbsp;|&nbsp;
    Generated: {self.report_date}
  </div>
  <div class="status-badge">{icon} Overall Status: {status}</div>
</div>
"""

    def _html_severity_summary(self) -> str:
        counts = self.get_severity_counts()
        cards = ""
        for sev, cnt in counts.items():
            color = SEVERITY_COLORS[sev]
            icon = SEVERITY_ICONS[sev]
            cards += f"""
      <div class="sev-card" style="background:{color}">
        <div class="count">{cnt}</div>
        <div class="label">{icon} {sev}</div>
      </div>"""
        return f"""
<div class="card">
  <h2>📊 Findings Summary ({len(self.findings)} total)</h2>
  <div class="severity-grid">{cards}
  </div>
</div>
"""

    def _html_metrics_section(self) -> str:
        if not self.metrics:
            return ""
        boxes = ""
        metric_format = {
            "accuracy": ("Accuracy", ".1%"),
            "precision": ("Precision", ".3f"),
            "recall": ("Recall", ".3f"),
            "f1": ("F1 Score", ".3f"),
            "roc_auc": ("ROC-AUC", ".3f"),
            "pr_auc": ("PR-AUC", ".3f"),
            "mse": ("MSE", ".4f"),
            "mae": ("MAE", ".4f"),
            "r2": ("R² Score", ".3f"),
        }
        for key, (label, fmt) in metric_format.items():
            if key in self.metrics:
                val = self.metrics[key]
                try:
                    formatted = format(val, fmt)
                except:
                    formatted = str(val)
                boxes += f"""
      <div class="metric-box">
        <div class="val">{formatted}</div>
        <div class="name">{label}</div>
      </div>"""

        # Any extra metrics not in the standard list
        for k, v in self.metrics.items():
            if k not in metric_format and isinstance(v, (int, float)):
                boxes += f"""
      <div class="metric-box">
        <div class="val">{v:.4g}</div>
        <div class="name">{k.replace('_', ' ').title()}</div>
      </div>"""

        return f"""
<div class="card">
  <h2>📈 Model Performance Metrics</h2>
  <div class="metrics-grid">{boxes}
  </div>
</div>
"""

    def _html_findings_section(self) -> str:
        if not self.findings:
            return """<div class="card"><h2>✅ No Issues Found</h2>
            <p>All checks passed. Model appears healthy.</p></div>"""

        # Sort findings by severity order
        severity_order = {Severity.CRITICAL: 0, Severity.HIGH: 1,
                          Severity.MEDIUM: 2, Severity.LOW: 3, Severity.OK: 4}
        sorted_findings = sorted(
            self.findings, key=lambda f: severity_order.get(f.severity, 99))

        rows = ""
        for f in sorted_findings:
            color = SEVERITY_COLORS.get(f.severity, "#95a5a6")
            icon = SEVERITY_ICONS.get(f.severity, "⚪")
            metric_str = ""
            if f.metric_value is not None:
                try:
                    metric_str = f" (measured: <strong>{f.metric_value:.4f}</strong>"
                    if f.threshold:
                        metric_str += f", threshold: {f.threshold:.4f}"
                    metric_str += ")"
                except:
                    pass
            rows += f"""
      <div class="finding" style="border-color:{color}">
        <div class="finding-cat">{icon} {f.severity} &nbsp;·&nbsp; {f.category}</div>
        <div class="finding-title">{f.title}</div>
        <div class="finding-desc">{f.description}{metric_str}</div>
        {'<div class="finding-rec">💡 ' + f.recommendation + '</div>' if f.recommendation else ''}
      </div>"""

        return f"""
<div class="card">
  <h2>🚨 Detailed Findings</h2>
  {rows}
</div>
"""

    def _html_fairness_section(self) -> str:
        if self.fairness_df is None:
            return ""

        # Build table
        df = self.fairness_df.reset_index()
        header_cells = "".join(f"<th>{col}</th>" for col in df.columns)
        rows = ""
        for _, row in df.iterrows():
            cells = ""
            for val in row:
                if isinstance(val, float):
                    cells += f"<td>{val:.4f}</td>"
                else:
                    cells += f"<td>{val}</td>"
            rows += f"<tr>{cells}</tr>"

        return f"""
<div class="card">
  <h2>⚖️ Fairness Metrics by Group</h2>
  <table><thead><tr>{header_cells}</tr></thead><tbody>{rows}</tbody></table>
  <p style="margin-top:12px; font-size:13px; color:#666">
    <strong>Key:</strong> TPR=True Positive Rate (Recall), FPR=False Positive Rate,
    FNR=False Negative Rate, PPV=Precision.
    Equal Fairness requires TPR and FPR to be equal across all groups.
  </p>
</div>
"""

    def _html_calibration_section(self) -> str:
        if self.calibration_df is None:
            return ""

        df = self.calibration_df
        header_cells = "".join(f"<th>{col}</th>" for col in df.columns)
        rows = ""
        for _, row in df.iterrows():
            cells = ""
            for val in row:
                if isinstance(val, float):
                    cells += f"<td>{val:.3f}</td>"
                else:
                    cells += f"<td>{val}</td>"
            rows += f"<tr>{cells}</tr>"

        return f"""
<div class="card">
  <h2>🎯 Calibration Analysis</h2>
  <table><thead><tr>{header_cells}</tr></thead><tbody>{rows}</tbody></table>
  <p style="margin-top:12px; font-size:13px; color:#666">
    Good calibration: mean_predicted ≈ actual_recid_rate in each bin.
    High calibration_error means the model is over/under-confident.
  </p>
</div>
"""

    def _html_significance_section(self) -> str:
        if not self.significance_results:
            return ""

        r = self.significance_results
        rows = ""
        for k, v in r.items():
            if isinstance(v, float):
                rows += f"<tr><td>{k}</td><td>{v:.6f}</td></tr>"
            elif isinstance(v, dict):
                rows += f"<tr><td>{k}</td><td>{json.dumps(v)}</td></tr>"
            else:
                rows += f"<tr><td>{k}</td><td>{v}</td></tr>"

        return f"""
<div class="card">
  <h2>📐 Statistical Significance</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>
"""

    def _html_explainability_section(self) -> str:
        if not self.explainability_results:
            return ""
        
        r = self.explainability_results
        global_imp = r.get("global_importance", [])
        global_rows = "".join(f"<li>{item['feature']}: {item['mean_abs_importance']:.4f}</li>" for item in global_imp)
        
        pos = r.get("positive_contributors", [])
        neg = r.get("negative_contributors", [])
        
        pos_list = "".join(f"<li>+ {item['feature']} ({item['value']:.4f})</li>" for item in pos)
        neg_list = "".join(f"<li>- {item['feature']} ({item['value']:.4f})</li>" for item in neg)
        
        return f"""
<div class="card">
  <h2>🧠 Explainability Center (SHAP)</h2>
  <div style="display: flex; gap: 24px;">
    <div style="flex: 1;">
      <h3>Global Feature Importance</h3>
      <ol>{global_rows}</ol>
    </div>
    <div style="flex: 1;">
      <h3>Local Explanation (Individual Prediction)</h3>
      <p><strong>Top Contributors:</strong></p>
      <ul style="color: #27ae60; list-style: none;">{pos_list}</ul>
      <p><strong>Negative Contributors:</strong></p>
      <ul style="color: #e74c3c; list-style: none;">{neg_list}</ul>
    </div>
  </div>
  <p style="margin-top: 16px; font-size: 13px; color: #666;">
    <em>Executive Summary:</em> {global_imp[0]['feature'] if global_imp else 'N/A'} is the strongest driver of model decisions.
  </p>
</div>
"""

    def _html_root_cause_section(self) -> str:
        if not self.root_cause_results:
            return ""
        
        r = self.root_cause_results
        scored = r.get("scored_causes", [])
        cause_rows = "".join(f"""
            <tr>
                <td>{c['cause']}</td>
                <td><div style="background:#eee; border-radius:4px; overflow:hidden;">
                    <div style="background:#3498db; width:{c['confidence']}%; padding:2px 8px; color:white; font-size:10px;">{c['confidence']}%</div>
                </div></td>
                <td>{c['impact']}</td>
            </tr>
        """ for c in scored)
        
        recs = r.get("recommendations", [])
        rec_list = "".join(f"<li>{rec}</li>" for rec in recs)
        
        timeline = r.get("timeline", [])
        timeline_items = "".join(f"<div>↓ {t['event']}</div>" for t in timeline)

        return f"""
<div class="card">
  <h2>🕵️ Root Cause Intelligence Center</h2>
  <h3>Identified Root Causes & Confidence</h3>
  <table>
    <thead><tr><th>Root Cause</th><th>Confidence</th><th>Impact</th></tr></thead>
    <tbody>{cause_rows}</tbody>
  </table>
  
  <div style="margin-top: 20px; display: flex; gap: 24px;">
    <div style="flex: 1;">
      <h3>Investigation Timeline</h3>
      <div style="font-family: monospace; font-size: 13px;">{timeline_items}</div>
    </div>
    <div style="flex: 1;">
      <h3>AI Recommendations</h3>
      <ul>{rec_list}</ul>
    </div>
  </div>
</div>
"""

    def _html_additional_sections(self) -> str:
        out = ""
        for sec in self.additional_sections:
            out += f"""
<div class="card">
  <h2>{sec['title']}</h2>
  {sec['content']}
</div>
"""
        return out

    def _html_recommendations(self) -> str:
        recs = [(f.severity, f.title, f.recommendation)
                for f in self.findings if f.recommendation]
        if not recs:
            return ""

        severity_order = {Severity.CRITICAL: 0, Severity.HIGH: 1,
                          Severity.MEDIUM: 2, Severity.LOW: 3, Severity.OK: 4}
        recs.sort(key=lambda x: severity_order.get(x[0], 99))

        items = ""
        for sev, title, rec in recs:
            icon = SEVERITY_ICONS.get(sev, "⚪")
            color = SEVERITY_COLORS.get(sev, "#95a5a6")
            items += f"""
      <tr>
        <td><span style="color:{color}">{icon} {sev}</span></td>
        <td><strong>{title}</strong></td>
        <td>{rec}</td>
      </tr>"""

        return f"""
<div class="card">
  <h2>💡 Prioritized Recommendations</h2>
  <table>
    <thead><tr><th>Priority</th><th>Issue</th><th>Action</th></tr></thead>
    <tbody>{items}</tbody>
  </table>
</div>
"""

    def _html_footer(self) -> str:
        return f"""
<div class="footer">
  Generated by ML Failure Investigation Engine &nbsp;·&nbsp;
  {self.report_date} &nbsp;·&nbsp;
  Model: {self.model_name} v{self.model_version}
</div>
</div></body></html>"""

    def generate(self, output_dir: str = "reports",
                 filename: Optional[str] = None) -> str:
        """
        Generate the complete HTML report.
        Returns the path to the saved report file.
        """
        os.makedirs(output_dir, exist_ok=True)

        if filename is None:
            safe_name = self.model_name.replace(" ", "_").replace("/", "_")
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"failure_report_{safe_name}_{ts}.html"

        output_path = os.path.join(output_dir, filename)

        html = (
            self._html_header()
            + self._html_report_header()
            + self._html_severity_summary()
            + self._html_metrics_section()
            + self._html_findings_section()
            + self._html_explainability_section()
            + self._html_root_cause_section()
            + self._html_fairness_section()
            + self._html_calibration_section()
            + self._html_significance_section()
            + self._html_additional_sections()
            + self._html_recommendations()
            + self._html_footer()
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"✅ Report generated: {output_path}")
        return output_path

    def save_json(self, output_dir: str = "reports",
                  filename: Optional[str] = None) -> str:
        """Save findings as JSON for programmatic consumption."""
        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"findings_{ts}.json"
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            f.write(self.to_json())
        print(f"✅ JSON saved: {path}")
        return path


# ─────────────────────────────────────────────────────────────────────────────
# DEMO — Full Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_demo():
    """Demonstrate the full report generator pipeline."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    print("=" * 65)
    print("  ML FAILURE REPORT GENERATOR — DEMO")
    print("=" * 65)

    # 1. Simulate classification metrics
    metrics = {
        "accuracy": 0.718,
        "precision": 0.654,
        "recall": 0.821,
        "f1": 0.728,
        "roc_auc": 0.693,
        "pr_auc": 0.631,
        "n_test_samples": 1500
    }

    # 2. Simulate fairness DataFrame
    fairness_data = {
        "group": ["Black", "White"],
        "n": [765, 735],
        "base_rate": [0.523, 0.391],
        "accuracy": [0.698, 0.739],
        "TPR": [0.791, 0.812],
        "FPR": [0.423, 0.198],     # ← Big disparity!
        "FNR": [0.209, 0.188],
        "PPV": [0.602, 0.718],
        "TP": [210, 190], "FP": [178, 85],
        "TN": [242, 343], "FN": [50, 42]
    }
    fairness_df = pd.DataFrame(fairness_data).set_index("group")

    # 3. Simulate calibration DataFrame
    cal_data = {
        "group": ["Black"] * 5 + ["White"] * 5,
        "score_bin": ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"] * 2,
        "n": [80, 120, 200, 230, 135, 95, 140, 180, 195, 125],
        "mean_predicted": [0.12, 0.31, 0.50, 0.70, 0.88,
                           0.11, 0.30, 0.51, 0.69, 0.87],
        "actual_recid_rate": [0.11, 0.29, 0.47, 0.69, 0.84,
                              0.10, 0.27, 0.50, 0.68, 0.83],
        "calibration_error": [0.01, 0.02, 0.03, 0.01, 0.04,
                              0.01, 0.03, 0.01, 0.01, 0.04]
    }
    cal_df = pd.DataFrame(cal_data)

    # 4. Simulate significance result
    significance = {
        "test": "McNemar's Test",
        "model1_acc": 0.718,
        "model2_acc": 0.721,
        "chi2_stat": 1.23,
        "p_value": 0.267,
        "reject_H0": False,
        "interpretation": "No significant difference between models"
    }

    # 5. Build and generate report
    gen = FailureReportGenerator(
        model_name="COMPAS Recidivism Risk Model",
        model_version="2.1",
        dataset_name="Synthetic COMPAS (N=5000)",
        analyst="Phase 2 Investigation Engine"
    )

    (gen
     .add_metrics(metrics)
     .add_fairness(fairness_df)
     .add_significance(significance)
     .add_calibration(cal_df)
     .add_section(
         "📚 Investigation Context",
         """<p>This report analyses the COMPAS recidivism prediction model for
         fairness violations across racial groups. The analysis follows
         ProPublica's 2016 methodology and includes the Chouldechova (2017)
         impossibility theorem implications.</p>
         <p style="margin-top:8px; color:#e74c3c">
         <strong>Important:</strong> This is a synthetic dataset for educational purposes.
         </p>"""
     ))

    # 6. Generate
    report_path = gen.generate(output_dir="/home/claude/ml_failure_engine/reports",
                               filename="compas_investigation_report.html")
    json_path   = gen.save_json(output_dir="/home/claude/ml_failure_engine/reports",
                                filename="compas_findings.json")

    # 7. Console summary
    counts = gen.get_severity_counts()
    print(f"\n  Overall Status : {gen.overall_status()}")
    print(f"  Total Findings : {len(gen.findings)}")
    for sev, cnt in counts.items():
        if cnt > 0:
            print(f"  {SEVERITY_ICONS[sev]} {sev:10s}: {cnt}")

    print(f"\n  HTML Report    : {report_path}")
    print(f"  JSON Report    : {json_path}")
    return report_path, json_path


if __name__ == "__main__":
    run_demo()
