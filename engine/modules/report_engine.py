"""
Engine Module — Report Engine
================================
Generates structured HTML and JSON investigation reports.

Aggregates findings from:
  • Evaluator     (classification / regression metrics)
  • Validator     (data quality, leakage, drift)
  • FairnessEngine (fairness audit across axes)
  • CalibrationEngine (calibration quality)

Outputs:
  • JSON report   — machine-readable structured findings
  • HTML report   — browser-viewable investigation report
  • Text report   — console-printable summary

Usage:
    from engine.modules.report_engine import ReportEngine

    re = ReportEngine(project_name="LoanApproval_v2")
    re.add_section("evaluation",   eval_report)
    re.add_section("fairness",     fairness_report)
    re.add_section("calibration",  cal_report)
    re.add_section("validation",   val_report)
    re.save_html("reports/investigation.html")
    re.save_json("reports/findings.json")
    print(re.text_summary())
"""

import json
import os
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# SEVERITY UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

_SEVERITY_RANK = {"OK": 0, "GOOD": 0, "LOW": 1, "MEDIUM": 2, "WARNING": 2,
                   "HIGH": 3, "POOR": 3, "CRITICAL": 4, "ERROR": 4}
_SEVERITY_COLOR = {
    "OK":       "#27ae60",
    "GOOD":     "#27ae60",
    "LOW":      "#f0ad4e",
    "MEDIUM":   "#e67e22",
    "WARNING":  "#e67e22",
    "HIGH":     "#e74c3c",
    "POOR":     "#e74c3c",
    "CRITICAL": "#8e44ad",
    "ERROR":    "#8e44ad",
}

def _severity_color(s: str) -> str:
    return _SEVERITY_COLOR.get(str(s).upper(), "#7f8c8d")

def _overall_severity(sections: dict) -> str:
    worst = "OK"
    for sec in sections.values():
        s = str(sec.get("severity") or sec.get("status") or sec.get("rating") or "OK").upper()
        if _SEVERITY_RANK.get(s, 0) > _SEVERITY_RANK.get(worst, 0):
            worst = s
    return worst


# ─────────────────────────────────────────────────────────────────────────────
# JSON SERIALISATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _json_safe(obj, depth=0):
    """Recursively make an object JSON-serialisable."""
    if depth > 10:
        return str(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v, depth + 1) for v in obj]
    return str(obj)


# ─────────────────────────────────────────────────────────────────────────────
# HTML HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _badge(text, color="#444"):
    return (f'<span style="background:{color};color:#fff;padding:2px 8px;'
            f'border-radius:4px;font-size:0.85em;font-weight:700">{text}</span>')


def _kv_table(d: dict, title: str = "") -> str:
    rows = ""
    for k, v in d.items():
        if isinstance(v, dict):
            v_str = f"<pre>{json.dumps(_json_safe(v), indent=2)}</pre>"
        elif isinstance(v, list):
            v_str = ", ".join(str(x) for x in v[:10]) + ("…" if len(v) > 10 else "")
        else:
            v_str = str(v)
        rows += f"<tr><td style='color:#999;white-space:nowrap'>{k}</td><td>{v_str}</td></tr>"
    return (f"<h4 style='margin:12px 0 4px'>{title}</h4>" if title else "") + \
           f"<table style='width:100%;border-collapse:collapse'>{rows}</table>"


def _section_html(title: str, content: str, severity: str = "OK") -> str:
    color = _severity_color(severity)
    badge = _badge(severity, color)
    return f"""
<div style='border-left:4px solid {color};padding:12px 16px;margin:16px 0;
            background:#1e1e2e;border-radius:0 6px 6px 0'>
  <h3 style='margin:0 0 8px;color:{color}'>{title} {badge}</h3>
  {content}
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# REPORT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class ReportEngine:
    """
    Aggregates module outputs into a unified investigation report.

    Methods
    -------
    add_section(name, data)       — register a module's output
    text_summary()                — console-friendly overview
    save_json(path)               — write machine-readable findings
    save_html(path)               — write browser-viewable report
    get_findings()                — return structured findings dict
    """

    def __init__(self, project_name: str = "ML Investigation",
                  model_name: str = "unnamed_model"):
        self.project_name = project_name
        self.model_name   = model_name
        self.timestamp    = datetime.utcnow().isoformat() + "Z"
        self._sections    = {}

    # ── ADD SECTION ──────────────────────────────────────────────────────

    def add_section(self, name: str, data: dict):
        """Register a module's report under `name`."""
        self._sections[name] = data
        return self

    # ── GET FINDINGS ─────────────────────────────────────────────────────

    def get_findings(self) -> dict:
        severity = _overall_severity(self._sections)
        return {
            "project":   self.project_name,
            "model":     self.model_name,
            "timestamp": self.timestamp,
            "severity":  severity,
            "sections":  _json_safe(self._sections),
        }

    # ── TEXT SUMMARY ─────────────────────────────────────────────────────

    def text_summary(self) -> str:
        sev  = _overall_severity(self._sections)
        lines = [
            "=" * 65,
            f"  ML FAILURE INVESTIGATION REPORT",
            f"  Project:   {self.project_name}",
            f"  Model:     {self.model_name}",
            f"  Generated: {self.timestamp}",
            f"  Overall:   {sev}",
            "=" * 65,
        ]
        for name, data in self._sections.items():
            sec_sev = str(data.get("severity") or data.get("status") or
                          data.get("rating") or "?").upper()
            summary = data.get("summary") or data.get("interpretation") or ""
            lines.append(f"\n  [{name.upper()}]  status={sec_sev}")
            lines.append(f"    {summary[:120]}" if summary else "    (no summary)")
        lines.append("=" * 65)
        return "\n".join(lines)

    # ── SAVE JSON ────────────────────────────────────────────────────────

    def save_json(self, path: str):
        """Write findings as JSON."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_findings(), f, indent=2, default=str)
        print(f"  [ReportEngine] JSON saved → {path}")
        return path

    # ── SAVE HTML ────────────────────────────────────────────────────────

    def save_html(self, path: str):
        """Write a styled HTML investigation report."""
        findings = self.get_findings()
        sev      = findings["severity"]
        sev_col  = _severity_color(sev)

        # ── build section HTML ────────────────────────────────────────
        sections_html = ""

        for name, data in self._sections.items():
            sec_sev = str(data.get("severity") or data.get("status") or
                          data.get("rating") or "OK").upper()
            content = ""

            # Evaluation section
            if name == "evaluation":
                cm = data.get("confusion_matrix", {})
                metrics = {k: v for k, v in data.items()
                           if k not in {"confusion_matrix", "model_name", "task",
                                        "n_samples", "grade", "summary", "status",
                                        "severity", "curve"}}
                content += _kv_table(metrics, "Metrics")
                if cm:
                    content += _kv_table(cm, "Confusion Matrix")

            # Fairness section
            elif name == "fairness":
                violations = data.get("violations", [])
                warnings   = data.get("warnings", [])
                if violations:
                    v_html = "".join(f"<li style='color:#e74c3c'>✗ {v}</li>"
                                     for v in violations)
                    content += f"<ul style='margin:8px 0'>{v_html}</ul>"
                if warnings:
                    w_html = "".join(f"<li style='color:#e67e22'>⚠ {w}</li>"
                                     for w in warnings)
                    content += f"<ul style='margin:8px 0'>{w_html}</ul>"
                for ax, ax_data in data.get("per_axis", {}).items():
                    gs = ax_data.get("group_stats", {})
                    if gs:
                        tbl = _kv_table(
                            {g: f"TPR={s.get('TPR','?')} FPR={s.get('FPR','?')} acc={s.get('accuracy','?')}"
                             for g, s in gs.items()},
                            f"Group Stats — {ax}"
                        )
                        content += tbl

            # Calibration section
            elif name == "calibration":
                cal_keys = {k: v for k, v in data.items()
                            if k not in {"curve", "model_name", "brier_decomp",
                                         "summary", "status", "severity"}}
                content += _kv_table(cal_keys, "Calibration Metrics")

            # Validation section
            elif name == "validation":
                for check in data.get("checks", []):
                    col = _severity_color(check.get("status", "OK"))
                    content += (f"<p style='color:{col};margin:4px 0'>"
                                f"{'✓' if check['status']=='OK' else '✗'} "
                                f"<b>{check.get('check','')}</b>: "
                                f"{check.get('detail','')}</p>")

            # Generic fallback
            else:
                summary = data.get("summary") or data.get("interpretation") or ""
                content += f"<p>{summary}</p>"

            sections_html += _section_html(name.upper(), content, sec_sev)

        # ── assemble full page ─────────────────────────────────────────
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{self.project_name} — ML Failure Investigation</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #12121f; color: #e0e0e0; font-family: 'Segoe UI', sans-serif;
             font-size: 14px; line-height: 1.5; }}
    .wrapper {{ max-width: 900px; margin: 0 auto; padding: 32px 16px; }}
    h1 {{ font-size: 1.6em; color: {sev_col}; margin-bottom: 4px; }}
    h2 {{ font-size: 1.1em; color: #aaa; font-weight: 400; margin-bottom: 24px; }}
    h3 {{ font-size: 1.05em; }}
    h4 {{ font-size: 0.9em; color: #aaa; text-transform: uppercase; letter-spacing: 0.05em; }}
    table {{ font-size: 0.88em; }}
    td {{ padding: 4px 8px; border-bottom: 1px solid #2a2a3a; vertical-align: top; }}
    pre {{ background: #0d0d1a; padding: 8px; border-radius: 4px; font-size: 0.82em;
           overflow-x: auto; color: #cdd6f4; }}
    .overall {{ padding: 16px; background: {sev_col}22; border: 1px solid {sev_col};
                border-radius: 8px; margin-bottom: 24px; }}
    .overall h2 {{ color: {sev_col}; font-weight: 700; font-size: 1.2em; margin-bottom: 4px; }}
    ul {{ padding-left: 20px; }}
    li {{ margin: 2px 0; }}
  </style>
</head>
<body>
<div class="wrapper">
  <h1>🔍 {self.project_name}</h1>
  <h2>ML Failure Investigation Report &nbsp;·&nbsp; {self.timestamp[:19].replace('T',' ')} UTC</h2>

  <div class="overall">
    <h2>Overall Severity: {sev} {_badge(sev, sev_col)}</h2>
    <p style='color:#ccc'>Model: <b>{self.model_name}</b> &nbsp;|&nbsp;
       Sections: {', '.join(self._sections.keys())}</p>
  </div>

  {sections_html}

  <p style='color:#555;margin-top:32px;font-size:0.8em'>
    Generated by ML Failure Investigation Engine &nbsp;·&nbsp;
    For research and educational purposes only.
  </p>
</div>
</body>
</html>"""

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  [ReportEngine] HTML saved → {path}")
        return path


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Minimal fake reports to test rendering pipeline
    eval_r = {
        "model_name": "DemoModel", "task": "classification", "n_samples": 400,
        "accuracy": 0.82, "precision": 0.79, "recall": 0.77, "f1_score": 0.78,
        "roc_auc": 0.87, "brier_score": 0.14,
        "confusion_matrix": {"TP": 140, "FP": 37, "TN": 188, "FN": 35},
        "grade": "B", "severity": "OK",
        "summary": "Classification report for DemoModel."
    }
    fairness_r = {
        "axes": ["race", "gender"], "severity": "HIGH",
        "violations": ["[RACE] Equalized Odds (gap=0.18)"],
        "warnings":   ["[GENDER] Demographic Parity (gap=0.09)"],
        "per_axis": {
            "race": {
                "severity": "HIGH",
                "group_stats": {
                    "White": {"TPR": 0.81, "FPR": 0.19, "accuracy": 0.85},
                    "Black": {"TPR": 0.63, "FPR": 0.37, "accuracy": 0.71},
                }
            }
        },
        "summary": "Fairness audit: HIGH severity. 1 violation detected."
    }
    cal_r = {
        "model_name": "DemoModel", "ece": 0.072, "mce": 0.14,
        "brier_score": 0.17, "rating": "FAIR", "status": "FAIR",
        "summary": "ECE=0.072 (FAIR). Model is overconfident."
    }
    val_r = {
        "status": "WARNING",
        "checks": [
            {"check": "split_ratio", "status": "OK", "detail": "Test=20% of 500. OK."},
            {"check": "label_balance", "status": "WARNING",
             "detail": "Positive rate: train=0.45, test=0.52, diff=0.07."},
        ],
        "summary": "Validation: WARNING. 1/2 checks passed."
    }

    re = ReportEngine("LoanApproval_Investigation", "XGBoost_v3")
    re.add_section("evaluation",  eval_r)
    re.add_section("fairness",    fairness_r)
    re.add_section("calibration", cal_r)
    re.add_section("validation",  val_r)

    print(re.text_summary())

    out_dir = "/mnt/user-data/outputs"
    os.makedirs(out_dir, exist_ok=True)
    re.save_json(f"{out_dir}/sample_findings.json")
    re.save_html(f"{out_dir}/sample_report.html")
    print("ReportEngine module OK.")
