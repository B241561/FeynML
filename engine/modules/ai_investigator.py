"""
Engine Module — AI Investigator
================================
AI-powered investigation agent that transforms raw investigation outputs
into human-readable reasoning, conclusions, risk assessments, and action plans.

Behaves like a senior ML engineer conducting a failure investigation.

Responsibilities:
- Read Investigation Object
- Analyze root causes
- Analyze evidence
- Analyze severity
- Generate narrative reasoning
- Estimate confidence
- Generate actionable recommendations
- Generate executive summary
- Generate technical summary
- Assess risk level

Usage:
    from engine.modules.ai_investigator import AIInvestigator
    from engine.modules.investigation import Investigation
    
    investigator = AIInvestigator()
    result = investigator.analyze(investigation)
"""

import sys
import os
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Add project root to path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from .investigation import Investigation
except ImportError:
    from investigation import Investigation


class AIInvestigator:
    """
    AI Investigator Agent for ML Failure Investigation.
    
    Transforms raw investigation outputs into human-readable reasoning,
    conclusions, risk assessments, and action plans.
    """
    
    def __init__(self, use_llm: bool = False, verbose: bool = False):
        """
        Initialize the AI Investigator.
        
        Args:
            use_llm: Whether to use LLM integration (if available)
            verbose: Enable verbose logging
        """
        self.use_llm = use_llm
        self.verbose = verbose
        self.llm_available = self._check_llm_availability()
        
        if self.use_llm and not self.llm_available:
            self._log("LLM requested but not available. Using deterministic fallback.")

    # ---------------------------------------------------------------------
    # Audience normalization / context
    # ---------------------------------------------------------------------
    # NOTE: The web UI stores audiences as keys like "ml_engineer", while the
    # narrative generators in this module historically used display labels like
    # "ML Engineer". Normalize to display labels everywhere to keep behavior
    # consistent across the stack.
    _AUDIENCE_KEY_TO_LABEL = {
        "ml_engineer": "ML Engineer",
        "ml engineer": "ML Engineer",
        "engineer": "ML Engineer",
        "executive": "Executive",
        "doctor": "Doctor",
        "loan_officer": "Loan Officer",
        "loan officer": "Loan Officer",
        "student": "Student",
        "hr_manager": "HR Manager",
        "hr manager": "HR Manager",
        "insurance_analyst": "Insurance Analyst",
        "insurance analyst": "Insurance Analyst",
        "legal_compliance": "Legal / Compliance Officer",
        "legal / compliance": "Legal / Compliance Officer",
        "legal/compliance": "Legal / Compliance Officer",
        "legal": "Legal / Compliance Officer",
        "compliance": "Legal / Compliance Officer",
        "researcher": "Researcher",
    }

    @staticmethod
    def normalize_audience(audience: str) -> str:
        """
        Normalize audience identifiers into the display labels used by this module.

        Accepts both key-style inputs (e.g. "ml_engineer") and display labels
        (e.g. "ML Engineer").
        """
        if not audience:
            return "ML Engineer"
        a = str(audience).strip()
        # Fast path: already a known display label
        if a in {
            "ML Engineer",
            "Executive",
            "Doctor",
            "Loan Officer",
            "Student",
            "HR Manager",
            "Insurance Analyst",
            "Legal / Compliance Officer",
            "Researcher",
        }:
            return a
        return AIInvestigator._AUDIENCE_KEY_TO_LABEL.get(a.lower(), "ML Engineer")
    
    def _log(self, msg: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[AIInvestigator] {msg}")
    
    def _check_llm_availability(self) -> bool:
        """
        Check if LLM integration is available.
        
        Returns:
            True if LLM is available, False otherwise
        """
        try:
            import google.genai as genai
            return True
        except ImportError:
            return False
    
    def analyze(self, investigation: Investigation, audience: str = "ML Engineer") -> Dict[str, Any]:
        """
        Analyze an investigation and generate human-readable report.
        
        Args:
            investigation: Investigation object from AutoRootCauseEngine
            audience: Target audience for the summary (ML Engineer, Executive, Doctor, Loan Officer, Student)
        
        Returns:
            Dictionary with executive_summary, investigation_findings, impact_assessment,
            confidence_explanation, recommended_actions, technical_notes, risk_level
        """
        self._log("Starting AI investigation analysis...")
        audience = self.normalize_audience(audience)
        
        # Determine which generation method to use
        if self.use_llm and self.llm_available:
            return self._generate_with_llm(investigation, audience)
        else:
            return self._generate_deterministic(investigation, audience)
    
    def _generate_deterministic(self, investigation: Investigation, audience: str = "ML Engineer") -> Dict[str, Any]:
        """
        Generate investigation report using deterministic logic (fallback).
        
        Args:
            investigation: Investigation object
            audience: Target audience for the summary
        
        Returns:
            Dictionary with investigation report
        """
        self._log("Using deterministic narrative generation.")
        
        # Assess risk level
        risk_level = self._assess_risk_level(investigation)
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(investigation, risk_level, audience)
        
        # Generate investigation findings
        investigation_findings = self._generate_investigation_findings(investigation, audience)
        
        # Generate impact assessment
        impact_assessment = self._generate_impact_assessment(investigation, audience)
        
        # Generate confidence explanation
        confidence_explanation = self._generate_confidence_explanation(investigation)
        
        # Generate recommended actions
        recommended_actions = self._generate_recommended_actions(investigation, audience)
        
        # Generate technical notes
        technical_notes = self._generate_technical_notes(investigation)
        
        return {
            "executive_summary": executive_summary,
            "investigation_findings": investigation_findings,
            "impact_assessment": impact_assessment,
            "confidence_explanation": confidence_explanation,
            "recommended_actions": recommended_actions,
            "technical_notes": technical_notes,
            "risk_level": risk_level,
            "risk_breakdown": (getattr(investigation, 'metadata', {}) or {}).get('risk'),
            "risk_explanation": (getattr(investigation, 'metadata', {}) or {}).get('risk_explanation'),
            "generated_at": datetime.now().isoformat(),
            "investigation_id": investigation.investigation_id
        }
    
    def _generate_with_llm(self, investigation: Investigation, audience: str = "ML Engineer") -> Dict[str, Any]:
        """
        Generate investigation report using LLM (if available).
        
        Args:
            investigation: Investigation object
            audience: Target audience for the summary
        
        Returns:
            Dictionary with investigation report
        """
        self._log("Using LLM for narrative generation.")
        
        try:
            import google.genai as genai
            
            # Prepare prompt
            prompt = self._build_llm_prompt(investigation, audience)
            
            # Generate response
            client = genai.Client()
            response = client.models.generate_content(
                model='gemini-pro',
                contents=prompt,
            )
            
            # Parse response (simplified - in production would need structured parsing)
            # For now, fallback to deterministic with LLM enhancement
            result = self._generate_deterministic(investigation, audience)
            result["llm_enhanced"] = True
            result["llm_raw_response"] = response.text
            
            return result
            
        except Exception as e:
            self._log(f"LLM generation failed: {e}. Falling back to deterministic.")
            return self._generate_deterministic(investigation, audience)
    
    def _build_llm_prompt(self, investigation: Investigation, audience: str = "ML Engineer") -> str:
        """
        Build LLM prompt for investigation analysis.
        
        Args:
            investigation: Investigation object
            audience: Target audience for the summary
        
        Returns:
            Prompt string
        """
        audiences_context = {
            "ML Engineer": "Use technical ML terminology.",
            "Doctor": "Use clinical medical terminology. Refer to predictions as clinical decisions. Use patient safety language.",
            "Loan Officer": "Use credit risk and finance terminology. Refer to predictions as loan approval decisions. Mention portfolio risk.",
            "HR Manager": "Use HR and people management terminology. Refer to predictions as employee decisions. Mention attrition and performance.",
            "Student": "Use simple academic language. Refer to predictions as academic performance assessments.",
            "Executive": "Use business impact language. Focus on ROI, revenue risk, and strategic decisions.",
            "Insurance Analyst": "Use actuarial and risk assessment terminology. Refer to predictions as risk scoring decisions.",
            "Legal / Compliance Officer": "Use legal and regulatory terminology. Mention compliance risk, disparate impact, and regulatory exposure.",
            "Researcher": "Use academic and statistical terminology. Mention methodology, validity, and generalizability.",
        }

        audience_context = audiences_context.get(audience, audiences_context["ML Engineer"])

        prompt = f"""You are a senior machine learning investigator. Your job is to analyze model failure investigations and provide clear, professional analysis.

AUDIENCE STYLE GUIDE:
Audience: {audience}
Guidance: {audience_context}

INVESTIGATION DATA:
"""
        prompt += f"\nHealth Status: {investigation.health_status}"
        prompt += f"\nConfidence: {investigation.confidence}%"
        prompt += f"\nInvestigation ID: {investigation.investigation_id}"
        prompt += f"\nTarget Audience: {audience}"
        
        if investigation.root_causes:
            prompt += "\n\nROOT CAUSES:\n"
            for i, cause in enumerate(investigation.root_causes[:5], 1):
                prompt += f"{i}. {cause.cause} (Score: {cause.score}, Severity: {cause.severity})\n"
                if cause.evidence:
                    for ev in cause.evidence[:2]:
                        prompt += f"   Evidence: {ev}\n"
        
        if investigation.recommendations:
            prompt += "\n\nRECOMMENDED ACTIONS:\n"
            for i, rec in enumerate(investigation.recommendations, 1):
                prompt += f"{i}. {rec}\n"
        
        prompt += """

TASK:
Generate a structured analysis with:
1. Executive Summary (max 120 words, synthesized narrative answering: What happened? Why does it matter? What should happen next?)
2. Investigation Findings (detailed analysis of what failed and why)
3. Impact Assessment (what this means for the business/model)
4. Confidence Explanation (why we are/aren't confident)
5. Recommended Actions (actionable next steps)
6. Technical Notes (ML-engineer-friendly details)

Be concise, professional, and avoid hallucinations. Only use the supplied evidence. Do not invent metrics or root causes.
Tailor ALL sections (not just the executive summary) to the target audience.
"""
        return prompt

    def _generate_recommended_actions(self, investigation: Investigation, audience: str = "ML Engineer") -> List[str]:
        """
        Generate audience-adapted recommended actions.
        """
        recs = list(investigation.recommendations or [])
        if not recs:
            return []

        if audience == "ML Engineer":
            return recs

        prefix = {
            "Executive": "Business action",
            "Doctor": "Clinical action",
            "Loan Officer": "Risk action",
            "HR Manager": "HR action",
            "Student": "Action",
            "Insurance Analyst": "Risk action",
            "Legal / Compliance Officer": "Compliance action",
            "Researcher": "Research action",
        }.get(audience, "Action")

        adapted = []
        for r in recs:
            # Light touch: keep the recommendation content, add audience framing.
            adapted.append(f"{prefix}: {r}")
        return adapted
    
    def _assess_risk_level(self, investigation: Investigation) -> str:
        """
        Assess risk level based on investigation data.
        
        Args:
            investigation: Investigation object
        
        Returns:
            Risk level: LOW, MEDIUM, HIGH, or CRITICAL
        """
        # Prefer pre-computed risk in investigation metadata (single source of truth)
        md = getattr(investigation, 'metadata', {}) or {}
        risk_md = md.get('risk') or {}
        level = None
        try:
            level = risk_md.get('level')
        except Exception:
            level = None

        if level:
            return level.upper()

        # Fallback to legacy heuristic if risk metadata not available
        health_status = investigation.health_status
        confidence = investigation.confidence
        high_severity_count = len(investigation.get_high_severity_causes())
        critical_count = len(investigation.get_critical_causes())

        # Critical risk
        if health_status == "Critical" or critical_count > 0:
            return "CRITICAL"

        # High risk
        if health_status == "Warning" and high_severity_count >= 2:
            return "HIGH"

        if confidence >= 80 and high_severity_count >= 1:
            return "HIGH"

        # Medium risk
        if health_status == "Warning" and high_severity_count >= 1:
            return "MEDIUM"

        if confidence >= 60 and len(investigation.root_causes) >= 2:
            return "MEDIUM"

        # Low risk
        if health_status == "Healthy":
            return "LOW"

        # Default to medium for unknown
        return "MEDIUM"
    
    _AUDIENCE_OPENERS = {
        "ML Engineer": "Investigation detected",
        "Executive": "Analysis identified",
        "Doctor": "Clinical review detected",
        "Loan Officer": "Credit review identified",
        "Student": "Review identified",
        "HR Manager": "Workforce review detected",
        "Insurance Analyst": "Underwriting review identified",
        "Legal / Compliance Officer": "Compliance review identified",
        "Researcher": "Study validation identified",
    }

    _AUDIENCE_RISK_PREFIX = {
        "ML Engineer": "Overall risk is classified as",
        "Executive": "Overall business risk is rated",
        "Doctor": "Clinical risk is rated",
        "Loan Officer": "Portfolio risk is rated",
        "Student": "Overall risk is rated",
        "HR Manager": "Workforce decision risk is rated",
        "Insurance Analyst": "Underwriting risk is rated",
        "Legal / Compliance Officer": "Governance risk is rated",
        "Researcher": "Study validity risk is rated",
    }

    def _generate_executive_summary(self, investigation: Investigation, risk_level: str, audience: str = "ML Engineer") -> str:
        """
        Generate evidence-grounded executive summary (max 120 words).

        References top root causes, issue-specific evidence, and risk rationale.
        Audience framing varies; factual findings are shared across audiences.
        """
        return self._compose_evidence_grounded_summary(investigation, risk_level, audience)

    @staticmethod
    def _join_names(names: List[str]) -> str:
        names = [n for n in names if n]
        if not names:
            return ""
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]} and {names[1]}"
        return ", ".join(names[:-1]) + f", and {names[-1]}"

    def _extract_feature_from_cause(self, cause) -> str:
        """Extract a display name from a root cause record."""
        text = cause.cause or ""
        category = cause.category or ""
        if category == "target_leakage":
            return text.replace(" target leakage", "")
        if category == "feature_drift":
            return text.replace(" feature drift", "")
        if category == "slice_degradation":
            return text.replace("Slice failure: ", "")
        if category == "missing_values":
            return text.replace("Missing values in ", "")
        if category == "outliers":
            return text.replace("Outlier increase in ", "")
        if category == "importance_drift":
            return text.replace("Feature importance drift: ", "")
        return text

    def _display_cause_name(self, cause) -> str:
        """Human-readable cause label for executive summaries."""
        name = self._extract_feature_from_cause(cause)
        if cause.category == "calibration":
            return "calibration degradation"
        if cause.category == "target_leakage":
            return name
        if cause.category == "feature_drift":
            return name
        if "label noise" in (cause.cause or "").lower():
            return "label noise"
        return name

    @staticmethod
    def _parse_metric(evidence_items: List[str], patterns: Tuple[str, ...]) -> Optional[str]:
        for item in evidence_items or []:
            for pattern in patterns:
                match = re.search(pattern, item, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        return None

    def _collect_all_evidence(self, investigation: Investigation) -> List[str]:
        evidence = list(investigation.evidence or [])
        for cause in investigation.root_causes:
            evidence.extend(cause.evidence or [])
        return evidence

    def _build_normalized_context(self, investigation: Investigation) -> Dict[str, Any]:
        """Build a normalized facts/context object used across narrative generators.

        Ensures all narrative sections derive from the same structured facts.
        """
        leakage_features = [
            self._extract_feature_from_cause(c)
            for c in investigation.root_causes
            if c.category == "target_leakage"
        ]

        drift_features = [
            self._extract_feature_from_cause(c)
            for c in investigation.root_causes
            if c.category == "feature_drift"
        ]

        # Augment with metadata when available
        md = getattr(investigation, 'metadata', {}) or {}
        if isinstance(md.get('leakage'), dict):
            suspects = (md.get('leakage', {}) or {}).get('findings', {})
            try:
                suspects_list = (suspects or {}).get('suspects') if isinstance(suspects, dict) else (md.get('leakage') or {}).get('suspects', [])
                for s in (suspects_list or []):
                    fname = s.get('feature') if isinstance(s, dict) else None
                    if fname and fname not in leakage_features:
                        leakage_features.append(str(fname))
            except Exception:
                pass

        if isinstance(md.get('drift'), dict):
            try:
                findings = (md.get('drift') or {}).get('findings') or md.get('drift') or {}
                drifted = findings.get('drifted') if isinstance(findings, dict) else []
                if not drifted and isinstance(findings.get('per_feature'), list):
                    drifted = [f.get('feature') for f in findings.get('per_feature', []) if f.get('status') == 'DRIFT']
                for f in (drifted or []):
                    if f and str(f) not in drift_features:
                        drift_features.append(str(f))
            except Exception:
                pass

        label_noise_rate = self._detect_label_noise_rate(investigation)
        calibration_status, calibration_metric = self._assess_calibration(investigation)

        high_severity = investigation.get_high_severity_causes()
        top_causes = investigation.root_causes[:3]
        top_cause_names = [self._display_cause_name(c) for c in top_causes]

        facts = {
            "high_severity_count": len(high_severity),
            "leakage_features": leakage_features,
            "label_noise_rate": label_noise_rate,
            "drift_features": drift_features,
            "calibration_status": calibration_status,
            "calibration_metric": calibration_metric,
            "top_cause_names": top_cause_names,
        }
        return facts

    def _detect_label_noise_rate(self, investigation: Investigation) -> Optional[str]:
        # 1) Check explicit metadata provided by the LabelNoiseEngine
        md = getattr(investigation, 'metadata', {}) or {}
        ln_md = md.get('label_noise') or md.get('label_noise', {})
        if isinstance(ln_md, dict):
            findings = ln_md.get('findings') or ln_md.get('result') or ln_md
            if isinstance(findings, dict):
                est = findings.get('estimated_noise_fraction') or findings.get('estimated_noise_rate')
                if est is not None:
                    try:
                        # expected as float fraction (0.157) or percentage string
                        if isinstance(est, (int, float)):
                            pct = float(est) * 100 if est <= 1 else float(est)
                            return f"{round(pct, 1)}%"
                        if isinstance(est, str):
                            return est if "%" in est else f"{est}%"
                    except Exception:
                        pass

        # 2) Fall back to scanning root causes' evidence (legacy behavior)
        for cause in investigation.root_causes:
            if cause.category == "label_noise" or "label noise" in (cause.cause or "").lower():
                rate = self._parse_metric(
                    cause.evidence,
                    (
                        r"(?:estimated[_ ]?)?(?:label[_ ]?)?noise[_ ]?(?:rate|fraction)[=:\s]+(\d+(?:\.\d+)?%?)",
                        r"noise[_ ]?fraction[=:\s]+(\d+(?:\.\d+)?%?)",
                        r"(\d+(?:\.\d+)?%)\s+label noise",
                    ),
                )
                if rate:
                    return rate if "%" in rate else f"{rate}%"

        rate = self._parse_metric(
            self._collect_all_evidence(investigation),
            (
                r"(?:estimated[_ ]?)?(?:label[_ ]?)?noise[_ ]?(?:rate|fraction)[=:\s]+(\d+(?:\.\d+)?%?)",
                r"noise[_ ]?fraction[=:\s]+(\d+(?:\.\d+)?%?)",
            ),
        )
        if rate:
            return rate if "%" in rate else f"{rate}%"
        return None

    def _assess_calibration(self, investigation: Investigation) -> Tuple[Optional[str], Optional[str]]:
        """
        Returns (status, metric) where status is 'issue', 'strong', or None.
        """
        calibration_causes = investigation.get_causes_by_category("calibration")
        all_evidence = self._collect_all_evidence(investigation)

        # 1) Inspect metadata from CalibrationEngine if available
        md = getattr(investigation, 'metadata', {}) or {}
        cal_md = md.get('calibration') or {}
        if isinstance(cal_md, dict):
            # CalibrationEngine.evaluate returns 'ece' or 'best_ece' depending on method
            # It may be nested under {'findings': ...} or present directly.
            findings = cal_md.get('findings') or cal_md
            ece_val = None
            if isinstance(findings, dict):
                ece_val = findings.get('ece') or findings.get('best_ece') or findings.get('raw_ece')
            if ece_val is not None:
                try:
                    ece_num = float(ece_val)
                    # treat as fraction if <=1
                    if ece_num <= 1:
                        ece_pct = round(ece_num * 100, 2)
                        ece_str = f"{ece_pct}%"
                    else:
                        ece_str = f"{round(ece_num, 2)}%"
                    # If calibration causes exist, mark as issue
                    if calibration_causes:
                        return "issue", ece_str
                    # Else, decide if strong
                    if ece_num <= 0.05 or (ece_num <= 5 and ece_num > 1):
                        return "strong", ece_str
                except Exception:
                    pass

        # 2) Fallback to parsing evidence like before
        ece = self._parse_metric(
            all_evidence,
            (r"ECE[=:\s]+(\d+(?:\.\d+)?%?)", r"expected calibration error[=:\s]+(\d+(?:\.\d+)?%?)"),
        )

        if calibration_causes:
            metric = ece
            if not metric and calibration_causes[0].evidence:
                metric = self._parse_metric(
                    calibration_causes[0].evidence,
                    (r"ECE[=:\s]+(\d+(?:\.\d+)?%?)",),
                )
            return "issue", metric or "elevated"

        if ece:
            ece_val = ece.replace("%", "")
            try:
                if float(ece_val) <= 5.0:
                    return "strong", ece if "%" in ece else f"{ece}%"
            except ValueError:
                pass
        return None, None

    def _severity_count_phrase(self, count: int) -> str:
        if count == 0:
            return "no high-severity issues"
        if count == 1:
            return "one high-severity issue"
        return f"{count} high-severity issues"

    def _build_risk_rationale(
        self, investigation: Investigation, risk_level: str, facts: Dict[str, Any]
    ) -> str:
        reasons: List[str] = []

        if risk_level == "LOW":
            return "no material findings warrant escalation"

        # Immediate critical rationale
        if risk_level == "CRITICAL" and investigation.get_critical_causes():
            reasons.append("critical-severity findings require immediate remediation")

        # Leakage rationale with explicit features when available
        if facts.get("leakage_features"):
            lf = facts.get("leakage_features")
            shown = self._join_names(lf[:3])
            reasons.append(f"leakage can invalidate model evaluation (features: {shown})")

        # Label noise rationale with estimated rate when available
        if facts.get("label_noise_rate"):
            ln = facts.get("label_noise_rate")
            reasons.append(f"noisy labels may reduce prediction reliability (estimated rate: {ln})")

        # Drift rationale with counts and examples
        if facts.get("drift_features"):
            df = facts.get("drift_features")
            if len(df) == 1:
                reasons.append(f"feature drift observed in {df[0]}")
            else:
                shown = self._join_names(df[:3])
                reasons.append(f"{len(df)} features exhibit measurable drift ({shown})")

        # Calibration rationale with metric
        if facts.get("calibration_status") == "issue":
            metric = facts.get("calibration_metric") or "elevated ECE"
            reasons.append(f"poor calibration can mislead decision thresholds (ECE={metric})")

        # High severity count rationale
        if facts.get("high_severity_count", 0) >= 2:
            reasons.append(f"{facts['high_severity_count']} high-severity issues compound operational risk")
        elif facts.get("high_severity_count", 0) == 1 and not reasons:
            top = facts.get("top_cause_names", ["a flagged issue"])[0]
            reasons.append(f"{top} is high severity and requires attention")

        if not reasons:
            if risk_level == "MEDIUM":
                reasons.append("multiple moderate findings warrant close monitoring")
            else:
                reasons.append(
                    f"health status is {investigation.health_status.lower()} with {investigation.confidence}% confidence"
                )

        # Return up to three concise reasons joined by semicolons
        return "; ".join(reasons[:3])

    def _compose_evidence_grounded_summary(
        self, investigation: Investigation, risk_level: str, audience: str
    ) -> str:
        opener = self._AUDIENCE_OPENERS.get(audience, self._AUDIENCE_OPENERS["ML Engineer"])
        risk_prefix = self._AUDIENCE_RISK_PREFIX.get(audience, self._AUDIENCE_RISK_PREFIX["ML Engineer"])

        if not investigation.root_causes:
            summary = (
                f"{opener} no significant degradation. Model operates within expected parameters "
                f"({investigation.confidence}% confidence). {risk_prefix} {risk_level} "
                f"because {self._build_risk_rationale(investigation, risk_level, {'high_severity_count': 0, 'leakage_features': [], 'label_noise_rate': None, 'drift_features': [], 'calibration_status': None, 'top_cause_names': []})}."
            )
            return self._truncate_to_word_limit(summary, 120)

        top_causes = investigation.root_causes[:3]
        high_severity = investigation.get_high_severity_causes()
        leakage_features = [
            self._extract_feature_from_cause(c)
            for c in investigation.root_causes
            if c.category == "target_leakage"
        ]
        drift_features = [
            self._extract_feature_from_cause(c)
            for c in investigation.root_causes
            if c.category == "feature_drift"
        ]

        # Augment lists with engine metadata (prefer metadata-sourced facts)
        md = getattr(investigation, 'metadata', {}) or {}
        # Leakage metadata: look for suspects -> list of {feature,...}
        try:
            leak_md = md.get('leakage') or {}
            suspects = (leak_md.get('findings') or leak_md).get('suspects') if isinstance((leak_md.get('findings') or leak_md), dict) else (leak_md.get('suspects') or [])
        except Exception:
            suspects = md.get('leakage', {}).get('suspects', []) if isinstance(md.get('leakage', {}), dict) else []
        for s in (suspects or []):
            fname = None
            if isinstance(s, dict):
                fname = s.get('feature') or s.get('name')
            elif isinstance(s, (list, tuple)) and len(s) > 0:
                fname = s[0]
            if fname and fname not in leakage_features:
                leakage_features.append(str(fname))

        # Drift metadata: prefer 'drifted' list or per_feature entries
        try:
            drift_md = md.get('drift') or {}
            findings = (drift_md.get('findings') or drift_md) if isinstance((drift_md.get('findings') or drift_md), dict) else drift_md
            drifted = findings.get('drifted') or findings.get('drifted_features') or []
            if not drifted and isinstance(findings.get('per_feature'), list):
                drifted = [f.get('feature') for f in findings.get('per_feature', []) if f.get('status') == 'DRIFT' or f.get('significant_drift')]
        except Exception:
            drifted = md.get('drift', {}).get('drifted', []) if isinstance(md.get('drift', {}), dict) else []
        for f in (drifted or []):
            if f and str(f) not in drift_features:
                drift_features.append(str(f))
        # Prefer metrics coming from investigation.metadata when available
        label_noise_rate = self._detect_label_noise_rate(investigation)
        calibration_status, calibration_metric = self._assess_calibration(investigation)
        top_cause_names = [self._display_cause_name(c) for c in top_causes]

        facts = {
            "high_severity_count": len(high_severity),
            "leakage_features": leakage_features,
            "label_noise_rate": label_noise_rate,
            "drift_features": drift_features,
            "calibration_status": calibration_status,
            "calibration_metric": calibration_metric,
            "top_cause_names": top_cause_names,
        }

        # Build concise executive summary (max 3 sentences) per new guidance.
        # Sentence 1: Overall risk level + confidence
        # Sentence 2: Top 2 most critical issues only
        # Sentence 3: Single most important action

        # Determine primary concerns: prefer high/critical severity root causes
        critical_or_high = [c for c in investigation.root_causes if c.severity in ("CRITICAL", "HIGH")]
        primary_issues = []
        if critical_or_high:
            # order by severity then score
            primary_sorted = sorted(critical_or_high, key=lambda c: (0 if c.severity == 'CRITICAL' else 1, -getattr(c, 'score', 0)))
            primary_issues = [self._display_cause_name(c) for c in primary_sorted[:2]]
        else:
            # fallback to top causes by score
            primary_issues = [self._display_cause_name(c) for c in investigation.root_causes[:2]]

        # Prepare action
        top_action = None
        try:
            if investigation.recommendations and len(investigation.recommendations) > 0:
                top_action = investigation.recommendations[0]
        except Exception:
            top_action = None

        # Sentence constructions
        sent1 = f"Model health is {risk_level} with {investigation.confidence}% confidence."

        if primary_issues:
            if len(primary_issues) == 1:
                sent2 = f"Primary concerns: {primary_issues[0]}."
            else:
                # primary_issues already ordered by severity then score (PSI-weighted)
                sent2 = f"Primary concerns: {primary_issues[0]} (fix first), then {primary_issues[1]}."
        else:
            sent2 = "Primary concerns: none identified."

        if top_action:
            try:
                if len(top_action) <= 80:
                    action = top_action.rstrip()
                    if not action.endswith('.'):
                        action = action + '.'
                else:
                    action = top_action[:80].rsplit(' ', 1)[0] + '.'
            except Exception:
                action = (top_action[:80].rsplit(' ', 1)[0] + '.') if top_action else 'Review root cause analysis for next steps.'
            # Prepend top-priority issue to action for clarity
            top_issue = primary_issues[0] if primary_issues else None
            if top_issue and top_issue.lower() not in action.lower():
                sent3 = f"Immediate action required: address {top_issue} first — {action}"
            else:
                sent3 = f"Immediate action required: {action}"
        else:
            sent3 = "Immediate action required: review root cause analysis for next steps."

        concise = " ".join([sent1, sent2, sent3])
        return concise

    @staticmethod
    def _join_issue_clauses(clauses: List[str]) -> str:
        if not clauses:
            return ""
        if len(clauses) == 1:
            return clauses[0]
        if len(clauses) == 2:
            return f"{clauses[0]} and {clauses[1]}"
        return ", ".join(clauses[:-1]) + f", and {clauses[-1]}"
    
    def _truncate_to_word_limit(self, text: str, max_words: int) -> str:
        """Truncate text to maximum word limit."""
        words = text.split()
        if len(words) <= max_words:
            return text
        return ' '.join(words[:max_words])
    
    def _generate_investigation_findings(self, investigation: Investigation, audience: str = "ML Engineer") -> str:
        """
        Generate detailed investigation findings as synthesized narrative.
        
        This provides interpretation layer, not evidence layer.
        Root Cause Analysis contains the raw evidence (PSI, KS, scores).
        
        Args:
            investigation: Investigation object
        
        Returns:
            Investigation findings string
        """
        if not investigation.root_causes:
            base = "No significant issues detected during investigation. All monitored metrics are within acceptable ranges."
            return self._apply_audience_context(base, audience, section="findings")

        # Use normalized context to ensure consistency across narrative sections
        facts = self._build_normalized_context(investigation)
        findings = []

        # Prepare optional top-cause headline to restore legacy test expectations
        top_causes = investigation.root_causes[:1]
        top_headline = None
        if top_causes:
            tc = top_causes[0]
            display = self._display_cause_name(tc)
            if tc.category == 'feature_drift':
                top_headline = f"{display} Drift detected."
            elif tc.category == 'target_leakage':
                top_headline = f"{display} Leakage detected."
            elif tc.category == 'calibration':
                top_headline = "Calibration degradation detected."
            else:
                # Generic headline for other top causes
                top_headline = f"{display} detected."

        # Feature drift interpretation
        if facts.get('drift_features'):
            n = len(facts['drift_features'])
            if n == 1:
                findings.append("A single feature exhibits distributional shift, indicating localized data drift that may affect predictions for specific segments.")
            elif n <= 3:
                findings.append(f"{n} features show distributional changes, suggesting broader shifts in the data landscape that could impact model reliability across multiple dimensions.")
            else:
                findings.append(f"Widespread distributional drift detected across {n} features, indicating significant changes in the underlying data distribution that may require comprehensive model retraining.")

        # Slice degradation interpretation (derive from root causes)
        slice_causes = [c for c in investigation.root_causes if c.category == 'slice_degradation']
        if slice_causes:
            if len(slice_causes) == 1:
                findings.append("Performance degradation identified in a specific data segment, indicating the model may not generalize well to that particular subgroup.")
            else:
                findings.append(f"Multiple data segments ({len(slice_causes)}) show performance degradation, suggesting systematic issues with model generalization across different population subgroups.")

        # Calibration interpretation
        if facts.get('calibration_status') == 'issue':
            findings.append("Calibration metrics indicate probability estimates have shifted, potentially affecting the reliability of risk assessments and decision-making thresholds.")
        elif facts.get('calibration_status') == 'strong':
            findings.append("Calibration remains strong based on reported ECE, indicating probability estimates are reliable.")

        # Data quality interpretation
        missing_causes = [c for c in investigation.root_causes if c.category in ['missing_values', 'outliers']]
        if missing_causes:
            findings.append("Data quality issues have been detected that may be contributing to model performance degradation, requiring investigation of upstream data pipelines.")

        # Feature importance interpretation
        importance_causes = [c for c in investigation.root_causes if c.category == 'importance_drift']
        if importance_causes:
            findings.append("Feature importance patterns have changed significantly, indicating the model's decision logic has shifted and may no longer align with the original training assumptions.")

        # Synthesize into cohesive narrative
        if findings:
            narrative = " ".join(findings)
            # Add context about severity (without recommendation language)
            high_severity = [c for c in investigation.root_causes if c.severity in ["HIGH", "CRITICAL"]]
            if high_severity:
                narrative += f" The presence of {len(high_severity)} high-severity issue(s) indicates significant model degradation."
            # Prepend top-cause headline if available to preserve legacy expectations
            if top_headline:
                narrative = f"{top_headline} {narrative}"
            return self._apply_audience_context(narrative, audience, section="findings")
        else:
            base = "The investigation identified several issues affecting model performance. Review the Root Cause Analysis for detailed evidence and severity assessments."
            # Even in the empty-findings fallback, include top_headline if present
            if top_headline:
                base = f"{top_headline} {base}"
            return self._apply_audience_context(base, audience, section="findings")
    
    def _generate_impact_assessment(self, investigation: Investigation, audience: str = "ML Engineer") -> str:
        """
        Generate impact assessment.
        
        Args:
            investigation: Investigation object
        
        Returns:
            Impact assessment string
        """
        if not investigation.root_causes:
            base = "No significant impact expected. Model performance remains stable."
            return self._apply_audience_context(base, audience, section="impact")

        facts = self._build_normalized_context(investigation)
        impact = ""

        if facts.get('drift_features'):
            impact += "Distributional drift may affect predictions for segments of the population that differ from training data. "

        slice_causes = [c for c in investigation.root_causes if c.category == 'slice_degradation']
        if slice_causes:
            impact += "Specific customer segments may experience reduced prediction accuracy. "

        if facts.get('calibration_status') == 'issue':
            impact += "Probability estimates may be less reliable for decision-making. "
        elif facts.get('calibration_status') == 'strong':
            impact += "Probability estimates appear reliable based on reported ECE. "

        missing_causes = [c for c in investigation.root_causes if c.category in ['missing_values', 'outliers']]
        if missing_causes:
            impact += "Data quality issues may be affecting model performance. "

        if any(c.category == 'importance_drift' for c in investigation.root_causes):
            impact += "Model behavior has shifted, potentially affecting feature importance and interpretation. "

        # Add severity context (without duplicating Investigation Findings)
        if investigation.health_status in ["Critical", "Warning"]:
            impact += "Prediction reliability may be compromised for affected segments."

        base = impact.strip() if impact else "Model performance may be affected by the identified issues."
        return self._apply_audience_context(base, audience, section="impact")

    def _apply_audience_context(self, text: str, audience: str, section: str) -> str:
        """
        Minimal, targeted audience adaptation for deterministic narratives.

        This does not attempt full rewriting (that would require an LLM), but it
        adds domain framing so non-ML audiences don't see purely technical wording.
        """
        if audience == "ML Engineer":
            return text

        framing = {
            "Doctor": {
                "findings": "Clinical framing: these signals suggest the decision support tool may behave differently for current patients than it did during validation.",
                "impact": "Clinical impact: this may increase the risk of missed or incorrect clinical decisions; use clinical judgment and consider re-validation.",
            },
            "Loan Officer": {
                "findings": "Credit framing: these shifts can change loan approval outcomes and increase portfolio risk if not addressed.",
                "impact": "Portfolio impact: decision accuracy may degrade for certain borrower segments, increasing default risk and potential policy exceptions.",
            },
            "HR Manager": {
                "findings": "HR framing: these changes can affect employee-related decisions (attrition, performance, hiring) for certain groups or time periods.",
                "impact": "People impact: inaccurate predictions may lead to suboptimal retention or performance interventions; validate before acting on results.",
            },
            "Student": {
                "findings": "Plain-language framing: this means the data the model sees now is different from what it learned from, so its outputs may be less reliable.",
                "impact": "Practical impact: decisions based on the model may be less accurate until the issues are fixed and the model is updated.",
            },
            "Executive": {
                "findings": "Business framing: these issues indicate the model may be drifting away from the conditions it was built for, increasing operational risk.",
                "impact": "Business impact: degraded decision quality can create revenue/ROI risk, higher operating costs, and reduced trust in automated workflows.",
            },
            "Insurance Analyst": {
                "findings": "Actuarial framing: these shifts can change risk scoring behavior and distort underwriting segmentation over time.",
                "impact": "Underwriting impact: mis-scoring can affect premium adequacy, loss ratio, and reserve assumptions for certain cohorts.",
            },
            "Legal / Compliance Officer": {
                "findings": "Governance framing: these issues can raise compliance and documentation risk, especially if outcomes differ across protected groups.",
                "impact": "Regulatory impact: degraded or shifting model behavior can increase disparate impact risk and regulatory exposure; document and review controls.",
            },
            "Researcher": {
                "findings": "Research framing: these signals may indicate dataset shift and threaten validity or generalizability of conclusions.",
                "impact": "Methodology impact: results may not generalize; consider re-sampling, recalibration, and reporting uncertainty alongside outcomes.",
            },
        }

        extra = framing.get(audience, {}).get(section)
        if not extra:
            return text
        return f"{extra} {text}"
    
    def _generate_confidence_explanation(self, investigation: Investigation) -> str:
        """
        Generate confidence explanation grounded in investigation evidence.

        Uses normalized facts built from the investigation object to assemble
        a concise, evidence-aware explanation for the computed confidence.
        """
        confidence = investigation.confidence

        # Build normalized facts (uses same extraction logic as other sections)
        facts = self._build_normalized_context(investigation)

        evidence_lines = []

        # High-severity findings
        try:
            hcount = int(facts.get('high_severity_count', 0) or 0)
            if hcount > 0:
                evidence_lines.append(f"{hcount} high-severity findings")
        except Exception:
            pass

        # Drift
        try:
            df = facts.get('drift_features') or []
            if df:
                evidence_lines.append(f"{len(df)} drifted features detected")
        except Exception:
            pass

        # Label noise
        try:
            ln = facts.get('label_noise_rate')
            if ln:
                # Expect ln like '34.9%'
                evidence_lines.append(f"{ln} estimated label noise")
        except Exception:
            pass

        # Calibration metric
        try:
            cal = facts.get('calibration_metric')
            if not cal:
                # Try to extract from investigation metadata directly if present
                md = getattr(investigation, 'metadata', {}) or {}
                cal_md = md.get('calibration') or {}
                if isinstance(cal_md, dict):
                    findings = cal_md.get('findings') or cal_md
                    if isinstance(findings, dict):
                        val = findings.get('ece') or findings.get('best_ece') or findings.get('raw_ece')
                        if val is not None:
                            try:
                                v = float(val)
                                if v <= 1:
                                    cal = f"{round(v*100,2)}%"
                                else:
                                    cal = f"{round(v,2)}%"
                            except Exception:
                                cal = str(val)
            if cal:
                evidence_lines.append(f"Calibration ECE of {cal}")
        except Exception:
            pass

        # Leakage
        try:
            leaks = facts.get('leakage_features') or []
            if leaks:
                evidence_lines.append(f"Target leakage detected in {len(leaks)} feature(s)")
            else:
                evidence_lines.append("No target leakage detected")
        except Exception:
            evidence_lines.append("No target leakage detected")

        # Compose the explanation
        text = (
            f"Confidence score of {confidence}% is based on:\n\n"
            + "\n".join(f"• {item}" for item in evidence_lines)
        )

        # Evidence quality summary
        try:
            if confidence >= 85:
                text += "\n\nEvidence quality is strong and supports a high-confidence diagnosis."
            elif confidence >= 70:
                text += "\n\nEvidence quality is moderate-to-strong and supports the investigation findings."
            else:
                text += "\n\nEvidence quality is limited; findings should be interpreted cautiously."
        except Exception:
            text += "\n\nEvidence quality is moderate."

        return text
    
    def _generate_technical_notes(self, investigation: Investigation) -> str:
        """
        Generate technical notes (ML-engineer-friendly).
        
        Args:
            investigation: Investigation object
        
        Returns:
            Technical notes string
        """
        notes = "Technical Summary:\n\n"
        
        # Group by category
        by_category = {}
        for cause in investigation.root_causes:
            cat = cause.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(cause)
        
        # Add category summaries
        for category, causes in by_category.items():
            notes += f"{category.upper()}:\n"
            for cause in causes[:3]:
                notes += f"  • {cause.cause} (Score: {cause.score}, Severity: {cause.severity})\n"
            notes += "\n"
        
        # Add source modules
        all_sources = set()
        for cause in investigation.root_causes:
            all_sources.update(cause.source_modules)
        
        if all_sources:
            notes += f"Analysis Engines Used: {', '.join(sorted(all_sources))}\n"
        
        # Add investigation metadata
        notes += f"\nInvestigation ID: {investigation.investigation_id}\n"
        notes += f"Generated At: {investigation.generated_at}\n"
        
        return notes
    
    def generate_investigation_report(self, investigation: Investigation) -> str:
        """
        Generate a full human-readable investigation report.
        
        Args:
            investigation: Investigation object
        
        Returns:
            Formatted report string
        """
        result = self.analyze(investigation)
        
        report = "=" * 70
        report += "\nAI INVESTIGATOR REPORT\n"
        report += "=" * 70
        report += f"\nRisk Level: {result['risk_level']}"
        report += f"\nInvestigation ID: {result['investigation_id']}"
        report += f"\nGenerated At: {result['generated_at']}"
        report += "\n\n"
        
        report += "EXECUTIVE SUMMARY\n"
        report += "-" * 70
        report += f"\n{result['executive_summary']}\n\n"
        
        report += "INVESTIGATION FINDINGS\n"
        report += "-" * 70
        report += f"\n{result['investigation_findings']}\n\n"
        
        report += "IMPACT ASSESSMENT\n"
        report += "-" * 70
        report += f"\n{result['impact_assessment']}\n\n"
        
        report += "CONFIDENCE EXPLANATION\n"
        report += "-" * 70
        report += f"\n{result['confidence_explanation']}\n\n"
        
        report += "RECOMMENDED ACTIONS\n"
        report += "-" * 70
        for i, action in enumerate(result['recommended_actions'], 1):
            report += f"\n{i}. {action}"
        report += "\n\n"
        
        report += "TECHNICAL NOTES\n"
        report += "-" * 70
        report += f"\n{result['technical_notes']}\n"
        
        report += "=" * 70
        
        return report
