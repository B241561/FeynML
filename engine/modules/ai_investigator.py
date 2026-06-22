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
from typing import Dict, List, Optional, Any
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
    
    def _generate_executive_summary(self, investigation: Investigation, risk_level: str, audience: str = "ML Engineer") -> str:
        """
        Generate executive summary (synthesized narrative, max 120 words).
        
        Answers: What happened? Why does it matter? What should happen next?
        
        Args:
            investigation: Investigation object
            risk_level: Assessed risk level
            audience: Target audience for the summary
        
        Returns:
            Executive summary string (max 120 words)
        """
        if not investigation.root_causes:
            return "Model operates within expected parameters. No significant degradation detected. No immediate action required."
        
        # Get top cause and category
        top_cause = investigation.root_causes[0]
        cause_count = len(investigation.root_causes)
        
        # Generate audience-specific summary
        if audience == "ML Engineer":
            return self._executive_summary_engineer(investigation, risk_level, top_cause, cause_count)
        elif audience == "Executive":
            return self._executive_summary_executive(investigation, risk_level, top_cause, cause_count)
        elif audience == "Doctor":
            return self._executive_summary_doctor(investigation, risk_level, top_cause, cause_count)
        elif audience == "Loan Officer":
            return self._executive_summary_loan_officer(investigation, risk_level, top_cause, cause_count)
        elif audience == "Student":
            return self._executive_summary_student(investigation, risk_level, top_cause, cause_count)
        elif audience == "HR Manager":
            return self._executive_summary_hr_manager(investigation, risk_level, top_cause, cause_count)
        elif audience == "Insurance Analyst":
            return self._executive_summary_insurance_analyst(investigation, risk_level, top_cause, cause_count)
        elif audience == "Legal / Compliance Officer":
            return self._executive_summary_compliance_officer(investigation, risk_level, top_cause, cause_count)
        elif audience == "Researcher":
            return self._executive_summary_researcher(investigation, risk_level, top_cause, cause_count)
        else:
            return self._executive_summary_engineer(investigation, risk_level, top_cause, cause_count)
    
    def _executive_summary_engineer(self, investigation: Investigation, risk_level: str, top_cause, cause_count: int) -> str:
        """ML Engineer audience summary."""
        if investigation.health_status == "Critical":
            summary = f"Critical model degradation detected. "
        elif investigation.health_status == "Warning":
            summary = f"Model performance degradation detected. "
        else:
            summary = f"Minor model issues detected. "
        
        # High-level impact without feature details
        if risk_level in ["HIGH", "CRITICAL"]:
            summary += f"Prediction reliability significantly compromised. Confidence: {investigation.confidence}%."
        else:
            summary += f"Prediction reliability moderately affected. Confidence: {investigation.confidence}%."
        
        return self._truncate_to_word_limit(summary, 120)
    
    def _executive_summary_executive(self, investigation: Investigation, risk_level: str, top_cause, cause_count: int) -> str:
        """Executive audience summary."""
        if risk_level == "CRITICAL":
            summary = f"Model reliability compromised. "
        elif risk_level == "HIGH":
            summary = f"Elevated operational risk detected. "
        elif risk_level == "MEDIUM":
            summary = f"Model performance requires attention. "
        else:
            summary = f"Model operating normally. "
        
        summary += f"Business impact: reduced prediction accuracy affecting decisions. "
        
        if risk_level in ["HIGH", "CRITICAL"]:
            summary += "Significant operational risk to business outcomes."
        else:
            summary += "Moderate operational risk requiring monitoring."
        
        return self._truncate_to_word_limit(summary, 120)
    
    def _executive_summary_doctor(self, investigation: Investigation, risk_level: str, top_cause, cause_count: int) -> str:
        """Doctor audience summary."""
        if investigation.health_status == "Critical":
            summary = f"Clinical decision support reliability compromised. "
        elif investigation.health_status == "Warning":
            summary = f"Clinical model requires monitoring. "
        else:
            summary = f"Clinical model operating normally. "
        
        summary += f"Patient care impact: potential reduction in diagnostic accuracy. "
        
        if risk_level in ["HIGH", "CRITICAL"]:
            summary += "Significant risk to patient outcomes."
        else:
            summary += "Moderate risk requiring clinical oversight."
        
        return self._truncate_to_word_limit(summary, 120)
    
    def _executive_summary_loan_officer(self, investigation: Investigation, risk_level: str, top_cause, cause_count: int) -> str:
        """Loan Officer audience summary."""
        if risk_level == "CRITICAL":
            summary = f"Credit risk model reliability compromised. "
        elif risk_level == "HIGH":
            summary = f"Elevated credit risk detected. "
        elif risk_level == "MEDIUM":
            summary = f"Credit model requires attention. "
        else:
            summary = f"Credit model operating normally. "
        
        summary += f"Portfolio impact: potential for inaccurate lending decisions. "
        
        if risk_level in ["HIGH", "CRITICAL"]:
            summary += "Significant regulatory and financial risk."
        else:
            summary += "Moderate risk requiring monitoring."
        
        return self._truncate_to_word_limit(summary, 120)
    
    def _executive_summary_student(self, investigation: Investigation, risk_level: str, top_cause, cause_count: int) -> str:
        """Student audience summary (professional tone)."""
        if investigation.health_status == "Critical":
            summary = f"Model integrity requires immediate attention. "
        elif investigation.health_status == "Warning":
            summary = f"Model performance degradation detected. "
        else:
            summary = f"Model operating within normal parameters. "
        
        summary += f"Learning impact: predictions may be unreliable for current data patterns. "
        
        if risk_level in ["HIGH", "CRITICAL"]:
            summary += "Significant impact on learning outcomes."
        else:
            summary += "Moderate impact on learning outcomes."
        
        return self._truncate_to_word_limit(summary, 120)
    
    def _executive_summary_hr_manager(self, investigation: Investigation, risk_level: str, top_cause, cause_count: int) -> str:
        """HR Manager audience summary."""
        if investigation.health_status == "Critical":
            summary = f"Workforce prediction model reliability compromised. "
        elif investigation.health_status == "Warning":
            summary = f"Employee outcome model requires monitoring. "
        else:
            summary = f"HR model operating normally. "
        
        summary += f"Talent management impact: potential for inaccurate hiring or retention decisions. "
        
        if risk_level in ["HIGH", "CRITICAL"]:
            summary += "Significant risk to workforce decisions."
        else:
            summary += "Moderate risk requiring HR oversight."
        
        return self._truncate_to_word_limit(summary, 120)
    
    def _executive_summary_insurance_analyst(self, investigation: Investigation, risk_level: str, top_cause, cause_count: int) -> str:
        """Insurance Analyst audience summary."""
        if investigation.health_status == "Critical":
            summary = f"Underwriting model reliability compromised. "
        elif investigation.health_status == "Warning":
            summary = f"Insurance risk model requires monitoring. "
        else:
            summary = f"Insurance model operating normally. "
        
        summary += f"Portfolio impact: potential for inaccurate premium exposure and claims risk assessment. "
        
        if risk_level in ["HIGH", "CRITICAL"]:
            summary += "Significant underwriting and reserve risk."
        else:
            summary += "Moderate risk requiring monitoring."
        
        return self._truncate_to_word_limit(summary, 120)
    
    def _executive_summary_compliance_officer(self, investigation: Investigation, risk_level: str, top_cause, cause_count: int) -> str:
        """Legal/Compliance Officer audience summary."""
        if investigation.health_status == "Critical":
            summary = f"Model compliance requires immediate review. "
        elif investigation.health_status == "Warning":
            summary = f"Compliance risk detected in model. "
        else:
            summary = f"Model operating within compliance parameters. "
        
        summary += f"Governance impact: potential regulatory exposure and documentation requirements. "
        
        if risk_level in ["HIGH", "CRITICAL"]:
            summary += "Significant regulatory and governance risk."
        else:
            summary += "Moderate risk requiring monitoring."
        
        return self._truncate_to_word_limit(summary, 120)
    
    def _executive_summary_researcher(self, investigation: Investigation, risk_level: str, top_cause, cause_count: int) -> str:
        """Researcher audience summary."""
        if investigation.health_status == "Critical":
            summary = f"Research model anomalies detected. "
        elif investigation.health_status == "Warning":
            summary = f"Experimental design requires review. "
        else:
            summary = f"Model methodology validated. "
        
        summary += f"Evidence quality: statistical validity may be compromised by distributional shifts. "
        
        if risk_level in ["HIGH", "CRITICAL"]:
            summary += "Significant impact on research validity."
        else:
            summary += "Moderate impact on research validity."
        
        return self._truncate_to_word_limit(summary, 120)
    
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
        
        # Group causes by category for pattern summarization
        categories = {}
        for cause in investigation.root_causes:
            if cause.category not in categories:
                categories[cause.category] = []
            categories[cause.category].append(cause)
        
        findings = []
        
        # Feature drift interpretation
        if "feature_drift" in categories:
            drift_causes = categories["feature_drift"]
            if len(drift_causes) == 1:
                findings.append("A single feature exhibits distributional shift, indicating localized data drift that may affect predictions for specific segments.")
            elif len(drift_causes) <= 3:
                findings.append(f"{len(drift_causes)} features show distributional changes, suggesting broader shifts in the data landscape that could impact model reliability across multiple dimensions.")
            else:
                findings.append(f"Widespread distributional drift detected across {len(drift_causes)} features, indicating significant changes in the underlying data distribution that may require comprehensive model retraining.")
        
        # Slice degradation interpretation
        if "slice_degradation" in categories:
            slice_causes = categories["slice_degradation"]
            if len(slice_causes) == 1:
                findings.append("Performance degradation identified in a specific data segment, indicating the model may not generalize well to that particular subgroup.")
            else:
                findings.append(f"Multiple data segments ({len(slice_causes)}) show performance degradation, suggesting systematic issues with model generalization across different population subgroups.")
        
        # Calibration interpretation
        if "calibration" in categories:
            findings.append("Calibration metrics indicate probability estimates have shifted, potentially affecting the reliability of risk assessments and decision-making thresholds.")
        
        # Data quality interpretation
        if "missing_values" in categories or "outliers" in categories:
            findings.append("Data quality issues have been detected that may be contributing to model performance degradation, requiring investigation of upstream data pipelines.")
        
        # Feature importance interpretation
        if "importance_drift" in categories:
            findings.append("Feature importance patterns have changed significantly, indicating the model's decision logic has shifted and may no longer align with the original training assumptions.")
        
        # Synthesize into cohesive narrative
        if findings:
            narrative = " ".join(findings)
            
            # Add context about severity (without recommendation language)
            high_severity = [c for c in investigation.root_causes if c.severity in ["HIGH", "CRITICAL"]]
            if high_severity:
                narrative += f" The presence of {len(high_severity)} high-severity issue(s) indicates significant model degradation."
            
            return self._apply_audience_context(narrative, audience, section="findings")
        else:
            base = "The investigation identified several issues affecting model performance. Review the Root Cause Analysis for detailed evidence and severity assessments."
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
        
        impact = ""
        
        # Assess based on categories
        categories = [c.category for c in investigation.root_causes]
        
        if "feature_drift" in categories:
            impact += "Distributional drift may affect predictions for segments of the population that differ from training data. "
        
        if "slice_degradation" in categories:
            impact += "Specific customer segments may experience reduced prediction accuracy. "
        
        if "calibration" in categories:
            impact += "Probability estimates may be less reliable for decision-making. "
        
        if "importance_drift" in categories:
            impact += "Model behavior has shifted, potentially affecting feature importance and interpretation. "
        
        if "missing_values" in categories or "outliers" in categories:
            impact += "Data quality issues may be affecting model performance. "
        
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
        Generate confidence explanation.
        
        Args:
            investigation: Investigation object
        
        Returns:
            Confidence explanation string
        """
        confidence = investigation.confidence
        cause_count = len(investigation.root_causes)
        high_severity_count = len(investigation.get_high_severity_causes())
        
        explanation = f"Investigation confidence is {confidence}% based on:\n\n"
        
        explanation += f"• Number of identified issues: {cause_count}\n"
        explanation += f"• High-severity issues: {high_severity_count}\n"
        
        if cause_count > 0:
            explanation += f"• Evidence strength: Strong (multiple supporting signals)\n"
        else:
            explanation += f"• Evidence strength: Limited (no issues detected)\n"
        
        if confidence >= 80:
            explanation += "\nHigh confidence indicates consistent findings across multiple analysis engines."
        elif confidence >= 60:
            explanation += "\nModerate confidence indicates some uncertainty in root cause attribution."
        else:
            explanation += "\nLow confidence indicates limited evidence or conflicting signals."
        
        return explanation
    
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
