"""
Audience Translation Layer for FeynML Phase 3.

Converts investigation findings into audience-specific language while keeping:
- evidence text unchanged
- recommendations unchanged
- downstream consequence claims evidence-safe
"""

from typing import Dict, List, Any, Optional


class AudienceTranslator:
    """
    Translates ML investigation findings into audience-specific reports.
    """

    AUDIENCES = [
        "ML Engineer",
        "Executive",
        "Doctor",
        "Loan Officer",
        "Student",
        "HR Manager",
        "Insurance Analyst",
        "Legal / Compliance Officer",
        "Researcher",
    ]

    AUDIENCE_PROFILES: Dict[str, Dict[str, Any]] = {
        "ML Engineer": {
            "title": "Technical Model Assessment",
            "status_label": "System status",
            "status_map": {
                "Healthy": "stable in monitoring",
                "Warning": "showing review-worthy signals",
                "Degraded": "degraded in monitoring",
                "Critical": "requires immediate technical review",
                "Unknown": "status unclear",
            },
            "issues_label": "Observed technical signals",
            "severity_map": {
                "CRITICAL": "Critical technical signal",
                "HIGH": "High-priority technical signal",
                "MEDIUM": "Moderate technical signal",
                "LOW": "Low-priority technical signal",
            },
            "summary_template": (
                "Monitoring flags the model as {status_text}. "
                "{issue_sentence} Review the evidence below before making pipeline changes."
            ),
            "issue_sentence_map": {
                0: "No technical review signals were observed.",
                1: "One technical review signal was observed.",
                "many": "{count} technical review signals were observed.",
            },
            "impact_assessment": (
                "This translation highlights observed monitoring evidence for technical diagnosis. "
                "It indicates where model behavior should be reviewed, but does not by itself prove downstream production outcomes."
            ),
            "confidence_explanation_template": (
                "The {confidence}% confidence score reflects how strongly the observed monitoring signals support the current diagnosis. "
                "It is confidence in the investigation, not a guarantee of downstream system impact."
            ),
            "technical_notes": (
                "Use the evidence lines as-is for debugging. Compare them with drift, calibration, and data-quality reports before changing the model or features."
            ),
            "note": "Treat the evidence as diagnostic input for technical review.",
            "include_score": True,
            "max_causes": 5,
            "max_evidence": 2,
            "show_risk_level": True,
            "risk_label": "Risk level",
        },
        "Executive": {
            "title": "Model Oversight Summary",
            "status_label": "Oversight status",
            "status_map": {
                "Healthy": "operating within expected bounds",
                "Warning": "showing signals that warrant review",
                "Degraded": "operating below expected bounds",
                "Critical": "requiring urgent review",
                "Unknown": "not yet clear",
            },
            "issues_label": "Evidence-backed review items",
            "severity_map": {
                "CRITICAL": "Urgent review item",
                "HIGH": "Priority review item",
                "MEDIUM": "Review item",
                "LOW": "Monitor item",
            },
            "summary_template": (
                "Monitoring shows the model is {status_text}. "
                "{issue_sentence} The wording below stays close to the observed evidence and avoids unsupported business claims."
            ),
            "issue_sentence_map": {
                0: "No monitoring signals require review.",
                1: "One monitoring signal requires review.",
                "many": "{count_word} monitoring signals require review.",
            },
            "impact_assessment": (
                "This summary identifies where model review may be needed for governance and operating decisions. "
                "It does not by itself establish revenue, customer, or other downstream business outcomes."
            ),
            "confidence_explanation_template": (
                "The {confidence}% confidence score indicates how consistently the available evidence supports this assessment. "
                "It is not a claim about business impact certainty."
            ),
            "technical_notes": "Additional technical detail can be reviewed separately if needed.",
            "note": "Use this summary for oversight and prioritization, not as proof of business loss.",
            "include_score": False,
            "max_causes": 3,
            "max_evidence": 1,
            "show_risk_level": True,
            "risk_label": "Risk level",
        },
        "Doctor": {
            "title": "Clinical Decision Support Review",
            "status_label": "Support model status",
            "status_map": {
                "Healthy": "reliable in current monitoring",
                "Warning": "showing signals that need review",
                "Degraded": "less reliable in current monitoring",
                "Critical": "requiring immediate review",
                "Unknown": "unclear from current information",
            },
            "issues_label": "Observed model signals",
            "severity_map": {
                "CRITICAL": "Urgent review signal",
                "HIGH": "Important review signal",
                "MEDIUM": "Moderate review signal",
                "LOW": "Low-level review signal",
            },
            "summary_template": (
                "Monitoring indicates the clinical decision-support model is {status_text}. "
                "{issue_sentence} Keep using clinical judgment while the evidence is reviewed."
            ),
            "issue_sentence_map": {
                0: "No clinical-model review signals were observed.",
                1: "One clinical-model review signal was observed.",
                "many": "{count_word} clinical-model review signals were observed.",
            },
            "impact_assessment": (
                "These findings describe monitored model behavior relevant to clinical review. "
                "They do not by themselves establish patient outcomes or care consequences."
            ),
            "confidence_explanation_template": (
                "The {confidence}% confidence score reflects how strongly the observed evidence supports the current model assessment. "
                "It is confidence in the investigation, not a clinical guarantee."
            ),
            "technical_notes": "Use the evidence lines to coordinate follow-up with the model monitoring or analytics team.",
            "note": "Clinical judgment should remain primary while these signals are investigated.",
            "include_score": False,
            "max_causes": 3,
            "max_evidence": 2,
            "show_risk_level": True,
            "risk_label": "Risk level",
        },
        "Loan Officer": {
            "title": "Credit Model Review",
            "status_label": "Model review status",
            "status_map": {
                "Healthy": "stable in monitoring",
                "Warning": "showing signals that need review",
                "Degraded": "showing reduced reliability in monitoring",
                "Critical": "requiring urgent review",
                "Unknown": "not yet clear",
            },
            "issues_label": "Observed review signals",
            "severity_map": {
                "CRITICAL": "Urgent lending review signal",
                "HIGH": "High-priority lending review signal",
                "MEDIUM": "Moderate lending review signal",
                "LOW": "Low-level lending review signal",
            },
            "summary_template": (
                "Monitoring indicates the credit model is {status_text}. "
                "{issue_sentence} Review the evidence before relying on the model in lending workflows."
            ),
            "issue_sentence_map": {
                0: "No lending-model review signals were observed.",
                1: "One lending-model review signal was observed.",
                "many": "{count} lending-model review signals were observed.",
            },
            "impact_assessment": (
                "This translation summarizes monitored model behavior relevant to lending review. "
                "It does not by itself prove borrower, portfolio, or default outcomes."
            ),
            "confidence_explanation_template": (
                "The {confidence}% confidence score shows how well the observed evidence supports this model assessment. "
                "It is not a guarantee about loan outcomes."
            ),
            "technical_notes": "Pair these findings with policy and governance review where needed.",
            "note": "Use the evidence for model review, alongside existing lending controls.",
            "include_score": False,
            "max_causes": 3,
            "max_evidence": 1,
            "show_risk_level": True,
            "risk_label": "Risk level",
        },
        "Student": {
            "title": "Model Health Explanation",
            "status_label": "Current status",
            "status_map": {
                "Healthy": "looking stable",
                "Warning": "showing signs that should be reviewed",
                "Degraded": "showing weaker performance signals",
                "Critical": "needing urgent review",
                "Unknown": "not yet clear",
            },
            "issues_label": "What the evidence shows",
            "severity_map": {
                "CRITICAL": "Very important signal",
                "HIGH": "Important signal",
                "MEDIUM": "Moderate signal",
                "LOW": "Minor signal",
            },
            "summary_template": (
                "The model is {status_text} according to monitoring. "
                "{issue_sentence} The wording below stays simple, but the evidence itself is unchanged."
            ),
            "issue_sentence_map": {
                0: "No important model issues were found.",
                1: "One important model issue was found.",
                "many": "{count_word} important model issues were found.",
            },
            "impact_assessment": (
                "This explanation points to where the model needs review based on observed evidence. "
                "It does not claim any specific real-world outcome by itself."
            ),
            "confidence_explanation_template": (
                "The {confidence}% confidence score tells you how strongly the available evidence supports this explanation. "
                "It is confidence in the diagnosis, not proof of what happens next."
            ),
            "technical_notes": "Read the evidence lines first, then compare them with the recommended actions.",
            "note": "Use this version to understand the investigation without changing the underlying facts.",
            "include_score": False,
            "max_causes": 3,
            "max_evidence": 2,
            "show_risk_level": True,
            "risk_label": "Risk level",
        },
        "HR Manager": {
            "title": "HR Model Review",
            "status_label": "Decision-support status",
            "status_map": {
                "Healthy": "reliable in monitoring",
                "Warning": "showing signals that need review",
                "Degraded": "less reliable in monitoring",
                "Critical": "requiring immediate review",
                "Unknown": "unclear from current information",
            },
            "issues_label": "Observed review signals",
            "severity_map": {
                "CRITICAL": "Urgent HR review signal",
                "HIGH": "Important HR review signal",
                "MEDIUM": "Moderate HR review signal",
                "LOW": "Low-level HR review signal",
            },
            "summary_template": (
                "Monitoring indicates the HR decision-support model is {status_text}. "
                "{issue_sentence} Use the evidence for review, alongside standard HR judgment."
            ),
            "issue_sentence_map": {
                0: "No HR model review signals were observed.",
                1: "One HR model review signal was observed.",
                "many": "{count} HR model review signals were observed.",
            },
            "impact_assessment": (
                "These findings summarize monitored model behavior relevant to HR review. "
                "They do not by themselves establish hiring, performance, or retention outcomes."
            ),
            "confidence_explanation_template": (
                "The {confidence}% confidence score reflects how strongly the observed evidence supports this assessment. "
                "It is not a guarantee about workforce outcomes."
            ),
            "technical_notes": "Escalate technical follow-up to the analytics team if evidence repeats across monitoring cycles.",
            "note": "HR judgment should remain primary while model signals are investigated.",
            "include_score": False,
            "max_causes": 3,
            "max_evidence": 1,
            "show_risk_level": True,
            "risk_label": "Risk level",
        },
        "Insurance Analyst": {
            "title": "Insurance Model Review",
            "status_label": "Model review status",
            "status_map": {
                "Healthy": "stable in monitoring",
                "Warning": "showing signals that need review",
                "Degraded": "showing reduced reliability in monitoring",
                "Critical": "requiring urgent review",
                "Unknown": "not yet clear",
            },
            "issues_label": "Observed review signals",
            "severity_map": {
                "CRITICAL": "Urgent underwriting review signal",
                "HIGH": "High-priority underwriting review signal",
                "MEDIUM": "Moderate underwriting review signal",
                "LOW": "Low-level underwriting review signal",
            },
            "summary_template": (
                "Monitoring indicates the insurance model is {status_text}. "
                "{issue_sentence} Review the evidence before making reliance decisions in underwriting workflows."
            ),
            "issue_sentence_map": {
                0: "No underwriting review signals were observed.",
                1: "One underwriting review signal was observed.",
                "many": "{count} underwriting review signals were observed.",
            },
            "impact_assessment": (
                "This translation summarizes monitored model behavior relevant to underwriting review. "
                "It does not by itself prove claims, pricing, or portfolio outcomes."
            ),
            "confidence_explanation_template": (
                "The {confidence}% confidence score reflects how consistently the available evidence supports this assessment. "
                "It is not a guarantee about underwriting outcomes."
            ),
            "technical_notes": "Use the evidence lines together with governance materials if a formal review is needed.",
            "note": "Use this report for model review and documentation, not as proof of downstream insurance outcomes.",
            "include_score": False,
            "max_causes": 3,
            "max_evidence": 1,
            "show_risk_level": True,
            "risk_label": "Risk level",
        },
        "Legal / Compliance Officer": {
            "title": "Model Governance Review",
            "status_label": "Governance status",
            "status_map": {
                "Healthy": "within expected monitoring bounds",
                "Warning": "showing review-worthy governance signals",
                "Degraded": "showing elevated governance concern",
                "Critical": "requiring immediate governance review",
                "Unknown": "not yet clear",
            },
            "issues_label": "Documented review signals",
            "severity_map": {
                "CRITICAL": "Urgent governance review signal",
                "HIGH": "High-priority governance review signal",
                "MEDIUM": "Moderate governance review signal",
                "LOW": "Low-level governance review signal",
            },
            "summary_template": (
                "Monitoring indicates the model is {status_text} from a governance perspective. "
                "{issue_sentence} The wording below stays close to the evidence and avoids unsupported legal conclusions."
            ),
            "issue_sentence_map": {
                0: "No governance review signals were observed.",
                1: "One governance review signal was observed.",
                "many": "{count} governance review signals were observed.",
            },
            "impact_assessment": (
                "This translation identifies monitored model signals relevant to governance review and documentation. "
                "It does not by itself establish legal violation, enforcement, or penalty outcomes."
            ),
            "confidence_explanation_template": (
                "The {confidence}% confidence score reflects the strength and consistency of the observed evidence for this review. "
                "It is not a legal conclusion."
            ),
            "technical_notes": "Retain the evidence lines verbatim if the review needs to be documented or escalated.",
            "note": "Document the observed signals and follow-up steps separately from any legal conclusion.",
            "include_score": False,
            "max_causes": 3,
            "max_evidence": 2,
            "show_risk_level": True,
            "risk_label": "Risk level",
        },
        "Researcher": {
            "title": "Research Model Assessment",
            "status_label": "Study status",
            "status_map": {
                "Healthy": "stable in current monitoring",
                "Warning": "showing signals worth investigation",
                "Degraded": "showing instability in current monitoring",
                "Critical": "requiring immediate investigation",
                "Unknown": "unclear from current evidence",
            },
            "issues_label": "Observed investigation signals",
            "severity_map": {
                "CRITICAL": "Urgent investigation signal",
                "HIGH": "High-priority investigation signal",
                "MEDIUM": "Moderate investigation signal",
                "LOW": "Low-level investigation signal",
            },
            "summary_template": (
                "Monitoring indicates the research model is {status_text}. "
                "{issue_sentence} Use the unchanged evidence lines when discussing validity or reproducibility."
            ),
            "issue_sentence_map": {
                0: "No investigation signals were detected.",
                1: "One investigation signal was detected.",
                "many": "{count_word} investigation signals were detected.",
            },
            "impact_assessment": (
                "These findings summarize monitored model behavior relevant to study review. "
                "They do not by themselves prove validity, reproducibility, or conclusion-level outcomes."
            ),
            "confidence_explanation_template": (
                "The {confidence}% confidence score indicates how strongly the observed evidence supports this assessment. "
                "It is confidence in the investigation, not proof of research conclusions."
            ),
            "technical_notes": "Preserve the evidence text exactly if you cite it in research notes or follow-up analysis.",
            "note": "Use this report to frame investigation and replication work, not to overstate conclusion-level effects.",
            "include_score": True,
            "max_causes": 5,
            "max_evidence": 2,
            "show_risk_level": True,
            "risk_label": "Risk level",
        },
        "General": {
            "title": "Model Review",
            "status_label": "Status",
            "status_map": {
                "Healthy": "stable in monitoring",
                "Warning": "showing signals that need review",
                "Degraded": "showing reduced reliability in monitoring",
                "Critical": "requiring urgent review",
                "Unknown": "not yet clear",
            },
            "issues_label": "Observed signals",
            "severity_map": {
                "CRITICAL": "Urgent review signal",
                "HIGH": "High-priority review signal",
                "MEDIUM": "Moderate review signal",
                "LOW": "Low-level review signal",
            },
            "summary_template": (
                "Monitoring indicates the model is {status_text}. {issue_sentence}"
            ),
            "issue_sentence_map": {
                0: "No review signals were observed.",
                1: "One review signal was observed.",
                "many": "{count} review signals were observed.",
            },
            "impact_assessment": (
                "This summary reflects observed evidence and indicates where model review may be needed. "
                "It does not by itself prove downstream outcomes."
            ),
            "confidence_explanation_template": (
                "The {confidence}% confidence score reflects how strongly the observed evidence supports this assessment."
            ),
            "technical_notes": "Consult the detailed monitoring report for more context.",
            "note": "",
            "include_score": False,
            "max_causes": 5,
            "max_evidence": 2,
            "show_risk_level": False,
        },
    }

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def translate(
        self,
        root_cause: Dict[str, Any],
        ai_investigator: Dict[str, Any],
        ai_investigator_by_audience: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Translate investigation findings for all supported audiences.
        """
        audience_reports = {}

        for audience in self.AUDIENCES:
            if self.verbose:
                print(f"Translating for audience: {audience}")

            ai_data = ai_investigator
            if ai_investigator_by_audience and isinstance(ai_investigator_by_audience, dict):
                ai_data = ai_investigator_by_audience.get(audience, ai_investigator)

            audience_reports[audience] = self._translate_for_audience(
                audience, root_cause, ai_data
            )

        return audience_reports

    def _translate_for_audience(
        self, audience: str, root_cause: Dict[str, Any], ai_investigator: Dict[str, Any]
    ) -> Dict[str, Any]:
        if audience == "ML Engineer":
            return self._translate_for_engineer(root_cause, ai_investigator)
        elif audience == "Executive":
            return self._translate_for_executive(root_cause, ai_investigator)
        elif audience == "Doctor":
            return self._translate_for_doctor(root_cause, ai_investigator)
        elif audience == "Loan Officer":
            return self._translate_for_loan_officer(root_cause, ai_investigator)
        elif audience == "Student":
            return self._translate_for_student(root_cause, ai_investigator)
        elif audience == "HR Manager":
            return self._translate_for_hr_manager(root_cause, ai_investigator)
        elif audience == "Insurance Analyst":
            return self._translate_for_insurance_analyst(root_cause, ai_investigator)
        elif audience == "Legal / Compliance Officer":
            return self._translate_for_compliance_officer(root_cause, ai_investigator)
        elif audience == "Researcher":
            return self._translate_for_researcher(root_cause, ai_investigator)
        else:
            return self._translate_generic(root_cause, ai_investigator)

    @staticmethod
    def _get_recommendations(root_cause: Dict[str, Any], ai_investigator: Dict[str, Any]) -> List[str]:
        recommendations = root_cause.get("recommended_actions", [])
        if recommendations:
            return list(recommendations)
        return list(ai_investigator.get("recommended_actions", []) or [])

    @staticmethod
    def _normalize_root_causes(root_cause: Dict[str, Any]) -> List[Dict[str, Any]]:
        return list(root_cause.get("root_causes", []) or [])

    @staticmethod
    def _count_word(count: int) -> str:
        small_counts = {
            0: "Zero",
            1: "One",
            2: "Two",
            3: "Three",
            4: "Four",
            5: "Five",
        }
        return small_counts.get(count, str(count))

    def _issue_sentence(self, audience: str, root_causes: List[Dict[str, Any]]) -> str:
        count = len(root_causes)
        profile = self.AUDIENCE_PROFILES.get(audience, self.AUDIENCE_PROFILES["General"])
        issue_map = profile.get("issue_sentence_map", {})
        if count in issue_map:
            return issue_map[count].format(count=count, count_word=self._count_word(count))
        template = issue_map.get("many", "{count} review signals were observed.")
        return template.format(count=count, count_word=self._count_word(count))

    def _build_executive_summary(
        self,
        audience: str,
        health_status: str,
        confidence: int,
        root_causes: List[Dict[str, Any]],
    ) -> str:
        profile = self.AUDIENCE_PROFILES.get(audience, self.AUDIENCE_PROFILES["General"])
        status_text = profile["status_map"].get(health_status, profile["status_map"]["Unknown"])
        return profile["summary_template"].format(
            status_text=status_text,
            confidence=confidence,
            issue_sentence=self._issue_sentence(audience, root_causes),
        )

    def _build_findings(
        self,
        audience: str,
        root_cause: Dict[str, Any],
        ai_investigator: Dict[str, Any],
    ) -> str:
        profile = self.AUDIENCE_PROFILES.get(audience, self.AUDIENCE_PROFILES["General"])
        health_status = root_cause.get("health_status", "Unknown")
        confidence = root_cause.get("confidence", 0)
        root_causes = self._normalize_root_causes(root_cause)
        risk_level = ai_investigator.get("risk_level", "UNKNOWN")

        lines = [
            f"{profile['title']}:",
            f"{profile['status_label']}: {profile['status_map'].get(health_status, profile['status_map']['Unknown'])}",
            f"Assessment confidence: {confidence}%",
        ]

        if profile.get("show_risk_level"):
            lines.append(f"{profile.get('risk_label', 'Risk level')}: {risk_level}")

        if root_causes:
            lines.extend(["", f"{profile['issues_label']}:"])
            # Display ALL root causes dynamically, not limited by max_causes
            for i, cause in enumerate(root_causes, 1):
                severity = str(cause.get("severity", "LOW")).upper()
                severity_text = profile["severity_map"].get(severity, severity)
                line = f"{i}. {cause.get('cause', 'Unknown')}: {severity_text}"
                if profile.get("include_score"):
                    line += f" (score={cause.get('score', 0):.3f})"
                lines.append(line)

                for evidence in (cause.get("evidence", []) or []):
                    lines.append(f"   Evidence: {evidence}")
        else:
            lines.extend(["", "No root causes were listed in the current summary."])

        note = profile.get("note", "").strip()
        if note:
            lines.extend(["", f"Note: {note}"])

        return "\n".join(lines)

    def _build_audience_report(
        self,
        audience: str,
        root_cause: Dict[str, Any],
        ai_investigator: Dict[str, Any],
    ) -> Dict[str, Any]:
        profile = self.AUDIENCE_PROFILES.get(audience, self.AUDIENCE_PROFILES["General"])
        health_status = root_cause.get("health_status", "Unknown")
        confidence = root_cause.get("confidence", 0)
        root_causes = self._normalize_root_causes(root_cause)

        return {
            "audience": audience if audience in self.AUDIENCE_PROFILES else "General",
            "executive_summary": self._build_executive_summary(
                audience, health_status, confidence, root_causes
            ),
            "findings": self._build_findings(audience, root_cause, ai_investigator),
            "impact_assessment": profile["impact_assessment"],
            "confidence_explanation": profile["confidence_explanation_template"].format(
                confidence=confidence
            ),
            "recommendations": self._get_recommendations(root_cause, ai_investigator),
            "technical_notes": profile["technical_notes"],
        }

    def _translate_for_hr_manager(
        self, root_cause: Dict[str, Any], ai_investigator: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._build_audience_report("HR Manager", root_cause, ai_investigator)

    def _translate_for_insurance_analyst(
        self, root_cause: Dict[str, Any], ai_investigator: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._build_audience_report("Insurance Analyst", root_cause, ai_investigator)

    def _translate_for_compliance_officer(
        self, root_cause: Dict[str, Any], ai_investigator: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._build_audience_report("Legal / Compliance Officer", root_cause, ai_investigator)

    def _translate_for_researcher(
        self, root_cause: Dict[str, Any], ai_investigator: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._build_audience_report("Researcher", root_cause, ai_investigator)

    def _translate_for_engineer(
        self, root_cause: Dict[str, Any], ai_investigator: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._build_audience_report("ML Engineer", root_cause, ai_investigator)

    def _translate_for_executive(
        self, root_cause: Dict[str, Any], ai_investigator: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._build_audience_report("Executive", root_cause, ai_investigator)

    def _translate_for_doctor(
        self, root_cause: Dict[str, Any], ai_investigator: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._build_audience_report("Doctor", root_cause, ai_investigator)

    def _translate_for_loan_officer(
        self, root_cause: Dict[str, Any], ai_investigator: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._build_audience_report("Loan Officer", root_cause, ai_investigator)

    def _translate_for_student(
        self, root_cause: Dict[str, Any], ai_investigator: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._build_audience_report("Student", root_cause, ai_investigator)

    def _translate_generic(
        self, root_cause: Dict[str, Any], ai_investigator: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._build_audience_report("General", root_cause, ai_investigator)
