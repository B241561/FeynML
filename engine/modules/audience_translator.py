"""
Audience Translation Layer for FeynML Phase 3

Converts investigation findings into audience-specific language.
"""

from typing import Dict, List, Any, Optional


class AudienceTranslator:
    """
    Translates ML investigation findings into audience-specific reports.
    
    Supported audiences:
    - ML Engineer: Technical depth, metrics, code-level details
    - Executive: Business impact, ROI, strategic recommendations
    - Doctor: Clinical impact, patient outcomes, medical terminology
    - Loan Officer: Risk assessment, financial impact, compliance
    - Student: Educational focus, learning objectives, simplified concepts
    """
    
    AUDIENCES = ["ML Engineer", "Executive", "Doctor", "Loan Officer", "Student", "HR Manager", "Insurance Analyst", "Legal / Compliance Officer", "Researcher"]
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    @staticmethod
    def _resolve_executive_summary(ai_investigator: Dict[str, Any]) -> str:
        """Use evidence-grounded summary from AI Investigator when available."""
        return (ai_investigator.get("executive_summary") or "").strip()
    
    def translate(
        self,
        root_cause: Dict[str, Any],
        ai_investigator: Dict[str, Any],
        ai_investigator_by_audience: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Translate investigation findings for all supported audiences.
        
        Args:
            root_cause: Root cause analysis output
            ai_investigator: AI investigator output
            
        Returns:
            Dictionary with audience-specific reports
        """
        audience_reports = {}
        
        for audience in self.AUDIENCES:
            if self.verbose:
                print(f"Translating for audience: {audience}")
            
            # If available, use audience-specific AI Investigator output so that
            # any sections that reference `ai_investigator.*` aren't stuck in a
            # single (often technical) style.
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
        """
        Translate findings for a specific audience.
        """
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
            
    def _translate_for_hr_manager(
        self, root_cause: Dict[str, Any], ai_investigator: Dict[str, Any]
    ) -> Dict[str, Any]:
        """HR-focused translation for employee attrition/performance models."""
        health_status = root_cause.get("health_status", "Unknown")
        confidence = root_cause.get("confidence", 0)
        root_causes = root_cause.get("root_causes", [])
        risk_level = ai_investigator.get("risk_level", "UNKNOWN")
        
        # HR terminology
        hr_status = {
            "Healthy": "Reliable",
            "Warning": "Needs Monitoring",
            "Degraded": "Unreliable",
            "Critical": "High Risk",
            "Unknown": "Uncertain"
        }
        
        reliability = hr_status.get(health_status, "Uncertain")
        
        findings = f"HR Model Assessment:\n"
        findings += f"Model reliability for HR decisions: {reliability}\n"
        findings += f"Assessment confidence: {confidence}%\n\n"
        
        if root_causes:
            findings += "Potential HR Impact:\n"
            for i, cause in enumerate(root_causes[:3], 1):
                severity = cause.get('severity', 'LOW')
                hr_severity = {
                    'CRITICAL': 'High - May affect employee decisions',
                    'HIGH': 'Moderate - Review recommended',
                    'MEDIUM': 'Low - Monitor',
                    'LOW': 'Minimal'
                }.get(severity, severity)
                
                findings += f"{i}. {cause.get('cause', 'Unknown')}: {hr_severity}\n"
        
        findings += f"\nOverall Risk Level: {risk_level}\n"
        findings += "\nNote: Always use HR judgment alongside model predictions."
        
        executive_summary = self._resolve_executive_summary(ai_investigator)
        if not executive_summary:
            executive_summary = (
                f"HR decision support model reliable. No significant issues detected. "
                f"Continue normal operations. Confidence: {confidence}%."
            )
        
        # HR recommendations
        hr_recommendations = []
        for rec in (recommendations := root_cause.get("recommended_actions", []) or ai_investigator.get("recommended_actions", [])):
            hr_recommendations.append(f"HR Action: {rec}")
        
        # Impact assessment (HR)
        impact_assessment = "HR implications include potential impact on hiring decisions, performance evaluations, and employee retention strategies."
        
        # Confidence explanation (HR)
        confidence_explanation = f"Confidence score of {confidence}% reflects the strength of evidence from model performance monitoring and HR validation."
        
        # Technical notes (HR)
        technical_notes = "Model performance and validation details are available in the full HR analytics report."
        
        return {
            "audience": "HR Manager",
            "executive_summary": executive_summary,
            "findings": findings,
            "impact_assessment": impact_assessment,
            "confidence_explanation": confidence_explanation,
            "recommendations": hr_recommendations,
            "technical_notes": technical_notes
        }
    
    def _translate_for_insurance_analyst(
        self, root_cause: Dict[str, Any], ai_investigator: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Insurance-focused translation for risk scoring models."""
        health_status = root_cause.get("health_status", "Unknown")
        confidence = root_cause.get("confidence", 0)
        root_causes = root_cause.get("root_causes", [])
        risk_level = ai_investigator.get("risk_level", "UNKNOWN")
        
        # Insurance terminology
        insurance_status = {
            "Healthy": "Low Risk",
            "Warning": "Monitor Closely",
            "Degraded": "Moderate Risk",
            "Critical": "High Risk",
            "Unknown": "Uncertain Risk"
        }
        
        assessment_risk = insurance_status.get(health_status, "Uncertain")
        
        findings = f"Insurance Risk Model Assessment:\n"
        findings += f"Model risk classification: {assessment_risk}\n"
        findings += f"Assessment confidence: {confidence}%\n\n"
        
        if root_causes:
            findings += "Risk Factors Identified:\n"
            for i, cause in enumerate(root_causes[:3], 1):
                severity = cause.get('severity', 'LOW')
                insurance_severity = {
                    'CRITICAL': 'High - May affect underwriting accuracy',
                    'HIGH': 'Moderate - Monitor closely',
                    'MEDIUM': 'Low - Track trends',
                    'LOW': 'Minimal'
                }.get(severity, severity)
                
                findings += f"{i}. {cause.get('cause', 'Unknown')}: {insurance_severity}\n"
        
        findings += f"\nOverall Risk Level: {risk_level}\n"
        findings += "\nCompliance Note: Ensure model meets insurance regulatory requirements."
        
        executive_summary = self._resolve_executive_summary(ai_investigator)
        if not executive_summary:
            executive_summary = (
                f"Insurance risk model reliable. No significant issues detected. "
                f"Continue normal operations. Risk level: {risk_level}."
            )
        
        # Insurance recommendations
        insurance_recommendations = []
        for rec in (recommendations := root_cause.get("recommended_actions", []) or ai_investigator.get("recommended_actions", [])):
            insurance_recommendations.append(f"Risk Management: {rec}")
        
        # Impact assessment (insurance)
        impact_assessment = "Portfolio impact includes potential for inaccurate risk assessments, increased claims costs, and compliance implications from unreliable model outputs."
        
        # Confidence explanation (insurance)
        confidence_explanation = f"Confidence score of {confidence}% is based on monitoring of model performance metrics, risk prediction accuracy, and compliance validation checks."
        
        # Technical notes (insurance)
        technical_notes = "Model performance and compliance documentation are available in the regulatory reporting package."
        
        return {
            "audience": "Insurance Analyst",
            "executive_summary": executive_summary,
            "findings": findings,
            "impact_assessment": impact_assessment,
            "confidence_explanation": confidence_explanation,
            "recommendations": insurance_recommendations,
            "technical_notes": technical_notes
        }
    
    def _translate_for_compliance_officer(
        self, root_cause: Dict[str, Any], ai_investigator: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compliance-focused translation for legal and fairness audits."""
        health_status = root_cause.get("health_status", "Unknown")
        confidence = root_cause.get("confidence", 0)
        root_causes = root_cause.get("root_causes", [])
        risk_level = ai_investigator.get("risk_level", "UNKNOWN")
        
        # Compliance terminology
        compliance_status = {
            "Healthy": "Compliant",
            "Warning": "Needs Review",
            "Degraded": "Potentially Non-Compliant",
            "Critical": "Non-Compliant",
            "Unknown": "Uncertain"
        }
        
        assessment_status = compliance_status.get(health_status, "Uncertain")
        
        findings = f"Model Compliance Assessment:\n"
        findings += f"Current compliance status: {assessment_status}\n"
        findings += f"Assessment confidence: {confidence}%\n\n"
        
        if root_causes:
            findings += "Potential Compliance Risks:\n"
            for i, cause in enumerate(root_causes[:3], 1):
                severity = cause.get('severity', 'LOW')
                compliance_severity = {
                    'CRITICAL': 'High - Immediate action required',
                    'HIGH': 'Moderate - Review and document',
                    'MEDIUM': 'Low - Monitor',
                    'LOW': 'Minimal'
                }.get(severity, severity)
                
                findings += f"{i}. {cause.get('cause', 'Unknown')}: {compliance_severity}\n"
        
        findings += f"\nOverall Risk Level: {risk_level}\n"
        findings += "\nNote: Document all findings and actions taken for audit trail purposes."
        
        executive_summary = self._resolve_executive_summary(ai_investigator)
        if not executive_summary:
            executive_summary = (
                f"Model compliant. No significant compliance issues detected. "
                f"Continue normal operations. Risk level: {risk_level}."
            )
        
        # Compliance recommendations
        compliance_recommendations = []
        for rec in (recommendations := root_cause.get("recommended_actions", []) or ai_investigator.get("recommended_actions", [])):
            compliance_recommendations.append(f"Compliance Action: {rec}")
        
        # Impact assessment (compliance)
        impact_assessment = "Compliance implications include potential for regulatory violations, fines, and reputational damage from unfair or unreliable model outputs."
        
        # Confidence explanation (compliance)
        confidence_explanation = f"Confidence score of {confidence}% is based on analysis of fairness metrics, bias detection, and model performance validation."
        
        # Technical notes (compliance)
        technical_notes = "Full audit trail and compliance documentation are available in the model governance package."
        
        return {
            "audience": "Legal / Compliance Officer",
            "executive_summary": executive_summary,
            "findings": findings,
            "impact_assessment": impact_assessment,
            "confidence_explanation": confidence_explanation,
            "recommendations": compliance_recommendations,
            "technical_notes": technical_notes
        }
    
    def _translate_for_researcher(
        self, root_cause: Dict[str, Any], ai_investigator: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Research-focused translation for academic/experimental models."""
        health_status = root_cause.get("health_status", "Unknown")
        confidence = root_cause.get("confidence", 0)
        root_causes = root_cause.get("root_causes", [])
        recommendations = root_cause.get("recommended_actions", []) or ai_investigator.get("recommended_actions", [])
        
        # Research terminology
        research_status = {
            "Healthy": "Stable",
            "Warning": "Anomalies Detected",
            "Degraded": "Unstable",
            "Critical": "Requires Immediate Attention",
            "Unknown": "Status Unclear"
        }
        
        model_status = research_status.get(health_status, "Unclear")
        
        findings = f"Research Model Assessment:\n"
        findings += f"Current model status: {model_status}\n"
        findings += f"Investigation confidence: {confidence}%\n\n"
        
        if root_causes:
            findings += "Identified Issues:\n"
            for i, cause in enumerate(root_causes[:5], 1):
                findings += f"{i}. {cause.get('cause', 'Unknown')} (Severity: {cause.get('severity', 'LOW')}, Score: {cause.get('score', 0):.3f})\n"
                if cause.get('evidence'):
                    findings += f"   Evidence: {', '.join(cause['evidence'][:2])}\n"
        
        # Use AI investigator findings if available
        if ai_investigator.get("investigation_findings"):
            findings += f"\nDetailed Analysis:\n{ai_investigator['investigation_findings']}\n"
        
        executive_summary = self._resolve_executive_summary(ai_investigator)
        if not executive_summary:
            executive_summary = (
                f"Research model stable. No significant issues detected. "
                f"Continue research operations. Confidence: {confidence}%."
            )
        
        # Impact assessment (research)
        impact_assessment = "Research implications include potential challenges to study validity, reproducibility, and the reliability of experimental conclusions."
        
        # Confidence explanation (research)
        confidence_explanation = f"Confidence score of {confidence}% is based on comprehensive analysis of model performance metrics, feature stability, and statistical validation."
        
        # Technical notes (research)
        technical_notes = "Detailed performance metrics, statistical analyses, and reproducibility information are available in the full research report."
        
        # Research recommendations
        research_recommendations = []
        for rec in recommendations:
            research_recommendations.append(f"Research Action: {rec}")
        
        return {
            "audience": "Researcher",
            "executive_summary": executive_summary,
            "findings": findings,
            "impact_assessment": impact_assessment,
            "confidence_explanation": confidence_explanation,
            "recommendations": research_recommendations,
            "technical_notes": technical_notes
        }
    
    def _translate_for_engineer(
        self, root_cause: Dict[str, Any], ai_investigator: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Technical translation for ML engineers."""
        health_status = root_cause.get("health_status", "Unknown")
        confidence = root_cause.get("confidence", 0)
        root_causes = root_cause.get("root_causes", [])
        recommendations = root_cause.get("recommended_actions", [])
        
        # Build technical summary
        findings = f"Model health status: {health_status}\n"
        findings += f"Confidence score: {confidence}%\n\n"
        
        if root_causes:
            findings += "Primary technical issues identified:\n"
            for i, cause in enumerate(root_causes[:5], 1):
                findings += f"{i}. {cause.get('cause', 'Unknown')} (Severity: {cause.get('severity', 'LOW')})\n"
                findings += f"   Score: {cause.get('score', 0):.3f}\n"
                if cause.get('evidence'):
                    findings += f"   Evidence: {', '.join(cause['evidence'][:2])}\n"
                findings += "\n"
        
        # Use AI investigator findings if available
        if ai_investigator.get("investigation_findings"):
            findings += f"\nDetailed Analysis:\n{ai_investigator['investigation_findings']}\n"
        
        executive_summary = self._resolve_executive_summary(ai_investigator)
        if not executive_summary:
            executive_summary = (
                f"Model operating normally. Minor issues detected. "
                f"Continue monitoring. Confidence: {confidence}%."
            )
        
        # Impact assessment (technical)
        impact_assessment = "Technical impact includes potential reduction in prediction accuracy, calibration drift, and increased risk of unreliable outputs for production workloads."
        
        # Confidence explanation (technical)
        confidence_explanation = f"Confidence score of {confidence}% is based on root cause analysis of drift metrics (PSI, KS), calibration error (ECE, Brier score), and feature importance stability."
        
        # Technical notes
        technical_notes = "Technical recommendations focus on feature drift mitigation, model recalibration, and retraining with updated datasets. Key metrics to monitor: PSI, KS, calibration ECE, feature importance variance over time."
        
        return {
            "audience": "ML Engineer",
            "executive_summary": executive_summary,
            "findings": findings,
            "impact_assessment": impact_assessment,
            "confidence_explanation": confidence_explanation,
            "recommendations": recommendations or ai_investigator.get("recommended_actions", []),
            "technical_notes": technical_notes
        }
    
    def _translate_for_executive(
        self, root_cause: Dict[str, Any], ai_investigator: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Business-focused translation for executives."""
        health_status = root_cause.get("health_status", "Unknown")
        confidence = root_cause.get("confidence", 0)
        root_causes = root_cause.get("root_causes", [])
        risk_level = ai_investigator.get("risk_level", "UNKNOWN")
        
        # Business impact language
        status_translation = {
            "Healthy": "operational",
            "Warning": "showing signs of degradation",
            "Degraded": "underperforming",
            "Critical": "at risk",
            "Unknown": "status unclear"
        }
        
        business_status = status_translation.get(health_status, "uncertain")
        
        # Build business summary
        findings = f"Business Impact Assessment:\n"
        findings += f"Current model performance is {business_status} with {confidence}% confidence in assessment.\n\n"
        
        if root_causes:
            findings += "Key Business Risks:\n"
            for i, cause in enumerate(root_causes[:3], 1):
                severity = cause.get('severity', 'LOW')
                business_severity = {
                    'CRITICAL': 'High - Immediate attention required',
                    'HIGH': 'High - Priority action needed',
                    'MEDIUM': 'Medium - Monitor closely',
                    'LOW': 'Low - Track for trends'
                }.get(severity, severity)
                
                findings += f"{i}. {cause.get('cause', 'Unknown')}: {business_severity}\n"
        
        findings += f"\nOverall Risk Level: {risk_level}\n"
        
        if ai_investigator.get("impact_assessment"):
            findings += f"\n{ai_investigator['impact_assessment']}\n"
        
        executive_summary = self._resolve_executive_summary(ai_investigator)
        if not executive_summary:
            executive_summary = (
                f"Model operating normally. No significant business impact detected. "
                f"Continue operations. Risk level: {risk_level}."
            )
        
        # Translate recommendations to business language
        business_recommendations = []
        for rec in (recommendations := root_cause.get("recommended_actions", []) or ai_investigator.get("recommended_actions", [])):
            # Simplify technical recommendations
            rec_lower = rec.lower()
            if "retrain" in rec_lower:
                business_recommendations.append("Schedule model retraining to improve accuracy")
            elif "feature" in rec_lower:
                business_recommendations.append("Review and optimize input data features")
            elif "drift" in rec_lower:
                business_recommendations.append("Monitor for data drift and update training data")
            elif "calibration" in rec_lower:
                business_recommendations.append("Improve model confidence calibration")
            else:
                business_recommendations.append(rec)
        
        # Impact assessment (business)
        impact_assessment = "Business risk includes potential revenue impact from incorrect decisions, operational friction from unreliable outputs, and increased scrutiny from stakeholders."
        
        # Confidence explanation (business)
        confidence_explanation = f"Confidence score of {confidence}% reflects our level of certainty in the identified issues and recommended actions based on comprehensive model health monitoring."
        
        # Technical notes (lightweight for exec)
        technical_notes = "Technical details are available upon request from the data science team."
        
        return {
            "audience": "Executive",
            "executive_summary": executive_summary,
            "findings": findings,
            "impact_assessment": impact_assessment,
            "confidence_explanation": confidence_explanation,
            "recommendations": business_recommendations,
            "technical_notes": technical_notes
        }
    
    def _translate_for_doctor(
        self, root_cause: Dict[str, Any], ai_investigator: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Clinical-focused translation for medical professionals."""
        health_status = root_cause.get("health_status", "Unknown")
        confidence = root_cause.get("confidence", 0)
        root_causes = root_cause.get("root_causes", [])
        
        # Clinical terminology
        clinical_status = {
            "Healthy": "reliable",
            "Warning": "needs monitoring",
            "Degraded": "variable",
            "Critical": "unreliable",
            "Unknown": "uncertain"
        }
        
        reliability = clinical_status.get(health_status, "uncertain")
        
        findings = f"Clinical Decision Support Assessment:\n"
        findings += f"Model reliability for clinical decisions: {reliability}\n"
        findings += f"Confidence in assessment: {confidence}%\n\n"
        
        if root_causes:
            findings += "Potential Clinical Impact:\n"
            for i, cause in enumerate(root_causes[:3], 1):
                severity = cause.get('severity', 'LOW')
                clinical_severity = {
                    'CRITICAL': 'High - May affect patient care',
                    'HIGH': 'Moderate - Review recommended',
                    'MEDIUM': 'Low - Monitor',
                    'LOW': 'Minimal'
                }.get(severity, severity)
                
                findings += f"{i}. {cause.get('cause', 'Unknown')}: {clinical_severity}\n"
        
        findings += "\nNote: Always use clinical judgment alongside model predictions."
        
        executive_summary = self._resolve_executive_summary(ai_investigator)
        if not executive_summary:
            executive_summary = (
                f"Clinical decision support reliable. No significant issues detected. "
                f"Continue normal operations. Confidence: {confidence}%."
            )
        
        # Clinical recommendations
        clinical_recommendations = []
        for rec in (recommendations := root_cause.get("recommended_actions", []) or ai_investigator.get("recommended_actions", [])):
            clinical_recommendations.append(f"Clinical: {rec}")
        
        # Impact assessment (clinical)
        impact_assessment = "Clinical implications include potential impact on diagnostic reliability, patient risk stratification, and overall quality of care decisions."
        
        # Confidence explanation (clinical)
        confidence_explanation = f"Confidence score of {confidence}% reflects the strength of evidence from model performance monitoring and clinical validation."
        
        # Technical notes (clinical)
        technical_notes = "Technical performance metrics and validation details are available in the full model monitoring report."
        
        return {
            "audience": "Doctor",
            "executive_summary": executive_summary,
            "findings": findings,
            "impact_assessment": impact_assessment,
            "confidence_explanation": confidence_explanation,
            "recommendations": clinical_recommendations,
            "technical_notes": technical_notes
        }
    
    def _translate_for_loan_officer(
        self, root_cause: Dict[str, Any], ai_investigator: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Risk-focused translation for loan officers."""
        health_status = root_cause.get("health_status", "Unknown")
        confidence = root_cause.get("confidence", 0)
        root_causes = root_cause.get("root_causes", [])
        risk_level = ai_investigator.get("risk_level", "UNKNOWN")
        
        # Risk assessment language
        risk_status = {
            "Healthy": "Low Risk",
            "Warning": "Monitor Closely",
            "Degraded": "Moderate Risk",
            "Critical": "High Risk",
            "Unknown": "Uncertain Risk"
        }
        
        assessment_risk = risk_status.get(health_status, "Uncertain")
        
        findings = f"Credit Risk Model Assessment:\n"
        findings += f"Model risk classification: {assessment_risk}\n"
        findings += f"Assessment confidence: {confidence}%\n\n"
        
        if root_causes:
            findings += "Risk Factors Identified:\n"
            for i, cause in enumerate(root_causes[:3], 1):
                severity = cause.get('severity', 'LOW')
                risk_impact = {
                    'CRITICAL': 'High - May affect approval accuracy',
                    'HIGH': 'Moderate - Monitor closely',
                    'MEDIUM': 'Low - Track trends',
                    'LOW': 'Minimal'
                }.get(severity, severity)
                
                findings += f"{i}. {cause.get('cause', 'Unknown')}: {risk_impact}\n"
        
        findings += f"\nOverall Risk Level: {risk_level}\n"
        findings += "\nCompliance Note: Ensure model meets regulatory requirements."
        
        executive_summary = self._resolve_executive_summary(ai_investigator)
        if not executive_summary:
            executive_summary = (
                f"Credit risk model reliable. No significant issues detected. "
                f"Continue normal operations. Risk level: {risk_level}."
            )
        
        # Risk-focused recommendations
        risk_recommendations = []
        for rec in (recommendations := root_cause.get("recommended_actions", []) or ai_investigator.get("recommended_actions", [])):
            risk_recommendations.append(f"Risk Management: {rec}")
        
        # Impact assessment (risk/loan)
        impact_assessment = "Portfolio impact includes potential for inaccurate credit decisions, increased default risk, and compliance implications from unreliable model outputs."
        
        # Confidence explanation (risk/loan)
        confidence_explanation = f"Confidence score of {confidence}% is based on monitoring of model performance metrics, default prediction accuracy, and compliance validation checks."
        
        # Technical notes (loan)
        technical_notes = "Model performance and compliance documentation are available in the regulatory reporting package."
        
        return {
            "audience": "Loan Officer",
            "executive_summary": executive_summary,
            "findings": findings,
            "impact_assessment": impact_assessment,
            "confidence_explanation": confidence_explanation,
            "recommendations": risk_recommendations,
            "technical_notes": technical_notes
        }
    
    def _translate_for_student(
        self, root_cause: Dict[str, Any], ai_investigator: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simplified professional translation for students."""
        health_status = root_cause.get("health_status", "Unknown")
        confidence = root_cause.get("confidence", 0)
        root_causes = root_cause.get("root_causes", [])
        recommendations = root_cause.get("recommended_actions", []) or ai_investigator.get("recommended_actions", [])
        
        # Professional but simplified language
        status_translation = {
            "Healthy": "operating within normal parameters",
            "Warning": "showing signs of degradation",
            "Degraded": "performance below acceptable thresholds",
            "Critical": "requires immediate attention",
            "Unknown": "status unclear"
        }
        
        model_status = status_translation.get(health_status, "status unclear")
        
        # Build professional findings
        findings = f"Model Health Assessment:\n"
        findings += f"Current status: {model_status}\n"
        findings += f"Investigation confidence: {confidence}%\n\n"
        
        if root_causes:
            findings += "Identified Issues:\n"
            for i, cause in enumerate(root_causes[:3], 1):
                severity = cause.get('severity', 'LOW')
                findings += f"{i}. {cause.get('cause', 'Unknown')} (Severity: {severity})\n"
        
        executive_summary = self._resolve_executive_summary(ai_investigator)
        if not executive_summary:
            executive_summary = (
                f"Model operating within normal parameters. No significant issues detected. "
                f"Continue monitoring."
            )
        
        # Impact assessment (simplified professional)
        impact_assessment = "Unreliable model predictions may lead to incorrect decisions. Addressing identified issues is critical to restoring model performance."
        
        # Confidence explanation (simplified)
        confidence_explanation = f"Confidence score of {confidence}% is based on analysis of model health metrics and detected anomalies."
        
        # Technical notes (simplified)
        technical_notes = "Review recommended actions to address identified issues and restore model performance."
        
        # Actionable recommendations with prefix
        action_recommendations = []
        for rec in recommendations:
            action_recommendations.append(f"ACTION REQUIRED: {rec}")
        
        return {
            "audience": "Student",
            "executive_summary": executive_summary,
            "findings": findings,
            "impact_assessment": impact_assessment,
            "confidence_explanation": confidence_explanation,
            "recommendations": action_recommendations,
            "technical_notes": technical_notes
        }
    
    def _translate_generic(
        self, root_cause: Dict[str, Any], ai_investigator: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generic translation for unknown audiences."""
        health_status = root_cause.get("health_status", "Unknown")
        confidence = root_cause.get("confidence", 0)
        root_causes = root_cause.get("root_causes", [])
        
        findings = f"Model Health: {health_status}\n"
        findings += f"Confidence: {confidence}%\n\n"
        
        if root_causes:
            findings += "Issues Identified:\n"
            for i, cause in enumerate(root_causes[:5], 1):
                findings += f"{i}. {cause.get('cause', 'Unknown')} (Severity: {cause.get('severity', 'LOW')})\n"
        
        executive_summary = self._resolve_executive_summary(ai_investigator)
        if not executive_summary:
            executive_summary = f"Model health status: {health_status} with {confidence}% confidence. "
            if root_causes:
                executive_summary += f"{len(root_causes)} issues identified."
        
        # Default fields
        impact_assessment = ai_investigator.get("impact_assessment", "Impact assessment not available.")
        confidence_explanation = ai_investigator.get("confidence_explanation", "Confidence explanation not available.")
        technical_notes = ai_investigator.get("technical_notes", "Technical notes not available.")
        
        return {
            "audience": "General",
            "executive_summary": executive_summary,
            "findings": findings,
            "impact_assessment": impact_assessment,
            "confidence_explanation": confidence_explanation,
            "recommendations": root_cause.get("recommended_actions", []) or ai_investigator.get("recommended_actions", []),
            "technical_notes": technical_notes
        }
