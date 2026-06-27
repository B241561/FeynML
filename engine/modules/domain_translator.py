import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Groq Configuration (same pattern as chatbot_routes.py)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile"


class DomainTranslator:
    AUDIENCE_MAP = {
        "ml_engineer": "ML Engineer",
        "executive": "Executive",
        "doctor": "Doctor",
        "loan_officer": "Loan Officer",
        "student": "Student",
        "hr_manager": "HR Manager",
        "insurance_analyst": "Insurance Analyst",
        "legal_compliance": "Legal / Compliance Officer",
        "researcher": "Researcher"
    }

    FALLBACK_TEMPLATES = {
        "drift": {
            "CRITICAL": "This feature has changed significantly since training. Model predictions may be unreliable. Immediate retraining recommended.",
            "HIGH": "This feature has changed substantially since training. Consider retraining the model soon.",
            "MEDIUM": "This feature has changed somewhat since training. Monitor performance closely.",
            "LOW": "No significant changes detected in this feature."
        },
        "leakage": {
            "CRITICAL": "This feature likely contains target information that would not be available at prediction time. Remove this feature immediately.",
            "HIGH": "This feature may contain inappropriate information that could leak future knowledge. Review this feature carefully.",
            "MEDIUM": "This feature shows some signs of potential leakage. Investigate further.",
            "LOW": "No significant leakage detected in this feature."
        },
        "noise": {
            "CRITICAL": "A large number of potentially mislabeled samples detected. Clean your dataset immediately.",
            "HIGH": "Many potentially mislabeled samples found. Review and correct labels.",
            "MEDIUM": "Some potentially mislabeled samples detected. Consider reviewing the dataset.",
            "LOW": "No significant label noise detected."
        },
        "fairness": {
            "CRITICAL": "Severe fairness issues detected. Model may be biased against certain groups. Correct immediately.",
            "HIGH": "Important fairness concerns identified. Review model performance across groups.",
            "MEDIUM": "Some fairness metrics are outside acceptable ranges. Monitor closely.",
            "LOW": "No significant fairness issues detected."
        },
        "calibration": {
            "CRITICAL": "Model is poorly calibrated. Confidence scores are unreliable. Recalibrate immediately.",
            "HIGH": "Model calibration is suboptimal. Consider recalibrating the model.",
            "MEDIUM": "Model shows some calibration issues. Monitor confidence scores.",
            "LOW": "Model is well-calibrated."
        }
    }

    def __init__(self, verbose=False):
        self.verbose = verbose

    def translate(self, finding_type, feature_name, severity, technical_details, audience):
        """Translate a technical finding to audience-specific language using Groq or fallback."""
        display_audience = self.AUDIENCE_MAP.get(audience, audience)

        # Rule-based fallback first (or if API fails)
        try:
            if not GROQ_API_KEY:
                if self.verbose:
                    print("DomainTranslator: GROQ_API_KEY not set, using fallback")
                return self._get_fallback(finding_type, severity)

            client = Groq(api_key=GROQ_API_KEY)

            system_prompt = f"""You are a domain expert translator for ML model failure findings. Convert technical ML diagnostics into clear actionable language for the given audience.

Audience: {display_audience}

Rules:
- Use terminology natural to THIS audience only
- Describe only effects directly supported by the supplied technical evidence.
- Do not infer business, financial, legal, clinical, underwriting, compliance, revenue, portfolio, reputational, or patient outcomes unless such evidence is explicitly provided in the input.
- Prefer evidence-grounded interpretation language like:
  * Feature distribution differs from training data.
  * Prediction reliability may decrease.
  * Model behavior should be reviewed.
  * Calibration quality has degraded.
  * Additional validation is recommended.
- Avoid consequence-based claims like:
  * Revenue loss.
  * Portfolio losses.
  * Incorrect loan approvals.
  * Regulatory exposure.
  * Reputational damage.
  * Patient harm.
- Give ONE specific recommended action
- CRITICAL severity = urgent, alarming language
- HIGH severity = important but not emergency
- MEDIUM = advisory, informational tone
- Maximum 2-3 sentences total
- Zero ML jargon unless audience is ml_engineer
"""

            user_message = f"""Finding: {finding_type}
Feature: {feature_name}
Severity: {severity}
Technical details: {technical_details}
Translate this for: {display_audience}
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]

            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
                max_tokens=200,
                top_p=1,
                stream=False,
                stop=None,
            )

            return completion.choices[0].message.content

        except Exception:
            return self._get_fallback(finding_type, severity)

    def _get_fallback(self, finding_type, severity):
        """Return pre-written template string based on finding_type + severity."""
        finding = finding_type.lower()
        if finding not in self.FALLBACK_TEMPLATES:
            finding = "drift"  # Default to drift if unknown finding type
        return self.FALLBACK_TEMPLATES[finding].get(severity, self.FALLBACK_TEMPLATES[finding]["LOW"])


if __name__ == "__main__":
    # Test the DomainTranslator
    print("--- Testing DomainTranslator ---")
    translator = DomainTranslator(verbose=True)
    
    # Test with various inputs
    test_translations = [
        ("drift", "age", "CRITICAL", "KS statistic = 0.32, PSI = 0.28", "executive"),
        ("leakage", "future_purchase", "HIGH", "Mutual information score = 0.91", "loan_officer"),
        ("noise", "diagnosis", "MEDIUM", "Label quality score average = 0.72", "doctor"),
    ]
    
    for finding_type, feature_name, severity, tech_details, audience in test_translations:
        print(f"\n=== Test: {finding_type} → {audience} ===")
        result = translator.translate(finding_type, feature_name, severity, tech_details, audience)
        print(f"Result: {result}")
