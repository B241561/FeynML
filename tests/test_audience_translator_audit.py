"""
Audit script for Audience Translation Layer runtime behavior.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from engine.modules.audience_translator import AudienceTranslator

# Sample data for testing
sample_root_cause = {
    "health_status": "Degraded",
    "confidence": 85,
    "root_causes": [
        {
            "cause": "Data Drift Detected",
            "score": 0.92,
            "severity": "HIGH",
            "evidence": ["Feature X drifted by 0.45", "Feature Y drifted by 0.32"]
        },
        {
            "cause": "Label Noise Present",
            "score": 0.78,
            "severity": "MEDIUM",
            "evidence": ["15% of labels may be incorrect"]
        }
    ],
    "recommended_actions": [
        "Retrain model with recent data",
        "Investigate label quality",
        "Monitor feature drift"
    ]
}

sample_ai_investigator = {
    "risk_level": "HIGH",
    "investigation_id": "INV-2024-001",
    "executive_summary": "Model shows signs of degradation due to data drift and label noise issues.",
    "investigation_findings": "Detailed analysis reveals significant drift in feature distributions and potential label contamination.",
    "impact_assessment": "Model performance may degrade by 15-20% if not addressed.",
    "confidence_explanation": "High confidence based on statistical significance of drift metrics.",
    "recommended_actions": [
        "Implement drift monitoring",
        "Review labeling process",
        "Schedule model retraining"
    ],
    "technical_notes": "Drift detected using KS test (p<0.01). Label noise estimated via consensus algorithms."
}

# Test audience translation
print("=" * 80)
print("AUDIENCE TRANSLATION AUDIT")
print("=" * 80)

translator = AudienceTranslator(verbose=True)
audience_reports = translator.translate(sample_root_cause, sample_ai_investigator)

print("\n" + "=" * 80)
print("AUDIENCE_REPORTS JSON SAMPLE")
print("=" * 80)
print(json.dumps(audience_reports, indent=2))

print("\n" + "=" * 80)
print("VERIFICATION: Reports exist for all 5 audiences")
print("=" * 80)
required_audiences = ["ML Engineer", "Executive", "Doctor", "Loan Officer", "Student"]
for audience in required_audiences:
    if audience in audience_reports:
        print(f"[OK] {audience}: EXISTS")
        print(f"  Fields: {list(audience_reports[audience].keys())}")
    else:
        print(f"[FAIL] {audience}: MISSING")

print("\n" + "=" * 80)
print("VERIFICATION: Field structure per audience")
print("=" * 80)
for audience in required_audiences:
    if audience in audience_reports:
        report = audience_reports[audience]
        print(f"\n{audience}:")
        for key, value in report.items():
            print(f"  {key}: {type(value).__name__} (length: {len(str(value))})")

print("\n" + "=" * 80)
print("BUG IDENTIFICATION")
print("=" * 80)
print("\nJavaScript expects these fields:")
print("  - executive_summary")
print("  - findings")
print("  - impact_assessment")
print("  - confidence_explanation")
print("  - recommendations")
print("  - technical_notes")

print("\nAudience Translator returns these fields:")
first_report = audience_reports.get("ML Engineer", {})
print(f"  - {list(first_report.keys())}")

print("\nMISSING FIELDS (JavaScript tries to update but don't exist):")
missing_fields = ["impact_assessment", "confidence_explanation", "technical_notes"]
for field in missing_fields:
    if field not in first_report:
        print(f"  [MISSING] {field}: This is the BUG!")
    else:
        print(f"  [OK] {field}: EXISTS")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("BUG: JavaScript tries to update fields that don't exist in audience_reports.")
print("The audience_translator.py only returns: audience, executive_summary, findings, recommendations")
print("But JavaScript tries to update: impact_assessment, confidence_explanation, technical_notes")
print("These fields are embedded in 'findings' string, not as separate fields.")
