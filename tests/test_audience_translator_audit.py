"""
Audit script for Audience Translation Layer runtime behavior.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from engine.modules.audience_translator import AudienceTranslator


sample_root_cause = {
    "health_status": "Degraded",
    "confidence": 85,
    "root_causes": [
        {
            "cause": "Data Drift Detected",
            "score": 0.92,
            "severity": "HIGH",
            "evidence": ["Feature X drifted by 0.45", "Feature Y drifted by 0.32"],
        },
        {
            "cause": "Label Noise Present",
            "score": 0.78,
            "severity": "MEDIUM",
            "evidence": ["15% of labels may be incorrect"],
        },
    ],
    "recommended_actions": [
        "Retrain model with recent data",
        "Investigate label quality",
        "Monitor feature drift",
    ],
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
        "Schedule model retraining",
    ],
    "technical_notes": "Drift detected using KS test (p<0.01). Label noise estimated via consensus algorithms.",
}

forbidden_phrases = [
    "revenue impact",
    "patient care",
    "regulatory violations",
    "fines",
    "default risk",
    "claims costs",
    "employee decisions",
    "retention strategies",
]

translator = AudienceTranslator(verbose=True)
audience_reports = translator.translate(sample_root_cause, sample_ai_investigator)

print("=" * 80)
print("AUDIENCE TRANSLATION AUDIT")
print("=" * 80)
print(json.dumps(audience_reports, indent=2))

print("\n" + "=" * 80)
print("VERIFICATION: Reports exist for all supported audiences")
print("=" * 80)
for audience in translator.AUDIENCES:
    report = audience_reports.get(audience)
    print(f"[OK] {audience}: {'EXISTS' if report else 'MISSING'}")

print("\n" + "=" * 80)
print("VERIFICATION: Shared structure")
print("=" * 80)
required_fields = [
    "audience",
    "executive_summary",
    "findings",
    "impact_assessment",
    "confidence_explanation",
    "recommendations",
    "technical_notes",
]
for audience in translator.AUDIENCES:
    report = audience_reports[audience]
    missing = [field for field in required_fields if field not in report]
    print(f"{audience}: {'OK' if not missing else f'MISSING {missing}'}")

print("\n" + "=" * 80)
print("VERIFICATION: Evidence preserved verbatim")
print("=" * 80)
for audience in translator.AUDIENCES:
    findings = audience_reports[audience]["findings"]
    checks = [
        "Feature X drifted by 0.45" in findings,
        "Feature Y drifted by 0.32" in findings,
        "15% of labels may be incorrect" in findings,
    ]
    print(f"{audience}: {'OK' if all(checks) else 'FAIL'}")

print("\n" + "=" * 80)
print("VERIFICATION: Recommendations unchanged")
print("=" * 80)
expected_recommendations = sample_root_cause["recommended_actions"]
for audience in translator.AUDIENCES:
    same = audience_reports[audience]["recommendations"] == expected_recommendations
    print(f"{audience}: {'OK' if same else 'FAIL'}")

print("\n" + "=" * 80)
print("VERIFICATION: No unsupported downstream consequence phrases")
print("=" * 80)
for audience in translator.AUDIENCES:
    report_text = json.dumps(audience_reports[audience]).lower()
    found = [phrase for phrase in forbidden_phrases if phrase in report_text]
    print(f"{audience}: {'OK' if not found else f'FOUND {found}'}")
