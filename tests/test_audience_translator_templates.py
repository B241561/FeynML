import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.modules.audience_translator import AudienceTranslator


SAMPLE_ROOT_CAUSE = {
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

SAMPLE_AI = {
    "risk_level": "HIGH",
    "executive_summary": "Model shows signs of degradation due to data drift and label noise issues.",
    "impact_assessment": "Model performance may degrade by 15-20% if not addressed.",
    "confidence_explanation": "High confidence based on statistical significance of drift metrics.",
    "recommended_actions": [
        "Implement drift monitoring",
        "Review labeling process",
        "Schedule model retraining",
    ],
}

FORBIDDEN_PHRASES = [
    "revenue impact",
    "patient care",
    "regulatory violations",
    "fines",
    "default risk",
    "claims costs",
    "employee decisions",
    "retention strategies",
]


def test_audience_translation_preserves_recommendations_and_evidence():
    translator = AudienceTranslator()
    reports = translator.translate(SAMPLE_ROOT_CAUSE, SAMPLE_AI)

    for audience in translator.AUDIENCES:
        report = reports[audience]
        assert report["recommendations"] == SAMPLE_ROOT_CAUSE["recommended_actions"]
        assert "Feature X drifted by 0.45" in report["findings"]
        assert "Feature Y drifted by 0.32" in report["findings"]
        assert "15% of labels may be incorrect" in report["findings"]


def test_audience_translation_varies_wording_by_audience():
    translator = AudienceTranslator()
    reports = translator.translate(SAMPLE_ROOT_CAUSE, SAMPLE_AI)

    assert reports["ML Engineer"]["executive_summary"] != reports["Executive"]["executive_summary"]
    assert reports["Doctor"]["findings"] != reports["Student"]["findings"]
    assert reports["Legal / Compliance Officer"]["impact_assessment"] != reports["Researcher"]["impact_assessment"]
    assert "2 evidence-backed issues were identified." not in reports["Executive"]["executive_summary"]
    assert "Two monitoring signals require review." in reports["Executive"]["executive_summary"]
    assert "Two clinical-model review signals were observed." in reports["Doctor"]["executive_summary"]
    assert "Two important model issues were found." in reports["Student"]["executive_summary"]
    assert "Two investigation signals were detected." in reports["Researcher"]["executive_summary"]


def test_audience_translation_renders_risk_level_for_all_audiences():
    translator = AudienceTranslator()
    reports = translator.translate(SAMPLE_ROOT_CAUSE, SAMPLE_AI)

    for audience in translator.AUDIENCES:
        assert "Risk level: HIGH" in reports[audience]["findings"]


def test_audience_translation_avoids_unsupported_consequence_claims():
    translator = AudienceTranslator()
    reports = translator.translate(SAMPLE_ROOT_CAUSE, SAMPLE_AI)

    for audience in translator.AUDIENCES:
        report_blob = " ".join(
            [
                reports[audience]["executive_summary"],
                reports[audience]["findings"],
                reports[audience]["impact_assessment"],
                reports[audience]["confidence_explanation"],
                reports[audience]["technical_notes"],
            ]
        ).lower()
        for forbidden in FORBIDDEN_PHRASES:
            assert forbidden not in report_blob
