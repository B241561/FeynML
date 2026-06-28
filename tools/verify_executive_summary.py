"""
Verification script for Executive Summary refactoring.
Tests the new synthesized narrative format with audience-specific styles.
"""

import sys
import os

# Add project root to path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine.modules.ai_investigator import AIInvestigator
from engine.modules.investigation import Investigation

def test_executive_summary_refactor():
    """Test the refactored executive summary generation."""
    print("=" * 70)
    print("VERIFICATION: Executive Summary Refactoring")
    print("=" * 70)
    
    # Create a test investigation with feature drift
    from engine.modules.investigation import RootCause
    
    cause = RootCause(
        cause="Attendance feature drift",
        score=0.85,
        severity="HIGH",
        category="feature_drift",
        evidence=["PSI: 0.45", "KS: 0.62"],
        source_modules=["drift_engine"]
    )
    
    investigation = Investigation(
        health_status="Warning",
        confidence=80,
        root_causes=[cause],
        recommendations=["Retrain model with recent data", "Monitor attendance patterns"]
    )
    
    investigator = AIInvestigator(use_llm=False, verbose=False)
    
    # Test all audience types
    audiences = ["ML Engineer", "Executive", "Doctor", "Loan Officer", "Student"]
    
    print("\n" + "=" * 70)
    print("TESTING AUDIENCE-SPECIFIC EXECUTIVE SUMMARIES")
    print("=" * 70)
    
    for audience in audiences:
        print(f"\n--- {audience} ---")
        result = investigator.analyze(investigation, audience)
        summary = result['executive_summary']
        word_count = len(summary.split())
        
        print(f"Summary: {summary}")
        print(f"Word count: {word_count}")
        
        # Verify word count limit
        if word_count <= 120:
            print("[PASS] Word count within 120-word limit")
        else:
            print(f"[FAIL] Word count exceeds 120-word limit ({word_count} words)")
        
        # Verify it's a synthesized narrative (not a list)
        if not summary.startswith(("1.", "2.", "3.", "-", "•")):
            print("[PASS] Synthesized narrative format (not a list)")
        else:
            print("[FAIL] Appears to be a list format")
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    
    # Test the audience translator as well
    print("\n" + "=" * 70)
    print("TESTING AUDIENCE TRANSLATOR EXECUTIVE SUMMARIES")
    print("=" * 70)
    
    from engine.modules.audience_translator import AudienceTranslator
    
    root_cause_dict = {
        "health_status": "Warning",
        "confidence": 80,
        "root_causes": [
            {
                "cause": "Attendance feature drift",
                "score": 0.85,
                "severity": "HIGH",
                "category": "feature_drift",
                "evidence": ["PSI: 0.45", "KS: 0.62"],
                "source_modules": ["drift_engine"]
            }
        ],
        "recommended_actions": ["Retrain model with recent data", "Monitor attendance patterns"]
    }
    
    ai_investigator_dict = {
        "risk_level": "HIGH",
        "executive_summary": "Test summary",
        "investigation_findings": "Test findings",
        "impact_assessment": "Test impact",
        "confidence_explanation": "Test confidence",
        "recommended_actions": ["Retrain model", "Monitor patterns"],
        "technical_notes": "Test notes"
    }
    
    translator = AudienceTranslator(verbose=False)
    audience_reports = translator.translate(root_cause_dict, ai_investigator_dict)
    
    for audience, report in audience_reports.items():
        print(f"\n--- {audience} ---")
        summary = report['executive_summary']
        word_count = len(summary.split())
        
        print(f"Summary: {summary}")
        print(f"Word count: {word_count}")
        
        # Verify word count limit
        if word_count <= 120:
            print("[PASS] Word count within 120-word limit")
        else:
            print(f"[FAIL] Word count exceeds 120-word limit ({word_count} words)")
    
    print("\n" + "=" * 70)
    print("ALL VERIFICATIONS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    test_executive_summary_refactor()
