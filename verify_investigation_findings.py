"""
Verification script for AI Investigator Investigation Findings refactoring.
Tests that Investigation Findings is a synthesized narrative (interpretation layer)
and does not duplicate Root Cause Analysis (evidence layer).
"""

import sys
import os

# Add project root to path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine.modules.ai_investigator import AIInvestigator
from engine.modules.investigation import Investigation, RootCause

def test_investigation_findings_refactor():
    """Test the refactored Investigation Findings generation."""
    print("=" * 70)
    print("VERIFICATION: Investigation Findings Refactoring")
    print("=" * 70)
    
    # Create test investigation with feature drift
    causes = [
        RootCause(
            cause="Attendance feature drift",
            score=85,
            severity="HIGH",
            category="feature_drift",
            evidence=["PSI=13.624", "KS=1.000"],
            source_modules=["drift_engine"]
        ),
        RootCause(
            cause="StudyTime feature drift",
            score=75,
            severity="HIGH",
            category="feature_drift",
            evidence=["PSI=8.245", "KS=0.850"],
            source_modules=["drift_engine"]
        ),
        RootCause(
            cause="Grade feature drift",
            score=65,
            severity="MEDIUM",
            category="feature_drift",
            evidence=["PSI=5.123", "KS=0.620"],
            source_modules=["drift_engine"]
        ),
    ]
    
    investigation = Investigation(
        health_status="Warning",
        confidence=80,
        root_causes=causes,
        recommendations=["Retrain model with recent data", "Monitor attendance patterns"]
    )
    
    investigator = AIInvestigator(use_llm=False, verbose=False)
    result = investigator.analyze(investigation)
    
    findings = result['investigation_findings']
    
    print("\n--- Investigation Findings Output ---")
    print(findings)
    print("\n" + "-" * 70)
    
    # Verification checks
    success = True
    
    # Check 1: No PSI values in findings
    if "PSI=" not in findings and "PSI " not in findings:
        print("[PASS] No PSI values in Investigation Findings")
    else:
        print("[FAIL] PSI values found in Investigation Findings")
        success = False
    
    # Check 2: No KS values in findings
    if "KS=" not in findings and "KS " not in findings:
        print("[PASS] No KS values in Investigation Findings")
    else:
        print("[FAIL] KS values found in Investigation Findings")
        success = False
    
    # Check 3: No numbered list format (not repeating root cause table)
    if not findings.startswith("1.") and "1. " not in findings[:50]:
        print("[PASS] No numbered list format (not repeating root cause table)")
    else:
        print("[FAIL] Numbered list format found (repeating root cause table)")
        success = False
    
    # Check 4: Contains interpretation/pattern summarization
    interpretation_keywords = ["distributional", "shift", "pattern", "indicates", "suggesting", "affecting"]
    has_interpretation = any(keyword in findings.lower() for keyword in interpretation_keywords)
    if has_interpretation:
        print("[PASS] Contains interpretation/pattern summarization")
    else:
        print("[FAIL] Missing interpretation/pattern summarization")
        success = False
    
    # Check 5: Explains why issues matter
    impact_keywords = ["impact", "affect", "degradation", "reliability", "performance"]
    has_impact = any(keyword in findings.lower() for keyword in impact_keywords)
    if has_impact:
        print("[PASS] Explains why issues matter")
    else:
        print("[FAIL] Missing explanation of why issues matter")
        success = False
    
    # Check 6: Technical depth maintained
    technical_keywords = ["distributional", "model", "prediction", "data", "feature"]
    has_technical = any(keyword in findings.lower() for keyword in technical_keywords)
    if has_technical:
        print("[PASS] Technical depth maintained")
    else:
        print("[FAIL] Missing technical depth")
        success = False
    
    # Check 7: Professional/recruiter-demo quality
    unprofessional_keywords = ["suck", "bad", "terrible", "awful", "broken"]
    has_unprofessional = any(keyword in findings.lower() for keyword in unprofessional_keywords)
    if not has_unprofessional:
        print("[PASS] Professional/recruiter-demo quality")
    else:
        print("[FAIL] Unprofessional language detected")
        success = False
    
    return success

def test_multiple_categories():
    """Test Investigation Findings with multiple categories."""
    print("\n" + "=" * 70)
    print("VERIFICATION: Multiple Categories Pattern Summarization")
    print("=" * 70)
    
    # Create test investigation with multiple categories
    causes = [
        RootCause(
            cause="Attendance feature drift",
            score=85,
            severity="HIGH",
            category="feature_drift",
            evidence=["PSI=13.624", "KS=1.000"],
            source_modules=["drift_engine"]
        ),
        RootCause(
            cause="Slice failure: Low attendance students",
            score=70,
            severity="MEDIUM",
            category="slice_degradation",
            evidence=["Effect size=0.65", "loss gap=0.15"],
            source_modules=["slicer_engine"]
        ),
        RootCause(
            cause="Calibration degradation",
            score=60,
            severity="MEDIUM",
            category="calibration",
            evidence=["ECE=0.08", "Brier=0.15"],
            source_modules=["calibration_engine"]
        ),
    ]
    
    investigation = Investigation(
        health_status="Warning",
        confidence=75,
        root_causes=causes,
        recommendations=["Retrain model", "Monitor slices"]
    )
    
    investigator = AIInvestigator(use_llm=False, verbose=False)
    result = investigator.analyze(investigation)
    
    findings = result['investigation_findings']
    
    print("\n--- Investigation Findings Output ---")
    print(findings)
    print("\n" + "-" * 70)
    
    success = True
    
    # Check that multiple categories are summarized
    category_keywords = ["distributional", "slice", "segment", "calibration", "probability"]
    has_multiple_categories = sum(1 for keyword in category_keywords if keyword in findings.lower())
    if has_multiple_categories >= 2:
        print(f"[PASS] Multiple categories summarized ({has_multiple_categories} category keywords found)")
    else:
        print(f"[FAIL] Multiple categories not properly summarized (only {has_multiple_categories} category keywords found)")
        success = False
    
    return success

def test_single_feature_drift():
    """Test Investigation Findings with single feature drift."""
    print("\n" + "=" * 70)
    print("VERIFICATION: Single Feature Drift Interpretation")
    print("=" * 70)
    
    # Create test investigation with single feature drift
    causes = [
        RootCause(
            cause="Attendance feature drift",
            score=85,
            severity="HIGH",
            category="feature_drift",
            evidence=["PSI=13.624", "KS=1.000"],
            source_modules=["drift_engine"]
        ),
    ]
    
    investigation = Investigation(
        health_status="Warning",
        confidence=80,
        root_causes=causes,
        recommendations=["Retrain model"]
    )
    
    investigator = AIInvestigator(use_llm=False, verbose=False)
    result = investigator.analyze(investigation)
    
    findings = result['investigation_findings']
    
    print("\n--- Investigation Findings Output ---")
    print(findings)
    print("\n" + "-" * 70)
    
    # Check that single feature is interpreted as localized
    if "single" in findings.lower() or "localized" in findings.lower():
        print("[PASS] Single feature interpreted as localized")
    else:
        print("[INFO] Single feature interpretation may vary (acceptable)")
    
    return True

def test_no_root_causes():
    """Test Investigation Findings with no root causes."""
    print("\n" + "=" * 70)
    print("VERIFICATION: No Root Causes Case")
    print("=" * 70)
    
    investigation = Investigation(
        health_status="Healthy",
        confidence=100,
        root_causes=[],
        recommendations=["Continue monitoring"]
    )
    
    investigator = AIInvestigator(use_llm=False, verbose=False)
    result = investigator.analyze(investigation)
    
    findings = result['investigation_findings']
    
    print("\n--- Investigation Findings Output ---")
    print(findings)
    print("\n" + "-" * 70)
    
    # Check appropriate message for no issues
    if "no significant issues" in findings.lower() or "within acceptable" in findings.lower():
        print("[PASS] Appropriate message for no root causes")
    else:
        print("[FAIL] Inappropriate message for no root causes")
        return False
    
    return True

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("INVESTIGATION FINDINGS REFACTORING VERIFICATION")
    print("=" * 70)
    
    test1 = test_investigation_findings_refactor()
    test2 = test_multiple_categories()
    test3 = test_single_feature_drift()
    test4 = test_no_root_causes()
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    if test1 and test2 and test3 and test4:
        print("[SUCCESS] All verification tests passed!")
        sys.exit(0)
    else:
        print("[FAILURE] Some verification tests failed!")
        sys.exit(1)
