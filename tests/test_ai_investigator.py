"""
Tests for AI Investigator
==========================
Tests for the AIInvestigator module.

Test Scenarios:
- High Risk Investigation
- Medium Risk Investigation
- No Failure Investigation
- Missing Evidence
- Missing Root Causes
- LLM Failure Fallback
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE = os.path.abspath(os.path.join(_HERE, "..", "engine", "modules"))
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)


def test_high_risk_investigation():
    """Test AI Investigator with high risk investigation."""
    from ai_investigator import AIInvestigator
    from investigation import Investigation
    
    # Create high-risk investigation
    inv = Investigation(
        health_status="Critical",
        confidence=87,
        evidence=["Evidence 1", "Evidence 2"],
        recommendations=["Action 1", "Action 2"]
    )
    
    # Add critical root cause
    inv.add_root_cause(
        cause="Age Drift",
        score=87,
        severity="CRITICAL",
        evidence=["PSI=0.31", "Accuracy dropped 18%"],
        category="feature_drift",
        source_modules=["drift_engine"]
    )
    
    investigator = AIInvestigator(use_llm=False, verbose=False)
    result = investigator.analyze(inv)
    
    # Verify structure
    assert "executive_summary" in result
    assert "investigation_findings" in result
    assert "impact_assessment" in result
    assert "confidence_explanation" in result
    assert "recommended_actions" in result
    assert "technical_notes" in result
    assert "risk_level" in result
    
    # Verify risk level
    assert result["risk_level"] == "CRITICAL"
    
    # Verify executive summary is not empty
    assert len(result["executive_summary"]) > 50
    
    # Verify investigation findings
    assert "Age Drift" in result["investigation_findings"]
    
    print("[PASS] High risk investigation test passed")


def test_medium_risk_investigation():
    """Test AI Investigator with medium risk investigation."""
    from ai_investigator import AIInvestigator
    from investigation import Investigation
    
    # Create medium-risk investigation
    inv = Investigation(
        health_status="Warning",
        confidence=65,
        evidence=["Evidence 1"],
        recommendations=["Action 1"]
    )
    
    # Add high severity root cause
    inv.add_root_cause(
        cause="Slice Failure",
        score=60,
        severity="HIGH",
        evidence=["Effect size=0.6"],
        category="slice_degradation",
        source_modules=["slicer_engine"]
    )
    
    investigator = AIInvestigator(use_llm=False, verbose=False)
    result = investigator.analyze(inv)
    
    # Verify risk level
    assert result["risk_level"] in ["MEDIUM", "HIGH"]
    
    # Verify executive summary
    assert len(result["executive_summary"]) > 50
    
    print("[PASS] Medium risk investigation test passed")


def test_no_failure_investigation():
    """Test AI Investigator with no failure investigation."""
    from ai_investigator import AIInvestigator
    from investigation import Investigation
    
    # Create healthy investigation
    inv = Investigation(
        health_status="Healthy",
        confidence=100,
        evidence=["No issues detected"],
        recommendations=["Continue monitoring"]
    )
    
    investigator = AIInvestigator(use_llm=False, verbose=False)
    result = investigator.analyze(inv)
    
    # Verify risk level
    assert result["risk_level"] == "LOW"
    
    # Verify executive summary mentions no degradation
    assert "no significant degradation" in result["executive_summary"].lower()
    
    print("[PASS] No failure investigation test passed")


def test_missing_evidence():
    """Test AI Investigator with missing evidence."""
    from ai_investigator import AIInvestigator
    from investigation import Investigation
    
    # Create investigation with missing evidence
    inv = Investigation(
        health_status="Warning",
        confidence=50,
        evidence=[],
        recommendations=[]
    )
    
    # Add root cause without evidence
    inv.add_root_cause(
        cause="Unknown Issue",
        score=40,
        severity="MEDIUM",
        evidence=[],
        category="unknown",
        source_modules=["unknown"]
    )
    
    investigator = AIInvestigator(use_llm=False, verbose=False)
    result = investigator.analyze(inv)
    
    # Should not crash
    assert result is not None
    assert "executive_summary" in result
    
    print("[PASS] Missing evidence test passed")


def test_missing_root_causes():
    """Test AI Investigator with missing root causes."""
    from ai_investigator import AIInvestigator
    from investigation import Investigation
    
    # Create investigation without root causes
    inv = Investigation(
        health_status="Healthy",
        confidence=100,
        evidence=[],
        recommendations=[]
    )
    
    investigator = AIInvestigator(use_llm=False, verbose=False)
    result = investigator.analyze(inv)
    
    # Should generate sensible narrative
    assert result is not None
    assert "no significant issues" in result["investigation_findings"].lower()
    
    print("[PASS] Missing root causes test passed")


def test_llm_fallback():
    """Test AI Investigator with LLM failure fallback."""
    from ai_investigator import AIInvestigator
    from investigation import Investigation
    
    # Create investigation
    inv = Investigation(
        health_status="Warning",
        confidence=70,
        evidence=["Evidence 1"],
        recommendations=["Action 1"]
    )
    
    inv.add_root_cause(
        cause="Test Cause",
        score=70,
        severity="HIGH",
        evidence=["Test evidence"],
        category="test",
        source_modules=["test_engine"]
    )
    
    # Request LLM but it should fallback to deterministic
    investigator = AIInvestigator(use_llm=True, verbose=False)
    result = investigator.analyze(inv)
    
    # Should still work
    assert result is not None
    assert "executive_summary" in result
    
    print("[PASS] LLM fallback test passed")


def test_investigation_report_generation():
    """Test full investigation report generation."""
    from ai_investigator import AIInvestigator
    from investigation import Investigation
    
    # Create investigation
    inv = Investigation(
        health_status="Warning",
        confidence=75,
        evidence=["Evidence 1", "Evidence 2"],
        recommendations=["Action 1", "Action 2"]
    )
    
    inv.add_root_cause(
        cause="Age Drift",
        score=75,
        severity="HIGH",
        evidence=["PSI=0.30", "KS=0.20"],
        category="feature_drift",
        source_modules=["drift_engine"]
    )
    
    investigator = AIInvestigator(use_llm=False, verbose=False)
    report = investigator.generate_investigation_report(inv)
    
    # Verify report structure
    assert "AI INVESTIGATOR REPORT" in report
    assert "EXECUTIVE SUMMARY" in report
    assert "INVESTIGATION FINDINGS" in report
    assert "IMPACT ASSESSMENT" in report
    assert "CONFIDENCE EXPLANATION" in report
    assert "RECOMMENDED ACTIONS" in report
    assert "TECHNICAL NOTES" in report
    assert "Risk Level:" in report
    
    print("[PASS] Investigation report generation test passed")


def test_risk_assessment_logic():
    """Test risk assessment logic."""
    from ai_investigator import AIInvestigator
    from investigation import Investigation
    
    investigator = AIInvestigator(use_llm=False, verbose=False)
    
    # Test Critical risk
    inv1 = Investigation(health_status="Critical", confidence=80)
    inv1.add_root_cause("Test", 90, "CRITICAL", [], "test", ["test"])
    risk1 = investigator._assess_risk_level(inv1)
    assert risk1 == "CRITICAL"
    
    # Test High risk
    inv2 = Investigation(health_status="Warning", confidence=85)
    inv2.add_root_cause("Test", 80, "HIGH", [], "test", ["test"])
    inv2.add_root_cause("Test2", 75, "HIGH", [], "test", ["test"])
    risk2 = investigator._assess_risk_level(inv2)
    assert risk2 == "HIGH"
    
    # Test Low risk
    inv3 = Investigation(health_status="Healthy", confidence=100)
    risk3 = investigator._assess_risk_level(inv3)
    assert risk3 == "LOW"
    
    print("[PASS] Risk assessment logic test passed")


def test_executive_summary_generation():
    """Test executive summary generation."""
    from ai_investigator import AIInvestigator
    from investigation import Investigation
    
    investigator = AIInvestigator(use_llm=False, verbose=False)
    
    # Test with feature drift
    inv = Investigation(health_status="Warning", confidence=75)
    inv.add_root_cause(
        "age feature drift",
        75,
        "HIGH",
        ["PSI=0.30"],
        "feature_drift",
        ["drift_engine"]
    )
    
    summary = investigator._generate_executive_summary(inv, "HIGH")
    
    assert len(summary) > 50
    assert "age" in summary.lower() or "drift" in summary.lower()
    
    print("[PASS] Executive summary generation test passed")


def run_all_tests():
    """Run all AI Investigator tests."""
    print("=" * 65)
    print("AI Investigator Tests")
    print("=" * 65)
    
    try:
        test_high_risk_investigation()
        test_medium_risk_investigation()
        test_no_failure_investigation()
        test_missing_evidence()
        test_missing_root_causes()
        test_llm_fallback()
        test_investigation_report_generation()
        test_risk_assessment_logic()
        test_executive_summary_generation()
        
        print("\n" + "=" * 65)
        print("All AI Investigator tests PASSED [OK]")
        print("=" * 65)
        return True
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_all_tests()
