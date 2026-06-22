"""
Tests for Auto Root Cause Engine
================================
Tests for the AutoRootCauseEngine module and related Phase 1.5 features.

Test Scenarios:
- High Drift Scenario
- Slice Failure Scenario
- Calibration Shift Scenario
- Mixed Failure Scenario
- No Failure Scenario
- Investigation Object Tests
- Investigation Service Tests
- SHAP Importance Drift Tests
- Edge Case Handling Tests
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE = os.path.abspath(os.path.join(_HERE, "..", "engine", "modules"))
_WEBAPP = os.path.abspath(os.path.join(_HERE, "..", "webapp", "services"))
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)
if _WEBAPP not in sys.path:
    sys.path.insert(0, _WEBAPP)


def test_high_drift_scenario():
    """Test root cause engine with high feature drift."""
    from root_cause_engine import AutoRootCauseEngine
    
    # Simulate high drift report
    drift_report = {
        "findings": {
            "per_feature": [
                {"feature": "age", "psi": 0.35, "ks_stat": 0.25, "status": "DRIFT"},
                {"feature": "income", "psi": 0.45, "ks_stat": 0.30, "status": "DRIFT"},
            ],
            "num_drifted_features": 2,
            "mean_psi": 0.40
        }
    }
    
    engine = AutoRootCauseEngine(verbose=False)
    result = engine.run(drift_report=drift_report)
    
    assert result["health_status"] in ["Warning", "Critical"]
    assert result["confidence"] > 50
    assert len(result["root_causes"]) > 0
    assert any("age" in c["cause"] for c in result["root_causes"])
    assert any("income" in c["cause"] for c in result["root_causes"])
    assert len(result["recommended_actions"]) > 0
    print("[PASS] High drift scenario test passed")


def test_slice_failure_scenario():
    """Test root cause engine with slice degradation."""
    from root_cause_engine import AutoRootCauseEngine
    
    # Simulate slice failure report
    slice_report = {
        "findings": {
            "slices": [
                {
                    "description": "age=senior, gender=female",
                    "effect_size": 0.85,
                    "slice_loss": 0.65,
                    "p_value": 0.01
                }
            ],
            "overall_loss": 0.25,
            "n_slices_found": 1
        }
    }
    
    engine = AutoRootCauseEngine(verbose=False)
    result = engine.run(slice_report=slice_report)
    
    assert result["health_status"] in ["Warning", "Critical"]
    assert len(result["root_causes"]) > 0
    assert any("Slice failure" in c["cause"] for c in result["root_causes"])
    assert any(c["severity"] in ["HIGH", "CRITICAL"] for c in result["root_causes"])
    print("[PASS] Slice failure scenario test passed")


def test_calibration_shift_scenario():
    """Test root cause engine with calibration degradation."""
    from root_cause_engine import AutoRootCauseEngine
    
    # Simulate calibration degradation report
    calibration_report = {
        "ece": 0.12,
        "brier_score": 0.25,
        "severity": "HIGH"
    }
    
    engine = AutoRootCauseEngine(verbose=False)
    result = engine.run(calibration_report=calibration_report)
    
    assert result["health_status"] in ["Warning", "Critical"]
    assert len(result["root_causes"]) > 0
    assert any("Calibration" in c["cause"] for c in result["root_causes"])
    print("[PASS] Calibration shift scenario test passed")


def test_mixed_failure_scenario():
    """Test root cause engine with multiple failure types."""
    from root_cause_engine import AutoRootCauseEngine
    
    # Simulate mixed failure report
    drift_report = {
        "findings": {
            "per_feature": [
                {"feature": "age", "psi": 0.30, "ks_stat": 0.20, "status": "DRIFT"},
            ],
            "num_drifted_features": 1
        }
    }
    
    calibration_report = {
        "ece": 0.08,
        "brier_score": 0.20
    }
    
    data_quality_report = {
        "checks": [
            {"check": "missing_values", "feature": "income", "missing_rate": 0.15}
        ]
    }
    
    engine = AutoRootCauseEngine(verbose=False)
    result = engine.run(
        drift_report=drift_report,
        calibration_report=calibration_report,
        data_quality_report=data_quality_report
    )
    
    assert result["health_status"] in ["Warning", "Critical"]
    assert len(result["root_causes"]) >= 2
    assert result["confidence"] >= 50
    assert len(result["recommended_actions"]) > 0
    print("[PASS] Mixed failure scenario test passed")


def test_no_failure_scenario():
    """Test root cause engine with no failures."""
    from root_cause_engine import AutoRootCauseEngine
    
    # Simulate no failure reports
    drift_report = {
        "findings": {
            "per_feature": [],
            "num_drifted_features": 0
        }
    }
    
    calibration_report = {
        "ece": 0.03,
        "brier_score": 0.10
    }
    
    engine = AutoRootCauseEngine(verbose=False)
    result = engine.run(
        drift_report=drift_report,
        calibration_report=calibration_report
    )
    
    assert result["health_status"] == "Healthy"
    assert result["confidence"] == 100
    assert len(result["root_causes"]) == 0
    assert "No critical issues" in result["recommended_actions"][0]
    print("[PASS] No failure scenario test passed")


def test_scoring_rules():
    """Test that scoring rules are correctly applied."""
    from root_cause_engine import AutoRootCauseEngine
    
    engine = AutoRootCauseEngine(verbose=False)
    rules = engine._scoring_rules
    
    # Check feature drift rules
    assert "feature_drift" in rules
    assert "psi_gt_025" in rules["feature_drift"]
    assert rules["feature_drift"]["psi_gt_025"]["score"] == 50
    assert rules["feature_drift"]["psi_gt_050"]["score"] == 75
    
    # Check slice degradation rules
    assert "slice_degradation" in rules
    assert "effect_size_gt_08" in rules["slice_degradation"]
    assert rules["slice_degradation"]["effect_size_gt_08"]["score"] == 65
    
    # Check calibration rules
    assert "calibration" in rules
    assert "ece_increase_gt_010" in rules["calibration"]
    assert rules["calibration"]["ece_increase_gt_010"]["score"] == 50
    
    print("[PASS] Scoring rules test passed")


def test_investigation_summary():
    """Test investigation summary generation."""
    from root_cause_engine import AutoRootCauseEngine
    
    drift_report = {
        "findings": {
            "per_feature": [
                {"feature": "age", "psi": 0.30, "ks_stat": 0.20, "status": "DRIFT"},
            ],
            "num_drifted_features": 1
        }
    }
    
    engine = AutoRootCauseEngine(verbose=False)
    result = engine.run(drift_report=drift_report)
    summary = engine.generate_investigation_summary(result)
    
    assert "INVESTIGATION SUMMARY" in summary
    assert "Model Health:" in summary
    assert "Confidence:" in summary
    assert "Evidence:" in summary
    assert "Most Likely Causes:" in summary
    assert "Recommended Actions:" in summary
    print("[PASS] Investigation summary test passed")


def test_json_structure():
    """Test that result has correct JSON structure."""
    from root_cause_engine import AutoRootCauseEngine
    import json
    
    drift_report = {
        "findings": {
            "per_feature": [
                {"feature": "age", "psi": 0.30, "ks_stat": 0.20, "status": "DRIFT"},
            ],
            "num_drifted_features": 1
        }
    }
    
    engine = AutoRootCauseEngine(verbose=False)
    result = engine.run(drift_report=drift_report)
    
    # Verify JSON serializable
    json_str = json.dumps(result)
    assert json_str is not None
    
    # Verify required keys
    assert "health_status" in result
    assert "confidence" in result
    assert "root_causes" in result
    assert "recommended_actions" in result
    # evidence_summary may be in metadata or as evidence list
    assert "evidence" in result or "metadata" in result
    assert "generated_at" in result
    assert "module" in result
    
    print("[PASS] JSON structure test passed")


def test_health_status_calculation():
    """Test health status calculation logic."""
    from root_cause_engine import AutoRootCauseEngine
    
    engine = AutoRootCauseEngine(verbose=False)
    
    # Test no causes
    status, confidence = engine._calculate_health_status([])
    assert status == "Healthy"
    assert confidence == 100
    
    # Test low score causes
    low_causes = [
        {"score": 20, "severity": "LOW"},
        {"score": 25, "severity": "LOW"}
    ]
    status, confidence = engine._calculate_health_status(low_causes)
    assert status == "Healthy"
    
    # Test medium score causes
    medium_causes = [
        {"score": 40, "severity": "MEDIUM"},
        {"score": 45, "severity": "MEDIUM"}
    ]
    status, confidence = engine._calculate_health_status(medium_causes)
    assert status == "Warning"
    
    # Test high score causes
    high_causes = [
        {"score": 70, "severity": "HIGH"},
        {"score": 80, "severity": "CRITICAL"}
    ]
    status, confidence = engine._calculate_health_status(high_causes)
    assert status == "Critical"
    assert confidence > 60
    
    print("[PASS] Health status calculation test passed")


def test_investigation_object():
    """Test Investigation object creation and methods."""
    from investigation import Investigation, RootCause
    
    # Create investigation
    inv = Investigation(
        health_status="Warning",
        confidence=75,
        evidence=["Evidence 1", "Evidence 2"],
        recommendations=["Action 1", "Action 2"]
    )
    
    # Add root causes
    inv.add_root_cause(
        cause="Test cause",
        score=80,
        severity="HIGH",
        evidence=["Test evidence"],
        category="test",
        source_modules=["test_engine"]
    )
    
    # Test methods
    assert len(inv.root_causes) == 1
    assert inv.investigation_id is not None
    assert inv.health_status == "Warning"
    assert inv.confidence == 75
    
    # Test to_dict
    inv_dict = inv.to_dict()
    assert "investigation_id" in inv_dict
    assert "health_status" in inv_dict
    assert "root_causes" in inv_dict
    
    # Test from_dict
    inv2 = Investigation.from_dict(inv_dict)
    assert inv2.health_status == inv.health_status
    assert len(inv2.root_causes) == len(inv.root_causes)
    
    print("[PASS] Investigation object test passed")


def test_investigation_service():
    """Test InvestigationService."""
    from investigation_service import InvestigationService
    
    service = InvestigationService(verbose=False)
    
    # Test with valid reports
    drift_report = {
        "findings": {
            "per_feature": [
                {"feature": "age", "psi": 0.30, "ks_stat": 0.20, "status": "DRIFT"},
            ],
            "num_drifted_features": 1
        }
    }
    
    investigation = service.run_investigation(drift_report=drift_report)
    assert investigation is not None
    assert investigation.health_status in ["Healthy", "Warning", "Critical"]
    assert investigation.investigation_id is not None
    
    # Test get_investigation_summary
    summary = service.get_investigation_summary(investigation)
    assert "INVESTIGATION SUMMARY" in summary
    assert investigation.investigation_id in summary
    
    print("[PASS] Investigation service test passed")


def test_shap_importance_drift():
    """Test SHAP importance drift detection."""
    from root_cause_engine import AutoRootCauseEngine
    
    engine = AutoRootCauseEngine(verbose=False)
    
    # Training and production importance
    training_importance = {
        "age": 0.12,
        "income": 0.25,
        "education": 0.15
    }
    
    production_importance = {
        "age": 0.31,  # Significant change
        "income": 0.24,
        "education": 0.16
    }
    
    # Detect drift
    drift = engine._detect_importance_drift(training_importance, production_importance, threshold=0.10)
    
    assert len(drift) >= 1
    assert any(d["feature"] == "age" for d in drift)
    assert drift[0]["change"] >= 0.10
    
    print("[PASS] SHAP importance drift test passed")


def test_shap_importance_drift_integration():
    """Test SHAP importance drift integration in full run."""
    from root_cause_engine import AutoRootCauseEngine
    
    engine = AutoRootCauseEngine(verbose=False)
    
    # Training and production importance
    training_importance = {
        "age": 0.12,
        "income": 0.25
    }
    
    production_importance = {
        "age": 0.35,  # Significant change
        "income": 0.26
    }
    
    result = engine.run(
        training_importance=training_importance,
        production_importance=production_importance
    )
    
    assert result is not None
    assert "root_causes" in result
    
    # Check if importance drift was detected
    importance_causes = [c for c in result["root_causes"] if c.get("category") == "importance_drift"]
    assert len(importance_causes) >= 1
    
    print("[PASS] SHAP importance drift integration test passed")


def test_source_modules_traceability():
    """Test source_modules field in root causes."""
    from root_cause_engine import AutoRootCauseEngine
    
    engine = AutoRootCauseEngine(verbose=False)
    
    drift_report = {
        "findings": {
            "per_feature": [
                {"feature": "age", "psi": 0.30, "ks_stat": 0.20, "status": "DRIFT"},
            ],
            "num_drifted_features": 1
        }
    }
    
    result = engine.run(drift_report=drift_report)
    
    # Check source_modules field
    for cause in result["root_causes"]:
        assert "source_modules" in cause
        assert isinstance(cause["source_modules"], list)
        assert len(cause["source_modules"]) > 0
    
    print("[PASS] Source modules traceability test passed")


def test_edge_case_handling():
    """Test edge case handling (empty reports, malformed data)."""
    from root_cause_engine import AutoRootCauseEngine
    
    engine = AutoRootCauseEngine(verbose=False)
    
    # Test with None reports
    result = engine.run(
        drift_report=None,
        slice_report=None,
        calibration_report=None,
        data_quality_report=None
    )
    
    assert result is not None
    assert result["health_status"] == "Healthy"
    assert len(result["root_causes"]) == 0
    
    # Test with malformed reports
    malformed_report = {"invalid": "data"}
    result = engine.run(drift_report=malformed_report)
    
    assert result is not None
    # Should not crash, just handle gracefully
    
    # Test with empty lists
    empty_report = {"findings": {"per_feature": []}}
    result = engine.run(drift_report=empty_report)
    
    assert result is not None
    
    print("[PASS] Edge case handling test passed")


def test_context_aware_recommendations():
    """Test context-aware recommendations."""
    from root_cause_engine import AutoRootCauseEngine
    
    engine = AutoRootCauseEngine(verbose=False)
    
    drift_report = {
        "findings": {
            "per_feature": [
                {"feature": "age", "psi": 0.30, "ks_stat": 0.20, "status": "DRIFT"},
            ],
            "num_drifted_features": 1
        }
    }
    
    result = engine.run(drift_report=drift_report)
    
    # Check recommendations include feature name and evidence
    recommendations = result.get("recommended_actions", [])
    assert len(recommendations) > 0
    
    # Recommendations should be context-aware
    for rec in recommendations:
        assert len(rec) > 20  # Should be detailed, not generic
    
    print("[PASS] Context-aware recommendations test passed")


def run_all_tests():
    """Run all Auto Root Cause Engine tests."""
    print("=" * 65)
    print("Auto Root Cause Engine Tests (Phase 1.5)")
    print("=" * 65)
    
    try:
        test_high_drift_scenario()
        test_slice_failure_scenario()
        test_calibration_shift_scenario()
        test_mixed_failure_scenario()
        test_no_failure_scenario()
        test_scoring_rules()
        test_investigation_summary()
        test_json_structure()
        test_health_status_calculation()
        test_investigation_object()
        test_investigation_service()
        test_shap_importance_drift()
        test_shap_importance_drift_integration()
        test_source_modules_traceability()
        test_edge_case_handling()
        test_context_aware_recommendations()
        
        print("\n" + "=" * 65)
        print("All Auto Root Cause Engine tests PASSED [OK]")
        print("=" * 65)
        return True
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
