"""
Verification script for identifier column exclusion in drift_engine and root_cause_engine.
Tests that identifier columns (id, *_id, studentid, userid, customerid, recordid) are excluded from analysis.
"""

import sys
import os

# Add project root to path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine.modules.drift_engine import DriftEngine
from engine.modules.root_cause_engine import AutoRootCauseEngine

def test_drift_engine_identifier_exclusion():
    """Test that drift engine excludes identifier columns."""
    print("=" * 70)
    print("VERIFICATION: Drift Engine Identifier Column Exclusion")
    print("=" * 70)
    
    # Create reference data with identifier columns
    feature_names = ["StudentID", "UserID", "Attendance", "StudyTime", "Grade"]
    X_reference = [
        [1, 101, 85.5, 10.2, 90],
        [2, 102, 90.0, 12.5, 95],
        [3, 103, 78.5, 8.0, 85],
        [4, 104, 92.0, 15.0, 92],
        [5, 105, 88.0, 11.5, 88],
    ]
    
    # Create production data with drift in non-identifier columns
    X_current = [
        [6, 106, 75.0, 9.0, 80],  # Attendance and StudyTime drifted
        [7, 107, 70.0, 7.5, 78],
        [8, 108, 65.0, 6.0, 75],
        [9, 109, 80.0, 10.0, 82],
        [10, 110, 72.0, 8.5, 79],
    ]
    
    de = DriftEngine(verbose=True)
    de.set_reference(X_reference, feature_names)
    result = de.run(X_current)
    
    print("\n--- Test Results ---")
    print(f"Excluded features: {result['findings'].get('excluded_features', [])}")
    print(f"Audit log: {result['findings'].get('audit_log', [])}")
    
    # Verify identifier columns are excluded
    excluded = result['findings'].get('excluded_features', [])
    expected_excluded = ["StudentID", "UserID"]
    
    success = True
    for feature in expected_excluded:
        if feature in excluded:
            print(f"[PASS] {feature} correctly excluded from drift analysis")
        else:
            print(f"[FAIL] {feature} NOT excluded from drift analysis")
            success = False
    
    # Verify non-identifier columns are analyzed
    per_feature = result['findings'].get('per_feature', [])
    analyzed_features = [f['feature'] for f in per_feature]
    
    for feature in ["Attendance", "StudyTime", "Grade"]:
        if feature in analyzed_features:
            print(f"[PASS] {feature} correctly analyzed")
        else:
            print(f"[FAIL] {feature} NOT analyzed")
            success = False
    
    # Verify StudentID and UserID are NOT in per_feature results
    for feature in ["StudentID", "UserID"]:
        if feature not in analyzed_features:
            print(f"[PASS] {feature} NOT in per_feature results")
        else:
            print(f"[FAIL] {feature} incorrectly appears in per_feature results")
            success = False
    
    return success

def test_root_cause_engine_identifier_exclusion():
    """Test that root cause engine excludes identifier columns."""
    print("\n" + "=" * 70)
    print("VERIFICATION: Root Cause Engine Identifier Column Exclusion")
    print("=" * 70)
    
    # Create drift report with identifier columns
    drift_report = {
        "findings": {
            "per_feature": [
                {"feature": "StudentID", "status": "DRIFT", "psi": 0.5, "ks_stat": 0.3},
                {"feature": "UserID", "status": "WARN", "psi": 0.15, "ks_stat": 0.12},
                {"feature": "Attendance", "status": "DRIFT", "psi": 0.45, "ks_stat": 0.28},
                {"feature": "StudyTime", "status": "WARN", "psi": 0.12, "ks_stat": 0.10},
            ],
            "drifted": ["StudentID", "Attendance"],
            "warned": ["UserID", "StudyTime"],
        }
    }
    
    # Create importance dictionaries with identifier columns
    training_importance = {
        "StudentID": 0.05,
        "UserID": 0.03,
        "Attendance": 0.25,
        "StudyTime": 0.30,
        "Grade": 0.37,
    }
    
    production_importance = {
        "StudentID": 0.15,  # Drifted
        "UserID": 0.10,     # Drifted
        "Attendance": 0.20,  # Drifted
        "StudyTime": 0.25,  # Drifted
        "Grade": 0.30,
    }
    
    arce = AutoRootCauseEngine(verbose=True)
    result = arce.run(
        drift_report=drift_report,
        training_importance=training_importance,
        production_importance=production_importance
    )
    
    print("\n--- Test Results ---")
    print(f"Excluded features: {result.get('excluded_features', [])}")
    print(f"Audit log: {result.get('audit_log', [])}")
    
    # Check root causes
    root_causes = result.get('root_causes', [])
    cause_features = [cause['cause'] for cause in root_causes]
    
    print(f"\nRoot causes: {cause_features}")
    
    success = True
    
    # Verify identifier columns are excluded
    excluded = result.get('excluded_features', [])
    expected_excluded = ["StudentID", "UserID"]
    
    for feature in expected_excluded:
        if feature in excluded:
            print(f"[PASS] {feature} correctly excluded from root cause analysis")
        else:
            print(f"[FAIL] {feature} NOT excluded from root cause analysis")
            success = False
    
    # Verify StudentID and UserID are NOT in root causes
    for feature in ["StudentID", "UserID"]:
        if feature not in cause_features:
            print(f"[PASS] {feature} NOT in root causes")
        else:
            print(f"[FAIL] {feature} incorrectly appears in root causes")
            success = False
    
    # Verify non-identifier columns can still appear in root causes
    non_identifier_found = False
    for cause in cause_features:
        if "Attendance" in cause or "StudyTime" in cause:
            non_identifier_found = True
            break
    
    if non_identifier_found:
        print("[PASS] Non-identifier columns correctly included in root causes")
    else:
        print("[FAIL] Non-identifier columns NOT included in root causes")
        success = False
    
    return success

def test_identifier_detection_patterns():
    """Test various identifier column naming patterns."""
    print("\n" + "=" * 70)
    print("VERIFICATION: Identifier Column Detection Patterns")
    print("=" * 70)
    
    from engine.modules.drift_engine import DriftEngine
    from engine.modules.root_cause_engine import AutoRootCauseEngine
    
    de = DriftEngine(verbose=False)
    arce = AutoRootCauseEngine(verbose=False)
    
    # Test various identifier patterns
    test_cases = [
        ("id", True, "Direct 'id'"),
        ("StudentID", True, "Direct 'StudentID'"),
        ("UserID", True, "Direct 'UserID'"),
        ("customerid", True, "Direct 'customerid' (lowercase)"),
        ("recordid", True, "Direct 'recordid'"),
        ("user_id", True, "Pattern '*_id'"),
        ("student_id", True, "Pattern '*_id'"),
        ("customer_id", True, "Pattern '*_id'"),
        ("id_number", True, "Pattern 'id_*'"),
        ("id_code", True, "Pattern 'id_*'"),
        ("Attendance", False, "Non-identifier"),
        ("StudyTime", False, "Non-identifier"),
        ("Grade", False, "Non-identifier"),
        ("Name", False, "Non-identifier"),
        ("Age", False, "Non-identifier"),
    ]
    
    success = True
    for feature_name, expected_identifier, description in test_cases:
        result_de = de._is_identifier_column(feature_name)
        result_arce = arce._is_identifier_column(feature_name)
        
        if result_de == expected_identifier and result_arce == expected_identifier:
            print(f"[PASS] {description}: '{feature_name}' -> {result_de}")
        else:
            print(f"[FAIL] {description}: '{feature_name}' -> DE={result_de}, ARCE={result_arce}, expected={expected_identifier}")
            success = False
    
    return success

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("IDENTIFIER COLUMN EXCLUSION VERIFICATION")
    print("=" * 70)
    
    test1 = test_drift_engine_identifier_exclusion()
    test2 = test_root_cause_engine_identifier_exclusion()
    test3 = test_identifier_detection_patterns()
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    if test1 and test2 and test3:
        print("[SUCCESS] All verification tests passed!")
        sys.exit(0)
    else:
        print("[FAILURE] Some verification tests failed!")
        sys.exit(1)
