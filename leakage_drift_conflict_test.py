"""
Leakage vs Drift Conflict Resolution Test
========================================
Test the conflict resolution logic to ensure leakage supersedes drift.
"""

import sys
import os

# Add project root to path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine.modules.root_cause_engine import AutoRootCauseEngine

def test_leakage_drift_conflict():
    """Test that leakage supersedes drift in conflict resolution."""
    print("=" * 80)
    print("LEAKAGE VS DRIFT CONFLICT RESOLUTION TEST")
    print("=" * 80)
    
    # Create test reports
    # RiskFlag appears in both leakage and drift
    drift_report = {
        "findings": {
            "per_feature": [
                {
                    "feature": "RiskFlag",
                    "psi": 0.30,
                    "ks_stat": 0.25,
                    "status": "DRIFT"
                },
                {
                    "feature": "Attendance",
                    "psi": 0.15,
                    "ks_stat": 0.10,
                    "status": "DRIFT"
                }
            ]
        }
    }
    
    leakage_report = {
        "findings": {
            "suspects": [
                {
                    "feature": "RiskFlag",
                    "leakage_confidence": 0.85
                }
            ]
        }
    }
    
    # Run root cause engine with both reports
    rc_engine = AutoRootCauseEngine(verbose=True)
    result = rc_engine.run(
        drift_report=drift_report,
        leakage_report=leakage_report
    )
    
    print("\n" + "=" * 80)
    print("ROOT CAUSE ANALYSIS RESULTS")
    print("=" * 80)
    
    root_causes = result.get('root_causes', [])
    
    print(f"\nTotal root causes found: {len(root_causes)}")
    
    for i, cause in enumerate(root_causes, 1):
        print(f"\n{i}. {cause.get('cause', 'Unknown')}")
        print(f"   Score: {cause.get('score', 0)}")
        print(f"   Severity: {cause.get('severity', 'Unknown')}")
        print(f"   Category: {cause.get('category', 'Unknown')}")
        print(f"   Evidence: {cause.get('evidence', [])}")
    
    recommendations = result.get('recommended_actions', [])
    print(f"\nRecommended Actions:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    # Check audit log for conflict resolution
    audit_log = result.get('audit_log', [])
    print(f"\nAudit Log:")
    for log in audit_log:
        print(f"  {log}")
    
    # Validation
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)
    
    # Check that RiskFlag appears only as leakage, not as drift
    riskflag_causes = [c for c in root_causes if "RiskFlag" in c.get('cause', '')]
    
    if len(riskflag_causes) == 1:
        cause = riskflag_causes[0]
        if cause.get('category') == 'target_leakage':
            print("[PASS] RiskFlag appears only as target leakage")
            print(f"      Category: {cause.get('category')}")
            print(f"      Severity: {cause.get('severity')}")
        else:
            print("[FAIL] RiskFlag appears but not as target leakage")
            print(f"      Category: {cause.get('category')}")
    elif len(riskflag_causes) == 0:
        print("[FAIL] RiskFlag does not appear in root causes")
    else:
        print("[FAIL] RiskFlag appears multiple times:")
        for cause in riskflag_causes:
            print(f"      - {cause.get('cause')} ({cause.get('category')})")
    
    # Check that Attendance still appears as drift
    attendance_causes = [c for c in root_causes if "Attendance" in c.get('cause', '')]
    if attendance_causes:
        cause = attendance_causes[0]
        if cause.get('category') == 'feature_drift':
            print("[PASS] Attendance appears as feature drift")
            print(f"      Category: {cause.get('category')}")
        else:
            print("[WARN] Attendance appears but not as feature drift")
            print(f"      Category: {cause.get('category')}")
    
    # Check that leakage has higher score than drift
    if riskflag_causes and attendance_causes:
        leakage_score = riskflag_causes[0].get('score', 0)
        drift_score = attendance_causes[0].get('score', 0)
        if leakage_score > drift_score:
            print("[PASS] Leakage score ({leakage_score}) > drift score ({drift_score})")
        else:
            print("[FAIL] Leakage score ({leakage_score}) not > drift score ({drift_score})")
    
    # Check recommendations for leakage-specific action
    leakage_rec_found = any("Exclude" in rec and "RiskFlag" in rec for rec in recommendations)
    if leakage_rec_found:
        print("[PASS] Leakage-specific recommendation found")
    else:
        print("[FAIL] Leakage-specific recommendation not found")
    
    # Check that drift recommendation for RiskFlag is NOT present
    riskflag_drift_rec = any("RiskFlag" in rec and "distribution shifted" in rec for rec in recommendations)
    if not riskflag_drift_rec:
        print("[PASS] RiskFlag drift recommendation removed")
    else:
        print("[FAIL] RiskFlag drift recommendation still present")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_leakage_drift_conflict()
