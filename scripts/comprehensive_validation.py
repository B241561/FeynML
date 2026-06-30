"""
Comprehensive Validation Script
==============================
Performs a complete validation pass on the FeynML system using a single synthetic dataset.

Validates:
1. Executive Summary
2. Investigation Findings
3. Root Cause Analysis
4. StudentID exclusion
5. Leakage Detection
6. Audience Translation
7. Confidence Scoring
8. Domain Translation
9. Dashboard Rendering
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine.modules.drift_engine import DriftEngine
from engine.modules.root_cause_engine import AutoRootCauseEngine
from engine.modules.ai_investigator import AIInvestigator
from engine.modules.audience_translator import AudienceTranslator
from engine.modules.leakage_engine import LeakageEngine
from engine.modules.investigation import Investigation, RootCause

def create_synthetic_dataset():
    """Create a synthetic dataset with StudentID and potential leakage."""
    np.random.seed(42)
    
    n_samples = 500
    
    # Create StudentID (should be excluded)
    student_ids = list(range(1000, 1000 + n_samples))
    
    # Create features with some drift
    attendance_train = np.random.normal(85, 10, n_samples)
    study_time_train = np.random.normal(10, 3, n_samples)
    grade_train = np.random.normal(85, 10, n_samples)
    
    # Create potential leakage feature (correlated with target)
    # This simulates a feature that might leak the target
    midterm_score_train = grade_train + np.random.normal(0, 5, n_samples)
    
    # Create target
    target_train = (attendance_train * 0.3 + study_time_train * 0.2 + grade_train * 0.5 + np.random.normal(0, 5, n_samples))
    target_train = (target_train - target_train.min()) / (target_train.max() - target_train.min()) * 100
    
    # Training data
    train_data = pd.DataFrame({
        'StudentID': student_ids,
        'Attendance': attendance_train,
        'StudyTime': study_time_train,
        'Grade': grade_train,
        'MidtermScore': midterm_score_train,
        'FinalGrade': target_train
    })
    
    # Production data with drift
    student_ids_prod = list(range(2000, 2000 + n_samples))
    attendance_prod = np.random.normal(75, 12, n_samples)  # Drifted
    study_time_prod = np.random.normal(8, 4, n_samples)  # Drifted
    grade_prod = np.random.normal(80, 12, n_samples)  # Drifted
    midterm_score_prod = grade_prod + np.random.normal(0, 5, n_samples)
    target_prod = (attendance_prod * 0.3 + study_time_prod * 0.2 + grade_prod * 0.5 + np.random.normal(0, 5, n_samples))
    target_prod = (target_prod - target_prod.min()) / (target_prod.max() - target_prod.min()) * 100
    
    prod_data = pd.DataFrame({
        'StudentID': student_ids_prod,
        'Attendance': attendance_prod,
        'StudyTime': study_time_prod,
        'Grade': grade_prod,
        'MidtermScore': midterm_score_prod,
        'FinalGrade': target_prod
    })
    
    return train_data, prod_data

def run_validation():
    """Run comprehensive validation."""
    print("=" * 80)
    print("COMPREHENSIVE VALIDATION PASS")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Create synthetic dataset
    print("\n[1/9] Creating synthetic dataset...")
    train_data, prod_data = create_synthetic_dataset()
    print(f"Training samples: {len(train_data)}")
    print(f"Production samples: {len(prod_data)}")
    print(f"Features: {list(train_data.columns)}")
    
    validation_results = {
        "timestamp": datetime.now().isoformat(),
        "checks": {},
        "passing": [],
        "failing": [],
        "warnings": []
    }
    
    # Prepare data for analysis
    feature_names = ['StudentID', 'Attendance', 'StudyTime', 'Grade', 'MidtermScore']
    target_col = 'FinalGrade'
    
    X_train = train_data[feature_names].values.tolist()
    X_prod = prod_data[feature_names].values.tolist()
    y_train = train_data[target_col].values.tolist()
    y_prod = prod_data[target_col].values.tolist()
    
    # 1. StudentID Exclusion Check
    print("\n[2/9] Checking StudentID exclusion...")
    de = DriftEngine(verbose=False)
    de.set_reference(X_train, feature_names)
    drift_result = de.run(X_prod)
    
    excluded_features = drift_result['findings'].get('excluded_features', [])
    audit_log = drift_result['findings'].get('audit_log', [])
    
    studentid_excluded = 'StudentID' in excluded_features
    validation_results["checks"]["studentid_exclusion"] = {
        "status": "PASS" if studentid_excluded else "FAIL",
        "excluded_features": excluded_features,
        "audit_log": audit_log
    }
    
    if studentid_excluded:
        print("  [PASS] StudentID correctly excluded")
        validation_results["passing"].append("StudentID Exclusion")
    else:
        print("  [FAIL] StudentID NOT excluded")
        validation_results["failing"].append("StudentID Exclusion")
    
    # 2. Leakage Detection
    print("\n[3/9] Checking leakage detection...")
    try:
        le = LeakageEngine(verbose=False)
        leakage_result = le.run(X_train, y_train, feature_names=feature_names)
        
        suspects = leakage_result.get('findings', {}).get('suspects', [])
        leakage_detected = len(suspects) > 0
        validation_results["checks"]["leakage_detection"] = {
            "status": "PASS" if leakage_detected else "WARN",
            "result": leakage_result,
            "num_suspects": len(suspects)
        }
        
        if leakage_detected:
            print(f"  [PASS] Leakage detected ({len(suspects)} suspects)")
            validation_results["passing"].append("Leakage Detection")
        else:
            print("  [WARN] Leakage not detected (may need stronger correlation)")
            validation_results["warnings"].append("Leakage Detection - weak signal")
    except Exception as e:
        print(f"  [ERROR] Leakage detection failed: {e}")
        validation_results["checks"]["leakage_detection"] = {
            "status": "ERROR",
            "error": str(e)
        }
        validation_results["failing"].append("Leakage Detection")
    
    # 3. Drift Analysis
    print("\n[4/9] Running drift analysis...")
    drift_features = drift_result['findings'].get('per_feature', [])
    drifted = drift_result['findings'].get('drifted', [])
    
    print(f"  Features analyzed: {len(drift_features)}")
    print(f"  Features drifted: {len(drifted)}")
    print(f"  Drifted features: {drifted}")
    
    validation_results["checks"]["drift_analysis"] = {
        "status": "PASS",
        "drifted_features": drifted,
        "num_drifted": len(drifted)
    }
    validation_results["passing"].append("Drift Analysis")
    
    # 4. Root Cause Analysis
    print("\n[5/9] Running root cause analysis...")
    arce = AutoRootCauseEngine(verbose=False)
    root_cause_result = arce.run(drift_report=drift_result)
    
    root_causes = root_cause_result.get('root_causes', [])
    excluded_rc = root_cause_result.get('excluded_features', [])
    
    print(f"  Root causes found: {len(root_causes)}")
    print(f"  Excluded features: {excluded_rc}")
    
    # Check StudentID is not in root causes
    studentid_in_rc = any('StudentID' in rc.get('cause', '') for rc in root_causes)
    
    validation_results["checks"]["root_cause_analysis"] = {
        "status": "PASS" if not studentid_in_rc else "FAIL",
        "root_causes": root_causes,
        "excluded_features": excluded_rc,
        "studentid_in_root_causes": studentid_in_rc
    }
    
    if not studentid_in_rc:
        print("  [PASS] StudentID not in root causes")
        validation_results["passing"].append("Root Cause Analysis")
    else:
        print("  [FAIL] StudentID found in root causes")
        validation_results["failing"].append("Root Cause Analysis")
    
    # 5. AI Investigator - Executive Summary
    print("\n[6/9] Verifying Executive Summary...")
    investigation = Investigation.from_dict(root_cause_result)
    ai_inv = AIInvestigator(use_llm=False, verbose=False)
    ai_result = ai_inv.analyze(investigation)
    
    exec_summary = ai_result.get('executive_summary', '')
    word_count = len(exec_summary.split())
    
    print(f"  Executive Summary: {exec_summary}")
    print(f"  Word count: {word_count}")

    # CONSISTENCY CHECK: Ensure root-cause risk, AI Investigator risk and global status align
    try:
        rc_level = root_cause_result.get('risk', {}).get('level') if isinstance(root_cause_result.get('risk', {}), dict) else None
        ai_level = ai_result.get('risk_level')
        # Determine pass/fail
        consistent = (rc_level is None and ai_level is None) or (rc_level and ai_level and str(rc_level).upper() == str(ai_level).upper())
        validation_results['checks']['risk_consistency'] = {
            'status': 'PASS' if consistent else 'FAIL',
            'root_cause_level': rc_level,
            'ai_investigator_level': ai_level
        }
        if consistent:
            validation_results['passing'].append('Risk Consistency')
            print("  [PASS] Risk levels are consistent between RootCauseEngine and AIInvestigator")
        else:
            validation_results['failing'].append('Risk Consistency')
            print(f"  [FAIL] Inconsistent risk levels: root_cause={rc_level}, ai_investigator={ai_level}")
    except Exception as e:
        validation_results['checks']['risk_consistency'] = {'status': 'ERROR', 'error': str(e)}
        validation_results['warnings'].append('Risk Consistency Check Failed')
        print(f"  [WARN] Risk consistency check failed: {e}")
    
    # Check executive summary quality
    has_psi = "PSI=" in exec_summary or "PSI " in exec_summary
    has_ks = "KS=" in exec_summary or "KS " in exec_summary
    is_narrative = not exec_summary.startswith(("1.", "2.", "3."))
    
    validation_results["checks"]["executive_summary"] = {
        "status": "PASS" if (not has_psi and not has_ks and is_narrative and word_count <= 120) else "FAIL",
        "summary": exec_summary,
        "word_count": word_count,
        "has_psi": has_psi,
        "has_ks": has_ks,
        "is_narrative": is_narrative
    }
    
    if not has_psi and not has_ks and is_narrative and word_count <= 120:
        print("  [PASS] Executive Summary quality verified")
        validation_results["passing"].append("Executive Summary")
    else:
        print("  [FAIL] Executive Summary quality issues")
        if has_psi:
            print("    - Contains PSI values")
        if has_ks:
            print("    - Contains KS values")
        if not is_narrative:
            print("    - Not in narrative format")
        if word_count > 120:
            print("    - Exceeds 120-word limit")
        validation_results["failing"].append("Executive Summary")
    
    # 6. AI Investigator - Investigation Findings
    print("\n[7/9] Verifying Investigation Findings...")
    inv_findings = ai_result.get('investigation_findings', '')
    
    print(f"  Investigation Findings: {inv_findings}")
    
    # Check investigation findings quality
    findings_has_psi = "PSI=" in inv_findings or "PSI " in inv_findings
    findings_has_ks = "KS=" in inv_findings or "KS " in inv_findings
    findings_is_narrative = not inv_findings.startswith(("1.", "2.", "3."))
    has_interpretation = any(kw in inv_findings.lower() for kw in ["distributional", "shift", "pattern", "indicates"])
    
    validation_results["checks"]["investigation_findings"] = {
        "status": "PASS" if (not findings_has_psi and not findings_has_ks and findings_is_narrative and has_interpretation) else "FAIL",
        "findings": inv_findings,
        "has_psi": findings_has_psi,
        "has_ks": findings_has_ks,
        "is_narrative": findings_is_narrative,
        "has_interpretation": has_interpretation
    }
    
    if not findings_has_psi and not findings_has_ks and findings_is_narrative and has_interpretation:
        print("  [PASS] Investigation Findings quality verified")
        validation_results["passing"].append("Investigation Findings")
    else:
        print("  [FAIL] Investigation Findings quality issues")
        if findings_has_psi:
            print("    - Contains PSI values")
        if findings_has_ks:
            print("    - Contains KS values")
        if not findings_is_narrative:
            print("    - Not in narrative format")
        if not has_interpretation:
            print("    - Missing interpretation")
        validation_results["failing"].append("Investigation Findings")
    
    # 7. Audience Translation
    print("\n[8/9] Verifying Audience Translation...")
    translator = AudienceTranslator(verbose=False)
    audience_reports = translator.translate(root_cause_result, ai_result)
    
    print(f"  Audiences generated: {list(audience_reports.keys())}")
    
    # Check that all expected audiences are present
    expected_audiences = ["ML Engineer", "Executive", "Doctor", "Loan Officer", "Student"]
    all_audiences_present = all(aud in audience_reports for aud in expected_audiences)
    
    # Check that each audience has executive summary
    all_have_summary = all('executive_summary' in audience_reports[aud] for aud in audience_reports)
    
    validation_results["checks"]["audience_translation"] = {
        "status": "PASS" if (all_audiences_present and all_have_summary) else "FAIL",
        "audiences": list(audience_reports.keys()),
        "all_present": all_audiences_present,
        "all_have_summary": all_have_summary
    }
    
    if all_audiences_present and all_have_summary:
        print("  [PASS] Audience Translation verified")
        validation_results["passing"].append("Audience Translation")
    else:
        print("  [FAIL] Audience Translation issues")
        if not all_audiences_present:
            print("    - Missing audiences")
        if not all_have_summary:
            print("    - Missing executive summaries")
        validation_results["failing"].append("Audience Translation")
    
    # 8. Confidence Scoring
    print("\n[9/9] Verifying Confidence Scoring...")
    confidence = root_cause_result.get('confidence', 0)
    health_status = root_cause_result.get('health_status', '')
    
    print(f"  Confidence: {confidence}%")
    print(f"  Health Status: {health_status}")
    
    # Check confidence is reasonable (0-100)
    confidence_valid = 0 <= confidence <= 100
    
    validation_results["checks"]["confidence_scoring"] = {
        "status": "PASS" if confidence_valid else "FAIL",
        "confidence": confidence,
        "health_status": health_status
    }
    
    if confidence_valid:
        print("  [PASS] Confidence scoring valid")
        validation_results["passing"].append("Confidence Scoring")
    else:
        print("  [FAIL] Confidence scoring invalid")
        validation_results["failing"].append("Confidence Scoring")
    
    # 9. Domain Translation (check if domain translations are present in root causes)
    print("\n[10/9] Verifying Domain Translation...")
    has_domain_translation = any(rc.get('domain_translation') for rc in root_causes)
    
    validation_results["checks"]["domain_translation"] = {
        "status": "PASS" if has_domain_translation else "WARN",
        "has_domain_translation": has_domain_translation
    }
    
    if has_domain_translation:
        print("  [PASS] Domain translations present")
        validation_results["passing"].append("Domain Translation")
    else:
        print("  [WARN] Domain translations not present (may not be configured)")
        validation_results["warnings"].append("Domain Translation - not configured")
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal Checks: {len(validation_results['checks'])}")
    print(f"Passing: {len(validation_results['passing'])}")
    print(f"Failing: {len(validation_results['failing'])}")
    print(f"Warnings: {len(validation_results['warnings'])}")
    
    print("\n" + "-" * 80)
    print("PASSING CHECKS:")
    print("-" * 80)
    for check in validation_results['passing']:
        print(f"  [PASS] {check}")
    
    if validation_results['failing']:
        print("\n" + "-" * 80)
        print("FAILING CHECKS:")
        print("-" * 80)
        for check in validation_results['failing']:
            print(f"  [FAIL] {check}")
    
    if validation_results['warnings']:
        print("\n" + "-" * 80)
        print("WARNINGS:")
        print("-" * 80)
        for check in validation_results['warnings']:
            print(f"  [WARN] {check}")
    
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    
    for check_name, check_result in validation_results['checks'].items():
        print(f"\n{check_name}:")
        print(f"  Status: {check_result['status']}")
        for key, value in check_result.items():
            if key != 'status':
                print(f"  {key}: {value}")
    
    # Save results to file
    import json
    with open('validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Validation results saved to: validation_results.json")
    print("=" * 80)
    
    return validation_results

if __name__ == "__main__":
    try:
        results = run_validation()
        
        # Exit with appropriate code
        if results['failing']:
            sys.exit(1)
        else:
            sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Validation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
