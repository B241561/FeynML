"""
Report Deduplication Audit
==========================
Audit the current report generation to identify duplicated information across sections.
"""

import sys
import os

# Add project root to path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine.modules.ai_investigator import AIInvestigator
from engine.modules.investigation import Investigation, RootCause

def audit_report_deduplication():
    """Audit report for duplicate information across sections."""
    print("=" * 80)
    print("REPORT DEDUPLICATION AUDIT")
    print("=" * 80)
    
    # Create test investigation
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
    ]
    
    investigation = Investigation(
        health_status="Warning",
        confidence=80,
        root_causes=causes,
        recommendations=["Retrain model", "Monitor attendance", "Collect recent samples"]
    )
    
    # Generate report
    ai_inv = AIInvestigator(use_llm=False, verbose=False)
    report = ai_inv.analyze(investigation, audience="ML Engineer")
    
    # Extract sections
    exec_summary = report.get('executive_summary', '')
    findings = report.get('investigation_findings', '')
    impact = report.get('impact_assessment', '')
    confidence = report.get('confidence_explanation', '')
    technical = report.get('technical_notes', '')
    recommendations = report.get('recommended_actions', [])
    
    print("\n" + "=" * 80)
    print("CURRENT REPORT SECTIONS")
    print("=" * 80)
    
    print("\n[1] EXECUTIVE SUMMARY")
    print("-" * 80)
    print(exec_summary)
    print(f"Word count: {len(exec_summary.split())}")
    
    print("\n[2] INVESTIGATION FINDINGS")
    print("-" * 80)
    print(findings)
    print(f"Word count: {len(findings.split())}")
    
    print("\n[3] IMPACT ASSESSMENT")
    print("-" * 80)
    print(impact)
    print(f"Word count: {len(impact.split())}")
    
    print("\n[4] CONFIDENCE EXPLANATION")
    print("-" * 80)
    print(confidence)
    print(f"Word count: {len(confidence.split())}")
    
    print("\n[5] TECHNICAL NOTES")
    print("-" * 80)
    print(technical)
    print(f"Word count: {len(technical.split())}")
    
    print("\n[6] RECOMMENDED ACTIONS")
    print("-" * 80)
    for i, action in enumerate(recommendations, 1):
        print(f"{i}. {action}")
    
    # Check for duplicates
    print("\n" + "=" * 80)
    print("DUPLICATE ANALYSIS")
    print("=" * 80)
    
    all_text = {
        "Executive Summary": exec_summary,
        "Investigation Findings": findings,
        "Impact Assessment": impact,
        "Confidence Explanation": confidence,
        "Technical Notes": technical
    }
    
    # Check for shared phrases
    print("\nShared phrase analysis:")
    for section1, text1 in all_text.items():
        for section2, text2 in all_text.items():
            if section1 >= section2:
                continue
            
            # Find common words (3+ word phrases)
            words1 = text1.lower().split()
            words2 = text2.lower().split()
            
            common_phrases = []
            for i in range(len(words1) - 2):
                phrase = ' '.join(words1[i:i+3])
                if phrase in ' '.join(words2):
                    common_phrases.append(phrase)
            
            if common_phrases:
                print(f"\n{section1} <-> {section2}:")
                print(f"  Common phrases: {set(common_phrases)}")
    
    # Check for feature names in Executive Summary
    print("\n" + "=" * 80)
    print("FEATURE-LEVEL DETECTION")
    print("=" * 80)
    
    feature_names = ["Attendance", "StudyTime", "Grade", "MidtermScore"]
    found_features = [f for f in feature_names if f.lower() in exec_summary.lower()]
    
    if found_features:
        print(f"[FAIL] Executive Summary contains feature-level details: {found_features}")
    else:
        print("[PASS] Executive Summary does not contain feature-level details")
    
    # Check for PSI/KS in Executive Summary
    if "PSI" in exec_summary or "KS" in exec_summary:
        print("[FAIL] Executive Summary contains PSI/KS values")
    else:
        print("[PASS] Executive Summary does not contain PSI/KS values")
    
    # Check for recommendations in Investigation Findings
    recommendation_keywords = ["recommend", "should", "must", "required", "action", "step"]
    findings_lower = findings.lower()
    found_rec_keywords = [kw for kw in recommendation_keywords if kw in findings_lower]
    
    if found_rec_keywords:
        print(f"[WARN] Investigation Findings may contain recommendation language: {found_rec_keywords}")
    else:
        print("[PASS] Investigation Findings does not contain recommendation language")
    
    # Check for root cause names in multiple sections
    print("\n" + "=" * 80)
    print("ROOT CAUSE DUPLICATION CHECK")
    print("=" * 80)
    
    root_cause_names = ["Attendance feature drift", "StudyTime feature drift"]
    
    for rc_name in root_cause_names:
        sections_with_rc = []
        if rc_name.lower() in exec_summary.lower():
            sections_with_rc.append("Executive Summary")
        if rc_name.lower() in findings.lower():
            sections_with_rc.append("Investigation Findings")
        if rc_name.lower() in impact.lower():
            sections_with_rc.append("Impact Assessment")
        if rc_name.lower() in technical.lower():
            sections_with_rc.append("Technical Notes")
        
        if len(sections_with_rc) > 1:
            print(f"[FAIL] '{rc_name}' appears in multiple sections: {sections_with_rc}")
        else:
            print(f"[PASS] '{rc_name}' appears only in: {sections_with_rc if sections_with_rc else 'Root Cause Analysis'}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nCurrent issues identified:")
    print("1. Executive Summary may contain feature-level details")
    print("2. Investigation Findings may contain recommendation language")
    print("3. Root cause names may appear in multiple sections")
    print("4. Impact Assessment may duplicate information from Findings")

if __name__ == "__main__":
    audit_report_deduplication()
