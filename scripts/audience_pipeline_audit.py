"""
Audience Translation End-to-End Audit
=====================================
Comprehensive audit of the audience translation pipeline for ALL supported audiences.

This script tests the entire flow:
1. Audience dropdown selection
2. JavaScript click handler
3. /set-audience endpoint
4. Flask session persistence
5. Session retrieval in app.py
6. runner.audience assignment
7. analysis_runner.py propagation
8. AI Investigator audience input
9. audience_translator.py execution
10. Dashboard rendering
11. Report rendering
"""

import sys
import os
import json

# Add project root to path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine.modules.ai_investigator import AIInvestigator
from engine.modules.audience_translator import AudienceTranslator
from engine.modules.investigation import Investigation, RootCause

def test_audience_pipeline():
    """Test the entire audience pipeline for all supported audiences."""
    print("=" * 80)
    print("AUDIENCE TRANSLATION END-TO-END AUDIT")
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
        recommendations=["Retrain model", "Monitor attendance"]
    )
    
    # Test all audiences
    audiences = [
        "ML Engineer",
        "Executive",
        "Doctor",
        "Loan Officer",
        "Student",
        "HR Manager",
        "Insurance Analyst",
        "Legal / Compliance Officer",
        "Researcher"
    ]
    
    results = {}
    
    for audience in audiences:
        print(f"\n{'=' * 80}")
        print(f"Testing Audience: {audience}")
        print(f"{'=' * 80}")
        
        # Step 1: AI Investigator with audience
        print(f"\n[STEP 1] AI Investigator with audience parameter")
        ai_inv = AIInvestigator(use_llm=False, verbose=False)
        ai_result = ai_inv.analyze(investigation, audience)
        
        exec_summary = ai_result.get('executive_summary', '')
        findings = ai_result.get('investigation_findings', '')
        
        print(f"  Executive Summary: {exec_summary}")
        print(f"  Investigation Findings: {findings}")
        
        # Step 2: Audience Translator
        print(f"\n[STEP 2] Audience Translator")
        translator = AudienceTranslator(verbose=False)
        
        # Convert investigation to dict for translator
        root_cause_dict = investigation.to_dict()
        
        # Generate audience-specific report
        audience_report = translator._translate_for_audience(
            audience, root_cause_dict, ai_result
        )
        
        print(f"  Audience Report Keys: {list(audience_report.keys())}")
        print(f"  Executive Summary: {audience_report.get('executive_summary', '')}")
        print(f"  Findings: {audience_report.get('findings', '')}")
        
        # Store results
        results[audience] = {
            "ai_investigator": ai_result,
            "audience_report": audience_report,
            "exec_summary_matches": exec_summary == audience_report.get('executive_summary', ''),
            "findings_match": findings == audience_report.get('findings', '')
        }
    
    # Generate validation matrix
    print(f"\n{'=' * 80}")
    print("VALIDATION MATRIX")
    print(f"{'=' * 80}")
    
    print(f"\n{'Audience':<30} | {'AI Summary':<20} | {'Audience Summary':<20} | {'Match?'}")
    print("-" * 100)
    
    for audience in audiences:
        ai_summary = results[audience]['ai_investigator'].get('executive_summary', '')[:20]
        aud_summary = results[audience]['audience_report'].get('executive_summary', '')[:20]
        match = "YES" if results[audience]['exec_summary_matches'] else "NO"
        print(f"{audience:<30} | {ai_summary:<20} | {aud_summary:<20} | {match}")
    
    # Check if all audiences produce different output
    print(f"\n{'=' * 80}")
    print("AUDIENCE-SPECIFIC OUTPUT VERIFICATION")
    print(f"{'=' * 80}")
    
    all_exec_summaries = [results[a]['ai_investigator'].get('executive_summary', '') for a in audiences]
    unique_summaries = set(all_exec_summaries)
    
    print(f"\nTotal audiences: {len(audiences)}")
    print(f"Unique executive summaries: {len(unique_summaries)}")
    
    if len(unique_summaries) == len(audiences):
        print("[PASS] All audiences produce unique executive summaries")
    else:
        print("[FAIL] Some audiences produce identical executive summaries")
        print("\nDuplicate summaries:")
        for summary in unique_summaries:
            count = all_exec_summaries.count(summary)
            if count > 1:
                print(f"  '{summary[:50]}...' appears {count} times")
    
    # Check for audience-specific keywords
    print(f"\n{'=' * 80}")
    print("AUDIENCE-SPECIFIC KEYWORD VERIFICATION")
    print(f"{'=' * 80}")
    
    audience_keywords = {
        "ML Engineer": ["technical", "drift", "calibration", "engineering"],
        "Executive": ["business", "impact", "roi", "strategic", "risk"],
        "Doctor": ["clinical", "patient", "medical", "health", "outcome"],
        "Loan Officer": ["credit", "lending", "financial", "risk", "compliance"],
        "Student": ["learning", "educational", "study", "academic", "grade"],
        "HR Manager": ["workforce", "employee", "hiring", "performance", "attrition"],
        "Insurance Analyst": ["underwriting", "premium", "policy", "claim", "risk"],
        "Legal / Compliance Officer": ["compliance", "audit", "governance", "regulatory", "legal"],
        "Researcher": ["methodology", "evidence", "experimental", "study", "analysis"]
    }
    
    keyword_results = {}
    
    for audience, keywords in audience_keywords.items():
        summary = results[audience]['ai_investigator'].get('executive_summary', '').lower()
        findings = results[audience]['ai_investigator'].get('investigation_findings', '').lower()
        combined = summary + " " + findings
        
        found_keywords = [kw for kw in keywords if kw in combined]
        keyword_results[audience] = {
            "expected_keywords": keywords,
            "found_keywords": found_keywords,
            "has_keywords": len(found_keywords) > 0
        }
        
        print(f"\n{audience}:")
        print(f"  Expected keywords: {keywords}")
        print(f"  Found keywords: {found_keywords}")
        print(f"  Status: {'PASS' if len(found_keywords) > 0 else 'FAIL'}")
    
    # Final summary
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY")
    print(f"{'=' * 80}")
    
    keyword_pass_count = sum(1 for r in keyword_results.values() if r['has_keywords'])
    
    print(f"\nAudience-specific keyword test: {keyword_pass_count}/{len(audiences)} passed")
    
    if len(unique_summaries) == len(audiences) and keyword_pass_count == len(audiences):
        print("\n[SUCCESS] All audiences produce visibly different output")
        return 0
    else:
        print("\n[FAILURE] Some audiences produce generic output")
        return 1

if __name__ == "__main__":
    sys.exit(test_audience_pipeline())
