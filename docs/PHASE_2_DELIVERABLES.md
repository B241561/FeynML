# FeynML Phase 2 - AI Investigator Agent

## Deliverables Summary

### 1. Architecture Changes

```
┌─────────────────────────────────────────────────────────────────┐
                    DATASET UPLOAD                                │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
                    VALIDATION                                   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
              PHASE 2: DIAGNOSTICS                               │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │ CalibrationEngine│  │  FairnessEngine  │                    │
│  └──────────────────┘  └──────────────────┘                    │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
              PHASE 3: OBSERVABILITY                              │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │   DriftEngine    │  │   SlicerEngine   │                    │
│  └──────────────────┘  └──────────────────┘                    │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
         PHASE 4: ROOT CAUSE ANALYSIS                             │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │ LabelNoiseEngine │  │  LeakageEngine   │                    │
│  └──────────────────┘  └──────────────────┘                    │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │MissingDataEngine │  │AutoRootCauseEng  │                    │
│  └──────────────────┘  └──────────────────┘                    │
│         │                      │                                  │
│         └──────────┬───────────┘                                  │
│                    ▼                                              │
│         ┌──────────────────────┐                                 │
│         │ Investigation Object │                                 │
│         └──────────────────────┘                                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
         PHASE 5: AI INVESTIGATION (NEW)                           │
│  ┌──────────────────────────────────────────────┐               │
│  │         AIInvestigator Agent                │               │
│  │  • Analyzes root causes                      │               │
│  │  • Generates narrative reasoning             │               │
│  │  • Assesses risk level                       │               │
│  │  • Creates executive summary                │               │
│  │  • Provides technical notes                  │               │
│  └──────────────────────────────────────────────┘               │
│         │                                                      │
│         ▼                                                      │
│  ┌──────────────────────────────────────────────┐               │
│  │  AI Investigation Report                     │               │
│  │  • Executive Summary                         │               │
│  │  • Investigation Findings                   │               │
│  │  • Impact Assessment                         │               │
│  │  • Confidence Explanation                    │               │
│  │  • Recommended Actions                      │               │
│  │  • Technical Notes                           │               │
│  └──────────────────────────────────────────────┘               │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
              REPORT GENERATION                                   │
│  • JSON Report (machine-readable)                               │
│  • HTML Report (browser-viewable)                               │
│  • AI Investigator Section (NEW)                                │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
              FLASK UI DISPLAY                                    │
│  • Dashboard                                                   │
│  • Report Page                                                 │
│  • Root Cause Card                                             │
│  • AI Investigator Card (NEW)                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Modified Files List

1. **engine/modules/report_engine.py**
   - Updated docstring to include AIInvestigator
   - Added ai_investigator example usage
   - Added AI Investigator section rendering with:
     - Risk level badge with icon
     - Executive summary display
     - Investigation findings in styled box
     - Impact assessment
     - Confidence explanation in styled box
     - Recommended actions as list
     - Technical notes in monospace box

2. **webapp/templates/report.html**
   - Added "AI Investigator Report" card after Root Cause section
   - Displays risk level with icon
   - Shows investigation ID
   - Renders executive summary
   - Shows investigation findings in styled box
   - Displays impact assessment
   - Shows confidence explanation in styled box
   - Lists recommended actions with icons
   - Displays technical notes in monospace box
   - Handles missing data gracefully

### 3. New Files List

1. **engine/modules/ai_investigator.py**
   - AIInvestigator class with full investigation analysis capabilities
   - Methods:
     - `analyze(investigation)` - Main analysis method
     - `_generate_deterministic(investigation)` - Fallback narrative generation
     - `_generate_with_llm(investigation)` - LLM integration (if available)
     - `_build_llm_prompt(investigation)` - Prompt engineering
     - `_assess_risk_level(investigation)` - Risk assessment logic
     - `_generate_executive_summary(investigation, risk_level)` - Business summary
     - `_generate_investigation_findings(investigation)` - Detailed findings
     - `_generate_impact_assessment(investigation)` - Impact analysis
     - `_generate_confidence_explanation(investigation)` - Confidence rationale
     - `_generate_technical_notes(investigation)` - Technical summary
     - `generate_investigation_report(investigation)` - Full text report
   - Features:
     - LLM integration with Gemini (optional)
     - Deterministic fallback (always available)
     - Graceful error handling
     - Edge case handling (missing data, empty evidence)

2. **tests/test_ai_investigator.py**
   - Comprehensive test suite for AIInvestigator
   - Test cases:
     - High risk investigation
     - Medium risk investigation
     - No failure investigation
     - Missing evidence
     - Missing root causes
     - LLM failure fallback
     - Investigation report generation
     - Risk assessment logic
     - Executive summary generation
   - All 9 tests passing

### 4. Example AI Investigation Output

```json
{
  "executive_summary": "The model shows signs of degradation that should be addressed. Primary issue appears to be distributional drift in age. Investigation confidence is high (87%) due to multiple supporting signals. Action should be taken soon to address the identified issues.",
  "investigation_findings": "The model degradation appears to be primarily driven by the following factors:\n\n1. age feature drift\n   Evidence:\n   • PSI=0.300, KS=0.200\n   Source: drift_engine\n\n2. Age Drift\n   Evidence:\n   • PSI=0.31\n   • Accuracy dropped 18%\n   Source: drift_engine",
  "impact_assessment": "Distributional drift may affect predictions for segments of the population that differ from training data. Specific customer segments may experience reduced prediction accuracy.\n\n2 high-severity issue(s) detected, indicating significant impact on model performance.",
  "confidence_explanation": "Investigation confidence is 87% based on:\n\n• Number of identified issues: 2\n• High-severity issues: 2\n• Evidence strength: Strong (multiple supporting signals)\n\nHigh confidence indicates consistent findings across multiple analysis engines.",
  "recommended_actions": [
    "age distribution shifted significantly (PSI=0.300, KS=0.200). Collect recent samples from the affected segment before retraining.",
    "Age Drift (PSI=0.31, Accuracy dropped 18%). Collect recent samples from the affected segment and retrain the model."
  ],
  "technical_notes": "Technical Summary:\n\nFEATURE_DRIFT:\n  • age feature drift (Score: 50, Severity: HIGH)\n  • Age Drift (Score: 87, Severity: CRITICAL)\n\nAnalysis Engines Used: drift_engine\n\nInvestigation ID: inv_abc123def456\nGenerated At: 2026-06-21T12:00:00.000000",
  "risk_level": "CRITICAL",
  "generated_at": "2026-06-21T12:00:00.000000",
  "investigation_id": "inv_abc123def456"
}
```

### 5. Example Report Section (HTML)

```html
<!-- AI Investigator Section -->
<div class="row g-4 mb-4">
    <div class="col-lg-12">
        <div class="card">
            <div class="card-header">AI Investigator Report</div>
            <div class="card-body p-4">
                <div class="row align-items-center mb-4">
                    <div class="col-md-6">
                        <div class="d-flex align-items-center">
                            <div class="h3 fw-bold mb-0 me-3">CRITICAL</div>
                            <div class="text-muted small">Risk Level</div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="text-muted small">
                            Investigation ID: inv_abc123def456
                        </div>
                    </div>
                </div>

                <h6 class="fw-bold small text-muted text-uppercase mb-3">Executive Summary</h6>
                <p class="mb-4">The model shows signs of degradation that should be addressed. Primary issue appears to be distributional drift in age. Investigation confidence is high (87%) due to multiple supporting signals. Action should be taken soon to address the identified issues.</p>

                <h6 class="fw-bold small text-muted text-uppercase mb-3">Investigation Findings</h6>
                <div class="bg-dark p-3 rounded mb-4" style="background-color: #1a1a2a !important;">
                    The model degradation appears to be primarily driven by the following factors:<br><br>
                    1. age feature drift<br>
                       Evidence:<br>
                       • PSI=0.300, KS=0.200<br>
                       Source: drift_engine<br><br>
                    2. Age Drift<br>
                       Evidence:<br>
                       • PSI=0.31<br>
                       • Accuracy dropped 18%<br>
                       Source: drift_engine
                </div>

                <h6 class="fw-bold small text-muted text-uppercase mb-3">Impact Assessment</h6>
                <p class="mb-4">Distributional drift may affect predictions for segments of the population that differ from training data. Specific customer segments may experience reduced prediction accuracy.<br><br>2 high-severity issue(s) detected, indicating significant impact on model performance.</p>

                <h6 class="fw-bold small text-muted text-uppercase mb-3">Confidence Explanation</h6>
                <div class="bg-dark p-3 rounded mb-4" style="background-color: #1a1a2a !important;">
                    Investigation confidence is 87% based on:<br><br>
                    • Number of identified issues: 2<br>
                    • High-severity issues: 2<br>
                    • Evidence strength: Strong (multiple supporting signals)<br><br>
                    High confidence indicates consistent findings across multiple analysis engines.
                </div>

                <h6 class="fw-bold small text-muted text-uppercase mb-3">Recommended Actions</h6>
                <ul class="list-group list-group-flush mb-4">
                    <li class="list-group-item">
                        <i class="bi bi-lightbulb-fill me-2 text-warning"></i>age distribution shifted significantly (PSI=0.300, KS=0.200). Collect recent samples from the affected segment before retraining.
                    </li>
                    <li class="list-group-item">
                        <i class="bi bi-lightbulb-fill me-2 text-warning"></i>Age Drift (PSI=0.31, Accuracy dropped 18%). Collect recent samples from the affected segment and retrain the model.
                    </li>
                </ul>

                <h6 class="fw-bold small text-muted text-uppercase mb-3">Technical Notes</h6>
                <div class="bg-dark p-3 rounded" style="background-color: #1a1a2a !important; font-family: monospace; font-size: 0.9em;">
                    Technical Summary:<br><br>
                    FEATURE_DRIFT:<br>
                      • age feature drift (Score: 50, Severity: HIGH)<br>
                      • Age Drift (Score: 87, Severity: CRITICAL)<br><br>
                    Analysis Engines Used: drift_engine<br><br>
                    Investigation ID: inv_abc123def456<br>
                    Generated At: 2026-06-21T12:00:00.000000
                </div>
            </div>
        </div>
    </div>
</div>
```

### 6. Example UI Rendering

The AI Investigator card appears in the Flask UI with:

- **Risk Level Badge**: Large, colored badge (CRITICAL in red, HIGH in orange, MEDIUM in yellow, LOW in green)
- **Investigation ID**: Small text showing the investigation identifier
- **Executive Summary**: Business-friendly 3-5 sentence summary
- **Investigation Findings**: Detailed findings in a dark-styled box with line breaks
- **Impact Assessment**: Clear impact description
- **Confidence Explanation**: Rationale for confidence score in a dark-styled box
- **Recommended Actions**: Bulleted list with lightbulb icons
- **Technical Notes**: Monospace font technical summary in a dark-styled box

All sections use existing styling patterns and are consistent with the rest of the application.

## Test Results

All 9 tests passed successfully:
- [PASS] High risk investigation test passed
- [PASS] Medium risk investigation test passed
- [PASS] No failure investigation test passed
- [PASS] Missing evidence test passed
- [PASS] Missing root causes test passed
- [PASS] LLM fallback test passed
- [PASS] Investigation report generation test passed
- [PASS] Risk assessment logic test passed
- [PASS] Executive summary generation test passed

## Key Features

1. **Dual-Mode Operation**: LLM-enhanced (if available) or deterministic fallback (always available)
2. **Risk Assessment**: Automated risk level calculation based on health status, severity, and confidence
3. **Business-Friendly Output**: Executive summaries written for non-technical stakeholders
4. **Technical Depth**: Technical notes provide ML-engineer-friendly details
5. **Edge Case Handling**: Graceful degradation for missing data, empty evidence, no failures
6. **Full Traceability**: All findings reference source modules and evidence
7. **Production Ready**: Comprehensive test coverage, error handling, and fallback mechanisms

## Integration Points

The AI Investigator integrates seamlessly with:
- **Investigation Object**: Consumes unified investigation from AutoRootCauseEngine
- **Report Engine**: New section in HTML and JSON reports
- **Flask UI**: Dedicated card with consistent styling
- **Investigation Service**: Can be called by monitoring systems

## Summary

Phase 2 successfully added an AI-powered investigation layer to FeynML that transforms raw technical findings into human-readable, business-friendly analysis. The system behaves like a senior ML engineer conducting a failure investigation, providing executive summaries, impact assessments, and actionable recommendations while maintaining full technical depth for engineering teams.
