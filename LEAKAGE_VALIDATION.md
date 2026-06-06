# LEAKAGE_VALIDATION: FeynML ML Failure Investigation Engine
**Date:** 2026-06-02
**Status:** VALIDATED

## 🔍 Audit Summary
The Leakage Engine was previously failing to report target leakage suspects when a model was not provided (e.g., during the initial diagnostic run). This was due to a conditional gate in the orchestration logic that restricted target leakage scoring to model-aware sessions only.

## 🧪 Test Results (Synthetic Data)

| Test Case | Description | Detected | Severity | Confidence |
|:---|:---|:---:|:---:|:---|
| **Test A** | Target Copied Directly | ✅ YES | **HIGH** | 100% |
| **Test B** | Target Scaled (x 0.99) | ✅ YES | **HIGH** | 100% |
| **Test C** | Target + Small Noise | ✅ YES | **HIGH** | 98% |
| **Test D** | Control (No Leakage) | ❌ NO | **NONE** | 100% |

## 📊 Detection Metrics
- **Thresholds**: 
  - Score > 0.8: **HIGH**
  - Score > 0.5: **MEDIUM**
- **Formula**: `0.7 * Correlation + 0.3 * Normalized MI`
- **Result**: Even without a trained model, the engine now correctly identifies features with extreme mutual information or correlation with the target.

## 🛠️ Root Cause & Fix
- **Root Cause**: The `scan()` function in `leakage_scanner.py` had an `if model is not None:` block wrapping the `leakage_score` calculation.
- **Fix**: Decoupled `leakage_score` (MI + Correlation) from the model requirement. Permutation Importance spikes still require a model, but statistical leakage detection now runs in all sessions.

---
**Verdict:** Leakage detection is now fully operational for both early-stage diagnostics and post-training model audits.
