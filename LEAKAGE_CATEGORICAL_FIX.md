# LEAKAGE_CATEGORICAL_FIX: FeynML ML Failure Investigation Engine
**Date:** 2026-06-02
**Status:** RESOLVED & VERIFIED

## 🔍 Root Cause
The `LeakageEngine` was crashing with a `ValueError: could not convert string to float: 'yes'` during the preprocessing phase of the `scan()` function. This occurred because the engine attempted to pass categorical string data (e.g., 'yes'/'no', 'male'/'female') directly into numerical routines such as `pandas.DataFrame.corr()` and `sklearn.feature_selection.mutual_info_classif`, which require numeric inputs.

## 🛠️ Fix Implemented
I have introduced a robust **Categorical Factorization Layer** in the `scan()` function of `leakage_scanner.py`.

**Key Changes:**
- **Automated Type Detection**: The engine now automatically identifies non-numeric columns in the input feature set.
- **Categorical Factorization**: Non-numeric columns are factorized into integer codes using `pandas.factorize()`.
- **NaN Preservation**: The factorization process explicitly preserves `NaN` values, ensuring that missing data integrity is maintained for downstream statistical tests.
- **Unified Processing**: The factorized data is used for both Target Correlation scans and Mutual Information scoring, while original feature names are preserved for reporting.

## 🧪 Verification Results
The fix was verified using a synthetic dataset containing categorical strings:
- **Test Data**: Features including `gender` (male/female), `married` (yes/no), and `target_copy` (yes/no).
- **Result**: 
  - **No Crashes**: The engine successfully processed the categorical strings.
  - **Leakage Detected**: The engine correctly identified `target_copy` as a high-severity leakage suspect based on its factorized correlation with the target variable.
  - **Severity**: Successfully assigned **HIGH** severity to the categorical leakage suspect.

---
**Verdict:** The Leakage Engine is now robust against categorical string data and can accurately detect leakage in mixed-type datasets.
