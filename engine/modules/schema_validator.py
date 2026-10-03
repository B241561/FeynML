"""
Engine Module — Schema Validator
=================================
Centralized schema validation for incoming datasets.

Validates dataset structure before downstream engines execute.

Responsibilities:
  • Dataset existence and readability
  • Dataset non-empty check
  • Column existence and validity
  • Duplicate column detection
  • Required column availability (target, prediction)
  • Data type detection (numerical, categorical, boolean, datetime)
  • Unsupported/inconsistent type detection

Usage:
    from engine.modules.schema_validator import SchemaValidator
    
    validator = SchemaValidator()
    result = validator.validate(df, target_col="target", prediction_col="prediction")
    
    if result["status"] == "INVALID":
        # Handle validation failure
        pass
"""

import sys
import os
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

try:
    from engine.base_module import BaseModule
except ImportError:
    from base_module import BaseModule


class SchemaValidator(BaseModule):
    """
    Centralized schema validation for ML Failure Investigation datasets.
    
    Validates dataset structure before engines execute to prevent
    downstream failures and provide clear error messages.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the schema validator.
        
        Args:
            verbose: Enable logging
        """
        super().__init__(verbose=verbose)
    
    def validate(self, df, target_col: Optional[str] = None, 
                 prediction_col: Optional[str] = None,
                 probability_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate dataset schema.
        
        Args:
            df: Input dataset (pandas DataFrame or dict-like)
            target_col: Name of target column (required for some analyses)
            prediction_col: Name of prediction column (required for some analyses)
            probability_col: Name of probability column (optional)
        
        Returns:
            Structured validation result with status, errors, warnings, and metadata
        """
        self._log("Starting schema validation...")
        
        errors = []
        warnings = []
        detected_types = {}
        missing_required = []
        
        # Import pandas if available
        try:
            import pandas as pd
        except ImportError:
            return self._result({
                "status": "INVALID",
                "errors": ["pandas is required for schema validation"],
                "warnings": warnings,
                "detected_types": {},
                "missing_required_columns": [],
                "columns": []
            }, severity="CRITICAL", module_name="SchemaValidator")
        
        # Convert to DataFrame if needed
        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except Exception as e:
                errors.append(f"Cannot convert input to DataFrame: {str(e)}")
                return self._result({
                    "status": "INVALID",
                    "errors": errors,
                    "warnings": warnings,
                    "detected_types": {},
                    "missing_required_columns": [],
                    "columns": []
                }, severity="CRITICAL", module_name="SchemaValidator")
        
        # Check if dataset is empty
        if df.empty:
            errors.append("Dataset is empty (0 rows)")
            return self._result({
                "status": "INVALID",
                "errors": errors,
                "warnings": warnings,
                "detected_types": {},
                "missing_required_columns": [],
                "columns": []
            }, severity="CRITICAL", module_name="SchemaValidator")
        
        # Check if dataset has columns
        if len(df.columns) == 0:
            errors.append("Dataset has no columns")
            return self._result({
                "status": "INVALID",
                "errors": errors,
                "warnings": warnings,
                "detected_types": {},
                "missing_required_columns": [],
                "columns": []
            }, severity="CRITICAL", module_name="SchemaValidator")
        
        # Check for duplicate column names
        columns = list(df.columns)
        column_counts = {}
        for col in columns:
            column_counts[col] = column_counts.get(col, 0) + 1
        
        duplicates = [col for col, count in column_counts.items() if count > 1]
        if duplicates:
            errors.append(f"Duplicate column names found: {duplicates}")
        
        # Validate column names (basic check for valid identifiers)
        invalid_col_names = []
        for col in columns:
            if not isinstance(col, str):
                invalid_col_names.append(f"{col} (non-string)")
            elif not col.strip():
                invalid_col_names.append(f"'{col}' (empty or whitespace)")
        
        if invalid_col_names:
            warnings.append(f"Column names may be problematic: {invalid_col_names[:5]}")
        
        # Check required columns
        if target_col is not None and target_col not in columns:
            missing_required.append(target_col)
            errors.append(f"Required target column '{target_col}' not found")
        
        if prediction_col is not None and prediction_col not in columns:
            missing_required.append(prediction_col)
            errors.append(f"Required prediction column '{prediction_col}' not found")
        
        if probability_col is not None and probability_col not in columns:
            missing_required.append(probability_col)
            warnings.append(f"Optional probability column '{probability_col}' not found")
        
        # Detect data types for each column
        for col in columns:
            detected_types[col] = self._detect_column_type(df[col])
        
        # Check for unsupported types
        unsupported_cols = []
        for col, dtype in detected_types.items():
            if dtype == "unsupported":
                unsupported_cols.append(col)
        
        if unsupported_cols:
            warnings.append(f"Columns with unsupported or mixed types: {unsupported_cols[:5]}")
        
        # Determine overall status
        if errors:
            status = "INVALID"
            severity = "CRITICAL"
        elif warnings:
            status = "WARNING"
            severity = "MEDIUM"
        else:
            status = "VALID"
            severity = "NONE"
        
        self._log(f"Schema validation complete: {status}")
        
        return self._result({
            "status": status,
            "errors": errors,
            "warnings": warnings,
            "detected_types": detected_types,
            "missing_required_columns": missing_required,
            "columns": columns,
            "n_rows": len(df),
            "n_columns": len(columns)
        }, severity=severity, module_name="SchemaValidator")
    
    def _detect_column_type(self, series) -> str:
        """
        Detect the data type of a column.
        
        Args:
            series: pandas Series
        
        Returns:
            One of: numerical, categorical, boolean, datetime, unsupported
        """
        # Check for boolean first (before datetime check)
        try:
            unique_vals = series.dropna().unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({True, False, 0, 1, 'True', 'False', 'true', 'false', 'T', 'F', 't', 'f'}):
                return "boolean"
        except Exception:
            pass
        
        # Check for numerical first (before datetime to avoid false positives)
        try:
            import pandas as pd
            if pd.api.types.is_numeric_dtype(series):
                return "numerical"
        except Exception:
            pass
        
        # Check for datetime (only if not already numeric)
        try:
            import pandas as pd
            if pd.api.types.is_datetime64_any_dtype(series):
                return "datetime"
            # Try to convert to datetime
            converted = pd.to_datetime(series, errors='coerce')
            # If conversion succeeds for most values, treat as datetime
            if converted.notna().sum() / len(series) > 0.8:
                return "datetime"
        except Exception:
            pass
        
        # Check for categorical
        try:
            unique_vals = series.dropna().unique()
            # If relatively few unique values compared to total, likely categorical
            if len(unique_vals) <= 100 or len(unique_vals) <= len(series) * 0.1:
                return "categorical"
        except Exception:
            pass
        
        # Default to unsupported
        return "unsupported"
    
    def _run(self, *args, **kwargs):
        """Override to use validate method."""
        return self.validate(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("pandas required for smoke test")
        sys.exit(1)
    
    # Valid dataset
    df_valid = pd.DataFrame({
        "target": [0, 1, 0, 1],
        "feature1": [1.0, 2.0, 3.0, 4.0],
        "feature2": ["A", "B", "A", "B"]
    })
    
    # Empty dataset
    df_empty = pd.DataFrame()
    
    # Missing target
    df_missing_target = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0],
        "feature2": ["A", "B", "A", "B"]
    })
    
    validator = SchemaValidator(verbose=True)
    
    print("\n=== Valid Dataset ===")
    result = validator.validate(df_valid, target_col="target")
    print(f"Status: {result['findings']['status']}")
    
    print("\n=== Empty Dataset ===")
    result = validator.validate(df_empty, target_col="target")
    print(f"Status: {result['findings']['status']}")
    
    print("\n=== Missing Target ===")
    result = validator.validate(df_missing_target, target_col="target")
    print(f"Status: {result['findings']['status']}")
    print(f"Errors: {result['findings']['errors']}")
    
    print("\n✓ SchemaValidator OK")
