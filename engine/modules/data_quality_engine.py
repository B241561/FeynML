"""
Engine Module — Data Quality Engine
====================================
Unified data quality reporting for ML Failure Investigation.

Provides comprehensive data quality assessment covering:
  • Missing values (rate, pattern)
  • Duplicates
  • Constant/near-constant features
  • High-cardinality categorical features
  • Invalid values
  • Outliers (statistical)
  • Target quality (imbalance, invalid values)

Integrates with existing validator.py and missing_data_engine.py where appropriate.

Usage:
    from engine.modules.data_quality_engine import DataQualityEngine
    
    engine = DataQualityEngine()
    report = engine.analyze(df, target_col="target")
"""

import sys
import os
from typing import Dict, List, Optional, Any, Set
from collections import Counter
import numpy as np
import pandas as pd

try:
    from engine.base_module import BaseModule
except ImportError:
    from base_module import BaseModule


class DataQualityEngine(BaseModule):
    """
    Unified data quality engine for ML Failure Investigation.
    
    Provides comprehensive data quality assessment to identify
    issues that could affect analysis reliability.
    """
    
    def __init__(self, verbose: bool = True, 
                 missing_threshold: float = 0.30,
                 constant_threshold: float = 0.01,
                 cardinality_threshold: int = 100):
        """
        Initialize the data quality engine.
        
        Args:
            verbose: Enable logging
            missing_threshold: Threshold for high missingness (default 30%)
            constant_threshold: Threshold for near-constant features (variance)
            cardinality_threshold: Threshold for high cardinality in categorical features
        """
        super().__init__(verbose=verbose)
        self.missing_threshold = missing_threshold
        self.constant_threshold = constant_threshold
        self.cardinality_threshold = cardinality_threshold
    
    def analyze(self, df, target_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze data quality.
        
        Args:
            df: Input dataset (pandas DataFrame)
            target_col: Name of target column for target-specific checks
        
        Returns:
            Structured data quality report
        """
        self._log("Starting data quality analysis...")
        
        # Import pandas if available
        try:
            import pandas as pd
        except ImportError:
            return self._result({
                "overall_status": "ERROR",
                "missingness": {},
                "duplicates": {},
                "constant_features": [],
                "cardinality": {},
                "outliers": {},
                "target_quality": {},
                "warnings": ["pandas is required for data quality analysis"],
                "evidence": []
            }, severity="CRITICAL", module_name="DataQualityEngine")
        
        # Convert to DataFrame if needed
        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except Exception as e:
                return self._result({
                    "overall_status": "ERROR",
                    "missingness": {},
                    "duplicates": {},
                    "constant_features": [],
                    "cardinality": {},
                    "outliers": {},
                    "target_quality": {},
                    "warnings": [f"Cannot convert input to DataFrame: {str(e)}"],
                    "evidence": []
                }, severity="CRITICAL", module_name="DataQualityEngine")
        
        warnings = []
        evidence = []
        
        # 1. Missingness analysis
        missingness = self._analyze_missingness(df)
        if missingness["high_missingness"]:
            warnings.append(f"High missingness features: {missingness['high_missingness']}")
            evidence.append(f"{len(missingness['high_missingness'])} features exceed {self.missing_threshold:.0%} missingness")
        
        # 2. Duplicate analysis
        duplicates = self._analyze_duplicates(df)
        if duplicates["duplicate_count"] > 0:
            warnings.append(f"Found {duplicates['duplicate_count']} duplicate rows ({duplicates['duplicate_rate']:.2%})")
            evidence.append(f"Duplicate rate: {duplicates['duplicate_rate']:.2%}")
        
        # 3. Constant/near-constant features
        constant_features = self._analyze_constant_features(df)
        if constant_features:
            warnings.append(f"Constant or near-constant features: {constant_features}")
            evidence.append(f"{len(constant_features)} features have low variance")
        
        # 4. Cardinality analysis
        cardinality = self._analyze_cardinality(df)
        high_cardinality = cardinality.get("high_cardinality", [])
        if high_cardinality:
            warnings.append(f"High cardinality categorical features: {high_cardinality}")
            evidence.append(f"{len(high_cardinality)} categorical features have > {self.cardinality_threshold} unique values")
        
        # 5. Outlier analysis (numerical features only)
        outliers = self._analyze_outliers(df)
        if outliers["features_with_outliers"]:
            warnings.append(f"Features with outliers: {outliers['features_with_outliers']}")
            evidence.append(f"{len(outliers['features_with_outliers'])} features contain statistical outliers")
        
        # 6. Target quality
        target_quality = {}
        if target_col and target_col in df.columns:
            target_quality = self._analyze_target_quality(df[target_col])
            if target_quality["issues"]:
                warnings.extend(target_quality["issues"])
                evidence.extend(target_quality["evidence"])
        
        # Determine overall status
        if len(warnings) == 0:
            overall_status = "GOOD"
            severity = "NONE"
        elif len(warnings) <= 2:
            overall_status = "ACCEPTABLE"
            severity = "LOW"
        elif len(warnings) <= 4:
            overall_status = "WARNING"
            severity = "MEDIUM"
        else:
            overall_status = "POOR"
            severity = "HIGH"
        
        self._log(f"Data quality analysis complete: {overall_status}")
        
        return self._result({
            "overall_status": overall_status,
            "missingness": missingness,
            "duplicates": duplicates,
            "constant_features": constant_features,
            "cardinality": cardinality,
            "outliers": outliers,
            "target_quality": target_quality,
            "warnings": warnings,
            "evidence": evidence
        }, severity=severity, module_name="DataQualityEngine")
    
    def _analyze_missingness(self, df) -> Dict[str, Any]:
        """Analyze missing values per feature."""
        missing_rates = {}
        high_missingness = []
        
        for col in df.columns:
            missing_rate = df[col].isna().sum() / len(df)
            missing_rates[col] = round(missing_rate, 4)
            
            if missing_rate > self.missing_threshold:
                high_missingness.append(col)
        
        return {
            "missing_rates": missing_rates,
            "high_missingness": high_missingness,
            "total_missing": df.isna().sum().sum(),
            "total_cells": len(df) * len(df.columns)
        }
    
    def _analyze_duplicates(self, df) -> Dict[str, Any]:
        """Analyze duplicate rows."""
        duplicate_count = df.duplicated().sum()
        duplicate_rate = duplicate_count / len(df) if len(df) > 0 else 0
        
        return {
            "duplicate_count": int(duplicate_count),
            "duplicate_rate": round(duplicate_rate, 4),
            "unique_rows": len(df) - duplicate_count
        }
    
    def _analyze_constant_features(self, df) -> List[str]:
        """Identify constant or near-constant features."""
        constant_features = []
        
        for col in df.columns:
            try:
                # Skip non-numeric columns for variance check
                if not pd.api.types.is_numeric_dtype(df[col]):
                    # Check if categorical column has only one unique value
                    if df[col].nunique() == 1:
                        constant_features.append(col)
                    continue
                
                # Check variance for numeric columns
                if df[col].var() < self.constant_threshold:
                    constant_features.append(col)
            except Exception:
                # If variance calculation fails, check unique values
                if df[col].nunique() == 1:
                    constant_features.append(col)
        
        return constant_features
    
    def _analyze_cardinality(self, df) -> Dict[str, Any]:
        """Analyze cardinality of categorical features."""
        cardinality = {}
        high_cardinality = []
        
        for col in df.columns:
            unique_count = df[col].nunique()
            cardinality[col] = unique_count
            
            # Consider high cardinality if categorical and many unique values
            if not pd.api.types.is_numeric_dtype(df[col]) and unique_count > self.cardinality_threshold:
                high_cardinality.append(col)
        
        return {
            "cardinality": cardinality,
            "high_cardinality": high_cardinality
        }
    
    def _analyze_outliers(self, df) -> Dict[str, Any]:
        """Analyze outliers in numerical features using IQR method."""
        features_with_outliers = []
        outlier_counts = {}
        
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            try:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR == 0:
                    continue
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_count = len(outliers)
                
                if outlier_count > 0:
                    features_with_outliers.append(col)
                    outlier_counts[col] = outlier_count
            except Exception:
                continue
        
        return {
            "features_with_outliers": features_with_outliers,
            "outlier_counts": outlier_counts
        }
    
    def _analyze_target_quality(self, target_series) -> Dict[str, Any]:
        """Analyze target column quality."""
        issues = []
        evidence = []
        
        # Check for missing values
        missing_rate = target_series.isna().sum() / len(target_series)
        if missing_rate > 0:
            issues.append(f"Target has {missing_rate:.2%} missing values")
            evidence.append(f"Target missingness: {missing_rate:.2%}")
        
        # Check for class imbalance (if categorical)
        unique_count = target_series.nunique()
        if unique_count <= 10:  # Likely classification
            value_counts = target_series.value_counts()
            minority_frac = value_counts.min() / len(target_series)
            
            if minority_frac < 0.05:
                issues.append(f"Severe class imbalance: minority class is {minority_frac:.2%} of data")
                evidence.append(f"Minority class fraction: {minority_frac:.2%}")
        
        # Check for invalid values (e.g., negative where not expected)
        # This is context-dependent, so we just report the distribution
        return {
            "unique_count": int(unique_count),
            "missing_rate": round(missing_rate, 4),
            "value_counts": target_series.value_counts().to_dict(),
            "issues": issues,
            "evidence": evidence
        }
    
    def _run(self, *args, **kwargs):
        """Override to use analyze method."""
        return self.analyze(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("pandas required for smoke test")
        sys.exit(1)
    
    # Dataset with quality issues
    df = pd.DataFrame({
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0],  # outlier
        "feature2": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        "constant": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],  # constant
        "missing_col": [1, None, 3, None, 5, None, 7, None, 9, None]  # 40% missing
    })
    
    # Add duplicate
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    
    engine = DataQualityEngine(verbose=True)
    
    print("\n=== Data Quality Analysis ===")
    result = engine.analyze(df, target_col="target")
    print(f"Overall status: {result['findings']['overall_status']}")
    print(f"Warnings: {result['findings']['warnings']}")
    print(f"Evidence: {result['findings']['evidence']}")
    
    print("\n✓ DataQualityEngine OK")
