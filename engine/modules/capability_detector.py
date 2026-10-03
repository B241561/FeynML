"""
Engine Module — Capability Detector
==================================
Centralized capability detection for ML Failure Investigation.

Determines what analyses a dataset can actually support before engines execute.

Responsibilities:
  • Classification vs regression detection
  • Binary vs multiclass classification detection
  • Temporal data detection
  • Categorical feature detection
  • Numerical feature detection
  • Target availability detection
  • Prediction availability detection
  • Probability availability detection
  • Reference/current dataset availability detection
  • Sufficient samples detection

Usage:
    from engine.modules.capability_detector import CapabilityDetector
    
    detector = CapabilityDetector()
    capabilities = detector.detect(df, target_col="target", prediction_col="prediction")
    
    if not capabilities["probabilities_available"]:
        # Skip calibration engine
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


class CapabilityDetector(BaseModule):
    """
    Centralized capability detection for ML Failure Investigation.
    
    Determines what analyses a dataset can support to prevent
    incompatible engines from executing.
    """
    
    def __init__(self, verbose: bool = True, min_samples: int = 30):
        """
        Initialize the capability detector.
        
        Args:
            verbose: Enable logging
            min_samples: Minimum samples required for reliable analysis
        """
        super().__init__(verbose=verbose)
        self.min_samples = min_samples
    
    def detect(self, df, target_col: Optional[str] = None,
               prediction_col: Optional[str] = None,
               probability_col: Optional[str] = None,
               reference_df=None) -> Dict[str, Any]:
        """
        Detect dataset capabilities.
        
        Args:
            df: Input dataset (pandas DataFrame)
            target_col: Name of target column
            prediction_col: Name of prediction column
            probability_col: Name of probability column
            reference_df: Reference dataset for drift/temporal analysis
        
        Returns:
            Structured capability report
        """
        self._log("Starting capability detection...")
        
        # Import pandas if available
        try:
            import pandas as pd
        except ImportError:
            return self._result({
                "capabilities": {},
                "limitations": ["pandas is required for capability detection"],
                "can_analyze": False
            }, severity="CRITICAL", module_name="CapabilityDetector")
        
        # Convert to DataFrame if needed
        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except Exception as e:
                return self._result({
                    "capabilities": {},
                    "limitations": [f"Cannot convert input to DataFrame: {str(e)}"],
                    "can_analyze": False
                }, severity="CRITICAL", module_name="CapabilityDetector")
        
        capabilities = {}
        limitations = []
        
        # Basic dataset checks
        n_samples = len(df)
        n_features = len(df.columns) - 1 if target_col and target_col in df.columns else len(df.columns)
        
        capabilities["sufficient_samples"] = n_samples >= self.min_samples
        if not capabilities["sufficient_samples"]:
            limitations.append(f"Insufficient samples: {n_samples} < {self.min_samples}")
        
        capabilities["n_samples"] = n_samples
        capabilities["n_features"] = n_features
        
        # Target availability
        capabilities["target_available"] = target_col is not None and target_col in df.columns
        if not capabilities["target_available"] and target_col:
            limitations.append(f"Target column '{target_col}' not found")
        
        # Prediction availability
        capabilities["predictions_available"] = prediction_col is not None and prediction_col in df.columns
        if not capabilities["predictions_available"] and prediction_col:
            limitations.append(f"Prediction column '{prediction_col}' not found")
        
        # Probability availability
        capabilities["probabilities_available"] = probability_col is not None and probability_col in df.columns
        if not capabilities["probabilities_available"] and probability_col:
            limitations.append(f"Probability column '{probability_col}' not found")
        
        # Reference dataset availability
        capabilities["reference_available"] = reference_df is not None and not reference_df.empty
        if not capabilities["reference_available"] and reference_df is not None:
            limitations.append("Reference dataset is empty or invalid")
        
        # Detect task type (classification vs regression)
        if capabilities["target_available"]:
            target_series = df[target_col].dropna()
            unique_targets = target_series.nunique()
            
            # Heuristic: if few unique values (and not continuous), likely classification
            if unique_targets <= 10 or (unique_targets <= n_samples * 0.05):
                capabilities["task_type"] = "classification"
                capabilities["classification"] = True
                capabilities["regression"] = False
                
                # Binary vs multiclass
                if unique_targets == 2:
                    capabilities["binary_classification"] = True
                    capabilities["multiclass_classification"] = False
                else:
                    capabilities["binary_classification"] = False
                    capabilities["multiclass_classification"] = True
                
                capabilities["n_classes"] = int(unique_targets)
            else:
                capabilities["task_type"] = "regression"
                capabilities["classification"] = False
                capabilities["regression"] = True
                capabilities["binary_classification"] = False
                capabilities["multiclass_classification"] = False
        else:
            capabilities["task_type"] = "unknown"
            capabilities["classification"] = False
            capabilities["regression"] = False
            capabilities["binary_classification"] = False
            capabilities["multiclass_classification"] = False
            limitations.append("Cannot determine task type without target column")
        
        # Detect feature types
        numerical_features = []
        categorical_features = []
        
        for col in df.columns:
            if col == target_col or col == prediction_col or col == probability_col:
                continue
            
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numerical_features.append(col)
                else:
                    categorical_features.append(col)
            except Exception:
                categorical_features.append(col)
        
        capabilities["numerical_features"] = numerical_features
        capabilities["categorical_features"] = categorical_features
        capabilities["has_numerical_features"] = len(numerical_features) > 0
        capabilities["has_categorical_features"] = len(categorical_features) > 0
        
        # Detect temporal data
        capabilities["temporal"] = False
        temporal_cols = []
        
        for col in df.columns:
            try:
                # Check if column is datetime or can be converted
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    temporal_cols.append(col)
                else:
                    # Try conversion
                    converted = pd.to_datetime(df[col], errors='coerce')
                    if converted.notna().sum() / len(df) > 0.8:
                        temporal_cols.append(col)
            except Exception:
                pass
        
        if temporal_cols:
            capabilities["temporal"] = True
            capabilities["temporal_columns"] = temporal_cols
        
        # Determine overall analyzability
        capabilities["can_analyze"] = (
            capabilities["sufficient_samples"] and
            (capabilities["target_available"] or capabilities["predictions_available"])
        )
        
        if not capabilities["can_analyze"]:
            limitations.append("Dataset cannot be analyzed: insufficient samples or missing target/predictions")
        
        self._log(f"Capability detection complete: {len(capabilities)} capabilities detected")
        
        severity = "NONE" if capabilities["can_analyze"] else "HIGH"
        
        return self._result({
            "capabilities": capabilities,
            "limitations": limitations,
            "can_analyze": capabilities["can_analyze"]
        }, severity=severity, module_name="CapabilityDetector")
    
    def can_run_engine(self, engine_name: str, capabilities: Dict[str, Any]) -> tuple:
        """
        Determine if a specific engine can run given the detected capabilities.
        
        Args:
            engine_name: Name of the engine (e.g., "calibration", "drift")
            capabilities: Capability dictionary from detect()
        
        Returns:
            Tuple of (can_run: bool, reason: str)
        """
        caps = capabilities.get("capabilities", {})
        
        engine_requirements = {
            "calibration": ["probabilities_available"],
            "drift": ["reference_available"],
            "fairness": ["predictions_available", "target_available"],
            "slicer": ["predictions_available", "target_available"],
            "calibration": ["probabilities_available"],
            "explainability": ["predictions_available"],
            "temporal": ["temporal"],
        }
        
        required = engine_requirements.get(engine_name.lower(), [])
        
        for req in required:
            if not caps.get(req, False):
                reason_map = {
                    "probabilities_available": "Prediction probabilities are unavailable",
                    "reference_available": "Reference dataset is unavailable",
                    "predictions_available": "Predictions are unavailable",
                    "target_available": "Target column is unavailable",
                    "temporal": "Temporal data is unavailable"
                }
                return False, reason_map.get(req, f"Required capability '{req}' not met")
        
        return True, "All requirements met"
    
    def _run(self, *args, **kwargs):
        """Override to use detect method."""
        return self.detect(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("pandas required for smoke test")
        sys.exit(1)
    
    # Classification dataset
    df_class = pd.DataFrame({
        "target": [0, 1, 0, 1, 0, 1, 0, 1],
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "feature2": ["A", "B", "A", "B", "A", "B", "A", "B"],
        "prediction": [0, 1, 0, 1, 0, 1, 0, 1],
        "probability": [0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.4, 0.6]
    })
    
    # Regression dataset
    df_reg = pd.DataFrame({
        "target": [10.5, 20.3, 15.7, 25.1, 18.9, 22.4],
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "prediction": [11.0, 19.5, 16.0, 24.8, 19.0, 22.0]
    })
    
    detector = CapabilityDetector(verbose=True)
    
    print("\n=== Classification Dataset ===")
    result = detector.detect(df_class, target_col="target", prediction_col="prediction", probability_col="probability")
    print(f"Task type: {result['findings']['capabilities']['task_type']}")
    print(f"Can analyze: {result['findings']['can_analyze']}")
    
    print("\n=== Regression Dataset ===")
    result = detector.detect(df_reg, target_col="target", prediction_col="prediction")
    print(f"Task type: {result['findings']['capabilities']['task_type']}")
    print(f"Can analyze: {result['findings']['can_analyze']}")
    
    print("\n=== Engine Eligibility ===")
    caps = detector.detect(df_class, target_col="target", prediction_col="prediction", probability_col="probability")["findings"]["capabilities"]
    
    for engine in ["calibration", "drift", "fairness"]:
        can_run, reason = detector.can_run_engine(engine, {"capabilities": caps})
        print(f"{engine}: {'CAN RUN' if can_run else 'SKIPPED'} - {reason}")
    
    print("\n✓ CapabilityDetector OK")
