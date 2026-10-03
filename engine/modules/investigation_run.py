"""
Engine Module — Investigation Run
=================================
Structured investigation run tracking for ML Failure Investigation.

Tracks run-level metadata including:
  • Run ID and timestamp
  • Input metadata
  • Dataset profile
  • Capabilities
  • Schema validation
  • Data quality
  • Per-engine status (SUCCESS/PARTIAL/SKIPPED/FAILED)
  • Findings and evidence
  • Overall status

Integrates with existing Investigation class from investigation.py.

Usage:
    from engine.modules.investigation_run import InvestigationRun
    
    run = InvestigationRun()
    run.start(df, target_col="target")
    run.record_engine_result("drift", drift_result)
    run.record_engine_result("calibration", calibration_result)
    run.complete()
"""

import sys
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

try:
    from engine.modules.investigation import Investigation
except ImportError:
    from investigation import Investigation


class InvestigationRun:
    """
    Structured investigation run tracking.
    
    Tracks run-level metadata and per-engine execution status
    to provide complete visibility into the investigation process.
    """
    
    def __init__(self, run_id: Optional[str] = None):
        """
        Initialize an investigation run.
        
        Args:
            run_id: Optional run ID (auto-generated if not provided)
        """
        self.run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        self.timestamp = datetime.now().isoformat()
        self.input_metadata = {}
        self.dataset_profile = {}
        self.capabilities = {}
        self.schema_validation = {}
        self.data_quality = {}
        self.engine_results = {}  # engine_name -> result dict
        self.investigation = None  # Investigation object from root cause engine
        self.status = "INITIALIZED"
        self.errors = []
        self.warnings = []
    
    def start(self, df, target_col: Optional[str] = None, 
              prediction_col: Optional[str] = None,
              probability_col: Optional[str] = None):
        """
        Initialize the investigation run with dataset information.
        
        Args:
            df: Input dataset
            target_col: Target column name
            prediction_col: Prediction column name
            probability_col: Probability column name
        """
        self.status = "RUNNING"
        
        # Record input metadata
        self.input_metadata = {
            "target_col": target_col,
            "prediction_col": prediction_col,
            "probability_col": probability_col,
            "n_rows": len(df) if hasattr(df, '__len__') else 0,
            "n_columns": len(df.columns) if hasattr(df, 'columns') else 0
        }
        
        # Dataset profile
        self.dataset_profile = {
            "columns": list(df.columns) if hasattr(df, 'columns') else [],
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()} if hasattr(df, 'dtypes') else {},
            "memory_usage": str(df.memory_usage(deep=True).sum()) if hasattr(df, 'memory_usage') else "unknown"
        }
    
    def record_schema_validation(self, validation_result: Dict[str, Any]):
        """
        Record schema validation results.
        
        Args:
            validation_result: Result from SchemaValidator
        """
        self.schema_validation = validation_result
        if validation_result.get("findings", {}).get("status") == "INVALID":
            self.status = "SCHEMA_INVALID"
            self.errors.extend(validation_result.get("findings", {}).get("errors", []))
    
    def record_capabilities(self, capability_result: Dict[str, Any]):
        """
        Record capability detection results.
        
        Args:
            capability_result: Result from CapabilityDetector
        """
        self.capabilities = capability_result
        if not capability_result.get("findings", {}).get("can_analyze", True):
            self.warnings.extend(capability_result.get("findings", {}).get("limitations", []))
    
    def record_data_quality(self, quality_result: Dict[str, Any]):
        """
        Record data quality results.
        
        Args:
            quality_result: Result from DataQualityEngine
        """
        self.data_quality = quality_result
        if quality_result.get("findings", {}).get("warnings"):
            self.warnings.extend(quality_result.get("findings", {}).get("warnings"))
    
    def record_engine_result(self, engine_name: str, result: Dict[str, Any]):
        """
        Record the result from an engine execution.
        
        Args:
            engine_name: Name of the engine
            result: Result envelope from the engine
        """
        self.engine_results[engine_name] = result
        
        # Track engine status
        engine_status = result.get("status", "UNKNOWN")
        
        if engine_status == "FAILED":
            self.status = "PARTIAL" if self.status == "RUNNING" else self.status
            error_msg = result.get("findings", {}).get("error", "Unknown error")
            self.errors.append(f"{engine_name} failed: {error_msg}")
        elif engine_status == "SKIPPED":
            skip_reason = result.get("skip_reason", "Unknown reason")
            self.warnings.append(f"{engine_name} skipped: {skip_reason}")
    
    def set_investigation(self, investigation: Investigation):
        """
        Set the final Investigation object from the root cause engine.
        
        Args:
            investigation: Investigation object
        """
        self.investigation = investigation
    
    def complete(self):
        """
        Mark the investigation run as complete and determine final status.
        """
        # Determine overall status based on engine results
        if self.status == "SCHEMA_INVALID":
            self.status = "FAILED"
        elif self.status == "RUNNING":
            # Check if any engines failed
            failed_engines = [name for name, result in self.engine_results.items() 
                            if result.get("status") == "FAILED"]
            if failed_engines:
                self.status = "PARTIAL"
            else:
                self.status = "SUCCESS"
        
        self.timestamp = datetime.now().isoformat()
    
    def get_engine_statuses(self) -> Dict[str, str]:
        """
        Get the status of all executed engines.
        
        Returns:
            Dictionary mapping engine names to their status
        """
        return {
            name: result.get("status", "UNKNOWN")
            for name, result in self.engine_results.items()
        }
    
    def get_skipped_engines(self) -> Dict[str, str]:
        """
        Get all skipped engines and their reasons.
        
        Returns:
            Dictionary mapping engine names to skip reasons
        """
        return {
            name: result.get("skip_reason", "Unknown reason")
            for name, result in self.engine_results.items()
            if result.get("status") == "SKIPPED"
        }
    
    def get_failed_engines(self) -> Dict[str, str]:
        """
        Get all failed engines and their errors.
        
        Returns:
            Dictionary mapping engine names to error messages
        """
        return {
            name: result.get("findings", {}).get("error", "Unknown error")
            for name, result in self.engine_results.items()
            if result.get("status") == "FAILED"
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the investigation run to a dictionary.
        
        Returns:
            Dictionary representation of the run
        """
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "status": self.status,
            "input_metadata": self.input_metadata,
            "dataset_profile": self.dataset_profile,
            "capabilities": self.capabilities,
            "schema_validation": self.schema_validation,
            "data_quality": self.data_quality,
            "engine_results": self.engine_results,
            "engine_statuses": self.get_engine_statuses(),
            "skipped_engines": self.get_skipped_engines(),
            "failed_engines": self.get_failed_engines(),
            "investigation": self.investigation.to_dict() if self.investigation else None,
            "errors": self.errors,
            "warnings": self.warnings
        }
    
    def get_summary(self) -> str:
        """
        Get a human-readable summary of the investigation run.
        
        Returns:
            Formatted summary string
        """
        lines = [
            "=" * 65,
            "INVESTIGATION RUN SUMMARY",
            "=" * 65,
            f"Run ID: {self.run_id}",
            f"Timestamp: {self.timestamp}",
            f"Status: {self.status}",
            ""
        ]
        
        if self.input_metadata:
            lines.append("Input Metadata:")
            for key, value in self.input_metadata.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        engine_statuses = self.get_engine_statuses()
        if engine_statuses:
            lines.append("Engine Statuses:")
            for engine, status in engine_statuses.items():
                lines.append(f"  {engine}: {status}")
            lines.append("")
        
        skipped = self.get_skipped_engines()
        if skipped:
            lines.append("Skipped Engines:")
            for engine, reason in skipped.items():
                lines.append(f"  {engine}: {reason}")
            lines.append("")
        
        failed = self.get_failed_engines()
        if failed:
            lines.append("Failed Engines:")
            for engine, error in failed.items():
                lines.append(f"  {engine}: {error}")
            lines.append("")
        
        if self.investigation:
            lines.append("Investigation Results:")
            lines.append(f"  Health Status: {self.investigation.health_status}")
            lines.append(f"  Confidence: {self.investigation.confidence}%")
            lines.append(f"  Root Causes: {len(self.investigation.root_causes)}")
            lines.append("")
        
        lines.append("=" * 65)
        
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("pandas required for smoke test")
        sys.exit(1)
    
    df = pd.DataFrame({
        "target": [0, 1, 0, 1, 0, 1],
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "prediction": [0, 1, 0, 1, 0, 1]
    })
    
    run = InvestigationRun()
    run.start(df, target_col="target", prediction_col="prediction")
    
    # Mock schema validation
    run.record_schema_validation({
        "findings": {
            "status": "VALID",
            "errors": [],
            "warnings": []
        }
    })
    
    # Mock capabilities
    run.record_capabilities({
        "findings": {
            "capabilities": {"classification": True, "regression": False},
            "can_analyze": True,
            "limitations": []
        }
    })
    
    # Mock engine results
    run.record_engine_result("drift", {
        "status": "SUCCESS",
        "severity": "NONE",
        "findings": {"drift_detected": False}
    })
    
    run.record_engine_result("calibration", {
        "status": "SKIPPED",
        "skip_reason": "Prediction probabilities unavailable",
        "severity": "NONE",
        "findings": {}
    })
    
    run.complete()
    
    print("\n=== Investigation Run Summary ===")
    print(run.get_summary())
    
    print("\n✓ InvestigationRun OK")
