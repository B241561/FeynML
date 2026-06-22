"""
Investigation Service - Service Layer for ML Failure Investigation
================================================================

Reusable service layer for orchestrating ML failure investigation.
Can be called by:
- Flask web application
- Future Monitor Page
- External monitoring systems
- Scheduled jobs

Responsibilities:
- Collect engine outputs
- Invoke Root Cause Engine
- Produce unified investigation object
- Handle edge cases gracefully
- Provide consistent interface

Usage:
    from webapp.services.investigation_service import InvestigationService
    
    service = InvestigationService()
    investigation = service.run_investigation(
        drift_report=drift_result,
        slice_report=slice_result,
        calibration_report=calib_result,
        data_quality_report=quality_result
    )
"""

import sys
import os
from typing import Dict, Optional, Any
import traceback

# Add project root to path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine.modules.root_cause_engine import AutoRootCauseEngine
from engine.modules.investigation import Investigation


class InvestigationService:
    """
    Service layer for ML failure investigation.
    
    Provides a consistent interface for running investigations
    across different contexts (web app, monitor page, scheduled jobs).
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the investigation service.
        
        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.root_cause_engine = AutoRootCauseEngine(verbose=verbose)
    
    def run_investigation(self,
                        drift_report: Optional[Dict] = None,
                        slice_report: Optional[Dict] = None,
                        calibration_report: Optional[Dict] = None,
                        data_quality_report: Optional[Dict] = None,
                        explainability_report: Optional[Dict] = None,
                        training_importance: Optional[Dict] = None,
                        production_importance: Optional[Dict] = None) -> Investigation:
        """
        Run a complete ML failure investigation.
        
        Args:
            drift_report: Output from DriftEngine
            slice_report: Output from SlicerEngine
            calibration_report: Output from CalibrationEngine
            data_quality_report: Output from data quality checks
            explainability_report: Output from ExplainabilityEngine
            training_importance: Training SHAP importance (dict: feature -> importance)
            production_importance: Production SHAP importance (dict: feature -> importance)
        
        Returns:
            Investigation object with unified results
        """
        try:
            if self.verbose:
                print("[InvestigationService] Starting investigation...")
            
            # Invoke AutoRootCauseEngine
            result_dict = self.root_cause_engine.run(
                drift_report=drift_report,
                slice_report=slice_report,
                calibration_report=calibration_report,
                data_quality_report=data_quality_report,
                explainability_report=explainability_report,
                training_importance=training_importance,
                production_importance=production_importance
            )
            
            # Convert to Investigation object
            investigation = Investigation.from_dict(result_dict)
            
            if self.verbose:
                print(f"[InvestigationService] Investigation complete. "
                      f"Health: {investigation.health_status}, "
                      f"Confidence: {investigation.confidence}")
            
            return investigation
            
        except Exception as e:
            if self.verbose:
                print(f"[InvestigationService] Error during investigation: {e}")
                traceback.print_exc()
            
            # Return degraded investigation on error
            return self._create_degraded_investigation(str(e))
    
    def run_investigation_from_results(self, engine_results: Dict[str, Any]) -> Investigation:
        """
        Run investigation from a dictionary of engine results.
        
        This is useful when you have all engine results in a single dict.
        
        Args:
            engine_results: Dictionary with engine outputs keyed by engine name
                          (e.g., {"drift": {...}, "calibration": {...}, ...})
        
        Returns:
            Investigation object with unified results
        """
        try:
            if self.verbose:
                print("[InvestigationService] Starting investigation from engine results...")
            
            # Extract individual reports
            drift_report = engine_results.get("drift")
            slice_report = engine_results.get("slice")
            calibration_report = engine_results.get("calibration")
            data_quality_report = engine_results.get("missing_data") or engine_results.get("data_quality")
            explainability_report = engine_results.get("explainability")
            
            # Run investigation
            return self.run_investigation(
                drift_report=drift_report,
                slice_report=slice_report,
                calibration_report=calibration_report,
                data_quality_report=data_quality_report,
                explainability_report=explainability_report
            )
            
        except Exception as e:
            if self.verbose:
                print(f"[InvestigationService] Error during investigation: {e}")
                traceback.print_exc()
            
            return self._create_degraded_investigation(str(e))
    
    def _create_degraded_investigation(self, error_message: str) -> Investigation:
        """
        Create a degraded investigation object when errors occur.
        
        Args:
            error_message: Error message to include in the investigation
        
        Returns:
            Investigation object with degraded status
        """
        investigation = Investigation(
            health_status="Unknown",
            confidence=0,
            evidence=[f"Investigation failed: {error_message}"],
            recommendations=["Review system logs and retry investigation."],
            metadata={"error": error_message, "degraded": True}
        )
        
        return investigation
    
    def get_investigation_summary(self, investigation: Investigation) -> str:
        """
        Get a human-readable summary of the investigation.
        
        Args:
            investigation: Investigation object
        
        Returns:
            Formatted summary string
        """
        lines = [
            "=" * 65,
            "INVESTIGATION SUMMARY",
            "=" * 65,
            f"Investigation ID: {investigation.investigation_id}",
            f"Health Status: {investigation.health_status}",
            f"Confidence: {investigation.confidence}%",
            f"Generated At: {investigation.generated_at}",
            ""
        ]
        
        if investigation.root_causes:
            lines.append("Top Root Causes:")
            for i, cause in enumerate(investigation.root_causes[:5], 1):
                lines.append(f"  {i}. {cause.cause} (Score: {cause.score}, Severity: {cause.severity})")
                if cause.evidence:
                    for ev in cause.evidence[:2]:
                        lines.append(f"     - {ev}")
            lines.append("")
        
        if investigation.recommendations:
            lines.append("Recommended Actions:")
            for i, rec in enumerate(investigation.recommendations, 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")
        
        lines.append("=" * 65)
        
        return "\n".join(lines)
    
    def is_investigation_critical(self, investigation: Investigation) -> bool:
        """
        Check if the investigation indicates a critical issue.
        
        Args:
            investigation: Investigation object
        
        Returns:
            True if critical, False otherwise
        """
        return investigation.is_critical()
    
    def get_critical_causes(self, investigation: Investigation) -> list:
        """
        Get all critical severity root causes.
        
        Args:
            investigation: Investigation object
        
        Returns:
            List of critical RootCause objects
        """
        return investigation.get_critical_causes()
    
    def get_causes_by_source(self, investigation: Investigation, source_module: str) -> list:
        """
        Get all root causes identified by a specific source module.
        
        Args:
            investigation: Investigation object
            source_module: Name of the source module (e.g., "drift_engine")
        
        Returns:
            List of RootCause objects from the specified module
        """
        return investigation.get_causes_by_source_module(source_module)


# Singleton instance for easy access
_investigation_service = None

def get_investigation_service(verbose: bool = False) -> InvestigationService:
    """
    Get the singleton investigation service instance.
    
    Args:
        verbose: Enable verbose logging
    
    Returns:
        InvestigationService instance
    """
    global _investigation_service
    if _investigation_service is None:
        _investigation_service = InvestigationService(verbose=verbose)
    return _investigation_service
