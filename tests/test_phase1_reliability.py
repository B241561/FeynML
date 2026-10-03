"""
Phase 1 Reliability Tests
=========================
Regression tests for Phase 1 reliability features:
- Schema Validator
- Capability Detector
- Data Quality Engine
- BaseModule can_run() and SKIPPED status
- Investigation Run
"""

import pytest
import sys
import os

# Add project root to path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas required")
class TestSchemaValidator:
    """Test schema validation functionality."""
    
    def test_valid_dataset(self):
        """Test validation of a valid dataset."""
        from engine.modules.schema_validator import SchemaValidator
        
        df = pd.DataFrame({
            "target": [0, 1, 0, 1],
            "feature1": [1.0, 2.0, 3.0, 4.0],
            "feature2": ["A", "B", "A", "B"]
        })
        
        validator = SchemaValidator(verbose=False)
        result = validator.validate(df, target_col="target")
        
        assert result["findings"]["status"] == "VALID"
        assert len(result["findings"]["errors"]) == 0
        assert result["findings"]["n_rows"] == 4
        assert result["findings"]["n_columns"] == 3
        assert result["severity"] == "NONE"
    
    def test_empty_dataset(self):
        """Test validation of an empty dataset."""
        from engine.modules.schema_validator import SchemaValidator
        
        df = pd.DataFrame()
        
        validator = SchemaValidator(verbose=False)
        result = validator.validate(df, target_col="target")
        
        assert result["findings"]["status"] == "INVALID"
        assert len(result["findings"]["errors"]) > 0
        assert any("empty" in err.lower() for err in result["findings"]["errors"])
        assert result["severity"] == "CRITICAL"
    
    def test_missing_target_column(self):
        """Test validation when required target column is missing."""
        from engine.modules.schema_validator import SchemaValidator
        
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0],
            "feature2": ["A", "B", "A", "B"]
        })
        
        validator = SchemaValidator(verbose=False)
        result = validator.validate(df, target_col="target")
        
        assert result["findings"]["status"] == "INVALID"
        assert "target" in result["findings"]["missing_required_columns"]
        assert result["severity"] == "CRITICAL"
    
    def test_duplicate_columns(self):
        """Test detection of duplicate column names."""
        from engine.modules.schema_validator import SchemaValidator
        
        df = pd.DataFrame({
            "target": [0, 1, 0, 1],
            "feature": [1.0, 2.0, 3.0, 4.0]
        })
        # Create duplicate columns
        df = pd.concat([df, df], axis=1)
        
        validator = SchemaValidator(verbose=False)
        result = validator.validate(df)
        
        assert result["findings"]["status"] == "INVALID"
        assert len(result["findings"]["errors"]) > 0
        assert any("duplicate" in err.lower() for err in result["findings"]["errors"])
    
    def test_type_detection(self):
        """Test data type detection for columns."""
        from engine.modules.schema_validator import SchemaValidator
        
        df = pd.DataFrame({
            "numerical": [1.0, 2.0, 3.0, 4.0],
            "categorical": ["A", "B", "A", "B"],
            "boolean": [True, False, True, False]
        })
        
        validator = SchemaValidator(verbose=False)
        result = validator.validate(df)
        
        assert result["findings"]["status"] == "VALID"
        detected_types = result["findings"]["detected_types"]
        assert detected_types["numerical"] == "numerical"
        assert detected_types["categorical"] == "categorical"
        assert detected_types["boolean"] == "boolean"


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas required")
class TestCapabilityDetector:
    """Test capability detection functionality."""
    
    def test_classification_detection(self):
        """Test detection of classification task."""
        from engine.modules.capability_detector import CapabilityDetector
        
        df = pd.DataFrame({
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0]
        })
        
        detector = CapabilityDetector(verbose=False)
        result = detector.detect(df, target_col="target")
        
        assert result["findings"]["capabilities"]["task_type"] == "classification"
        assert result["findings"]["capabilities"]["classification"] is True
        assert result["findings"]["capabilities"]["regression"] is False
        assert result["findings"]["capabilities"]["binary_classification"] is True
        assert result["findings"]["can_analyze"] is True
    
    def test_regression_detection(self):
        """Test detection of regression task."""
        from engine.modules.capability_detector import CapabilityDetector
        
        # Use more unique values to trigger regression detection
        df = pd.DataFrame({
            "target": [10.5, 20.3, 15.7, 25.1, 18.9, 22.4, 30.0, 35.5, 40.2, 45.8, 50.1, 55.3],
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        })
        
        detector = CapabilityDetector(verbose=False)
        result = detector.detect(df, target_col="target")
        
        assert result["findings"]["capabilities"]["task_type"] == "regression"
        assert result["findings"]["capabilities"]["classification"] is False
        assert result["findings"]["capabilities"]["regression"] is True
    
    def test_probability_availability(self):
        """Test detection of probability column availability."""
        from engine.modules.capability_detector import CapabilityDetector
        
        df = pd.DataFrame({
            "target": [0, 1, 0, 1],
            "prediction": [0, 1, 0, 1],
            "probability": [0.2, 0.8, 0.3, 0.7]
        })
        
        detector = CapabilityDetector(verbose=False)
        result = detector.detect(df, target_col="target", probability_col="probability")
        
        assert result["findings"]["capabilities"]["probabilities_available"] is True
    
    def test_insufficient_samples(self):
        """Test detection of insufficient samples."""
        from engine.modules.capability_detector import CapabilityDetector
        
        df = pd.DataFrame({
            "target": [0, 1],
            "feature1": [1.0, 2.0]
        })
        
        detector = CapabilityDetector(verbose=False, min_samples=30)
        result = detector.detect(df, target_col="target")
        
        assert result["findings"]["capabilities"]["sufficient_samples"] is False
        assert not result["findings"]["can_analyze"]
        assert len(result["findings"]["limitations"]) > 0
    
    def test_engine_eligibility(self):
        """Test engine eligibility checking."""
        from engine.modules.capability_detector import CapabilityDetector
        
        df = pd.DataFrame({
            "target": [0, 1, 0, 1],
            "prediction": [0, 1, 0, 1]
        })
        
        detector = CapabilityDetector(verbose=False)
        result = detector.detect(df, target_col="target", prediction_col="prediction")
        caps = result["findings"]["capabilities"]
        
        # Calibration should be skipped without probabilities
        can_run, reason = detector.can_run_engine("calibration", {"capabilities": caps})
        assert can_run is False
        assert "probabilit" in reason.lower()  # Check for substring
        
        # Fairness should run with predictions and target
        can_run, reason = detector.can_run_engine("fairness", {"capabilities": caps})
        assert can_run is True


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas required")
class TestDataQualityEngine:
    """Test data quality engine functionality."""
    
    def test_missingness_detection(self):
        """Test detection of missing values."""
        from engine.modules.data_quality_engine import DataQualityEngine
        
        df = pd.DataFrame({
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "missing_col": [1, None, 3, None, 5, None, 7, None, 9, None]
        })
        
        engine = DataQualityEngine(verbose=False)
        result = engine.analyze(df, target_col="target")
        
        assert "missingness" in result["findings"]
        assert result["findings"]["missingness"]["missing_rates"]["missing_col"] > 0
        assert result["findings"]["missingness"]["high_missingness"] == ["missing_col"]
    
    def test_duplicate_detection(self):
        """Test detection of duplicate rows."""
        from engine.modules.data_quality_engine import DataQualityEngine
        
        df = pd.DataFrame({
            "target": [0, 1, 0, 1],
            "feature": [1, 2, 1, 2]
        })
        # Add duplicate
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        
        engine = DataQualityEngine(verbose=False)
        result = engine.analyze(df)
        
        assert result["findings"]["duplicates"]["duplicate_count"] > 0
        assert result["findings"]["duplicates"]["duplicate_rate"] > 0
    
    def test_constant_feature_detection(self):
        """Test detection of constant features."""
        from engine.modules.data_quality_engine import DataQualityEngine
        
        df = pd.DataFrame({
            "target": [0, 1, 0, 1],
            "constant": [5, 5, 5, 5],
            "feature": [1, 2, 3, 4]
        })
        
        engine = DataQualityEngine(verbose=False)
        result = engine.analyze(df)
        
        assert "constant" in result["findings"]["constant_features"]
    
    def test_target_quality(self):
        """Test target quality analysis."""
        from engine.modules.data_quality_engine import DataQualityEngine
        
        df = pd.DataFrame({
            "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # Balanced
            "feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        
        engine = DataQualityEngine(verbose=False)
        result = engine.analyze(df, target_col="target")
        
        assert "target_quality" in result["findings"]
        assert result["findings"]["target_quality"]["unique_count"] == 2
    
    def test_overall_status(self):
        """Test overall status determination."""
        from engine.modules.data_quality_engine import DataQualityEngine
        
        # Clean dataset
        df_clean = pd.DataFrame({
            "target": [0, 1, 0, 1],
            "feature": [1, 2, 3, 4]
        })
        
        engine = DataQualityEngine(verbose=False)
        result = engine.analyze(df_clean)
        
        assert result["findings"]["overall_status"] in ["GOOD", "ACCEPTABLE"]


class TestBaseModuleReliability:
    """Test BaseModule reliability enhancements."""
    
    def test_can_run_default(self):
        """Test default can_run() implementation."""
        from engine.base_module import BaseModule
        
        class TestModule(BaseModule):
            def _run(self, *args, **kwargs):
                return self._result({}, severity="NONE")
        
        module = TestModule(verbose=False)
        can_run, reason = module.can_run()
        
        assert can_run is True
        assert reason == "No specific requirements"
    
    def test_skipped_status_in_result(self):
        """Test SKIPPED status in result envelope."""
        from engine.base_module import BaseModule
        
        class TestModule(BaseModule):
            def _run(self, *args, **kwargs):
                return self._result(
                    {}, 
                    severity="NONE", 
                    status="SKIPPED",
                    skip_reason="Test skip reason"
                )
        
        module = TestModule(verbose=False)
        result = module.run()
        
        assert result["status"] == "SKIPPED"
        assert result["skip_reason"] == "Test skip reason"
    
    def test_status_inference_from_severity(self):
        """Test automatic status inference from severity."""
        from engine.base_module import BaseModule
        
        class TestModule(BaseModule):
            def _run(self, *args, **kwargs):
                return self._result({}, severity="HIGH")
        
        module = TestModule(verbose=False)
        result = module.run()
        
        assert result["status"] == "FAILED"
    
    def test_backward_compatibility(self):
        """Test backward compatibility of result envelope."""
        from engine.base_module import BaseModule
        
        class TestModule(BaseModule):
            def _run(self, *args, **kwargs):
                # Old-style call without status parameter
                return self._result({}, severity="LOW")
        
        module = TestModule(verbose=False)
        result = module.run()
        
        # All old fields should still be present
        assert "module" in result
        assert "timestamp" in result
        assert "severity" in result
        assert "passed" in result
        assert "findings" in result
        assert "log" in result
        # New field should be added
        assert "status" in result


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas required")
class TestInvestigationRun:
    """Test investigation run tracking."""
    
    def test_run_initialization(self):
        """Test investigation run initialization."""
        from engine.modules.investigation_run import InvestigationRun
        
        run = InvestigationRun()
        
        assert run.run_id is not None
        assert run.status == "INITIALIZED"
        assert len(run.engine_results) == 0
    
    def test_run_start(self):
        """Test starting an investigation run."""
        from engine.modules.investigation_run import InvestigationRun
        
        df = pd.DataFrame({
            "target": [0, 1, 0, 1],
            "feature": [1, 2, 3, 4]
        })
        
        run = InvestigationRun()
        run.start(df, target_col="target")
        
        assert run.status == "RUNNING"
        assert run.input_metadata["target_col"] == "target"
        assert run.input_metadata["n_rows"] == 4
        assert len(run.dataset_profile["columns"]) == 2
    
    def test_engine_result_recording(self):
        """Test recording engine results."""
        from engine.modules.investigation_run import InvestigationRun
        
        run = InvestigationRun()
        run.record_engine_result("drift", {
            "status": "SUCCESS",
            "severity": "NONE",
            "findings": {"drift_detected": False}
        })
        
        assert "drift" in run.engine_results
        assert run.engine_results["drift"]["status"] == "SUCCESS"
    
    def test_skipped_engine_tracking(self):
        """Test tracking of skipped engines."""
        from engine.modules.investigation_run import InvestigationRun
        
        run = InvestigationRun()
        run.record_engine_result("calibration", {
            "status": "SKIPPED",
            "skip_reason": "Probabilities unavailable",
            "severity": "NONE",
            "findings": {}
        })
        
        skipped = run.get_skipped_engines()
        assert "calibration" in skipped
        assert "probabilities" in skipped["calibration"].lower()
    
    def test_failed_engine_tracking(self):
        """Test tracking of failed engines."""
        from engine.modules.investigation_run import InvestigationRun
        
        run = InvestigationRun()
        run.record_engine_result("temporal", {
            "status": "FAILED",
            "severity": "HIGH",
            "findings": {"error": "No temporal data"}
        })
        
        failed = run.get_failed_engines()
        assert "temporal" in failed
        assert "temporal" in failed["temporal"].lower()
    
    def test_run_completion(self):
        """Test run completion and status determination."""
        from engine.modules.investigation_run import InvestigationRun
        
        run = InvestigationRun()
        run.status = "RUNNING"
        
        # Record successful engine
        run.record_engine_result("drift", {
            "status": "SUCCESS",
            "severity": "NONE",
            "findings": {}
        })
        
        run.complete()
        
        assert run.status == "SUCCESS"
    
    def test_run_to_dict(self):
        """Test conversion to dictionary."""
        from engine.modules.investigation_run import InvestigationRun
        
        run = InvestigationRun()
        run_dict = run.to_dict()
        
        assert "run_id" in run_dict
        assert "timestamp" in run_dict
        assert "status" in run_dict
        assert "engine_results" in run_dict
        assert "engine_statuses" in run_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
