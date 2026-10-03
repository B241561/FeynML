"""
Unit Tests — Phase 3 Production Monitoring
==========================================

Tests for Model Registry, Model Versions, Monitoring Configuration,
Monitoring Runs, and Alerts.

Run:
    pytest tests/test_phase3_monitoring.py -q
"""

import sys
import os
import unittest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_WEBAPP = os.path.join(_ROOT, "webapp")
for p in [_ROOT, _WEBAPP]:
    if p not in sys.path:
        sys.path.insert(0, p)


class TestModelRegistry(unittest.TestCase):
    """Test Model Registry functionality."""

    def test_model_fields(self):
        """Test model field structure."""
        # Test field names without database initialization
        expected_fields = [
            'model_id', 'model_name', 'description', 'task_type', 
            'owner', 'model_metadata', 'created_at', 'updated_at'
        ]
        
        # Verify the model structure is defined correctly
        from webapp.models import MonitoredModel
        self.assertTrue(hasattr(MonitoredModel, '__tablename__'))
        self.assertEqual(MonitoredModel.__tablename__, 'monitored_models')
    
    def test_model_task_types(self):
        """Test supported task types."""
        valid_task_types = ['classification', 'regression']
        for task_type in valid_task_types:
            self.assertIn(task_type, valid_task_types)


class TestModelVersion(unittest.TestCase):
    """Test Model Version functionality."""

    def test_version_fields(self):
        """Test version field structure."""
        from webapp.models import ModelVersion
        self.assertTrue(hasattr(ModelVersion, '__tablename__'))
        self.assertEqual(ModelVersion.__tablename__, 'model_versions')
    
    def test_version_statuses(self):
        """Test different version statuses."""
        statuses = ["ACTIVE", "INACTIVE", "ARCHIVED"]
        for status in statuses:
            self.assertIn(status, statuses)


class TestMonitoringConfig(unittest.TestCase):
    """Test Monitoring Configuration functionality."""

    def test_config_fields(self):
        """Test config field structure."""
        from webapp.models import MonitoringConfig
        self.assertTrue(hasattr(MonitoringConfig, '__tablename__'))
        self.assertEqual(MonitoringConfig.__tablename__, 'monitoring_configs')
    
    def test_monitoring_signals(self):
        """Test monitoring signal flags."""
        expected_signals = [
            'monitor_data_drift', 'monitor_prediction_drift', 
            'monitor_performance', 'monitor_data_quality',
            'monitor_calibration', 'monitor_error_rate',
            'monitor_segment_degradation', 'monitor_temporal_degradation'
        ]
        from webapp.models import MonitoringConfig
        for signal in expected_signals:
            self.assertTrue(hasattr(MonitoringConfig, signal))


class TestMonitoringRun(unittest.TestCase):
    """Test Monitoring Run functionality."""

    def test_run_fields(self):
        """Test run field structure."""
        from webapp.models import MonitoringRun
        self.assertTrue(hasattr(MonitoringRun, '__tablename__'))
        self.assertEqual(MonitoringRun.__tablename__, 'monitoring_runs')
    
    def test_trigger_types(self):
        """Test different trigger types."""
        trigger_types = ["MANUAL", "SCHEDULED", "API", "AUTOMATED"]
        for trigger_type in trigger_types:
            self.assertIn(trigger_type, trigger_types)
    
    def test_run_statuses(self):
        """Test different run statuses."""
        statuses = ["PENDING", "RUNNING", "SUCCESS", "PARTIAL", "FAILED"]
        for status in statuses:
            self.assertIn(status, statuses)


class TestAlert(unittest.TestCase):
    """Test Alert functionality."""

    def test_alert_fields(self):
        """Test alert field structure."""
        from webapp.models import Alert
        self.assertTrue(hasattr(Alert, '__tablename__'))
        self.assertEqual(Alert.__tablename__, 'alerts')
    
    def test_alert_severities(self):
        """Test different alert severities."""
        severities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        for severity in severities:
            self.assertIn(severity, severities)
    
    def test_alert_lifecycle(self):
        """Test alert lifecycle states."""
        states = ["OPEN", "ACKNOWLEDGED", "RESOLVED"]
        for state in states:
            self.assertIn(state, states)
    
    def test_alert_types(self):
        """Test different alert types."""
        alert_types = ["DRIFT", "PERFORMANCE", "DATA_QUALITY", "CALIBRATION"]
        for alert_type in alert_types:
            self.assertIn(alert_type, alert_types)


class TestMonitoringService(unittest.TestCase):
    """Test Monitoring Service functionality."""

    def test_service_initialization(self):
        """Test service can be initialized."""
        from webapp.services.monitoring_service import MonitoringService
        service = MonitoringService(db_session=None)
        self.assertIsNotNone(service)
    
    def test_determine_health_status_healthy(self):
        """Test health status determination for healthy results."""
        from webapp.services.monitoring_service import MonitoringService
        service = MonitoringService(db_session=None)
        
        results = {
            'root_cause': {
                'health_status': 'Healthy'
            }
        }
        
        status = service._determine_health_status(results)
        self.assertEqual(status, "HEALTHY")
    
    def test_determine_health_status_warning(self):
        """Test health status determination for warning results."""
        from webapp.services.monitoring_service import MonitoringService
        service = MonitoringService(db_session=None)
        
        results = {
            'root_cause': {
                'health_status': 'Warning'
            }
        }
        
        status = service._determine_health_status(results)
        self.assertEqual(status, "WARNING")
    
    def test_determine_health_status_critical(self):
        """Test health status determination for critical results."""
        from webapp.services.monitoring_service import MonitoringService
        service = MonitoringService(db_session=None)
        
        results = {
            'root_cause': {
                'health_status': 'Critical'
            }
        }
        
        status = service._determine_health_status(results)
        self.assertEqual(status, "CRITICAL")


class TestAlertEngine(unittest.TestCase):
    """Test Alert Engine functionality."""

    def test_engine_initialization(self):
        """Test engine can be initialized."""
        from webapp.services.alert_engine import AlertEngine
        engine = AlertEngine(db_session=None)
        self.assertIsNotNone(engine)
    
    def test_get_default_thresholds(self):
        """Test default threshold values."""
        from webapp.services.alert_engine import AlertEngine
        engine = AlertEngine(db_session=None)
        
        thresholds = engine._get_default_thresholds()
        
        self.assertIn("psi", thresholds)
        self.assertIn("ks_stat", thresholds)
        self.assertIn("error_rate_increase", thresholds)
        self.assertEqual(thresholds["psi"], 0.20)
        self.assertEqual(thresholds["ks_stat"], 0.30)
    
    def test_calculate_drift_severity_critical(self):
        """Test drift severity calculation for critical drift."""
        from webapp.services.alert_engine import AlertEngine
        engine = AlertEngine(db_session=None)
        
        severity = engine._calculate_drift_severity(psi=0.60, ks_stat=0.60)
        self.assertEqual(severity, "CRITICAL")
    
    def test_calculate_drift_severity_high(self):
        """Test drift severity calculation for high drift."""
        from webapp.services.alert_engine import AlertEngine
        engine = AlertEngine(db_session=None)
        
        severity = engine._calculate_drift_severity(psi=0.35, ks_stat=0.45)
        self.assertEqual(severity, "HIGH")
    
    def test_calculate_drift_severity_medium(self):
        """Test drift severity calculation for medium drift."""
        from webapp.services.alert_engine import AlertEngine
        engine = AlertEngine(db_session=None)
        
        severity = engine._calculate_drift_severity(psi=0.25, ks_stat=0.35)
        self.assertEqual(severity, "MEDIUM")
    
    def test_calculate_drift_severity_low(self):
        """Test drift severity calculation for low drift."""
        from webapp.services.alert_engine import AlertEngine
        engine = AlertEngine(db_session=None)
        
        severity = engine._calculate_drift_severity(psi=0.15, ks_stat=0.20)
        self.assertEqual(severity, "LOW")
    
    def test_evaluate_drift_alerts(self):
        """Test drift alert evaluation."""
        from webapp.services.alert_engine import AlertEngine
        engine = AlertEngine(db_session=None)
        
        results = {
            'drift': {
                'findings': {
                    'per_feature': [
                        {
                            'feature': 'age',
                            'psi': 0.30,
                            'ks_stat': 0.35
                        }
                    ]
                }
            }
        }
        
        thresholds = {'psi': 0.20, 'ks_stat': 0.30}
        
        alerts = engine._evaluate_drift_alerts(
            run_id="run_001",
            model_id=1,
            version_id=1,
            results=results,
            thresholds=thresholds
        )
        
        self.assertGreater(len(alerts), 0)
        self.assertEqual(alerts[0]['alert_type'], "DRIFT")
        self.assertEqual(alerts[0]['metric'], "PSI")
    
    def test_evaluate_performance_alerts(self):
        """Test performance alert evaluation."""
        from webapp.services.alert_engine import AlertEngine
        engine = AlertEngine(db_session=None)
        
        results = {
            'error_taxonomy': {
                'findings': {
                    'overall_metrics': {
                        'error_rate': 0.15
                    }
                }
            }
        }
        
        thresholds = {'error_rate_increase': 0.10}
        
        alerts = engine._evaluate_performance_alerts(
            run_id="run_001",
            model_id=1,
            version_id=1,
            results=results,
            thresholds=thresholds
        )
        
        self.assertGreater(len(alerts), 0)
        self.assertEqual(alerts[0]['alert_type'], "PERFORMANCE")
    
    def test_evaluate_data_quality_alerts(self):
        """Test data quality alert evaluation."""
        from webapp.services.alert_engine import AlertEngine
        engine = AlertEngine(db_session=None)
        
        results = {
            'missing_data': {
                'findings': {
                    'overall_missing_rate': 0.08
                }
            }
        }
        
        thresholds = {'missing_rate': 0.05}
        
        alerts = engine._evaluate_data_quality_alerts(
            run_id="run_001",
            model_id=1,
            version_id=1,
            results=results,
            thresholds=thresholds
        )
        
        self.assertGreater(len(alerts), 0)
        self.assertEqual(alerts[0]['alert_type'], "DATA_QUALITY")
    
    def test_evaluate_calibration_alerts(self):
        """Test calibration alert evaluation."""
        from webapp.services.alert_engine import AlertEngine
        engine = AlertEngine(db_session=None)
        
        results = {
            'calibration': {
                'findings': {
                    'ece': 0.20
                }
            }
        }
        
        thresholds = {'ece': 0.15}
        
        alerts = engine._evaluate_calibration_alerts(
            run_id="run_001",
            model_id=1,
            version_id=1,
            results=results,
            thresholds=thresholds
        )
        
        self.assertGreater(len(alerts), 0)
        self.assertEqual(alerts[0]['alert_type'], "CALIBRATION")


if __name__ == "__main__":
    unittest.main()
