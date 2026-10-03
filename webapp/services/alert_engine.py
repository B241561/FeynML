"""
Alert Engine
============

Generates and manages monitoring alerts from investigation results.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any


class AlertEngine:
    """
    Engine for generating and managing monitoring alerts.
    
    Evaluates investigation results against configured thresholds
    and generates alerts when thresholds are exceeded.
    """
    
    def __init__(self, db_session=None):
        """
        Initialize Alert Engine.
        
        Args:
            db_session: SQLAlchemy database session
        """
        self.db_session = db_session
    
    def evaluate_alerts(self,
                       run_id: str,
                       model_id: int,
                       version_id: Optional[int],
                       results: Dict[str, Any],
                       thresholds: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Evaluate investigation results against thresholds and generate alerts.
        
        Args:
            run_id: Monitoring run ID
            model_id: Model ID
            version_id: Model version ID
            results: Investigation results from AnalysisRunner
            thresholds: Configured thresholds
        
        Returns:
            List of generated alerts
        """
        if thresholds is None:
            thresholds = self._get_default_thresholds()
        
        alerts = []
        
        # Evaluate drift alerts
        drift_alerts = self._evaluate_drift_alerts(run_id, model_id, version_id, results, thresholds)
        alerts.extend(drift_alerts)
        
        # Evaluate performance alerts
        perf_alerts = self._evaluate_performance_alerts(run_id, model_id, version_id, results, thresholds)
        alerts.extend(perf_alerts)
        
        # Evaluate data quality alerts
        dq_alerts = self._evaluate_data_quality_alerts(run_id, model_id, version_id, results, thresholds)
        alerts.extend(dq_alerts)
        
        # Evaluate calibration alerts
        cal_alerts = self._evaluate_calibration_alerts(run_id, model_id, version_id, results, thresholds)
        alerts.extend(cal_alerts)
        
        # Store alerts in database
        stored_alerts = []
        for alert in alerts:
            stored = self._store_alert(alert)
            if stored:
                stored_alerts.append(stored)
        
        return stored_alerts
    
    def _get_default_thresholds(self) -> Dict[str, float]:
        """
        Get default alert thresholds.
        
        Returns:
            Dictionary of default thresholds
        """
        return {
            "psi": 0.20,
            "ks_stat": 0.30,
            "error_rate_increase": 0.10,
            "performance_decrease": 0.10,
            "missing_rate": 0.05,
            "ece": 0.15
        }
    
    def _evaluate_drift_alerts(self,
                              run_id: str,
                              model_id: int,
                              version_id: Optional[int],
                              results: Dict[str, Any],
                              thresholds: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Evaluate drift-related alerts.
        
        Args:
            run_id: Monitoring run ID
            model_id: Model ID
            version_id: Model version ID
            results: Investigation results
            thresholds: Configured thresholds
        
        Returns:
            List of drift alerts
        """
        alerts = []
        
        # Check feature drift
        drift_results = results.get('drift', {})
        findings = drift_results.get('findings', {})
        per_feature = findings.get('per_feature', [])
        
        for feature_drift in per_feature:
            psi = feature_drift.get('psi', 0)
            ks_stat = feature_drift.get('ks_stat', 0)
            feature = feature_drift.get('feature', 'unknown')
            
            # Check PSI threshold
            if psi >= thresholds.get('psi', 0.20):
                severity = self._calculate_drift_severity(psi, ks_stat)
                alerts.append({
                    "alert_id": f"alert_{uuid.uuid4().hex[:8]}",
                    "model_id": model_id,
                    "version_id": version_id,
                    "run_id": run_id,
                    "alert_type": "DRIFT",
                    "severity": severity,
                    "metric": "PSI",
                    "observed_value": psi,
                    "threshold": thresholds.get('psi', 0.20),
                    "message": f"Feature drift detected for {feature}: PSI={psi:.3f} exceeds threshold {thresholds.get('psi', 0.20)}",
                    "deduplication_key": f"drift_{feature}",
                    "status": "OPEN"
                })
        
        # Check prediction drift
        pred_drift = results.get('prediction_drift', {})
        pred_findings = pred_drift.get('findings', {})
        drift_metrics = pred_findings.get('drift_metrics', {})
        
        if drift_metrics:
            psi = drift_metrics.get('psi', 0)
            ks_stat = drift_metrics.get('ks_stat', 0)
            
            if psi >= thresholds.get('psi', 0.20) or ks_stat >= thresholds.get('ks_stat', 0.30):
                severity = self._calculate_drift_severity(psi, ks_stat)
                alerts.append({
                    "alert_id": f"alert_{uuid.uuid4().hex[:8]}",
                    "model_id": model_id,
                    "version_id": version_id,
                    "run_id": run_id,
                    "alert_type": "DRIFT",
                    "severity": severity,
                    "metric": "prediction_drift",
                    "observed_value": psi,
                    "threshold": thresholds.get('psi', 0.20),
                    "message": f"Prediction drift detected: PSI={psi:.3f}, KS={ks_stat:.3f}",
                    "deduplication_key": "prediction_drift",
                    "status": "OPEN"
                })
        
        return alerts
    
    def _evaluate_performance_alerts(self,
                                     run_id: str,
                                     model_id: int,
                                     version_id: Optional[int],
                                     results: Dict[str, Any],
                                     thresholds: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Evaluate performance-related alerts.
        
        Args:
            run_id: Monitoring run ID
            model_id: Model ID
            version_id: Model version ID
            results: Investigation results
            thresholds: Configured thresholds
        
        Returns:
            List of performance alerts
        """
        alerts = []
        
        # Check error taxonomy
        error_taxonomy = results.get('error_taxonomy', {})
        error_findings = error_taxonomy.get('findings', {})
        overall_metrics = error_findings.get('overall_metrics', {})
        error_rate = overall_metrics.get('error_rate', 0)
        
        if error_rate >= thresholds.get('error_rate_increase', 0.10):
            severity = "HIGH" if error_rate >= 0.20 else "MEDIUM"
            alerts.append({
                "alert_id": f"alert_{uuid.uuid4().hex[:8]}",
                "model_id": model_id,
                "version_id": version_id,
                "run_id": run_id,
                "alert_type": "PERFORMANCE",
                "severity": severity,
                "metric": "error_rate",
                "observed_value": error_rate,
                "threshold": thresholds.get('error_rate_increase', 0.10),
                "message": f"High error rate detected: {error_rate:.3f}",
                "deduplication_key": "high_error_rate",
                "status": "OPEN"
            })
        
        return alerts
    
    def _evaluate_data_quality_alerts(self,
                                     run_id: str,
                                     model_id: int,
                                     version_id: Optional[int],
                                     results: Dict[str, Any],
                                     thresholds: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Evaluate data quality-related alerts.
        
        Args:
            run_id: Monitoring run ID
            model_id: Model ID
            version_id: Model version ID
            results: Investigation results
            thresholds: Configured thresholds
        
        Returns:
            List of data quality alerts
        """
        alerts = []
        
        # Check missing data
        missing_data = results.get('missing_data', {})
        missing_findings = missing_data.get('findings', {})
        overall_missing = missing_findings.get('overall_missing_rate', 0)
        
        if overall_missing >= thresholds.get('missing_rate', 0.05):
            severity = "HIGH" if overall_missing >= 0.10 else "MEDIUM"
            alerts.append({
                "alert_id": f"alert_{uuid.uuid4().hex[:8]}",
                "model_id": model_id,
                "version_id": version_id,
                "run_id": run_id,
                "alert_type": "DATA_QUALITY",
                "severity": severity,
                "metric": "missing_rate",
                "observed_value": overall_missing,
                "threshold": thresholds.get('missing_rate', 0.05),
                "message": f"High missing data rate: {overall_missing:.3f}",
                "deduplication_key": "high_missing_rate",
                "status": "OPEN"
            })
        
        return alerts
    
    def _evaluate_calibration_alerts(self,
                                     run_id: str,
                                     model_id: int,
                                     version_id: Optional[int],
                                     results: Dict[str, Any],
                                     thresholds: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Evaluate calibration-related alerts.
        
        Args:
            run_id: Monitoring run ID
            model_id: Model ID
            version_id: Model version ID
            results: Investigation results
            thresholds: Configured thresholds
        
        Returns:
            List of calibration alerts
        """
        alerts = []
        
        # Check calibration
        calibration = results.get('calibration', {})
        cal_findings = calibration.get('findings', {})
        ece = cal_findings.get('ece', 0)
        
        if ece >= thresholds.get('ece', 0.15):
            severity = "HIGH" if ece >= 0.25 else "MEDIUM"
            alerts.append({
                "alert_id": f"alert_{uuid.uuid4().hex[:8]}",
                "model_id": model_id,
                "version_id": version_id,
                "run_id": run_id,
                "alert_type": "CALIBRATION",
                "severity": severity,
                "metric": "ECE",
                "observed_value": ece,
                "threshold": thresholds.get('ece', 0.15),
                "message": f"Poor calibration detected: ECE={ece:.3f}",
                "deduplication_key": "poor_calibration",
                "status": "OPEN"
            })
        
        return alerts
    
    def _calculate_drift_severity(self, psi: float, ks_stat: float) -> str:
        """
        Calculate drift severity based on PSI and KS statistic.
        
        Args:
            psi: Population Stability Index
            ks_stat: Kolmogorov-Smirnov statistic
        
        Returns:
            Severity level: LOW, MEDIUM, HIGH, CRITICAL
        """
        if psi >= 0.50 or ks_stat >= 0.50:
            return "CRITICAL"
        elif psi >= 0.30 or ks_stat >= 0.40:
            return "HIGH"
        elif psi >= 0.20 or ks_stat >= 0.30:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _store_alert(self, alert: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Store alert in database with deduplication.
        
        Args:
            alert: Alert dictionary
        
        Returns:
            Stored alert dictionary or None if duplicate
        """
        from webapp.models import Alert
        
        if not self.db_session:
            return alert
        
        # Check for duplicate open alert with same deduplication key
        dedup_key = alert.get('deduplication_key')
        if dedup_key:
            existing = self.db_session.query(Alert).filter(
                Alert.model_id == alert['model_id'],
                Alert.deduplication_key == dedup_key,
                Alert.status == 'OPEN'
            ).first()
            
            if existing:
                # Update existing alert instead of creating new one
                existing.updated_at = datetime.utcnow()
                existing.observed_value = alert.get('observed_value')
                existing.message = alert.get('message')
                self.db_session.commit()
                return None
        
        # Create new alert
        db_alert = Alert(
            alert_id=alert['alert_id'],
            model_id=alert['model_id'],
            version_id=alert.get('version_id'),
            run_id=alert['run_id'],
            alert_type=alert['alert_type'],
            severity=alert['severity'],
            metric=alert['metric'],
            observed_value=alert.get('observed_value'),
            threshold=alert.get('threshold'),
            message=alert.get('message'),
            status=alert['status'],
            deduplication_key=alert.get('deduplication_key')
        )
        
        self.db_session.add(db_alert)
        self.db_session.commit()
        
        return alert
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID
        
        Returns:
            True if successful, False otherwise
        """
        from webapp.models import Alert
        
        if not self.db_session:
            return False
        
        alert = self.db_session.query(Alert).filter_by(alert_id=alert_id).first()
        
        if not alert:
            return False
        
        alert.status = 'ACKNOWLEDGED'
        alert.acknowledged_at = datetime.utcnow()
        alert.updated_at = datetime.utcnow()
        self.db_session.commit()
        
        return True
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: Alert ID
        
        Returns:
            True if successful, False otherwise
        """
        from webapp.models import Alert
        
        if not self.db_session:
            return False
        
        alert = self.db_session.query(Alert).filter_by(alert_id=alert_id).first()
        
        if not alert:
            return False
        
        alert.status = 'RESOLVED'
        alert.resolved_at = datetime.utcnow()
        alert.updated_at = datetime.utcnow()
        self.db_session.commit()
        
        return True
    
    def get_alerts(self,
                   model_id: Optional[int] = None,
                   version_id: Optional[int] = None,
                   status: Optional[str] = None,
                   severity: Optional[str] = None,
                   limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve alerts with optional filters.
        
        Args:
            model_id: Filter by model ID
            version_id: Filter by version ID
            status: Filter by status
            severity: Filter by severity
            limit: Maximum number of alerts to return
        
        Returns:
            List of alert dictionaries
        """
        from webapp.models import Alert
        
        if not self.db_session:
            return []
        
        query = self.db_session.query(Alert)
        
        if model_id:
            query = query.filter_by(model_id=model_id)
        
        if version_id:
            query = query.filter_by(version_id=version_id)
        
        if status:
            query = query.filter_by(status=status)
        
        if severity:
            query = query.filter_by(severity=severity)
        
        query = query.order_by(Alert.created_at.desc()).limit(limit)
        
        alerts = query.all()
        
        return [
            {
                "alert_id": alert.alert_id,
                "model_id": alert.model_id,
                "version_id": alert.version_id,
                "run_id": alert.run_id,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "metric": alert.metric,
                "observed_value": alert.observed_value,
                "threshold": alert.threshold,
                "message": alert.message,
                "status": alert.status,
                "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                "created_at": alert.created_at.isoformat() if alert.created_at else None
            }
            for alert in alerts
        ]
