"""
Monitoring Service
==================

Orchestrates production monitoring runs using Phase 1 and Phase 2 engines.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd


class MonitoringService:
    """
    Service for executing production monitoring runs.
    
    Orchestrates Phase 1 validation and Phase 2 investigation engines
    for continuous model monitoring.
    """
    
    def __init__(self, db_session=None):
        """
        Initialize Monitoring Service.
        
        Args:
            db_session: SQLAlchemy database session
        """
        self.db_session = db_session
    
    def create_monitoring_run(self, 
                             model_id: int,
                             version_id: Optional[int],
                             config_id: Optional[int],
                             trigger_type: str = "MANUAL",
                             reference_info: Optional[Dict] = None,
                             current_info: Optional[Dict] = None) -> str:
        """
        Create a new monitoring run.
        
        Args:
            model_id: ID of the monitored model
            version_id: ID of the model version
            config_id: ID of the monitoring configuration
            trigger_type: Type of trigger (MANUAL, SCHEDULED, API, AUTOMATED)
            reference_info: Reference dataset information
            current_info: Current dataset information
        
        Returns:
            run_id: Unique identifier for the monitoring run
        """
        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        from webapp.models import MonitoringRun
        
        monitoring_run = MonitoringRun(
            run_id=run_id,
            model_id=model_id,
            version_id=version_id,
            config_id=config_id,
            trigger_type=trigger_type,
            status="PENDING",
            started_at=datetime.utcnow(),
            reference_info=reference_info,
            current_info=current_info
        )
        
        if self.db_session:
            self.db_session.add(monitoring_run)
            self.db_session.commit()
        
        return run_id
    
    def execute_monitoring_run(self,
                              run_id: str,
                              reference_data: pd.DataFrame,
                              current_data: pd.DataFrame,
                              y_true: Optional[pd.Series] = None,
                              y_pred: Optional[pd.Series] = None,
                              y_prob: Optional[pd.Series] = None,
                              target_column: Optional[str] = None,
                              feature_columns: Optional[List[str]] = None,
                              timestamp_column: Optional[str] = None,
                              config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute a monitoring run using Phase 1 and Phase 2 engines.
        
        Args:
            run_id: Monitoring run ID
            reference_data: Reference dataset
            current_data: Current monitoring dataset
            y_true: True labels
            y_pred: Predictions
            y_prob: Prediction probabilities
            target_column: Name of target column
            feature_columns: List of feature column names
            timestamp_column: Name of timestamp column
            config: Monitoring configuration
        
        Returns:
            Dictionary with monitoring results
        """
        from webapp.models import MonitoringRun
        from webapp.services.analysis_runner import AnalysisRunner
        
        # Update run status to RUNNING
        if self.db_session:
            monitoring_run = self.db_session.query(MonitoringRun).filter_by(run_id=run_id).first()
            if monitoring_run:
                monitoring_run.status = "RUNNING"
                self.db_session.commit()
        
        try:
            # Prepare kwargs for AnalysisRunner
            runner_kwargs = {
                'target_column': target_column,
                'feature_columns': feature_columns,
                'reference_data': reference_data,
                'current_data': current_data
            }
            
            # Initialize AnalysisRunner
            runner = AnalysisRunner(
                df=current_data,
                target=target_column,
                features=feature_columns,
                reference_df=reference_data,
                **runner_kwargs
            )
            
            # Run full investigation (Phase 1 + Phase 2)
            runner.run()
            
            # Extract results
            results = runner.results
            
            # Determine overall health status
            health_status = self._determine_health_status(results)
            confidence_score = results.get('root_cause', {}).get('confidence', 0)
            
            # Update monitoring run with results
            if self.db_session:
                monitoring_run = self.db_session.query(MonitoringRun).filter_by(run_id=run_id).first()
                if monitoring_run:
                    monitoring_run.status = "SUCCESS"
                    monitoring_run.completed_at = datetime.utcnow()
                    monitoring_run.results = results
                    monitoring_run.health_status = health_status
                    monitoring_run.confidence_score = confidence_score
                    self.db_session.commit()
            
            return {
                "run_id": run_id,
                "status": "SUCCESS",
                "health_status": health_status,
                "confidence_score": confidence_score,
                "results": results
            }
            
        except Exception as e:
            # Update monitoring run with failure
            if self.db_session:
                monitoring_run = self.db_session.query(MonitoringRun).filter_by(run_id=run_id).first()
                if monitoring_run:
                    monitoring_run.status = "FAILED"
                    monitoring_run.completed_at = datetime.utcnow()
                    monitoring_run.results = {"error": str(e)}
                    self.db_session.commit()
            
            return {
                "run_id": run_id,
                "status": "FAILED",
                "error": str(e)
            }
    
    def _determine_health_status(self, results: Dict[str, Any]) -> str:
        """
        Determine overall health status from investigation results.
        
        Args:
            results: Investigation results from AnalysisRunner
        
        Returns:
            Health status: HEALTHY, WARNING, CRITICAL
        """
        # Check root cause health status
        root_cause = results.get('root_cause', {})
        health_status = root_cause.get('health_status', 'Healthy')
        
        if health_status in ['Critical', 'CRITICAL']:
            return 'CRITICAL'
        elif health_status in ['Warning', 'WARNING']:
            return 'WARNING'
        else:
            return 'HEALTHY'
    
    def get_monitoring_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a monitoring run by ID.
        
        Args:
            run_id: Monitoring run ID
        
        Returns:
            Dictionary with monitoring run data or None
        """
        from webapp.models import MonitoringRun
        
        if not self.db_session:
            return None
        
        monitoring_run = self.db_session.query(MonitoringRun).filter_by(run_id=run_id).first()
        
        if not monitoring_run:
            return None
        
        return {
            "run_id": monitoring_run.run_id,
            "model_id": monitoring_run.model_id,
            "version_id": monitoring_run.version_id,
            "config_id": monitoring_run.config_id,
            "trigger_type": monitoring_run.trigger_type,
            "status": monitoring_run.status,
            "started_at": monitoring_run.started_at.isoformat() if monitoring_run.started_at else None,
            "completed_at": monitoring_run.completed_at.isoformat() if monitoring_run.completed_at else None,
            "reference_info": monitoring_run.reference_info,
            "current_info": monitoring_run.current_info,
            "health_status": monitoring_run.health_status,
            "confidence_score": monitoring_run.confidence_score,
            "created_at": monitoring_run.created_at.isoformat() if monitoring_run.created_at else None
        }
    
    def get_monitoring_history(self, 
                              model_id: Optional[int] = None,
                              version_id: Optional[int] = None,
                              limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve monitoring run history.
        
        Args:
            model_id: Filter by model ID
            version_id: Filter by version ID
            limit: Maximum number of runs to return
        
        Returns:
            List of monitoring run dictionaries
        """
        from webapp.models import MonitoringRun
        
        if not self.db_session:
            return []
        
        query = self.db_session.query(MonitoringRun)
        
        if model_id:
            query = query.filter_by(model_id=model_id)
        
        if version_id:
            query = query.filter_by(version_id=version_id)
        
        query = query.order_by(MonitoringRun.created_at.desc()).limit(limit)
        
        runs = query.all()
        
        return [
            {
                "run_id": run.run_id,
                "model_id": run.model_id,
                "version_id": run.version_id,
                "trigger_type": run.trigger_type,
                "status": run.status,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "health_status": run.health_status,
                "confidence_score": run.confidence_score,
                "created_at": run.created_at.isoformat() if run.created_at else None
            }
            for run in runs
        ]
