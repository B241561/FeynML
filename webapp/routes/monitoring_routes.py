"""
Monitoring API Routes
=====================

API endpoints for production monitoring functionality.
"""

from flask import Blueprint, request, jsonify
from webapp.extensions import db
from webapp.models import MonitoredModel, ModelVersion, MonitoringConfig, MonitoringRun, Alert
from webapp.services.monitoring_service import MonitoringService
from webapp.services.alert_engine import AlertEngine
import uuid

monitoring_bp = Blueprint('monitoring', __name__)


# ============================================================================
# Model Registry Endpoints
# ============================================================================

@monitoring_bp.route('/api/models', methods=['POST'])
def create_model():
    """
    Create a new monitored model.
    
    Request Body:
        {
            "model_id": "fraud-model",
            "model_name": "Fraud Detection Model",
            "description": "Detects fraudulent transactions",
            "task_type": "classification",
            "owner": "data-science-team",
            "metadata": {}
        }
    """
    try:
        data = request.get_json()
        
        # Validation
        if not data or not data.get('model_id'):
            return jsonify({"error": "model_id is required"}), 400
        
        if not data.get('model_name'):
            return jsonify({"error": "model_name is required"}), 400
        
        if not data.get('task_type'):
            return jsonify({"error": "task_type is required"}), 400
        
        # Check for duplicate model_id
        existing = MonitoredModel.query.filter_by(model_id=data['model_id']).first()
        if existing:
            return jsonify({"error": "Model with this model_id already exists"}), 409
        
        # Create model
        model = MonitoredModel(
            model_id=data['model_id'],
            model_name=data['model_name'],
            description=data.get('description'),
            task_type=data['task_type'],
            owner=data.get('owner'),
            metadata=data.get('metadata')
        )
        
        db.session.add(model)
        db.session.commit()
        
        return jsonify({
            "model_id": model.model_id,
            "model_name": model.model_name,
            "description": model.description,
            "task_type": model.task_type,
            "owner": model.owner,
            "created_at": model.created_at.isoformat() if model.created_at else None
        }), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@monitoring_bp.route('/api/models', methods=['GET'])
def list_models():
    """
    List all monitored models.
    
    Query Params:
        task_type: Filter by task type (optional)
    """
    try:
        task_type = request.args.get('task_type')
        
        query = MonitoredModel.query
        
        if task_type:
            query = query.filter_by(task_type=task_type)
        
        models = query.order_by(MonitoredModel.created_at.desc()).all()
        
        return jsonify([
            {
                "model_id": model.model_id,
                "model_name": model.model_name,
                "description": model.description,
                "task_type": model.task_type,
                "owner": model.owner,
                "created_at": model.created_at.isoformat() if model.created_at else None,
                "updated_at": model.updated_at.isoformat() if model.updated_at else None
            }
            for model in models
        ]), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@monitoring_bp.route('/api/models/<model_id>', methods=['GET'])
def get_model(model_id):
    """
    Get a specific monitored model by ID.
    """
    try:
        model = MonitoredModel.query.filter_by(model_id=model_id).first()
        
        if not model:
            return jsonify({"error": "Model not found"}), 404
        
        return jsonify({
            "model_id": model.model_id,
            "model_name": model.model_name,
            "description": model.description,
            "task_type": model.task_type,
            "owner": model.owner,
            "metadata": model.metadata,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "updated_at": model.updated_at.isoformat() if model.updated_at else None
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Model Version Endpoints
# ============================================================================

@monitoring_bp.route('/api/models/<model_id>/versions', methods=['POST'])
def create_version(model_id):
    """
    Create a new model version.
    
    Request Body:
        {
            "version_name": "v1",
            "status": "ACTIVE",
            "task_type": "classification",
            "evaluation_metadata": {},
            "deployment_metadata": {}
        }
    """
    try:
        data = request.get_json()
        
        # Validation
        if not data or not data.get('version_name'):
            return jsonify({"error": "version_name is required"}), 400
        
        # Check model exists
        model = MonitoredModel.query.filter_by(model_id=model_id).first()
        if not model:
            return jsonify({"error": "Model not found"}), 404
        
        # Check for duplicate version
        existing = ModelVersion.query.filter_by(
            model_id=model.id,
            version_name=data['version_name']
        ).first()
        if existing:
            return jsonify({"error": "Version with this name already exists for this model"}), 409
        
        # Create version
        version_id = f"{model_id}_{data['version_name']}"
        version = ModelVersion(
            model_id=model.id,
            version_id=version_id,
            version_name=data['version_name'],
            status=data.get('status', 'ACTIVE'),
            task_type=data.get('task_type', model.task_type),
            evaluation_metadata=data.get('evaluation_metadata'),
            deployment_metadata=data.get('deployment_metadata')
        )
        
        db.session.add(version)
        db.session.commit()
        
        return jsonify({
            "version_id": version.version_id,
            "version_name": version.version_name,
            "status": version.status,
            "task_type": version.task_type,
            "created_at": version.created_at.isoformat() if version.created_at else None
        }), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@monitoring_bp.route('/api/models/<model_id>/versions', methods=['GET'])
def list_versions(model_id):
    """
    List all versions for a model.
    """
    try:
        model = MonitoredModel.query.filter_by(model_id=model_id).first()
        
        if not model:
            return jsonify({"error": "Model not found"}), 404
        
        versions = ModelVersion.query.filter_by(model_id=model.id).order_by(
            ModelVersion.created_at.desc()
        ).all()
        
        return jsonify([
            {
                "version_id": v.version_id,
                "version_name": v.version_name,
                "status": v.status,
                "task_type": v.task_type,
                "evaluation_metadata": v.evaluation_metadata,
                "deployment_metadata": v.deployment_metadata,
                "created_at": v.created_at.isoformat() if v.created_at else None
            }
            for v in versions
        ]), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Monitoring Configuration Endpoints
# ============================================================================

@monitoring_bp.route('/api/monitoring/config', methods=['POST'])
def create_monitoring_config():
    """
    Create a monitoring configuration.
    
    Request Body:
        {
            "model_id": "fraud-model",
            "config_name": "production-config",
            "thresholds": {"psi": 0.20, "error_rate_increase": 0.10},
            "schedule_enabled": false,
            "schedule_interval": "daily"
        }
    """
    try:
        data = request.get_json()
        
        # Validation
        if not data or not data.get('model_id'):
            return jsonify({"error": "model_id is required"}), 400
        
        if not data.get('config_name'):
            return jsonify({"error": "config_name is required"}), 400
        
        # Check model exists
        model = MonitoredModel.query.filter_by(model_id=data['model_id']).first()
        if not model:
            return jsonify({"error": "Model not found"}), 404
        
        # Create config
        config = MonitoringConfig(
            model_id=model.id,
            config_name=data['config_name'],
            is_active=data.get('is_active', True),
            monitor_data_drift=data.get('monitor_data_drift', True),
            monitor_prediction_drift=data.get('monitor_prediction_drift', True),
            monitor_performance=data.get('monitor_performance', True),
            monitor_data_quality=data.get('monitor_data_quality', True),
            monitor_calibration=data.get('monitor_calibration', True),
            monitor_error_rate=data.get('monitor_error_rate', True),
            monitor_segment_degradation=data.get('monitor_segment_degradation', True),
            monitor_temporal_degradation=data.get('monitor_temporal_degradation', True),
            thresholds=data.get('thresholds'),
            schedule_enabled=data.get('schedule_enabled', False),
            schedule_interval=data.get('schedule_interval'),
            schedule_timezone=data.get('schedule_timezone')
        )
        
        db.session.add(config)
        db.session.commit()
        
        return jsonify({
            "config_id": config.id,
            "config_name": config.config_name,
            "model_id": data['model_id'],
            "is_active": config.is_active,
            "thresholds": config.thresholds,
            "schedule_enabled": config.schedule_enabled,
            "schedule_interval": config.schedule_interval,
            "created_at": config.created_at.isoformat() if config.created_at else None
        }), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@monitoring_bp.route('/api/monitoring/config/<model_id>', methods=['GET'])
def get_monitoring_config(model_id):
    """
    Get monitoring configuration for a model.
    """
    try:
        model = MonitoredModel.query.filter_by(model_id=model_id).first()
        
        if not model:
            return jsonify({"error": "Model not found"}), 404
        
        config = MonitoringConfig.query.filter_by(model_id=model.id, is_active=True).first()
        
        if not config:
            return jsonify({"error": "No active monitoring configuration found"}), 404
        
        return jsonify({
            "config_id": config.id,
            "config_name": config.config_name,
            "model_id": model_id,
            "is_active": config.is_active,
            "monitor_data_drift": config.monitor_data_drift,
            "monitor_prediction_drift": config.monitor_prediction_drift,
            "monitor_performance": config.monitor_performance,
            "monitor_data_quality": config.monitor_data_quality,
            "monitor_calibration": config.monitor_calibration,
            "monitor_error_rate": config.monitor_error_rate,
            "monitor_segment_degradation": config.monitor_segment_degradation,
            "monitor_temporal_degradation": config.monitor_temporal_degradation,
            "thresholds": config.thresholds,
            "schedule_enabled": config.schedule_enabled,
            "schedule_interval": config.schedule_interval,
            "created_at": config.created_at.isoformat() if config.created_at else None
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Monitoring Run Endpoints
# ============================================================================

@monitoring_bp.route('/api/monitoring/run', methods=['POST'])
def create_monitoring_run():
    """
    Create and execute a monitoring run.
    
    Request Body:
        {
            "model_id": "fraud-model",
            "version_name": "v1",
            "trigger_type": "MANUAL",
            "reference_data": {...},
            "current_data": {...}
        }
    """
    try:
        data = request.get_json()
        
        # Validation
        if not data or not data.get('model_id'):
            return jsonify({"error": "model_id is required"}), 400
        
        # Check model exists
        model = MonitoredModel.query.filter_by(model_id=data['model_id']).first()
        if not model:
            return jsonify({"error": "Model not found"}), 404
        
        # Get version if specified
        version_id = None
        if data.get('version_name'):
            version = ModelVersion.query.filter_by(
                model_id=model.id,
                version_name=data['version_name']
            ).first()
            if version:
                version_id = version.id
        
        # Get active config
        config = MonitoringConfig.query.filter_by(model_id=model.id, is_active=True).first()
        config_id = config.id if config else None
        
        # Create monitoring run
        service = MonitoringService(db.session)
        run_id = service.create_monitoring_run(
            model_id=model.id,
            version_id=version_id,
            config_id=config_id,
            trigger_type=data.get('trigger_type', 'MANUAL'),
            reference_info=data.get('reference_info'),
            current_info=data.get('current_info')
        )
        
        return jsonify({
            "run_id": run_id,
            "status": "PENDING",
            "model_id": data['model_id'],
            "version_name": data.get('version_name')
        }), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@monitoring_bp.route('/api/monitoring/runs', methods=['GET'])
def list_monitoring_runs():
    """
    List monitoring runs with optional filters.
    
    Query Params:
        model_id: Filter by model ID
        version_id: Filter by version ID
        limit: Maximum number of runs (default: 50)
    """
    try:
        model_id = request.args.get('model_id')
        version_id = request.args.get('version_id')
        limit = int(request.args.get('limit', 50))
        
        # Convert model_id to database ID if provided
        db_model_id = None
        if model_id:
            model = MonitoredModel.query.filter_by(model_id=model_id).first()
            if model:
                db_model_id = model.id
        
        # Convert version_id to database ID if provided
        db_version_id = None
        if version_id:
            version = ModelVersion.query.filter_by(version_id=version_id).first()
            if version:
                db_version_id = version.id
        
        service = MonitoringService(db.session)
        runs = service.get_monitoring_history(
            model_id=db_model_id,
            version_id=db_version_id,
            limit=limit
        )
        
        return jsonify(runs), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@monitoring_bp.route('/api/monitoring/runs/<run_id>', methods=['GET'])
def get_monitoring_run(run_id):
    """
    Get a specific monitoring run by ID.
    """
    try:
        service = MonitoringService(db.session)
        run = service.get_monitoring_run(run_id)
        
        if not run:
            return jsonify({"error": "Monitoring run not found"}), 404
        
        return jsonify(run), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Alert Endpoints
# ============================================================================

@monitoring_bp.route('/api/alerts', methods=['GET'])
def list_alerts():
    """
    List alerts with optional filters.
    
    Query Params:
        model_id: Filter by model ID
        version_id: Filter by version ID
        status: Filter by status (OPEN, ACKNOWLEDGED, RESOLVED)
        severity: Filter by severity (LOW, MEDIUM, HIGH, CRITICAL)
        limit: Maximum number of alerts (default: 50)
    """
    try:
        model_id = request.args.get('model_id')
        version_id = request.args.get('version_id')
        status = request.args.get('status')
        severity = request.args.get('severity')
        limit = int(request.args.get('limit', 50))
        
        # Convert model_id to database ID if provided
        db_model_id = None
        if model_id:
            model = MonitoredModel.query.filter_by(model_id=model_id).first()
            if model:
                db_model_id = model.id
        
        # Convert version_id to database ID if provided
        db_version_id = None
        if version_id:
            version = ModelVersion.query.filter_by(version_id=version_id).first()
            if version:
                db_version_id = version.id
        
        engine = AlertEngine(db.session)
        alerts = engine.get_alerts(
            model_id=db_model_id,
            version_id=db_version_id,
            status=status,
            severity=severity,
            limit=limit
        )
        
        return jsonify(alerts), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@monitoring_bp.route('/api/alerts/<alert_id>', methods=['GET'])
def get_alert(alert_id):
    """
    Get a specific alert by ID.
    """
    try:
        alert = Alert.query.filter_by(alert_id=alert_id).first()
        
        if not alert:
            return jsonify({"error": "Alert not found"}), 404
        
        return jsonify({
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
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@monitoring_bp.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    """
    Acknowledge an alert.
    """
    try:
        engine = AlertEngine(db.session)
        success = engine.acknowledge_alert(alert_id)
        
        if not success:
            return jsonify({"error": "Alert not found"}), 404
        
        return jsonify({"status": "acknowledged"}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@monitoring_bp.route('/api/alerts/<alert_id>/resolve', methods=['POST'])
def resolve_alert(alert_id):
    """
    Resolve an alert.
    """
    try:
        engine = AlertEngine(db.session)
        success = engine.resolve_alert(alert_id)
        
        if not success:
            return jsonify({"error": "Alert not found"}), 404
        
        return jsonify({"status": "resolved"}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
