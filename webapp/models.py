from datetime import datetime, timedelta
import secrets
from flask_login import UserMixin
from .extensions import db, bcrypt

class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(180), nullable=False, unique=True, index=True)
    password_hash = db.Column(db.String(128), nullable=False)
    joined_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    analyses = db.relationship('Analysis', backref='user', lazy='dynamic')
    reports = db.relationship('Report', backref='user', lazy='dynamic')
    datasets = db.relationship('Dataset', backref='user', lazy='dynamic')
    notes = db.relationship('Note', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    tasks = db.relationship('Task', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    usage_stat = db.relationship('UsageStat', uselist=False, backref='user')

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

    @property
    def total_analyses(self):
        return self.analyses.count()

    @property
    def datasets_uploaded(self):
        return self.datasets.count()

    @property
    def reports_generated(self):
        return self.reports.count()

    @property
    def storage_used(self):
        if self.usage_stat:
            return self.usage_stat.storage_bytes
        return sum(dataset.size_bytes or 0 for dataset in self.datasets)

    @property
    def join_date(self):
        return self.joined_at.strftime('%b %d, %Y')

    def __repr__(self):
        return f'<User {self.email}>'

class Analysis(db.Model):
    __tablename__ = 'analyses'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    dataset_name = db.Column(db.String(255), nullable=False)
    report_id = db.Column(db.String(128), nullable=True, index=True)
    status = db.Column(db.String(50), nullable=False, default='PENDING')
    config = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)

    reports = db.relationship('Report', backref='analysis', lazy='dynamic')

    def __repr__(self):
        return f'<Analysis {self.id} user={self.user_id} report={self.report_id}>'

class Report(db.Model):
    __tablename__ = 'reports'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analyses.id'), nullable=True)
    report_id = db.Column(db.String(128), nullable=False, unique=True, index=True)
    title = db.Column(db.String(255), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    size_bytes = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f'<Report {self.report_id}>'

class Dataset(db.Model):
    __tablename__ = 'datasets'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    original_name = db.Column(db.String(255), nullable=False)
    stored_name = db.Column(db.String(255), nullable=False)
    upload_path = db.Column(db.String(1024), nullable=False)
    size_bytes = db.Column(db.Integer, nullable=False, default=0)
    uploaded_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f'<Dataset {self.original_name}>'

class Note(db.Model):
    __tablename__ = 'notes'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False, default='')
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<Note {self.title} user={self.user_id}>'

class Task(db.Model):
    __tablename__ = 'tasks'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    priority = db.Column(db.String(20), nullable=False, default='Medium')
    due_date = db.Column(db.Date, nullable=True)
    is_completed = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<Task {self.title} priority={self.priority} completed={self.is_completed}>'

class UsageStat(db.Model):
    __tablename__ = 'usage_stats'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True)
    analyses_total = db.Column(db.Integer, nullable=False, default=0)
    uploads_total = db.Column(db.Integer, nullable=False, default=0)
    storage_bytes = db.Column(db.BigInteger, nullable=False, default=0)
    last_analysis_at = db.Column(db.DateTime, nullable=True)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<UsageStat user={self.user_id} analyses={self.analyses_total}>'


class AppSetting(db.Model):
    """Simple key/value store for application settings."""
    __tablename__ = 'app_settings'

    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(128), nullable=False, unique=True, index=True)
    value = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<AppSetting {self.key}={self.value}>'


class AdminProfile(db.Model):
    """
    Stores the single admin account (Feyn_admin).
    Only ONE admin account is allowed to exist.
    """
    __tablename__ = 'admin_profiles'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(128), nullable=False, unique=True, index=True)
    email = db.Column(db.String(180), nullable=False, unique=True, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    otp_tokens = db.relationship('OTPToken', backref='admin', lazy='dynamic', cascade='all, delete-orphan')

    def set_password(self, password):
        """Hash and set the admin password using bcrypt."""
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        """Verify the provided password against the hash."""
        return bcrypt.check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<AdminProfile username={self.username}>'


class OTPToken(db.Model):
    """
    Stores OTP tokens for admin password reset flow.
    Each OTP is single-use with a 10-minute expiry.
    """
    __tablename__ = 'otp_tokens'

    id = db.Column(db.Integer, primary_key=True)
    admin_id = db.Column(db.Integer, db.ForeignKey('admin_profiles.id'), nullable=False, index=True)
    otp_code = db.Column(db.String(6), nullable=False, index=True)
    is_used = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)

    @staticmethod
    def generate_otp():
        """Generate a secure 6-digit OTP."""
        return ''.join([str(secrets.randbelow(10)) for _ in range(6)])

    @classmethod
    def create_otp_for_admin(cls, admin_id, expiry_minutes=10):
        """Create a new OTP for the specified admin with given expiry in minutes."""
        otp_code = cls.generate_otp()
        expires_at = datetime.utcnow() + timedelta(minutes=expiry_minutes)
        otp_token = cls(admin_id=admin_id, otp_code=otp_code, expires_at=expires_at)
        db.session.add(otp_token)
        db.session.commit()
        return otp_token

    def is_valid(self):
        """Check if OTP is valid (not used and not expired)."""
        return not self.is_used and self.expires_at > datetime.utcnow()

    def mark_as_used(self):
        """Mark OTP as used."""
        self.is_used = True
        db.session.commit()

    def __repr__(self):
        return f'<OTPToken admin_id={self.admin_id} is_used={self.is_used}>'


# ============================================================================
# Phase 3: Production Monitoring Models
# ============================================================================

class MonitoredModel(db.Model):
    """Model Registry - tracks registered ML models for monitoring."""
    __tablename__ = 'monitored_models'

    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.String(128), nullable=False, unique=True, index=True)
    model_name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    task_type = db.Column(db.String(50), nullable=False)  # classification, regression
    owner = db.Column(db.String(255), nullable=True)
    model_metadata = db.Column(db.JSON, nullable=True)  # Renamed from 'metadata' (reserved keyword)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    versions = db.relationship('ModelVersion', backref='model', lazy='dynamic', cascade='all, delete-orphan')
    monitoring_configs = db.relationship('MonitoringConfig', backref='model', lazy='dynamic', cascade='all, delete-orphan')
    monitoring_runs = db.relationship('MonitoringRun', backref='model', lazy='dynamic', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<MonitoredModel {self.model_id}>'


class ModelVersion(db.Model):
    """Model Versions - tracks different versions of a monitored model."""
    __tablename__ = 'model_versions'

    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, db.ForeignKey('monitored_models.id'), nullable=False, index=True)
    version_id = db.Column(db.String(128), nullable=False, index=True)
    version_name = db.Column(db.String(50), nullable=False)  # v1, v2, etc.
    status = db.Column(db.String(20), nullable=False, default='ACTIVE')  # ACTIVE, INACTIVE, ARCHIVED
    task_type = db.Column(db.String(50), nullable=False)
    evaluation_metadata = db.Column(db.JSON, nullable=True)  # metrics, training info
    deployment_metadata = db.Column(db.JSON, nullable=True)  # deployment info
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    monitoring_runs = db.relationship('MonitoringRun', backref='version', lazy='dynamic')

    __table_args__ = (
        db.UniqueConstraint('model_id', 'version_name', name='unique_model_version'),
    )

    def __repr__(self):
        return f'<ModelVersion {self.version_id}>'


class MonitoringConfig(db.Model):
    """Monitoring Configuration - defines what to monitor for a model."""
    __tablename__ = 'monitoring_configs'

    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, db.ForeignKey('monitored_models.id'), nullable=False, index=True)
    config_name = db.Column(db.String(255), nullable=False)
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    
    # Monitoring signals
    monitor_data_drift = db.Column(db.Boolean, nullable=False, default=True)
    monitor_prediction_drift = db.Column(db.Boolean, nullable=False, default=True)
    monitor_performance = db.Column(db.Boolean, nullable=False, default=True)
    monitor_data_quality = db.Column(db.Boolean, nullable=False, default=True)
    monitor_calibration = db.Column(db.Boolean, nullable=False, default=True)
    monitor_error_rate = db.Column(db.Boolean, nullable=False, default=True)
    monitor_segment_degradation = db.Column(db.Boolean, nullable=False, default=True)
    monitor_temporal_degradation = db.Column(db.Boolean, nullable=False, default=True)
    
    # Thresholds (JSON for flexibility)
    thresholds = db.Column(db.JSON, nullable=True)  # {"psi": 0.20, "error_rate_increase": 0.10}
    
    # Scheduling
    schedule_enabled = db.Column(db.Boolean, nullable=False, default=False)
    schedule_interval = db.Column(db.String(50), nullable=True)  # hourly, daily, weekly
    schedule_timezone = db.Column(db.String(50), nullable=True)
    
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    monitoring_runs = db.relationship('MonitoringRun', backref='config', lazy='dynamic')

    def __repr__(self):
        return f'<MonitoringConfig {self.config_name}>'


class MonitoringRun(db.Model):
    """Monitoring Runs - stores individual monitoring execution results."""
    __tablename__ = 'monitoring_runs'

    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.String(128), nullable=False, unique=True, index=True)
    model_id = db.Column(db.Integer, db.ForeignKey('monitored_models.id'), nullable=False, index=True)
    version_id = db.Column(db.Integer, db.ForeignKey('model_versions.id'), nullable=True, index=True)
    config_id = db.Column(db.Integer, db.ForeignKey('monitoring_configs.id'), nullable=True)
    
    # Run metadata
    trigger_type = db.Column(db.String(20), nullable=False)  # MANUAL, SCHEDULED, API, AUTOMATED
    status = db.Column(db.String(20), nullable=False, default='PENDING')  # PENDING, RUNNING, SUCCESS, PARTIAL, FAILED
    started_at = db.Column(db.DateTime, nullable=False)
    completed_at = db.Column(db.DateTime, nullable=True)
    
    # Input information
    reference_info = db.Column(db.JSON, nullable=True)  # reference dataset info
    current_info = db.Column(db.JSON, nullable=True)  # current dataset info
    
    # Results (stores InvestigationRun results)
    results = db.Column(db.JSON, nullable=True)
    
    # Summary
    health_status = db.Column(db.String(20), nullable=True)  # HEALTHY, WARNING, CRITICAL
    confidence_score = db.Column(db.Integer, nullable=True)
    
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    alerts = db.relationship('Alert', backref='monitoring_run', lazy='dynamic', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<MonitoringRun {self.run_id}>'


class Alert(db.Model):
    """Alerts - stores monitoring alerts generated from threshold violations."""
    __tablename__ = 'alerts'

    id = db.Column(db.Integer, primary_key=True)
    alert_id = db.Column(db.String(128), nullable=False, unique=True, index=True)
    model_id = db.Column(db.Integer, db.ForeignKey('monitored_models.id'), nullable=False, index=True)
    version_id = db.Column(db.Integer, db.ForeignKey('model_versions.id'), nullable=True, index=True)
    run_id = db.Column(db.Integer, db.ForeignKey('monitoring_runs.id'), nullable=False, index=True)
    
    # Alert details
    alert_type = db.Column(db.String(50), nullable=False)  # DRIFT, PERFORMANCE, DATA_QUALITY, CALIBRATION
    severity = db.Column(db.String(20), nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL
    metric = db.Column(db.String(100), nullable=False)  # PSI, accuracy, missing_rate, etc.
    observed_value = db.Column(db.Float, nullable=True)
    threshold = db.Column(db.Float, nullable=True)
    message = db.Column(db.Text, nullable=True)
    
    # Alert lifecycle
    status = db.Column(db.String(20), nullable=False, default='OPEN')  # OPEN, ACKNOWLEDGED, RESOLVED
    acknowledged_at = db.Column(db.DateTime, nullable=True)
    resolved_at = db.Column(db.DateTime, nullable=True)
    
    # Deduplication
    deduplication_key = db.Column(db.String(255), nullable=True, index=True)  # for grouping similar alerts
    
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<Alert {self.alert_id} severity={self.severity}>'
