import os
import json
import re
import traceback
from datetime import datetime, timedelta
from functools import wraps
from uuid import uuid4
from urllib.parse import urljoin, urlparse
from werkzeug.utils import secure_filename
import io
import csv
import plotly
import plotly.graph_objects as go
import plotly.express as px

from webapp.chart_colors import FEYNML_CHART_COLORS

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file, send_from_directory
from sqlalchemy import func, or_
from flask_login import current_user, login_user, logout_user, login_required
from webapp.extensions import db, login_manager, bcrypt, migrate
from webapp.services.analysis_runner import AnalysisRunner
from webapp.services.email_service import EmailService

email_service = EmailService()
from webapp.models import (
    User,
    Dataset,
    Analysis,
    Report,
    AdminProfile,
    OTPToken,
    UsageStat,
    Note,
    Task,
    AppSetting,
)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

app = Flask(__name__)

# Register optional blueprints if their modules and optional deps are available.
try:
    from webapp.routes.phase4_routes import phase4_bp
    app.register_blueprint(phase4_bp)
except Exception:
    pass

try:
    from webapp.routes.chatbot_routes import chatbot_bp
    app.register_blueprint(chatbot_bp, url_prefix='/api/chatbot')
except Exception:
    pass
app.debug = True
app.config["PROPAGATE_EXCEPTIONS"] = True
app.config["TRAP_HTTP_EXCEPTIONS"] = True
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'feyn-ml-investigator-key-2024')

# Database configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
DATABASE_PATH = os.path.join(ROOT_DIR, 'feynml.db')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', f'sqlite:///{DATABASE_PATH}')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Path configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
REPORT_FOLDER = os.path.join(BASE_DIR, 'reports')
LOG_FILE = os.path.join(BASE_DIR, 'logs', 'app.log')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def _normalize_recommended_actions(actions):
    """Convert legacy recommendation blobs into structured action objects."""
    if not actions:
        return []

    section_pattern = re.compile(
        r"(Root Cause|Highest Priority|Evidence|Why it matters|Recommended workflow):\s*",
        re.IGNORECASE,
    )

    def _parse_steps(workflow_text):
        workflow_text = str(workflow_text or "").strip()
        if not workflow_text:
            return []

        numbered_steps = re.findall(
            r"(?:^|\n)\s*\d+\.\s*(.+?)(?=(?:\n\s*\d+\.\s)|$)",
            workflow_text,
            flags=re.DOTALL,
        )
        if numbered_steps:
            return [" ".join(step.split()) for step in numbered_steps if step.strip()]

        fallback = " ".join(workflow_text.split())
        return [fallback] if fallback else []

    normalized = []
    for action in actions:
        if isinstance(action, dict):
            title = str(action.get("title") or "Recommended action").strip()
            evidence = str(action.get("evidence") or "").strip()
            why = str(action.get("why") or "").strip()
            steps = action.get("steps") or []
            if isinstance(steps, str):
                steps = _parse_steps(steps)
            elif isinstance(steps, list):
                steps = [str(step).strip() for step in steps if str(step).strip()]
            normalized.append({
                "title": title or "Recommended action",
                "evidence": evidence,
                "why": why,
                "steps": steps,
            })
            continue

        text = str(action or "").strip()
        if not text:
            continue

        matches = list(section_pattern.finditer(text))
        sections = {}
        if matches:
            for idx, match in enumerate(matches):
                start = match.end()
                end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
                label = match.group(1).strip().lower()
                sections[label] = text[start:end].strip()

        root_cause = sections.get("root cause", "")
        highest_priority = sections.get("highest priority", "")
        title = root_cause or "Recommended action"
        if highest_priority:
            title = f"{title} (Highest priority: {highest_priority})"

        evidence = sections.get("evidence", "")
        why = sections.get("why it matters", "")
        steps = _parse_steps(sections.get("recommended workflow", ""))

        if not matches:
            why = " ".join(text.split())

        normalized.append({
            "title": title,
            "evidence": evidence,
            "why": why,
            "steps": steps,
        })

    return normalized
app.config['REPORT_FOLDER'] = REPORT_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'json'}
_df_cache = {}
# Lazy analysis runner: avoid importing heavy engine modules at import time
runner = None


def get_runner():
    """Lazily instantiate AnalysisRunner when first needed.

    Keeps module import lightweight for test collection. If the
    analysis runner or its optional dependencies are unavailable,
    this returns None and callers should handle that gracefully.
    """
    global runner
    if runner is None:
        try:
            from webapp.services.analysis_runner import AnalysisRunner
            runner = AnalysisRunner()
        except Exception:
            runner = None
    return runner

# Instantiate a runner on module import so startup and test code can access it,
# while get_runner() still preserves the lazy-loading semantics for route handlers.
runner = get_runner()

db.init_app(app)
bcrypt.init_app(app)
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'


def is_safe_url(target):
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and ref_url.netloc == test_url.netloc


def _is_admin_authenticated():
    return session.get('admin_authenticated', False) is True


def admin_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if not _is_admin_authenticated():
            next_url = request.path
            return redirect(url_for('admin_login', next=next_url))
        return view_func(*args, **kwargs)
    return wrapped_view


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _get_df(report_id: str, x_col: str = None, dataset_path: str = None):
    """
    Smart dataset loader.
    1. Check cache.
    2. If dataset_path provided and exists, use it.
    3. Pass 1 → scan every CSV in UPLOAD_FOLDER, pick the one whose
             headers contain x_col.  Correct for multi-CSV situations.
    4. Pass 2 → fallback to the largest CSV (best proxy for 'main dataset').
    """
    if report_id in _df_cache:
        return _df_cache[report_id], None

    if dataset_path and os.path.exists(dataset_path):
        try:
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            else:
                try:
                    df = pd.read_json(dataset_path, lines=True)
                except Exception:
                    df = pd.read_json(dataset_path)
            print(f"[_get_df] LOADED from provided path -> {os.path.basename(dataset_path)} shape={df.shape}")
            _df_cache[report_id] = df
            return df, None
        except Exception as exc:
            print(f"[_get_df] Failed to load from provided path {dataset_path}: {exc}")

    import glob
    csv_candidates = glob.glob(os.path.join(UPLOAD_FOLDER, '*.csv'))
    json_candidates = glob.glob(os.path.join(UPLOAD_FOLDER, '*.json'))
    all_candidates = csv_candidates + json_candidates

    # If dataset_path is missing but we have a filename, try to find it in uploads
    if dataset_path:
        filename = os.path.basename(dataset_path)
        for cand in all_candidates:
            if os.path.basename(cand) == filename:
                try:
                    if cand.endswith('.csv'):
                        df = pd.read_csv(cand)
                    else:
                        try:
                            df = pd.read_json(cand, lines=True)
                        except Exception:
                            df = pd.read_json(cand)
                    print(f"[_get_df] RECOVERED by filename -> {os.path.basename(cand)} shape={df.shape}")
                    _df_cache[report_id] = df
                    return df, None
                except Exception:
                    continue

    print(f"\n[_get_df] report_id={repr(report_id)}  x_col hint={repr(x_col)}")
    print(f"[_get_df] Candidates in folder: {[os.path.basename(p) for p in all_candidates]}")

    if not all_candidates:
        return None, (
            f"No dataset files found in uploads folder: {UPLOAD_FOLDER}\n"
            "Upload a dataset first before generating charts."
        )

    # ── Pass 1: column-match ──────────────────────────────────────────
    if x_col:
        for path in sorted(all_candidates):
            try:
                if path.endswith('.csv'):
                    headers = pd.read_csv(path, nrows=0).columns.tolist()
                else:
                    headers = pd.read_json(path, lines=True, nrows=0).columns.tolist()
                
                if x_col in headers:
                    if path.endswith('.csv'):
                        df = pd.read_csv(path)
                    else:
                        df = pd.read_json(path, lines=True)
                    print(f"[_get_df] MATCHED '{x_col}' -> {os.path.basename(path)}  "
                          f"shape={df.shape}  cols={headers[:6]}")
                    _df_cache[report_id] = df
                    return df, None
            except Exception as exc:
                print(f"[_get_df] header-read failed for {path}: {exc}")

        print(f"[_get_df] Column '{x_col}' not found in any dataset — using fallback")

    # ── Pass 2: largest file fallback ─────────────────────────────────
    try:
        best_path = max(all_candidates, key=os.path.getsize)
        if best_path.endswith('.csv'):
            df = pd.read_csv(best_path)
        else:
            df = pd.read_json(best_path, lines=True)
        print(f"[_get_df] FALLBACK -> {os.path.basename(best_path)}  shape={df.shape}")
        _df_cache[report_id] = df
        return df, None
    except Exception as exc:
        return None, f"All dataset load attempts failed: {exc}"


def get_or_create_usage(user):
    if not user.usage_stat:
        usage = UsageStat(user_id=user.id)
        db.session.add(usage)
        db.session.commit()
        return usage
    return user.usage_stat


def create_dataset_record(user, original_name, stored_name, filepath):
    size_bytes = os.path.getsize(filepath) if os.path.exists(filepath) else 0
    dataset = Dataset(
        user_id=user.id,
        original_name=original_name,
        stored_name=stored_name,
        upload_path=filepath,
        size_bytes=size_bytes
    )
    db.session.add(dataset)
    db.session.commit()

    usage = get_or_create_usage(user)
    usage.uploads_total += 1
    usage.storage_bytes += size_bytes
    db.session.commit()
    return dataset


def register_analysis(report_id, user, config=None):
    analysis = Analysis.query.filter_by(report_id=report_id, user_id=user.id).first()
    if not analysis:
        analysis = Analysis(
            user_id=user.id,
            dataset_name=session.get('filename', 'Unknown dataset'),
            report_id=report_id,
            status='COMPLETED',
            config=session.get('analysis_config') or config,
            completed_at=datetime.utcnow()
        )
        db.session.add(analysis)
    else:
        analysis.status = 'COMPLETED'
        analysis.completed_at = analysis.completed_at or datetime.utcnow()
        if config:
            analysis.config = analysis.config or config
    db.session.commit()
    return analysis


def register_report(report_id, user, analysis):
    report = Report.query.filter_by(report_id=report_id, user_id=user.id).first()
    if report:
        return report

    filepath = os.path.join(REPORT_FOLDER, f"{report_id}.json")
    size_bytes = os.path.getsize(filepath) if os.path.exists(filepath) else 0
    report = Report(
        user_id=user.id,
        analysis_id=analysis.id,
        report_id=report_id,
        title=f"Investigation {report_id}",
        filename=os.path.basename(filepath),
        size_bytes=size_bytes
    )
    db.session.add(report)
    db.session.commit()

    usage = get_or_create_usage(user)
    usage.analyses_total += 1
    usage.last_analysis_at = datetime.utcnow()
    db.session.commit()
    return report


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@login_manager.unauthorized_handler
def unauthorized():
    flash('Please log in to access this page.', 'warning')
    return redirect(url_for('login'))


@app.errorhandler(401)
def unauthorized_error(error):
    return redirect(url_for('login'))


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )


@app.route('/')
def landing():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('landing.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            if next_page and is_safe_url(next_page):
                return redirect(next_page)
            flash('Welcome back to FeynML.', 'success')
            return redirect(url_for('index'))

        flash('Invalid email or password.', 'danger')

    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not name or not email or not password:
            flash('Please fill all required fields.', 'warning')
            return redirect(url_for('signup'))

        if User.query.filter_by(email=email).first():
            flash('An account already exists with that email.', 'warning')
            return redirect(url_for('signup'))

        user = User(name=name, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        login_user(user)
        flash('Account created successfully. Welcome to FeynML.', 'success')
        return redirect(url_for('index'))

    return render_template('signup.html')


@app.route('/logout')
def logout():
    logout_user()
    flash('You have been signed out.', 'info')
    return redirect(url_for('landing'))


@app.route('/admin/setup', methods=['GET', 'POST'])
def admin_setup():
    """
    First-time admin setup wizard.
    Redirects to login if admin already exists.
    """
    # Check if admin already exists
    admin_exists = AdminProfile.query.filter_by(username='Feyn_admin').first()
    if admin_exists:
        flash('Admin account already configured. Please log in.', 'warning')
        return redirect(url_for('admin_login'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        password_confirm = request.form.get('password_confirm', '')

        # Validate username (must be Feyn_admin)
        if username != 'Feyn_admin':
            flash('Admin username must be Feyn_admin.', 'danger')
            return render_template('admin_setup.html')

        # Validate passwords match
        if password != password_confirm:
            flash('Passwords do not match.', 'danger')
            return render_template('admin_setup.html')

        # Validate password strength
        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'danger')
            return render_template('admin_setup.html')

        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in '!@#$%^&*' for c in password)

        if not (has_upper and has_lower and has_digit and has_special):
            flash('Password must contain uppercase, lowercase, numbers, and special characters.', 'danger')
            return render_template('admin_setup.html')

        # Validate email
        if not email or '@' not in email:
            flash('Please enter a valid email address.', 'danger')
            return render_template('admin_setup.html')

        # Check if email already used (should not be, but double check)
        if AdminProfile.query.filter_by(email=email).first():
            flash('This email is already associated with an admin account.', 'danger')
            return render_template('admin_setup.html')

        # Create admin profile
        try:
            admin = AdminProfile(
                username='Feyn_admin',
                email=email
            )
            admin.set_password(password)
            db.session.add(admin)
            db.session.commit()

            app.logger.info(f"First-time admin setup completed for Feyn_admin with email {email}")
            flash('Admin account created successfully! Please log in with your credentials.', 'success')
            return redirect(url_for('admin_login'))
        except Exception as exc:
            app.logger.error(f"Error creating admin profile: {exc}")
            db.session.rollback()
            flash('An error occurred while creating the admin account. Please try again.', 'danger')
            return render_template('admin_setup.html')

    return render_template('admin_setup.html')


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """
    Admin login with AdminProfile database model.
    Redirects to setup if no admin exists.
    """
    # Check if admin exists - if not, redirect to setup
    admin_exists = AdminProfile.query.filter_by(username='Feyn_admin').first()
    if not admin_exists:
        return redirect(url_for('admin_setup'))

    if _is_admin_authenticated():
        return redirect(url_for('admin'))

    blocked_until = None
    attempts = session.get('admin_login_attempts', 0)
    last_attempt = session.get('admin_login_last_failure')
    if attempts >= 5 and last_attempt:
        block_seconds = 900 - int(datetime.utcnow().timestamp() - last_attempt)
        if block_seconds > 0:
            blocked_until = block_seconds
        else:
            session.pop('admin_login_attempts', None)
            session.pop('admin_login_last_failure', None)
            attempts = 0

    if request.method == 'POST':
        if blocked_until:
            flash('Too many failed login attempts. Please try again later.', 'danger')
            return render_template('admin_login.html', blocked_until=blocked_until)

        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        # Find admin by username
        admin = AdminProfile.query.filter_by(username=username).first()

        if admin and admin.check_password(password):
            session['admin_authenticated'] = True
            session['admin_id'] = admin.id
            session.pop('admin_login_attempts', None)
            session.pop('admin_login_last_failure', None)
            next_page = request.form.get('next') or request.args.get('next')
            if next_page and is_safe_url(next_page):
                return redirect(next_page)
            flash('Admin console access granted.', 'success')
            app.logger.info(f"Admin {username} logged in successfully")
            return redirect(url_for('admin'))

        attempts = attempts + 1
        session['admin_login_attempts'] = attempts
        session['admin_login_last_failure'] = datetime.utcnow().timestamp()
        if attempts >= 5:
            flash('Too many failed login attempts. Please try again in 15 minutes.', 'danger')
            app.logger.warning(f"Admin login rate limited after 5 failed attempts")
        else:
            flash('Invalid admin username or password.', 'danger')

    next_url = request.args.get('next', '')
    return render_template('admin_login.html', blocked_until=blocked_until, next=next_url)


@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_authenticated', None)
    session.pop('admin_id', None)
    flash('Admin session signed out.', 'info')
    return redirect(url_for('admin_login'))


@app.route('/admin/forgot-password', methods=['GET', 'POST'])
def admin_forgot_password():
    """
    Admin password recovery - Step 1: Request OTP.
    User must enter the admin username (Feyn_admin) and registered email.
    """
    # Check if admin exists - if not, redirect to setup
    admin_exists = AdminProfile.query.filter_by(username='Feyn_admin').first()
    if not admin_exists:
        flash('Please set up the admin account first.', 'warning')
        return redirect(url_for('admin_setup'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()

        # Validate username is exactly 'Feyn_admin'
        if username != 'Feyn_admin':
            flash('Invalid admin username. Only Feyn_admin can request password recovery.', 'danger')
            return render_template('admin_forgot_password.html')

        # Check if admin profile exists with this username and email
        admin = AdminProfile.query.filter_by(username=username, email=email).first()
        if not admin:
            # Security: Don't reveal which username/email combination is valid
            flash('No admin account found with that username and email combination.', 'danger')
            app.logger.warning(f"Failed forgot-password attempt for username={username}, email={email}")
            return render_template('admin_forgot_password.html')

        # Generate OTP and store in database
        otp_token = OTPToken.create_otp_for_admin(admin.id, expiry_minutes=10)

        # Send OTP via email
        email_sent = email_service.send_otp_email(admin.email, otp_token.otp_code, admin.username)
        
        if email_sent:
            app.logger.info(f"OTP generated and sent for admin {admin.username}")
            # Store necessary info in session for next step
            session['otp_admin_id'] = admin.id
            session['otp_admin_username'] = admin.username
            session['otp_expiry'] = otp_token.expires_at.timestamp()
            flash('OTP has been sent to your registered email address.', 'success')
            return redirect(url_for('admin_verify_otp'))
        else:
            app.logger.error(f"Failed to send OTP email for admin {admin.username}")
            flash('Failed to send OTP email. Please try again later.', 'danger')

    return render_template('admin_forgot_password.html')


@app.route('/admin/verify-otp', methods=['GET', 'POST'])
def admin_verify_otp():
    """
    Admin password recovery - Step 2: Verify OTP.
    User must enter the 6-digit OTP received via email.
    """
    # Check if user came from forgot-password step
    admin_id = session.get('otp_admin_id')
    admin_username = session.get('otp_admin_username')
    otp_expiry = session.get('otp_expiry')

    if not admin_id or not admin_username:
        flash('Please start from the forgot password page.', 'warning')
        return redirect(url_for('admin_forgot_password'))

    if request.method == 'POST':
        otp_code = request.form.get('otp_code', '').strip()
        submitted_username = request.form.get('admin_username', '').strip()

        # Validate username hasn't changed
        if submitted_username != admin_username:
            flash('Username mismatch. Please start over.', 'danger')
            session.pop('otp_admin_id', None)
            session.pop('otp_admin_username', None)
            session.pop('otp_expiry', None)
            return redirect(url_for('admin_forgot_password'))

        # Validate OTP format
        if not otp_code or not otp_code.isdigit() or len(otp_code) != 6:
            flash('Invalid OTP format. Please enter a 6-digit code.', 'danger')
            return render_template('admin_verify_otp.html', 
                                 admin_username=admin_username,
                                 time_remaining=int(otp_expiry - datetime.utcnow().timestamp()) if otp_expiry else 0)

        # Find the most recent OTP for this admin
        otp_token = OTPToken.query.filter_by(
            admin_id=admin_id,
            otp_code=otp_code
        ).order_by(OTPToken.created_at.desc()).first()

        if not otp_token:
            flash('Invalid OTP. Please check and try again.', 'danger')
            return render_template('admin_verify_otp.html', 
                                 admin_username=admin_username,
                                 time_remaining=int(otp_expiry - datetime.utcnow().timestamp()) if otp_expiry else 0)

        # Check if OTP is valid
        if not otp_token.is_valid():
            if otp_token.is_used:
                flash('This OTP has already been used. Please request a new one.', 'danger')
            else:
                flash('This OTP has expired. Please request a new one.', 'danger')
            session.pop('otp_admin_id', None)
            session.pop('otp_admin_username', None)
            session.pop('otp_expiry', None)
            return redirect(url_for('admin_forgot_password'))

        # Mark OTP as used
        otp_token.mark_as_used()

        # Store OTP verification in session
        session['otp_verified'] = True
        session['otp_verified_admin_id'] = admin_id
        session['otp_verified_admin_username'] = admin_username

        app.logger.info(f"OTP verified successfully for admin {admin_username}")
        flash('OTP verified successfully. Please set your new password.', 'success')
        return redirect(url_for('admin_reset_password'))

    # Calculate remaining time
    time_remaining = int(otp_expiry - datetime.utcnow().timestamp()) if otp_expiry else 0
    if time_remaining <= 0:
        session.pop('otp_admin_id', None)
        session.pop('otp_admin_username', None)
        session.pop('otp_expiry', None)
        flash('OTP has expired. Please request a new one.', 'warning')
        return redirect(url_for('admin_forgot_password'))

    return render_template('admin_verify_otp.html', 
                         admin_username=admin_username,
                         time_remaining=time_remaining)


@app.route('/admin/reset-password', methods=['GET', 'POST'])
def admin_reset_password():
    """
    Admin password recovery - Step 3: Reset password.
    User must enter a new password after OTP verification.
    """
    # Check if user has verified OTP
    if not session.get('otp_verified'):
        flash('Please verify your OTP first.', 'warning')
        return redirect(url_for('admin_forgot_password'))

    admin_id = session.get('otp_verified_admin_id')
    admin_username = session.get('otp_verified_admin_username')

    if not admin_id or not admin_username:
        flash('Session expired. Please start over.', 'warning')
        return redirect(url_for('admin_forgot_password'))

    if request.method == 'POST':
        password = request.form.get('password', '')
        password_confirm = request.form.get('password_confirm', '')
        submitted_username = request.form.get('admin_username', '').strip()

        # Validate username
        if submitted_username != admin_username:
            flash('Username mismatch. Please start over.', 'danger')
            session.pop('otp_verified', None)
            session.pop('otp_verified_admin_id', None)
            session.pop('otp_verified_admin_username', None)
            return redirect(url_for('admin_forgot_password'))

        # Validate password
        if not password or not password_confirm:
            flash('Both password fields are required.', 'warning')
            return render_template('admin_reset_password.html', admin_username=admin_username)

        if password != password_confirm:
            flash('Passwords do not match.', 'danger')
            return render_template('admin_reset_password.html', admin_username=admin_username)

        # Validate password strength
        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'danger')
            return render_template('admin_reset_password.html', admin_username=admin_username)

        # Get admin profile and update password
        admin = AdminProfile.query.get(admin_id)
        if not admin:
            flash('Admin profile not found. Please contact system administrator.', 'danger')
            app.logger.error(f"Admin profile {admin_id} not found during password reset")
            session.pop('otp_verified', None)
            session.pop('otp_verified_admin_id', None)
            session.pop('otp_verified_admin_username', None)
            return redirect(url_for('admin_forgot_password'))

        # Update password
        try:
            admin.set_password(password)
            admin.updated_at = datetime.utcnow()
            db.session.commit()

            app.logger.info(f"Password reset successfully for admin {admin.username}")

            # Send confirmation email
            email_service.send_password_reset_confirmation(admin.email, admin.username)

            # Clear session
            session.pop('otp_admin_id', None)
            session.pop('otp_admin_username', None)
            session.pop('otp_expiry', None)
            session.pop('otp_verified', None)
            session.pop('otp_verified_admin_id', None)
            session.pop('otp_verified_admin_username', None)

            flash('Password reset successfully. Please log in with your new password.', 'success')
            return redirect(url_for('admin_login'))
        except Exception as exc:
            app.logger.error(f"Error during password reset for admin {admin_username}: {exc}")
            flash('An error occurred while resetting your password. Please try again.', 'danger')
            db.session.rollback()
            return render_template('admin_reset_password.html', admin_username=admin_username)

    return render_template('admin_reset_password.html', admin_username=admin_username)


@app.route('/analyze')
@login_required
def index():
    return render_template('index.html')


@app.route('/features')
def features():
    return render_template('features.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/faq')
def faq():
    return render_template('faq.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/admin')
@admin_required
def admin():
    return render_template('admin.html')


@app.route('/admin/analytics')
@admin_required
def admin_analytics():
    analytics_error = False
    error_message = None

    def safe_count(query):
        try:
            return query.count()
        except Exception as exc:
            app.logger.exception('Analytics count failed: %s', exc)
            return 0

    def safe_trend(model, date_field, start_date):
        try:
            rows = db.session.query(
                db.func.date(date_field).label('day'),
                func.count().label('count')
            ).filter(date_field >= start_date).group_by(db.func.date(date_field)).order_by(db.func.date(date_field)).all()
            return {str(row.day): row.count for row in rows}
        except Exception as exc:
            app.logger.exception('Analytics trend query failed: %s', exc)
            return None

    today = datetime.utcnow().date()
    week_start = today - timedelta(days=6)
    labels = [(week_start + timedelta(days=i)).isoformat() for i in range(7)]

    total_users = safe_count(User.query)
    new_users_today = 0
    try:
        new_users_today = User.query.filter(db.func.date(User.created_at) == today).count()
    except Exception as exc:
        app.logger.exception('Analytics new users today failed: %s', exc)
        analytics_error = True

    uploaded_datasets = safe_count(Dataset.query)
    models_trained = 0
    try:
        models_trained = Analysis.query.filter(Analysis.status != 'PENDING').count()
    except Exception as exc:
        app.logger.exception('Analytics models trained failed: %s', exc)
        analytics_error = True

    reports_generated = safe_count(Report.query)

    active_users = 0
    try:
        active_users = db.session.query(func.count(func.distinct(Analysis.user_id))).filter(Analysis.created_at >= week_start).scalar() or 0
    except Exception as exc:
        app.logger.exception('Analytics active users query failed: %s', exc)
        analytics_error = True

    user_growth_map = safe_trend(User, User.created_at, week_start)
    dataset_trend_map = safe_trend(Dataset, Dataset.uploaded_at, week_start)
    report_trend_map = safe_trend(Report, Report.created_at, week_start)

    if user_growth_map is None or dataset_trend_map is None or report_trend_map is None:
        analytics_error = True

    user_growth = [user_growth_map.get(label, 0) if user_growth_map else 0 for label in labels]
    dataset_trend = [dataset_trend_map.get(label, 0) if dataset_trend_map else 0 for label in labels]
    report_trend = [report_trend_map.get(label, 0) if report_trend_map else 0 for label in labels]

    return render_template(
        'analytics.html',
        total_users=total_users,
        new_users_today=new_users_today,
        uploaded_datasets=uploaded_datasets,
        models_trained=models_trained,
        reports_generated=reports_generated,
        active_users=active_users,
        labels=labels,
        user_growth=user_growth,
        dataset_trend=dataset_trend,
        report_trend=report_trend,
        analytics_error=analytics_error,
        error_message=error_message,
    )


def _is_admin_user():
    """Determine whether the current user is an admin.

    By default this checks a comma-separated list in the environment variable
    `ADMIN_EMAILS`. If that is not set, the first registered user (id == 1)
    is treated as admin for single-user deployments.
    """
    admin_emails = os.environ.get('ADMIN_EMAILS', '')
    if admin_emails:
        emails = [e.strip().lower() for e in admin_emails.split(',') if e.strip()]
        return current_user.is_authenticated and current_user.email and current_user.email.lower() in emails

    # Fallback: treat user with id==1 as admin if logged in
    return current_user.is_authenticated and getattr(current_user, 'id', None) == 1


@app.route('/admin/users')
@admin_required
def admin_users():

    q = request.args.get('q', '').strip()
    sort = request.args.get('sort', 'created_desc')

    users_q = User.query
    if q:
        like = f"%{q}%"
        users_q = users_q.filter((User.name.ilike(like)) | (User.email.ilike(like)))

    if sort == 'created_asc':
        users_q = users_q.order_by(User.created_at.asc())
    else:
        users_q = users_q.order_by(User.created_at.desc())

    users = users_q.all()
    total = users_q.count()

    # Only expose non-sensitive fields
    safe_users = [
        {
            'id': u.id,
            'name': u.name,
            'email': u.email,
            'joined_at': u.joined_at.strftime('%Y-%m-%d %H:%M') if getattr(u, 'joined_at', None) else '',
            'created_at': u.created_at.strftime('%Y-%m-%d %H:%M') if getattr(u, 'created_at', None) else ''
        }
        for u in users
    ]

    return render_template('admin_users.html', users=safe_users, total=total, q=q, sort=sort)


@app.route('/admin/export-logs')
@admin_required
def admin_export_logs():

    q = request.args.get('q', '').strip()
    level = request.args.get('level', 'all')

    # Read log file
    entries = []
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        else:
            lines = []
    except Exception:
        lines = []

    # Parse lines formatted as: timestamp - LEVEL - message
    parsed = []
    for ln in lines[-1000:]:
        ln = ln.strip('\n')
        parts = ln.split(' - ', 2)
        if len(parts) == 3:
            ts, lvl, msg = parts
        else:
            ts = ''
            lvl = 'INFO'
            msg = ln

        parsed.append({'timestamp': ts, 'level': lvl, 'message': msg})

    # Apply filters and search, then take last 100
    def matches(p):
        if level and level != 'all' and p['level'] != level:
            return False
        if q and q.lower() not in p['message'].lower() and q.lower() not in p['timestamp'].lower():
            return False
        return True

    filtered = [p for p in parsed if matches(p)]
    entries = filtered[-100:][::-1]

    return render_template('admin_export_logs.html', entries=entries, q=q, level=level)


@app.route('/admin/export-logs/download')
@admin_required
def admin_export_logs_download():

    fmt = request.args.get('fmt', 'txt')
    q = request.args.get('q', '').strip()
    level = request.args.get('level', 'all')

    # reuse parsing
    try:
        with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        lines = []

    parsed = []
    for ln in lines:
        ln = ln.strip('\n')
        parts = ln.split(' - ', 2)
        if len(parts) == 3:
            ts, lvl, msg = parts
        else:
            ts = ''
            lvl = 'INFO'
            msg = ln
        parsed.append({'timestamp': ts, 'level': lvl, 'message': msg})

    def matches(p):
        if level and level != 'all' and p['level'] != level:
            return False
        if q and q.lower() not in p['message'].lower() and q.lower() not in p['timestamp'].lower():
            return False
        return True

    filtered = [p for p in parsed if matches(p)]

    if fmt == 'csv':
        import io, csv
        si = io.StringIO()
        cw = csv.writer(si)
        cw.writerow(['timestamp', 'level', 'message'])
        for p in filtered:
            cw.writerow([p['timestamp'], p['level'], p['message']])
        output = io.BytesIO()
        output.write(si.getvalue().encode('utf-8'))
        output.seek(0)
        return send_file(output, as_attachment=True, download_name='app_logs.csv', mimetype='text/csv')

    # default: txt
    import io
    sio = io.StringIO()
    for p in filtered:
        sio.write(f"{p['timestamp']} - {p['level']} - {p['message']}\n")
    out = io.BytesIO()
    out.write(sio.getvalue().encode('utf-8'))
    out.seek(0)
    return send_file(out, as_attachment=True, download_name='app_logs.txt', mimetype='text/plain')


@app.route('/admin/settings', methods=['GET', 'POST'])
@admin_required
def admin_settings():

    if request.method == 'POST':
        site_name = request.form.get('site_name', '').strip()
        theme = request.form.get('theme', 'dark')
        maintenance_mode = 'true' if request.form.get('maintenance_mode') else 'false'
        upload_limit_mb = request.form.get('upload_limit_mb', '50')

        # Save each setting into AppSetting table
        def upsert(key, val):
            s = AppSetting.query.filter_by(key=key).first()
            if not s:
                s = AppSetting(key=key, value=str(val))
                db.session.add(s)
            else:
                s.value = str(val)

        upsert('site_name', site_name)
        upsert('theme', theme)
        upsert('maintenance_mode', maintenance_mode)
        upsert('upload_limit_mb', upload_limit_mb)
        db.session.commit()

        flash('Settings saved successfully.', 'success')
        return redirect(url_for('admin_settings'))

    # GET: load settings
    settings_q = AppSetting.query.all()
    settings = {s.key: s.value for s in settings_q}
    return render_template('admin_settings.html', settings=settings)

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'dataset' not in request.files:
        flash('No file part', 'warning')
        return redirect(url_for('index'))

    file = request.files['dataset']
    if file.filename == '':
        flash('No selected file', 'warning')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            file.save(filepath)
            if filename.endswith('.csv'):
                pd.read_csv(filepath, nrows=1)
            elif filename.endswith('.json'):
                pd.read_json(filepath, lines=True, nrows=1)

            create_dataset_record(current_user, file.filename, filename, filepath)
            session['filename'] = filename
            session['dataset_id'] = Dataset.query.filter_by(user_id=current_user.id, stored_name=filename).order_by(Dataset.uploaded_at.desc()).first().id
            return redirect(url_for('configure_schema'))
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            flash(f'Upload failure: {str(e)}', 'danger')
            return redirect(url_for('index'))

    flash('Invalid file type. Please upload a CSV or JSON file.', 'danger')
    return redirect(url_for('index'))


# ============================================================
# NEW ROUTE — Model upload (.pkl / .joblib)
# Placed here, right after upload_file(), following the same
# structure and error-handling pattern as the dataset upload.
# ============================================================
@app.route('/upload-model', methods=['POST'])
@login_required
def upload_model():
    if 'model_file' not in request.files:
        flash('No file part', 'warning')
        return redirect(url_for('index'))

    file = request.files['model_file']
    if file.filename == '':
        flash('No selected file', 'warning')
        return redirect(url_for('index'))

    if file and (file.filename.endswith('.pkl') or file.filename.endswith('.joblib')):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            file.save(filepath)

            # Sanity-check: make sure the file actually loads before trusting it
            import joblib
            joblib.load(filepath)

            session['model_filename'] = filename
            flash('Model uploaded successfully!', 'success')
            return redirect(url_for('index'))
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            flash(f'Model upload failed: {str(e)}', 'danger')
            return redirect(url_for('index'))

    flash('Invalid file type. Please upload a .pkl or .joblib file.', 'danger')
    return redirect(url_for('index'))
# ============================================================
# END NEW ROUTE
# ============================================================


@app.route('/configure')
@login_required
def configure_schema():
    filename = session.get('filename')
    if not filename:
        flash('Please upload a dataset first.', 'warning')
        return redirect(url_for('index'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath, nrows=1)
            columns = df.columns.tolist()
        elif filename.endswith('.json'):
            with open(filepath, 'r') as f:
                first_line = f.readline()
                data = json.loads(first_line)
                columns = list(data.keys()) if isinstance(data, dict) else []
        else:
            columns = []
    except Exception as e:
        traceback.print_exc()
        flash(f'Error reading file headers: {str(e)}', 'danger')
        return redirect(url_for('index'))

    return render_template('configure.html', columns=columns, filename=filename)


@app.route('/configure-model')
@login_required
def configure_model_schema():
    filename = session.get('filename')
    model_filename = session.get('model_filename')
    if not filename or not model_filename:
        flash('Please upload both a dataset and a model first.', 'warning')
        return redirect(url_for('index'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], model_filename)

    try:
        if filename.endswith('.csv'):
            dataset_columns = pd.read_csv(filepath, nrows=1).columns.tolist()
        else:
            with open(filepath, 'r') as f:
                data = json.loads(f.readline())
                dataset_columns = list(data.keys()) if isinstance(data, dict) else []
    except Exception as e:
        flash(f'Error reading dataset columns: {str(e)}', 'danger')
        return redirect(url_for('configure_schema'))

    try:
        from webapp.services.model_loader import load_model
        model = load_model(model_path)
    except Exception as e:
        flash(f'Error loading model: {str(e)}', 'danger')
        return redirect(url_for('index'))

    target_col = session.get('analysis_config', {}).get('target_col')
    candidate_cols = [c for c in dataset_columns if c != target_col]

    if hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
        named = True
    elif hasattr(model, 'n_features_in_'):
        expected_features = [f"feature_{i+1}" for i in range(model.n_features_in_)]
        named = False
    else:
        expected_features = candidate_cols
        named = False

    import difflib
    suggested_mapping = {}
    used = set()
    for ef in expected_features:
        match = None
        if named:
            for c in candidate_cols:
                if c.lower() == ef.lower() and c not in used:
                    match = c
                    break
            if not match:
                close = difflib.get_close_matches(
                    ef, [c for c in candidate_cols if c not in used], n=1, cutoff=0.6
                )
                if close:
                    match = close[0]
        suggested_mapping[ef] = match
        if match:
            used.add(match)

    return render_template(
        'configure_model.html',
        expected_features=expected_features,
        candidate_cols=candidate_cols,
        suggested_mapping=suggested_mapping,
        named=named,
        model_filename=model_filename
    )


@app.route('/save-model-mapping', methods=['POST'])
@login_required
def save_model_mapping():
    filename = session.get('filename')
    model_filename = session.get('model_filename')
    if not filename or not model_filename:
        flash('Session expired. Please upload again.', 'warning')
        return redirect(url_for('index'))

    mapping_fields = {
        key: value for key, value in request.form.items()
        if key.startswith('map__')
    }
    missing_features = [
        key[len('map__'):] for key, value in mapping_fields.items()
        if not value
    ]
    if not mapping_fields or missing_features:
        flash('Please map every feature before running the analysis.', 'warning')
        return redirect(url_for('configure_model_schema'))

    feature_mapping = {
        key[len('map__'):]: value for key, value in mapping_fields.items()
    }

    session['feature_mapping'] = feature_mapping
    session['feature_mapping_key'] = f"{model_filename}::{filename}"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], model_filename)

    try:
        r = get_runner()
        if r is None:
            flash('Analysis runner not available in this environment.', 'warning')
            return redirect(url_for('configure_schema'))
        r.audience = session.get('selected_audience', 'ml_engineer')
        r.run(
            filepath,
            session['analysis_config'],
            model_path=model_path,
            feature_mapping=feature_mapping
        )
        return redirect(url_for('analysis_progress'))
    except Exception as e:
        traceback.print_exc()
        flash(f'Analysis failed to start: {str(e)}', 'danger')
        return redirect(url_for('configure_schema'))


@app.route('/run_analysis', methods=['POST'])
@login_required
def run_analysis():
    try:
        target_col = request.form.get('target_col')
        pred_col = request.form.get('pred_col')
        sensitive_col = request.form.get('sensitive_col')
        time_col = request.form.get('time_col')
        auto_predict = request.form.get('auto_predict') == 'true'

        if not target_col:
            flash('Target column is required.', 'warning')
            return redirect(url_for('configure_schema'))

        if target_col == pred_col:
            flash('Warning: Prediction column should contain model outputs, not the actual target column.', 'info')

        filename = session.get('filename')
        if not filename:
            flash('Session expired. Please upload again.', 'warning')
            return redirect(url_for('index'))

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        session['analysis_config'] = {
            'target_col': target_col,
            'pred_col': pred_col,
            'sensitive_col': sensitive_col,
            'timestamp_col': time_col,
            'auto_predict': auto_predict
        }

        try:
            r = get_runner()
            if r is None:
                flash('Analysis runner not available in this environment.', 'warning')
                return redirect(url_for('configure_schema'))
            r.audience = session.get('selected_audience', 'ml_engineer')

            model_path = None
            model_filename = session.get('model_filename')
            if model_filename:
                model_path = os.path.join(app.config['UPLOAD_FOLDER'], model_filename)

                # A model is attached — route through feature mapping first,
                # unless this exact model+dataset pair was already mapped.
                mapping_key = f"{model_filename}::{filename}"
                if session.get('feature_mapping_key') != mapping_key:
                    return redirect(url_for('configure_model_schema'))

                r.run(
                    filepath,
                    session['analysis_config'],
                    model_path=model_path,
                    feature_mapping=session.get('feature_mapping')
                )
            else:
                r.run(filepath, session['analysis_config'], model_path=model_path)

            return redirect(url_for('analysis_progress'))
        except Exception as e:
            traceback.print_exc()
            flash(f'Analysis failed to start: {str(e)}', 'danger')
            return redirect(url_for('configure_schema'))
    except Exception as e:
        traceback.print_exc()
        raise


@app.route('/analysis-progress')
@login_required
def analysis_progress():
    try:
        r = get_runner()
        if r is None or r.status == 'idle':
            return redirect(url_for('index'))
        return render_template('analysis_running.html')
    except Exception:
        traceback.print_exc()
        raise


@app.route('/analysis-status')
@login_required
def analysis_status():
    try:
        r = get_runner()
        if r is None:
            return jsonify({'status': 'unavailable', 'progress': 0, 'logs': [], 'error': 'runner unavailable', 'report_id': None})
        report_id = None
        if r.report_path:
            report_id = os.path.basename(r.report_path).replace('.json', '')
        return jsonify({
            'status': r.status,
            'progress': r.progress,
            'logs': r.logs,
            'error': r.error,
            'report_id': report_id
        })
    except Exception:
        traceback.print_exc()
        raise


@app.route('/reset-analysis')
@login_required
def reset_analysis():
    r = get_runner()
    if r:
        r.status = 'idle'
        r.progress = 0
        r.logs = []
        r.error = None
        r.report_path = None
    flash('Analysis runner reset. You can start a new analysis.', 'info')
    return redirect(url_for('index'))


@app.route('/set-audience', methods=['POST'])
@login_required
def set_audience():
    try:
        from flask import request, session
        audience = request.form.get('audience')
        if audience:
            session['selected_audience'] = audience
        return jsonify({'success': True, 'audience': audience})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/analysis-results')
@login_required
def analysis_results():
    try:
        if runner.status != 'completed':
            return redirect(url_for('analysis_progress'))

        report_id = os.path.basename(runner.report_path).replace('.json', '')
        return redirect(url_for('view_dashboard', report_id=report_id))
    except Exception:
        traceback.print_exc()
        raise


@app.route('/dashboard/<report_id>')
@login_required
def view_dashboard(report_id):
    filepath = os.path.join(REPORT_FOLDER, f"{report_id}.json")
    if not os.path.exists(filepath):
        flash('Report not found.', 'warning')
        return redirect(url_for('index'))

    report = Report.query.filter_by(report_id=report_id, user_id=current_user.id).first()
    if report is None:
        if 'filename' not in session:
            flash('Report not found in your workspace.', 'warning')
            return redirect(url_for('saved_reports'))

        analysis = register_analysis(report_id, current_user, session.get('analysis_config'))
        report = register_report(report_id, current_user, analysis)

    with open(filepath, 'r') as f:
        data = json.load(f)

    # ── Unwrap ReportEngine "sections" wrapper if present ──────────────
    # ReportEngine.save_json() nests all module outputs under "sections".
    # view_dashboard expects flat top-level keys. Unwrap if needed.
    if 'sections' in data and isinstance(data.get('sections'), dict):
        sections_data = data.pop('sections')
        for k, v in sections_data.items():
            if k not in data:
                data[k] = v

    # Detect Phase 4 reports and redirect to the specialized phase4 viewer
    if data.get('results', {}).get('module'):
        return redirect(url_for('phase4.view_report', report_id=report_id))

    data.setdefault('calibration', {'status': 'SKIPPED', 'severity': 'NONE', 'findings': {}})
    _raw_fairness = data.get('fairness', {})
    if _raw_fairness and 'per_axis' in _raw_fairness:
        _per_axis = _raw_fairness.get('per_axis', {})
        _dp_gaps = []
        _groups = []
        for _axis, _axis_data in _per_axis.items():
            _groups.append(_axis)
            _dp = _axis_data.get('demographic_parity', {})
            _dp_gaps.append(_dp.get('max_difference', 0.0))
        _max_disparity = max(_dp_gaps) if _dp_gaps else 0.0
        data['fairness'] = {
            'status':   _raw_fairness.get('severity', 'LOW'),
            'severity': _raw_fairness.get('severity', 'LOW'),
            'findings': {
                'disparity_ratio': round(_max_disparity, 3),
                'max_disparity':   round(_max_disparity, 3),
                'groups':          _groups,
                'violations':      _raw_fairness.get('violations', []),
                'per_axis':        _per_axis,
            }
        }
    else:
        data.setdefault('fairness', {
            'status':   'INSUFFICIENT_DATA',
            'severity': 'NONE',
            'findings': {
                'disparity_ratio': None,
                'max_disparity':   None,
                'groups':          [],
                'violations':      [],
            }
        })
    data.setdefault('drift', {'status': 'SKIPPED', 'severity': 'NONE', 'findings': {}})
    data.setdefault('label_noise', {'status': 'SKIPPED', 'severity': 'NONE', 'findings': {}})
    data.setdefault('leakage', {'status': 'SKIPPED', 'severity': 'NONE', 'findings': {}})
    data.setdefault('missing_data', {'status': 'SKIPPED', 'severity': 'NONE', 'findings': {}})
    data.setdefault('root_cause', {'health_status': 'Unknown', 'confidence': 0, 'root_causes': [], 'recommended_actions': []})
    # If ai_investigator exists but has empty fields, try to populate from
    # ai_investigator_by_audience (which may have been generated instead)
    existing_ai = data.get('ai_investigator', {})
    if not existing_ai.get('executive_summary'):
        ai_by_aud = data.get('ai_investigator_by_audience', {})
        if ai_by_aud:
            # Use first available audience report as fallback
            fallback_audience = (
                data.get('selected_audience') or
                session.get('selected_audience') or
                'ML Engineer'
            )
            fallback_ai = ai_by_aud.get(fallback_audience) or next(iter(ai_by_aud.values()), {})
            if fallback_ai.get('executive_summary'):
                # Merge fallback into ai_investigator
                for field in ['executive_summary', 'investigation_findings',
                               'impact_assessment', 'confidence_explanation',
                               'recommended_actions', 'technical_notes', 'risk_level']:
                    if not existing_ai.get(field) and fallback_ai.get(field):
                        existing_ai[field] = fallback_ai[field]
                data['ai_investigator'] = existing_ai

    data.setdefault('ai_investigator', {
        'risk_level': 'UNKNOWN',
        'executive_summary': '',
        'investigation_findings': '',
        'impact_assessment': '',
        'confidence_explanation': '',
        'recommended_actions': [],
        'technical_notes': ''
    })
    # If audience_reports is empty but ai_investigator_by_audience exists,
    # mirror it into audience_reports so the JS switcher works
    if not data.get('audience_reports') and data.get('ai_investigator_by_audience'):
        # Normalize keys: strip extra whitespace, preserve original too
        raw_aud = data['ai_investigator_by_audience']
        normalized = {}
        for k, v in raw_aud.items():
            normalized[k] = v
            normalized[k.strip()] = v  # whitespace-safe copy
        data['audience_reports'] = normalized
    data.setdefault('audience_reports', {})

    if isinstance(data.get('root_cause'), dict):
        data['root_cause']['recommended_actions'] = _normalize_recommended_actions(
            data['root_cause'].get('recommended_actions', [])
        )

    filename = session.get('filename', report.filename if report else 'Unknown')
    severities = [
        data.get('calibration', {}).get('severity', 'NONE'),
        data.get('fairness', {}).get('severity', 'NONE') if data.get('fairness') else 'NONE',
        data.get('drift', {}).get('severity', 'NONE'),
        data.get('label_noise', {}).get('severity', 'NONE'),
        data.get('leakage', {}).get('severity', 'NONE'),
        data.get('missing_data', {}).get('severity', 'NONE')
    ]
    severities = [s.upper() if s else 'NONE' for s in severities]

    # Initialize variables before conditional logic
    critical_count = 0
    alerts_count = 0
    risk_level = "STABLE"

    # Use canonical status from RiskScorer (single source of truth)
    # RiskScorer level is stored in data['root_cause']['metadata']['risk']['level']
    canonical_status = data.get('root_cause', {}).get('metadata', {}).get('risk', {}).get('level')
    if canonical_status:
        risk_level = canonical_status
    else:
        # Fallback to legacy calculation if RiskScorer data not available
        critical_count = severities.count('CRITICAL') + severities.count('HIGH')
        alerts_count = severities.count('MEDIUM') + severities.count('LOW')
        if critical_count > 0:
            risk_level = 'HIGH'
        elif severities.count('MEDIUM') > 0:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'

    charts = {}
    cal_data = data.get('calibration', {})
    confidences = []
    accuracies = []
    if 'findings' in cal_data:
        findings_block = cal_data['findings']
        # Try direct keys first, then nested curve fallback
        confidences = (
            findings_block.get('confidences') or
            findings_block.get('curve', {}).get('mean_predicted') or
            cal_data.get('curve', {}).get('mean_predicted', [])
        )
        accuracies = (
            findings_block.get('accuracies') or
            findings_block.get('curve', {}).get('fraction_pos') or
            cal_data.get('curve', {}).get('fraction_pos', [])
        )
    elif 'curve' in cal_data:
        confidences = cal_data['curve'].get('mean_predicted', [])
        accuracies  = cal_data['curve'].get('fraction_pos',   [])
    else:
        confidences = []
        accuracies  = []

    if confidences and accuracies:
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Bar(
            x=confidences,
            y=accuracies,
            name='Actual Accuracy',
            marker_color=FEYNML_CHART_COLORS['primary'],
            hovertemplate="Confidence: %{x:.2f}<br>Accuracy: %{y:.2f}<extra></extra>"
        ))
        fig_cal.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfectly Calibrated',
            line=dict(dash='dash', color=FEYNML_CHART_COLORS['secondary'])
        ))
        fig_cal.update_layout(
            title='Reliability Diagram (Calibration)',
            xaxis_title='Predicted Confidence',
            yaxis_title='Actual Accuracy',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter, sans-serif', color='white')
        )
        charts['calibration'] = json.dumps(fig_cal, cls=plotly.utils.PlotlyJSONEncoder)

    drift_features = data.get('drift', {}).get('findings', {}).get('per_feature', [])
    if drift_features:
        drift_features = sorted(drift_features, key=lambda x: x.get('ks_stat', 0), reverse=True)
        features = [f['feature'] for f in drift_features]
        ks_stats = [f.get('ks_stat', 0) for f in drift_features]
        psis = [f.get('psi', 0) for f in drift_features]
        colors = []
        for f in drift_features:
            status = f.get('status', 'STABLE')
            if status == 'DRIFT':
                colors.append(FEYNML_CHART_COLORS['critical'])
            elif status == 'WARN':
                colors.append(FEYNML_CHART_COLORS['critical'])
            else:
                colors.append(FEYNML_CHART_COLORS['primary'])

        fig_drift = go.Figure()
        fig_drift.add_trace(go.Bar(
            x=features,
            y=ks_stats,
            marker_color=colors,
            name='KS Statistic',
            customdata=list(zip(psis, [f.get('status', 'STABLE') for f in drift_features])),
            hovertemplate="<b>%{x}</b><br>KS: %{y:.3f}<br>PSI: %{customdata[0]:.3f}<br>Status: %{customdata[1]}<extra></extra>"
        ))
        fig_drift.add_hline(y=0.2, line_dash='dash', line_color=FEYNML_CHART_COLORS['critical'], annotation_text='Drift Threshold')
        fig_drift.update_layout(
            title='Feature Drift Analysis',
            xaxis_title='Feature',
            yaxis_title='KS Statistic',
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter, sans-serif', color='white')
        )
        charts['drift'] = json.dumps(fig_drift, cls=plotly.utils.PlotlyJSONEncoder)

    ln_findings = data.get('label_noise', {}).get('findings', {})
    if ln_findings:
        errors = ln_findings.get('num_errors_detected', 0)
        total = ln_findings.get('total_samples')
        if total is None:
            fraction = ln_findings.get('estimated_noise_fraction', 0)
            total = int(round(errors / fraction)) if fraction > 0 else errors
        if total < errors:
            total = errors
        clean = total - errors
        fig_noise = go.Figure(go.Pie(
            labels=['Clean Labels', 'Suspected Noise'],
            values=[clean, errors],
            hole=.4,
            marker_colors=[FEYNML_CHART_COLORS['stable'], FEYNML_CHART_COLORS['critical']],
            textinfo='label+percent',
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
        ))
        fig_noise.update_layout(
            title='Label Integrity Profile',
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter, sans-serif', color='white')
        )
        charts['label_noise'] = json.dumps(fig_noise, cls=plotly.utils.PlotlyJSONEncoder)

    miss_findings = data.get('missing_data', {}).get('findings', {})
    if miss_findings and 'missing_rates' in miss_findings:
        m_rates = miss_findings['missing_rates']
        sorted_rates = sorted(m_rates.items(), key=lambda x: x[1], reverse=True)
        features = [x[0] for x in sorted_rates]
        rates = [x[1] * 100 for x in sorted_rates]
        colors = []
        for r in rates:
            if r >= 10:
                colors.append(FEYNML_CHART_COLORS['critical'])
            elif r > 0:
                colors.append(FEYNML_CHART_COLORS['critical'])
            else:
                colors.append(FEYNML_CHART_COLORS['stable'])
        fig_miss = go.Figure(go.Bar(
            y=features,
            x=rates,
            orientation='h',
            marker_color=colors,
            hovertemplate="Feature: %{y}<br>Missing: %{x:.2f}%<extra></extra>"
        ))
        fig_miss.update_layout(
            title='Missingness Analytics (Feature-wise %)',
            xaxis_title='Missing Percentage (%)',
            yaxis_title='Feature',
            height=max(350, len(features) * 25),
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter, sans-serif', color='white')
        )
        charts['missing_data'] = json.dumps(fig_miss, cls=plotly.utils.PlotlyJSONEncoder)

    # Ensure the dashboard knows the currently selected audience so it can render
    # audience-specific sections (e.g. AI Investigator report).
    # Priority:
    # 1) session['selected_audience'] (live selection)
    # 2) report JSON data['selected_audience'] (persisted selection)
    selected_audience = session.get('selected_audience') or data.get('selected_audience', 'ML Engineer')
    try:
        data['selected_audience'] = selected_audience
    except Exception:
        pass

    return render_template(
        'dashboard.html',
        data=data,
        report_id=report_id,
        filename=filename,
        risk_level=risk_level,
        critical_count=critical_count,
        alerts_count=alerts_count,
        charts=charts,
        selected_audience=selected_audience,
        why_risk=data.get('why_risk'),

        # Dashboard AI context
        root_cause_data=data.get('root_cause', {}),
        ai_investigator_data=data.get('ai_investigator', {}),
        audience_reports=data.get('audience_reports', {})
    )


@app.route('/dashboard')
@login_required
def dashboard_home():
    return redirect(url_for('saved_reports'))


@app.route('/reports')
@login_required
def reports_home():
    return redirect(url_for('saved_reports'))


@app.route('/report/<report_id>')
@login_required
def view_report(report_id):
    report = Report.query.filter_by(report_id=report_id, user_id=current_user.id).first()
    if not report:
        flash('Report not found in your workspace.', 'warning')
        return redirect(url_for('saved_reports'))

    filepath = os.path.join(REPORT_FOLDER, f"{report_id}.json")
    if not os.path.exists(filepath):
        flash('Report artifact missing.', 'danger')
        return redirect(url_for('saved_reports'))

    with open(filepath, 'r') as f:
        data = json.load(f)

    data.setdefault('calibration', {'status': 'SKIPPED', 'severity': 'NONE', 'findings': {}})
    _raw_fairness = data.get('fairness', {})
    if _raw_fairness and 'per_axis' in _raw_fairness:
        _per_axis = _raw_fairness.get('per_axis', {})
        _dp_gaps = []
        _groups = []
        for _axis, _axis_data in _per_axis.items():
            _groups.append(_axis)
            _dp = _axis_data.get('demographic_parity', {})
            _dp_gaps.append(_dp.get('max_difference', 0.0))
        _max_disparity = max(_dp_gaps) if _dp_gaps else 0.0
        data['fairness'] = {
            'status':   _raw_fairness.get('severity', 'LOW'),
            'severity': _raw_fairness.get('severity', 'LOW'),
            'findings': {
                'disparity_ratio': round(_max_disparity, 3),
                'max_disparity':   round(_max_disparity, 3),
                'groups':          _groups,
                'violations':      _raw_fairness.get('violations', []),
                'per_axis':        _per_axis,
            }
        }
    else:
        data.setdefault('fairness', {
            'status':   'INSUFFICIENT_DATA',
            'severity': 'NONE',
            'findings': {
                'disparity_ratio': None,
                'max_disparity':   None,
                'groups':          [],
                'violations':      [],
            }
        })
    data.setdefault('drift', {'status': 'SKIPPED', 'severity': 'NONE', 'findings': {}})
    data.setdefault('label_noise', {'status': 'SKIPPED', 'severity': 'NONE', 'findings': {}})
    data.setdefault('leakage', {'status': 'SKIPPED', 'severity': 'NONE', 'findings': {}})
    data.setdefault('missing_data', {'status': 'SKIPPED', 'severity': 'NONE', 'findings': {}})
    data.setdefault('root_cause', {'health_status': 'Unknown', 'confidence': 0, 'root_causes': [], 'recommended_actions': []})
    data.setdefault('ai_investigator', {'risk_level': 'UNKNOWN', 'executive_summary': '', 'investigation_findings': '', 'impact_assessment': '', 'confidence_explanation': '', 'recommended_actions': [], 'technical_notes': ''})
    data.setdefault('audience_reports', {})

    severities = [
        data.get('calibration', {}).get('severity', 'NONE'),
        data.get('fairness', {}).get('severity', 'NONE') if data.get('fairness') else 'NONE',
        data.get('drift', {}).get('severity', 'NONE'),
        data.get('label_noise', {}).get('severity', 'NONE'),
        data.get('leakage', {}).get('severity', 'NONE'),
        data.get('missing_data', {}).get('severity', 'NONE')
    ]
    severities = [s.upper() if s else 'NONE' for s in severities]

    # Use canonical status from RiskScorer (single source of truth)
    canonical_status = data.get('root_cause', {}).get('metadata', {}).get('risk', {}).get('level')
    if canonical_status:
        risk_level = canonical_status
    else:
        # Fallback to legacy calculation if RiskScorer data not available
        if 'CRITICAL' in severities or 'HIGH' in severities:
            risk_level = 'HIGH'
        elif 'MEDIUM' in severities:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'

    # Check for pre-generated charts in the report JSON
    charts = data.get('charts', {})
    print(f"DEBUG: Found {len(charts)} pre-generated charts in report JSON.")
    if charts:
        print(f"DEBUG: Chart keys: {list(charts.keys())}")
    
    # Extract suspicious samples for the table
    ln_findings = data.get('label_noise', {}).get('findings', {})
    suspicious_samples = ln_findings.get('suspicious_samples', [])[:20]

    # If charts are missing from JSON (legacy reports), generate them on the fly
    if not charts:
        # --- PHASE 2: CALIBRATION CHARTS ---
        cal_data = data.get('calibration', {})
        confidences = []
        accuracies = []
        if 'findings' in cal_data:
            findings_block = cal_data['findings']
            # Try direct keys first, then nested curve fallback
            confidences = (
                findings_block.get('confidences') or
                findings_block.get('curve', {}).get('mean_predicted') or
                cal_data.get('curve', {}).get('mean_predicted', [])
            )
            accuracies = (
                findings_block.get('accuracies') or
                findings_block.get('curve', {}).get('fraction_pos') or
                cal_data.get('curve', {}).get('fraction_pos', [])
            )
        elif 'curve' in cal_data:
            confidences = cal_data['curve'].get('mean_predicted', [])
            accuracies  = cal_data['curve'].get('fraction_pos',   [])
        else:
            confidences = []
            accuracies  = []
        
        # Prediction Distribution - histogram
        if confidences:
            fig_pred_dist = go.Figure()
            fig_pred_dist.add_trace(go.Histogram(
                x=confidences,
                name='Predicted Probabilities',
                marker_color=FEYNML_CHART_COLORS['primary'],
                nbinsx=30,
                opacity=0.7
            ))
            fig_pred_dist.update_layout(
                title='Prediction Confidence Distribution',
                xaxis_title='Predicted Probability',
                yaxis_title='Frequency',
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter, sans-serif', color='white'),
                showlegend=True
            )
            charts['prediction_dist'] = json.dumps(fig_pred_dist, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Residuals Plot
        if confidences and accuracies:
            residuals = [a - c for a, c in zip(accuracies, confidences)]
            fig_residuals = go.Figure()
            fig_residuals.add_trace(go.Scatter(
                x=confidences,
                y=residuals,
                mode='markers',
                marker=dict(size=8, color=FEYNML_CHART_COLORS['critical'], opacity=0.6),
                name='Residuals',
                hovertemplate='Predicted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>'
            ))
            fig_residuals.add_hline(y=0, line_dash='dash', line_color=FEYNML_CHART_COLORS['secondary'], annotation_text='Perfect Calibration')
            fig_residuals.update_layout(
                title='Residuals Plot (Actual - Predicted)',
                xaxis_title='Predicted Confidence',
                yaxis_title='Residual',
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter, sans-serif', color='white')
            )
            charts['residuals'] = json.dumps(fig_residuals, cls=plotly.utils.PlotlyJSONEncoder)
        
        # --- PHASE 3: DRIFT CHARTS ---
        drift_findings = data.get('drift', {}).get('findings', {})
        drift_features = drift_findings.get('per_feature', [])
        
        # PSI Heatmap
        if drift_features and len(drift_features) > 1:
            feature_names = [f['feature'] for f in drift_features]
            psi_values = [f.get('psi', 0) for f in drift_features]
            
            fig_psi = go.Figure(data=go.Heatmap(
                z=[psi_values],
                x=feature_names,
                y=['PSI Score'],
                colorscale='RdYlGn_r',
                text=[[f'{v:.3f}' for v in psi_values]],
                texttemplate='%{text}',
                textfont={"size": 12},
                colorbar=dict(title='PSI')
            ))
            fig_psi.update_layout(
                title='Population Stability Index (PSI) Across Features',
                height=200,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter, sans-serif', color='white'),
                xaxis=dict(side='bottom')
            )
            charts['psi_heatmap'] = json.dumps(fig_psi, cls=plotly.utils.PlotlyJSONEncoder)
        
        # KS Statistic Ranked Bar Chart
        if drift_features:
            sorted_features = sorted(drift_features, key=lambda x: x.get('ks_stat', 0), reverse=True)
            ks_names = [f['feature'] for f in sorted_features]
            ks_stats = [f.get('ks_stat', 0) for f in sorted_features]
            colors = [
                FEYNML_CHART_COLORS['critical'] if f.get('status') == 'DRIFT'
                else FEYNML_CHART_COLORS['high'] if f.get('status') == 'WARN'
                else FEYNML_CHART_COLORS['stable']
                for f in sorted_features
            ]
            
            fig_ks = go.Figure(data=go.Bar(
                y=ks_names,
                x=ks_stats,
                orientation='h',
                marker_color=colors,
                text=[f'{v:.3f}' for v in ks_stats],
                textposition='auto',
                hovertemplate='%{y}<br>KS Stat: %{x:.4f}<extra></extra>'
            ))
            fig_ks.update_layout(
                title='KS Statistic Ranked by Feature',
                xaxis_title='KS Statistic',
                height=300,
                margin=dict(l=150, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter, sans-serif', color='white'),
                showlegend=False
            )
            charts['ks_ranked'] = json.dumps(fig_ks, cls=plotly.utils.PlotlyJSONEncoder)
        
        # --- LABEL NOISE CHARTS ---
        # Noise Score Distribution histogram
        noise_scores = ln_findings.get('noise_scores', [])
        if noise_scores:
            fig_noise_dist = go.Figure()
            fig_noise_dist.add_trace(go.Histogram(
                x=noise_scores,
                name='Noise Scores',
                marker_color=FEYNML_CHART_COLORS['stable'],
                nbinsx=30,
                opacity=0.7
            ))
            fig_noise_dist.update_layout(
                title='Label Noise Score Distribution',
                xaxis_title='Noise Score',
                yaxis_title='Frequency',
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter, sans-serif', color='white')
            )
            charts['noise_score_dist'] = json.dumps(fig_noise_dist, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Existing Label Noise Pie Chart
        if ln_findings:
            errors = ln_findings.get('num_errors_detected', 0)
            total = ln_findings.get('total_samples')
            if total is None:
                fraction = ln_findings.get('estimated_noise_fraction', 0)
                total = int(round(errors / fraction)) if fraction > 0 else errors
            if total < errors:
                total = errors
            clean = total - errors
            fig_noise = go.Figure(go.Pie(
                labels=['Clean', 'Noisy'],
                values=[clean, errors],
                hole=.4,
                marker_colors=[FEYNML_CHART_COLORS['stable'], FEYNML_CHART_COLORS['critical']],
                textinfo='label+percent',
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
            ))
            fig_noise.update_layout(
                title='Label Integrity',
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter, sans-serif', color='white')
            )
            charts['label_noise'] = json.dumps(fig_noise, cls=plotly.utils.PlotlyJSONEncoder)
        
        # --- LEAKAGE CHARTS ---
        leakage_findings = data.get('leakage', {}).get('findings', {})
        
        # Feature Correlation Heatmap
        correlations = leakage_findings.get('correlations', {})
        if correlations and isinstance(correlations, dict):
            features_list = list(correlations.keys())
            corr_matrix = []
            for f1 in features_list:
                row = []
                for f2 in features_list:
                    if f1 == f2:
                        row.append(1.0)
                    else:
                        row.append(correlations.get(f1, {}).get(f2, 0) if isinstance(correlations.get(f1), dict) else 0)
                corr_matrix.append(row)
            
            if corr_matrix:
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix,
                    x=features_list,
                    y=features_list,
                    colorscale='RdBu',
                    zmid=0,
                    text=[[f'{v:.2f}' for v in row] for row in corr_matrix],
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    colorbar=dict(title='Correlation')
                ))
                fig_corr.update_layout(
                    title='Feature Correlation Matrix',
                    height=400,
                    margin=dict(l=100, r=20, t=40, b=100),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter, sans-serif', color='white')
                )
                charts['correlation_heatmap'] = json.dumps(fig_corr, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Leakage Score Bar Chart
        leakage_suspects = leakage_findings.get('suspects', [])
        if leakage_suspects:
            suspect_names = [s.get('feature', 'Unknown') for s in leakage_suspects]
            suspect_scores = [s.get('score', 0) for s in leakage_suspects]
            
            fig_leakage = go.Figure(data=go.Bar(
                x=suspect_names,
                y=suspect_scores,
                marker_color=FEYNML_CHART_COLORS['critical'],
                text=[f'{v:.3f}' for v in suspect_scores],
                textposition='auto',
                hovertemplate='%{x}<br>Leakage Score: %{y:.4f}<extra></extra>'
            ))
            fig_leakage.update_layout(
                title='Feature Leakage Scores',
                xaxis_title='Feature',
                yaxis_title='Leakage Score',
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter, sans-serif', color='white')
            )
            charts['leakage_scores'] = json.dumps(fig_leakage, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Missing Data chart (existing)
        miss_findings = data.get('missing_data', {}).get('findings', {})
        if miss_findings and ('missingness_rates' in miss_findings
                              or 'missing_rates' in miss_findings):
            m_rates = (miss_findings.get('missingness_rates')
                       or miss_findings.get('missing_rates', {}))
            features = list(m_rates.keys())
            rates = [v * 100 for v in m_rates.values()]
            fig_miss = go.Figure(go.Bar(x=features, y=rates, marker_color=FEYNML_CHART_COLORS['secondary']))
            fig_miss.update_layout(title='Missing Data (%)', xaxis_title='Feature', yaxis_title='Missing %', height=350, margin=dict(l=20, r=20, t=40, b=20), font=dict(color='white'))
            charts['missing_data'] = json.dumps(fig_miss, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('report.html', data=data, report_id=report_id, risk_level=risk_level, charts=charts, suspicious_samples=suspicious_samples)


@app.route('/visual_explorer/<report_id>')
@login_required
def visual_explorer(report_id):
    """Visual Explorer page for ad-hoc chart generation"""
    report = Report.query.filter_by(report_id=report_id, user_id=current_user.id).first()
    if not report:
        flash('Report not found in your workspace.', 'warning')
        return redirect(url_for('saved_reports'))

    # Load report data
    filepath = os.path.join(REPORT_FOLDER, f"{report_id}.json")
    if not os.path.exists(filepath):
        flash('Report artifact missing.', 'danger')
        return redirect(url_for('saved_reports'))

    with open(filepath, 'r') as f:
        report_data = json.load(f)

    # Load dataset from the original analysis dataset record, or derive it from the saved report metadata
    columns = []
    dataset_path = None
    dataset_name = None

    if report.analysis and getattr(report.analysis, 'dataset_name', None):
        dataset_name = report.analysis.dataset_name

    if not dataset_name:
        dataset_name = report_data.get('dataset', {}).get('name')

    if dataset_name:
        dataset = Dataset.query.filter_by(user_id=current_user.id, original_name=dataset_name).order_by(Dataset.uploaded_at.desc()).first()
        if dataset and dataset.upload_path and os.path.exists(dataset.upload_path):
            dataset_path = dataset.upload_path

    if not dataset_path:
        # Fallback: try a dataset path in the current session if available
        session_dataset_id = session.get('dataset_id')
        if session_dataset_id:
            dataset = Dataset.query.filter_by(id=session_dataset_id, user_id=current_user.id).first()
            if dataset and dataset.upload_path and os.path.exists(dataset.upload_path):
                dataset_path = dataset.upload_path
                dataset_name = dataset.original_name

    column_meta = {}
    try:
        df, err = _get_df(report_id, dataset_path=dataset_path)
        if df is not None:
            columns = df.columns.tolist()
            for col in columns:
                column_meta[col] = {
                    'dtype': str(df[col].dtype),
                    'nunique': int(df[col].nunique(dropna=False))
                }
        else:
            print(f"Visual Explorer dataset not found for report_id={report_id}, err={err}")
            columns = []
    except Exception as e:
        print(f"Error loading dataset for Visual Explorer: {str(e)}")
        columns = []
        column_meta = {}

    # Generate suggestions based on analysis results
    suggestions = generate_chart_suggestions(report_data, columns)

    return render_template(
        'visual_explorer.html',
        report_id=report_id,
        columns=columns,
        column_meta=column_meta,
        suggestions=suggestions,
        report_data=report_data,
        body_class='dark-mode'
    )




def generate_chart_suggestions(report_data, columns):
    """Generate suggested charts based on analysis findings"""
    suggestions = []
    
    # Suggestion 1: Drift analysis if drift detected
    drift_data = report_data.get('drift', {})
    if drift_data.get('severity') in ['HIGH', 'CRITICAL']:
        drift_features = drift_data.get('findings', {}).get('per_feature', [])
        if drift_features and len(columns) >= 2:
            # Sort by KS stat
            top_feature = sorted(drift_features, key=lambda x: x.get('ks_stat', 0), reverse=True)[0]
            feature_name = top_feature.get('feature', columns[0] if columns else 'Feature')
            suggestions.append({
                'title': 'Top Drifted Feature Distribution',
                'description': f'Compare distribution of {feature_name} across time periods',
                'x_axis': 'Bins' if feature_name in columns else columns[0] if columns else '',
                'y_axis': 'Frequency',
                'chart_type': 'Bar Chart',
                'color_by': None
            })
    
    # Suggestion 2: Label noise histogram if label noise detected
    ln_data = report_data.get('label_noise', {})
    if ln_data.get('severity') in ['HIGH', 'CRITICAL']:
        suggestions.append({
            'title': 'Noise Score Distribution',
            'description': 'Histogram of confidence scores for predicted labels',
            'x_axis': 'Confidence Score Bins',
            'y_axis': 'Sample Count',
            'chart_type': 'Histogram',
            'color_by': None
        })
    
    # Suggestion 3: Feature correlation scatter if multiple numeric columns
    if len(columns) >= 2:
        numeric_cols = columns[:2]  # Use first two as defaults
        suggestions.append({
            'title': 'Feature Relationship',
            'description': f'Explore relationship between {numeric_cols[0]} and {numeric_cols[1]}',
            'x_axis': numeric_cols[0],
            'y_axis': numeric_cols[1],
            'chart_type': 'Scatter Plot',
            'color_by': numeric_cols[1] if len(columns) > 2 else None
        })
    
    return suggestions[:3]  # Return top 3 suggestions


def _save_report_visualization(report_id, user_id, chart_data, chart_title, chart_description=None):
    """Persist a custom visualization snapshot into the report JSON."""
    report = Report.query.filter_by(report_id=report_id, user_id=user_id).first()
    if not report:
        return None, ({'error': 'Report not found'}, 404)

    filepath = os.path.join(REPORT_FOLDER, f"{report_id}.json")
    if not os.path.exists(filepath):
        return None, ({'error': 'Report artifact missing'}, 404)

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except Exception as exc:
        return None, ({'error': f'Unable to read report artifact: {exc}'}, 500)

    if not isinstance(payload, dict):
        return None, ({'error': 'Invalid report payload'}, 500)

    custom_visualizations = payload.setdefault('custom_visualizations', [])
    if not isinstance(custom_visualizations, list):
        custom_visualizations = []
        payload['custom_visualizations'] = custom_visualizations

    visualization = {
        'id': uuid4().hex,
        'title': chart_title or 'Untitled Visualization',
        'description': chart_description or '',
        'chart_json': chart_data,
        'created_at': datetime.utcnow().isoformat()
    }
    custom_visualizations.append(visualization)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    report.size_bytes = os.path.getsize(filepath)
    db.session.commit()
    return visualization, None


@app.route('/add_to_report', methods=['POST'])
@app.route('/api/add-to-report', methods=['POST'])
@login_required
def add_to_report():
    data = request.get_json(silent=True) or {}
    report_id = str(data.get('report_id') or '').strip()
    chart_data = data.get('chart_data') or data.get('chart_json')
    chart_title = data.get('chart_title')
    chart_description = data.get('chart_description')

    if not report_id:
        return jsonify({'error': 'report_id is required'}), 400
    if not chart_data:
        return jsonify({'error': 'chart_data is required'}), 400

    visualization, error = _save_report_visualization(
        report_id=report_id,
        user_id=current_user.id,
        chart_data=chart_data,
        chart_title=chart_title,
        chart_description=chart_description,
    )
    if error:
        body, status = error
        return jsonify(body), status

    return jsonify({
        'success': True,
        'message': 'Visualization added to report',
        'visualization': visualization
    })


@app.route('/api/save-visualization', methods=['POST'])
@app.route('/save_visualization', methods=['POST'])
@login_required
def save_visualization():
    data = request.get_json(silent=True) or {}
    report_id = str(data.get('report_id') or '').strip()
    chart_data = data.get('chart_data') or data.get('chart_json')
    chart_title = data.get('chart_title')
    chart_description = data.get('chart_description')

    if not report_id:
        return jsonify({'error': 'report_id is required'}), 400
    if not chart_data:
        return jsonify({'error': 'chart_data is required'}), 400

    visualization, error = _save_report_visualization(
        report_id=report_id,
        user_id=current_user.id,
        chart_data=chart_data,
        chart_title=chart_title,
        chart_description=chart_description,
    )
    if error:
        body, status = error
        return jsonify(body), status

    return jsonify({
        'success': True,
        'message': 'Visualization saved',
        'visualization': visualization
    })


@app.route('/generate_chart', methods=['POST'])
@app.route('/api/generate-chart', methods=['POST'])
@login_required
def generate_chart():
    """Generate a chart based on user selections"""
    try:
        data = request.get_json()
        print("=== CHART REQUEST ===")
        print("Full request data:", data)
        print("chart_type raw:", repr(data.get('chart_type')))
        print("x_col:", data.get('x_col') or data.get('x_axis'))
        print("y_col:", data.get('y_col') or data.get('y_axis'))
        print("====================")

        report_id = str(data.get('report_id') or '').strip()
        print('\n=== REPORT DEBUG ===')
        print('Current User:', current_user.id)
        print('Authenticated:', current_user.is_authenticated)
        print('Report ID:', report_id)
        print('====================\n')
        x_axis = data.get('x_axis')
        y_axis = data.get('y_axis')
        chart_type_raw = data.get('chart_type', 'scatter')
        chart_type = (chart_type_raw or 'scatter').strip().lower()
        chart_type_display = {
            'scatter': 'Scatter',
            'bar': 'Bar',
            'histogram': 'Histogram',
            'box': 'Box',
            'line': 'Line',
            'heatmap': 'Heatmap',
            'violin': 'Violin'
        }.get(chart_type, chart_type.title())
        print("Normalized chart_type:", chart_type)
        color_by = data.get('color_by')
        show_legend = data.get('show_legend', True)
        filters = data.get('filters', [])
        
        # Verify report ownership
        report_lookup = Report.query.filter_by(report_id=report_id, user_id=current_user.id).first()
        print('Report Lookup:', report_lookup)
        report = report_lookup
        if not report:
            return jsonify({
                'error':
                    f'Report not found. '
                    f'report_id={report_id}, '
                    f'user_id={current_user.id}'
            }), 403
        
        # Load dataset from the analysis record or fallback to session dataset
        dataset_path = None
        dataset = None
        dataset_name = None

        if report.analysis and getattr(report.analysis, 'dataset_name', None):
            dataset_name = report.analysis.dataset_name

        if not dataset_name:
            report_json_path = os.path.join(REPORT_FOLDER, f"{report_id}.json")
            try:
                if os.path.exists(report_json_path):
                    with open(report_json_path, 'r') as report_file:
                        report_json = json.load(report_file)
                        dataset_name = report_json.get('dataset', {}).get('name')
            except Exception:
                dataset_name = None

        if dataset_name:
            dataset = Dataset.query.filter_by(user_id=current_user.id, original_name=dataset_name).order_by(Dataset.uploaded_at.desc()).first()
            if dataset and dataset.upload_path and os.path.exists(dataset.upload_path):
                dataset_path = dataset.upload_path

        if not dataset_path:
            session_dataset_id = session.get('dataset_id')
            if session_dataset_id:
                dataset = Dataset.query.filter_by(id=session_dataset_id, user_id=current_user.id).first()
                if dataset and dataset.upload_path and os.path.exists(dataset.upload_path):
                    dataset_path = dataset.upload_path
                    dataset_name = dataset.original_name

        if not dataset_path:
            return jsonify({'error': 'Dataset file not found for report'}), 404

        df, err = _get_df(report_id, x_col=x_axis, dataset_path=dataset_path)
        if err:
            return jsonify({'error': err}), 404

        # ── Guard: dataset must have actual rows ───────────────────────────
        if df is None or len(df) < 5:
            return jsonify({'error': 'Dataset not found or contains too few rows to generate a chart.'}), 404

        df = df.copy()
        
        # Apply filters
        for filter_item in filters:
            col = filter_item.get('column')
            op = filter_item.get('operator')
            val = filter_item.get('value')
            if col in df.columns:
                if op == '>':
                    df = df[df[col] > float(val)]
                elif op == '<':
                    df = df[df[col] < float(val)]
                elif op == '==':
                    df = df[df[col] == val]
        
        # For Scatter, sample large datasets to improve performance
        if chart_type == 'scatter' and len(df) > 3000:
            df = df.sample(min(1500, len(df)), random_state=42)
        
        # Validate dataframe is not empty after filtering
        if len(df) == 0:
            return jsonify({'error': 'No data available after applying filters. Try adjusting your filter criteria.'}), 400
        
        # Validate columns exist
        if chart_type in ['scatter', 'bar', 'line', 'box', 'violin']:
            if x_axis not in df.columns or y_axis not in df.columns:
                return jsonify({'error': f'Selected columns not found in dataset. X: {x_axis}, Y: {y_axis}'}), 400
        
        if chart_type == 'histogram' and x_axis not in df.columns:
            return jsonify({'error': f'Selected column "{x_axis}" not found in dataset.'}), 400
        
        if chart_type == 'heatmap':
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            if len(numeric_cols) < 2:
                return jsonify({'error': 'At least two numeric columns are required for a Heatmap.'}), 400
        
        numeric_charts = [
            'scatter',
            'line',
            'bar',
            'box',
            'violin'
        ]

        if chart_type in numeric_charts:
            if not pd.api.types.is_numeric_dtype(df[y_axis]):
                return jsonify({
                    'error': f'Column "{y_axis}" must be numeric for {chart_type} chart.'
                }), 400
        
        # Validate Bar Chart constraints
        if chart_type == 'bar':
            if x_axis == y_axis:
                return jsonify({'error': 'X and Y axis cannot be the same column for a Bar chart. Use Histogram instead.'}), 400
        
        color = None if not color_by or color_by == 'None' else color_by
        
        # Generate chart based on type
        fig = None
        
        if chart_type == 'scatter':
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color)
        
        elif chart_type == 'bar':
            if pd.api.types.is_numeric_dtype(df[x_axis]):
                df['_binned'] = pd.cut(df[x_axis], bins=12)
                grouped = (
                    df.groupby('_binned')[y_axis]
                    .mean()
                    .reset_index()
                )
                grouped['_binned'] = grouped['_binned'].astype(str)
                fig = px.bar(grouped, x='_binned', y=y_axis)
            else:
                grouped = (
                    df.groupby(x_axis)[y_axis]
                    .mean()
                    .reset_index()
                )
                fig = px.bar(grouped, x=x_axis, y=y_axis)
        
        elif chart_type == 'histogram':
            fig = px.histogram(df, x=x_axis, color=color, nbins=30)
        
        elif chart_type == 'box':
            fig = px.box(df, x=x_axis, y=y_axis, color=color)
        
        elif chart_type == 'line':
            df_sorted = df.sort_values(by=x_axis)
            fig = px.line(df_sorted, x=x_axis, y=y_axis, color=color)
        
        elif chart_type == 'heatmap':
            corr = df.select_dtypes(include='number').corr()
            fig = px.imshow(
                corr,
                text_auto='.2f',
                color_continuous_scale='Blues'
            )
        
        elif chart_type == 'violin':
            fig = px.violin(df, x=x_axis, y=y_axis, color=color, box=True, points='outliers')
        
        else:
            return jsonify({'error': f"Unknown chart type received: '{chart_type}'"}), 400
        
        if fig is None:
            return jsonify({'error': 'Could not generate chart'}), 400
        
        # Update layout
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(
                family='Inter',
                color='#e2e8f0',
                size=12
            ),
            margin=dict(
                l=60,
                r=20,
                t=50,
                b=100
            ),
            hovermode='closest',
            legend=dict(
                bgcolor='rgba(15,23,42,0.85)',
                bordercolor='#334155',
                borderwidth=1
            ),
            title=dict(
                text=f'{chart_type_display}: {x_axis} vs {y_axis}',
                font=dict(size=15)
            ),
            showlegend=show_legend
        )
        
        if chart_type == 'bar':
            fig.update_xaxes(tickmode='auto', nticks=20)

        print('Chart Type:', chart_type)
        print('Rows:', len(df))
        print('Columns:', list(df.columns))
        print('X:', x_axis)
        print('Y:', y_axis)

        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({
            'chart_json': chart_json,
            'chart': chart_json,
            'success': True
        })
    
    except Exception as e:
        print(f"Error generating chart: {str(e)}")
        return jsonify({'error': str(e)}), 500







@app.route('/export/json/<report_id>')
@login_required
def export_json(report_id):
    report = Report.query.filter_by(report_id=report_id, user_id=current_user.id).first()
    if not report:
        flash('Report not found in your workspace.', 'warning')
        return redirect(url_for('saved_reports'))

    filepath = os.path.join(REPORT_FOLDER, f"{report_id}.json")
    if not os.path.exists(filepath):
        flash('Report artifact missing.', 'danger')
        return redirect(url_for('saved_reports'))

    return send_file(filepath, as_attachment=True, download_name=f"report_{report_id}.json")


@app.route('/export/csv/<report_id>')
@login_required
def export_csv(report_id):
    report = Report.query.filter_by(report_id=report_id, user_id=current_user.id).first()
    if not report:
        flash('Report not found in your workspace.', 'warning')
        return redirect(url_for('saved_reports'))

    filepath = os.path.join(REPORT_FOLDER, f"{report_id}.json")
    if not os.path.exists(filepath):
        flash('Report artifact missing.', 'danger')
        return redirect(url_for('saved_reports'))

    with open(filepath, 'r') as f:
        data = json.load(f)

    calibration_ece = data.get('calibration', {}).get('ece', 0.0)
    fairness_sev = data.get('fairness', {}).get('severity', 'N/A')
    drift_sev = data.get('drift', {}).get('severity', 'N/A')
    drift_count = data.get('drift', {}).get('findings', {}).get('num_drifted_features', 0)
    noise_frac = data.get('label_noise', {}).get('findings', {}).get('estimated_noise_fraction', 0.0)
    leakage_count = data.get('leakage', {}).get('findings', {}).get('num_suspects', 0)

    mechanism = 'N/A'
    if data.get('missing_data', {}).get('findings', {}).get('mechanisms'):
        mech_vals = list(data['missing_data']['findings']['mechanisms'].values())
        if mech_vals:
            mechanism = mech_vals[0]

    severities = [
        data.get('calibration', {}).get('severity', 'NONE'),
        data.get('fairness', {}).get('severity', 'NONE') if data.get('fairness') else 'NONE',
        data.get('drift', {}).get('severity', 'NONE'),
        data.get('label_noise', {}).get('severity', 'NONE'),
        data.get('leakage', {}).get('severity', 'NONE'),
        data.get('missing_data', {}).get('severity', 'NONE')
    ]
    severities = [s.upper() if s else 'NONE' for s in severities]
    critical_count = severities.count('CRITICAL') + severities.count('HIGH')
    risk_level = 'HIGH' if critical_count > 0 else ('MEDIUM' if (severities.count('MEDIUM') + severities.count('LOW')) > 0 else 'LOW')

    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Metric', 'Value'])
    cw.writerow(['Report ID', report_id])
    cw.writerow(['Global Risk', risk_level])
    cw.writerow(['Calibration ECE', f"{calibration_ece:.4f}"])
    cw.writerow(['Fairness Severity', fairness_sev])
    cw.writerow(['Drift Severity', drift_sev])
    cw.writerow(['Drifted Feature Count', drift_count])
    cw.writerow(['Label Noise %', f"{noise_frac*100:.2f}%"])
    cw.writerow(['Leakage Suspect Count', leakage_count])
    cw.writerow(['Missingness Mechanism', mechanism])
    cw.writerow(['Generated Timestamp', data.get('drift', {}).get('timestamp', 'Unknown')])
    
    # Add custom visualizations metadata
    custom_viz = data.get('custom_visualizations', [])
    if custom_viz:
        cw.writerow([''])
        cw.writerow(['Custom Visualizations', len(custom_viz)])
        for i, vis in enumerate(custom_viz, 1):
            cw.writerow([f"  - {vis.get('title', 'Untitled')}", vis.get('created_at', 'Unknown')[:10]])

    output = io.BytesIO()
    output.write(si.getvalue().encode('utf-8'))
    output.seek(0)
    return send_file(output, as_attachment=True, download_name=f"investigation_summary_{report_id}.csv", mimetype='text/csv')


@app.route('/export/pdf/<report_id>')
@login_required
def export_pdf(report_id):
    report = Report.query.filter_by(report_id=report_id, user_id=current_user.id).first()
    if not report:
        flash('Report not found in your workspace.', 'warning')
        return redirect(url_for('saved_reports'))

    filepath = os.path.join(REPORT_FOLDER, f"{report_id}.json")
    if not os.path.exists(filepath):
        flash('Report artifact missing.', 'danger')
        return redirect(url_for('saved_reports'))

    with open(filepath, 'r') as f:
        data = json.load(f)

    severities = [
        data.get('calibration', {}).get('severity', 'NONE'),
        data.get('fairness', {}).get('severity', 'NONE') if data.get('fairness') else 'NONE',
        data.get('drift', {}).get('severity', 'NONE'),
        data.get('label_noise', {}).get('severity', 'NONE'),
        data.get('leakage', {}).get('severity', 'NONE'),
        data.get('missing_data', {}).get('severity', 'NONE')
    ]
    severities = [s.upper() if s else 'NONE' for s in severities]
    if 'CRITICAL' in severities or 'HIGH' in severities:
        risk_level = 'HIGH'
    elif 'MEDIUM' in severities:
        risk_level = 'MEDIUM'
    else:
        risk_level = 'LOW'

    # Generate charts and suspicious samples
    charts = {}
    suspicious_samples = []
    
    try:
        import plotly.graph_objects as go
        import plotly.io as pio

        PRINT_LAYOUT = dict(
            paper_bgcolor='#ffffff',
            plot_bgcolor='#ffffff',
            font=dict(color='#111111', family='Inter, Arial, sans-serif', size=11),
            margin=dict(l=48, r=24, t=48, b=48),
            xaxis=dict(gridcolor='#e2e8f0', linecolor='#cbd5e1', tickfont=dict(size=10)),
            yaxis=dict(gridcolor='#e2e8f0', linecolor='#cbd5e1', tickfont=dict(size=10)),
        )

        def make_layout(title, xtitle='', ytitle='', **kwargs):
            l = dict(**PRINT_LAYOUT)
            l['title'] = dict(text=title, font=dict(size=13, color='#0f172a'), x=0)
            if xtitle: l['xaxis'] = dict(**l['xaxis'], title=dict(text=xtitle, font=dict(size=11)))
            if ytitle: l['yaxis'] = dict(**l['yaxis'], title=dict(text=ytitle, font=dict(size=11)))
            l.update(kwargs)
            return l

        # ── Calibration: Prediction Distribution ──
        if data.get('calibration', {}).get('prediction_distribution'):
            pred_dist_data = data['calibration']['prediction_distribution']
            charts['prediction_dist'] = go.Figure(
                data=[go.Histogram(
                    x=pred_dist_data,
                    marker_color=FEYNML_CHART_COLORS['primary'],
                    opacity=0.85,
                    name='Score Distribution'
                )],
                layout=make_layout(
                    'Prediction Score Distribution',
                    xtitle='Predicted Probability',
                    ytitle='Sample Count'
                )
            ).to_json()

        # ── Calibration: Residuals ──
        if data.get('calibration', {}).get('residuals'):
            residuals = data['calibration']['residuals']
            charts['residuals'] = go.Figure(
                data=[go.Scatter(
                    y=residuals,
                    mode='markers',
                    marker=dict(color=FEYNML_CHART_COLORS['critical'], size=4, opacity=0.6),
                    name='Residual'
                )],
                layout=make_layout(
                    'Calibration Residuals',
                    xtitle='Sample Index',
                    ytitle='Residual Value (Predicted − Actual)'
                )
            ).to_json()

        # ── Drift: PSI Bar Chart ──
        if data.get('drift', {}).get('findings', {}).get('psi_scores'):
            psi_scores = data['drift']['findings']['psi_scores']
            features = list(psi_scores.keys())
            psi_vals  = list(psi_scores.values())
            colors = [FEYNML_CHART_COLORS['critical'] if v > 0.25 else FEYNML_CHART_COLORS['critical'] if v > 0.1 else FEYNML_CHART_COLORS['stable']
                      for v in psi_vals]
            charts['psi_heatmap'] = go.Figure(
                data=[go.Bar(
                    x=features, y=psi_vals,
                    marker_color=colors,
                    name='PSI',
                    text=[f'{v:.3f}' for v in psi_vals],
                    textposition='outside'
                )],
                layout=make_layout(
                    'Population Stability Index (PSI) — Feature Drift Severity',
                    xtitle='Feature',
                    ytitle='PSI Score',
                    shapes=[
                        dict(type='line', y0=0.1, y1=0.1, x0=0, x1=1,
                                xref='paper', line=dict(color=FEYNML_CHART_COLORS['critical'], dash='dash', width=1)),
                            dict(type='line', y0=0.25, y1=0.25, x0=0, x1=1,
                                xref='paper', line=dict(color=FEYNML_CHART_COLORS['critical'], dash='dash', width=1))
                    ],
                    annotations=[
                            dict(x=1, y=0.1, xref='paper', text='Warn (0.1)',
                                showarrow=False, font=dict(size=9, color=FEYNML_CHART_COLORS['critical']), xanchor='right'),
                            dict(x=1, y=0.25, xref='paper', text='Critical (0.25)',
                                showarrow=False, font=dict(size=9, color=FEYNML_CHART_COLORS['critical']), xanchor='right')
                    ]
                )
            ).to_json()

        # ── Drift: KS Ranked ──
        per_feature = data.get('drift', {}).get('findings', {}).get('per_feature', [])
        if per_feature:
            sorted_feats = sorted(
                [f for f in per_feature if isinstance(f, dict) and f.get('ks_stat') is not None],
                key=lambda x: x.get('ks_stat', 0), reverse=True
            )[:20]
            if sorted_feats:
                feat_names = [f['feature'] for f in sorted_feats]
                ks_vals    = [f.get('ks_stat', 0) for f in sorted_feats]
                ks_colors  = [FEYNML_CHART_COLORS['critical'] if f.get('status') == 'DRIFT'
                               else FEYNML_CHART_COLORS['critical'] if f.get('status') == 'WARN'
                               else FEYNML_CHART_COLORS['stable'] for f in sorted_feats]
                charts['ks_ranked'] = go.Figure(
                    data=[go.Bar(
                        x=ks_vals, y=feat_names,
                        orientation='h',
                        marker_color=ks_colors,
                        name='KS Statistic',
                        text=[f'{v:.3f}' for v in ks_vals],
                        textposition='outside'
                    )],
                    layout=make_layout(
                        'KS Statistic Ranking — Top Drifted Features',
                        xtitle='KS Statistic (higher = more drift)',
                        ytitle='Feature'
                    )
                ).to_json()

        # ── Label Noise: Score Distribution ──
        if data.get('label_noise', {}).get('findings', {}).get('noise_scores'):
            noise_scores = data['label_noise']['findings']['noise_scores']
            charts['noise_score_dist'] = go.Figure(
                data=[go.Histogram(
                    x=noise_scores,
                    nbinsx=30,
                    marker_color=FEYNML_CHART_COLORS['primary'],
                    opacity=0.85,
                    name='Noise Score'
                )],
                layout=make_layout(
                    'Label Noise Score Distribution',
                    xtitle='Noise Score (0 = clean, 1 = likely mislabeled)',
                    ytitle='Number of Samples'
                )
            ).to_json()

        # ── Leakage: Score Bar Chart ──
        if data.get('leakage', {}).get('findings', {}).get('leakage_scores'):
            leakage_scores = data['leakage']['findings']['leakage_scores']
            if isinstance(leakage_scores, dict):
                sorted_lk = sorted(leakage_scores.items(), key=lambda x: x[1], reverse=True)[:20]
                lk_feats  = [x[0] for x in sorted_lk]
                lk_vals   = [x[1] for x in sorted_lk]
                charts['leakage_scores'] = go.Figure(
                    data=[go.Bar(
                        x=lk_feats, y=lk_vals,
                        marker_color=FEYNML_CHART_COLORS['critical'],
                        opacity=0.85,
                        name='Leakage Score',
                        text=[f'{v:.3f}' for v in lk_vals],
                        textposition='outside'
                    )],
                    layout=make_layout(
                        'Feature Leakage Score Ranking',
                        xtitle='Feature',
                        ytitle='Leakage Score (higher = higher risk)'
                    )
                ).to_json()

    except Exception as e:
        print(f"[PDF Export] Chart generation error: {str(e)}")
    
    # Extract suspicious samples from label_noise findings
    if data.get('label_noise', {}).get('findings', {}).get('noisy_instances'):
        noisy = data['label_noise']['findings']['noisy_instances'][:20]
        for item in noisy:
            suspicious_samples.append({
                'index': item.get('idx', 'N/A'),
                'actual_label': item.get('label', 'N/A'),
                'confidence_score': item.get('noise_score', 0),
                'noise_indicator': 'Noisy' if item.get('noise_score', 0) > 0.5 else 'Clean'
            })

    try:
        from xhtml2pdf import pisa
        html = render_template(
            'report.html',
            data=data,
            report_id=report_id,
            risk_level=risk_level,
            is_pdf=True,
            charts=charts,
            suspicious_samples=suspicious_samples
        )
        result = io.BytesIO()
        pdf = pisa.pisaDocument(io.BytesIO(html.encode('UTF-8')), result)
        if not pdf.err:
            result.seek(0)
            return send_file(result, as_attachment=True, download_name=f"investigation_report_{report_id}.pdf", mimetype='application/pdf')
    except ImportError:
        pass

    return render_template(
        'report.html',
        data=data,
        report_id=report_id,
        risk_level=risk_level,
        print_on_load=True,
        charts=charts,
        suspicious_samples=suspicious_samples
    )


@app.route('/saved-reports')
@login_required
def saved_reports():
    reports = current_user.reports.order_by(Report.created_at.desc()).all()
    return render_template('saved_reports.html', reports=reports)


@app.route('/saved-reports/delete/<string:report_id>', methods=['POST', 'GET'])
@login_required
def delete_report(report_id):
    report = Report.query.filter_by(report_id=report_id, user_id=current_user.id).first()
    if not report:
        flash('Report not found in your workspace.', 'warning')
        return redirect(url_for('saved_reports'))

    filepath = os.path.join(REPORT_FOLDER, f"{report_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)

    db.session.delete(report)
    db.session.commit()
    flash('Report deleted successfully.', 'success')
    return redirect(url_for('saved_reports'))


@app.route('/notes')
@login_required
def notes():
    q = request.args.get('q', '').strip()
    notes_query = Note.query.filter_by(user_id=current_user.id)
    if q:
        notes_query = notes_query.filter(or_(Note.title.ilike(f'%{q}%'), Note.content.ilike(f'%{q}%')))
    notes_list = notes_query.order_by(Note.updated_at.desc()).all()
    return render_template('notes.html', notes=notes_list, q=q)


@app.route('/notes/save', methods=['POST'])
@login_required
def save_note():
    note_id = request.form.get('note_id')
    title = request.form.get('title', '').strip()
    content = request.form.get('content', '').strip()
    if not title or not content:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'status': 'error', 'message': 'Both title and content are required for notes.'}), 400
        flash('Both title and content are required for notes.', 'warning')
        return redirect(url_for('notes'))

    note = None
    if note_id:
        note = Note.query.filter_by(id=int(note_id), user_id=current_user.id).first()
        if note:
            note.title = title
            note.content = content
            note.updated_at = datetime.utcnow()
        else:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'status': 'error', 'message': 'Note not found.'}), 404
            flash('Note not found.', 'warning')
            return redirect(url_for('notes'))
    else:
        note = Note(user_id=current_user.id, title=title, content=content)
        db.session.add(note)

    db.session.commit()
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'status': 'success', 'note_id': note.id, 'message': 'Note saved.'})

    flash('Note saved successfully.', 'success')
    return redirect(url_for('notes'))


@app.route('/notes/delete/<int:note_id>', methods=['POST', 'GET'])
@login_required
def delete_note(note_id):
    note = Note.query.filter_by(id=note_id, user_id=current_user.id).first()
    if not note:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'status': 'error', 'message': 'Note not found.'}), 404
        flash('Note not found.', 'warning')
        return redirect(url_for('notes'))

    db.session.delete(note)
    db.session.commit()
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'status': 'success', 'message': 'Note deleted.'})
    flash('Note deleted successfully.', 'success')
    return redirect(url_for('notes'))


@app.route('/tasks')
@login_required
def tasks():
    status_filter = request.args.get('status', 'all')
    tasks_query = Task.query.filter_by(user_id=current_user.id)
    if status_filter == 'pending':
        tasks_query = tasks_query.filter_by(is_completed=False)
    elif status_filter == 'completed':
        tasks_query = tasks_query.filter_by(is_completed=True)
    tasks_list = tasks_query.order_by(Task.is_completed.asc(), Task.due_date.asc().nulls_last(), Task.updated_at.desc()).all()
    return render_template('tasks.html', tasks=tasks_list, status_filter=status_filter)


@app.route('/tasks/save', methods=['POST'])
@login_required
def save_task():
    task_id = request.form.get('task_id')
    title = request.form.get('title', '').strip()
    description = request.form.get('description', '').strip()
    priority = request.form.get('priority', 'Medium').strip().capitalize()
    due_date = request.form.get('due_date', '').strip()
    is_completed = request.form.get('is_completed') == 'on'

    if not title:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'status': 'error', 'message': 'Task title cannot be empty.'}), 400
        flash('Task title cannot be empty.', 'warning')
        return redirect(url_for('tasks'))

    if priority not in ['Low', 'Medium', 'High']:
        priority = 'Medium'

    due_date_obj = None
    if due_date:
        try:
            due_date_obj = datetime.strptime(due_date, '%Y-%m-%d').date()
        except ValueError:
            due_date_obj = None

    task = None
    if task_id:
        task = Task.query.filter_by(id=int(task_id), user_id=current_user.id).first()
        if task:
            task.title = title
            task.description = description
            task.priority = priority
            task.due_date = due_date_obj
            task.is_completed = is_completed
            task.updated_at = datetime.utcnow()
        else:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'status': 'error', 'message': 'Task not found.'}), 404
            flash('Task not found.', 'warning')
            return redirect(url_for('tasks'))
    else:
        task = Task(
            user_id=current_user.id,
            title=title,
            description=description,
            priority=priority,
            due_date=due_date_obj,
            is_completed=is_completed
        )
        db.session.add(task)

    db.session.commit()
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'status': 'success', 'task_id': task.id, 'message': 'Task saved.'})

    flash('Task saved successfully.', 'success')
    return redirect(url_for('tasks'))


@app.route('/tasks/delete/<int:task_id>', methods=['POST', 'GET'])
@login_required
def delete_task(task_id):
    task = Task.query.filter_by(id=task_id, user_id=current_user.id).first()
    if not task:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'status': 'error', 'message': 'Task not found.'}), 404
        flash('Task not found.', 'warning')
        return redirect(url_for('tasks'))

    db.session.delete(task)
    db.session.commit()
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'status': 'success', 'message': 'Task deleted.'})
    flash('Task deleted successfully.', 'success')
    return redirect(url_for('tasks'))


@app.route('/tasks/toggle/<int:task_id>', methods=['POST'])
@login_required
def toggle_task_completion(task_id):
    task = Task.query.filter_by(id=task_id, user_id=current_user.id).first()
    if not task:
        return jsonify({'status': 'error', 'message': 'Task not found.'}), 404

    task.is_completed = not task.is_completed
    task.updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify({'status': 'success', 'completed': task.is_completed, 'message': 'Task updated.'})


@app.route('/datasets')
@login_required
def datasets():

    datasets = current_user.datasets.order_by(Dataset.uploaded_at.desc()).all()
    return render_template('datasets.html', datasets=datasets)


@app.route('/datasets/delete/<int:dataset_id>', methods=['POST', 'GET'])
@login_required
def delete_dataset(dataset_id):
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first()
    if not dataset:
        flash('Dataset not found.', 'warning')
        return redirect(url_for('datasets'))

    if os.path.exists(dataset.upload_path):
        os.remove(dataset.upload_path)

    usage = get_or_create_usage(current_user)
    usage.storage_bytes = max(0, usage.storage_bytes - (dataset.size_bytes or 0))
    db.session.delete(dataset)
    db.session.commit()

    flash('Dataset removed from your library.', 'success')
    return redirect(url_for('datasets'))


@app.route('/usage')
@login_required
def usage_dashboard():
    usage = get_or_create_usage(current_user)
    analyses = current_user.analyses.order_by(Analysis.completed_at.desc()).limit(6).all()
    totals = {
        'total_analyses': current_user.total_analyses,
        'total_uploads': current_user.datasets_uploaded,
        'storage_used_mb': round((usage.storage_bytes or 0) / 1024 / 1024, 2),
        'last_analysis': usage.last_analysis_at.strftime('%b %d, %Y %H:%M') if usage.last_analysis_at else 'N/A'
    }
    return render_template('usage.html', totals=totals, analyses=analyses)


@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')


@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        new_password = request.form.get('new_password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not name or not email:
            flash('Name and email are required.', 'warning')
            return redirect(url_for('settings'))

        if email != current_user.email and User.query.filter_by(email=email).first():
            flash('That email is already in use.', 'warning')
            return redirect(url_for('settings'))

        current_user.name = name
        current_user.email = email

        if new_password:
            if new_password != confirm_password:
                flash('New passwords do not match.', 'warning')
                return redirect(url_for('settings'))
            current_user.set_password(new_password)

        db.session.commit()
        flash('Settings saved successfully.', 'success')
        return redirect(url_for('profile'))

    return render_template('settings.html')

with app.app_context():
    db.create_all()
if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=True, reloader_options={"exclude_patterns": [".venv*"]})
