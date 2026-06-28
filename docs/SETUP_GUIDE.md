# FeynML - Complete Setup Guide

**Understanding Why Models Fail**

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended for analysis)
- **Disk**: 2GB free space
- **OS**: Windows, macOS, or Linux
- **Browser**: Chrome, Firefox, Safari, or Edge

---

## 1. FOLDER STRUCTURE

```
ml_failure_engine_reorganized/
│
├── webapp/                          # Flask web application
│   ├── __init__.py
│   ├── app.py                       # Main Flask app with SaaS routes
│   ├── extensions.py                # SQLAlchemy, Flask-Login, Bcrypt initialization
│   ├── models.py                    # Database models (User, Analysis, Report, Dataset, UsageStat)
│   │
│   ├── templates/                   # HTML templates
│   │   ├── base.html               # Base layout with navbar and footer
│   │   ├── landing.html            # Marketing/landing page
│   │   ├── login.html              # Authentication
│   │   ├── signup.html             # User registration
│   │   ├── profile.html            # User profile page
│   │   ├── index.html              # Dataset upload page
│   │   ├── configure.html          # Schema mapping
│   │   ├── analysis_running.html   # Analysis progress
│   │   ├── dashboard.html          # Interactive report dashboard
│   │   ├── report.html             # Full investigation report
│   │   ├── saved_reports.html      # Report history/library
│   │   ├── datasets.html           # Dataset library
│   │   ├── usage.html              # Usage analytics
│   │   ├── settings.html           # Account settings
│   │   ├── features.html           # Features page
│   │   ├── about.html              # About page
│   │   ├── faq.html                # FAQ page
│   │   ├── contact.html            # Contact page
│   │   └── admin.html              # Admin console
│   │
│   ├── static/                      # Static assets
│   │   └── css/
│   │       └── style.css           # Global glassmorphism theme
│   │
│   ├── services/
│   │   └── analysis_runner.py      # Analysis engine runner
│   │
│   ├── uploads/                     # Temporary dataset uploads (created at runtime)
│   ├── reports/                     # Generated JSON reports (created at runtime)
│   └── feynml.db                  # SQLite database (created on first run)
│
├── engine/                          # ML diagnostic engines
│   ├── __init__.py
│   ├── base_module.py              # Base diagnostic module
│   ├── failure_report_generator.py # Report generation
│   │
│   └── modules/                     # Individual diagnostic engines
│       ├── calibration_engine.py    # Model calibration audit
│       ├── causal_engine.py         # Causal inference
│       ├── drift_engine.py          # Feature drift detection
│       ├── evaluator.py             # Model evaluation
│       ├── explainability_engine.py # SHAP/LIME explanations
│       ├── fairness_engine.py       # Fairness audit
│       ├── label_noise_engine.py    # Label quality
│       ├── leakage_engine.py        # Data leakage detection
│       ├── missing_data_engine.py   # Missingness analysis
│       ├── report_engine.py         # Report synthesis
│       ├── slicer_engine.py         # Data slicing
│       └── validator.py             # Data validation
│
├── scratch/                         # Development/scratch work (can be ignored)
│   ├── phase0/
│   ├── phase1/
│   ├── phase2/
│   ├── phase3/
│   └── phase4/
│
├── tests/                           # Unit tests
│   ├── test_calibration.py
│   ├── test_causal_inference.py
│   ├── test_drift_engine.py
│   ├── test_fairness.py
│   ├── test_feature_leakage.py
│   ├── test_label_noise.py
│   ├── test_missing_data.py
│   └── ... (more tests)
│
├── docs/                            # Documentation
│   ├── phase4_architecture.md
│   └── ... (other docs)
│
├── reports/                         # Sample/archived reports (for reference)
│   └── *.json (report files)
│
├── migrations/                      # Flask-Alembic database migrations
│   ├── README.md
│   └── versions/
│
├── requirements.txt                 # Python dependencies
├── manage.py                        # Database migration manager
├── run_phase0.py                   # Legacy CLI scripts
├── run_phase2.py
│
├── README.md                        # Project readme
├── PROJECT_AUDIT.md                # Architecture documentation
├── PORTFOLIO_GUIDE.md              # Portfolio documentation
├── LEAKAGE_VALIDATION.md           # Validation notes
└── .gitignore                       # Git ignore file
```

---

## 2. INSTALLATION STEPS (Windows PowerShell / CMD)

### Step 1: Navigate to Project Directory

```powershell
cd "C:\Users\arman\OneDrive\Desktop\DATA ANALYTICS\ML_investigator\ml_failure_engine_reorganized"
```

### Step 2: Create Python Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate it (PowerShell)
.\venv\Scripts\Activate.ps1

# OR if using CMD.exe
venv\Scripts\activate.bat
```

**Output**: You should see `(venv)` at the start of your terminal line.

### Step 3: Upgrade pip, setuptools, wheel

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### Step 4: Install Dependencies

```powershell
pip install -r requirements.txt
```

**Expected time**: 3-5 minutes depending on internet speed.

### Step 5: Initialize Database (First Time Only)

```powershell
# Create SQLite database with schema
python manage.py db upgrade
```

If you get an error about migrations not existing, run:
```powershell
python manage.py db init
python manage.py db migrate -m "Initial schema"
python manage.py db upgrade
```

This creates `feynml.db` in your project root with all tables (users, analyses, reports, datasets, usage_stats).

---

## 3. RUNNING THE APPLICATION

### Option A: Run with Built-in Flask Server (Development)

```powershell
# From the project root
python manage.py

# OR directly
python -c "from webapp.app import app; app.run(debug=True, port=5000)"
```

**Output**:
```
 * Serving Flask app 'webapp.app'
 * Debug mode: on
 * WARNING: This is a development server. Do not use it in production.
 * Running on http://127.0.0.1:5000
```

### Option B: Run with Gunicorn (Production-like)

```powershell
# Install gunicorn
pip install gunicorn

# Run
gunicorn -w 4 -b 127.0.0.1:5000 webapp.app:app
```

### Option C: Run in Another PowerShell Tab (Parallel)

Keep the first terminal running the server, open a new PowerShell tab and:

```powershell
cd "C:\Users\arman\OneDrive\Desktop\DATA ANALYTICS\ML_investigator\ml_failure_engine_reorganized"
.\venv\Scripts\Activate.ps1
# Server is already running in the other tab
```

---

## 4. ACCESSING THE APPLICATION IN CHROME

### Step 1: Open Chrome

Click the Chrome icon on your desktop or press `Win + R` and type:
```
chrome
```

### Step 2: Navigate to Localhost

In the address bar, type:
```
http://localhost:5000
```

OR

```
http://127.0.0.1:5000
```

### Step 3: You Should See the Landing Page

You'll land on the **FeynML** marketing/landing page with:
- Hero section
- Features overview
- Platform capabilities
- Roadmap
- Footer with recruiter links

---

## 5. FIRST-TIME USER WORKFLOW

### Create an Account

1. Click **"Sign Up"** button in the top-right navbar
2. Enter:
   - **Name**: Your full name
   - **Email**: Your email address
   - **Password**: A secure password
3. Click **"Create Account"**
4. You're automatically logged in and redirected to the upload page

### Upload & Analyze a Dataset

1. Click **"Get Started"** or go to http://localhost:5000/analyze
2. **Upload CSV or JSON file** (sample datasets provided in `/reports`)
3. **Configure schema**:
   - Select **Target Column** (actual outcome)
   - Select **Prediction Column** (optional, or auto-generate)
   - Select **Sensitive Attribute** (for fairness audit, optional)
   - Select **Timestamp** (for drift analysis, optional)
4. Click **"Execute Analysis"**
5. Wait for analysis to complete (1-3 minutes depending on dataset size)
6. View interactive dashboard with:
   - Calibration charts
   - Feature drift analysis
   - Label integrity
   - Missing data patterns
   - Fairness metrics

### Download Reports

- **Dashboard**: Interactive view (web only)
- **Full Report**: Comprehensive HTML report with tables and recommendations
- **PDF**: Printable investigation report
- **JSON**: Raw data export
- **CSV**: Summary metrics table

### Manage Your Workspace

- **Saved Reports**: View all past analyses
- **Dataset Library**: Track uploaded datasets
- **Usage Analytics**: See total analyses, uploads, storage used
- **Profile**: View account info and statistics
- **Settings**: Update email/password

---

## 6. DETAILED COMMAND REFERENCE

### Database Commands

```powershell
# Initialize migrations folder (one-time)
python manage.py db init

# Create a new migration after schema changes
python manage.py db migrate -m "Description of change"

# Apply migrations to database
python manage.py db upgrade

# View migration status
python manage.py db current
```

### Running Tests

```powershell
# Install pytest
pip install pytest

# Run all tests
pytest tests/

# Run specific test
pytest tests/test_calibration.py

# Run with verbose output
pytest tests/ -v
```

### Running Analysis from CLI

```powershell
# Import and run directly
python -c "
from engine.modules.drift_engine import DriftEngine
from engine.base_module import BaseModule

# Example usage (see test files for detailed examples)
"
```

### Useful Python Commands

```powershell
# Check installed packages
pip list

# Freeze current environment
pip freeze > requirements-exact.txt

# Interactive Python shell
python

# Within Python:
>>> from webapp.models import User, Analysis, Report
>>> from webapp.app import db, app
>>> with app.app_context():
...     users = User.query.all()
...     print(f"Total users: {len(users)}")
```

---

## 7. IMPORTANT FILE LOCATIONS & THEIR PURPOSES

| File/Folder | Purpose | Notes |
|---|---|---|
| `webapp/app.py` | Main Flask application with all routes | ~880 lines |
| `webapp/models.py` | Database schema (User, Analysis, Report, etc.) | SQLAlchemy ORM |
| `engine/modules/*.py` | Individual diagnostic engines | 6 core modules |
| `webapp/templates/` | HTML templates (Jinja2) | 18+ templates |
| `webapp/static/css/style.css` | Global glassmorphism theme | Dark/light mode support |
| `feynml.db` | SQLite database (auto-created) | Don't commit to git |
| `uploads/` | Temporary dataset storage (auto-created) | Cleaned up after analysis |
| `reports/` | Generated JSON reports (auto-created) | Persistent (backed by DB) |
| `requirements.txt` | Python dependencies | Flask, SQLAlchemy, Plotly, etc. |
| `manage.py` | Database migration manager | Uses Flask-Migrate |

---

## 8. COMMON TASKS & COMMANDS

### Start Fresh

```powershell
# Deactivate current environment
deactivate

# Remove old database
Remove-Item feynml.db -ErrorAction SilentlyContinue

# Remove venv and recreate
Remove-Item venv -Recurse -ErrorAction SilentlyContinue
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python manage.py db upgrade

# Start server
python manage.py
```

### View Database Contents

```powershell
# Install sqlite3 CLI
# (Usually comes with Python)

# Then:
sqlite3 feynml.db

# In SQLite shell:
.tables
.schema users
SELECT * FROM users;
SELECT * FROM analyses;
.quit
```

### Check Logs

```powershell
# Flask logs print to console where you started the server
# Look for:
# - "INFO: ..."
# - "WARNING: ..."
# - "ERROR: ..."

# To save logs to file:
python manage.py 2>&1 | Tee-Object -FilePath app.log
```

### Troubleshooting

```powershell
# Check if port 5000 is in use
netstat -ano | findstr :5000

# Kill process on port 5000 (if needed)
Stop-Process -Id <PID> -Force

# Check Python version
python --version

# Check pip version
pip --version

# Test imports
python -c "from webapp.app import app; print('App imported successfully')"
```

---

## 9. FOLDER STRUCTURE EXPLANATION

### `/webapp`
Contains the entire Flask web application, including routes, templates, and static assets. This is where the UI/SaaS layer lives.

### `/engine`
The ML diagnostic logic. Each module (calibration, drift, fairness, etc.) runs independently and can be used as CLI tools or integrated into the Flask app.

### `/scratch`
Legacy development code organized by "phase" (phase0-4). Can be ignored for normal usage; useful for reference.

### `/tests`
Unit tests for engines and utilities. Run with `pytest`.

### `/migrations`
Database schema version control. Don't edit manually—use `flask db` commands.

### `/docs`
Architecture documentation and design notes.

---

## 10. CHROME BOOKMARKS (FOR QUICK ACCESS)

After logging in, bookmark these URLs:

```
http://localhost:5000/                      # Landing page
http://localhost:5000/analyze               # Upload new dataset
http://localhost:5000/saved-reports         # View all reports
http://localhost:5000/datasets              # Dataset library
http://localhost:5000/usage                 # Usage dashboard
http://localhost:5000/profile               # Your profile
http://localhost:5000/settings              # Account settings
```

---

## 11. QUICK REFERENCE CHECKLIST

- [ ] Python 3.8+ installed
- [ ] Navigate to project directory
- [ ] Create virtual environment: `python -m venv venv`
- [ ] Activate venv: `.\venv\Scripts\Activate.ps1`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Initialize database: `python manage.py db upgrade`
- [ ] Start server: `python manage.py`
- [ ] Open Chrome: `http://localhost:5000`
- [ ] Sign up for account
- [ ] Upload test dataset
- [ ] Run analysis
- [ ] View results

---

## 12. TROUBLESHOOTING COMMON ISSUES

### Issue: "ModuleNotFoundError: No module named 'webapp'"

**Solution**: Make sure you're running commands from the project root:
```powershell
cd "C:\Users\arman\OneDrive\Desktop\DATA ANALYTICS\ML_investigator\ml_failure_engine_reorganized"
```

### Issue: "Port 5000 already in use"

**Solution**: 
```powershell
# Find process using port 5000
netstat -ano | findstr :5000

# Kill it
taskkill /PID <PID> /F

# Or use a different port:
python -c "from webapp.app import app; app.run(port=5001)"
```

### Issue: Database errors on startup

**Solution**:
```powershell
# Backup old database
Copy-Item feynml.db feynml.db.backup

# Reinitialize
Remove-Item feynml.db
python manage.py db upgrade
```

### Issue: Virtual environment won't activate

**Solution**:
```powershell
# If Activate.ps1 fails, try:
python -m venv venv --upgrade-deps

# Or use CMD.exe instead:
venv\Scripts\activate.bat
```

### Issue: "Permission denied" on Windows

**Solution**: Run PowerShell as Administrator or use CMD.exe

---

## 13. PERFORMANCE TIPS

- **Analysis time**: 1-3 minutes for datasets < 100k rows
- **Maximum dataset size**: 500k rows (tested)
- **Memory usage**: ~500MB-2GB during analysis
- **Disk usage**: Reports are JSON (~1-5 MB each)

---

## 14. NEXT STEPS AFTER SETUP

1. **Explore the UI**: Upload a test CSV, configure, and run an analysis
2. **Customize theme**: Edit `webapp/static/css/style.css`
3. **Add custom columns**: Edit `webapp/models.py` and create migrations
4. **Deploy to cloud**: Use Heroku, AWS, or DigitalOcean (requires environment variables)
5. **Add HTTPS**: Use nginx or Apache reverse proxy

---

## 15. ENVIRONMENT VARIABLES (Optional)

Create a `.env` file in the project root:

```env
# Flask config
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here

# Database (optional, defaults to SQLite)
DATABASE_URL=sqlite:///feynml.db
# Or PostgreSQL:
# DATABASE_URL=postgresql://user:password@localhost/feynml

# Upload limits
MAX_CONTENT_LENGTH=52428800  # 50MB in bytes
```

Load in Python:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

Good luck! You're ready to run FeynML locally. Start with Step 1-5 above, then access it in Chrome.
