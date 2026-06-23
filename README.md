# FeynML: ML Failure Investigation Engine

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-2.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-production--ready-success.svg)

**Understanding Why Models Fail**

FeynML is a production-grade ML observability and diagnostic platform that automates the investigation of model failures. It provides a comprehensive multi-phase audit of datasets and models, covering calibration, fairness, distribution drift, and root-cause data integrity issues with professional-grade reporting.

---

## 🎯 Problem Statement

Machine learning models in production inevitably degrade over time due to data drift, concept drift, and changing real-world conditions. Traditional monitoring tools alert you **that** something is wrong, but fail to explain **why**. FeynML bridges this gap by:

- **Automating Root Cause Analysis**: Identifying the specific features and patterns causing model degradation
- **Providing Actionable Insights**: Delivering synthesized narratives that explain issues in business context
- **Supporting Multiple Stakeholders**: Translating technical findings for executives, doctors, loan officers, and other domain experts
- **Ensuring Data Integrity**: Detecting leakage, missing data patterns, and label noise before they impact production

---

## ✨ Key Features

### 🔍 Automated Diagnostic Pipeline
- **4-Phase Audit Architecture**: Diagnostics → Observability → Root Cause → Integrity
- **Parallelized Analysis**: Executes multiple diagnostic engines simultaneously
- **Intelligent Scoring**: Ranks issues by severity and confidence

### 📊 Feature Drift Detection
- **Statistical Monitoring**: KS-tests and Population Stability Index (PSI)
- **Domain Classifier Drift**: Detects distributional shifts using adversarial classifiers
- **Automatic Identifier Exclusion**: Prevents ID columns from being treated as predictive features

### 🎯 Model Calibration Audit
- **Reliability Diagrams**: Visualizes probability calibration
- **Expected Calibration Error (ECE)**: Quantifies calibration quality
- **Brier Score**: Measures overall probability accuracy

### 🔬 Data Integrity Analysis
- **Leakage Detection**: Identifies features that leak target information
- **Missing Data Analysis**: Classifies missingness patterns (MCAR/MAR/MNAR)
- **Label Noise Detection**: Flags potential mislabeling in training data

### 🎨 Professional Analytics Dashboard
- **Interactive Visualizations**: Plotly-powered charts and graphs
- **High-Density KPI Monitoring**: Real-time health status tracking
- **Responsive Design**: Works across desktop and mobile devices

### 📝 Multi-Format Export
- **JSON Export**: Raw data for automated processing
- **CSV Summary**: Spreadsheet-friendly format
- **PDF Reports**: Professional print-ready documentation

### 🌐 Audience-Specific Reporting
- **ML Engineer**: Technical details with statistical evidence
- **Executive**: Business impact and risk assessment
- **Doctor**: Clinical implications and patient safety
- **Loan Officer**: Credit risk and regulatory compliance
- **Student**: Learning outcomes and academic performance

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Web Application Layer                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Dashboard  │  │   Reports    │  │    API       │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Orchestration Layer                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Analysis Runner (Parallel Execution)         │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Engine Modules                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │   Drift  │ │Calibration│ │ Leakage  │ │ Fairness │          │
│  │  Engine  │ │  Engine  │ │  Engine  │ │  Engine  │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │Root Cause│ │   Slice  │ │  Data    │ │ Explain  │          │
│  │  Engine  │ │  Engine  │ │ Quality  │ │ability  │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Investigation Layer                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              AI Investigator (Narrative Synthesis)          │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Audience Translator (Domain Adaptation)       │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```text
.
├── engine/                      # Core ML Diagnostic Logic
│   ├── modules/                 # Specialized Engines
│   │   ├── drift_engine.py      # Distribution drift detection
│   │   ├── calibration_engine.py # Model calibration analysis
│   │   ├── leakage_engine.py    # Target leakage detection
│   │   ├── root_cause_engine.py # Root cause identification
│   │   ├── ai_investigator.py   # Narrative synthesis
│   │   └── audience_translator.py # Domain-specific translation
│   ├── base_module.py           # Abstract base for all engines
│   └── investigation.py         # Investigation data structures
├── webapp/                      # Flask Presentation Layer
│   ├── static/                  # CSS, JavaScript, assets
│   ├── templates/               # HTML templates
│   │   ├── dashboard.html       # Main dashboard
│   │   └── report.html          # Investigation reports
│   ├── services/                # Orchestration layer
│   │   └── analysis_runner.py   # Parallel analysis execution
│   └── app.py                   # Application entry point
├── tests/                       # Unit and integration tests
├── scratch/                     # Pure statistical implementations
├── requirements.txt             # Project dependencies
└── README.md                    # This file
```

---

## 🛠️ Tech Stack

### Backend
- **Python 3.8+**: Core language
- **Flask 2.0+**: Web framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **SciPy**: Statistical tests
- **Scikit-learn**: Machine learning utilities

### Frontend
- **HTML5**: Markup
- **Bootstrap 5**: CSS framework
- **Plotly.js**: Interactive visualizations
- **Custom CSS**: FeynML design system

### Export & Reporting
- **xhtml2pdf**: PDF generation
- **JSON/CSV**: Data serialization

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/feynml-ml-engine.git
cd feynml-ml-engine
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Run the application**:
```bash
cd webapp
python app.py
```

5. **Access the Dashboard**: Open `http://localhost:5000` in your browser

---

## 📖 Usage

### Basic Workflow

1. **Upload Dataset**: Provide a CSV or Parquet file with your data
2. **Map Attributes**: Define target variable, predictions, and sensitive attributes
3. **Select Analysis**: Choose which diagnostic phases to run
4. **Review Results**: Explore the interactive dashboard
5. **Generate Report**: Export findings in your preferred format

### Python API Usage

```python
from engine.modules.drift_engine import DriftEngine
from engine.modules.root_cause_engine import AutoRootCauseEngine
from engine.modules.ai_investigator import AIInvestigator

# Initialize engines
drift_engine = DriftEngine()
root_cause_engine = AutoRootCauseEngine()
ai_investigator = AIInvestigator()

# Set reference data
X_train = [[85.5, 10.2, 90], [90.0, 12.5, 95], ...]
feature_names = ['Attendance', 'StudyTime', 'Grade']
drift_engine.set_reference(X_train, feature_names)

# Analyze production data
X_prod = [[75.0, 9.0, 80], [70.0, 7.5, 78], ...]
drift_result = drift_engine.run(X_prod)

# Identify root causes
root_cause_result = root_cause_engine.run(drift_report=drift_result)

# Generate narrative investigation
investigation = Investigation.from_dict(root_cause_result)
ai_result = ai_investigator.analyze(investigation)

print(f"Executive Summary: {ai_result['executive_summary']}")
print(f"Investigation Findings: {ai_result['investigation_findings']}")
```

---

## 📊 Example Investigation Report

### Executive Summary
```
Critical model degradation detected. Primary cause: distributional drift affecting prediction reliability. Immediate retraining recommended. Confidence: 90%.
```

### Investigation Findings
```
Widespread distributional drift detected across 4 features, indicating significant changes in the underlying data distribution that may require comprehensive model retraining. The presence of 4 high-severity issue(s) indicates these problems require immediate attention to prevent further model degradation.
```

### Root Cause Analysis
| Feature | Score | Severity | Evidence | Category |
|---------|-------|----------|----------|----------|
| Attendance | 100 | CRITICAL | PSI=0.826, KS=0.356 | feature_drift |
| StudyTime | 75 | HIGH | PSI=0.481, KS=0.290 | feature_drift |
| Grade | 75 | HIGH | PSI=0.355, KS=0.234 | feature_drift |
| MidtermScore | 75 | HIGH | PSI=0.369, KS=0.238 | feature_drift |

### Audit Log
```
[Feature Excluded] StudentID (Identifier Column)
```

### Recommended Actions
1. Collect recent samples from affected segments before retraining
2. Investigate upstream data pipeline for distributional changes
3. Consider targeted data collection for drifted features
4. Monitor model performance post-retraining

---

## 🎨 Screenshots

*(Note: Screenshots would be added here showing the dashboard interface, analysis results, and report generation)*

### Dashboard View
![Dashboard](screenshots/dashboard.png)

### Drift Analysis
![Drift Analysis](screenshots/drift_analysis.png)

### Investigation Report
![Investigation Report](screenshots/investigation_report.png)

---

## 🔮 Future Roadmap

### Short Term
- [ ] Real-time streaming data support
- [ ] Alerting and notification system
- [ ] Model comparison and A/B testing
- [ ] Enhanced mobile experience

### Medium Term
- [ ] Multi-model orchestration
- [ ] Automated retraining pipelines
- [ ] Integration with MLOps platforms
- [ ] Advanced explainability techniques

### Long Term
- [ ] Federated learning support
- [ ] AutoML integration
- [ ] Cloud-native deployment
- [ ] Enterprise SSO and RBAC

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

Developed as a high-fidelity ML Observability solution for data-driven teams.

---

## 🙏 Acknowledgments

- Statistical methods inspired by industry best practices in ML monitoring
- Visualization powered by Plotly.js
- Web framework built on Flask

---

*Built with ❤️ for production ML systems*
# force redeploy Tue Jun 23 12:06:51 IST 2026
