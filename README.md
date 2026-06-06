# FeynML: ML Failure Investigation Engine

Understanding Why Models Fail

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-2.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**FeynML** is a professional-grade ML observability and diagnostic platform designed to automate the investigation of model failures. It provides a multi-phase audit of datasets and models, covering everything from calibration and fairness to distribution drift and root-cause data integrity issues.

## 🚀 Key Features

*   **Automated Diagnostic Pipeline**: Executes a 4-phase audit (Diagnostics, Observability, Root Cause, Integrity).
*   **Feature Drift Detection**: Statistical monitoring using KS-tests and PSI to identify data distribution shifts.
*   **Model Calibration Audit**: Visualizes reliability diagrams and calculates Expected Calibration Error (ECE).
*   **Label Noise Analysis**: Identifies potential mislabeling in training data using automated integrity checks.
*   **Leakage & Missingness Detection**: Flags target leakage and analyzes missing data mechanisms (MCAR/MAR/MNAR).
*   **Professional Analytics Dashboard**: Interactive Plotly visualizations and high-density KPI monitoring.
*   **Multi-Format Export**: Export findings as raw JSON, summary CSV, or professional print-ready PDF reports.

## 🏗️ Architecture

The project follows a modular "Engine" architecture, separating statistical logic from the web presentation layer:

*   **`engine/modules/`**: Core diagnostic engines (Drift, Calibration, Fairness, etc.).
*   **`webapp/`**: Flask-based application layer with Plotly integration.
*   **`scratch/`**: Pure statistical implementation of underlying ML tests.
*   **`services/`**: Orchestration layer for running parallelized analysis.

## 📁 Repository Structure

```text
.
├── engine/                 # Core ML Diagnostic Logic
│   ├── modules/            # Specialized Engines (Drift, Leakage, etc.)
│   └── base_module.py      # Abstract Base for all Engines
├── webapp/                 # Flask Presentation Layer
│   ├── static/             # "FeynML" CSS & Design System
│   ├── templates/          # Responsive HTML (Dashboard, Reports)
│   ├── services/           # Analysis Runner & Orchestration
│   └── app.py              # Application Entry Point & Export Routes
├── tests/                  # Unit tests for Engine reliability
└── requirements.txt        # Project Dependencies
```

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/feynml-ml-engine.git
    cd feynml-ml-engine
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**:
    ```bash
    cd webapp
    python app.py
    ```

4.  **Access the Dashboard**: Open `http://localhost:5000` in your browser.

## 📊 Example Workflow

1.  **Upload**: Provide a CSV/Parquet dataset and model predictions.
2.  **Map**: Define Target, Predictions, and Sensitive attributes via the UI.
3.  **Analyze**: FeynML executes parallelized audits across all four phases.
4.  **Investigate**: Use the interactive dashboard to identify the "Weakest Link" in your ML pipeline.
5.  **Export**: Generate a PDF report for stakeholder review or a CSV summary for further automated processing.

## 💻 Technologies Used

*   **Backend**: Python, Flask, Pandas, Scikit-learn, SciPy
*   **Frontend**: HTML5, "FeynML" CSS (Custom Design System), Bootstrap 5
*   **Visualization**: Plotly.js
*   **Export**: xhtml2pdf, CSV/JSON Serializers

---
*Developed as a high-fidelity ML Observability solution for data-driven teams.*
