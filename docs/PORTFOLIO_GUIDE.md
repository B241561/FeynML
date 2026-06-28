# Portfolio & Career Guide: ML Failure Engine

This guide provides professional descriptions and talking points for showcasing the FeynML ML Failure Engine in your portfolio, resume, and interviews.

## 📄 Resume Descriptions

### 2-Line Version
Developed "FeynML," a professional ML observability platform using Flask and Plotly that automates model failure investigation. Implemented a 4-phase diagnostic pipeline to detect feature drift, miscalibration, and data integrity issues.

### 5-Line Version
Engineered "FeynML," a comprehensive ML diagnostic engine designed to identify the root causes of model performance degradation. Developed modular Python engines for statistical drift detection (KS-tests), Expected Calibration Error (ECE) analysis, and label noise identification. Built a responsive web dashboard with interactive Plotly visualizations for real-time failure investigation. Implemented a robust export layer generating multi-format reports (PDF, CSV, JSON) for stakeholder communication.

### ATS-Friendly Version (Skills-Focused)
**ML Observability Platform (FeynML)** | *Python, Flask, Plotly, Pandas, Scikit-learn, SciPy*
*   Designed a modular architecture to automate ML model audits, reducing manual failure investigation time by implementing statistical tests for Feature Drift (KS-test, PSI).
*   Built interactive reliability diagrams and calibration curves to diagnose model uncertainty and Expected Calibration Error (ECE).
*   Developed data integrity modules to detect target leakage and label noise using automated statistical heuristics.
*   Architected a responsive frontend with a custom CSS design system and integrated multi-format report export capabilities (PDF, CSV, JSON).

---

## 💬 Interview Talking Points

### Technical Challenges Solved
*   **Challenge**: Normalizing disparate engine outputs into a unified "Global Risk" score.
*   **Solution**: Implemented a standardized severity mapping layer (NONE -> CRITICAL) across modular diagnostic engines, ensuring consistent risk aggregation in the presentation layer.
*   **Challenge**: Ensuring backward compatibility for persistent report data after schema changes.
*   **Solution**: Developed a defensive rendering strategy in Jinja2 templates that supports multiple key variations (`n_drifted` vs `num_drifted_features`), preventing UI breaks on legacy data.

### Architecture Decisions
*   **Decoupled Logic**: I chose to separate the "Engine" (statistical logic) from the "Webapp" (presentation). This allows the diagnostic engines to be used as a CLI tool or integrated into a CI/CD pipeline independently of the Flask UI.
*   **Weakest Link Strategy**: The Global Risk calculation uses a "Max Severity" approach. This ensures that even if most metrics are healthy, a single critical failure (like high feature drift) is surfaced immediately to the user.

### Debugging Lessons
*   **Data Consistency**: During development, I discovered that UI metrics were reporting 0 drift while the data tables showed significant KS values. This taught me the importance of strict schema validation between backend JSON outputs and frontend template consumption, leading to a more robust export/import layer.

---

## 📦 GitHub Release Notes: v1.0.0 (The "FeynML" Update)

### 🚀 New Features
*   **Phase 1-4 Audit**: Full implementation of Diagnostics, Observability, Root Cause, and Data Integrity phases.
*   **Visual Analytics**: Interactive Plotly charts for Calibration, Drift, and Label Noise.
*   **Professional Export**: Added one-click export for PDF Reports, Summary CSVs, and Raw JSON data.
*   **FeynML UI**: Completely redesigned user interface with a high-density dashboard and custom "Observability" design system.

### 🔧 Enhancements & Fixes
*   Improved Drift detection logic using KS-test and PSI with industry-standard thresholds.
*   Normalized Calibration severity mapping for consistent global risk calculation.
*   Enhanced backward compatibility for older report schemas.
*   Added print-optimized styling for professional PDF generation.

### 📦 Dependencies
*   Flask, Plotly, Pandas, Scikit-learn, SciPy, xhtml2pdf.
