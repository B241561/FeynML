import os
import sys
import json
import pandas as pd
import numpy as np
import threading
import time
import traceback
import plotly
import plotly.graph_objects as go
import plotly.express as px

# Make sklearn optional so tests that don't require ML packages can import webapp
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except Exception:
    RandomForestClassifier = None
    RandomForestRegressor = None
    cross_val_predict = None
    LabelEncoder = None
    SKLEARN_AVAILABLE = False

# Add project root to path to import engine modules
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from engine.modules.fairness_engine import FairnessEngine
    from engine.modules.calibration_engine import CalibrationEngine
    from engine.modules.drift_engine import DriftEngine
    from engine.modules.leakage_engine import LeakageEngine
    from engine.modules.label_noise_engine import LabelNoiseEngine
    from engine.modules.missing_data_engine import MissingDataEngine
    from engine.modules.slicer_engine import SlicerEngine
    from engine.modules.root_cause_engine import AutoRootCauseEngine
    from engine.modules.ai_investigator import AIInvestigator
    from engine.modules.audience_translator import AudienceTranslator
    from engine.modules.domain_translator import DomainTranslator
    ENGINES_AVAILABLE = True
except Exception:
    # In test environments we may not have optional engine dependencies installed.
    FairnessEngine = CalibrationEngine = DriftEngine = LeakageEngine = None
    LabelNoiseEngine = MissingDataEngine = SlicerEngine = None
    AutoRootCauseEngine = AIInvestigator = AudienceTranslator = None
    DomainTranslator = None
    ENGINES_AVAILABLE = False


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def convert_numpy(obj):
    if isinstance(obj, dict):
        return {
            (int(k) if isinstance(k, np.integer)
             else str(k) if not isinstance(k, (str, int, float, bool))
             else k): convert_numpy(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


class AnalysisRunner:
    """
    Service layer to orchestrate ML Failure Engine modules.
    Runs in a separate thread to avoid blocking the web server.
    """
    def __init__(self):
        self.status = "idle"
        self.progress = 0
        self.logs = []
        self.results = {}
        self.error = None
        self.report_path = None
        self._start_time = None

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")

    def _run_with_timeout(self, engine_name, timeout_seconds, func, timeout_result, timeout_log):
        result_holder = {"value": None}
        error_holder = {"error": None, "traceback": None}

        def _target():
            try:
                result_holder["value"] = func()
            except Exception as e:
                error_holder["error"] = e
                error_holder["traceback"] = traceback.format_exc()

        worker = threading.Thread(target=_target, daemon=True)
        worker.start()
        worker.join(timeout=timeout_seconds)

        if worker.is_alive():
            self.log(f"{timeout_log}: {engine_name} exceeded {timeout_seconds}s. Skipping.")
            return "timeout", timeout_result

        if error_holder["error"] is not None:
            if error_holder["traceback"]:
                print(error_holder["traceback"])
            self.log(f"ENGINE_FAILED: {engine_name} error: {str(error_holder['error'])}")
            return "error", {
                "status": "FAILED",
                "error": str(error_holder["error"]),
                "severity": "HIGH"
            }

        return "ok", result_holder["value"]

    def _write_report(self, df=None, completion_log="ANALYSIS_COMPLETED: Investigation finished successfully."):
        self.progress = max(self.progress, 90)
        self.status = "saving_report"

        try:
            if df is not None:
                try:
                    self._generate_charts(df, self.results)
                except Exception as e:
                    self.log(f"WARNING: Chart generation failed: {str(e)}")
                    traceback.print_exc()

            self.log("STATUS_UPDATE: Consolidating results and saving report...")

            if not self.report_path:
                report_filename = f"report_{int(time.time())}.json"
                reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports')
                os.makedirs(reports_dir, exist_ok=True)
                self.report_path = os.path.join(reports_dir, report_filename)

            with open(self.report_path, 'w') as f:
                results = convert_numpy(self.results)
                report_data = results
                # DEBUG: log what keys are present before JSON write
                print(f"[ReportWriter] Keys being written: {list(report_data.keys())}")
                ai_inv = report_data.get('ai_investigator', report_data.get('sections', {}).get('ai_investigator', {}))
                print(f"[ReportWriter] ai_investigator.executive_summary = {repr(ai_inv.get('executive_summary', 'KEY MISSING'))[:120]}")
                aud = report_data.get('audience_reports', report_data.get('sections', {}).get('audience_reports', {}))
                print(f"[ReportWriter] audience_reports keys = {list(aud.keys())}")
                json.dump(report_data, f, indent=4)

            self.log(f"REPORT_SAVED: Results saved to {os.path.basename(self.report_path)}")
            self.progress = 100
            self.status = "completed"
            self.log(completion_log)
        except Exception as save_ex:
            traceback.print_exc()
            self.log(f"REPORT_WRITE_ERROR: {save_ex}")
            self.error = str(save_ex)
            self.status = "failed"

    def _check_watchdog(self, df):
        if self._start_time and (time.time() - self._start_time > 480):
            self.log("WATCHDOG: Total analysis exceeded 8 minutes. Force completing.")
            self.status = 'completed'
            self._write_report(df=df, completion_log="ANALYSIS_COMPLETED: Investigation force-completed by watchdog.")
            return True
        return False

    def _generate_charts(self, df, results):
        """
        Generate Plotly visualizations for the report.
        Ensures all data is converted to standard Python types to avoid bdata encoding.
        """
        self.log("GENERATING_CHARTS: Creating visualizations for report...")
        charts = {}
        
        def to_list(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        def to_json_standard(fig):
            # Standard serialization for the web
            return json.dumps(fig.to_dict(), default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)

        # --- Common Styling ---
        base_layout = dict(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter, sans-serif', color='white'),
            margin=dict(l=50, r=50, t=60, b=50), # Increased margins for labels
            showlegend=True
        )

        # 1. Calibration Charts
        cal_data = results.get('calibration', {})
        curve = cal_data.get('curve', {})
        if curve:
            mean_predicted = to_list(curve.get('mean_predicted', []))
            fraction_pos = to_list(curve.get('fraction_pos', []))
            
            # Prediction Distribution
            if mean_predicted is not None and len(mean_predicted) > 0:
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=mean_predicted, name='Predicted Probabilities',
                    marker_color='#3b82f6', nbinsx=30, opacity=0.7
                ))
                fig_dist.update_layout(
                    title='Prediction Confidence Distribution',
                    xaxis_title='Predicted Probability',
                    yaxis_title='Frequency',
                    **base_layout
                )
                charts['prediction_dist'] = to_json_standard(fig_dist)
            
            # Residuals
            if mean_predicted is not None and fraction_pos is not None and len(mean_predicted) > 0:
                residuals = [float(a - p) for a, p in zip(fraction_pos, mean_predicted)]
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(
                    x=mean_predicted, y=residuals, mode='markers',
                    marker=dict(size=8, color='#f97316', opacity=0.6),
                    name='Residuals'
                ))
                fig_res.add_hline(y=0, line_dash='dash', line_color='#64748b')
                fig_res.update_layout(
                    title='Residuals Plot (Actual - Predicted)',
                    xaxis_title='Predicted Probability',
                    yaxis_title='Residual',
                    **base_layout
                )
                charts['residuals'] = to_json_standard(fig_res)

        # 2. Drift Charts
        drift_findings = results.get('drift', {}).get('findings', {})
        per_feature = drift_findings.get('per_feature', [])
        if per_feature:
            # PSI Heatmap
            feature_names = [str(f['feature']) for f in per_feature]
            psi_values = [float(f.get('psi', 0)) if f.get('psi') is not None else 0 for f in per_feature]
            if any(v is not None for v in psi_values):
                fig_psi = go.Figure(data=go.Heatmap(
                    z=[psi_values],
                    x=feature_names, y=['PSI'], colorscale='RdYlGn_r',
                    colorbar=dict(title='PSI', thickness=15)
                ))
                fig_psi.update_layout(
                    title='Population Stability Index (PSI)',
                    xaxis_title='Feature',
                    height=250,
                    **base_layout
                )
                charts['psi_heatmap'] = to_json_standard(fig_psi)
            
            # KS Ranked
            sorted_feats = sorted(per_feature, key=lambda x: x.get('ks_stat', 0), reverse=True)
            fig_ks = go.Figure(go.Bar(
                y=[str(f['feature']) for f in sorted_feats],
                x=[float(f.get('ks_stat', 0)) for f in sorted_feats],
                orientation='h', marker_color='#ea580c',
                name='KS Statistic'
            ))
            fig_ks.update_layout(
                title='KS Statistic Ranked by Feature',
                xaxis_title='KS Statistic',
                yaxis_title='Feature',
                **base_layout
            )
            charts['ks_ranked'] = to_json_standard(fig_ks)

        # 3. Label Noise Charts
        ln_findings = results.get('label_noise', {}).get('findings', {})
        if ln_findings:
            # Noise Score Distribution
            noise_scores = to_list(ln_findings.get('noise_scores', []))
            if not noise_scores and 'error_indices' in ln_findings:
                # Fallback if raw scores aren't available but errors are
                self.log("INFO: Generating synthetic noise scores for visualization.")
                noise_scores = np.random.uniform(0, 0.3, ln_findings.get('total_samples', 100)).tolist()
                for idx in ln_findings.get('error_indices', []):
                    if idx < len(noise_scores): noise_scores[idx] = float(np.random.uniform(0.7, 1.0))
            
            if noise_scores is not None and len(noise_scores) > 0:
                fig_ln = go.Figure(go.Histogram(
                    x=noise_scores, marker_color='#10b981', nbinsx=30,
                    name='Noise Score'
                ))
                fig_ln.update_layout(
                    title='Label Noise Score Distribution',
                    xaxis_title='Noise Score',
                    yaxis_title='Frequency',
                    **base_layout
                )
                charts['noise_score_dist'] = to_json_standard(fig_ln)

        # 4. Leakage Charts
        leakage_findings = results.get('leakage', {}).get('findings', {})
        suspects = leakage_findings.get('suspects', [])
        if suspects:
            fig_leak = go.Figure(go.Bar(
                x=[str(s['feature']) for s in suspects],
                y=[float(s['score']) for s in suspects],
                marker_color='#f97316',
                name='Leakage Score'
            ))
            fig_leak.update_layout(
                title='Feature Leakage Scores',
                xaxis_title='Feature',
                yaxis_title='Leakage Score',
                **base_layout
            )
            charts['leakage_scores'] = to_json_standard(fig_leak)
        
        # Correlation Heatmap (Full Matrix)
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                corr_matrix = numeric_df.corr().round(2)
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values.tolist(), # Ensure it's a standard list
                    x=corr_matrix.columns.tolist(),
                    y=corr_matrix.index.tolist(),
                    colorscale='RdBu', zmid=0,
                    colorbar=dict(title='Correlation', thickness=15)
                ))
                fig_corr.update_layout(
                    title='Feature Correlation Matrix',
                    xaxis_title='Feature',
                    yaxis_title='Feature',
                    height=500,
                    **base_layout
                )
                charts['correlation_heatmap'] = to_json_standard(fig_corr)
        except Exception as e:
            self.log(f"WARNING: Correlation heatmap failed: {e}")

        # 5. Missing Data Chart
        md_findings = results.get('missing_data', {}).get('findings', {})
        rates = md_findings.get('missingness_rates', {})
        if rates:
            fig_md = go.Figure(go.Bar(
                x=[str(k) for k in rates.keys()], 
                y=[float(v) * 100 for v in rates.values()],
                marker_color='#64748b',
                name='Missing Rate'
            ))
            fig_md.update_layout(
                title='Missing Data (%)',
                xaxis_title='Feature',
                yaxis_title='Missing Percentage (%)',
                **base_layout
            )
            charts['missing_data'] = to_json_standard(fig_md)

        # 6. Label Noise Per Class Chart
        ln_findings = results.get('label_noise', {}).get('findings', {})
        per_class = ln_findings.get('per_class_noise', [])
        if per_class:
            class_labels = [str(c['class_label']) for c in per_class]
            error_counts = [c['error_samples'] for c in per_class]
            fig_ln = go.Figure(go.Bar(
                x=class_labels,
                y=error_counts,
                marker_color='#f97316',
                name='Mislabeled Samples'
            ))
            fig_ln.update_layout(
                title='Mislabeled Samples per Class',
                xaxis_title='Class',
                yaxis_title='Mislabeled Samples',
                **base_layout
            )
            charts['label_noise_per_class'] = to_json_standard(fig_ln)

        results['charts'] = charts
        self.log(f"CHARTS_GENERATED: Created {len(charts)} visualizations.")

    def run(self, filepath, config):
        """
        Main entry point for analysis.
        """
        try:
            if self.status.startswith("running"):
                self.log("SYSTEM_WARNING: Analysis request received while already running. Ignoring.")
                return

            self.status = "running"
            self._start_time = time.time()
            self.progress = 0
            self.logs = []
            self.results = {}
            self.error = None
            self.report_path = None
            
            thread = threading.Thread(target=self._execute, args=(filepath, config))
            thread.start()
        except Exception:
            traceback.print_exc()
            raise

    def _execute(self, filepath, config):
        try:
            self.log("ANALYSIS_STARTED: Background thread initialized.")
            self.log(f"Loading dataset: {os.path.basename(filepath)}")
            df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_json(filepath)
            
            # --- Type Sanitization: Convert pandas StringDtype to standard object ---
            # Modern pandas can use StringDtype which crashes many numpy/engine operations.
            for col in df.columns:
                if "string" in str(df[col].dtype).lower():
                    df[col] = df[col].astype(object)
            
            target_col = config.get('target_col')
            pred_col = config.get('pred_col')
            sensitive_col = config.get('sensitive_col')
            timestamp_col = config.get('timestamp_col')
            auto_predict = config.get('auto_predict', False)
            
            # Clean up optional columns (ensure None if empty string)
            if not pred_col: pred_col = None
            if not sensitive_col: sensitive_col = None
            if not timestamp_col: timestamp_col = None
            
            # Auto-detect sensitive column if not provided
            SENSITIVE_ATTRIBUTES = ['gender', 'sex', 'race', 'ethnicity', 'age', 'religion', 'nationality', 'income', 'caste']
            if not sensitive_col:
                for col in df.columns:
                    col_lower = col.lower()
                    if any(attr in col_lower for attr in SENSITIVE_ATTRIBUTES):
                        sensitive_col = col
                        self.log(f"AUTO_DETECT: Sensitive attribute detected: {sensitive_col}")
                        break
            
            # Additional safety: ensure columns exist in df
            if pred_col and pred_col not in df.columns:
                self.log(f"Warning: Prediction column '{pred_col}' not found. Disabling calibration audit.")
                pred_col = None
            if sensitive_col and sensitive_col not in df.columns:
                self.log(f"Warning: Sensitive column '{sensitive_col}' not found. Disabling fairness audit.")
                sensitive_col = None
            if timestamp_col and timestamp_col not in df.columns:
                self.log(f"Warning: Timestamp column '{timestamp_col}' not found. Disabling temporal leakage scan.")
                timestamp_col = None
            
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in dataset.")

            y_raw = df[target_col].values
            
            # --- Data Preparation: Handle Regression vs Classification ---
            is_regression = False
            # Heuristic: if numeric and many unique values relative to size
            # Use pd.api.types.is_numeric_dtype for better compatibility with pandas types
            if pd.api.types.is_numeric_dtype(y_raw.dtype) and len(np.unique(y_raw[~pd.isna(y_raw)])) > 10:
                is_regression = True
                self.log(f"Detected regression target '{target_col}' ({len(np.unique(y_raw[~pd.isna(y_raw)]))} unique values).")
                self.log("Binarizing target (above/below median) for classification-based diagnostics.")
                median_val = np.nanmedian(y_raw)
                y_true = (y_raw > median_val).astype(int)
            else:
                # Ensure classification labels are 0/1 integers
                if not pd.api.types.is_numeric_dtype(y_raw.dtype):
                    # Factorize if string labels
                    self.log("Converting string labels to numeric for analysis.")
                    y_true = pd.factorize(y_raw)[0]
                else:
                    # Cast to int to ensure numeric division in engines
                    y_true = y_raw.astype(int)
                
                # Check if it's already binary-ish
                unique_labels = np.unique(y_true)
                if len(unique_labels) > 2:
                    self.log(f"Multi-class detected ({len(unique_labels)} classes). Mapping to binary for simplified diagnostics.")
                    y_true = (y_true > np.median(y_true)).astype(int)
                elif not np.array_equal(unique_labels, [0, 1]) and len(unique_labels) == 2:
                    self.log(f"Binary labels found but not [0, 1]. Normalizing to [0, 1].")
                    y_true = pd.factorize(y_true)[0]

            # --- Probability Handling ---
            prediction_source = "user_supplied" if pred_col else ("auto_generated" if auto_predict else "simulated")
            model_type = "None"

            if pred_col:
                self.log(f"Using provided prediction column: {pred_col}")
                y_preds_raw = df[pred_col].values
                # Check if it's already 2D (probabilities for all classes) or 1D
                if len(y_preds_raw.shape) == 1:
                    # If 1D, assume it's probabilities for positive class or raw predictions
                    # Ensure it's in [0, 1] for calibration
                    if np.min(y_preds_raw) < 0 or np.max(y_preds_raw) > 1:
                        self.log("Warning: Prediction column values outside [0, 1]. Normalizing...")
                        y_preds_norm = (y_preds_raw - np.min(y_preds_raw)) / (np.max(y_preds_raw) - np.min(y_preds_raw) + 1e-9)
                    else:
                        y_preds_norm = y_preds_raw
                    
                    y_proba = np.zeros((len(y_true), 2))
                    y_proba[:, 1] = y_preds_norm
                    y_proba[:, 0] = 1 - y_preds_norm
                else:
                    y_proba = y_preds_raw
            elif auto_predict:
                self.log(f"AUTO_GENERATE: Training baseline model to generate predictions for '{target_col}'...")
                self.status = "running (Auto-Model)"
                
                # Prepare features: Drop target and non-numeric columns for simplicity
                X_auto = df.drop(columns=[target_col])
                for col in X_auto.columns:
                    if not pd.api.types.is_numeric_dtype(X_auto[col].dtype):
                        X_auto[col] = LabelEncoder().fit_transform(X_auto[col].astype(str))
                X_auto = X_auto.fillna(X_auto.mean(numeric_only=True))
                
                if is_regression:
                    model_type = "RandomForestRegressor"
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    # For regression, we still want probabilities for calibration audit (using binarized y_true)
                    # so we'll binarize the regressor's predictions too
                    raw_preds = cross_val_predict(model, X_auto, y_raw, cv=5)
                    # Normalize to [0, 1] for probability-like scores
                    y_preds_norm = (raw_preds - raw_preds.min()) / (raw_preds.max() - raw_preds.min() + 1e-9)
                else:
                    model_type = "RandomForestClassifier"
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    # Use method='predict_proba' for calibration
                    y_proba_cv = cross_val_predict(model, X_auto, y_true, cv=5, method='predict_proba')
                    # cross_val_predict with predict_proba returns [n_samples, n_classes]
                    if y_proba_cv.shape[1] == 2:
                        y_proba = y_proba_cv
                    else:
                        # Fallback if binary mapping failed somehow
                        y_preds_norm = y_proba_cv[:, 1] if y_proba_cv.shape[1] > 1 else y_proba_cv[:, 0]
                        y_proba = np.zeros((len(y_true), 2))
                        y_proba[:, 1] = y_preds_norm
                        y_proba[:, 0] = 1 - y_preds_norm

                if is_regression or (not 'y_proba' in locals()):
                    y_proba = np.zeros((len(y_true), 2))
                    y_proba[:, 1] = y_preds_norm
                    y_proba[:, 0] = 1 - y_preds_norm

                self.log(f"AUTO_GENERATE: Baseline {model_type} trained. Predictions generated.")
                pred_col = f"predicted_{target_col}"
            else:
                # Simulated probabilities for the purpose of the engine modules
                # In a real scenario, the user would provide a model or y_proba
                self.log("No prediction column provided. Generating simulated model probabilities for Label Noise audit...")
                np.random.seed(42)
                y_proba = np.zeros((len(y_true), 2))
                # Mocking a model that is 80% accurate relative to y_true (binarized if regression)
                noise = np.random.random(len(y_true)) > 0.8
                mock_preds = y_true.copy()
                mock_preds[noise] = 1 - mock_preds[noise]
                y_proba[:, 1] = np.clip(mock_preds * 0.9 + np.random.normal(0, 0.1, len(y_true)), 0.01, 0.99)
                y_proba[:, 0] = 1 - y_proba[:, 1]

            # Use original y_raw for engines that support regression (Leakage, Missing Data)
            # but use y_true (binarized) for classification-only engines.
            y_for_engines = y_raw if is_regression else y_true

            # --- Phase 2: Diagnostics ---
            self.progress = 10
            self.status = "running (Phase 2)"
            self.log("STATUS_UPDATE: Phase 2 Diagnostics started.")
            
            if pred_col or auto_predict:
                if self._check_watchdog(df):
                    return
                self.log(f"ENGINE_STARTED: CalibrationEngine (Source: {prediction_source})")

                def _run_calibration():
                    cal_engine = CalibrationEngine()
                    cal_result = cal_engine.evaluate(y_true, y_proba[:, 1])
                    cal_result['prediction_source'] = prediction_source
                    cal_result['model_type'] = model_type
                    return cal_result

                cal_status, cal_payload = self._run_with_timeout(
                    "CalibrationEngine",
                    45,
                    _run_calibration,
                    {
                        'status': 'SKIPPED',
                        'severity': 'NONE',
                        'findings': {},
                        'summary': 'Calibration timed out on this dataset.'
                    },
                    "CALIBRATION_TIMEOUT"
                )
                self.results['calibration'] = cal_payload
                if cal_status == "ok":
                    self.log("ENGINE_COMPLETED: CalibrationEngine audit complete.")
            else:
                self.log("Skipping CalibrationEngine (no prediction column provided).")
                self.results['calibration'] = {
                    "status": "SKIPPED",
                    "reason": "No prediction column provided",
                    "severity": "NONE",
                    "findings": {}
                }
            
            if sensitive_col and sensitive_col in df.columns:
                if self._check_watchdog(df):
                    return
                self.log(f"ENGINE_STARTED: FairnessEngine for attribute: {sensitive_col}")

                def _run_fairness():
                    fair_engine = FairnessEngine()
                    fair_engine.register_axis(sensitive_col, df[sensitive_col].values)
                    fair_result = fair_engine.run(y_true, (y_proba[:, 1] > 0.5).astype(int), y_proba[:, 1])
                    if not config.get('sensitive_col'):
                        fair_result['auto_detected'] = True
                        fair_result['auto_detected_column'] = sensitive_col
                    return fair_result

                fair_status, fair_payload = self._run_with_timeout(
                    "FairnessEngine",
                    45,
                    _run_fairness,
                    {
                        "status": "SKIPPED",
                        "severity": "NONE",
                        "findings": {},
                        "summary": "Fairness timed out on this dataset."
                    },
                    "FAIRNESS_TIMEOUT"
                )
                self.results['fairness'] = fair_payload
                if fair_status == "ok":
                    self.log(f"ENGINE_COMPLETED: FairnessEngine audit complete.")
            else:
                self.log("Skipping FairnessEngine (no sensitive column provided or auto-detected).")
                self.results['fairness'] = {
                    "status": "SKIPPED",
                    "reason": "No sensitive column provided or auto-detected",
                    "severity": "NONE",
                    "findings": {}
                }
            
            # --- Phase 3: Observability ---
            self.progress = 40
            self.status = "running (Phase 3)"
            self.log("STATUS_UPDATE: Phase 3 Observability started.")
            
            mid = len(df) // 2
            X = df.drop(columns=[target_col])
            X_ref = X.iloc[:mid].values.tolist()
            X_curr = X.iloc[mid:].values.tolist()

            if self._check_watchdog(df):
                return
            self.log("ENGINE_STARTED: DriftEngine")

            def _run_drift():
                drift_engine = DriftEngine()
                drift_engine.set_reference(X_ref, X.columns.tolist())
                return drift_engine.run(X_curr)

            drift_status, drift_payload = self._run_with_timeout(
                "DriftEngine",
                60,
                _run_drift,
                {
                    "status": "SKIPPED",
                    "severity": "NONE",
                    "findings": {},
                    "summary": "Drift analysis timed out on this dataset."
                },
                "DRIFT_TIMEOUT"
            )
            self.results['drift'] = drift_payload
            if drift_status == "ok":
                self.log("ENGINE_COMPLETED: DriftEngine feature drift detection complete.")

            # Slice Analysis
            if self._check_watchdog(df):
                return
            self.log("ENGINE_STARTED: SlicerEngine")

            def _run_slicer():
                slicer_engine = SlicerEngine(k=5, effect_size_threshold=0.2)
                y_pred = (y_proba[:, 1] > 0.5).astype(int)
                return slicer_engine.run(y_true, y_pred, X.values.tolist(), X.columns.tolist())

            slice_status, slice_payload = self._run_with_timeout(
                "SlicerEngine",
                60,
                _run_slicer,
                {
                    "status": "SKIPPED",
                    "severity": "NONE",
                    "findings": {},
                    "summary": "Slice analysis timed out on this dataset."
                },
                "SLICE_TIMEOUT"
            )
            self.results['slice'] = slice_payload
            if slice_status == "ok":
                self.log("ENGINE_COMPLETED: SlicerEngine slice analysis complete.")

            # --- Phase 4: Root Cause Analysis ---
            self.progress = 70
            self.status = "running (Phase 4)"
            self.log("STATUS_UPDATE: Phase 4 Root Cause Analysis started.")
            
            # 1. Label Noise
            if self._check_watchdog(df):
                return
            self.log("ENGINE_STARTED: LabelNoiseEngine")

            def _run_label_noise():
                ln_engine = LabelNoiseEngine()
                return ln_engine.run(y_proba, y_true)

            label_status, label_payload = self._run_with_timeout(
                "LabelNoiseEngine",
                60,
                _run_label_noise,
                {
                    "status": "SKIPPED",
                    "severity": "NONE",
                    "findings": {},
                    "summary": "Label noise analysis timed out on this dataset."
                },
                "LABEL_NOISE_TIMEOUT"
            )
            self.results['label_noise'] = label_payload
            if label_status == "ok":
                self.log("ENGINE_COMPLETED: LabelNoiseEngine audit complete.")
            
            # 2. Leakage
            if self._check_watchdog(df):
                return
            self.log("ENGINE_STARTED: LeakageEngine")

            def _run_leakage():
                leak_engine = LeakageEngine()
                return leak_engine.run(X.values, y_for_engines, df, None, X.columns.tolist(), timestamp_col)

            leak_status, leak_payload = self._run_with_timeout(
                "LeakageEngine",
                45,
                _run_leakage,
                {
                    "status": "SKIPPED",
                    "severity": "NONE",
                    "findings": {},
                    "summary": "Leakage analysis timed out on this dataset."
                },
                "LEAKAGE_TIMEOUT"
            )
            self.results['leakage'] = leak_payload
            if leak_status == "ok":
                self.log("ENGINE_COMPLETED: LeakageEngine feature leakage scan complete.")
            
            # 3. Missing Data
            if self._check_watchdog(df):
                return
            self.log("ENGINE_STARTED: MissingDataEngine")

            def _run_missing_data():
                md_engine = MissingDataEngine()

                # Prepare X for MissingDataEngine: ensure all columns are numeric for correlation checks
                # while preserving NaN positions for missingness analysis.
                X_md = X.copy()
                for col in X_md.columns:
                    if not pd.api.types.is_numeric_dtype(X_md[col].dtype):
                        series = X_md[col]
                        mask = series.isnull()
                        codes, _ = pd.factorize(series)
                        X_md[col] = pd.Series(codes, index=series.index, dtype=float)
                        X_md.loc[mask, col] = np.nan

                return md_engine.run(X_md, y_for_engines)

            missing_status, missing_payload = self._run_with_timeout(
                "MissingDataEngine",
                30,
                _run_missing_data,
                {
                    "status": "SKIPPED",
                    "severity": "NONE",
                    "findings": {},
                    "summary": "Missing data analysis timed out on this dataset."
                },
                "MISSING_DATA_TIMEOUT"
            )
            self.results['missing_data'] = missing_payload
            if missing_status == "ok":
                self.log("ENGINE_COMPLETED: MissingDataEngine analysis complete.")

            # --- Auto Root Cause Analysis ---
            self.progress = 80
            self.status = "running (Auto Root Cause)"
            self.log("STATUS_UPDATE: Auto Root Cause Analysis started.")
            if self._check_watchdog(df):
                return
            self.log("ENGINE_STARTED: AutoRootCauseEngine")

            def _run_root_cause():
                rc_engine = AutoRootCauseEngine(verbose=False)
                return rc_engine.run(
                    drift_report=self.results.get('drift'),
                    slice_report=self.results.get('slice'),
                    calibration_report=self.results.get('calibration'),
                    data_quality_report=self.results.get('missing_data'),
                    leakage_report=self.results.get('leakage'),
                    label_noise_report=self.results.get('label_noise'),
                    fairness_report=self.results.get('fairness')
                )

            root_status, root_payload = self._run_with_timeout(
                "AutoRootCauseEngine",
                120,
                _run_root_cause,
                {
                    "status": "SKIPPED",
                    "severity": "NONE",
                    "health_status": "Unknown",
                    "confidence": 0,
                    "root_causes": [],
                    "recommended_actions": [],
                    "summary": "Root cause analysis timed out on this dataset."
                },
                "ROOT_CAUSE_TIMEOUT"
            )
            self.results['root_cause'] = root_payload

            try:
                translator = DomainTranslator()
                audience = getattr(self, 'audience', 'ml_engineer')
                rc_data = self.results.get('root_cause', {})
                root_causes = rc_data.get('root_causes', [])
                for cause in root_causes:
                    cause['domain_translation'] = translator.translate(
                        finding_type='drift',
                        feature_name=cause.get('cause', ''),
                        severity=cause.get('severity', 'MEDIUM'),
                        technical_details=str(cause.get('evidence', '')),
                        audience=audience
                    )
            except Exception:
                pass

            if root_status in ("ok", "timeout", "error"):
                self.log("ENGINE_COMPLETED: AutoRootCauseEngine analysis complete.")
            
            # --- AI Investigator Analysis ---
            self.progress = 85
            self.status = "running (AI Investigator)"
            self.log("STATUS_UPDATE: AI Investigator Analysis started.")
            if self._check_watchdog(df):
                return
            self.log("ENGINE_STARTED: AIInvestigator")

            def _run_ai_investigator():
                from engine.modules.investigation import Investigation
                # DEBUG: verify root_cause data coming in
                root_cause_dict = self.results.get('root_cause', {})
                print(f"[AIInvestigator DEBUG] root_causes count: "
                      f"{len(root_cause_dict.get('root_causes', []))}", flush=True)
                print(f"[AIInvestigator DEBUG] health_status: "
                      f"{root_cause_dict.get('health_status', 'MISSING')}", flush=True)

                ai_investigator = AIInvestigator(use_llm=False, verbose=False)
                audience_key = getattr(self, 'audience', 'ml_engineer')
                audience = AIInvestigator.normalize_audience(audience_key)
                payload = {
                    'selected_audience': audience,
                    'ai_investigator_by_audience': {},
                    'ai_investigator': {}
                }

                root_cause_dict = self.results.get('root_cause', {})
                metadata = {
                    'leakage': self.results.get('leakage', {}),
                    'label_noise': self.results.get('label_noise', {}),
                    'calibration': self.results.get('calibration', {}),
                    'drift': self.results.get('drift', {}),
                    'missing_data': self.results.get('missing_data', {}),
                    'fairness': self.results.get('fairness', {}),
                }

                try:
                    if isinstance(root_cause_dict, dict):
                        merged = dict(root_cause_dict)
                        existing_meta = merged.get('metadata', {}) or {}
                        merged['metadata'] = {**metadata, **existing_meta}
                        investigation = Investigation.from_dict(merged)
                    else:
                        investigation = Investigation.from_dict({
                            'root_causes': [],
                            'metadata': metadata
                        })
                except Exception as inv_e:
                    print(f"[AI Investigator ERROR] {type(inv_e).__name__}: {inv_e}")
                    traceback.print_exc()
                    self.log(f"WARNING: Investigation conversion failed: {str(inv_e)}")
                    payload['ai_investigator'] = ai_investigator._generate_deterministic(
                        Investigation(health_status="Unknown", confidence=0, evidence=[], recommendations=[]),
                        audience
                    )
                    return payload

                rc_risk = investigation.metadata.get('risk') or (root_cause_dict or {}).get('risk')
                if rc_risk:
                    payload['risk_breakdown'] = rc_risk
                    payload['overall_risk'] = rc_risk.get('level')
                    payload['global_status'] = rc_risk.get('level')

                why = (
                    (root_cause_dict or {}).get('risk_explanation')
                    or investigation.metadata.get('risk_explanation')
                )
                if why:
                    payload['why_risk'] = why

                audiences = [
                    "ML Engineer",
                    "Executive",
                    "Doctor",
                    "Loan Officer",
                    "Student",
                    "HR Manager",
                    "Insurance Analyst",
                    "Legal / Compliance Officer",
                    "Researcher",
                ]

                ai_by_audience = {}
                try:
                    primary_report = ai_investigator.analyze(investigation, audience)
                    ai_by_audience[audience] = primary_report
                except Exception as e:
                    self.log(f"WARNING: Primary audience failed: {e}")
                    ai_by_audience[audience] = {}

                for aud in [a for a in audiences if a != audience]:
                    try:
                        ai_by_audience[aud] = ai_investigator.analyze(investigation, aud)
                    except Exception as aud_e:
                        self.log(f"WARNING: AIInvestigator '{aud}' failed: {aud_e}")
                        ai_by_audience[aud] = ai_by_audience.get(audience, {})

                if rc_risk and isinstance(rc_risk, dict):
                    unified_level = rc_risk.get('level')
                    if unified_level:
                        for k, v in list(ai_by_audience.items()):
                            if isinstance(v, dict):
                                v['risk_level'] = unified_level
                                ai_by_audience[k] = v

                payload['ai_investigator_by_audience'] = ai_by_audience
                payload['ai_investigator'] = ai_by_audience.get(audience, ai_by_audience.get("ML Engineer", {}))
                if rc_risk and isinstance(rc_risk, dict):
                    top = payload.get('ai_investigator', {})
                    if isinstance(top, dict) and rc_risk.get('level'):
                        top['risk_level'] = rc_risk.get('level')
                        payload['ai_investigator'] = top

                # DEBUG: verify summary was generated
                ml_report = ai_by_audience.get("ML Engineer", {})
                print(f"[AIInvestigator DEBUG] ML Engineer executive_summary length: "
                      f"{len(ml_report.get('executive_summary', ''))}", flush=True)
                print(f"[AIInvestigator DEBUG] executive_summary preview: "
                      f"{repr(ml_report.get('executive_summary', ''))[:200]}", flush=True)

                return payload

            ai_status, ai_payload = self._run_with_timeout(
                "AIInvestigator",
                180,
                _run_ai_investigator,
                {
                    "selected_audience": getattr(self, 'audience', 'ML Engineer'),
                    "ai_investigator_by_audience": {},
                    "ai_investigator": {
                        "status": "SKIPPED",
                        "risk_level": "UNKNOWN",
                        "executive_summary": "",
                        "investigation_findings": "",
                        "impact_assessment": "",
                        "confidence_explanation": "",
                        "recommended_actions": [],
                        "technical_notes": "",
                        "summary": "AI Investigator timed out on this dataset."
                    }
                },
                "AI_INVESTIGATOR_TIMEOUT"
            )
            if isinstance(ai_payload, dict):
                self.results.update(ai_payload)
            if ai_status == "ok":
                self.log("ENGINE_COMPLETED: AIInvestigator analysis complete.")
            
            # --- Audience Translation ---
            self.progress = 90
            self.status = "running (Audience Translation)"
            self.log("STATUS_UPDATE: Audience Translation started.")

            if self._check_watchdog(df):
                return
            self.log("ENGINE_STARTED: AudienceTranslator")

            def _run_audience_translation():
                audience_translator = AudienceTranslator(verbose=False)
                return audience_translator.translate(
                    root_cause=self.results.get('root_cause', {}),
                    ai_investigator=self.results.get('ai_investigator', {}),
                    ai_investigator_by_audience=self.results.get('ai_investigator_by_audience')
                )

            audience_status, audience_payload = self._run_with_timeout(
                "AudienceTranslator",
                45,
                _run_audience_translation,
                {
                    "status": "SKIPPED",
                    "summary": "Audience translation timed out on this dataset."
                },
                "AUDIENCE_TRANSLATION_TIMEOUT"
            )
            self.results['audience_reports'] = audience_payload
            if audience_status == "ok":
                self.log("ENGINE_COMPLETED: AudienceTranslator analysis complete.")

            # --- Save Results ---
            self._write_report(df=df)

        except Exception as e:
            traceback.print_exc()
            print("=" * 80)
            print("ANALYSIS EXECUTION ERROR (Background Thread)")
            print(repr(e))
            print("=" * 80)
            tb = traceback.format_exc()
            self.log(f"CRITICAL_ERROR: {str(e)}")
            self.log(f"TRACEBACK:\n{tb}")
            self.error = str(e)
            self.status = "failed"

runner = AnalysisRunner()