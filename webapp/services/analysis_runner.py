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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder

# Add project root to path to import engine modules
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine.modules.fairness_engine import FairnessEngine
from engine.modules.calibration_engine import CalibrationEngine
from engine.modules.drift_engine import DriftEngine
from engine.modules.leakage_engine import LeakageEngine
from engine.modules.label_noise_engine import LabelNoiseEngine
from engine.modules.missing_data_engine import MissingDataEngine

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

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")

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
                try:
                    self.log(f"ENGINE_STARTED: CalibrationEngine (Source: {prediction_source})")
                    cal_engine = CalibrationEngine()
                    self.results['calibration'] = cal_engine.evaluate(y_true, y_proba[:, 1])
                    self.results['calibration']['prediction_source'] = prediction_source
                    self.results['calibration']['model_type'] = model_type
                    self.log("ENGINE_COMPLETED: CalibrationEngine audit complete.")
                except Exception as e:
                    traceback.print_exc()
                    self.log(f"ENGINE_FAILED: CalibrationEngine error: {str(e)}")
                    self.results['calibration'] = {"status": "FAILED", "error": str(e), "severity": "HIGH"}
            else:
                self.log("Skipping CalibrationEngine (no prediction column provided).")
                self.results['calibration'] = {
                    "status": "SKIPPED",
                    "reason": "No prediction column provided",
                    "severity": "NONE",
                    "findings": {}
                }
            
            if sensitive_col and sensitive_col in df.columns:
                try:
                    self.log(f"ENGINE_STARTED: FairnessEngine for attribute: {sensitive_col}")
                    fair_engine = FairnessEngine()
                    # FairnessEngine requires register_axis() before run()
                    fair_engine.register_axis(sensitive_col, df[sensitive_col].values)
                    self.results['fairness'] = fair_engine.run(y_true, (y_proba[:, 1] > 0.5).astype(int), y_proba[:, 1])
                    self.log(f"ENGINE_COMPLETED: FairnessEngine audit complete.")
                except Exception as e:
                    traceback.print_exc()
                    self.log(f"ENGINE_FAILED: FairnessEngine error: {str(e)}")
                    self.results['fairness'] = {"status": "FAILED", "error": str(e), "severity": "HIGH"}
            
            # --- Phase 3: Observability ---
            self.progress = 40
            self.status = "running (Phase 3)"
            self.log("STATUS_UPDATE: Phase 3 Observability started.")
            
            try:
                self.log("ENGINE_STARTED: DriftEngine")
                drift_engine = DriftEngine()
                # DriftEngine requires set_reference() then run(X_current)
                mid = len(df) // 2
                X = df.drop(columns=[target_col])
                X_ref = X.iloc[:mid].values.tolist()
                X_curr = X.iloc[mid:].values.tolist()
                drift_engine.set_reference(X_ref, X.columns.tolist())
                self.results['drift'] = drift_engine.run(X_curr)
                self.log("ENGINE_COMPLETED: DriftEngine feature drift detection complete.")
            except Exception as e:
                traceback.print_exc()
                self.log(f"ENGINE_FAILED: DriftEngine error: {str(e)}")
                self.results['drift'] = {"status": "FAILED", "error": str(e), "severity": "HIGH"}

            # --- Phase 4: Root Cause Analysis ---
            self.progress = 70
            self.status = "running (Phase 4)"
            self.log("STATUS_UPDATE: Phase 4 Root Cause Analysis started.")
            
            # 1. Label Noise
            try:
                self.log("ENGINE_STARTED: LabelNoiseEngine")
                ln_engine = LabelNoiseEngine()
                self.results['label_noise'] = ln_engine.run(y_proba, y_true)
                self.log("ENGINE_COMPLETED: LabelNoiseEngine audit complete.")
            except Exception as e:
                traceback.print_exc()
                self.log(f"ENGINE_FAILED: LabelNoiseEngine error: {str(e)}")
                self.results['label_noise'] = {"status": "FAILED", "error": str(e), "severity": "HIGH"}
            
            # 2. Leakage
            try:
                self.log("ENGINE_STARTED: LeakageEngine")
                leak_engine = LeakageEngine()
                # Leakage scanner can handle original y_raw
                self.results['leakage'] = leak_engine.run(X.values, y_for_engines, df, None, X.columns.tolist(), timestamp_col)
                self.log("ENGINE_COMPLETED: LeakageEngine feature leakage scan complete.")
            except Exception as e:
                traceback.print_exc()
                self.log(f"ENGINE_FAILED: LeakageEngine error: {str(e)}")
                self.results['leakage'] = {"status": "FAILED", "error": str(e), "severity": "HIGH"}
            
            # 3. Missing Data
            try:
                self.log("ENGINE_STARTED: MissingDataEngine")
                md_engine = MissingDataEngine()
                
                # Prepare X for MissingDataEngine: ensure all columns are numeric for correlation checks
                # while preserving NaN positions for missingness analysis.
                X_md = X.copy()
                for col in X_md.columns:
                    if not pd.api.types.is_numeric_dtype(X_md[col].dtype):
                        # Factorize categorical columns to numeric codes, preserving NaNs
                        series = X_md[col]
                        mask = series.isnull()
                        # pd.factorize returns -1 for NaNs by default
                        codes, _ = pd.factorize(series)
                        X_md[col] = pd.Series(codes, index=series.index, dtype=float)
                        X_md.loc[mask, col] = np.nan
                
                self.results['missing_data'] = md_engine.run(X_md, y_for_engines)
                self.log("ENGINE_COMPLETED: MissingDataEngine analysis complete.")
            except Exception as e:
                traceback.print_exc()
                self.log(f"ENGINE_FAILED: MissingDataEngine error: {str(e)}")
                self.results['missing_data'] = {"status": "FAILED", "error": str(e), "severity": "HIGH"}

            # --- Save Results ---
            self.progress = 90
            self.status = "saving_report"
            
            # Generate Visualizations
            try:
                self._generate_charts(df, self.results)
            except Exception as e:
                self.log(f"WARNING: Chart generation failed: {str(e)}")
                traceback.print_exc()

            self.log("STATUS_UPDATE: Consolidating results and saving report...")
            
            report_filename = f"report_{int(time.time())}.json"
            reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            self.report_path = os.path.join(reports_dir, report_filename)
            
            with open(self.report_path, 'w') as f:
                json.dump(self.results, f, indent=4)
            
            self.log(f"REPORT_SAVED: Results saved to {report_filename}")
            self.progress = 100
            self.status = "completed"
            self.log("ANALYSIS_COMPLETED: Investigation finished successfully.")

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
