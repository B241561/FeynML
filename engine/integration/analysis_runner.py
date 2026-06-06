import os
import pandas as pd
import numpy as np
import traceback
from datetime import datetime

from uuid import uuid4


def _read_dataset(path):
    try:
        if path.endswith('.csv'):
            return pd.read_csv(path)
        elif path.endswith('.json'):
            return pd.read_json(path, lines=True)
        else:
            return pd.DataFrame()
    except Exception:
        raise


def run_analysis(dataset_path, analysis_type, config):
    """
    Orchestrate calls to Phase 4 engine modules.
    Returns JSON-serializable dict.
    """
    execution_logs = []
    def log(msg):
        execution_logs.append(f"[{analysis_type.upper()}] {msg}")
        print(f"[{analysis_type.upper()}] {msg}")

    log(f"Starting analysis: {analysis_type}")
    log(f"Dataset path: {dataset_path}")

    try:
        df = _read_dataset(dataset_path)
        log(f"Dataset loaded successfully. Shape: {df.shape}")
    except Exception as e:
        log(f"Failed to read dataset: {e}")
        return {"status": "failed", "error": f"Failed to read dataset: {e}", "log": execution_logs}

    try:
        if analysis_type == 'label_noise':
            from engine.label_noise import LabelNoiseAnalyzer
            target = config.get('target_col')
            if not target:
                log("Target column is required for Label Noise analysis")
                return {"status": "failed", "error": "Target column is required for Label Noise analysis", "log": execution_logs}
            
            analyzer = LabelNoiseAnalyzer()
            try:
                res = analyzer.run(df, target)
                res["log"] = execution_logs + res.get("log", [])
                return {"status": "ok", "results": res}
            except Exception as e:
                log(f"Label Noise analysis failed: {e}")
                return {"status": "failed", "error": str(e), "log": execution_logs}

        if analysis_type == 'leakage':
            from engine.leakage_detector import LeakageDetector
            target = config.get('target_col')
            date_col = config.get('date_col')
            
            detector = LeakageDetector()
            try:
                res = detector.run(df, target_col=target, date_col=date_col)
                res["log"] = execution_logs + res.get("log", [])
                return {"status": "ok", "results": res}
            except Exception as e:
                log(f"Leakage analysis failed: {e}")
                return {"status": "failed", "error": str(e), "log": execution_logs}

        if analysis_type == 'missing_data':
            from engine.missing_data import MissingDataAnalyzer
            target = config.get('target_col')
            analyzer = MissingDataAnalyzer()
            try:
                res = analyzer.run(df, target_col=target)
                res["log"] = execution_logs + res.get("log", [])
                return {"status": "ok", "results": res}
            except Exception as e:
                log(f"Missing data analysis failed: {e}")
                return {"status": "failed", "error": str(e), "log": execution_logs}

        if analysis_type == 'causal':
            # causal: run causal graph builder and inference
            from engine.causal_thinking import CausalGraphBuilder
            from engine.causal_inference import CausalInferenceEngine

            nodes = config.get('nodes')
            edges = config.get('edges')
            treatment = config.get('treatment_col')
            outcome = config.get('outcome_col')
            
            engine = CausalInferenceEngine()
            try:
                # CausalInferenceEngine.run expects treatment_col and outcome_col
                inf_res = engine.run(df, treatment_col=treatment, outcome_col=outcome, covariates=config.get('covariates', []))
                
                builder = CausalGraphBuilder()
                # Placeholder for graph if not provided
                graph_res = builder.build_from_text(nodes=nodes, edges=edges) if (nodes and edges and hasattr(builder, 'build_from_text')) else {"message": "No DAG specified"}
                
                # Wrap in result envelope manually since CausalInferenceEngine might not be a BaseModule
                res = {
                    "module": "CausalInferenceEngine",
                    "status": "ok",
                    "results": {"graph": graph_res, "inference": inf_res},
                    "findings": {"graph": graph_res, "inference": inf_res},
                    "log": execution_logs,
                    "severity": "NONE",
                    "passed": True
                }
                return {"status": "ok", "results": res}
            except Exception as e:
                log(f"Causal analysis failed: {e}")
                return {"status": "failed", "error": str(e), "log": execution_logs}

        if analysis_type == 'explainability':
            from engine.modules.explainability_engine import ExplainabilityEngine
            import sklearn.ensemble
            
            target = config.get('target_col')
            if not target:
                log("Target column is required for Explainability analysis")
                return {"status": "failed", "error": "Target column is required for Explainability analysis", "log": execution_logs}
            
            if target not in df.columns:
                log(f"Target column '{target}' not found in dataset columns: {df.columns.tolist()}")
                return {"status": "failed", "error": f"Target column '{target}' not found", "log": execution_logs}

            log(f"Selected target column: {target}")
            log(f"Dataset shape: {df.shape}")
            
            # Train a baseline model if none provided
            X = df.drop(columns=[target]).select_dtypes(include=[np.number])
            y = df[target]
            
            if X.empty:
                log("No numeric features found for explainability analysis.")
                return {"status": "failed", "error": "No numeric features found", "log": execution_logs}
            
            # Basic preprocessing: Fill NaNs for baseline model
            X = X.fillna(X.mean())
            if y.isnull().any():
                log("Warning: Target column contains NaNs. Dropping these rows.")
                valid_mask = y.notnull()
                X = X[valid_mask]
                y = y[valid_mask]

            feature_names = X.columns.tolist()
            log(f"Number of features analyzed: {len(feature_names)}")
            log(f"Features for analysis: {feature_names}")
            
            # Simple heuristic for classification vs regression
            is_classification = y.nunique() < 20
            log(f"Model type detected: {'Classification' if is_classification else 'Regression'}")
            
            try:
                if is_classification:
                    log("Training RandomForestClassifier baseline...")
                    model = sklearn.ensemble.RandomForestClassifier(n_estimators=10, random_state=42)
                    model.fit(X, y)
                    log("Model training status: SUCCESS")
                    # Wrap model_fn to ensure scalar float output and 2D input for sklearn
                    model_fn = lambda x: float(model.predict_proba(np.atleast_2d(x))[0, 1])
                else:
                    log("Training RandomForestRegressor baseline...")
                    # Ensure y is numeric for regression
                    if not np.issubdtype(y.dtype, np.number):
                        log(f"Error: Target column '{target}' is not numeric but detected as regression (nunique={y.nunique()})")
                        return {"status": "failed", "error": f"Target column '{target}' must be numeric for regression", "log": execution_logs}
                    
                    model = sklearn.ensemble.RandomForestRegressor(n_estimators=10, random_state=42)
                    model.fit(X, y)
                    log("Model training status: SUCCESS")
                    # Wrap model_fn to ensure scalar float output and 2D input for sklearn
                    model_fn = lambda x: float(model.predict(np.atleast_2d(x))[0])
                log("Baseline model trained successfully.")
            except Exception as e:
                log(f"Model training status: FAILED")
                log(f"Failed to train baseline model: {str(e)}")
                log(f"Traceback: {traceback.format_exc()}")
                return {"status": "failed", "error": f"Model training failed: {str(e)}", "log": execution_logs}

            engine = ExplainabilityEngine(method="shap")
            # Explain a random instance from the dataset
            x_instance = X.iloc[0].tolist()
            X_background = X.sample(min(50, len(X))).values.tolist()
            
            try:
                log("SHAP execution status: STARTING")
                log("Running SHAP explain_instance...")
                res = engine.run(model_fn, x_instance, X_background, feature_names=feature_names)
                
                log("Running SHAP explain_batch for global importance...")
                batch_res = engine.explain_batch(model_fn, X.sample(min(20, len(X))).values.tolist(), X_background, feature_names=feature_names)
                
                log("SHAP execution status: SUCCESS")
                log("Explainability analysis completed successfully.")
                
                res["findings"]["global_importance"] = batch_res["global_importance"]
                res["findings"]["model_type"] = "classification" if is_classification else "regression"
                res["log"] = execution_logs + res.get("log", [])
                
                return {"status": "ok", "results": res}
            except Exception as e:
                log("SHAP execution status: FAILED")
                log(f"SHAP execution failed: {str(e)}")
                tb = traceback.format_exc()
                log(f"Traceback: {tb}")
                return {"status": "failed", "error": f"SHAP failed: {str(e)}", "log": execution_logs}

        if analysis_type == 'root_cause':
            from engine.modules.root_cause_engine import RootCauseEngine
            
            target = config.get('target_col')
            # Split dataset into ref and cur for demonstration if not provided
            # In production, these would be separate uploads
            split_idx = int(len(df) * 0.7)
            ref_df = df.iloc[:split_idx]
            cur_df = df.iloc[split_idx:]
            
            log(f"Running root cause analysis on split: {len(ref_df)} ref, {len(cur_df)} cur")
            engine = RootCauseEngine()
            try:
                # RootCauseEngine.run might return a raw dict, let's wrap it if needed
                res = engine.run(ref_df, cur_df, target_col=target)
                if "module" not in res:
                    res = {
                        "module": "RootCauseEngine",
                        "findings": res,
                        "log": execution_logs,
                        "severity": "NONE",
                        "passed": True,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    res["log"] = execution_logs + res.get("log", [])
                return {"status": "ok", "results": res}
            except Exception as e:
                log(f"Root cause analysis failed: {e}")
                return {"status": "failed", "error": str(e), "log": execution_logs}

        log(f"Unknown analysis_type: {analysis_type}")
        return {"status": "failed", "error": f"Unknown analysis_type: {analysis_type}", "log": execution_logs}

    except Exception as e:
        tb = traceback.format_exc()
        log(f"Unexpected error in run_analysis: {e}")
        log(f"Traceback: {tb}")
        return {"status": "failed", "error": str(e), "trace": tb, "log": execution_logs}
