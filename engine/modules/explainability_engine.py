import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False

class ExplainabilityEngine:
    """
    Computes SHAP-based global feature importance for an uploaded model.
    Falls back gracefully if shap is not installed, or if the model is
    incompatible with the available SHAP explainers.
    """

    def run(self, model, X, feature_names, max_samples=100):
        if not SHAP_AVAILABLE:
            return {
                "status": "SKIPPED",
                "severity": "NONE",
                "findings": {},
                "summary": "shap package not installed. Run: pip install shap"
            }

        try:
            # Keep SHAP computation fast on larger datasets
            X_sample = X.iloc[:max_samples] if hasattr(X, 'iloc') else X[:max_samples]

            try:
                # Fast path: native support for tree-based models
                explainer = shap.Explainer(model, X_sample)
                shap_values = explainer(X_sample)
                values = shap_values.values
            except Exception:
                # Fallback: model-agnostic path via predict_proba
                explainer = shap.Explainer(model.predict_proba, X_sample)
                shap_values = explainer(X_sample)
                values = shap_values.values

            # Some explainers return a 3D array for multi-class outputs
            # (n_samples, n_features, n_classes) — take the positive class
            if values.ndim == 3:
                values = values[:, :, 1]

            mean_abs_shap = np.abs(values).mean(axis=0)
            importance = sorted(
                [
                    {"feature": str(f), "importance": float(v)}
                    for f, v in zip(feature_names, mean_abs_shap)
                ],
                key=lambda x: x["importance"],
                reverse=True
            )

            top_feature = importance[0]["feature"] if importance else None

            return {
                "status": "COMPLETED",
                "severity": "INFO",
                "findings": {
                    "global_importance": importance,
                    "top_feature": top_feature,
                    "samples_used": int(len(X_sample))
                },
                "summary": (
                    f"Top feature driving predictions: {top_feature}."
                    if top_feature else "No dominant feature found."
                )
            }

        except Exception as e:
            return {
                "status": "FAILED",
                "severity": "NONE",
                "findings": {},
                "error": str(e),
                "summary": f"SHAP explainability failed: {str(e)}"
            }
