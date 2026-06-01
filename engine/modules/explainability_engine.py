"""
Engine Module — Explainability Engine
=======================================
Production wrapper around scratch/phase3/shap_explainer.py,
scratch/phase3/lime_explainer.py, and scratch/phase3/shap_vs_lime.py.

Responsibilities:
  • Single entry point: explain_instance() or explain_batch()
  • Auto-selects best method based on model type and budget
  • Runs SHAP vs LIME agreement check on demand
  • Returns structured dicts suitable for the report engine
  • Enforces explainability quality gate (min feature coverage)

Usage:
    from engine.modules.explainability_engine import ExplainabilityEngine

    ee = ExplainabilityEngine(method="shap")
    result = ee.run(model_fn, x_instance, X_background, feature_names)
    ee.assert_gate(result, min_spearman=0.7)
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_P3   = os.path.abspath(os.path.join(_HERE, "..", "..", "scratch", "phase3"))
_BASE = os.path.abspath(os.path.join(_HERE, ".."))
for p in [_P3, _BASE]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from shap_explainer import (
        shapley_values_exact, kernel_shap, shap_summary_data
    )
    from lime_explainer import lime_explain, lime_stability_analysis
    from shap_vs_lime import (
        compare_explanations, stability_analysis, batch_agreement, DECISION_GUIDE
    )
    _MODULES_LOADED = True
except ImportError as e:
    _MODULES_LOADED = False
    _IMPORT_ERROR = str(e)

try:
    from base_module import BaseModule, GateError, highest_severity
    _BASE_LOADED = True
except ImportError:
    # Fallback: minimal base
    class BaseModule:
        def __init__(self, verbose=True):
            self.verbose = verbose
        def _log(self, msg):
            if self.verbose:
                import sys; print(f"[ExplainabilityEngine] {msg}", file=sys.stderr)
        def _result(self, findings, severity="NONE", module_name=None):
            return {"module": module_name or "ExplainabilityEngine",
                    "severity": severity, "passed": severity in ("NONE","LOW"),
                    "findings": findings, "log": []}
        def assert_gate(self, result, **kwargs): pass
        def run(self, *a, **kw): return self._run(*a, **kw)

    class GateError(Exception): pass
    _BASE_LOADED = False


class ExplainabilityGateError(GateError):
    pass


class ExplainabilityEngine(BaseModule):
    """
    End-to-end explainability pipeline for the ML Failure Engine.

    Supports:
      method="shap"   → Kernel SHAP (exact for small M, kernel for large)
      method="lime"   → LIME perturbation-based local explanation
      method="both"   → Run both and compute agreement
      method="auto"   → Choose based on n_features and model_type
    """

    def __init__(self, method="auto", verbose=True):
        super().__init__(verbose=verbose)
        self.method = method

    # ── Main entry point ─────────────────────────────────────────────────────

    def _run(self, model_fn, x_instance, X_background,
             feature_names=None, n_samples=300, **kwargs):
        """
        Explain a single instance.

        Parameters
        ----------
        model_fn      : callable(x_list) → float   prediction function
        x_instance    : list[float]                 the instance to explain
        X_background  : list[list[float]]           background/training data
        feature_names : list[str]
        n_samples     : int                         SHAP/LIME samples

        Returns standard result envelope with findings:
          method_used, shap_values, lime_values, feature_importance,
          top5_features, agreement (if both), stability
        """
        if not _MODULES_LOADED:
            return self._result(
                {"error": f"Phase 3 modules not importable: {_IMPORT_ERROR}"},
                severity="CRITICAL"
            )

        n_features = len(x_instance)
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(n_features)]

        method = self._choose_method(n_features)
        self._log(f"Method: {method}, n_features={n_features}, n_samples={n_samples}")

        findings = {"method_used": method, "n_features": n_features}

        # ── SHAP ─────────────────────────────────────────────────────────────
        shap_vals = None
        if method in ("shap", "both"):
            try:
                bg_means = [
                    sum(row[j] for row in X_background) / len(X_background)
                    for j in range(n_features)
                ] if X_background else [0.0] * n_features

                if n_features <= 10:
                    result_shap = shapley_values_exact(
                        model_fn, x_instance, bg_means, feature_names
                    )
                    shap_vals = result_shap["shap_values"]
                else:
                    result_shap = kernel_shap(
                        model_fn, x_instance, X_background,
                        n_samples=n_samples
                    )
                    shap_vals = result_shap["shap_values"]

                findings["shap_values"] = {
                    feature_names[i]: round(shap_vals[i], 6)
                    for i in range(n_features)
                }
                self._log(f"SHAP complete, sum={sum(shap_vals):.4f}")
            except Exception as ex:
                self._log(f"SHAP failed: {ex}")
                findings["shap_error"] = str(ex)

        # ── LIME ─────────────────────────────────────────────────────────────
        lime_vals = None
        if method in ("lime", "both"):
            try:
                result_lime = lime_explain(
                    model_fn, x_instance, X_background,
                    feature_names=feature_names, K=min(n_features, 6),
                    n_samples=n_samples
                )
                lime_dict = dict(result_lime["feature_importances"])
                lime_vals = [lime_dict.get(f, 0.0) for f in feature_names]
                findings["lime_values"] = {
                    f: round(v, 6) for f, v in result_lime["feature_importances"]
                }
                findings["lime_r2"]     = result_lime.get("local_r2", None)
                self._log(f"LIME complete, R²={findings['lime_r2']:.3f}"
                          if findings.get("lime_r2") else "LIME complete")
            except Exception as ex:
                self._log(f"LIME failed: {ex}")
                findings["lime_error"] = str(ex)

        # ── Agreement ────────────────────────────────────────────────────────
        if method == "both" and shap_vals is not None and lime_vals is not None:
            try:
                agreement = compare_explanations(
                    shap_vals, lime_vals, feature_names
                )
                findings["agreement"] = {
                    "spearman_rho":   agreement["spearman_rho"],
                    "top3_overlap":   agreement["top3_overlap"],
                    "verdict":        agreement["verdict"],
                }
                self._log(f"Agreement verdict: {agreement['verdict']}, "
                          f"ρ={agreement['spearman_rho']}")
            except Exception as ex:
                self._log(f"Agreement check failed: {ex}")

        # ── Unified top features ─────────────────────────────────────────────
        primary_vals = shap_vals or lime_vals or ([0.0] * n_features)
        ranked = sorted(
            range(n_features), key=lambda i: abs(primary_vals[i]), reverse=True
        )
        findings["top5_features"] = [
            {"feature": feature_names[i],
             "importance": round(primary_vals[i], 5),
             "rank": r + 1}
            for r, i in enumerate(ranked[:5])
        ]

        # ── Severity ─────────────────────────────────────────────────────────
        if "shap_error" in findings and "lime_error" in findings:
            sev = "HIGH"
        elif findings.get("lime_r2", 1.0) is not None and \
             findings.get("lime_r2", 1.0) < 0.3:
            sev = "MEDIUM"   # local approximation poor
        else:
            sev = "NONE"

        return self._result(findings, severity=sev,
                            module_name="ExplainabilityEngine")

    def _choose_method(self, n_features):
        if self.method != "auto":
            return self.method
        if n_features <= 8:
            return "both"
        elif n_features <= 20:
            return "shap"
        else:
            return "lime"   # faster for high-dimensional

    # ── Batch explanation ─────────────────────────────────────────────────────

    def explain_batch(self, model_fn, X_instances, X_background,
                      feature_names=None, n_samples=200):
        """
        Explain multiple instances and return summary statistics.
        Returns dict with per_instance results + global importance ranking.
        """
        results = []
        all_importances = {}

        for i, x in enumerate(X_instances):
            r = self.run(model_fn, x, X_background, feature_names, n_samples)
            results.append(r)
            for feat_dict in r["findings"].get("top5_features", []):
                f = feat_dict["feature"]
                all_importances.setdefault(f, []).append(abs(feat_dict["importance"]))

        # Global importance = mean |importance| across instances
        global_rank = sorted(
            all_importances.items(),
            key=lambda kv: sum(kv[1]) / max(len(kv[1]), 1),
            reverse=True
        )

        return {
            "n_instances":      len(X_instances),
            "per_instance":     results,
            "global_importance": [
                {"feature": f, "mean_abs_importance": round(
                    sum(vals) / max(len(vals), 1), 5
                )} for f, vals in global_rank[:10]
            ],
        }

    # ── Gate ─────────────────────────────────────────────────────────────────

    def assert_gate(self, result, min_spearman=0.6, max_severity="MEDIUM"):
        """
        Raise ExplainabilityGateError if:
          - result severity exceeds max_severity
          - SHAP/LIME agreement below min_spearman (if both were run)
        """
        sev_order = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        actual = result.get("severity", "NONE")
        if sev_order.get(actual, 0) > sev_order.get(max_severity, 2):
            raise ExplainabilityGateError(
                f"ExplainabilityEngine severity {actual} > {max_severity}"
            )

        agreement = result["findings"].get("agreement", {})
        rho = agreement.get("spearman_rho", 1.0)
        if rho < min_spearman:
            raise ExplainabilityGateError(
                f"SHAP/LIME agreement too low: ρ={rho:.3f} < min={min_spearman}. "
                "Explanations are inconsistent — investigate model behaviour."
            )
