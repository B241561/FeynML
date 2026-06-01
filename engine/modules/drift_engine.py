"""
Engine Module — Drift Engine
==============================
Production wrapper around scratch/phase3/drift_detection.py.

Responsibilities:
  • Register reference (training) distribution per feature
  • Run drift checks on new production batches
  • Classify each feature as STABLE / WARN / DRIFT
  • Emit structured alerts ready for monitoring dashboards
  • Enforce drift gate: block if critical drift detected

Usage:
    from engine.modules.drift_engine import DriftEngine

    de = DriftEngine()
    de.set_reference(X_train, feature_names)
    result = de.run(X_production)
    de.assert_gate(result, max_severity="MEDIUM")
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
    from drift_detection import (
        ks_test, psi, chi2_test_categorical,
        domain_classifier_drift, full_drift_report,
    )
    _MODULES_LOADED = True
except ImportError as e:
    _MODULES_LOADED = False
    _IMPORT_ERROR = str(e)

try:
    from base_module import BaseModule, GateError
    _BASE_LOADED = True
except ImportError:
    class BaseModule:
        def __init__(self, verbose=True):
            self.verbose = verbose
        def _log(self, msg):
            if self.verbose:
                import sys; print(f"[DriftEngine] {msg}", file=sys.stderr)
        def _result(self, findings, severity="NONE", module_name=None):
            return {"module": module_name or "DriftEngine",
                    "severity": severity, "passed": severity in ("NONE","LOW"),
                    "findings": findings, "log": []}
        def assert_gate(self, result, **kwargs): pass
        def run(self, *a, **kw): return self._run(*a, **kw)

    class GateError(Exception): pass
    _BASE_LOADED = False

# Thresholds
PSI_THRESHOLDS = {"STABLE": 0.10, "WARN": 0.20}   # ≥0.20 → DRIFT
KS_ALPHA       = 0.05


class DriftGateError(GateError):
    pass


class DriftEngine(BaseModule):
    """
    Production drift monitoring engine.

    Call flow:
      1. de.set_reference(X_train, feature_names, categorical_cols=[])
      2. de.run(X_new)  →  result envelope
      3. de.assert_gate(result)  →  raises if critical drift
    """

    def __init__(self, psi_thresholds=None, ks_alpha=KS_ALPHA, verbose=True):
        super().__init__(verbose=verbose)
        self.psi_thresholds  = psi_thresholds or PSI_THRESHOLDS
        self.ks_alpha        = ks_alpha
        self._reference      = None
        self._feature_names  = None
        self._categorical    = set()

    def set_reference(self, X_reference, feature_names=None, categorical_cols=None):
        """
        Register the reference (training) dataset.

        X_reference    : list[list[float]]
        feature_names  : list[str]
        categorical_cols: list[int] or list[str]  column indices/names that are categorical
        """
        self._reference     = X_reference
        n_feat = len(X_reference[0]) if X_reference else 0
        self._feature_names = feature_names or [f"f{i}" for i in range(n_feat)]
        if categorical_cols:
            if isinstance(categorical_cols[0], str):
                self._categorical = {
                    self._feature_names.index(c) for c in categorical_cols
                    if c in self._feature_names
                }
            else:
                self._categorical = set(categorical_cols)
        self._log(f"Reference set: {len(X_reference)} samples, "
                  f"{n_feat} features, {len(self._categorical)} categorical.")

    # ── Main run ─────────────────────────────────────────────────────────────

    def _run(self, X_current):
        """
        Run drift detection comparing X_current to the registered reference.

        Returns result envelope with findings:
          per_feature    : [{feature, ks_pvalue, psi, status}]
          drifted        : [feature names flagged as DRIFT]
          warned         : [feature names flagged as WARN]
          domain_drift   : domain classifier AUC (0.5=no drift, 1.0=complete drift)
          overall_psi    : mean PSI across features
        """
        if not _MODULES_LOADED:
            return self._result(
                {"error": f"Phase 3 drift modules not importable: {_IMPORT_ERROR}"},
                severity="CRITICAL"
            )
        if self._reference is None:
            return self._result(
                {"error": "No reference data. Call set_reference() first."},
                severity="HIGH"
            )

        n_feat = len(self._feature_names)
        per_feature = []
        drifted, warned = [], []

        for j, fname in enumerate(self._feature_names):
            ref_col = [row[j] for row in self._reference
                       if isinstance(row[j], (int, float))]
            cur_col = [row[j] for row in X_current
                       if isinstance(row[j], (int, float))]
            if not ref_col or not cur_col:
                continue

            if j in self._categorical:
                # Chi-squared for categorical
                from collections import Counter
                ref_cats = [str(v) for v in ref_col]
                cur_cats = [str(v) for v in cur_col]
                test_result = chi2_test_categorical(ref_cats, cur_cats, fname)
                feature_entry = {
                    "feature":   fname,
                    "type":      "categorical",
                    "pvalue":    round(test_result.get("p_value", 1.0), 5),
                    "chi2_stat": round(test_result.get("chi2_statistic", 0.0), 4),
                    "psi":       None,
                    "significant_drift": test_result.get("significant_drift", False),
                }
            else:
                # KS test
                ks_result = ks_test(ref_col, cur_col, fname, alpha=self.ks_alpha)
                # PSI
                psi_val = psi(ref_col, cur_col)
                psi_score = psi_val.get("psi", 0.0)
                feature_entry = {
                    "feature":  fname,
                    "type":     "numeric",
                    "ks_stat":  round(ks_result.get("ks_statistic", 0.0), 4),
                    "pvalue":   round(ks_result.get("p_value", 1.0), 5),
                    "psi":      round(psi_score, 4),
                    "significant_drift": ks_result.get("significant_drift", False),
                }

            # Status classification
            p = feature_entry["pvalue"]
            psi_v = feature_entry.get("psi") or 0.0
            if p < self.ks_alpha or psi_v >= self.psi_thresholds["WARN"]:
                if psi_v >= self.psi_thresholds["WARN"] or p < 0.001:
                    feature_entry["status"] = "DRIFT"
                    drifted.append(fname)
                else:
                    feature_entry["status"] = "WARN"
                    warned.append(fname)
            else:
                feature_entry["status"] = "STABLE"

            per_feature.append(feature_entry)
            self._log(f"  {fname}: {feature_entry['status']} "
                      f"(p={p:.4f}, psi={psi_v:.4f})")

        # Domain classifier (overall drift check)
        domain_auc = 0.5
        try:
            dc = domain_classifier_drift(self._reference, X_current, self._feature_names)
            domain_auc = dc.get("auc", 0.5)
        except Exception as ex:
            self._log(f"Domain classifier failed: {ex}")

        # PSI summary
        psi_vals = [f["psi"] for f in per_feature if f.get("psi") is not None]
        mean_psi = sum(psi_vals) / max(len(psi_vals), 1)

        # Severity
        if len(drifted) > n_feat * 0.5 or domain_auc > 0.85:
            sev = "CRITICAL"
        elif drifted:
            sev = "HIGH"
        elif warned:
            sev = "MEDIUM"
        else:
            sev = "NONE"

        findings = {
            "per_feature":   per_feature,
            "drifted":       drifted,
            "warned":        warned,
            "domain_auc":    round(domain_auc, 4),
            "mean_psi":      round(mean_psi, 4),
            "n_features":    n_feat,
            "n_drifted":     len(drifted),
            "n_warned":      len(warned),
            "alerts": self._build_alerts(drifted, warned, domain_auc, mean_psi),
        }

        return self._result(findings, severity=sev, module_name="DriftEngine")

    def _build_alerts(self, drifted, warned, domain_auc, mean_psi):
        alerts = []
        if drifted:
            alerts.append({
                "level":   "CRITICAL" if len(drifted) > 3 else "HIGH",
                "message": f"{len(drifted)} feature(s) show significant drift: "
                           f"{drifted[:5]}. Model retraining recommended.",
            })
        if warned:
            alerts.append({
                "level":   "MEDIUM",
                "message": f"{len(warned)} feature(s) show borderline drift: "
                           f"{warned[:5]}. Monitor closely.",
            })
        if domain_auc > 0.75:
            alerts.append({
                "level":   "HIGH",
                "message": f"Domain classifier AUC={domain_auc:.3f} indicates "
                           "the production distribution is distinguishable from training.",
            })
        if not alerts:
            alerts.append({"level": "INFO", "message": "No drift detected."})
        return alerts

    def assert_gate(self, result, max_severity="HIGH"):
        sev_order = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        actual = result.get("severity", "NONE")
        if sev_order.get(actual, 0) > sev_order.get(max_severity, 3):
            n_drifted = result["findings"].get("n_drifted", 0)
            raise DriftGateError(
                f"DriftEngine gate FAILED. Severity: {actual}. "
                f"{n_drifted} features drifted. Investigate before serving."
            )
