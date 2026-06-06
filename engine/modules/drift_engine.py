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


def _safe_pvalue(raw, default=None):
    """
    Coerce scratch test p_value to float, or None if unavailable (e.g. 'scipy required').
    Avoids TypeError from round() on non-numeric values.
    """
    if raw is None:
        return default
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _round_pvalue(raw, default=None, decimals=5):
    p = _safe_pvalue(raw, default=default)
    if p is None:
        return None
    return round(p, decimals)


def _significant_drift(test_result):
    """Normalize scratch keys: shift_detected (KS/PSI/chi2) or legacy significant_drift."""
    if not test_result:
        return False
    if "significant_drift" in test_result and test_result["significant_drift"] is not None:
        return bool(test_result["significant_drift"])
    if "shift_detected" in test_result and test_result["shift_detected"] is not None:
        return bool(test_result["shift_detected"])
    return False


def _chi2_statistic(test_result):
    """Normalize chi2 vs chi2_statistic keys from scratch drift_detection."""
    if "chi2_statistic" in test_result:
        return float(test_result["chi2_statistic"])
    return float(test_result.get("chi2", 0.0))


def _domain_auc(dc):
    """Normalize domain_classifier_drift output (domain_auc or legacy auc)."""
    if not dc or dc.get("error"):
        return 0.5
    for key in ("domain_auc", "auc"):
        val = dc.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return 0.5


def _feature_column(rows, col_idx, categorical=False):
    """Extract one feature column; categoricals include all values as strings."""
    if categorical:
        return [str(row[col_idx]) for row in rows if col_idx < len(row)]
    return [
        row[col_idx] for row in rows
        if col_idx < len(row) and isinstance(row[col_idx], (int, float))
    ]


def _classify_feature_status(pvalue, psi_v, ks_stat, significant, ks_alpha, thresholds):
    """Map normalized test outputs to STABLE / WARN / DRIFT."""
    psi_v = psi_v or 0.0
    ks_v  = ks_stat or 0.0
    
    # 1. DRIFT triggers (Critical/High)
    # Trigger if PSI >= 0.25 (industry standard) or KS >= 0.20 or very low p-value
    if psi_v >= 0.25 or ks_v >= 0.20 or (pvalue is not None and pvalue < 0.001):
        return "DRIFT"
        
    # 2. WARN triggers (Medium)
    # Trigger if PSI >= 0.10 or KS >= 0.10 or p-value < alpha
    sig_p = pvalue is not None and pvalue < ks_alpha
    if sig_p or significant or psi_v >= 0.10 or ks_v >= 0.10:
        return "WARN"
        
    return "STABLE"


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
            is_cat = j in self._categorical
            ref_col = _feature_column(self._reference, j, categorical=is_cat)
            cur_col = _feature_column(X_current, j, categorical=is_cat)
            if not ref_col or not cur_col:
                continue

            if is_cat:
                test_result = chi2_test_categorical(
                    ref_col, cur_col, fname, alpha=self.ks_alpha
                )
                feature_entry = {
                    "feature":   fname,
                    "type":      "categorical",
                    "pvalue":    _round_pvalue(test_result.get("p_value")),
                    "pvalue_unavailable": _safe_pvalue(test_result.get("p_value")) is None,
                    "chi2_stat": round(_chi2_statistic(test_result), 4),
                    "psi":       None,
                    "significant_drift": _significant_drift(test_result),
                }
            else:
                ks_result = ks_test(ref_col, cur_col, fname, alpha=self.ks_alpha)
                psi_val = psi(ref_col, cur_col)
                psi_score = psi_val.get("psi", 0.0)
                feature_entry = {
                    "feature":  fname,
                    "type":     "numeric",
                    "ks_stat":  round(ks_result.get("ks_statistic", 0.0), 4),
                    "pvalue":   _round_pvalue(ks_result.get("p_value"), default=1.0),
                    "pvalue_unavailable": False,
                    "psi":      round(psi_score, 4),
                    "significant_drift": _significant_drift(ks_result),
                }

            p = feature_entry["pvalue"]
            psi_v = feature_entry.get("psi")
            ks_v = feature_entry.get("ks_stat") if feature_entry["type"] == "numeric" else None
            
            feature_entry["status"] = _classify_feature_status(
                p,
                psi_v,
                ks_v,
                feature_entry["significant_drift"],
                self.ks_alpha,
                self.psi_thresholds,
            )
            if feature_entry["status"] == "DRIFT":
                drifted.append(fname)
            elif feature_entry["status"] == "WARN":
                warned.append(fname)

            per_feature.append(feature_entry)
            p_str = f"{p:.4f}" if p is not None else "n/a"
            psi_str = f"{psi_v:.4f}" if psi_v is not None else "n/a"
            self._log(f"  {fname}: {feature_entry['status']} (p={p_str}, psi={psi_str})")

        domain_auc = 0.5
        try:
            dc = domain_classifier_drift(
                self._reference, X_current, self._feature_names
            )
            domain_auc = _domain_auc(dc)
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
            "num_drifted_features": len(drifted),
            "num_warned_features":  len(warned),
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
            n_drifted = result["findings"].get("num_drifted_features", 0)
            raise DriftGateError(
                f"DriftEngine gate FAILED. Severity: {actual}. "
                f"{n_drifted} features drifted. Investigate before serving."
            )
