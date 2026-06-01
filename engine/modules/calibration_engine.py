"""
Engine Module — Calibration Engine
=====================================
Production wrapper around scratch/phase2/calibration.py.

Responsibilities:
  • Auto-select best calibration method (Platt / Temperature / Isotonic)
  • Compare calibration before and after correction
  • Per-group calibration audit (fairness-aware)
  • Enforce calibration quality gates
  • Return structured findings for the report engine

Usage:
    from engine.modules.calibration_engine import CalibrationEngine

    ce = CalibrationEngine()
    result = ce.auto_calibrate(y_train, scores_train, y_test, scores_test)
    ce.assert_gate(result, max_ece=0.05)
"""

import sys
import os
import math

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "scratch", "phase2"))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from calibration import (
        calibration_summary, expected_calibration_error,
        maximum_calibration_error, brier_score,
        platt_scaling, temperature_scaling, isotonic_regression_calibration,
        compare_calibration, calibration_by_group, reliability_curve,
    )
    _MODULES_LOADED = True
except ImportError as e:
    _MODULES_LOADED = False
    _IMPORT_ERROR   = str(e)


class CalibrationGateError(Exception):
    """Raised when a model fails the calibration quality gate."""
    pass


class CalibrationEngine:
    """
    End-to-end calibration analysis and correction for the ML Failure Engine.

    Methods
    -------
    evaluate(y_true, y_prob)                    — summarize calibration quality
    auto_calibrate(y_cal, s_cal, y_test, s_test) — fit + apply best method
    compare(y_true, raw_probs, cal_probs)        — before vs after
    audit_by_group(y_true, y_prob, groups)       — per-demographic calibration
    assert_gate(report, max_ece)                 — raise if ECE too high
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins   = n_bins
        self._results = {}

    # ── EVALUATE ─────────────────────────────────────────────────────────

    def evaluate(self, y_true, y_prob, model_name: str = "model") -> dict:
        """
        Full calibration summary: ECE, MCE, Brier, reliability curve, rating.
        """
        if not _MODULES_LOADED:
            return {"error": _IMPORT_ERROR}
        result = calibration_summary(y_true, y_prob, self.n_bins, model_name)
        self._results["last_eval"] = result
        return result

    # ── AUTO-CALIBRATE ────────────────────────────────────────────────────

    def auto_calibrate(self, y_cal, scores_cal, y_test, scores_test,
                        methods=None) -> dict:
        """
        Fit all three calibration methods on (y_cal, scores_cal),
        apply to scores_test, and return the one with lowest ECE on y_test.

        Parameters
        ----------
        y_cal        : calibration-set labels
        scores_cal   : raw model scores for calibration set
        y_test       : test-set labels
        scores_test  : raw model scores for test set
        methods      : list of method names to try; default all three

        Returns
        -------
        dict with best method, calibrated_probs, ECE before/after, all results.
        """
        if not _MODULES_LOADED:
            return {"error": _IMPORT_ERROR}

        if methods is None:
            methods = ["platt", "temperature", "isotonic"]

        ece_before = expected_calibration_error(y_test, scores_test, self.n_bins)
        results    = {"raw_ece": ece_before, "methods": {}}
        best_ece   = ece_before
        best_name  = "raw"
        best_probs = scores_test

        # ── Platt scaling ────────────────────────────────────────────
        if "platt" in methods:
            try:
                platt = platt_scaling(y_cal, scores_cal, scores_test)
                cal_p = platt["calibrated_probs"]
                ece_p = expected_calibration_error(y_test, cal_p, self.n_bins)
                results["methods"]["platt"] = {
                    "ece":  ece_p, "params": {"A": platt["A"], "B": platt["B"]},
                    "calibrated_probs": cal_p,
                }
                if ece_p < best_ece:
                    best_ece, best_name, best_probs = ece_p, "platt", cal_p
            except Exception as ex:
                results["methods"]["platt"] = {"error": str(ex)}

        # ── Temperature scaling ──────────────────────────────────────
        if "temperature" in methods:
            try:
                eps    = 1e-6
                logits = [math.log(max(p, eps) / max(1 - p, eps)) for p in scores_cal]
                temp   = temperature_scaling(y_cal, logits)
                T      = temp["T"]
                cal_t  = [
                    1.0 / (1.0 + math.exp(-math.log(max(p, eps) / max(1-p, eps)) / T))
                    for p in scores_test
                ]
                ece_t  = expected_calibration_error(y_test, cal_t, self.n_bins)
                results["methods"]["temperature"] = {
                    "ece": ece_t, "params": {"T": T},
                    "calibrated_probs": cal_t,
                }
                if ece_t < best_ece:
                    best_ece, best_name, best_probs = ece_t, "temperature", cal_t
            except Exception as ex:
                results["methods"]["temperature"] = {"error": str(ex)}

        # ── Isotonic regression ──────────────────────────────────────
        if "isotonic" in methods:
            try:
                iso    = isotonic_regression_calibration(y_cal, scores_cal)
                # Build monotone lookup from calibration set scores → calibrated
                sorted_pairs = sorted(zip(scores_cal, iso["calibrated_probs"]))
                cal_scores_sorted = [p for p, _ in sorted_pairs]
                cal_vals_sorted   = [v for _, v in sorted_pairs]

                def _iso_predict(p):
                    # Binary search for nearest calibration point
                    lo, hi = 0, len(cal_scores_sorted) - 1
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if cal_scores_sorted[mid] < p:
                            lo = mid + 1
                        else:
                            hi = mid
                    return cal_vals_sorted[lo]

                cal_i = [_iso_predict(p) for p in scores_test]
                ece_i = expected_calibration_error(y_test, cal_i, self.n_bins)
                results["methods"]["isotonic"] = {
                    "ece": ece_i, "params": {"n_blocks": iso["n_blocks"]},
                    "calibrated_probs": cal_i,
                }
                if ece_i < best_ece:
                    best_ece, best_name, best_probs = ece_i, "isotonic", cal_i
            except Exception as ex:
                results["methods"]["isotonic"] = {"error": str(ex)}

        improvement = round((ece_before - best_ece) / max(ece_before, 1e-8) * 100, 2)

        results.update({
            "best_method":        best_name,
            "best_ece":           round(best_ece, 6),
            "ece_improvement_pct": improvement,
            "calibrated_probs":   best_probs,
            "interpretation": (
                f"Best method: {best_name}. ECE improved from "
                f"{ece_before:.4f} → {best_ece:.4f} ({improvement:.1f}%)."
                if best_name != "raw" else
                f"No calibration method improved ECE below raw ({ece_before:.4f})."
            )
        })

        self._results["auto_calibrate"] = results
        return results

    # ── COMPARE ──────────────────────────────────────────────────────────

    def compare(self, y_true, raw_probs, cal_probs,
                 raw_name="raw", cal_name="calibrated") -> dict:
        if not _MODULES_LOADED:
            return {"error": _IMPORT_ERROR}
        return compare_calibration(y_true, raw_probs, cal_probs, raw_name, cal_name)

    # ── GROUP CALIBRATION ─────────────────────────────────────────────────

    def audit_by_group(self, y_true, y_prob, groups) -> dict:
        if not _MODULES_LOADED:
            return {"error": _IMPORT_ERROR}
        return calibration_by_group(y_true, y_prob, groups, self.n_bins)

    # ── DEPLOYMENT GATE ───────────────────────────────────────────────────

    def assert_gate(self, report: dict, max_ece: float = 0.05):
        """
        Block deployment if ECE exceeds `max_ece`.
        Looks for "ece" key in the report dict.
        """
        ece = report.get("ece") or report.get("best_ece", 0.0)
        if ece is None:
            return True
        if ece > max_ece:
            raise CalibrationGateError(
                f"CALIBRATION GATE FAILED: ECE={ece:.4f} exceeds "
                f"max allowed {max_ece:.4f}. "
                "Apply Platt / Temperature / Isotonic calibration before deployment."
            )
        return True

    # ── RELIABILITY PLOT DATA ─────────────────────────────────────────────

    def reliability_plot_data(self, y_true, y_prob) -> dict:
        """
        Return data ready for plotting a reliability diagram.
        """
        if not _MODULES_LOADED:
            return {"error": _IMPORT_ERROR}
        curve = reliability_curve(y_true, y_prob, self.n_bins)
        bins  = [(lo + hi) / 2 for lo, hi in curve["bin_edges"]]
        return {
            "bin_centers":       bins,
            "mean_predicted":    curve["mean_predicted"],
            "fraction_positive": curve["fraction_positive"],
            "bin_counts":        curve["bin_counts"],
            "ece":               expected_calibration_error(y_true, y_prob, self.n_bins),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    if not _MODULES_LOADED:
        print(f"Cannot run smoke test: {_IMPORT_ERROR}")
    else:
        rng = random.Random(5)
        n   = 600
        y_t = [rng.randint(0, 1) for _ in range(n)]
        raw = [min(0.999, max(0.001, rng.gauss(0.7 if yt else 0.3, 0.12))) for yt in y_t]
        split = n // 2

        ce     = CalibrationEngine()
        report = ce.evaluate(y_t, raw, "DemoModel")
        print(f"ECE: {report['ece']}  Rating: {report['rating']}")

        ac = ce.auto_calibrate(y_t[:split], raw[:split], y_t[split:], raw[split:])
        print(f"Auto-calibrate: {ac['interpretation']}")

        try:
            ce.assert_gate(ac, max_ece=0.10)
            print("Calibration gate: PASSED")
        except CalibrationGateError as e:
            print(f"Calibration gate: {str(e)[:80]}")

        print("CalibrationEngine module OK.")
