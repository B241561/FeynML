"""
Engine Module — Fairness Engine
================================
Production wrapper around scratch/phase2/fairness_metrics.py and
scratch/phase2/fairness_audit.py.

Responsibilities:
  • Register sensitive attribute columns with their privileged groups
  • Run the full fairness audit pipeline in one call
  • Apply fairness-aware threshold adjustment (equalize FPR or TPR)
  • Emit structured findings suitable for the report engine
  • Enforce fairness gates (block deployment if severity ≥ threshold)

Usage:
    from engine.modules.fairness_engine import FairnessEngine

    fe = FairnessEngine()
    fe.register_axis("race", race_col, privileged="White", unprivileged="Black")
    fe.register_axis("gender", gender_col)
    report = fe.run(y_true, y_pred, y_prob)
    fe.assert_gate(report, max_severity="HIGH")  # raises if CRITICAL
"""

import sys
import os

# Make scratch modules importable from within the engine
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "scratch", "phase2"))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from fairness_metrics import (
        demographic_parity, equalized_odds, equal_opportunity,
        predictive_parity, disparate_impact, full_fairness_report, _group_rates,
    )
    from fairness_audit import (
        audit_by_group, multi_axis_audit, intersectional_audit,
        generate_audit_report,
    )
    _MODULES_LOADED = True
except ImportError as e:
    _MODULES_LOADED = False
    _IMPORT_ERROR   = str(e)


# ─────────────────────────────────────────────────────────────────────────────
# FAIRNESS GATE EXCEPTION
# ─────────────────────────────────────────────────────────────────────────────

class FairnessGateError(Exception):
    """Raised when a model fails the fairness deployment gate."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# FAIRNESS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class FairnessEngine:
    """
    End-to-end fairness orchestration for the ML Failure Engine.

    Attributes
    ----------
    axes : dict  — {axis_name: {labels, privileged, unprivileged}}
    """

    _SEVERITY_RANK = {"OK": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

    def __init__(self):
        self.axes     = {}
        self._results = {}

    # ── AXIS REGISTRATION ────────────────────────────────────────────────

    def register_axis(self, name: str, labels,
                       privileged=None, unprivileged=None):
        """
        Register a sensitive attribute axis.

        Parameters
        ----------
        name           : str         — axis identifier ("race", "gender", ...)
        labels         : list[str]   — group membership per sample
        privileged     : str | None  — majority/reference group
        unprivileged   : str | None  — minority group for disparate impact
        """
        from collections import Counter
        counts = Counter(labels)
        if privileged is None:
            privileged   = max(counts, key=counts.get)
        if unprivileged is None:
            unprivileged = min(counts, key=counts.get)
        self.axes[name] = {
            "labels":       list(labels),
            "privileged":   privileged,
            "unprivileged": unprivileged,
        }
        return self

    def clear_axes(self):
        self.axes = {}
        return self

    # ── MAIN RUN ─────────────────────────────────────────────────────────

    def run(self, y_true, y_pred, y_prob=None, threshold: float = 0.1) -> dict:
        """
        Run the full fairness audit across all registered axes.

        Parameters
        ----------
        y_true     : list[int]    — ground-truth labels
        y_pred     : list[int]    — hard predictions
        y_prob     : list[float]  — probabilities (optional)
        threshold  : float        — gap threshold for pass/fail

        Returns structured dict with per-axis reports + overall summary.
        """
        if not _MODULES_LOADED:
            return {
                "error": f"Fairness modules not loaded: {_IMPORT_ERROR}",
                "hint":  "Ensure scratch/phase2/fairness_metrics.py is importable."
            }

        if not self.axes:
            return {"warning": "No axes registered. Call register_axis() first."}

        # ── multi-axis audit ──────────────────────────────────────────
        axis_labels = {ax: cfg["labels"] for ax, cfg in self.axes.items()}
        axis_kwargs = {}   # threshold is passed per-axis below

        per_axis = {}
        for ax, cfg in self.axes.items():
            per_axis[ax] = audit_by_group(
                y_true, y_pred,
                groups=cfg["labels"],
                axis_name=ax,
                privileged_group=cfg["privileged"],
                unprivileged_group=cfg["unprivileged"],
                threshold=threshold,
            )

        # ── aggregate summary ─────────────────────────────────────────
        all_violations = []
        all_warnings   = []
        for ax, result in per_axis.items():
            for v in result["violations"]:
                all_violations.append(f"[{ax.upper()}] {v}")
            for w in result["warnings"]:
                all_warnings.append(f"[{ax.upper()}] {w}")

        severities = [r["severity"] for r in per_axis.values()]
        worst = max(severities, key=lambda s: self._SEVERITY_RANK.get(s, 0))

        # ── intersectional (if ≥ 2 axes) ─────────────────────────────
        inter = None
        if len(self.axes) >= 2:
            ax_cols = {ax: cfg["labels"] for ax, cfg in self.axes.items()}
            inter   = intersectional_audit(y_true, y_pred, ax_cols=ax_cols)

        self._results = {
            "axes":           list(self.axes.keys()),
            "per_axis":       per_axis,
            "intersectional": inter,
            "violations":     all_violations,
            "warnings":       all_warnings,
            "severity":       worst,
            "threshold":      threshold,
            "summary": (
                f"Fairness audit across {len(self.axes)} axes. "
                f"Severity: {worst}. "
                f"{len(all_violations)} violations, {len(all_warnings)} warnings."
            )
        }
        return self._results

    # ── THRESHOLD OPTIMIZATION ────────────────────────────────────────────

    def find_fair_threshold(self, y_true, y_prob, groups,
                             criterion: str = "equalized_odds",
                             n_thresholds: int = 50) -> dict:
        """
        Grid-search for the threshold that minimizes the fairness gap
        while preserving as much F1 as possible.

        Parameters
        ----------
        criterion : "equalized_odds" | "demographic_parity" | "equal_opportunity"
        """
        best_thr    = 0.5
        best_gap    = float("inf")
        best_f1     = 0.0
        results_log = []

        for t_idx in range(n_thresholds):
            thr = (t_idx + 1) / (n_thresholds + 1)
            y_pred_t = [1 if p >= thr else 0 for p in y_prob]

            if criterion == "equalized_odds":
                eo  = equalized_odds(y_true, y_pred_t, groups)
                gap = eo["max_gap"]
            elif criterion == "demographic_parity":
                dp  = demographic_parity(y_pred_t, groups)
                gap = dp["max_difference"]
            else:
                eop = equal_opportunity(y_true, y_pred_t, groups)
                gap = eop["TPR_gap"]

            # F1
            TP = sum(1 for a, b in zip(y_true, y_pred_t) if a == 1 and b == 1)
            FP = sum(1 for a, b in zip(y_true, y_pred_t) if a == 0 and b == 1)
            FN = sum(1 for a, b in zip(y_true, y_pred_t) if a == 1 and b == 0)
            prec = TP / (TP + FP) if (TP + FP) > 0 else 0
            rec  = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

            # Pareto-optimal: minimize gap, then maximize F1
            if gap < best_gap or (gap == best_gap and f1 > best_f1):
                best_thr = thr
                best_gap = gap
                best_f1  = f1

            results_log.append({"threshold": round(thr, 3), "gap": round(gap, 4),
                                  "f1": round(f1, 4)})

        return {
            "criterion":      criterion,
            "best_threshold": round(best_thr, 4),
            "best_gap":       round(best_gap, 4),
            "best_f1":        round(best_f1, 4),
            "log":            results_log,
            "interpretation": (
                f"Threshold {best_thr:.3f} minimizes {criterion} gap "
                f"({best_gap:.4f}) with F1={best_f1:.4f}."
            )
        }

    # ── DEPLOYMENT GATE ───────────────────────────────────────────────────

    def assert_gate(self, report: dict, max_severity: str = "HIGH"):
        """
        Block deployment if severity exceeds `max_severity`.

        Raises FairnessGateError with a detailed message.
        """
        actual   = report.get("severity", "OK")
        rank_max = self._SEVERITY_RANK.get(max_severity, 3)
        rank_act = self._SEVERITY_RANK.get(actual, 0)

        if rank_act > rank_max:
            violations = report.get("violations", [])
            raise FairnessGateError(
                f"DEPLOYMENT BLOCKED: Fairness severity {actual} exceeds "
                f"max allowed {max_severity}.\n"
                f"Violations: {violations}\n"
                "Remediate before deploying."
            )
        return True

    # ── TEXT REPORT ───────────────────────────────────────────────────────

    def format_report(self, report: dict = None) -> str:
        """Return a human-readable string report."""
        r = report or self._results
        if not r:
            return "No results. Call run() first."
        lines = [
            "=" * 65,
            "ML FAILURE ENGINE — FAIRNESS REPORT",
            "=" * 65,
            f"  Severity:   {r.get('severity', 'N/A')}",
            f"  Axes:       {', '.join(r.get('axes', []))}",
            f"  Violations: {r.get('violations', [])}",
            f"  Warnings:   {r.get('warnings', [])}",
            "",
        ]
        for ax, result in r.get("per_axis", {}).items():
            lines.append(generate_audit_report(result))
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    if not _MODULES_LOADED:
        print(f"Cannot run smoke test: {_IMPORT_ERROR}")
    else:
        rng = random.Random(99)
        n   = 400
        y_t = [rng.randint(0, 1) for _ in range(n)]
        y_p = [1 if rng.random() < (0.7 if yt else 0.4) else 0 for yt in y_t]
        y_s = [min(0.99, max(0.01, rng.gauss(0.7 if yt else 0.3, 0.1))) for yt in y_t]
        genders = [rng.choice(["M", "F"]) for _ in range(n)]
        races   = [rng.choice(["White", "Black", "Hispanic"]) for _ in range(n)]

        fe = FairnessEngine()
        fe.register_axis("gender", genders)
        fe.register_axis("race", races, privileged="White", unprivileged="Black")

        report = fe.run(y_t, y_p, threshold=0.1)
        print(report["summary"])

        # Threshold optimization
        opt = fe.find_fair_threshold(y_t, y_s, races, criterion="equalized_odds")
        print(f"Fair threshold: {opt['interpretation']}")

        # Gate (should pass since threshold is permissive)
        try:
            fe.assert_gate(report, max_severity="CRITICAL")
            print("Gate: PASSED")
        except FairnessGateError as e:
            print(f"Gate: BLOCKED — {str(e)[:80]}...")

        print("FairnessEngine module OK.")
