"""
Engine — Base Module
=====================
Abstract base class that every engine module inherits from.

Provides:
  • Standardised run() interface
  • Logging with timestamps
  • Structured result envelope
  • Input validation helpers
  • Gate enforcement (raise on severity threshold)
"""

import sys
import math
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# SEVERITY ORDERING
# ─────────────────────────────────────────────────────────────────────────────

SEVERITY_ORDER = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


class GateError(Exception):
    """Raised when a module's quality gate is not met."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# BASE MODULE
# ─────────────────────────────────────────────────────────────────────────────

class BaseModule:
    """
    Abstract base for all ML Failure Engine modules.

    Subclasses must implement:
      _run(self, *args, **kwargs) -> dict

    And call:
      self._result(findings, severity, module_name)

    to wrap the output in a standard envelope.
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self._log_lines = []

    # ── Logging ──────────────────────────────────────────────────────────────

    def _log(self, msg, level="INFO"):
        ts  = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] [{level}] [{self.__class__.__name__}] {msg}"
        self._log_lines.append(line)
        if self.verbose:
            print(line, file=sys.stderr)

    def _warn(self, msg):
        self._log(msg, level="WARN")

    def _error(self, msg):
        self._log(msg, level="ERROR")

    # ── Result envelope ──────────────────────────────────────────────────────

    def _result(self, findings, severity="NONE", module_name=None):
        """
        Wrap findings in a standard envelope consumed by the report engine.

        Parameters
        ----------
        findings    : dict  — module-specific output
        severity    : str   — NONE / LOW / MEDIUM / HIGH / CRITICAL
        module_name : str   — defaults to class name

        Returns
        -------
        dict:
          module     : str
          timestamp  : str
          severity   : str
          passed     : bool  (severity <= LOW)
          findings   : dict
          log        : list[str]
        """
        return {
            "module":    module_name or self.__class__.__name__,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "severity":  severity,
            "passed":    SEVERITY_ORDER.get(severity, 0) <= 1,
            "findings":  findings,
            "log":       list(self._log_lines),
        }

    # ── Input validation ─────────────────────────────────────────────────────

    def _check_lengths(self, **arrays):
        """Assert all arrays have the same length. Raises ValueError if not."""
        lengths = {name: len(arr) for name, arr in arrays.items()}
        unique  = set(lengths.values())
        if len(unique) > 1:
            raise ValueError(
                f"Array length mismatch: {lengths}. "
                f"All inputs must have the same number of samples."
            )
        return lengths[next(iter(lengths))]

    def _check_binary(self, y, name="y"):
        """Assert array contains only 0 and 1."""
        unique = set(y)
        if not unique.issubset({0, 1}):
            raise ValueError(
                f"'{name}' must be binary (0/1). Found values: {unique - {0, 1}}"
            )

    def _check_probs(self, p, name="y_prob"):
        """Assert array values are in [0, 1]."""
        if any(v < 0 or v > 1 for v in p):
            lo = min(p)
            hi = max(p)
            raise ValueError(
                f"'{name}' must be in [0, 1]. Got range [{lo:.4f}, {hi:.4f}]."
            )

    def _check_min_samples(self, n, minimum=10, context=""):
        """Warn if n is below the minimum recommended sample count."""
        if n < minimum:
            self._warn(
                f"Only {n} samples {context}— results may be unreliable "
                f"(recommended ≥ {minimum})."
            )

    # ── Gate enforcement ─────────────────────────────────────────────────────

    def assert_gate(self, result, max_severity="LOW"):
        """
        Raise GateError if result severity exceeds max_severity.

        Parameters
        ----------
        result       : dict from _result()
        max_severity : str  maximum acceptable severity level

        Raises
        ------
        GateError with details about the violation.
        """
        actual_sev  = result.get("severity", "NONE")
        max_level   = SEVERITY_ORDER.get(max_severity.upper(), 1)
        actual_level = SEVERITY_ORDER.get(actual_sev.upper(), 0)

        if actual_level > max_level:
            module = result.get("module", "unknown")
            raise GateError(
                f"[{module}] Quality gate FAILED. "
                f"Severity: {actual_sev} > allowed max: {max_severity}.\n"
                f"Findings: {result.get('findings', {})}"
            )

    # ── Public interface ─────────────────────────────────────────────────────

    def run(self, *args, **kwargs):
        """
        Public entry point. Calls _run() and returns the result envelope.
        Subclasses implement _run(), not run().
        """
        self._log_lines.clear()
        self._log(f"Starting {self.__class__.__name__}")
        result = self._run(*args, **kwargs)
        self._log(f"Done — severity: {result.get('severity', 'NONE')}")
        return result

    def _run(self, *args, **kwargs):
        """Override in subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _run()"
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(verbose={self.verbose})"


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS (shared across modules)
# ─────────────────────────────────────────────────────────────────────────────

def highest_severity(*severities):
    """Return the most severe label from a collection."""
    return max(severities, key=lambda s: SEVERITY_ORDER.get(s, 0))


def format_table(headers, rows, col_width=14):
    """Simple ASCII table formatter for console output."""
    lines = []
    header_str = "  " + "".join(f"{h:<{col_width}}" for h in headers)
    lines.append(header_str)
    lines.append("  " + "-" * (col_width * len(headers)))
    for row in rows:
        lines.append("  " + "".join(f"{str(v):<{col_width}}" for v in row))
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    class DemoModule(BaseModule):
        """Minimal subclass to demonstrate the base."""
        def _run(self, y_true, y_pred):
            n = self._check_lengths(y_true=y_true, y_pred=y_pred)
            self._check_binary(y_true, "y_true")
            self._check_binary(y_pred, "y_pred")
            self._check_min_samples(n, minimum=50)
            acc = sum(a == b for a, b in zip(y_true, y_pred)) / n
            sev = "LOW" if acc > 0.7 else "HIGH"
            return self._result({"accuracy": round(acc, 4), "n": n}, severity=sev)

    import random
    random.seed(0)
    n = 100
    y_t = [random.randint(0, 1) for _ in range(n)]
    y_p = [random.randint(0, 1) for _ in range(n)]

    mod = DemoModule(verbose=True)
    result = mod.run(y_t, y_p)
    print(f"\nResult envelope:")
    for k, v in result.items():
        if k != "log":
            print(f"  {k}: {v}")

    try:
        mod.assert_gate(result, max_severity="NONE")
    except GateError as e:
        print(f"\nGate correctly raised: {e}")

    print("\n✓ BaseModule OK")
