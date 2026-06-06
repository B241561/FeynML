"""
Engine Module — Slicer Engine
================================
Production wrapper around scratch/phase3/slice_finder.py.

Responsibilities:
  • Automatic feature discretization for numeric columns
  • Run SliceFinder (lattice search) to find problematic data slices
  • Rank slices by effect size and statistical significance
  • Emit structured findings per slice, ready for the report engine
  • Enforce slice gate: block if CRITICAL slices found

Reference: Chung et al. (2019) "Automated Data Slicing for Model Validation"
           Algorithm 1 (Lattice Searching) — arXiv 1807.06068v3

Usage:
    from engine.modules.slicer_engine import SlicerEngine

    se = SlicerEngine(k=5, effect_size_threshold=0.2)
    result = se.run(y_true, y_pred, X_val, feature_names)
    se.assert_gate(result, max_severity="HIGH")
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
    from slice_finder import (
        SliceFinder, discretize_numeric_features,
        accuracy_loss_per_sample, log_loss_per_sample,
        squared_error_per_sample,
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
                import sys; print(f"[SlicerEngine] {msg}", file=sys.stderr)
        def _result(self, findings, severity="NONE", module_name=None):
            return {"module": module_name or "SlicerEngine",
                    "severity": severity, "passed": severity in ("NONE","LOW"),
                    "findings": findings, "log": []}
        def assert_gate(self, result, **kwargs): pass
        def run(self, *a, **kw): return self._run(*a, **kw)

    class GateError(Exception): pass
    _BASE_LOADED = False


# Effect size thresholds (Cohen's rule of thumb, from §2.3 of paper)
EFFECT_SIZE_LEVELS = {
    "small":     0.2,
    "medium":    0.5,
    "large":     0.8,
    "very_large": 1.3,
}


class SlicerGateError(GateError):
    pass


def _matrix_to_data_rows(X, feature_names):
    """Convert list[list] feature matrix to list[dict] for slice_finder."""
    return [
        {feature_names[j]: X[i][j] for j in range(len(feature_names))}
        for i in range(len(X))
    ]


def _prepare_slice_rows(X, feature_names, n_bins):
    """
    Build categorical string rows + feature column list for SliceFinder.

    Numeric columns are discretized via scratch discretize_numeric_features
    (expects list[dict]); slice predicates use '{name}_binned' columns.
    """
    n = len(X)
    data_rows = _matrix_to_data_rows(X, feature_names)

    numeric_names = []
    for j, fname in enumerate(feature_names):
        col = [X[i][j] for i in range(n)]
        if col and all(isinstance(v, (int, float)) for v in col):
            numeric_names.append(fname)

    if numeric_names:
        data_rows, _bin_maps = discretize_numeric_features(
            data_rows, numeric_names, n_bins=n_bins
        )
        feature_cols = [
            f"{fname}_binned" if fname in numeric_names else fname
            for fname in feature_names
        ]
    else:
        feature_cols = list(feature_names)

    for row in data_rows:
        for fname in feature_cols:
            if fname in row:
                row[fname] = str(row[fname])

    return data_rows, feature_cols


def _counterpart_mean_loss(slice_loss, slice_size, overall_loss, n_samples):
    """Mean loss on samples outside the slice (from dataset overall mean)."""
    if slice_size <= 0 or slice_size >= n_samples:
        return overall_loss
    total = overall_loss * n_samples
    rest_size = n_samples - slice_size
    return (total - slice_loss * slice_size) / rest_size


def _format_slice_entry(raw, n_samples):
    """Map find_slices() entry to engine findings dict."""
    sl = raw.get("slice")
    predicate = dict(sl.predicates) if sl is not None else {}
    slice_loss = raw.get("slice_loss", 0.0)
    overall_loss = raw.get("overall_loss", 0.0)
    slice_size = raw.get("slice_size", 0)
    return {
        "predicate":   predicate,
        "description": raw.get("description", str(sl) if sl else ""),
        "size":        slice_size,
        "effect_size": round(raw.get("effect_size", 0.0), 4),
        "slice_loss":  round(slice_loss, 5),
        "rest_loss":   round(
            _counterpart_mean_loss(slice_loss, slice_size, overall_loss, n_samples), 5
        ),
        "p_value":     round(raw.get("p_value", 1.0), 5),
    }


class SlicerEngine(BaseModule):
    """
    Automated data slicing for model validation.

    Finds the top-k data subsets (slices) where the model
    underperforms relative to its counterpart (rest of the data).

    Slices are defined as feature-value predicates:
      e.g. {gender=Female, age=[40,50)} → loss=0.55 (effect_size=0.82)
    """

    def __init__(self, k=5, effect_size_threshold=0.20,
                 alpha=0.05, n_bins=4, verbose=True):
        """
        k                    : number of top slices to find
        effect_size_threshold: minimum Cohen's d to report (0.2=small)
        alpha                : significance level for Welch's t-test
        n_bins               : bins for discretizing numeric features
        """
        super().__init__(verbose=verbose)
        self.k                    = k
        self.effect_size_threshold = effect_size_threshold
        self.alpha                = alpha
        self.n_bins               = n_bins

    # ── Main run ─────────────────────────────────────────────────────────────

    def _run(self, y_true, y_pred, X, feature_names=None,
             task="classification", y_prob=None):
        """
        Find problematic data slices.

        Parameters
        ----------
        y_true        : list[int/float]
        y_pred        : list[int/float]
        X             : list[list]  validation feature matrix
        feature_names : list[str]
        task          : "classification" | "regression"
        y_prob        : list[float]  probabilities (for log-loss, optional)

        Returns result envelope with findings:
          slices         : list of problematic slice dicts
          n_slices_found : int
          worst_slice    : slice with largest effect size
          overall_loss   : float  mean loss on full dataset
        """
        if not _MODULES_LOADED:
            return self._result(
                {"error": f"SliceFinder not importable: {_IMPORT_ERROR}"},
                severity="CRITICAL"
            )

        n = len(y_true)
        n_features = len(X[0]) if X else 0
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(n_features)]

        self._log(f"Slicing: n={n}, k={self.k}, effect_threshold={self.effect_size_threshold}")

        # ── Compute per-sample losses ─────────────────────────────────────
        if task == "regression":
            losses = squared_error_per_sample(y_true, y_pred)
        elif y_prob is not None:
            losses = log_loss_per_sample(y_true, y_prob)
        else:
            losses = accuracy_loss_per_sample(y_true, y_pred)

        overall_loss = sum(losses) / max(len(losses), 1)
        self._log(f"Overall {task} loss: {overall_loss:.4f}")

        # ── Discretize & build categorical rows ───────────────────────────
        try:
            data_rows, feature_cols = _prepare_slice_rows(
                X, feature_names, self.n_bins
            )
        except Exception as ex:
            self._log(f"Discretization failed: {ex}. Using stringified raw X.")
            data_rows = _matrix_to_data_rows(X, feature_names)
            feature_cols = list(feature_names)
            for row in data_rows:
                for fname in feature_cols:
                    row[fname] = str(row[fname])

        # ── Run SliceFinder ───────────────────────────────────────────────
        try:
            sf = SliceFinder(data_rows, losses, feature_cols)
            slice_result = sf.find_slices(
                k=self.k,
                effect_threshold=self.effect_size_threshold,
                alpha=self.alpha,
            )
            slices_raw = slice_result.get("slices", [])
        except Exception as ex:
            self._log(f"SliceFinder failed: {ex}")
            return self._result(
                {"error": str(ex), "overall_loss": round(overall_loss, 5)},
                severity="HIGH"
            )

        # ── Format results ────────────────────────────────────────────────
        slices_out = []
        for s in slices_raw:
            entry = _format_slice_entry(s, n)
            entry["severity"] = self._slice_severity(entry["effect_size"])
            slices_out.append(entry)

        # Sort by effect size
        slices_out.sort(key=lambda s: s["effect_size"], reverse=True)
        worst = slices_out[0] if slices_out else None

        # ── Overall severity ─────────────────────────────────────────────
        critical_slices = [s for s in slices_out if s["severity"] == "CRITICAL"]
        high_slices     = [s for s in slices_out if s["severity"] == "HIGH"]
        if critical_slices:
            sev = "CRITICAL"
        elif high_slices:
            sev = "HIGH"
        elif slices_out:
            sev = "MEDIUM"
        else:
            sev = "NONE"

        findings = {
            "slices":           slices_out,
            "n_slices_found":   len(slices_out),
            "worst_slice":      worst,
            "overall_loss":     round(overall_loss, 5),
            "task":             task,
            "n_samples":        n,
            "recommendations":  self._build_recs(slices_out),
        }

        worst_eff = worst["effect_size"] if worst else 0.0
        self._log(f"Found {len(slices_out)} slices. Worst effect={worst_eff:.3f}")
        return self._result(findings, severity=sev, module_name="SlicerEngine")

    def _slice_severity(self, effect_size):
        if effect_size >= EFFECT_SIZE_LEVELS["very_large"]:
            return "CRITICAL"
        elif effect_size >= EFFECT_SIZE_LEVELS["large"]:
            return "HIGH"
        elif effect_size >= EFFECT_SIZE_LEVELS["medium"]:
            return "MEDIUM"
        elif effect_size >= EFFECT_SIZE_LEVELS["small"]:
            return "LOW"
        return "NONE"

    def _build_recs(self, slices):
        if not slices:
            return ["No problematic slices found — model is uniformly calibrated."]
        recs = []
        for s in slices[:3]:
            recs.append(
                f"Slice '{s['description']}': loss={s['slice_loss']:.3f} "
                f"(effect={s['effect_size']:.2f}, n={s['size']}). "
                "Investigate data quality and representation for this subgroup."
            )
        return recs

    def assert_gate(self, result, max_severity="HIGH"):
        sev_order = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        actual = result.get("severity", "NONE")
        if sev_order.get(actual, 0) > sev_order.get(max_severity, 3):
            n = result["findings"].get("n_slices_found", 0)
            worst = result["findings"].get("worst_slice", {})
            raise SlicerGateError(
                f"SlicerEngine gate FAILED. Severity: {actual}. "
                f"{n} problematic slices found. "
                f"Worst: '{worst.get('description', 'N/A')}' "
                f"effect={worst.get('effect_size', 0):.2f}"
            )
