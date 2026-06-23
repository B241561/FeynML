"""
Risk Scorer
===========
Provides a single source-of-truth risk scoring engine used by AutoRootCauseEngine,
AIInvestigator and the webapp.

Enhancements:
- Supports per-component weighting so certain findings (e.g., leakage, root cause)
  can contribute more to the overall score.
- Uses percentage-based thresholds derived from the configured weights so the
  final level is stable regardless of weight scale.

Returns a breakdown with per-component severities, weighted numeric score,
percentage score and final level and reasons.
"""

from typing import Dict, Any, Optional, List

DEFAULT_SEVERITY_MAP = {
    "NONE": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}

# Default component weights (can be overridden by instantiating RiskScorer)
DEFAULT_COMPONENT_WEIGHTS = {
    'leakage': 2.0,
    'label_noise': 1.5,
    'drift': 1.0,
    'calibration': 1.0,
    'missing_data': 0.8,
    'fairness': 1.0,
    'root_cause': 2.0,
}

# Percentage-based thresholds (applied to weighted-percent score)
DEFAULT_PERCENT_THRESHOLDS = {
    'LOW': (0.0, 15.0),
    'MEDIUM': (15.0, 40.0),
    'HIGH': (40.0, 70.0),
    'CRITICAL': (70.0, 100.0),
}


class RiskScorer:
    def __init__(self,
                 severity_map: Dict[str, int] = None,
                 component_weights: Dict[str, float] = None,
                 percent_thresholds: Dict[str, tuple] = None):
        self.severity_map = severity_map or DEFAULT_SEVERITY_MAP
        self.component_weights = component_weights or DEFAULT_COMPONENT_WEIGHTS
        self.percent_thresholds = percent_thresholds or DEFAULT_PERCENT_THRESHOLDS

    def _map_severity(self, sev: Optional[str]) -> int:
        if not sev:
            return self.severity_map.get("NONE", 0)
        return self.severity_map.get(sev.upper(), self.severity_map.get("NONE", 0))

    def _normalize_report_severity(self, report: Any) -> str:
        """Extract severity label from a module report envelope if possible.

        Adds extra safeguards for fairness-style reports where metric
        computations can produce misleading defaults (e.g., DI=1.0 due
        to division-by-zero). If the report indicates there are no
        positive labels (all base_rate == 0) or no selections across
        groups (all selection_rate == 0), treat severity as NONE.
        """
        if not report:
            return "NONE"
        if isinstance(report, dict):
            # Quick extract if severity explicitly provided
            sev = report.get('severity') or report.get('status')
            # Detect per-group rates in common report shapes (audit_by_group / fairness)
            per_group_rates = None
            # report may contain per_group_rates at top level
            if isinstance(report.get('per_group_rates'), dict):
                per_group_rates = report.get('per_group_rates')
            # or nested inside 'findings' or inside per-axis structures
            findings = report.get('findings') if 'findings' in report else report
            if isinstance(findings, dict):
                if per_group_rates is None and isinstance(findings.get('per_group_rates'), dict):
                    per_group_rates = findings.get('per_group_rates')
                # If per-axis results exist, aggregate across axes
                if per_group_rates is None and isinstance(report.get('per_axis'), dict):
                    # Collect per_group_rates from first axis that has it
                    for ax_res in report.get('per_axis', {}).values():
                        if isinstance(ax_res, dict) and isinstance(ax_res.get('per_group_rates'), dict):
                            per_group_rates = ax_res.get('per_group_rates')
                            break
            try:
                if per_group_rates and isinstance(per_group_rates, dict):
                    all_zero_base = all(v.get('base_rate', 0) == 0 for v in per_group_rates.values())
                    all_zero_selection = all(v.get('selection_rate', 0) == 0 for v in per_group_rates.values())
                    if all_zero_base or all_zero_selection:
                        return 'NONE'
            except Exception:
                # If inspection fails, fall back to provided severity
                pass

            if isinstance(sev, str):
                return sev.upper()

            # Fallback to findings.severity if present
            if isinstance(findings, dict):
                sev2 = findings.get('severity')
                if isinstance(sev2, str):
                    return sev2.upper()
        return 'NONE'

    def score(self,
              leakage_report: Any = None,
              label_noise_report: Any = None,
              drift_report: Any = None,
              calibration_report: Any = None,
              missing_data_report: Any = None,
              fairness_report: Any = None,
              root_causes: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Compute the weighted aggregated risk score and breakdown."""
        breakdown = {}

        # Extract per-component severities (prefer envelope severity)
        breakdown['leakage'] = self._normalize_report_severity(leakage_report)
        breakdown['label_noise'] = self._normalize_report_severity(label_noise_report)
        breakdown['drift'] = self._normalize_report_severity(drift_report)
        breakdown['calibration'] = self._normalize_report_severity(calibration_report)
        breakdown['missing_data'] = self._normalize_report_severity(missing_data_report)
        breakdown['fairness'] = self._normalize_report_severity(fairness_report)

        # Root cause severity: derive from provided root_causes list (use worst)
        root_sev = 'NONE'
        if root_causes and isinstance(root_causes, list) and len(root_causes) > 0:
            sevs = [r.get('severity', 'NONE') for r in root_causes]
            highest = max(sevs, key=lambda s: self._map_severity(s))
            root_sev = highest.upper()
        breakdown['root_cause'] = root_sev

        # Compute weighted numeric score
        weighted_numeric = 0.0
        max_possible = 0.0
        max_sev = max(self.severity_map.values()) if self.severity_map else 4
        for comp, sev_label in breakdown.items():
            weight = float(self.component_weights.get(comp, 1.0))
            sev_val = float(self._map_severity(sev_label))
            weighted_numeric += sev_val * weight
            max_possible += max_sev * weight

        # Convert to percentage (0-100)
        pct_score = (weighted_numeric / max_possible * 100.0) if max_possible > 0 else 0.0

        # Map pct_score into levels using percent thresholds
        level = 'MEDIUM'
        for lvl, (lo, hi) in self.percent_thresholds.items():
            if lo <= pct_score <= hi:
                level = lvl
                break

        # Build reasons list with component-level details
        reasons = []
        for comp, sev in breakdown.items():
            if sev and sev != 'NONE':
                reasons.append({'component': comp, 'severity': sev, 'weight': self.component_weights.get(comp, 1.0)})

        return {
            'score': round(weighted_numeric, 3),
            'score_pct': round(pct_score, 2),
            'level': level,
            'breakdown': breakdown,
            'reasons': reasons,
            'percent_thresholds': self.percent_thresholds,
            'severity_map': self.severity_map,
            'component_weights': self.component_weights,
        }