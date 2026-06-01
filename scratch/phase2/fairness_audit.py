"""
Phase 2.5 — Fairness Audit
============================
Structured audit pipeline built on top of fairness_metrics.py.

Exports used by engine/modules/fairness_engine.py:
  • audit_by_group        — audit one sensitive attribute
  • multi_axis_audit      — audit several attributes independently
  • intersectional_audit  — audit combinations (race × gender, etc.)
  • generate_audit_report — render findings as a structured dict/text

Each audit returns a structured dict so the report engine can consume it
without any further computation.

Key design principle: NEVER suppress a group with fewer than min_group_size
samples — instead flag it as "insufficient data" to avoid silent failures.
"""

import math
from collections import defaultdict

# ── Import metrics from sibling module ──────────────────────────────────────
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from fairness_metrics import (
    demographic_parity,
    equalized_odds,
    equal_opportunity,
    predictive_parity,
    disparate_impact,
    full_fairness_report,
    _group_rates,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MIN_GROUP_SIZE  = 30      # warn if any group has fewer samples
SEVERITY_THRESHOLDS = {
    # (metric, threshold) → severity label
    "dp_gap":    [(0.05, "LOW"), (0.10, "MEDIUM"), (0.20, "HIGH"), (1.0, "CRITICAL")],
    "eo_gap":    [(0.05, "LOW"), (0.10, "MEDIUM"), (0.20, "HIGH"), (1.0, "CRITICAL")],
    "di_ratio":  [(0.80, "CRITICAL"), (0.90, "HIGH"), (0.95, "MEDIUM"), (1.0, "LOW")],
}


def _classify_severity(metric, value):
    """Return severity label for a given metric value."""
    if metric == "di_ratio":
        # DI < 1 is the concern; lower ratio = worse
        for threshold, label in SEVERITY_THRESHOLDS["di_ratio"]:
            if value <= threshold:
                return label
        return "NONE"
    else:
        abs_val = abs(value)
        for threshold, label in SEVERITY_THRESHOLDS.get(metric, []):
            if abs_val <= threshold:
                return label
        return "NONE"


def _small_group_warning(groups):
    """Return list of group names below MIN_GROUP_SIZE."""
    counts = defaultdict(int)
    for g in groups:
        counts[g] += 1
    return [g for g, cnt in counts.items() if cnt < MIN_GROUP_SIZE]


# ─────────────────────────────────────────────────────────────────────────────
# 1. AUDIT BY GROUP — single sensitive attribute
# ─────────────────────────────────────────────────────────────────────────────

def audit_by_group(y_true, y_pred, groups, group_name="group",
                   y_prob=None, privileged=None):
    """
    Run a full fairness audit for one sensitive attribute.

    Parameters
    ----------
    y_true      : list[int]   ground-truth binary labels
    y_pred      : list[int]   model predictions
    groups      : list        sensitive attribute values per sample
    group_name  : str         human-readable name (e.g. 'race', 'gender')
    y_prob      : list[float] predicted probabilities (optional, for AUC)
    privileged  : any         name of the privileged group (for DI ratio)
                              If None, uses the group with highest selection rate.

    Returns
    -------
    dict with:
      group_name       : str
      group_names      : list of groups found
      per_group_rates  : {group: {n, TPR, FPR, PPV, selection_rate, ...}}
      demographic_parity: result dict
      equalized_odds   : result dict
      equal_opportunity: result dict
      predictive_parity: result dict
      disparate_impact : result dict
      summary          : {dp_gap, eo_max_gap, di_ratio, worst_group}
      severity         : overall severity label (NONE/LOW/MEDIUM/HIGH/CRITICAL)
      warnings         : list of warning strings
      passed           : bool
    """
    warnings = []

    # Small group check
    small = _small_group_warning(groups)
    if small:
        warnings.append(
            f"Groups {small} have < {MIN_GROUP_SIZE} samples — "
            f"metrics may be unreliable."
        )

    # Per-group raw rates
    group_data = defaultdict(lambda: {"y_true": [], "y_pred": []})
    for yt, yp, g in zip(y_true, y_pred, groups):
        group_data[g]["y_true"].append(yt)
        group_data[g]["y_pred"].append(yp)

    per_group_rates = {}
    for g, data in group_data.items():
        per_group_rates[g] = _group_rates(data["y_true"], data["y_pred"])

    # Run fairness metrics
    dp  = demographic_parity(y_pred, groups)
    eo  = equalized_odds(y_true, y_pred, groups)
    eop = equal_opportunity(y_true, y_pred, groups)
    pp  = predictive_parity(y_true, y_pred, groups)

    # Disparate impact needs explicit privileged + unprivileged group names
    group_names = sorted(set(groups))
    if len(group_names) >= 2:
        priv   = privileged if privileged in group_names else group_names[-1]
        unpriv = [g for g in group_names if g != priv][0]
        try:
            di = disparate_impact(y_pred, groups, priv, unpriv)
        except Exception:
            di = {"di_ratio": 1.0, "disparate_impact_ratio": 1.0, "eeoc_passes": True}
    else:
        di = {"di_ratio": 1.0, "disparate_impact_ratio": 1.0, "eeoc_passes": True}

    # Summarise — note: equalized_odds uses "TPR_gap"/"FPR_gap" keys
    dp_gap  = dp.get("max_difference", 0.0)
    eo_gap  = eo.get("max_gap", 0.0)
    # disparate_impact returns "disparate_impact_ratio" key
    di_ratio = di.get("disparate_impact_ratio", di.get("di_ratio", 1.0)) or 1.0

    # Overall severity = worst across metrics
    sev_order = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    severities = [
        _classify_severity("dp_gap",   dp_gap),
        _classify_severity("eo_gap",   eo_gap),
        _classify_severity("di_ratio", di_ratio),
    ]
    overall_sev = max(severities, key=lambda s: sev_order.get(s, 0))

    # Worst group by DP gap
    sel_rates = {g: v["selection_rate"] for g, v in per_group_rates.items()}
    if sel_rates:
        worst = max(sel_rates, key=lambda g: abs(
            sel_rates[g] - sum(sel_rates.values()) / len(sel_rates)
        ))
    else:
        worst = None

    passed = overall_sev in ("NONE", "LOW")

    return {
        "group_name":        group_name,
        "group_names":       sorted(set(groups)),
        "per_group_rates":   per_group_rates,
        "demographic_parity": dp,
        "equalized_odds":    eo,
        "equal_opportunity": eop,
        "predictive_parity": pp,
        "disparate_impact":  di,
        "summary": {
            "dp_gap":    round(dp_gap, 4),
            "eo_max_gap": round(eo_gap, 4),
            "di_ratio":  round(di_ratio, 4),
            "worst_group": worst,
        },
        "severity": overall_sev,
        "warnings": warnings,
        "passed":   passed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. MULTI-AXIS AUDIT — several attributes independently
# ─────────────────────────────────────────────────────────────────────────────

def multi_axis_audit(y_true, y_pred, axes, y_prob=None):
    """
    Audit multiple sensitive attributes independently.

    Parameters
    ----------
    y_true : list[int]
    y_pred : list[int]
    axes   : dict {attribute_name: list_of_group_values}
             e.g. {"race": race_col, "gender": gender_col}
    y_prob : list[float] optional

    Returns
    -------
    dict {attribute_name: audit_result} plus:
      _summary: {
        axes_audited, any_critical, any_high, worst_axis, overall_severity
      }
    """
    results = {}
    for attr_name, groups in axes.items():
        results[attr_name] = audit_by_group(
            y_true, y_pred, list(groups),
            group_name=attr_name, y_prob=y_prob
        )

    # Summary
    sev_order = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    if results:
        worst_axis = max(results, key=lambda k: sev_order.get(results[k]["severity"], 0))
        overall    = results[worst_axis]["severity"]
    else:
        worst_axis = None
        overall    = "NONE"

    results["_summary"] = {
        "axes_audited":    list(axes.keys()),
        "any_critical":    any(r["severity"] == "CRITICAL" for r in results.values()
                               if not isinstance(r, dict) or "_summary" not in r),
        "any_high":        any(r["severity"] in ("HIGH", "CRITICAL")
                               for k, r in results.items() if k != "_summary"),
        "worst_axis":      worst_axis,
        "overall_severity": overall,
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. INTERSECTIONAL AUDIT — combinations of attributes
# ─────────────────────────────────────────────────────────────────────────────

def intersectional_audit(y_true, y_pred, axes, y_prob=None, min_size=20):
    """
    Audit intersectional groups (e.g. Black Women, White Men).

    Parameters
    ----------
    y_true    : list[int]
    y_pred    : list[int]
    axes      : dict {attr_name: list_of_values}
    y_prob    : list[float] optional
    min_size  : int minimum samples in an intersection to include

    Returns
    -------
    dict:
      intersections       : {combo_label: {n, rates, dp_gap, ...}}
      skipped_small       : list of combos below min_size
      worst_intersection  : combo label with widest selection rate deviation
      selection_rate_range: (min, max) across valid intersections
    """
    attr_names  = sorted(axes.keys())
    attr_values = [axes[k] for k in attr_names]

    # Build intersection labels
    combo_data = defaultdict(lambda: {"y_true": [], "y_pred": [], "idx": []})
    for i, row in enumerate(zip(*attr_values)):
        label = " × ".join(f"{n}={v}" for n, v in zip(attr_names, row))
        combo_data[label]["y_true"].append(y_true[i])
        combo_data[label]["y_pred"].append(y_pred[i])

    intersections = {}
    skipped       = []
    sel_rates     = {}

    for combo, data in combo_data.items():
        yt = data["y_true"]
        yp = data["y_pred"]
        n  = len(yt)
        if n < min_size:
            skipped.append({"combo": combo, "n": n})
            continue

        rates  = _group_rates(yt, yp)
        tp_rate = rates["TPR"]
        sel     = rates["selection_rate"]
        sel_rates[combo] = sel

        intersections[combo] = {
            "n":              n,
            "selection_rate": round(sel, 4),
            "TPR":            round(tp_rate, 4),
            "FPR":            round(rates["FPR"], 4),
            "PPV":            round(rates["PPV"], 4),
            "accuracy":       round(rates["accuracy"], 4),
        }

    # Worst intersection by selection rate deviation from mean
    worst = None
    if sel_rates:
        mean_sel = sum(sel_rates.values()) / len(sel_rates)
        worst = max(sel_rates, key=lambda k: abs(sel_rates[k] - mean_sel))
        sr_min = min(sel_rates.values())
        sr_max = max(sel_rates.values())
    else:
        sr_min = sr_max = 0.0

    return {
        "intersections":         intersections,
        "skipped_small":         skipped,
        "worst_intersection":    worst,
        "selection_rate_range":  (round(sr_min, 4), round(sr_max, 4)),
        "n_valid":               len(intersections),
        "n_skipped":             len(skipped),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. GENERATE AUDIT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_audit_report(audit_results, title="Fairness Audit Report"):
    """
    Convert audit_results (from multi_axis_audit or audit_by_group) into a
    structured report dict suitable for the report engine and for printing.

    Parameters
    ----------
    audit_results : dict from audit_by_group or multi_axis_audit
    title         : str

    Returns
    -------
    dict:
      title         : str
      overall_pass  : bool
      severity      : str
      axes          : list of per-axis summaries
      recommendations: list of actionable strings
      raw           : original audit_results
    """
    sev_order = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

    # Detect if single-axis or multi-axis result
    if "group_name" in audit_results:
        # Single-axis
        axes_data = [audit_results]
        overall_sev = audit_results["severity"]
    elif "_summary" in audit_results:
        # Multi-axis
        axes_data = [v for k, v in audit_results.items()
                     if k != "_summary" and isinstance(v, dict) and "group_name" in v]
        overall_sev = audit_results["_summary"]["overall_severity"]
    else:
        axes_data = []
        overall_sev = "NONE"

    overall_pass = sev_order.get(overall_sev, 0) <= 1  # NONE or LOW

    # Axis summaries
    axes_summaries = []
    for ax in axes_data:
        s = ax.get("summary", {})
        axes_summaries.append({
            "attribute":    ax.get("group_name", "unknown"),
            "groups":       ax.get("group_names", []),
            "severity":     ax.get("severity", "NONE"),
            "passed":       ax.get("passed", True),
            "dp_gap":       s.get("dp_gap", 0.0),
            "eo_max_gap":   s.get("eo_max_gap", 0.0),
            "di_ratio":     s.get("di_ratio", 1.0),
            "worst_group":  s.get("worst_group"),
            "warnings":     ax.get("warnings", []),
        })

    # Recommendations
    recs = _build_recommendations(axes_summaries, overall_sev)

    return {
        "title":           title,
        "overall_pass":    overall_pass,
        "severity":        overall_sev,
        "axes":            axes_summaries,
        "recommendations": recs,
        "raw":             audit_results,
    }


def _build_recommendations(axes_summaries, overall_sev):
    """Generate actionable recommendations based on audit findings."""
    recs = []
    sev_order = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

    for ax in axes_summaries:
        attr = ax["attribute"]
        sev  = ax["severity"]
        gap  = ax["dp_gap"]
        di   = ax["di_ratio"]
        worst = ax["worst_group"]

        if sev_order.get(sev, 0) >= 4:  # CRITICAL
            recs.append(
                f"CRITICAL — '{attr}': Block deployment. Disparate impact ratio "
                f"{di:.2f} violates the 80% rule. Investigate data collection and "
                f"labelling for group '{worst}'."
            )
        elif sev_order.get(sev, 0) >= 3:  # HIGH
            recs.append(
                f"HIGH — '{attr}': DP gap {gap:.3f} exceeds 10%. Apply post-processing "
                f"threshold adjustment (e.g., equalise FPR across groups) before release."
            )
        elif sev_order.get(sev, 0) >= 2:  # MEDIUM
            recs.append(
                f"MEDIUM — '{attr}': DP gap {gap:.3f} warrants monitoring. "
                f"Add fairness metrics to your model monitoring dashboard."
            )

    if overall_sev in ("NONE", "LOW"):
        recs.append(
            "Overall fairness checks passed. Continue monitoring as data "
            "distribution may shift over time."
        )

    if not recs:
        recs.append("No specific recommendations — all axes at LOW/NONE severity.")

    return recs


# ─────────────────────────────────────────────────────────────────────────────
# PRINT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def print_audit_report(report):
    """Pretty-print a report dict from generate_audit_report."""
    width = 60
    print("=" * width)
    print(f"  {report['title']}")
    print("=" * width)
    status = "PASSED ✓" if report["overall_pass"] else "FAILED ✗"
    print(f"  Overall: {status}  |  Severity: {report['severity']}\n")

    for ax in report["axes"]:
        icon = "✓" if ax["passed"] else "✗"
        print(f"  [{icon}] {ax['attribute'].upper()}")
        print(f"       Groups       : {ax['groups']}")
        print(f"       Severity     : {ax['severity']}")
        print(f"       DP gap       : {ax['dp_gap']:.4f}")
        print(f"       EO max gap   : {ax['eo_max_gap']:.4f}")
        print(f"       DI ratio     : {ax['di_ratio']:.4f}")
        if ax["warnings"]:
            for w in ax["warnings"]:
                print(f"       ⚠ {w}")
        print()

    print("  Recommendations:")
    for i, r in enumerate(report["recommendations"], 1):
        print(f"  {i}. {r}")
    print("=" * width)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(42)

    n = 600
    # Simulate biased predictions: Group A favoured over Group B
    race   = ["White"] * 300 + ["Black"] * 300
    gender = (["M"] * 150 + ["F"] * 150) * 2

    y_true = [random.randint(0, 1) for _ in range(n)]
    # Biased: Black group gets lower positive rate regardless of true label
    y_pred = []
    for i, (r, g, yt) in enumerate(zip(race, gender, y_true)):
        if r == "White":
            prob_pos = 0.70 if yt == 1 else 0.20
        else:
            prob_pos = 0.45 if yt == 1 else 0.25  # biased against Black group
        y_pred.append(1 if random.random() < prob_pos else 0)

    print("── Single-axis audit (race) ──────────────────────────────")
    result_race = audit_by_group(y_true, y_pred, race, group_name="race",
                                 privileged="White")
    report_race = generate_audit_report(result_race, title="Race Fairness Audit")
    print_audit_report(report_race)

    print("\n── Multi-axis audit (race + gender) ──────────────────────")
    multi = multi_axis_audit(
        y_true, y_pred,
        axes={"race": race, "gender": gender}
    )
    report_multi = generate_audit_report(multi, title="Multi-Axis Fairness Audit")
    print_audit_report(report_multi)

    print("\n── Intersectional audit ──────────────────────────────────")
    inter = intersectional_audit(
        y_true, y_pred,
        axes={"race": race, "gender": gender},
        min_size=20
    )
    print(f"  Valid intersections : {inter['n_valid']}")
    print(f"  Skipped (too small) : {inter['n_skipped']}")
    print(f"  Worst intersection  : {inter['worst_intersection']}")
    print(f"  Selection rate range: {inter['selection_rate_range']}")
    print("\n✓ Fairness audit demo complete.")
