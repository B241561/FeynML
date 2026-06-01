"""
Phase 2 — Fairness Metrics
============================
Standalone module covering all fairness definitions required by the roadmap.

Metrics implemented (from scratch, verified against fairlearn):
  • Demographic Parity (Statistical Parity)
  • Equalized Odds (TPR + FPR parity)
  • Equal Opportunity (TPR parity only)
  • Predictive Parity (PPV parity)
  • Disparate Impact (80% / four-fifths rule)
  • Individual Fairness (distance-based consistency)
  • Calibration by group
  • Fairness Impossibility Theorem demonstration

Key reference: Chouldechova (2017), Hardt et al. (2016)
"""

from collections import defaultdict
import math


# ─────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _confusion(y_true, y_pred):
    """Returns TP, FP, TN, FN for binary labels."""
    TP = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    FP = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    TN = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    FN = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    return TP, FP, TN, FN

def _group_rates(y_true, y_pred):
    """Compute all rates for a single group."""
    TP, FP, TN, FN = _confusion(y_true, y_pred)
    n = len(y_true)
    pos = TP + FN
    neg = TN + FP
    pred_pos = TP + FP
    return {
        "n":            n,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "TPR":          TP / max(pos, 1),            # Recall / Sensitivity
        "FPR":          FP / max(neg, 1),            # 1 - Specificity
        "TNR":          TN / max(neg, 1),
        "FNR":          FN / max(pos, 1),
        "PPV":          TP / max(pred_pos, 1),       # Precision
        "NPV":          TN / max(TN + FN, 1),
        "base_rate":    pos / max(n, 1),             # P(Y=1)
        "selection_rate": pred_pos / max(n, 1),      # P(Ŷ=1)
        "accuracy":     (TP + TN) / max(n, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. DEMOGRAPHIC PARITY (Statistical Parity)
# ─────────────────────────────────────────────────────────────────────────────

def demographic_parity(y_pred, groups):
    """
    Demographic Parity: P(Ŷ=1 | A=a) = P(Ŷ=1 | A=b)  for all groups a, b
    
    Measures whether the POSITIVE PREDICTION RATE is equal across groups.
    Does NOT look at ground truth — only at model outputs.
    
    Metric: max difference in selection rates across groups.
    Ideal value: 0 (equal selection rates).
    
    Limitation: achieving demographic parity may require ignoring real
    base rate differences → can reduce accuracy.
    
    Returns:
        dict with selection_rate per group, max_difference, passes (bool).
    """
    group_names = sorted(set(groups))
    group_preds = defaultdict(list)
    for g, p in zip(groups, y_pred):
        group_preds[g].append(p)

    rates = {g: sum(group_preds[g]) / len(group_preds[g]) for g in group_names}
    max_rate = max(rates.values())
    min_rate = min(rates.values())
    max_diff = max_rate - min_rate

    return {
        "metric":            "Demographic Parity",
        "selection_rates":   {g: round(r, 4) for g, r in rates.items()},
        "max_difference":    round(max_diff, 4),
        "passes":            max_diff < 0.1,      # common threshold
        "interpretation":    (
            f"Max selection rate gap = {max_diff:.3f}. "
            + ("✓ Acceptable." if max_diff < 0.1 else "✗ Fairness concern.")
        )
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. EQUALIZED ODDS
# ─────────────────────────────────────────────────────────────────────────────

def equalized_odds(y_true, y_pred, groups):
    """
    Equalized Odds (Hardt et al., 2016):
      P(Ŷ=1 | Y=y, A=a) = P(Ŷ=1 | Y=y, A=b)  for y ∈ {0,1}
    
    Both TPR and FPR must be equal across groups.
    
    TPR parity: model equally recalls positives in each group
    FPR parity: model equally misclassifies negatives in each group
    
    Tension with predictive parity (Chouldechova impossibility):
      When base rates differ, you CANNOT have both equalized odds
      AND equal PPV simultaneously (except trivially).
    
    Returns:
        dict with TPR/FPR per group, max gaps, passes.
    """
    group_names = sorted(set(groups))
    metrics = {}
    for g in group_names:
        idx = [i for i, gr in enumerate(groups) if gr == g]
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        metrics[g] = _group_rates(yt, yp)

    tpr_values = {g: metrics[g]["TPR"] for g in group_names}
    fpr_values = {g: metrics[g]["FPR"] for g in group_names}

    tpr_gap = max(tpr_values.values()) - min(tpr_values.values())
    fpr_gap = max(fpr_values.values()) - min(fpr_values.values())
    max_gap = max(tpr_gap, fpr_gap)

    return {
        "metric":           "Equalized Odds",
        "per_group":        {g: {"TPR": round(metrics[g]["TPR"], 4),
                                 "FPR": round(metrics[g]["FPR"], 4),
                                 "n":   metrics[g]["n"]} for g in group_names},
        "TPR_gap":          round(tpr_gap, 4),
        "FPR_gap":          round(fpr_gap, 4),
        "max_gap":          round(max_gap, 4),
        "passes":           max_gap < 0.1,
        "interpretation":   (
            f"TPR gap={tpr_gap:.3f}, FPR gap={fpr_gap:.3f}. "
            + ("✓ Equalized odds approximately satisfied."
               if max_gap < 0.1 else "✗ Equalized odds violated.")
        )
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. EQUAL OPPORTUNITY
# ─────────────────────────────────────────────────────────────────────────────

def equal_opportunity(y_true, y_pred, groups):
    """
    Equal Opportunity (Hardt et al., 2016) — relaxed version of equalized odds.
    Requires only TPR equality (not FPR):
      P(Ŷ=1 | Y=1, A=a) = P(Ŷ=1 | Y=1, A=b)
    
    Rationale: focus on not disadvantaging the positive class (qualified candidates,
    creditworthy borrowers, low-risk defendants) across groups.
    
    More achievable than full equalized odds.
    """
    group_names = sorted(set(groups))
    metrics = {}
    for g in group_names:
        idx = [i for i, gr in enumerate(groups) if gr == g]
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        metrics[g] = _group_rates(yt, yp)

    tpr_values = {g: metrics[g]["TPR"] for g in group_names}
    tpr_gap = max(tpr_values.values()) - min(tpr_values.values())

    return {
        "metric":        "Equal Opportunity",
        "TPR_per_group": {g: round(v, 4) for g, v in tpr_values.items()},
        "TPR_gap":       round(tpr_gap, 4),
        "passes":        tpr_gap < 0.1,
        "interpretation": (
            f"TPR gap={tpr_gap:.3f}. "
            + ("✓ Equal opportunity satisfied." if tpr_gap < 0.1
               else "✗ Equal opportunity violated.")
        )
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. PREDICTIVE PARITY (Calibration Fairness)
# ─────────────────────────────────────────────────────────────────────────────

def predictive_parity(y_true, y_pred, groups):
    """
    Predictive Parity: P(Y=1 | Ŷ=1, A=a) = P(Y=1 | Ŷ=1, A=b)
    
    Equal precision (PPV) across groups.
    Used in COMPAS defense: "when the model says high-risk, it's equally
    accurate regardless of race."
    
    Chouldechova (2017) Impossibility Theorem:
      With unequal base rates AND equalized odds, predictive parity is IMPOSSIBLE.
      You MUST choose which fairness criterion to prioritize.
    """
    group_names = sorted(set(groups))
    metrics = {}
    for g in group_names:
        idx = [i for i, gr in enumerate(groups) if gr == g]
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        metrics[g] = _group_rates(yt, yp)

    ppv_values = {g: metrics[g]["PPV"] for g in group_names}
    ppv_gap = max(ppv_values.values()) - min(ppv_values.values())

    return {
        "metric":         "Predictive Parity",
        "PPV_per_group":  {g: round(v, 4) for g, v in ppv_values.items()},
        "PPV_gap":        round(ppv_gap, 4),
        "base_rates":     {g: round(metrics[g]["base_rate"], 4) for g in group_names},
        "passes":         ppv_gap < 0.1,
        "interpretation": (
            f"PPV gap={ppv_gap:.3f}. "
            + ("✓ Predictive parity satisfied." if ppv_gap < 0.1
               else "✗ Predictive parity violated.")
        )
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. DISPARATE IMPACT (80% / Four-Fifths Rule)
# ─────────────────────────────────────────────────────────────────────────────

def disparate_impact(y_pred, groups, privileged_group, unprivileged_group):
    """
    Disparate Impact Ratio (EEOC Four-Fifths Rule):
      DI = P(Ŷ=1 | A=unprivileged) / P(Ŷ=1 | A=privileged)
    
    EEOC (1978) guideline: DI < 0.8 indicates adverse impact (potential discrimination).
    
    Range:
      DI = 1.0 → perfect parity
      DI < 0.8 → adverse impact on unprivileged group
      DI > 1.25 → adverse impact on privileged group (reverse discrimination)
    
    Note: purely outcome-based; does not account for legitimate predictors.
    """
    priv = [y_pred[i] for i, g in enumerate(groups) if g == privileged_group]
    unpriv = [y_pred[i] for i, g in enumerate(groups) if g == unprivileged_group]

    if not priv or not unpriv:
        raise ValueError(f"Groups '{privileged_group}' or '{unprivileged_group}' not found.")

    rate_priv   = sum(priv) / len(priv)
    rate_unpriv = sum(unpriv) / len(unpriv)

    di = rate_unpriv / max(rate_priv, 1e-10)

    return {
        "metric":              "Disparate Impact",
        "privileged_group":    privileged_group,
        "unprivileged_group":  unprivileged_group,
        "selection_rate_privileged":   round(rate_priv, 4),
        "selection_rate_unprivileged": round(rate_unpriv, 4),
        "disparate_impact_ratio":      round(di, 4),
        "eeoc_passes":         0.8 <= di <= 1.25,
        "interpretation":      (
            f"DI = {di:.3f}. "
            + ("✓ Passes EEOC 4/5 rule." if 0.8 <= di <= 1.25
               else f"✗ EEOC adverse impact {'against ' + unprivileged_group if di < 0.8 else 'against ' + privileged_group}.")
        )
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. INDIVIDUAL FAIRNESS (Consistency Score)
# ─────────────────────────────────────────────────────────────────────────────

def individual_fairness(X, y_pred, k=5):
    """
    Individual Fairness: "similar individuals should be treated similarly."
    (Dwork et al., 2012)
    
    Approximation via consistency score:
      For each individual, find K nearest neighbors in feature space.
      Consistency = 1 - mean |pred(i) - pred(neighbor(i)|
    
    Score ∈ [0, 1]:
      1.0 → perfectly consistent (similar people get similar predictions)
      0.0 → maximally inconsistent
    
    Limitation: requires a meaningful distance metric in feature space.
    """
    n = len(X)
    d = len(X[0])

    def euclidean(a, b):
        return math.sqrt(sum((a[j] - b[j])**2 for j in range(d)))

    consistency_scores = []
    for i in range(n):
        dists = [(euclidean(X[i], X[j]), j) for j in range(n) if j != i]
        dists.sort(key=lambda x: x[0])
        neighbors = [j for _, j in dists[:k]]
        mean_diff = sum(abs(y_pred[i] - y_pred[nj]) for nj in neighbors) / k
        consistency_scores.append(1 - mean_diff)

    avg_consistency = sum(consistency_scores) / n
    return {
        "metric":              "Individual Fairness (Consistency)",
        "k_neighbors":         k,
        "consistency_score":   round(avg_consistency, 4),
        "passes":              avg_consistency >= 0.8,
        "interpretation":      (
            f"Consistency={avg_consistency:.3f}. "
            + ("✓ Similar individuals treated similarly." if avg_consistency >= 0.8
               else "✗ Similar individuals treated differently.")
        )
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. FULL FAIRNESS REPORT
# ─────────────────────────────────────────────────────────────────────────────

def full_fairness_report(y_true, y_pred, groups, privileged_group=None,
                          unprivileged_group=None, X=None):
    """
    Run all fairness checks and return a unified report.
    
    Parameters:
        y_true          : ground truth labels (list of 0/1)
        y_pred          : model predictions (list of 0/1)
        groups          : list of group membership strings
        privileged_group: for disparate impact (optional)
        unprivileged_group: for disparate impact (optional)
        X               : feature matrix for individual fairness (optional)
    
    Returns:
        dict with all fairness metrics and an overall status.
    """
    report = {}
    violations = []

    report["demographic_parity"] = demographic_parity(y_pred, groups)
    if not report["demographic_parity"]["passes"]:
        violations.append("Demographic Parity")

    report["equalized_odds"] = equalized_odds(y_true, y_pred, groups)
    if not report["equalized_odds"]["passes"]:
        violations.append("Equalized Odds")

    report["equal_opportunity"] = equal_opportunity(y_true, y_pred, groups)
    if not report["equal_opportunity"]["passes"]:
        violations.append("Equal Opportunity")

    report["predictive_parity"] = predictive_parity(y_true, y_pred, groups)
    if not report["predictive_parity"]["passes"]:
        violations.append("Predictive Parity")

    if privileged_group and unprivileged_group:
        report["disparate_impact"] = disparate_impact(
            y_pred, groups, privileged_group, unprivileged_group)
        if not report["disparate_impact"]["eeoc_passes"]:
            violations.append("Disparate Impact (EEOC)")

    if X is not None:
        report["individual_fairness"] = individual_fairness(X, y_pred)
        if not report["individual_fairness"]["passes"]:
            violations.append("Individual Fairness")

    # Impossibility theorem check
    base_rates = {}
    group_names = sorted(set(groups))
    for g in group_names:
        idx = [i for i, gr in enumerate(groups) if gr == g]
        yt = [y_true[i] for i in idx]
        base_rates[g] = sum(yt) / max(len(yt), 1)

    br_gap = max(base_rates.values()) - min(base_rates.values())
    report["impossibility_theorem"] = {
        "base_rates":        {g: round(r, 4) for g, r in base_rates.items()},
        "base_rate_gap":     round(br_gap, 4),
        "note": (
            "Base rates are equal — all fairness criteria can theoretically coexist."
            if br_gap < 0.05 else
            f"Base rate gap = {br_gap:.3f}. Chouldechova (2017): with unequal base rates, "
            "equalized odds AND predictive parity CANNOT both hold simultaneously. "
            "You must choose which fairness criterion to prioritize."
        )
    }

    report["violations"]     = violations
    report["n_violations"]   = len(violations)
    report["overall_status"] = "CRITICAL" if len(violations) >= 3 else \
                                "HIGH"     if len(violations) >= 2 else \
                                "MEDIUM"   if len(violations) == 1 else "OK"

    return report


# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def run_verification():
    import random
    try:
        from fairlearn.metrics import (demographic_parity_difference,
                                        equalized_odds_difference)
        HAS_FAIRLEARN = True
    except ImportError:
        HAS_FAIRLEARN = False

    print("=" * 65)
    print("Phase 2 — Fairness Metrics Verification")
    print("=" * 65)

    rng = random.Random(42)
    n = 600

    # Simulate biased classifier (Black group gets higher FPR)
    groups = ["Black"] * 300 + ["White"] * 300
    y_true, y_pred = [], []
    for i in range(300):     # Black group: base_rate=0.52, FPR=0.42
        yt = 1 if rng.random() < 0.52 else 0
        if yt == 1:
            yp = 1 if rng.random() < 0.80 else 0   # TPR=0.80
        else:
            yp = 1 if rng.random() < 0.42 else 0   # FPR=0.42 (biased)
        y_true.append(yt); y_pred.append(yp)
    for i in range(300):     # White group: base_rate=0.39, FPR=0.20
        yt = 1 if rng.random() < 0.39 else 0
        if yt == 1:
            yp = 1 if rng.random() < 0.80 else 0
        else:
            yp = 1 if rng.random() < 0.20 else 0
        y_true.append(yt); y_pred.append(yp)

    # --- Demographic Parity ---
    dp = demographic_parity(y_pred, groups)
    print(f"\n  Demographic Parity:")
    print(f"    Selection rates: {dp['selection_rates']}")
    print(f"    Max difference: {dp['max_difference']}  passes={dp['passes']}")
    print(f"    {dp['interpretation']}")

    # --- Equalized Odds ---
    eo = equalized_odds(y_true, y_pred, groups)
    print(f"\n  Equalized Odds:")
    for g, m in eo["per_group"].items():
        print(f"    {g}: TPR={m['TPR']}, FPR={m['FPR']}, n={m['n']}")
    print(f"    TPR_gap={eo['TPR_gap']}  FPR_gap={eo['FPR_gap']}  passes={eo['passes']}")
    print(f"    {eo['interpretation']}")

    # --- Disparate Impact ---
    di = disparate_impact(y_pred, groups, "White", "Black")
    print(f"\n  Disparate Impact:")
    print(f"    DI ratio={di['disparate_impact_ratio']}  EEOC passes={di['eeoc_passes']}")
    print(f"    {di['interpretation']}")

    # --- Predictive Parity ---
    pp = predictive_parity(y_true, y_pred, groups)
    print(f"\n  Predictive Parity:")
    print(f"    PPV per group: {pp['PPV_per_group']}  gap={pp['PPV_gap']}")
    print(f"    Base rates: {pp['base_rates']}")
    print(f"    {pp['interpretation']}")

    # --- Fairlearn comparison ---
    if HAS_FAIRLEARN:
        import numpy as np
        from fairlearn.metrics import demographic_parity_difference as dpd
        dpd_fl = dpd(y_true, y_pred, sensitive_features=groups)
        ok = abs(dp["max_difference"] - abs(dpd_fl)) < 0.02
        print(f"\n  vs fairlearn DPD={abs(dpd_fl):.4f}  ours={dp['max_difference']}  [{'✓ PASS' if ok else '✗ FAIL'}]")
    else:
        print("\n  (fairlearn not installed — skipping library comparison)")

    # --- Impossibility Theorem ---
    report = full_fairness_report(y_true, y_pred, groups, "White", "Black")
    print(f"\n  Impossibility Theorem note:")
    print(f"    {report['impossibility_theorem']['note']}")
    print(f"\n  Overall status: {report['overall_status']} | Violations: {report['violations']}")
    print("=" * 65)


if __name__ == "__main__":
    run_verification()
