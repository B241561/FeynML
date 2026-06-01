"""
Unit Tests — Fairness Metrics & Audit
=======================================
Tests scratch/phase2/fairness_metrics.py and
      scratch/phase2/fairness_audit.py

Run:
    python tests/test_fairness.py
"""

import sys
import os
import unittest
import random

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_P2   = os.path.join(_ROOT, "scratch", "phase2")
for p in [_ROOT, _P2]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fairness_metrics import (
    demographic_parity,
    equalized_odds,
    equal_opportunity,
    predictive_parity,
    disparate_impact,
    full_fairness_report,
    _group_rates,
)
from fairness_audit import (
    audit_by_group,
    multi_axis_audit,
    intersectional_audit,
    generate_audit_report,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_fair_data(n=200, seed=42):
    """Identical positive rates across groups A and B."""
    random.seed(seed)
    groups = ["A"] * (n // 2) + ["B"] * (n // 2)
    y_true = [random.randint(0, 1) for _ in range(n)]
    y_pred = list(y_true)   # perfect predictions → identical group rates
    return y_true, y_pred, groups

def _make_biased_data(n=200, seed=7):
    """Group B gets positive rate ~30%, Group A gets ~70%."""
    random.seed(seed)
    y_true, y_pred, groups = [], [], []
    for i in range(n):
        g = "A" if i < n // 2 else "B"
        yt = random.randint(0, 1)
        if g == "A":
            yp = 1 if random.random() < 0.70 else 0
        else:
            yp = 1 if random.random() < 0.30 else 0
        y_true.append(yt)
        y_pred.append(yp)
        groups.append(g)
    return y_true, y_pred, groups


# ─────────────────────────────────────────────────────────────────────────────
# _group_rates
# ─────────────────────────────────────────────────────────────────────────────

class TestGroupRates(unittest.TestCase):

    def test_keys_present(self):
        yt = [1, 0, 1, 0]
        yp = [1, 0, 0, 1]
        rates = _group_rates(yt, yp)
        for key in ["n", "TP", "FP", "TN", "FN", "TPR", "FPR", "PPV", "selection_rate"]:
            self.assertIn(key, rates)

    def test_perfect_prediction_rates(self):
        yt = [1, 1, 0, 0]
        yp = [1, 1, 0, 0]
        r = _group_rates(yt, yp)
        self.assertAlmostEqual(r["TPR"], 1.0)
        self.assertAlmostEqual(r["FPR"], 0.0)

    def test_all_wrong_rates(self):
        yt = [1, 1, 0, 0]
        yp = [0, 0, 1, 1]
        r = _group_rates(yt, yp)
        self.assertAlmostEqual(r["TPR"], 0.0)
        self.assertAlmostEqual(r["FPR"], 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# DEMOGRAPHIC PARITY
# ─────────────────────────────────────────────────────────────────────────────

class TestDemographicParity(unittest.TestCase):

    def test_fair_data_small_gap(self):
        y_true, y_pred, groups = _make_fair_data()
        dp = demographic_parity(y_pred, groups)
        self.assertIn("max_difference", dp)
        # Perfect predictions → DP gap ≈ 0
        self.assertLess(dp["max_difference"], 0.15)

    def test_biased_data_large_gap(self):
        _, y_pred, groups = _make_biased_data()
        dp = demographic_parity(y_pred, groups)
        # Selection rate difference should be visible
        self.assertGreater(dp["max_difference"], 0.1)

    def test_output_structure(self):
        y_pred  = [1, 0, 1, 0, 1, 0]
        groups  = ["A", "A", "A", "B", "B", "B"]
        dp = demographic_parity(y_pred, groups)
        self.assertIn("selection_rates", dp)
        self.assertIn("max_difference", dp)
        self.assertIn("passes", dp)

    def test_single_group(self):
        y_pred = [1, 0, 1]
        groups = ["A", "A", "A"]
        dp = demographic_parity(y_pred, groups)
        self.assertAlmostEqual(dp["max_difference"], 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# DISPARATE IMPACT
# ─────────────────────────────────────────────────────────────────────────────

class TestDisparateImpact(unittest.TestCase):

    def test_di_ratio_fair(self):
        """Equal selection rates → DI ≈ 1.0."""
        y_pred = [1, 1, 0, 0, 1, 1, 0, 0]
        groups = ["A"] * 4 + ["B"] * 4
        di = disparate_impact(y_pred, groups, privileged="A")
        self.assertAlmostEqual(di["di_ratio"], 1.0, places=4)

    def test_di_ratio_biased(self):
        """Privileged 100%, unprivileged 50% → DI = 0.5."""
        y_pred = [1, 1, 1, 1, 1, 0, 1, 0]
        groups = ["A", "A", "A", "A", "B", "B", "B", "B"]
        di = disparate_impact(y_pred, groups, privileged="A")
        # A: 4/4=1.0, B: 2/4=0.5 → DI=0.5
        self.assertAlmostEqual(di["di_ratio"], 0.5, places=4)

    def test_di_fails_below_80(self):
        y_pred = [1, 1, 1, 1, 1, 0, 1, 0]
        groups = ["A", "A", "A", "A", "B", "B", "B", "B"]
        di = disparate_impact(y_pred, groups, privileged="A")
        self.assertFalse(di["passes"])


# ─────────────────────────────────────────────────────────────────────────────
# EQUALIZED ODDS
# ─────────────────────────────────────────────────────────────────────────────

class TestEqualizedOdds(unittest.TestCase):

    def test_perfect_predictor_equal_odds(self):
        y_true, y_pred, groups = _make_fair_data()
        eo = equalized_odds(y_true, y_pred, groups)
        self.assertIn("tpr_gap", eo)
        self.assertIn("fpr_gap", eo)
        self.assertIn("max_gap", eo)
        # With identical per-group predictions, gaps should be small
        self.assertLess(eo["max_gap"], 0.20)

    def test_output_structure(self):
        y_true = [1, 0, 1, 0, 1, 0]
        y_pred = [1, 0, 0, 1, 1, 0]
        groups = ["A", "A", "A", "B", "B", "B"]
        eo = equalized_odds(y_true, y_pred, groups)
        for key in ["tpr_gap", "fpr_gap", "max_gap", "per_group"]:
            self.assertIn(key, eo)


# ─────────────────────────────────────────────────────────────────────────────
# FULL FAIRNESS REPORT
# ─────────────────────────────────────────────────────────────────────────────

class TestFullFairnessReport(unittest.TestCase):

    def test_report_structure(self):
        y_true, y_pred, groups = _make_fair_data()
        report = full_fairness_report(y_true, y_pred, groups)
        self.assertIsInstance(report, dict)
        # Should contain at least one fairness metric key
        self.assertGreater(len(report), 0)

    def test_report_not_empty(self):
        y_true, y_pred, groups = _make_biased_data()
        report = full_fairness_report(y_true, y_pred, groups)
        self.assertIsInstance(report, dict)


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT BY GROUP
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditByGroup(unittest.TestCase):

    def test_fair_passes(self):
        y_true, y_pred, groups = _make_fair_data(n=400)
        result = audit_by_group(y_true, y_pred, groups, group_name="test_attr")
        # Fair data should not be CRITICAL
        self.assertNotEqual(result["severity"], "CRITICAL")

    def test_biased_fails(self):
        _, y_pred, groups = _make_biased_data(n=400)
        y_true = [random.randint(0, 1) for _ in range(400)]
        result = audit_by_group(y_true, y_pred, groups,
                                group_name="test_attr", privileged="A")
        # Large bias → should not pass
        self.assertFalse(result["passed"])

    def test_structure(self):
        y_true, y_pred, groups = _make_fair_data()
        result = audit_by_group(y_true, y_pred, groups, group_name="race")
        for key in ["group_name", "group_names", "per_group_rates", "severity",
                    "summary", "warnings", "passed"]:
            self.assertIn(key, result)

    def test_small_group_warning(self):
        """Groups below 30 samples should trigger a warning."""
        y_true = [1, 0] * 15 + [1, 0]   # 32 total
        y_pred = y_true[:]
        groups = ["tiny"] * 2 + ["large"] * 30  # tiny has 2 samples
        result = audit_by_group(y_true, y_pred, groups)
        self.assertTrue(len(result["warnings"]) > 0)


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-AXIS AUDIT
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiAxisAudit(unittest.TestCase):

    def test_audits_all_axes(self):
        y_true, y_pred, race = _make_fair_data(n=300)
        gender = (["M"] * 75 + ["F"] * 75) * 2
        result = multi_axis_audit(y_true, y_pred,
                                  axes={"race": race, "gender": gender})
        self.assertIn("race", result)
        self.assertIn("gender", result)
        self.assertIn("_summary", result)

    def test_summary_present(self):
        y_true, y_pred, race = _make_fair_data(n=200)
        result = multi_axis_audit(y_true, y_pred, axes={"race": race})
        summ = result["_summary"]
        self.assertIn("axes_audited", summ)
        self.assertIn("overall_severity", summ)


# ─────────────────────────────────────────────────────────────────────────────
# INTERSECTIONAL AUDIT
# ─────────────────────────────────────────────────────────────────────────────

class TestIntersectionalAudit(unittest.TestCase):

    def test_finds_intersections(self):
        y_true, y_pred, race = _make_fair_data(n=400)
        gender = (["M"] * 100 + ["F"] * 100) * 2
        result = intersectional_audit(
            y_true, y_pred,
            axes={"race": race, "gender": gender},
            min_size=20
        )
        self.assertGreater(result["n_valid"], 0)

    def test_skips_small_groups(self):
        y_true, y_pred, race = _make_fair_data(n=100)
        gender = ["M"] * 98 + ["F"] * 2   # only 2 F samples
        result = intersectional_audit(
            y_true, y_pred,
            axes={"race": race, "gender": gender},
            min_size=20
        )
        # At least one intersection should be skipped
        self.assertGreater(result["n_skipped"], 0)

    def test_output_structure(self):
        y_true, y_pred, race = _make_fair_data(n=200)
        gender = (["M"] * 50 + ["F"] * 50) * 2
        result = intersectional_audit(y_true, y_pred,
                                      axes={"race": race, "gender": gender})
        for key in ["intersections", "skipped_small", "worst_intersection",
                    "selection_rate_range", "n_valid", "n_skipped"]:
            self.assertIn(key, result)


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE AUDIT REPORT
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateAuditReport(unittest.TestCase):

    def test_report_keys(self):
        y_true, y_pred, groups = _make_fair_data()
        single = audit_by_group(y_true, y_pred, groups, group_name="race")
        report = generate_audit_report(single)
        for key in ["title", "overall_pass", "severity", "axes",
                    "recommendations", "raw"]:
            self.assertIn(key, report)

    def test_recommendations_list(self):
        y_true, y_pred, groups = _make_biased_data()
        single = audit_by_group(y_true, y_pred, groups)
        report = generate_audit_report(single)
        self.assertIsInstance(report["recommendations"], list)
        self.assertGreater(len(report["recommendations"]), 0)

    def test_multi_axis_report(self):
        y_true, y_pred, race = _make_fair_data(n=300)
        gender = (["M"] * 75 + ["F"] * 75) * 2
        multi  = multi_axis_audit(y_true, y_pred,
                                  axes={"race": race, "gender": gender})
        report = generate_audit_report(multi, title="Test Multi-Axis")
        self.assertEqual(report["title"], "Test Multi-Axis")
        self.assertIsInstance(report["axes"], list)
        self.assertGreaterEqual(len(report["axes"]), 1)


# ─────────────────────────────────────────────────────────────────────────────
# EDGE CASES
# ─────────────────────────────────────────────────────────────────────────────

class TestFairnessEdgeCases(unittest.TestCase):

    def test_single_group(self):
        y_true = [1, 0, 1]
        y_pred = [1, 0, 1]
        groups = ["A", "A", "A"]
        dp = demographic_parity(y_pred, groups)
        self.assertAlmostEqual(dp["max_difference"], 0.0)

    def test_all_positive_predictions(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 1, 1, 1]
        groups = ["A", "A", "B", "B"]
        dp = demographic_parity(y_pred, groups)
        self.assertAlmostEqual(dp["max_difference"], 0.0)  # Both groups 100%

    def test_empty_group_handled(self):
        y_true = [1, 1, 0]
        y_pred = [1, 1, 0]
        groups = ["A", "A", "B"]
        # Should not raise even with tiny group B
        result = audit_by_group(y_true, y_pred, groups)
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
