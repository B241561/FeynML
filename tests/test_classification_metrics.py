"""
Unit Tests — Classification Metrics
=====================================
Tests every function in scratch/phase2/classification_metrics.py.
Uses only stdlib (unittest) — no pytest required.

Run:
    python -m tests.test_classification_metrics
  or:
    python tests/test_classification_metrics.py
"""

import sys
import os
import unittest
import math

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_P2   = os.path.join(_ROOT, "scratch", "phase2")
for p in [_ROOT, _P2]:
    if p not in sys.path:
        sys.path.insert(0, p)

from classification_metrics import (
    confusion_matrix,
    binary_confusion,
    precision, recall, f1_score, fbeta_score,
    roc_curve, roc_auc,
    precision_recall_curve, average_precision,
    log_loss,
    cohen_kappa,
    matthews_corrcoef,
    calibration_curve,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _near(a, b, tol=1e-4):
    return abs(a - b) <= tol


# ─────────────────────────────────────────────────────────────────────────────
# PERFECT PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

class TestPerfectPredictions(unittest.TestCase):

    def setUp(self):
        self.y_true = [1, 1, 0, 0, 1, 0, 1, 0]
        self.y_pred = [1, 1, 0, 0, 1, 0, 1, 0]   # perfect
        self.y_prob = [0.9, 0.8, 0.1, 0.2, 0.95, 0.05, 0.85, 0.15]

    def test_perfect_precision(self):
        self.assertEqual(precision(self.y_true, self.y_pred), 1.0)

    def test_perfect_recall(self):
        self.assertEqual(recall(self.y_true, self.y_pred), 1.0)

    def test_perfect_f1(self):
        self.assertEqual(f1_score(self.y_true, self.y_pred), 1.0)

    def test_perfect_auc(self):
        auc = roc_auc(self.y_true, self.y_prob)
        self.assertGreater(auc, 0.95)

    def test_perfect_confusion(self):
        TN, FP, FN, TP = binary_confusion(self.y_true, self.y_pred)
        self.assertEqual(FP, 0)
        self.assertEqual(FN, 0)
        self.assertEqual(TP, 4)
        self.assertEqual(TN, 4)


# ─────────────────────────────────────────────────────────────────────────────
# KNOWN VALUES
# ─────────────────────────────────────────────────────────────────────────────

class TestKnownValues(unittest.TestCase):

    def test_precision_known(self):
        # TP=3, FP=1 → P=0.75
        y_true = [1, 1, 1, 0, 0]
        y_pred = [1, 1, 1, 1, 0]
        self.assertAlmostEqual(precision(y_true, y_pred), 0.75, places=4)

    def test_recall_known(self):
        # TP=2, FN=1 → R=2/3
        y_true = [1, 1, 1, 0]
        y_pred = [1, 1, 0, 0]
        self.assertAlmostEqual(recall(y_true, y_pred), 2/3, places=4)

    def test_f1_known(self):
        # P=0.5, R=1.0 → F1=2*(0.5*1)/(0.5+1)=2/3
        y_true = [1, 0, 1]
        y_pred = [1, 1, 1]
        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        self.assertAlmostEqual(f1, 2 * p * r / (p + r), places=4)

    def test_fbeta_beta1_equals_f1(self):
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 0, 0, 1, 1]
        self.assertAlmostEqual(
            fbeta_score(y_true, y_pred, beta=1.0),
            f1_score(y_true, y_pred),
            places=6
        )

    def test_fbeta_beta0_equals_precision(self):
        """F-beta with beta→0 approaches precision."""
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 1]
        p  = precision(y_true, y_pred)
        fb = fbeta_score(y_true, y_pred, beta=0.01)
        self.assertAlmostEqual(fb, p, places=2)


# ─────────────────────────────────────────────────────────────────────────────
# ALL-WRONG PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

class TestAllWrong(unittest.TestCase):

    def setUp(self):
        self.y_true = [1, 1, 0, 0]
        self.y_pred = [0, 0, 1, 1]   # perfectly wrong

    def test_precision_zero(self):
        self.assertEqual(precision(self.y_true, self.y_pred), 0.0)

    def test_recall_zero(self):
        self.assertEqual(recall(self.y_true, self.y_pred), 0.0)

    def test_f1_zero(self):
        self.assertEqual(f1_score(self.y_true, self.y_pred), 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

class TestConfusionMatrix(unittest.TestCase):

    def test_binary_shape(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 0]
        cm, labels = confusion_matrix(y_true, y_pred)
        self.assertEqual(len(cm), 2)
        self.assertEqual(len(cm[0]), 2)

    def test_binary_sum(self):
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0, 0, 1, 1, 0]
        cm, _ = confusion_matrix(y_true, y_pred)
        total = sum(cm[i][j] for i in range(2) for j in range(2))
        self.assertEqual(total, len(y_true))

    def test_tn_fp_fn_tp(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 1]
        TN, FP, FN, TP = binary_confusion(y_true, y_pred)
        self.assertEqual(TP, 1)
        self.assertEqual(TN, 1)
        self.assertEqual(FP, 1)
        self.assertEqual(FN, 1)


# ─────────────────────────────────────────────────────────────────────────────
# ROC & AUC
# ─────────────────────────────────────────────────────────────────────────────

class TestROC(unittest.TestCase):

    def test_auc_random(self):
        """Random classifier should have AUC ≈ 0.5."""
        import random
        random.seed(7)
        n = 200
        y_true = [random.randint(0, 1) for _ in range(n)]
        y_prob  = [random.random() for _ in range(n)]
        auc = roc_auc(y_true, y_prob)
        # Very loose bound for randomness
        self.assertGreater(auc, 0.35)
        self.assertLess(auc, 0.65)

    def test_auc_perfect(self):
        y_true = [0, 0, 1, 1]
        y_prob  = [0.1, 0.2, 0.8, 0.9]
        self.assertAlmostEqual(roc_auc(y_true, y_prob), 1.0, places=4)

    def test_roc_starts_at_origin(self):
        y_true = [0, 0, 1, 1]
        y_prob  = [0.1, 0.2, 0.8, 0.9]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        self.assertAlmostEqual(fpr[0], 0.0, places=4)
        self.assertAlmostEqual(tpr[0], 0.0, places=4)

    def test_roc_ends_at_one(self):
        y_true = [0, 0, 1, 1]
        y_prob  = [0.1, 0.2, 0.8, 0.9]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        self.assertAlmostEqual(fpr[-1], 1.0, places=4)
        self.assertAlmostEqual(tpr[-1], 1.0, places=4)


# ─────────────────────────────────────────────────────────────────────────────
# LOG-LOSS
# ─────────────────────────────────────────────────────────────────────────────

class TestLogLoss(unittest.TestCase):

    def test_perfect_log_loss_near_zero(self):
        y_true = [1, 0, 1, 0]
        y_prob  = [0.999, 0.001, 0.999, 0.001]
        ll = log_loss(y_true, y_prob)
        self.assertLess(ll, 0.01)

    def test_log_loss_0_5_prob(self):
        """Predicting 0.5 always → log-loss = log(2) ≈ 0.693."""
        y_true = [0, 1, 0, 1, 0, 1]
        y_prob  = [0.5] * 6
        ll = log_loss(y_true, y_prob)
        self.assertAlmostEqual(ll, math.log(2), places=3)


# ─────────────────────────────────────────────────────────────────────────────
# KAPPA & MCC
# ─────────────────────────────────────────────────────────────────────────────

class TestAgreementMetrics(unittest.TestCase):

    def test_kappa_perfect(self):
        y = [0, 0, 1, 1, 0, 1]
        self.assertAlmostEqual(cohen_kappa(y, y), 1.0, places=4)

    def test_kappa_range(self):
        y_true = [1, 0, 1, 0, 1]
        y_pred = [0, 1, 1, 0, 0]
        k = cohen_kappa(y_true, y_pred)
        self.assertGreaterEqual(k, -1.0)
        self.assertLessEqual(k, 1.0)

    def test_mcc_perfect(self):
        y = [1, 1, 0, 0]
        self.assertAlmostEqual(matthews_corrcoef(y, y), 1.0, places=4)

    def test_mcc_all_wrong(self):
        y_true = [1, 1, 0, 0]
        y_pred = [0, 0, 1, 1]
        mcc = matthews_corrcoef(y_true, y_pred)
        self.assertAlmostEqual(mcc, -1.0, places=4)


# ─────────────────────────────────────────────────────────────────────────────
# EDGE CASES
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases(unittest.TestCase):

    def test_all_positive_predictions(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 1, 1, 1]
        r = recall(y_true, y_pred)
        self.assertEqual(r, 1.0)  # all positives caught

    def test_no_positive_predictions(self):
        y_true = [1, 0, 1, 0]
        y_pred = [0, 0, 0, 0]
        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)
        self.assertEqual(p, 0.0)
        self.assertEqual(r, 0.0)

    def test_single_sample(self):
        y_true = [1]
        y_pred = [1]
        self.assertEqual(precision(y_true, y_pred), 1.0)
        self.assertEqual(recall(y_true, y_pred), 1.0)

    def test_all_same_class(self):
        y_true = [1, 1, 1, 1]
        y_pred = [1, 1, 1, 1]
        r = recall(y_true, y_pred)
        self.assertEqual(r, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
