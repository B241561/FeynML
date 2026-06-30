"""
run_phase2.py — Phase 2 Full Verification Runner
=================================================
Runs every Phase 2 module (Metrics, Validation, Fairness, Calibration)
and prints a structured PASS/FAIL report.

Usage:
    cd ml_failure_engine
    python scratch/phase2/run_phase2.py
"""

import sys
import os
import time
import random
import math

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_P2 = os.path.join(_ROOT, "scratch", "phase2")
if _P2 not in sys.path:
    sys.path.insert(0, _P2)


def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def run_check(name, fn):
    try:
        t0  = time.time()
        fn()
        elapsed = time.time() - t0
        print(f"  ✓  {name:<45} ({elapsed:.2f}s)")
        return True
    except Exception as e:
        print(f"  ✗  {name:<45} → {e}")
        return False


def main():
    print("\n" + "="*60)
    print("  PHASE 2 — Model Evaluation Full Suite")
    print("  Classification · Regression · Validation · Fairness · Calibration")
    print("="*60)

    random.seed(42)
    results = {}

    # ── 1. Classification Metrics ─────────────────────────────────
    section("1. Classification Metrics")
    try:
        from scratch.phase2.classification_metrics import (
            binary_confusion, accuracy, precision, recall, f1_score,
            matthews_corrcoef, cohen_kappa, roc_auc_score,
            average_precision_score, log_loss, binary_metrics_summary
        )
        import sklearn.metrics as skm

        n       = 400
        y_true  = [1 if random.random() < 0.15 else 0 for _ in range(n)]
        y_prob  = [(random.random()*0.4+0.5 if y else random.random()*0.5) for y in y_true]
        y_pred  = [1 if p >= 0.5 else 0 for p in y_prob]

        def chk_acc():
            assert abs(accuracy(y_true, y_pred) - skm.accuracy_score(y_true, y_pred)) < 0.001
        def chk_prec():
            assert abs(precision(y_true, y_pred) - skm.precision_score(y_true, y_pred, zero_division=0)) < 0.001
        def chk_rec():
            assert abs(recall(y_true, y_pred) - skm.recall_score(y_true, y_pred, zero_division=0)) < 0.001
        def chk_f1():
            assert abs(f1_score(y_true, y_pred) - skm.f1_score(y_true, y_pred, zero_division=0)) < 0.001
        def chk_mcc():
            assert abs(matthews_corrcoef(y_true, y_pred) - skm.matthews_corrcoef(y_true, y_pred)) < 0.005
        def chk_kappa():
            assert abs(cohen_kappa(y_true, y_pred) - skm.cohen_kappa_score(y_true, y_pred)) < 0.005
        def chk_auc():
            assert abs(roc_auc_score(y_true, y_prob) - skm.roc_auc_score(y_true, y_prob)) < 0.005
        def chk_ap():
            assert abs(average_precision_score(y_true, y_prob) - skm.average_precision_score(y_true, y_prob)) < 0.01
        def chk_logloss():
            assert abs(log_loss(y_true, y_prob) - skm.log_loss(y_true, y_prob)) < 0.01
        def chk_summary():
            s = binary_metrics_summary(y_true, y_pred, y_prob)
            assert "f1_score" in s and "roc_auc" in s and "confusion" in s

        for name_f, fn in [
            ("accuracy",             chk_acc),
            ("precision",            chk_prec),
            ("recall",               chk_rec),
            ("f1_score",             chk_f1),
            ("matthews_corrcoef",    chk_mcc),
            ("cohen_kappa",          chk_kappa),
            ("roc_auc_score",        chk_auc),
            ("average_precision",    chk_ap),
            ("log_loss",             chk_logloss),
            ("binary_metrics_summary", chk_summary),
        ]:
            results[f"classification.{name_f}"] = run_check(f"classification_metrics.{name_f}", fn)
    except ImportError as e:
        print(f"  ✗  Import failed: {e}")

    # ── 2. Regression Metrics ─────────────────────────────────────
    section("2. Regression Metrics")
    try:
        from scratch.phase2.regression_metrics import (
            mse, rmse, mae, r2_score, regression_summary
        )
        import sklearn.metrics as skm

        n    = 300
        y_t  = [2*random.random() + i*0.5 for i in range(n)]
        y_p  = [y + random.gauss(0, 0.3)  for y in y_t]

        def chk_mse():
            assert abs(mse(y_t, y_p) - skm.mean_squared_error(y_t, y_p)) < 1e-4
        def chk_mae():
            assert abs(mae(y_t, y_p) - skm.mean_absolute_error(y_t, y_p)) < 1e-4
        def chk_r2():
            assert abs(r2_score(y_t, y_p) - skm.r2_score(y_t, y_p)) < 1e-4
        def chk_rmse():
            assert abs(rmse(y_t, y_p) - math.sqrt(skm.mean_squared_error(y_t, y_p))) < 1e-4
        def chk_summary():
            s = regression_summary(y_t, y_p)
            assert "rmse" in s and "r2" in s and "mae" in s

        for name_f, fn in [
            ("mse", chk_mse), ("mae", chk_mae),
            ("r2_score", chk_r2), ("rmse", chk_rmse),
            ("regression_summary", chk_summary),
        ]:
            results[f"regression.{name_f}"] = run_check(f"regression_metrics.{name_f}", fn)
    except ImportError as e:
        print(f"  ✗  Import failed: {e}")

    # ── 3. Validation Strategy ────────────────────────────────────
    section("3. Validation Strategy")
    try:
        from scratch.phase2.validation_strategy import (
            kfold_split, stratified_kfold_split, time_series_split,
            detect_target_leakage, check_temporal_ordering
        )

        n  = 100
        X  = [[random.random() for _ in range(4)] for _ in range(n)]
        y  = [random.randint(0, 1) for _ in range(n)]

        def chk_kfold():
            splits = kfold_split(n, k=5)
            all_val = []
            for tr, vl in splits:
                assert len(set(tr) & set(vl)) == 0
                all_val.extend(vl)
            assert sorted(all_val) == list(range(n))

        def chk_stratified():
            splits = stratified_kfold_split(y, k=5)
            overall = sum(y) / len(y)
            for tr, vl in splits:
                fold_rate = sum(y[i] for i in vl) / len(vl)
                assert abs(fold_rate - overall) < 0.2

        def chk_ts():
            ts = time_series_split(n, n_splits=4)
            for tr, vl in ts:
                assert max(tr) < min(vl)

        def chk_leakage():
            X_l = [[y[i] + random.gauss(0,0.01)] + [random.random() for _ in range(3)]
                   for i in range(n)]
            r = detect_target_leakage(X_l, y, ["leaky","f1","f2","f3"])
            assert r["n_suspects"] >= 1

        def chk_temporal():
            ts = list(range(n))
            r = check_temporal_ordering(ts, list(range(80)), list(range(80, 100)))
            assert not r["temporal_leakage"]

        for name_f, fn in [
            ("kfold_split", chk_kfold), ("stratified_kfold", chk_stratified),
            ("time_series_split", chk_ts), ("leakage_detection", chk_leakage),
            ("temporal_check", chk_temporal),
        ]:
            results[f"validation.{name_f}"] = run_check(f"validation_strategy.{name_f}", fn)
    except ImportError as e:
        print(f"  ✗  Import failed: {e}")

    # ── 4. Fairness Metrics ───────────────────────────────────────
    section("4. Fairness Metrics")
    try:
        from scratch.phase2.fairness_metrics import (
            demographic_parity, equalized_odds, disparate_impact,
            predictive_parity, fairness_report
        )

        n    = 200
        sens = [random.randint(0, 1) for _ in range(n)]
        y_t  = [random.randint(0, 1) for _ in range(n)]
        y_p  = [random.randint(0, 1) for _ in range(n)]

        def chk_dp():
            dp = demographic_parity(y_p, sens)
            assert "group_0" in dp and "group_1" in dp and "parity_gap" in dp

        def chk_eo():
            eo = equalized_odds(y_t, y_p, sens)
            assert "TPR_gap" in eo and "FPR_gap" in eo

        def chk_di():
            di = disparate_impact(y_p, sens)
            assert 0.0 <= di <= 1.0

        def chk_pp():
            pp = predictive_parity(y_t, y_p, sens)
            assert "PPV_gap" in pp

        def chk_report():
            r = fairness_report(y_t, y_p, sens, group_names=["group_A", "group_B"])
            assert "demographic_parity" in r and "disparate_impact" in r

        for name_f, fn in [
            ("demographic_parity", chk_dp), ("equalized_odds", chk_eo),
            ("disparate_impact", chk_di), ("predictive_parity", chk_pp),
            ("fairness_report", chk_report),
        ]:
            results[f"fairness.{name_f}"] = run_check(f"fairness_metrics.{name_f}", fn)
    except ImportError as e:
        print(f"  ✗  Import failed: {e}")

    # ── 5. Calibration ────────────────────────────────────────────
    section("5. Calibration")
    try:
        from scratch.phase2.calibration import (
            reliability_curve, expected_calibration_error,
            brier_score, platt_scaling
        )

        n    = 300
        y_t  = [random.randint(0, 1) for _ in range(n)]
        y_pb = [min(0.99, max(0.01, random.gauss(0.6 if y else 0.4, 0.15))) for y in y_t]

        def chk_reliability():
            mp, fp, counts = reliability_curve(y_t, y_pb, n_bins=10)
            assert len(mp) > 0 and len(fp) == len(mp)

        def chk_ece():
            ece = expected_calibration_error(y_t, y_pb, n_bins=10)
            assert 0.0 <= ece <= 1.0

        def chk_brier():
            bs = brier_score(y_t, y_pb)
            assert 0.0 <= bs <= 1.0

        def chk_platt():
            cal_probs = platt_scaling(y_t, y_pb, y_pb)
            assert len(cal_probs) == n
            assert all(0.0 <= p <= 1.0 for p in cal_probs)

        for name_f, fn in [
            ("reliability_curve", chk_reliability), ("ece", chk_ece),
            ("brier_score", chk_brier), ("platt_scaling", chk_platt),
        ]:
            results[f"calibration.{name_f}"] = run_check(f"calibration.{name_f}", fn)
    except ImportError as e:
        print(f"  ✗  Import failed: {e}")

    # ── 6. Engine Modules ─────────────────────────────────────────
    section("6. Engine Modules (importability check)")
    for mod_name in ["evaluator", "validator", "fairness_engine",
                     "calibration_engine", "report_engine"]:
        def make_import_check(name):
            def fn():
                import importlib
                m = importlib.import_module(f"engine.modules.{name}")
                assert m is not None
            return fn
        results[f"engine.{mod_name}"] = run_check(
            f"engine.modules.{mod_name}", make_import_check(mod_name))

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 2 SUMMARY")
    print("="*60)

    by_group = {}
    for key, ok in results.items():
        group = key.split(".")[0]
        by_group.setdefault(group, []).append(ok)

    total_pass = sum(1 for v in results.values() if v)
    total_all  = len(results)

    for group, oks in by_group.items():
        gp = sum(oks)
        gt = len(oks)
        icon = "✓" if gp == gt else "⚠️"
        print(f"  {icon}  {group:<25} {gp}/{gt} checks passed")

    print(f"\n  Overall: {total_pass}/{total_all} checks passed")

    if total_pass == total_all:
        print("\n  🎉 PHASE 2 COMPLETE — Ready for Phase 3 (SHAP/LIME/Slices)!")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n  ⚠️  {len(failed)} checks failed:")
        for f in failed:
            print(f"     • {f}")
    print("="*60 + "\n")

    return total_pass == total_all


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
