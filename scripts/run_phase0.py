"""
run_phase0.py — Phase 0 Verification Runner
=============================================
Runs all Phase 0 (Prerequisites) verification suites in sequence.
Each module verifies its implementations against numpy/scipy and prints PASS/FAIL.

Usage:
    python scratch/phase0/run_phase0.py
    # or from project root:
    python -m scratch.phase0.run_phase0
"""

import sys
import os
import time

# Ensure project root is on path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def run_module(module_name, verify_fn):
    print(f"\n{'='*60}")
    print(f"  Running: {module_name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        verify_fn()
        elapsed = time.time() - t0
        print(f"\n  ✓ {module_name} completed in {elapsed:.2f}s")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  ✗ {module_name} FAILED after {elapsed:.2f}s")
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("  PHASE 0 — Prerequisites Verification Suite")
    print("  Python & Mathematics from Scratch")
    print("="*60)

    results = {}

    # 1. Linear Algebra
    try:
        from scratch.phase0.linear_algebra import run_verification as la_verify
        results["Linear Algebra"] = run_module("Linear Algebra", la_verify)
    except ImportError as e:
        print(f"  ✗ Could not import linear_algebra: {e}")
        results["Linear Algebra"] = False

    # 2. Calculus & Optimization
    try:
        from scratch.phase0.calculus_optimization import run_verification as calc_verify
        results["Calculus & Optimization"] = run_module("Calculus & Optimization", calc_verify)
    except ImportError as e:
        print(f"  ✗ Could not import calculus_optimization: {e}")
        results["Calculus & Optimization"] = False

    # 3. Statistics
    try:
        from scratch.phase0.statistics import run_verification as stats_verify
        results["Statistics"] = run_module("Statistics", stats_verify)
    except ImportError as e:
        print(f"  ✗ Could not import statistics: {e}")
        results["Statistics"] = False

    # 4. Probability & Statistics (extended)
    try:
        from scratch.phase0.probability_statistics import run_verification as prob_verify
        results["Probability & Statistics"] = run_module("Probability & Statistics", prob_verify)
    except ImportError as e:
        print(f"  ✗ Could not import probability_statistics: {e}")
        results["Probability & Statistics"] = False

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 0 SUMMARY")
    print("="*60)
    passed = sum(1 for v in results.values() if v)
    total  = len(results)
    for name, ok in results.items():
        icon = "✓" if ok else "✗"
        print(f"  {icon}  {name}")
    print(f"\n  Result: {passed}/{total} modules passed")

    if passed == total:
        print("\n  🎉 PHASE 0 COMPLETE — Ready for Phase 1!")
    else:
        print("\n  ⚠️  Fix failing modules before proceeding to Phase 1.")
    print("="*60 + "\n")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
