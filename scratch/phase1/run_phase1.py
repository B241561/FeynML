"""
Phase 1 — Runner Script
========================
Runs all Phase 1 verifications in sequence.
Usage: python run_phase1.py
"""

import sys
import time

def run_module(name, module_path):
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"{'='*60}")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.run_verification()
        print(f"  ✓ {name} completed successfully")
        return True
    except Exception as e:
        print(f"  ✗ {name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import os
    base = os.path.dirname(__file__)

    modules = [
        ("Linear Regression",     os.path.join(base, "linear_regression.py")),
        ("Logistic Regression",   os.path.join(base, "logistic_regression.py")),
        ("Decision Tree",         os.path.join(base, "decision_tree.py")),
        ("Ensemble Models",       os.path.join(base, "ensemble_models.py")),
        ("K-Means & PCA",         os.path.join(base, "kmeans_pca.py")),
    ]

    results = []
    start = time.time()
    for name, path in modules:
        ok = run_module(name, path)
        results.append((name, ok))

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  PHASE 1 SUMMARY  ({elapsed:.1f}s)")
    print(f"{'='*60}")
    for name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {name}")

    n_pass = sum(1 for _, ok in results if ok)
    print(f"\n  {n_pass}/{len(results)} modules passed")
    sys.exit(0 if n_pass == len(results) else 1)
