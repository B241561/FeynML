import sys
import os

# Ensure project root is on sys.path when this file is executed directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the test functions from phase4_verification; map them to the expected names
from engine.phase4_verification import (
    test_label_noise as verify_label_noise,
    test_leakage_detection as verify_leakage_detection,
    test_missing_data as verify_missing_data,
    test_causal_thinking as verify_causal_thinking,
    test_causal_inference as verify_causal_inference,
)


def run_verification():
    results = {}
    tests = {
        "Label Noise":       verify_label_noise,
        "Leakage Detection": verify_leakage_detection,
        "Missing Data":      verify_missing_data,
        "Causal Thinking":   verify_causal_thinking,
        "Causal Inference":  verify_causal_inference,
    }
    for name, fn in tests.items():
        try:
            # Call the verification function; it should raise on failure or return False
            rv = fn()
            if rv is False:
                raise RuntimeError("verification returned False")
            results[name] = "PASS"
            print(f"  [PASS] {name}")
        except Exception as e:
            results[name] = f"FAIL: {e}"
            print(f"  [FAIL] {name}: {e}")

    all_passed = all("PASS" == v for v in results.values())
    print(f"\nResult: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
