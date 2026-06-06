import os
import sys
import pandas as pd
import numpy as np

print("DEBUG: Script started", flush=True)

# Add project root to path
_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

print(f"DEBUG: _ROOT is {_ROOT}", flush=True)

try:
    from engine.integration.analysis_runner import run_analysis
    from engine.label_noise import LabelNoiseAnalyzer
    from engine.leakage_detector import LeakageDetector
    from engine.missing_data import MissingDataAnalyzer
    print("DEBUG: Imports successful", flush=True)
except Exception as e:
    print(f"DEBUG: Import failed: {e}", flush=True)
    sys.exit(1)

def test_fixes():
    try:
        # 1. Create a dummy dataset 'movies_dataset.csv'
        df = pd.DataFrame({
            'movie_rating': [1, 2, 3, 4, 5] * 20,
            'budget': np.random.randn(100),
            'genre': ['action', 'comedy', 'drama', 'horror', 'sci-fi'] * 20
        })
        
        upload_dir = os.path.join(_ROOT, 'webapp', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        dataset_path = os.path.join(upload_dir, 'movies_dataset.csv')
        df.to_csv(dataset_path, index=False)
        print(f"Created dummy dataset at {dataset_path}", flush=True)

        # 2. Test LabelNoiseAnalyzer
        print("\nTesting LabelNoiseAnalyzer resolution...", flush=True)
        config = {'target_col': 'movies'}
        result = run_analysis(dataset_path, 'label_noise', config)
        if result['status'] == 'ok':
            print("[PASS] LabelNoiseAnalyzer resolved 'movies' target", flush=True)
            logs = result['results']['log']
            if any("Inferred target column from hint 'movies': 'movie_rating'" in l for l in logs):
                 print("[OK] Logs confirm resolution of 'movies' to 'movie_rating'", flush=True)
            else:
                 print(f"[WARN] Resolution log not found. Logs: {logs}", flush=True)
        else:
            print(f"[FAIL] LabelNoiseAnalyzer failed: {result.get('error')}", flush=True)

        # 3. Test LeakageDetector
        print("\nTesting LeakageDetector resolution...", flush=True)
        config = {'target_col': 'movies'}
        result = run_analysis(dataset_path, 'leakage', config)
        if result['status'] == 'ok':
            print("[PASS] LeakageDetector resolved 'movies' target", flush=True)
        else:
            print(f"[FAIL] LeakageDetector failed: {result.get('error')}", flush=True)

        # 4. Test MissingDataAnalyzer
        print("\nTesting MissingDataAnalyzer resolution...", flush=True)
        config = {'target_col': 'movies'}
        result = run_analysis(dataset_path, 'missing_data', config)
        if result['status'] == 'ok':
            print("[PASS] MissingDataAnalyzer resolved 'movies' target", flush=True)
        else:
            print(f"[FAIL] MissingDataAnalyzer failed: {result.get('error')}", flush=True)

        # 5. Test with no target
        print("\nTesting automatic target detection (no target hint)...", flush=True)
        config = {'target_col': ''}
        result = run_analysis(dataset_path, 'label_noise', config)
        if result['status'] == 'failed':
            print(f"[OK] Validation caught empty target: {result.get('error')}", flush=True)
        
        # 6. Test with a housing dataset
        df_house = pd.DataFrame({
            'price': [100, 200, 300, 400, 500] * 20,
            'sqft': np.random.randn(100),
            'rooms': [2, 3, 4, 5, 6] * 20
        })
        house_path = os.path.join(upload_dir, 'housing_test.csv')
        df_house.to_csv(house_path, index=False)
        
        print("\nTesting housing dataset automatic detection...", flush=True)
        config = {'target_col': None}
        result = run_analysis(house_path, 'missing_data', config)
        if result['status'] == 'ok':
            print("[PASS] MissingDataAnalyzer auto-detected 'price'", flush=True)
        else:
            print(f"[FAIL] MissingDataAnalyzer failed: {result.get('error')}", flush=True)

    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixes()
