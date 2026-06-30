import os
import sys
import pandas as pd
import json

# Add project root to path
_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine.integration.analysis_runner import run_analysis

def verify_module(name, dataset, config):
    print(f"\n--- Verifying {name} ---", flush=True)
    dataset_path = os.path.join(_ROOT, 'webapp', 'uploads', dataset)
    
    try:
        result = run_analysis(dataset_path, name, config)
        status = "PASS" if result['status'] == 'ok' else "FAIL"
        
        # Try to extract resolved target from logs
        resolved_target = "N/A"
        if 'results' in result and 'log' in result['results']:
            for log in result['results']['log']:
                if "Inferred target column" in log:
                    resolved_target = log.split("'")[-2]
        
        print(f"Status: {status}", flush=True)
        print(f"Target Column Selected: {resolved_target}", flush=True)
        
        if status == "PASS":
            # Print a summary of the output
            res_data = result['results']
            # Convert numpy types for printing
            def clean_dict(d):
                if isinstance(d, dict):
                    return {k: clean_dict(v) for k, v in d.items()}
                elif isinstance(d, list):
                    return [clean_dict(v) for v in d]
                elif hasattr(d, 'to_dict'): # For DataFrame/Series
                    return d.to_dict()
                elif hasattr(d, 'tolist'): # For ndarray
                    return d.tolist()
                return d

            summary = {k: clean_dict(v) for k, v in res_data.items() if k not in ['log', 'data']}
            print(f"Output Generated: {json.dumps(summary, indent=2)[:500]}...", flush=True)
        else:
            print(f"Error: {result.get('error')}", flush=True)
            
        return status
    except Exception as e:
        print(f"Exception: {str(e)}", flush=True)
        return "FAIL"

if __name__ == "__main__":
    results = {}
    
    # 1. Label Noise Analysis
    # Dataset: movies_dataset.csv, Hint: 'movies' (should resolve to movie_rating)
    results['Label Noise'] = verify_module(
        'label_noise', 
        'movies_dataset.csv', 
        {'target_col': 'movies'}
    )
    
    # 2. Leakage Detection
    # Dataset: house-price.csv, Hint: 'price'
    results['Leakage Detection'] = verify_module(
        'leakage', 
        'house-price.csv', 
        {'target_col': 'price'}
    )
    
    # 3. Missing Data
    # Dataset: hospital_dataset.csv, Hint: 'hospital' (should resolve to something if available)
    # Let's check hospital columns first if possible, or just use auto-detection
    results['Missing Data'] = verify_module(
        'missing_data', 
        'hospital_dataset.csv', 
        {'target_col': ''}
    )
    
    # 4. Causal Analysis
    # Dataset: house-price.csv, Treatment: area, Outcome: price
    results['Causal Analysis'] = verify_module(
        'causal', 
        'house-price.csv', 
        {
            'treatment_col': 'area',
            'outcome_col': 'price',
            'nodes': 'area, price',
            'edges': 'area -> price'
        }
    )
    
    print("\n\n=== FINAL SUMMARY ===", flush=True)
    for k, v in results.items():
        print(f"{k}: {v}", flush=True)
