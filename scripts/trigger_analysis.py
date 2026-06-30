print("DEBUG: Script started")
import os
import time
print("DEBUG: Imports done")
from webapp.services.analysis_runner import AnalysisRunner
print("DEBUG: AnalysisRunner imported")

def test_runner():
    runner = AnalysisRunner()
    filepath = os.path.abspath("webapp/uploads/house-price.csv")
    config = {
        'target_col': 'price',
        'pred_col': None,
        'sensitive_col': None,
        'timestamp_col': None,
        'auto_predict': True
    }
    
    print(f"Starting test analysis on {filepath}...")
    runner.run(filepath, config)
    
    # Wait for completion or failure
    while runner.status.startswith("running") or runner.status == "saving_report":
        print(f"Status: {runner.status}, Progress: {runner.progress}%")
        time.sleep(2)
    
    print(f"\nFinal Status: {runner.status}")
    if runner.error:
        print(f"Error: {runner.error}")
    
    if runner.logs:
        print("\nLogs:")
        for log in runner.logs:
            print(log)

if __name__ == "__main__":
    test_runner()
