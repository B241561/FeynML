import os
import sys
import json
import time
import pandas as pd

# Add current directory to path so we can import webapp
sys.path.append(os.path.abspath('.'))

from webapp.services.analysis_runner import AnalysisRunner

def verify():
    print("--- Starting Verification Script ---")
    
    # Use existing test_iris.csv
    dataset_path = os.path.abspath('test_iris.csv')
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return

    # Mock config
    config = {
        'target_col': 'species',
        'prediction_col': None,  # AnalysisRunner will auto-predict
        'sensitive_col': 'sepal length (cm)',
        'timestamp_col': None,
        'task_type': 'classification'
    }

    runner = AnalysisRunner()
    print(f"Running analysis on {dataset_path}...")
    
    # AnalysisRunner.run normally runs in a thread, let's wait for it
    runner.run(dataset_path, config)
    
    # Since run() might be async in some implementations (though we saw it called directly in our previous Read),
    # let's check status. In the current AnalysisRunner.run it is a blocking call if called directly.
    while runner.status not in ['completed', 'failed']:
        time.sleep(1)
        print(f"Status: {runner.status} ({runner.progress}%)")

    if runner.status == 'failed':
        print(f"Analysis failed: {runner.error}")
        return

    report_path = runner.report_path
    print(f"Report saved to: {report_path}")

    # Verify JSON content
    with open(report_path, 'r') as f:
        data = json.load(f)

    if 'charts' not in data:
        print("FAILED: 'charts' key missing from report JSON.")
        return

    charts = data['charts']
    print(f"SUCCESS: 'charts' key found with {len(charts)} charts.")
    
    for key in charts.keys():
        payload_len = len(charts[key])
        print(f" - {key}: {payload_len} chars")

    required_charts = ['prediction_dist', 'ks_ranked', 'leakage_scores', 'correlation_heatmap']
    for rc in required_charts:
        if rc in charts:
            print(f" - [OK] {rc} exists")
        else:
            print(f" - [MISSING] {rc}")

    print("\nVerification JSON check complete.")
    print(f"New report ID: {os.path.basename(report_path).replace('.json', '')}")

if __name__ == "__main__":
    verify()
