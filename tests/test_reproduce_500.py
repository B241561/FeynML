import os
import pytest
import traceback
from webapp.app import app, runner

def test_run_analysis_traceback():
    print("\n--- STARTING ANALYSIS TRACEBACK TEST ---")
    with app.test_request_context():
        # Prepare session
        with app.test_client() as client:
            with client.session_transaction() as sess:
                sess['filename'] = 'house-price.csv'
            
            filepath = os.path.abspath("webapp/uploads/house-price.csv")
            config = {
                'target_col': 'price',
                'pred_col': None,
                'sensitive_col': None,
                'timestamp_col': None,
                'auto_predict': True
            }
            
            print(f"DEBUG: filepath={filepath}")
            
            try:
                # Call the internal execution logic directly to avoid threading
                # and capture the exception in this main thread
                runner._execute(filepath, config)
                print("DEBUG: _execute finished without raising (check logs for internal engine failures)")
            except Exception as e:
                print("\n" + "!"*40)
                print("CAPTURED EXCEPTION IN TEST:")
                traceback.print_exc()
                print("!"*40 + "\n")
                raise e
