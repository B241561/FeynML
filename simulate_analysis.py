import os
import traceback
from webapp.app import app, runner
from flask import session

def simulate_run_analysis():
    print("DEBUG: Starting simulation")
    with app.test_request_context():
        # Setup session
        with app.test_client() as client:
            with client.session_transaction() as sess:
                sess['filename'] = 'house-price.csv'
                sess['analysis_config'] = {
                    'target_col': 'price',
                    'pred_col': None,
                    'sensitive_col': None,
                    'timestamp_col': None,
                    'auto_predict': True
                }
            
            filepath = os.path.abspath("webapp/uploads/house-price.csv")
            print(f"DEBUG: Calling runner.run with {filepath}")
            try:
                runner.run(filepath, sess['analysis_config'])
                print("DEBUG: runner.run called successfully")
            except Exception as e:
                print("=" * 80)
                print("SIMULATED ERROR")
                traceback.print_exc()
                print("=" * 80)

if __name__ == "__main__":
    simulate_run_analysis()
