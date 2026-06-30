import os
import sys
print("DEBUG: Script starting")
import traceback
import json
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from webapp.app import app, runner

def reproduce():
    with open('debug_output.txt', 'w') as f:
        f.write("Starting reproduction script...\n")
        
        # Configuration
        filepath = os.path.abspath("webapp/uploads/house-price.csv")
        config = {
            'target_col': 'price',
            'pred_col': None,
            'sensitive_col': None,
            'timestamp_col': None,
            'auto_predict': True
        }
        
        f.write(f"Filepath: {filepath}\n")
        f.write(f"Config: {config}\n")
        
        # Ensure the file exists
        if not os.path.exists(filepath):
            f.write(f"ERROR: File {filepath} not found.\n")
            return

        # Use app context
        with app.app_context():
            try:
                f.write("Calling runner._execute directly to capture traceback...\n")
                runner._execute(filepath, config)
                f.write("Analysis finished.\n")
                f.write(f"Runner status: {runner.status}\n")
                f.write(f"Runner error: {runner.error}\n")
                f.write("Runner logs:\n")
                for log in runner.logs:
                    f.write(f"  {log}\n")
            except Exception as e:
                f.write("\n" + "="*80 + "\n")
                f.write("REPRODUCED EXCEPTION:\n")
                traceback.print_exc(file=f)
                f.write("="*80 + "\n")

if __name__ == "__main__":
    reproduce()
