"""
Quick runner for local testing (synchronous) — invokes AnalysisRunner._execute
Use from project root (Windows cmd.exe):

python run_local.py C:\path\to\loan_dataset.csv target_column pred_column sensitive_column

This will run analysis and print report path.
"""
import sys
import time
from webapp.services.analysis_runner import runner

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python run_local.py <csv_path> <target_col> [pred_col] [sensitive_col]")
        sys.exit(1)

    filepath = sys.argv[1]
    target_col = sys.argv[2]
    pred_col = sys.argv[3] if len(sys.argv) > 3 else None
    sensitive_col = sys.argv[4] if len(sys.argv) > 4 else None

    config = {
        'target_col': target_col,
        'pred_col': pred_col,
        'sensitive_col': sensitive_col,
        'auto_predict': False
    }

    print(f"Starting analysis for {filepath} (target={target_col}, pred={pred_col}, sensitive={sensitive_col})")
    # Call private method synchronously for testing convenience
    try:
        runner._execute(filepath, config)
        print(f"Report saved to: {runner.report_path}")
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise
