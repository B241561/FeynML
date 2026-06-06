import os
import pytest
from webapp.app import app, runner

def test_analysis_startup():
    with app.test_request_context():
        filepath = os.path.abspath("webapp/uploads/house-price.csv")
        config = {
            'target_col': 'price',
            'pred_col': None,
            'sensitive_col': None,
            'timestamp_col': None,
            'auto_predict': True
        }
        # This should at least trigger the startup logic
        runner.run(filepath, config)
        assert runner.status in ["running", "completed", "failed"]
