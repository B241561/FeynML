import os
import json
import traceback
import sys
import pathlib

# Ensure project root is on sys.path so `webapp` imports work
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from webapp.app import app
from webapp.extensions import db
from webapp.models import User, Dataset, Analysis, Report

try:
    with app.app_context():
        db.create_all()
        # Create test user
        email = 'test+ve@example.com'
        user = User.query.filter_by(email=email).first()
        if not user:
            user = User(name='Test User', email=email)
            user.set_password('password')
            db.session.add(user)
            db.session.commit()
        else:
            user.set_password('password')
            db.session.commit()

        # Prepare uploads
        uploads = app.config['UPLOAD_FOLDER']
        os.makedirs(uploads, exist_ok=True)
        csvpath = os.path.join(uploads, 'test_data.csv')
        import pandas as pd
        df = pd.DataFrame({'a': list(range(100)), 'b': [x*2 for x in range(100)], 'c': [1 for _ in range(100)]})
        df.to_csv(csvpath, index=False)

        ds = Dataset.query.filter_by(user_id=user.id, original_name='test_data.csv').first()
        if not ds:
            ds = Dataset(user_id=user.id, original_name='test_data.csv', stored_name='test_data.csv', upload_path=csvpath, size_bytes=os.path.getsize(csvpath))
            db.session.add(ds)
            db.session.commit()

        report_id = 'testreport1'
        an = Analysis.query.filter_by(report_id=report_id, user_id=user.id).first()
        if not an:
            an = Analysis(user_id=user.id, dataset_name='test_data.csv', report_id=report_id, status='COMPLETED')
            db.session.add(an)
            db.session.commit()

        rpt = Report.query.filter_by(report_id=report_id, user_id=user.id).first()
        if not rpt:
            rpt = Report(user_id=user.id, analysis_id=an.id, report_id=report_id, title='Test Report', filename=f'{report_id}.json', size_bytes=0)
            db.session.add(rpt)
            db.session.commit()

        # Write report JSON
        reports_dir = os.path.join(os.path.dirname(__file__), '..', 'webapp', 'reports')
        reports_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'webapp', 'reports'))
        os.makedirs(reports_dir, exist_ok=True)
        report_path = os.path.join(reports_dir, f'{report_id}.json')
        with open(report_path, 'w') as f:
            json.dump({'dataset': {'name': 'test_data.csv'}, 'drift':{}, 'calibration':{}, 'label_noise':{}, 'leakage':{}, 'missing_data':{}}, f)

    # Use test client
    with app.test_client() as client:
        # Login
        rv = client.post('/login', data={'email': email, 'password': 'password'}, follow_redirects=True)
        print('Login:', rv.status_code)
        if rv.status_code != 200:
            print('Login failed, response data:', rv.data[:200])

        def post_chart(payload):
            r = client.post('/api/generate-chart', json=payload)
            try:
                j = r.get_json()
            except Exception:
                j = None
            return r.status_code, j

        print('Testing Scatter...')
        st, sj = post_chart({'report_id': report_id, 'chart_type': 'scatter', 'x_axis': 'a', 'y_axis': 'b'})
        print('Scatter:', st, type(sj))

        print('Testing Histogram...')
        st, sj = post_chart({'report_id': report_id, 'chart_type': 'histogram', 'x_axis': 'a'})
        print('Histogram:', st)

        print('Testing Heatmap...')
        st, sj = post_chart({'report_id': report_id, 'chart_type': 'heatmap'})
        print('Heatmap:', st)

        print('Testing Violin...')
        st, sj = post_chart({'report_id': report_id, 'chart_type': 'violin', 'x_axis': 'c', 'y_axis': 'b'})
        print('Violin:', st)

        # Add to report
        print('Testing add_to_report...')
        # Reuse previous scatter chart JSON if available
        st_scatter, scatter_json = post_chart({'report_id': report_id, 'chart_type': 'scatter', 'x_axis': 'a', 'y_axis': 'b'})
        chart_json = None
        if scatter_json:
            chart_json = scatter_json.get('chart_json') or scatter_json.get('chart')
        rv = client.post('/add_to_report', json={'report_id': report_id, 'chart_data': chart_json, 'chart_title': 'Test Scatter'})
        print('add_to_report:', rv.status_code, rv.get_json())

        print('Testing save_visualization...')
        rv = client.post('/api/save-visualization', json={'report_id': report_id, 'chart_data': chart_json, 'chart_title': 'Saved Scatter', 'chart_description': 'desc'})
        print('save_visualization:', rv.status_code, rv.get_json())

except Exception:
    traceback.print_exc()
    raise
