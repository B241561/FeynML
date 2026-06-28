import io
import os
import uuid
from sklearn.datasets import load_iris
from webapp.app import app
from webapp.extensions import db
from webapp.models import User

print('script start')
# Ensure DB exists and app context is ready
with app.app_context():
    db.create_all()

iris = load_iris()
cols = iris.feature_names + ["species"]
rows = [list(iris.data[i]) + [int(iris.target[i])] for i in range(len(iris.data))]

user_email = f"client_test_{uuid.uuid4().hex[:8]}@example.com"
password = "Password123"

with app.app_context():
    user = User(name="Client Test", email=user_email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

client = app.test_client()

# Signup route may also work; using direct test user creation above.

# Log in to receive session cookie
resp = client.post('/login', data={'email': user_email, 'password': password}, follow_redirects=True)
print('login', resp.status_code, resp.request.path)

# Create iris CSV in memory and upload
output = io.StringIO()
output.write(','.join(cols) + '\n')
for row in rows:
    output.write(','.join(map(str, row)) + '\n')
output.seek(0)
csv_bytes = output.getvalue().encode('utf-8')
resp = client.post('/upload', data={
    'dataset': (io.BytesIO(csv_bytes), 'test_iris.csv')
}, content_type='multipart/form-data', follow_redirects=True)
print('upload', resp.status_code, resp.request.path)

# Fetch label-noise page and parse first dataset ID
resp = client.get('/analysis/label-noise')
print('/analysis/label-noise GET', resp.status_code)
html = resp.get_data(as_text=True)
if '<option value="' not in html:
    raise SystemExit('No dataset option found')
dataset_id = html.split('<option value="')[1].split('"')[0]
print('dataset_id', dataset_id)

# Run label-noise analysis
resp = client.post('/analysis/label-noise/run', data={'dataset_id': dataset_id, 'target_col': 'species'}, follow_redirects=False)
print('label-noise run', resp.status_code, resp.location)

# Follow redirect to report page if any
if resp.status_code in (302, 303) and resp.location:
    report_url = resp.location
    if report_url.startswith('/'):
        report_url = report_url
    resp2 = client.get(report_url)
    print('report page', resp2.status_code, report_url)
    content = resp2.get_data(as_text=True)
    print('report snippet', content[:500].replace('\n',' '))
    report_id = report_url.split('/analysis/reports/phase4/')[-1]
    print('report_id', report_id)
    print('html_exists', os.path.exists(os.path.join('reports', 'generated', report_id + '.html')))
    print('json_exists', os.path.exists(os.path.join('reports', 'generated', report_id + '.json')))
else:
    print('unexpected run response', resp.status_code, resp.get_data(as_text=True)[:1000])
