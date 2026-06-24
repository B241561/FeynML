import sys
import os
# Ensure repository root is on sys.path for imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from webapp import create_app, db
from webapp.models import AdminProfile

app = create_app()
app.config.setdefault('TESTING', True)
app.config.setdefault('SERVER_NAME', 'localhost')

with app.app_context():
    db.create_all()
    admin = AdminProfile.query.filter_by(username='Feyn_admin').first()
    if not admin:
        admin = AdminProfile(username='Feyn_admin', email='admin@feynml.com')
        admin.set_password('SecurePassword123!')
        db.session.add(admin)
        db.session.commit()

    client = app.test_client()
    with client.session_transaction() as sess:
        sess['otp_verified'] = True
        sess['otp_verified_admin_id'] = admin.id
        sess['otp_verified_admin_username'] = admin.username

    resp = client.post('/admin/reset-password', data={
        'password': 'weak',
        'password_confirm': 'weak',
        'admin_username': 'Feyn_admin'
    }, follow_redirects=True)

    print('STATUS:', resp.status_code)
    print('LENGTH:', len(resp.data))
    print(resp.data.decode('utf-8', errors='replace'))
