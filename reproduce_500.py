import os
import sys
import json
import traceback

print("Starting reproduction script...")
try:
    from webapp.app import app, db
    from webapp.models import User
    print("Imports successful.")
except Exception:
    print("IMPORT ERROR:")
    traceback.print_exc()
    sys.exit(1)

from io import BytesIO

def reproduce():
    print("Setting up test client...")
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    
    client = app.test_client()
    
    with app.app_context():
        # Create user
        print("Ensuring test user exists...")
        user = User.query.filter_by(email='test@example.com').first()
        if not user:
            user = User(name='Test User', email='test@example.com')
            user.set_password('password123')
            db.session.add(user)
            db.session.commit()
        
        # Login
        print("Logging in...")
        client.post('/login', data={'email': 'test@example.com', 'password': 'password123'})
        
        # Upload file
        print("Uploading test CSV...")
        csv_content = "price,size,rooms\n100,10,1\n200,20,2\n300,30,3"
        data = {
            'dataset': (BytesIO(csv_content.encode()), 'test.csv')
        }
        client.post('/upload', data=data, content_type='multipart/form-data')
        
        # Run analysis
        config_data = {
            'target_col': 'price',
            'pred_col': '',
            'sensitive_col': '',
            'time_col': '',
            'auto_predict': 'true'
        }
        print("\nTriggering /run_analysis...")
        response = client.post('/run_analysis', data=config_data)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 500:
            print("REPRODUCED 500 ERROR IN /run_analysis")
            return

        # Check status
        print("\nChecking /analysis-status...")
        response = client.get('/analysis-status')
        print(f"Response status: {response.status_code}")
        print(f"Response data: {response.get_data(as_text=True)}")

if __name__ == "__main__":
    try:
        reproduce()
    except Exception:
        print("EXECUTION ERROR:")
        traceback.print_exc()
