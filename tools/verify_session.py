import os
import sys
from webapp.app import app

def verify():
    print("Checking SECRET_KEY...")
    secret_key = app.config.get('SECRET_KEY')
    if secret_key:
        print(f"SUCCESS: SECRET_KEY is set to: {secret_key}")
    else:
        print("FAILURE: SECRET_KEY is not set.")
        sys.exit(1)

    print("\nChecking UPLOAD_FOLDER...")
    upload_folder = app.config.get('UPLOAD_FOLDER')
    if upload_folder:
        print(f"SUCCESS: UPLOAD_FOLDER is set to: {upload_folder}")
        if os.path.exists(upload_folder):
            print(f"SUCCESS: UPLOAD_FOLDER exists.")
        else:
            print(f"FAILURE: UPLOAD_FOLDER does not exist.")
            sys.exit(1)
    else:
        print("FAILURE: UPLOAD_FOLDER is not set.")
        sys.exit(1)

    print("\nChecking Flask session...")
    with app.test_request_context():
        from flask import session
        try:
            session['test'] = 'value'
            if session.get('test') == 'value':
                print("SUCCESS: Session is working correctly.")
            else:
                print("FAILURE: Session value not stored.")
                sys.exit(1)
        except RuntimeError as e:
            print(f"FAILURE: Session error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    verify()
