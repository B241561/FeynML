from webapp.app import app
from webapp.models import User

def verify():
    print(f"Using database at: {app.config['SQLALCHEMY_DATABASE_URI']}")
    with app.app_context():
        try:
            user_count = User.query.count()
            print(f"SUCCESS: Found {user_count} users in the database.")
            first_user = User.query.first()
            if first_user:
                print(f"SUCCESS: First user email: {first_user.email}")
            else:
                print("INFO: No users found, but table exists.")
        except Exception as e:
            print(f"FAILURE: Database error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    verify()
