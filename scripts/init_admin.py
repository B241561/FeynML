#!/usr/bin/env python3
"""
Admin Profile Initialization Script

This script initializes the admin profile with username 'Feyn_admin'.
Only ONE admin account is allowed to exist.

Usage:
    python init_admin.py
    python init_admin.py --email admin@example.com
    python init_admin.py --reset  # Reset existing admin password
"""

import os
import sys
import argparse
from getpass import getpass
from datetime import datetime

# Add the parent directory to path so we can import webapp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from webapp import db
from webapp.models import AdminProfile
from webapp.extensions import bcrypt
from flask import Flask


def create_app():
    """Create Flask app context."""
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
        'DATABASE_URL'
    ) or f"sqlite:///{os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'feynml.db')}"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    bcrypt.init_app(app)
    return app


def init_admin(email=None, reset=False):
    """
    Initialize admin profile.
    
    Args:
        email: Email address for admin (prompted if not provided)
        reset: If True, reset existing admin password
    """
    app = create_app()

    with app.app_context():
        # Create tables if they don't exist
        db.create_all()

        # Check if admin already exists
        existing_admin = AdminProfile.query.filter_by(username='Feyn_admin').first()

        if existing_admin and not reset:
            print("✓ Admin profile 'Feyn_admin' already exists.")
            print(f"  Email: {existing_admin.email}")
            print(f"  Created: {existing_admin.created_at}")
            print(f"  Updated: {existing_admin.updated_at}")
            print("\nUse --reset flag to reset the password.")
            return True

        # Get email if not provided
        if not email:
            if existing_admin and reset:
                email = existing_admin.email
            else:
                email = input("Enter admin email address: ").strip().lower()
                if not email or '@' not in email:
                    print("✗ Invalid email address.")
                    return False

        # Validate email isn't already used by another admin
        if existing_admin and existing_admin.email != email and not reset:
            other_admin = AdminProfile.query.filter_by(email=email).first()
            if other_admin:
                print(f"✗ Email '{email}' is already associated with an admin account.")
                return False

        # Get password
        password = getpass("Enter admin password: ").strip()
        if not password or len(password) < 8:
            print("✗ Password must be at least 8 characters long.")
            return False

        # Confirm password
        password_confirm = getpass("Confirm admin password: ").strip()
        if password != password_confirm:
            print("✗ Passwords do not match.")
            return False

        # Validate password strength
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in '!@#$%^&*' for c in password)

        criteria_count = sum([has_upper, has_lower, has_digit, has_special])
        if criteria_count < 3:
            print("✗ Password should contain uppercase, lowercase, numbers, and special characters.")
            print("  Current strength: {}/4 criteria met".format(criteria_count))
            return False

        # Create or update admin profile
        try:
            if existing_admin and reset:
                admin = existing_admin
                admin.set_password(password)
                admin.updated_at = datetime.utcnow()
                action = "reset"
            else:
                admin = AdminProfile(
                    username='Feyn_admin',
                    email=email
                )
                admin.set_password(password)
                db.session.add(admin)
                action = "created"

            db.session.commit()

            print(f"\n✓ Admin profile '{admin.username}' {action} successfully!")
            print(f"  Email: {admin.email}")
            print(f"  Created: {admin.created_at}")
            print(f"  Updated: {admin.updated_at}")
            print("\n✓ You can now log in at /admin/login with:")
            print(f"  Username: Feyn_admin")
            print("  Password: [the password you just entered]")
            return True

        except Exception as exc:
            print(f"✗ Error creating/updating admin profile: {exc}")
            db.session.rollback()
            return False


def main():
    """Parse arguments and run initialization."""
    parser = argparse.ArgumentParser(
        description='Initialize admin profile for FeynML'
    )
    parser.add_argument(
        '--email',
        help='Email address for admin (prompted if not provided)'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset password for existing admin'
    )

    args = parser.parse_args()

    if init_admin(email=args.email, reset=args.reset):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
