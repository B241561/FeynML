# Admin Authentication System - Implementation Guide

## Overview

This document describes the extended Admin Authentication System for FeynML. The system enforces a single administrator account (`Feyn_admin`) with secure password recovery via OTP (One-Time Password).

## Architecture

### Key Components

1. **AdminProfile Model** - Stores the single admin account
2. **OTPToken Model** - Manages OTP generation and verification
3. **Email Service** - Handles sending OTP and confirmation emails
4. **Password Recovery Routes** - Three-step recovery flow

### Database Schema

#### AdminProfile Table
```sql
CREATE TABLE admin_profiles (
    id INTEGER PRIMARY KEY,
    username VARCHAR(128) UNIQUE NOT NULL,  -- Always 'Feyn_admin'
    email VARCHAR(180) UNIQUE NOT NULL,
    password_hash VARCHAR(256) NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
)
```

#### OTPToken Table
```sql
CREATE TABLE otp_tokens (
    id INTEGER PRIMARY KEY,
    admin_id INTEGER NOT NULL REFERENCES admin_profiles(id),
    otp_code VARCHAR(6) NOT NULL,           -- 6-digit numeric code
    is_used BOOLEAN DEFAULT 0,
    created_at DATETIME NOT NULL,
    expires_at DATETIME NOT NULL            -- 10 minutes from creation
)
```

## Setup Instructions

### 1. Database Migration

Apply the migration to create the necessary tables:

```bash
# Option A: Using Flask-Migrate
flask db upgrade

# Option B: Manual (if migration not available)
python -c "from webapp import db, create_app; app = create_app(); db.create_all()"
```

### 2. Initialize Admin Profile

Create the initial admin account:

```bash
# Interactive setup
python init_admin.py

# With email specified
python init_admin.py --email admin@yourdomain.com

# Reset existing admin password
python init_admin.py --reset
```

Follow the prompts to set a secure password. The admin username will always be `Feyn_admin`.

### 3. Configure Email Service (Optional)

For email OTP delivery, configure environment variables:

```bash
# Enable email sending
export MAIL_ENABLED=true
export MAIL_SERVER=smtp.gmail.com
export MAIL_PORT=587
export MAIL_USE_TLS=true
export MAIL_USERNAME=your-email@gmail.com
export MAIL_PASSWORD=your-app-password
export MAIL_DEFAULT_SENDER=noreply@yourdomain.com
```

If `MAIL_ENABLED=false` (default), emails are logged to console only.

## API Routes

### 1. Admin Login
**Route:** `/admin/login`  
**Method:** GET, POST  
**Description:** Authenticate with username and password

**POST Parameters:**
```json
{
  "username": "Feyn_admin",
  "password": "SecurePassword123!"
}
```

**Response:**
- Success: Redirects to `/admin`
- Failure: Shows error message

### 2. Forgot Password
**Route:** `/admin/forgot-password`  
**Method:** GET, POST  
**Description:** Step 1 of password recovery - request OTP

**POST Parameters:**
```json
{
  "username": "Feyn_admin",
  "email": "admin@yourdomain.com"
}
```

**Validations:**
- Username must be exactly `Feyn_admin`
- Email must match registered admin email
- Returns generic error if admin not found (security)

**Response:**
- Success: Redirects to `/admin/verify-otp`
- Failure: Shows error and returns to form

### 3. Verify OTP
**Route:** `/admin/verify-otp`  
**Method:** GET, POST  
**Description:** Step 2 of password recovery - verify OTP code

**POST Parameters:**
```json
{
  "otp_code": "123456",
  "admin_username": "Feyn_admin"
}
```

**Validations:**
- OTP must be 6 digits
- OTP must not be expired (10-minute window)
- OTP must not have been used already
- Username must match

**Response:**
- Success: Redirects to `/admin/reset-password`
- Failure: Shows error with remaining time

### 4. Reset Password
**Route:** `/admin/reset-password`  
**Method:** GET, POST  
**Description:** Step 3 of password recovery - set new password

**POST Parameters:**
```json
{
  "password": "NewSecurePassword123!",
  "password_confirm": "NewSecurePassword123!",
  "admin_username": "Feyn_admin"
}
```

**Validations:**
- Password minimum 8 characters
- Passwords must match
- Recommended: Mix of upper, lower, digits, special chars
- Username must match

**Response:**
- Success: Redirects to `/admin/login` with confirmation
- Failure: Shows error and returns to form

### 5. Admin Logout
**Route:** `/admin/logout`  
**Method:** GET  
**Description:** End admin session

**Response:**
- Redirects to `/admin/login` with logout message

## Security Features

### Single Admin Enforcement
- Only one `AdminProfile` record allowed
- Application logic prevents creation of secondary admin accounts
- No admin registration endpoints

### Password Security
- Passwords hashed with bcrypt (cost factor: 12)
- Passwords never logged or transmitted in plaintext
- Updated timestamp tracks password changes

### OTP Security
- 6-digit numeric code (1 million possible combinations)
- 10-minute expiration window
- Single-use only - marked as `is_used` after verification
- Stored securely in database with encryption recommended for production

### Rate Limiting
- Admin login: 5 failed attempts = 15-minute lockout
- Prevents brute force attacks

### Email Verification
- OTP only sent to registered admin email
- Email address validated on form submission
- Confirmation emails sent after password reset

### Session Management
- Session data cleared after each step
- Cannot skip steps in password recovery flow
- Session timeout recommendations: 30 minutes for admin sessions

## Email Templates

### OTP Email
```
Subject: FeynML Admin - Password Reset OTP

Dear Admin,

Your OTP (One-Time Password) for password reset is:

[OTP_CODE]

This OTP will expire in 10 minutes. Please do not share this code with anyone.

If you did not request this code, please ignore this email.

Best regards,
FeynML Admin System
```

### Password Reset Confirmation
```
Subject: FeynML Admin - Password Successfully Reset

Dear Admin,

Your password has been successfully reset on [TIMESTAMP] UTC.

You can now log in at: /admin/login

If this was not you, please contact system administrator immediately.

Best regards,
FeynML Admin System
```

## Testing

Run comprehensive test suite:

```bash
# Run all admin auth tests
pytest tests/test_admin_auth.py -v

# Run specific test class
pytest tests/test_admin_auth.py::TestAdminProfileModel -v

# Run with coverage
pytest tests/test_admin_auth.py --cov=webapp --cov-report=html
```

### Test Categories
- **Model Tests**: AdminProfile and OTPToken creation and validation
- **Route Tests**: All three password recovery routes
- **Security Tests**: Enforcement of single admin, password hashing, OTP expiry
- **Access Control Tests**: Protected route redirects
- **Integration Tests**: Full password recovery flow

## Verification Checklist

```
✓ Only Feyn_admin can log in
✓ Only Feyn_admin can request OTP
✓ OTP sent only to registered admin email
✓ Password reset works with valid OTP
✓ Login works with new password
✓ No additional admin accounts can exist
✓ Failed login attempts trigger rate limiting
✓ Session cleared after logout
✓ OTP expires after 10 minutes
✓ OTP can only be used once
✓ Passwords are hashed with bcrypt
✓ Email service logs to console in dev mode
```

## Troubleshooting

### OTP Not Being Sent
1. Check `MAIL_ENABLED` environment variable
2. In dev mode, check application logs for email content
3. Verify email configuration matches your SMTP provider
4. Check firewall/network restrictions on SMTP port

### Admin Profile Not Found
1. Run `python init_admin.py` to create admin profile
2. Verify admin profile exists: `python -c "from webapp.models import AdminProfile; from webapp import db, create_app; app = create_app(); print(AdminProfile.query.all())"`

### OTP Verification Failing
1. Verify OTP hasn't expired (10-minute window)
2. Verify OTP matches exactly (copy-paste from email)
3. Verify OTP hasn't been used already
4. Check session data in browser dev tools

### Password Reset Not Working
1. Verify new password meets requirements (8+ chars)
2. Verify password confirmation matches exactly
3. Check database permissions for UPDATE
4. Check application logs for specific errors

## Production Deployment

### Recommendations

1. **Database**
   - Use PostgreSQL or MySQL (not SQLite)
   - Enable column encryption for email field
   - Regular backups

2. **Email Service**
   - Use production SMTP provider (SendGrid, AWS SES, etc.)
   - Implement email templates with HTML
   - Set up bounce handling

3. **Security**
   - Enable HTTPS only
   - Set secure session cookies (`SESSION_COOKIE_SECURE=True`)
   - Implement CSRF protection
   - Add rate limiting middleware
   - Log all admin actions

4. **Monitoring**
   - Alert on multiple failed login attempts
   - Monitor OTP request frequency
   - Track admin session duration
   - Log password reset events

5. **Backup & Recovery**
   - Regular database backups
   - Document admin recovery procedures
   - Keep offline backup of admin credentials

## API Examples

### Complete Password Recovery Flow

```bash
# Step 1: Request OTP
curl -X POST http://localhost:5000/admin/forgot-password \
  -d "username=Feyn_admin&email=admin@example.com"

# Step 2: Verify OTP
curl -X POST http://localhost:5000/admin/verify-otp \
  -d "otp_code=123456&admin_username=Feyn_admin"

# Step 3: Reset Password
curl -X POST http://localhost:5000/admin/reset-password \
  -d "password=NewPassword123!&password_confirm=NewPassword123!&admin_username=Feyn_admin"
```

## File Structure

```
webapp/
├── models.py                          # AdminProfile, OTPToken models
├── app.py                             # Routes: forgot-password, verify-otp, reset-password
├── services/
│   └── email_service.py              # EmailService for OTP delivery
├── templates/
│   ├── admin_forgot_password.html    # Step 1: Request OTP
│   ├── admin_verify_otp.html         # Step 2: Verify OTP
│   └── admin_reset_password.html     # Step 3: Reset password
└── migrations/
    └── versions/
        └── 001_add_admin_auth.py     # Database migration

tests/
└── test_admin_auth.py                # Comprehensive test suite

init_admin.py                         # Admin profile initialization script
```

## Support & Maintenance

### Regular Tasks

1. **Monthly**: Review admin login logs for suspicious activity
2. **Quarterly**: Audit admin password age, recommend refresh if > 90 days
3. **Semi-annually**: Review and update security policies
4. **Annually**: Security audit of admin authentication system

### Monitoring Dashboard Metrics

- Admin login success/failure rate
- OTP request frequency
- Password reset frequency
- Average admin session duration
- Failed login attempts by IP

## References

- NIST Password Guidelines: https://pages.nist.gov/800-63-3/
- OWASP Authentication Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html
- RFC 6238 (TOTP): https://tools.ietf.org/html/rfc6238

---

**Last Updated:** June 2024  
**Version:** 1.0  
**Status:** Production Ready
