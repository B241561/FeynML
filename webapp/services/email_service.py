"""
Email service for sending OTP and password reset notifications.
Supports both SMTP and development/testing modes.
"""

import os
import logging
from datetime import datetime
from flask import render_template_string

logger = logging.getLogger(__name__)


class EmailService:
    """Service for sending emails via SMTP or development mode."""

    def __init__(self, app=None):
        """Initialize email service with Flask app configuration."""
        self.app = app
        self.use_smtp = os.environ.get('MAIL_ENABLED', 'false').lower() == 'true'
        self.mail_server = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
        self.mail_port = int(os.environ.get('MAIL_PORT', 587))
        self.mail_use_tls = os.environ.get('MAIL_USE_TLS', 'true').lower() == 'true'
        self.mail_username = os.environ.get('MAIL_USERNAME', '')
        self.mail_password = os.environ.get('MAIL_PASSWORD', '')
        self.mail_default_sender = os.environ.get('MAIL_DEFAULT_SENDER', 'noreply@feynml.com')

    def send_otp_email(self, admin_email, otp_code, admin_username='Feyn_admin'):
        """
        Send OTP email to admin.
        
        Args:
            admin_email: Email address to send OTP to
            otp_code: 6-digit OTP code
            admin_username: Admin username for context
        
        Returns:
            bool: True if sent successfully or in dev mode, False if failed
        """
        if not self.use_smtp:
            logger.info(f"[DEV MODE] OTP Email would be sent to {admin_email}")
            logger.info(f"[DEV MODE] OTP Code: {otp_code}")
            logger.info(f"[DEV MODE] Recipient: {admin_username}")
            return True

        try:
            subject = "FeynML Admin - Password Reset OTP"
            body = f"""
            Dear Admin,

            Your OTP (One-Time Password) for password reset is:

            {otp_code}

            This OTP will expire in 10 minutes. Please do not share this code with anyone.

            If you did not request this code, please ignore this email.

            Best regards,
            FeynML Admin System
            """

            return self._send_via_smtp(admin_email, subject, body)
        except Exception as exc:
            logger.error(f"Failed to send OTP email to {admin_email}: {exc}")
            return False

    def send_password_reset_confirmation(self, admin_email, admin_username='Feyn_admin'):
        """
        Send password reset confirmation email.
        
        Args:
            admin_email: Email address to send confirmation to
            admin_username: Admin username for context
        
        Returns:
            bool: True if sent successfully or in dev mode, False if failed
        """
        if not self.use_smtp:
            logger.info(f"[DEV MODE] Password Reset Confirmation Email would be sent to {admin_email}")
            logger.info(f"[DEV MODE] Recipient: {admin_username}")
            logger.info(f"[DEV MODE] Reset Date: {datetime.utcnow().isoformat()}")
            return True

        try:
            subject = "FeynML Admin - Password Successfully Reset"
            body = f"""
            Dear Admin,

            Your password has been successfully reset on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC.

            You can now log in with your new password at: /admin/login

            If this was not you, please contact system administrator immediately.

            Best regards,
            FeynML Admin System
            """

            return self._send_via_smtp(admin_email, subject, body)
        except Exception as exc:
            logger.error(f"Failed to send password reset confirmation to {admin_email}: {exc}")
            return False

    def send_failed_login_alert(self, admin_email, attempt_count=3):
        """
        Send alert for failed admin login attempts.
        
        Args:
            admin_email: Email address to send alert to
            attempt_count: Number of failed attempts
        
        Returns:
            bool: True if sent successfully or in dev mode, False if failed
        """
        if not self.use_smtp:
            logger.warning(f"[DEV MODE] Failed Login Alert Email would be sent to {admin_email}")
            logger.warning(f"[DEV MODE] Failed Attempts: {attempt_count}")
            return True

        try:
            subject = "FeynML Admin - Failed Login Attempts Alert"
            body = f"""
            Dear Admin,

            We detected {attempt_count} failed login attempts to your admin account on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC.

            If this was you, you can reset your password at: /admin/forgot-password

            If this was not you, please take immediate action to secure your account.

            Best regards,
            FeynML Admin System
            """

            return self._send_via_smtp(admin_email, subject, body)
        except Exception as exc:
            logger.error(f"Failed to send login alert to {admin_email}: {exc}")
            return False

    def _send_via_smtp(self, recipient, subject, body):
        """
        Internal method to send email via SMTP.
        
        Args:
            recipient: Email recipient
            subject: Email subject
            body: Email body text
        
        Returns:
            bool: True if sent successfully, False otherwise
        """
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.mail_default_sender
            msg['To'] = recipient

            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(self.mail_server, self.mail_port) as server:
                if self.mail_use_tls:
                    server.starttls()
                if self.mail_username and self.mail_password:
                    server.login(self.mail_username, self.mail_password)
                server.sendmail(self.mail_default_sender, recipient, msg.as_string())

            logger.info(f"Email sent successfully to {recipient}")
            return True
        except Exception as exc:
            logger.error(f"SMTP error while sending to {recipient}: {exc}")
            return False


def send_otp_email_direct(admin_email, otp_code, admin_username='Feyn_admin'):
    """
    Convenience function to send OTP email without instantiating service.
    Used when Flask app context might not be fully initialized.
    """
    service = EmailService()
    return service.send_otp_email(admin_email, otp_code, admin_username)


def send_password_reset_confirmation_direct(admin_email, admin_username='Feyn_admin'):
    """
    Convenience function to send password reset confirmation without instantiating service.
    Used when Flask app context might not be fully initialized.
    """
    service = EmailService()
    return service.send_password_reset_confirmation(admin_email, admin_username)
