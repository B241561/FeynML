from .app import app
from .extensions import db

# Compatibility factory: return the Flask app instance created in webapp.app
def create_app():
    """Return the existing Flask application instance for compatibility."""
    return app

# Expose db at package level for `from webapp import db` imports
__all__ = ["create_app", "db", "app"]
