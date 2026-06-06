from .app import app

# Register Phase 4 routes blueprint
try:
	from webapp.routes.phase4_routes import phase4_bp
	app.register_blueprint(phase4_bp)
except Exception:
	# Fail quietly during import-time operations (tests, linters)
	pass
