from webapp.app import app
from webapp.extensions import db, migrate

migrate.init_app(app, db)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
