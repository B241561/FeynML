# Database Migrations

This project uses Flask-Migrate and Alembic for database migrations.

To initialize the migrations directory and create the initial schema, run:

```bash
# from the repository root
flask --app manage db init
flask --app manage db migrate -m "Initial schema"
flask --app manage db upgrade
```

If you are already using SQLite, the fallback database file is created at `feynml.db`.
