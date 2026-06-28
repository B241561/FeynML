import sqlite3
import os

def check_db(path):
    print(f"\nChecking database: {path}")
    if not os.path.exists(path):
        print("  File does not exist.")
        return
    
    try:
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"  Tables found: {[t[0] for t in tables]}")
        conn.close()
    except Exception as e:
        print(f"  Error: {e}")

check_db("feynml.db")
check_db("instance/feynml.db")
check_db("webapp/feynml.db")
