"""Create sample SQLite DB (company.db) with employees table."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "company.db"


def create_db():
    if DB_PATH.exists():
        return
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE employees (
            employee_id INTEGER PRIMARY KEY,
            name TEXT,
            department TEXT,
            salary INTEGER,
            age INTEGER
        )
        """
    )
    cur.executemany(
        "INSERT INTO employees (name, department, salary, age) VALUES (?, ?, ?, ?)",
        [
            ("Aarav", "Engineering", 85000, 28),
            ("Priya", "Marketing", 52000, 31),
            ("Rohan", "Engineering", 72000, 26),
            ("Sneha", "HR", 45000, 35),
            ("Karan", "Sales", 48000, 29),
            ("Meera", "Sales", 61000, 38),
            ("Vikram", "Marketing", 39000, 24),
            ("Anjali", "HR", 55000, 41),
        ],
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    create_db()
    print("company.db ready")
