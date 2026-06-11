import sqlite3
from datetime import datetime

DB_NAME = "neuro.db"


# ================= INIT DB =================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image TEXT,
        prediction TEXT,
        confidence REAL,
        calibrated_conf REAL,
        uncertainty REAL,
        report TEXT,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()


# ================= INSERT REPORT =================
def insert_report(image, prediction, confidence, calibrated_conf, uncertainty, report_text):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # ISO format ensures correct sorting
    created_at = datetime.now().isoformat()

    c.execute("""
        INSERT INTO reports (
            image,
            prediction,
            confidence,
            calibrated_conf,
            uncertainty,
            report,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        image,
        prediction,
        confidence,
        calibrated_conf,
        uncertainty,
        report_text,
        created_at
    ))

    conn.commit()
    conn.close()


# ================= GET HISTORY (LATEST FIRST) =================
def get_reports():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("""
        SELECT * FROM reports
        ORDER BY created_at DESC
    """)

    rows = c.fetchall()
    conn.close()

    return rows