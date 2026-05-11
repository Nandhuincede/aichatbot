import sqlite3
import os
import logging
from datetime import datetime, timezone, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'chatbot.db')

IST = timezone(timedelta(hours=5, minutes=30))

def now_ist() -> str:
    """Return current IST time as a string for DB storage."""
    return datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'active'
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS contact_details (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        name TEXT,
        phone TEXT,
        email TEXT,
        description TEXT,
        collected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS conversation_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        message TEXT NOT NULL,
        step TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS error_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        error_type TEXT,
        error_message TEXT,
        context TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')

    conn.commit()
    conn.close()
    logging.info("Database initialized successfully.")

def create_session(session_id: str):
    conn = get_db()
    try:
        ts = now_ist()
        conn.execute(
            "INSERT INTO sessions (session_id, created_at, updated_at) VALUES (?, ?, ?)",
            (session_id, ts, ts)
        )
        conn.commit()
    finally:
        conn.close()

def update_session(session_id: str):
    conn = get_db()
    try:
        conn.execute("UPDATE sessions SET updated_at=? WHERE session_id=?", (now_ist(), session_id))
        conn.commit()
    finally:
        conn.close()

def close_session(session_id: str):
    """Mark session as completed when conversation finishes."""
    conn = get_db()
    try:
        conn.execute(
            "UPDATE sessions SET status='completed', updated_at=? WHERE session_id=?",
            (now_ist(), session_id)
        )
        conn.commit()
    finally:
        conn.close()


def log_message(session_id: str, role: str, message: str, step: str = None):
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO conversation_logs (session_id, role, message, step, timestamp) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, message, step, now_ist())
        )
        conn.commit()
    finally:
        conn.close()

def log_error(session_id: str, error_type: str, error_message: str, context: str = None):
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO error_logs (session_id, error_type, error_message, context, timestamp) VALUES (?, ?, ?, ?, ?)",
            (session_id, error_type, error_message, context, now_ist())
        )
        conn.commit()
    finally:
        conn.close()

def save_contact(session_id: str, name: str = None, phone: str = None, email: str = None, description: str = None):
    conn = get_db()
    try:
        existing = conn.execute("SELECT id FROM contact_details WHERE session_id=?", (session_id,)).fetchone()
        if existing:
            conn.execute(
                "UPDATE contact_details SET name=COALESCE(?,name), phone=COALESCE(?,phone), email=COALESCE(?,email), description=COALESCE(?,description) WHERE session_id=?",
                (name, phone, email, description, session_id)
            )
        else:
            conn.execute(
                "INSERT INTO contact_details (session_id, name, phone, email, description) VALUES (?,?,?,?,?)",
                (session_id, name, phone, email, description)
            )
        conn.commit()
    finally:
        conn.close()

def get_all_sessions(page: int = 1, per_page: int = 10, status: str = None, search: str = None):
    conn = get_db()
    try:
        # Build dynamic WHERE clause
        conditions = []
        params = []

        if status == 'completed':
            conditions.append("s.status = 'completed'")
        elif status == 'active':
            conditions.append("s.status != 'completed'")

        if search:
            like = f"%{search}%"
            conditions.append("(cd.name LIKE ? OR cd.phone LIKE ? OR cd.email LIKE ?)")
            params.extend([like, like, like])

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        base_query = f"""
            FROM sessions s
            LEFT JOIN contact_details cd ON s.session_id = cd.session_id
            {where_clause}
        """

        # Filtered total (for pagination info)
        total = conn.execute(f"SELECT COUNT(*) {base_query}", params).fetchone()[0]

        # Global stats (unfiltered)
        total_all = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        completed_count = conn.execute("SELECT COUNT(*) FROM sessions WHERE status='completed'").fetchone()[0]

        # Today count in IST (stored as IST strings)
        today_ist = datetime.now(IST).strftime('%Y-%m-%d')
        today_count = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE created_at LIKE ?",
            (f"{today_ist}%",)
        ).fetchone()[0]

        offset = (page - 1) * per_page
        rows = conn.execute(f"""
            SELECT s.session_id, s.created_at, s.updated_at, s.status,
                   cd.name, cd.phone, cd.email, cd.description
            {base_query}
            ORDER BY s.updated_at DESC
            LIMIT ? OFFSET ?
        """, params + [per_page, offset]).fetchall()

        return {
            "sessions": [dict(r) for r in rows],
            "total": total,
            "total_all": total_all,
            "completed_count": completed_count,
            "today_count": today_count,
            "page": page,
            "per_page": per_page,
            "total_pages": max(1, (total + per_page - 1) // per_page)
        }
    finally:
        conn.close()

def get_session_detail(session_id: str):
    conn = get_db()
    try:
        session = conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        contact = conn.execute("SELECT * FROM contact_details WHERE session_id=?", (session_id,)).fetchone()
        convo = conn.execute(
            "SELECT * FROM conversation_logs WHERE session_id=? ORDER BY timestamp ASC", (session_id,)
        ).fetchall()
        errors = conn.execute(
            "SELECT * FROM error_logs WHERE session_id=? ORDER BY timestamp ASC", (session_id,)
        ).fetchall()
        return {
            "session": dict(session) if session else {},
            "contact": dict(contact) if contact else {},
            "conversation": [dict(r) for r in convo],
            "errors": [dict(r) for r in errors]
        }
    finally:
        conn.close()

def get_sessions_paginated(page: int = 1, per_page: int = 10):
    conn = get_db()
    try:
        offset = (page - 1) * per_page
        total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        rows = conn.execute("""
            SELECT s.session_id, s.created_at, s.updated_at, s.status,
                   cd.name, cd.phone, cd.email, cd.description
            FROM sessions s
            LEFT JOIN contact_details cd ON s.session_id = cd.session_id
            ORDER BY s.updated_at DESC
            LIMIT ? OFFSET ?
        """, (per_page, offset)).fetchall()
        return {
            "sessions": [dict(r) for r in rows],
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page
        }
    finally:
        conn.close()