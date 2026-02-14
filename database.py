import sqlite3
import datetime
from pathlib import Path

DB_PATH = "sessions.db"

def init_db():
    """Create the sessions table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            total_beans INTEGER NOT NULL,
            good_beans INTEGER NOT NULL,
            bad_beans INTEGER NOT NULL,
            avg_confidence REAL,
            duration_sec REAL,
            report_text TEXT
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"✅ Database initialized: {DB_PATH}")

def save_session(timestamp, total, good, bad, avg_conf, duration, report):
    """Save a new session to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO sessions (timestamp, total_beans, good_beans, bad_beans, avg_confidence, duration_sec, report_text)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, total, good, bad, avg_conf, duration, report))
    
    conn.commit()
    session_id = cursor.lastrowid
    conn.close()
    
    return session_id

def get_all_sessions():
    """Fetch all sessions, sorted by newest first."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            id, 
            timestamp as start_time, 
            total_beans as total_count, 
            good_beans as good_count, 
            bad_beans as bad_count,
            report_text as ai_report
        FROM sessions 
        ORDER BY timestamp DESC
    """)
    
    sessions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return sessions

def get_session(session_id):
    """Fetch a single session by ID."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            id, 
            timestamp as start_time, 
            total_beans as total_count, 
            good_beans as good_count, 
            bad_beans as bad_count,
            report_text as ai_report
        FROM sessions WHERE id = ?
    """, (session_id,))
    row = cursor.fetchone()
    conn.close()
    
    return dict(row) if row else None

def delete_session(session_id):
    """Delete a session by ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()

def get_stats():
    """Get overall statistics."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total_sessions,
            SUM(total_beans) as total_beans_sum,
            SUM(good_beans) as total_good_sum
        FROM sessions
    """)
    
    row = cursor.fetchone()
    conn.close()
    
    total_sessions = row[0] or 0
    total_beans = row[1] or 0
    total_good = row[2] or 0
    
    avg_good = (total_good / total_beans * 100) if total_beans > 0 else 0
    
    return {
        "total_sessions": total_sessions,
        "total_beans": total_beans,
        "avg_good_percent": avg_good
    }

if __name__ == "__main__":
    # Initialize database when run directly
    init_db()
    print("Database ready!")
