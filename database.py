import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(os.getenv("DATABASE_PATH", "sessions.db")).expanduser()


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(reset=False):
    """Initialize DB schema. If reset=True, recreate from scratch."""
    if reset and DB_PATH.exists():
        DB_PATH.unlink()

    conn = _connect()
    cursor = conn.cursor()

    cursor.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS remember_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token_hash TEXT NOT NULL UNIQUE,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            total_beans INTEGER NOT NULL,
            good_beans INTEGER NOT NULL,
            bad_beans INTEGER NOT NULL,
            avg_confidence REAL,
            duration_sec REAL,
            report_text TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_user_time
        ON sessions(user_id, timestamp DESC);

        CREATE INDEX IF NOT EXISTS idx_remember_tokens_hash
        ON remember_tokens(token_hash);
        """
    )

    conn.commit()
    conn.close()
    cleanup_expired_tokens()
    print(f"Database initialized: {DB_PATH}")


def create_user(username, password_hash):
    """Create a user. Returns (user_id, error_message)."""
    conn = _connect()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO users (username, password_hash, created_at)
            VALUES (?, ?, ?)
            """,
            (username, password_hash, _utc_now_iso()),
        )
        conn.commit()
        return cursor.lastrowid, None
    except sqlite3.IntegrityError:
        return None, "Username is already taken."
    finally:
        conn.close()


def get_user_by_username(username):
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, username, password_hash, created_at
        FROM users
        WHERE username = ?
        """,
        (username,),
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id):
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, username, created_at
        FROM users
        WHERE id = ?
        """,
        (user_id,),
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def save_remember_token(user_id, token_hash, expires_at):
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO remember_tokens (user_id, token_hash, expires_at, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (user_id, token_hash, expires_at, _utc_now_iso()),
    )
    conn.commit()
    conn.close()


def delete_remember_token(token_hash):
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM remember_tokens WHERE token_hash = ?", (token_hash,))
    conn.commit()
    conn.close()


def cleanup_expired_tokens():
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        DELETE FROM remember_tokens
        WHERE expires_at <= ?
        """,
        (_utc_now_iso(),),
    )
    conn.commit()
    conn.close()


def get_user_by_valid_token(token_hash):
    cleanup_expired_tokens()
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT u.id, u.username, u.created_at
        FROM remember_tokens t
        JOIN users u ON u.id = t.user_id
        WHERE t.token_hash = ? AND t.expires_at > ?
        LIMIT 1
        """,
        (token_hash, _utc_now_iso()),
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def save_session(user_id, timestamp, total, good, bad, avg_conf, duration, report):
    """Save a new session to the database for a specific user."""
    conn = _connect()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO sessions (
            user_id, timestamp, total_beans, good_beans, bad_beans,
            avg_confidence, duration_sec, report_text
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, timestamp, total, good, bad, avg_conf, duration, report),
    )

    conn.commit()
    session_id = cursor.lastrowid
    conn.close()
    return session_id


def get_all_sessions(user_id):
    """Fetch all sessions for one user, sorted by newest first."""
    conn = _connect()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            timestamp AS start_time,
            total_beans AS total_count,
            good_beans AS good_count,
            bad_beans AS bad_count,
            report_text AS ai_report
        FROM sessions
        WHERE user_id = ?
        ORDER BY timestamp DESC
        """,
        (user_id,),
    )

    sessions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return sessions


def get_session(user_id, session_id):
    """Fetch one session by ID scoped to user."""
    conn = _connect()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            timestamp AS start_time,
            total_beans AS total_count,
            good_beans AS good_count,
            bad_beans AS bad_count,
            report_text AS ai_report
        FROM sessions
        WHERE id = ? AND user_id = ?
        """,
        (session_id, user_id),
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_session(user_id, session_id):
    """Delete a session by ID scoped to user."""
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM sessions WHERE id = ? AND user_id = ?",
        (session_id, user_id),
    )
    conn.commit()
    conn.close()


def get_stats(user_id):
    """Get overall statistics for one user."""
    conn = _connect()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            COUNT(*) AS total_sessions,
            SUM(total_beans) AS total_beans_sum,
            SUM(good_beans) AS total_good_sum
        FROM sessions
        WHERE user_id = ?
        """,
        (user_id,),
    )

    row = cursor.fetchone()
    conn.close()

    total_sessions = row["total_sessions"] or 0
    total_beans = row["total_beans_sum"] or 0
    total_good = row["total_good_sum"] or 0

    avg_good = (total_good / total_beans * 100) if total_beans > 0 else 0
    return {
        "total_sessions": total_sessions,
        "total_beans": total_beans,
        "avg_good_percent": avg_good,
    }


if __name__ == "__main__":
    init_db()
    print("Database ready!")
