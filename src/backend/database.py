"""
MODULE 5 — Database (PostgreSQL)
===================================
Purpose : All PostgreSQL interaction — connection management,
          schema creation, and every query the app needs.

Schema overview:
  users        — one row per user, keyed by username
  sessions     — one row per login/logout cycle
  interactions — every item shown, liked, or skipped (explicit or implicit)

Action values:
  'shown'   — item was displayed (neutral, not used in embedding)
  'liked'   — user explicitly clicked Like       (+positive weight)
  'skipped' — implicit OR explicit negative signal (-negative weight)

Implicit skip logic (handled in frontend):
  When the user clicks Like or Refresh, the frontend checks how long
  each currently visible product was shown. If a product was viewed
  for more than 15 seconds without being liked, it is automatically
  logged as 'skipped'. This is a passive negative signal — the user
  looked long enough to form an opinion but chose not to engage.

view_duration_seconds:
  Stored for all interactions where duration is known.
  NULL for 'shown' events (duration not yet known at log time).
  Used for analytics — e.g. average view time per cluster,
  threshold tuning (is 15s the right cutoff?).
"""

import os
import psycopg2
import psycopg2.extras

from contextlib import contextmanager


# ── Connection ─────────────────────────────────────────────────────────────────

def get_connection():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise EnvironmentError("DATABASE_URL environment variable is not set.")
    return psycopg2.connect(url)


@contextmanager
def db_cursor():
    """Auto-commit context manager. Yields a RealDictCursor (rows as dicts)."""
    conn = get_connection()
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                yield cur
    finally:
        conn.close()


# ── Schema ─────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id    UUID      PRIMARY KEY DEFAULT gen_random_uuid(),
    username   TEXT      UNIQUE NOT NULL,     -- login key, chosen by user
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id UUID      PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    UUID      REFERENCES users(user_id) ON DELETE CASCADE,
    started_at TIMESTAMP DEFAULT NOW(),
    ended_at   TIMESTAMP DEFAULT NULL          -- NULL = session still active
);

CREATE TABLE IF NOT EXISTS interactions (
    id                     SERIAL    PRIMARY KEY,
    session_id             UUID      REFERENCES sessions(session_id) ON DELETE CASCADE,
    user_id                UUID      REFERENCES users(user_id)       ON DELETE CASCADE,
    asin                   TEXT      NOT NULL,
    action                 TEXT      NOT NULL CHECK (action IN ('shown', 'liked', 'skipped')),
    view_duration_seconds  FLOAT     DEFAULT NULL,  -- NULL for 'shown'; set for liked/skipped
    created_at             TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_interactions_user
    ON interactions (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_sessions_user
    ON sessions (user_id, ended_at);
"""


def init_db() -> None:
    """Create all tables and indexes. Safe to call on every startup."""
    with db_cursor() as cur:
        cur.execute(SCHEMA)
    print("[db] Schema initialised (users, sessions, interactions)")


# ── Auth ───────────────────────────────────────────────────────────────────────

def get_user_by_username(username: str) -> dict | None:
    """
    Look up a user by username.
    Returns the user row dict if found, None if this is a new user.
    The caller decides whether to login or register based on this result.
    """
    with db_cursor() as cur:
        cur.execute(
            "SELECT user_id, username, created_at FROM users WHERE username = %s",
            (username.strip().lower(),)   # normalise: strip spaces, lowercase
        )
        row = cur.fetchone()
    return dict(row) if row else None


def create_user(username: str) -> dict:
    """
    Register a brand new user. Returns the created user row.
    Only called when get_user_by_username() returns None.
    """
    with db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO users (username)
            VALUES (%s)
            RETURNING user_id, username, created_at
            """,
            (username.strip().lower(),)
        )
        row = cur.fetchone()
    return dict(row)


# ── Sessions ───────────────────────────────────────────────────────────────────

def create_session(user_id: str) -> str:
    """
    Open a new session on login. Returns session_id as a string.
    A user accumulates sessions over their lifetime — each login
    creates one. History from all past sessions informs the embedding.
    """
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO sessions (user_id) VALUES (%s) RETURNING session_id",
            (user_id,)
        )
        row = cur.fetchone()
    return str(row["session_id"])


def close_session(session_id: str) -> None:
    """
    Mark session as ended on logout (sets ended_at = NOW()).
    Interaction history is preserved — it will influence future sessions.
    """
    with db_cursor() as cur:
        cur.execute(
            """
            UPDATE sessions SET ended_at = NOW()
            WHERE  session_id = %s AND ended_at IS NULL
            """,
            (session_id,)
        )


# ── Interactions ───────────────────────────────────────────────────────────────

def log_interaction(
    session_id            : str,
    user_id               : str,
    asin                  : str,
    action                : str,       # 'shown' | 'liked' | 'skipped'
    view_duration_seconds : float | None = None,
) -> None:
    """
    Append one interaction event to the log.

    view_duration_seconds:
      Pass None for 'shown' events (duration unknown at display time).
      Pass the elapsed seconds for 'liked' and 'skipped' events —
      computed by the frontend from (now - shown_at) for each product.
    """
    with db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO interactions
                (session_id, user_id, asin, action, view_duration_seconds)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (session_id, user_id, asin, action, view_duration_seconds)
        )


def get_user_interactions(user_id: str) -> list[dict]:
    """
    Return all LIKED and SKIPPED interactions for a user across all sessions,
    ordered newest first.

    'shown' is deliberately excluded — neutral impressions do not contribute
    to the user embedding. Only explicit signals (liked / skipped) do.
    This history spans ALL past sessions, so returning users immediately
    benefit from everything they interacted with before.
    """
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT   asin, action, created_at
            FROM     interactions
            WHERE    user_id = %s
              AND    action  IN ('liked', 'skipped')
            ORDER BY created_at DESC
            """,
            (user_id,)
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


# ── Health ─────────────────────────────────────────────────────────────────────

def health_check() -> bool:
    try:
        with db_cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:
        return False