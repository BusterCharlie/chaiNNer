"""Users, login keys, sessions, and login rate limiting.

Security model
--------------

- Every reviewer gets their own random 256-bit *login key*, delivered as an
  invite link ``https://host/#key=...``.  The key sits in the URL fragment, so
  it never reaches server logs, proxies, or ``Referer`` headers; the page posts
  it once to ``/api/login`` and then removes it from the address bar.
- Only SHA-256 hashes of keys and session tokens are stored, so a leaked
  database cannot be used to log in.
- A login creates a session cookie (``HttpOnly``, ``SameSite=Strict``, and
  ``Secure`` over HTTPS).  A key can be used on several devices; rotating or
  revoking it ends all of that user's sessions.
- Failed logins are rate limited per client address and globally.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass

from .db import now, transaction
from .tags import CONTROL_CHARS

SESSION_TTL = 30 * 24 * 3600
ROLES = ("reviewer", "admin")
_NAME_MAX = 40


@dataclass(frozen=True)
class User:
    id: int
    name: str
    role: str
    weight: float

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def new_secret() -> str:
    return secrets.token_urlsafe(32)


def _validate_name(name: str) -> str:
    name = name.strip()
    if not name or len(name) > _NAME_MAX or CONTROL_CHARS.search(name):
        raise ValueError(f"User name must be 1-{_NAME_MAX} printable characters")
    return name


def create_user(
    conn: sqlite3.Connection, name: str, role: str = "reviewer", weight: float = 1.0
) -> str:
    """Create a user and return their login key (shown once, never stored)."""
    if role not in ROLES:
        raise ValueError(f"Role must be one of {ROLES}")
    if not 0 <= weight <= 10:
        raise ValueError("Weight must be between 0 and 10")
    key = new_secret()
    with transaction(conn):
        conn.execute(
            "INSERT INTO users(name, role, weight, key_hash, created_at) "
            "VALUES(?, ?, ?, ?, ?)",
            (_validate_name(name), role, weight, hash_secret(key), now()),
        )
    return key


def rotate_key(conn: sqlite3.Connection, name: str) -> str:
    """Issue a new login key, invalidating the old one and all sessions."""
    key = new_secret()
    with transaction(conn):
        row = _get_user_row(conn, name)
        conn.execute(
            "UPDATE users SET key_hash = ?, revoked = 0 WHERE id = ?",
            (hash_secret(key), row["id"]),
        )
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (row["id"],))
    return key


def revoke_user(conn: sqlite3.Connection, name: str) -> None:
    """Block a user.  Their votes stop counting until they are re-enabled
    with :func:`rotate_key` (or removed with ``purge_votes``)."""
    with transaction(conn):
        row = _get_user_row(conn, name)
        conn.execute(
            "UPDATE users SET revoked = 1, key_hash = NULL WHERE id = ?", (row["id"],)
        )
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (row["id"],))


def set_user(
    conn: sqlite3.Connection,
    name: str,
    *,
    role: str | None = None,
    weight: float | None = None,
) -> None:
    with transaction(conn):
        row = _get_user_row(conn, name)
        if role is not None:
            if role not in ROLES:
                raise ValueError(f"Role must be one of {ROLES}")
            conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, row["id"]))
        if weight is not None:
            if not 0 <= weight <= 10:
                raise ValueError("Weight must be between 0 and 10")
            conn.execute(
                "UPDATE users SET weight = ? WHERE id = ?", (weight, row["id"])
            )


def _get_user_row(conn: sqlite3.Connection, name: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM users WHERE name = ?", (name,)).fetchone()
    if row is None:
        raise KeyError(f"No user named {name!r}")
    return row


def login(conn: sqlite3.Connection, key: str) -> tuple[User, str] | None:
    """Exchange a login key for a new session token."""
    if not key or len(key) > 200:
        return None
    row = conn.execute(
        "SELECT * FROM users WHERE key_hash = ? AND revoked = 0", (hash_secret(key),)
    ).fetchone()
    if row is None:
        return None
    token = new_secret()
    t = now()
    with transaction(conn):
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (t,))
        conn.execute(
            "INSERT INTO sessions(token_hash, user_id, created_at, expires_at) "
            "VALUES(?, ?, ?, ?)",
            (hash_secret(token), row["id"], t, t + SESSION_TTL),
        )
        conn.execute("UPDATE users SET last_seen = ? WHERE id = ?", (t, row["id"]))
    return _user(row), token


def session_user(conn: sqlite3.Connection, token: str | None) -> User | None:
    if not token or len(token) > 200:
        return None
    t = now()
    row = conn.execute(
        """
        SELECT u.*, s.expires_at, s.token_hash FROM sessions s
        JOIN users u ON u.id = s.user_id
        WHERE s.token_hash = ? AND s.expires_at > ? AND u.revoked = 0
        """,
        (hash_secret(token), t),
    ).fetchone()
    if row is None:
        return None
    # Sliding expiry, refreshed at most once an hour to avoid write churn.
    if row["expires_at"] - t < SESSION_TTL - 3600:
        with transaction(conn):
            conn.execute(
                "UPDATE sessions SET expires_at = ? WHERE token_hash = ?",
                (t + SESSION_TTL, row["token_hash"]),
            )
            conn.execute("UPDATE users SET last_seen = ? WHERE id = ?", (t, row["id"]))
    return _user(row)


def logout(conn: sqlite3.Connection, token: str) -> None:
    with transaction(conn):
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (hash_secret(token),))


def _user(row: sqlite3.Row) -> User:
    return User(id=row["id"], name=row["name"], role=row["role"], weight=row["weight"])


class LoginThrottle:
    """Sliding-window limiter on *failed* logins.

    Keys are 256-bit random values, so brute force is hopeless anyway; this is
    here to keep the logs quiet and make scanning pointless.
    """

    def __init__(
        self,
        per_client: int = 10,
        global_limit: int = 100,
        window: float = 15 * 60,
    ) -> None:
        self.per_client = per_client
        self.global_limit = global_limit
        self.window = window
        self._lock = threading.Lock()
        self._by_client: dict[str, deque[float]] = {}
        self._global: deque[float] = deque()

    def _trim(self, q: deque[float], t: float) -> None:
        while q and q[0] < t - self.window:
            q.popleft()

    def allowed(self, client: str) -> bool:
        t = time.monotonic()
        with self._lock:
            self._trim(self._global, t)
            q = self._by_client.get(client)
            if q is not None:
                self._trim(q, t)
                if not q:
                    del self._by_client[client]
                    q = None
            return len(self._global) < self.global_limit and (
                q is None or len(q) < self.per_client
            )

    def failed(self, client: str) -> None:
        t = time.monotonic()
        with self._lock:
            self._global.append(t)
            self._by_client.setdefault(client, deque()).append(t)
