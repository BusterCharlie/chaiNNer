"""SQLite storage for a review directory."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import REVIEW_DB_NAME

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS images (
    id           INTEGER PRIMARY KEY,
    rel_path     TEXT NOT NULL UNIQUE,
    src_size     INTEGER NOT NULL DEFAULT 0,
    src_mtime    REAL NOT NULL DEFAULT 0,
    width        INTEGER NOT NULL DEFAULT 0,
    height       INTEGER NOT NULL DEFAULT 0,
    orig_width   INTEGER NOT NULL DEFAULT 0,
    orig_height  INTEGER NOT NULL DEFAULT 0,
    need         REAL NOT NULL DEFAULT 1.0,
    -- 1 when the source file disappeared on a later sync (kept for its votes)
    missing      INTEGER NOT NULL DEFAULT 0,
    -- caption fragments that are prose rather than tags; preserved on export
    caption_extra TEXT NOT NULL DEFAULT '[]',
    updated_at   REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS images_need ON images(need);

-- Tag vocabulary: every tag known from the booru/e621 tag database plus every
-- tag seen in the dataset or added by a reviewer.
CREATE TABLE IF NOT EXISTS tags (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    category    TEXT NOT NULL DEFAULT 'unknown',
    post_count  INTEGER NOT NULL DEFAULT 0,
    in_vocab    INTEGER NOT NULL DEFAULT 0,
    -- 1 when the category was set by hand (CLI, categories file or admin UI)
    -- so a later tag-database import does not overwrite it.
    category_locked INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tag_aliases (
    alias   TEXT PRIMARY KEY,
    tag_id  INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE
);

-- One row per candidate (image, tag) pair, with cached consensus.
CREATE TABLE IF NOT EXISTS image_tags (
    image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    tag_id      INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    in_caption  INTEGER NOT NULL DEFAULT 0,
    caption_pos INTEGER,
    score       REAL,
    added_by    INTEGER REFERENCES users(id) ON DELETE SET NULL,
    yes         REAL NOT NULL DEFAULT 0,
    no          REAL NOT NULL DEFAULT 0,
    n_votes     INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'unreviewed',
    posterior   REAL NOT NULL DEFAULT 0.5,
    need        REAL NOT NULL DEFAULT 0,
    resolution  INTEGER,
    resolved_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    PRIMARY KEY (image_id, tag_id)
);
CREATE INDEX IF NOT EXISTS image_tags_tag ON image_tags(tag_id);
CREATE INDEX IF NOT EXISTS image_tags_status ON image_tags(status);

CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    role        TEXT NOT NULL DEFAULT 'reviewer',
    weight      REAL NOT NULL DEFAULT 1.0,
    key_hash    TEXT UNIQUE,
    created_at  REAL NOT NULL,
    revoked     INTEGER NOT NULL DEFAULT 0,
    last_seen   REAL
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash  TEXT PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  REAL NOT NULL,
    expires_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id);

CREATE TABLE IF NOT EXISTS votes (
    image_id    INTEGER NOT NULL,
    tag_id      INTEGER NOT NULL,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    vote        INTEGER NOT NULL CHECK (vote IN (-1, 1)),
    mode        TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    PRIMARY KEY (image_id, tag_id, user_id),
    FOREIGN KEY (image_id, tag_id) REFERENCES image_tags(image_id, tag_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS votes_user ON votes(user_id);

-- Which images a user has already handled in which mode, so the queue does
-- not show them the same picture twice.
CREATE TABLE IF NOT EXISTS submissions (
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    mode        TEXT NOT NULL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (user_id, image_id, mode)
);
"""


def db_path(review_dir: Path) -> Path:
    return review_dir / REVIEW_DB_NAME


def connect(review_dir: Path, *, create: bool = False) -> sqlite3.Connection:
    path = db_path(review_dir)
    if not create and not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Publish a dataset into {review_dir} first."
        )
    review_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=15, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=15000")
    if create:
        conn.executescript(_SCHEMA)
        if get_meta(conn, "schema_version") is None:
            set_meta(conn, "schema_version", SCHEMA_VERSION)
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """``BEGIN IMMEDIATE`` transaction: takes the write lock up front so two
    concurrent voters never deadlock on a lock upgrade."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def get_meta(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    return json.loads(row[0])


def set_meta(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, json.dumps(value)),
    )


def now() -> float:
    return time.time()
