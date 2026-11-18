"""Queries behind the review UI: picking images, voting, adding tags, browsing."""

from __future__ import annotations

import random
import sqlite3
from typing import Any

from .auth import User
from .consensus import VERIFY_SQL, Settings, recompute
from .db import now, transaction
from .tags import (
    category_sort_key,
    display_tag,
    get_or_create_tag,
    normalize_category,
    normalize_tag,
    resolve_alias,
)

MODES = ("verify", "missing", "random")
MAX_TAGS_PER_REQUEST = 40
MAX_VOTES_PER_REQUEST = 200


class ReviewError(ValueError):
    """A client error with a message safe to show to the user."""


def _missing_sql(s: Settings) -> tuple[str, dict[str, float]]:
    return (
        """(
            it.status = 'unreviewed' AND it.in_caption = 0
            AND it.score >= :miss_lo AND it.score < :miss_hi
        )""",
        {"miss_lo": s.missing_min_score, "miss_hi": s.present_threshold},
    )


# ---------------------------------------------------------------------------
# Picking the next image


def pick_image(
    conn: sqlite3.Connection,
    user: User,
    mode: str,
    exclude: frozenset[int] = frozenset(),
) -> int | None:
    """Choose the next image for ``user``; ``exclude`` holds ids the client
    already has queued (it prefetches one ahead)."""
    if mode not in MODES:
        raise ReviewError(f"Unknown mode {mode!r}")
    s = Settings.load(conn)
    submitted_mode = "missing" if mode == "missing" else "verify"
    not_done = (
        "NOT EXISTS (SELECT 1 FROM submissions sub WHERE sub.image_id = i.id "
        "AND sub.user_id = :uid AND sub.mode = :smode)"
    )
    params: dict[str, Any] = {"uid": user.id, "smode": submitted_mode}

    if mode == "random":
        ids = [
            r[0]
            for r in conn.execute(
                f"SELECT i.id FROM images i WHERE i.missing = 0 AND {not_done}",
                params,
            ).fetchall()
            if r[0] not in exclude
        ]
        return random.choice(ids) if ids else None

    if mode == "verify":
        rows = conn.execute(
            f"SELECT i.id, i.need FROM images i "
            f"WHERE i.missing = 0 AND i.need > 0 AND {not_done}",
            params,
        ).fetchall()
    else:
        miss, miss_params = _missing_sql(s)
        rows = conn.execute(
            f"""
            SELECT i.id, SUM(it.score) AS need FROM images i
            JOIN image_tags it ON it.image_id = i.id
            WHERE i.missing = 0 AND {miss} AND {not_done}
            GROUP BY i.id
            """,
            {**params, **miss_params},
        ).fetchall()
    rows = [r for r in rows if r[0] not in exclude]
    if not rows:
        return None
    # Weighted random: images that need more review come up more often, but
    # everything with any need can appear.
    return random.choices(
        [r[0] for r in rows], weights=[max(r[1], 1e-6) for r in rows]
    )[0]


# ---------------------------------------------------------------------------
# Image payloads


def _image_row(conn: sqlite3.Connection, image_id: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM images WHERE id = ? AND missing = 0", (image_id,)
    ).fetchone()
    if row is None:
        raise ReviewError("Image not found")
    return row


def _tag_dict(r: sqlite3.Row) -> dict[str, Any]:
    return {
        "tag_id": r["tag_id"],
        "name": r["name"],
        "label": display_tag(r["name"]),
        "category": r["category"],
        "score": None if r["score"] is None else round(r["score"], 3),
        "in_caption": bool(r["in_caption"]),
        "status": r["status"],
        "yes": r["yes"],
        "no": r["no"],
        "posterior": round(r["posterior"], 3),
        "resolution": r["resolution"],
        "my_vote": r["my_vote"],
        "user_added": r["added_by"] is not None,
    }


_TAG_SELECT = """
    SELECT it.*, t.name, t.category,
        (SELECT v.vote FROM votes v WHERE v.image_id = it.image_id
            AND v.tag_id = it.tag_id AND v.user_id = :uid) AS my_vote
    FROM image_tags it JOIN tags t ON t.id = it.tag_id
"""


def image_payload(
    conn: sqlite3.Connection,
    user: User,
    image_id: int,
    mode: str,
    n: int = 12,
) -> dict[str, Any]:
    """Image plus the tags to show for ``mode``.

    ``verify``/``random``: tags currently believed present, most uncertain
    first, skipping ones this user already answered.  Default answer: correct.

    ``missing``: likely-missing tags (moderate auto score, not in caption).
    Default answer: absent.

    ``detail``: every tag with votes, for the browse view.
    """
    s = Settings.load(conn)
    img = _image_row(conn, image_id)
    n = max(1, min(int(n), MAX_TAGS_PER_REQUEST))
    params: dict[str, Any] = {"uid": user.id, "iid": image_id, "n": n}

    if mode in ("verify", "random"):
        rows = conn.execute(
            f"""{_TAG_SELECT}
            WHERE it.image_id = :iid AND {VERIFY_SQL} AND my_vote IS NULL
            ORDER BY it.need DESC, it.posterior ASC LIMIT :n""",
            params,
        ).fetchall()
        default = 1
    elif mode == "missing":
        miss, miss_params = _missing_sql(s)
        rows = conn.execute(
            f"""{_TAG_SELECT}
            WHERE it.image_id = :iid AND {miss} AND my_vote IS NULL
            ORDER BY it.score DESC LIMIT :n""",
            {**params, **miss_params},
        ).fetchall()
        default = -1
    elif mode == "detail":
        rows = conn.execute(
            f"""{_TAG_SELECT}
            WHERE it.image_id = :iid AND (
                {VERIFY_SQL} OR it.n_votes > 0 OR it.resolution IS NOT NULL
                OR it.score >= :lo
            )""",
            {**params, "lo": s.missing_min_score},
        ).fetchall()
        default = 0
    else:
        raise ReviewError(f"Unknown mode {mode!r}")

    tags = [_tag_dict(r) for r in rows]
    tags.sort(key=lambda t: (category_sort_key(t["category"]), t["name"]))
    payload: dict[str, Any] = {
        "image": {
            "id": img["id"],
            "width": img["width"],
            "height": img["height"],
            "url": f"/img/{img['id']}.webp?v={int(img['updated_at'])}",
            "thumb": f"/thumb/{img['id']}.webp?v={int(img['updated_at'])}",
        },
        "mode": mode,
        "default_vote": default,
        "tags": tags,
    }
    if user.is_admin:
        payload["image"]["path"] = img["rel_path"]
    return payload


# ---------------------------------------------------------------------------
# Voting


def _parse_votes(raw: object) -> dict[int, int]:
    if not isinstance(raw, list) or len(raw) > MAX_VOTES_PER_REQUEST:
        raise ReviewError("votes must be a list")
    out: dict[int, int] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ReviewError("bad vote")
        tag_id, vote = item.get("tag_id"), item.get("vote")
        if (
            not isinstance(tag_id, int)
            or isinstance(tag_id, bool)
            or not isinstance(vote, int)
            or isinstance(vote, bool)
            or vote not in (-1, 0, 1)
        ):
            raise ReviewError("bad vote")
        out[tag_id] = vote
    return out


def submit_votes(
    conn: sqlite3.Connection,
    user: User,
    image_id: int,
    mode: str,
    raw_votes: object,
) -> None:
    """Record votes (1 = correct, -1 = wrong, 0 = clear my vote).

    Submitting from ``verify``/``random``/``missing`` also marks the image as
    done for this user in that mode, even if no votes were cast (a skip).
    """
    if mode not in (*MODES, "detail"):
        raise ReviewError(f"Unknown mode {mode!r}")
    votes = _parse_votes(raw_votes)
    _image_row(conn, image_id)
    t = now()
    with transaction(conn):
        if votes:
            ids = list(votes)
            valid = {
                r[0]
                for r in conn.execute(
                    f"SELECT tag_id FROM image_tags WHERE image_id = ? "
                    f"AND tag_id IN ({','.join('?' * len(ids))})",
                    [image_id, *ids],
                ).fetchall()
            }
            if valid != set(ids):
                raise ReviewError("Unknown tag for this image")
        for tag_id, vote in votes.items():
            if vote == 0:
                conn.execute(
                    "DELETE FROM votes WHERE image_id = ? AND tag_id = ? "
                    "AND user_id = ?",
                    (image_id, tag_id, user.id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO votes(image_id, tag_id, user_id, vote, mode,
                                      created_at, updated_at)
                    VALUES(?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(image_id, tag_id, user_id) DO UPDATE SET
                        vote = excluded.vote, mode = excluded.mode,
                        updated_at = excluded.updated_at
                    """,
                    (image_id, tag_id, user.id, vote, mode, t, t),
                )
        if mode != "detail":
            conn.execute(
                "INSERT OR IGNORE INTO submissions(user_id, image_id, mode, "
                "created_at) VALUES(?, ?, ?, ?)",
                (user.id, image_id, "missing" if mode == "missing" else "verify", t),
            )
        recompute(conn, image_id, votes)
        # A reviewer-added tag that nobody votes for any more disappears.
        conn.execute(
            """DELETE FROM image_tags WHERE image_id = ? AND added_by IS NOT NULL
               AND in_caption = 0 AND score IS NULL AND n_votes = 0
               AND resolution IS NULL""",
            (image_id,),
        )


def add_tag(
    conn: sqlite3.Connection, user: User, image_id: int, raw_name: object
) -> dict[str, Any]:
    """Add a tag to an image (or upvote it if already a candidate)."""
    if not isinstance(raw_name, str):
        raise ReviewError("Tag name required")
    name = normalize_tag(raw_name)
    if name is None:
        raise ReviewError("Not a valid tag")
    _image_row(conn, image_id)
    s = Settings.load(conn)
    t = now()
    with transaction(conn):
        name = resolve_alias(conn, name)
        row = conn.execute(
            "SELECT id, in_vocab FROM tags WHERE name = ?", (name,)
        ).fetchone()
        if row is None and not s.allow_unknown_tags:
            raise ReviewError("That tag is not in the tag database")
        tag_id = row[0] if row is not None else get_or_create_tag(conn, name)
        conn.execute(
            "INSERT OR IGNORE INTO image_tags(image_id, tag_id, added_by) "
            "VALUES(?, ?, ?)",
            (image_id, tag_id, user.id),
        )
        conn.execute(
            """
            INSERT INTO votes(image_id, tag_id, user_id, vote, mode, created_at,
                              updated_at)
            VALUES(?, ?, ?, 1, 'add', ?, ?)
            ON CONFLICT(image_id, tag_id, user_id) DO UPDATE SET
                vote = 1, updated_at = excluded.updated_at
            """,
            (image_id, tag_id, user.id, t, t),
        )
        recompute(conn, image_id, [tag_id], s)
    row = conn.execute(
        f"{_TAG_SELECT} WHERE it.image_id = :iid AND it.tag_id = :tid",
        {"uid": user.id, "iid": image_id, "tid": tag_id},
    ).fetchone()
    return _tag_dict(row)


# ---------------------------------------------------------------------------
# Browsing

BROWSE_FILTERS = ("all", "needs_review", "contested", "unreviewed", "mine")


def browse(
    conn: sqlite3.Connection,
    user: User,
    *,
    filt: str = "all",
    tag: str | None = None,
    after: int = 0,
    limit: int = 60,
) -> dict[str, Any]:
    if filt not in BROWSE_FILTERS:
        raise ReviewError("Unknown filter")
    limit = max(1, min(int(limit), 200))
    where = ["i.missing = 0", "i.id > :after"]
    params: dict[str, Any] = {"after": after, "limit": limit, "uid": user.id}
    if filt == "needs_review":
        where.append("i.need > 0")
    elif filt == "contested":
        where.append(
            "EXISTS (SELECT 1 FROM image_tags c WHERE c.image_id = i.id "
            "AND c.status = 'contested')"
        )
    elif filt == "unreviewed":
        where.append("NOT EXISTS (SELECT 1 FROM votes v WHERE v.image_id = i.id)")
    elif filt == "mine":
        where.append(
            "EXISTS (SELECT 1 FROM votes v WHERE v.image_id = i.id "
            "AND v.user_id = :uid)"
        )
    if tag:
        name = normalize_tag(tag)
        if name is None:
            raise ReviewError("Not a valid tag")
        params["tag"] = resolve_alias(conn, name)
        where.append(
            f"""EXISTS (SELECT 1 FROM image_tags it JOIN tags t ON t.id = it.tag_id
                WHERE it.image_id = i.id AND t.name = :tag AND {VERIFY_SQL})"""
        )
    rows = conn.execute(
        f"""
        SELECT i.id, i.width, i.height, i.need, i.updated_at,
            (SELECT COUNT(*) FROM image_tags c WHERE c.image_id = i.id
                AND c.status = 'contested') AS contested
        FROM images i WHERE {" AND ".join(where)}
        ORDER BY i.id LIMIT :limit
        """,
        params,
    ).fetchall()
    items = [
        {
            "id": r["id"],
            "width": r["width"],
            "height": r["height"],
            "need": round(r["need"], 2),
            "contested": r["contested"],
            "thumb": f"/thumb/{r['id']}.webp?v={int(r['updated_at'])}",
        }
        for r in rows
    ]
    return {
        "items": items,
        "next": items[-1]["id"] if len(items) == limit else None,
    }


# ---------------------------------------------------------------------------
# Admin


def contested(
    conn: sqlite3.Connection, *, after: tuple[int, int] = (0, 0), limit: int = 50
) -> dict[str, Any]:
    limit = max(1, min(int(limit), 200))
    rows = conn.execute(
        """
        SELECT it.image_id, it.tag_id, it.yes, it.no, it.n_votes, it.score,
               it.in_caption, t.name, t.category, i.updated_at
        FROM image_tags it
        JOIN tags t ON t.id = it.tag_id
        JOIN images i ON i.id = it.image_id
        WHERE it.status = 'contested' AND i.missing = 0
          AND (it.image_id, it.tag_id) > (?, ?)
        ORDER BY it.image_id, it.tag_id LIMIT ?
        """,
        (*after, limit),
    ).fetchall()
    items = []
    for r in rows:
        voters = conn.execute(
            "SELECT u.name, v.vote FROM votes v JOIN users u ON u.id = v.user_id "
            "WHERE v.image_id = ? AND v.tag_id = ? AND u.revoked = 0 "
            "ORDER BY v.updated_at",
            (r["image_id"], r["tag_id"]),
        ).fetchall()
        items.append(
            {
                "image_id": r["image_id"],
                "tag_id": r["tag_id"],
                "name": r["name"],
                "label": display_tag(r["name"]),
                "category": r["category"],
                "score": r["score"],
                "in_caption": bool(r["in_caption"]),
                "yes": r["yes"],
                "no": r["no"],
                "voters": [{"name": v["name"], "vote": v["vote"]} for v in voters],
                "thumb": f"/thumb/{r['image_id']}.webp?v={int(r['updated_at'])}",
            }
        )
    nxt = [items[-1]["image_id"], items[-1]["tag_id"]] if len(items) == limit else None
    return {"items": items, "next": nxt}


def resolve(
    conn: sqlite3.Connection,
    user: User,
    image_id: int,
    tag_id: int,
    resolution: object,
) -> None:
    if resolution not in (0, 1, None) or isinstance(resolution, bool):
        raise ReviewError("resolution must be 1, 0 or null")
    with transaction(conn):
        cur = conn.execute(
            "UPDATE image_tags SET resolution = ?, resolved_by = ? "
            "WHERE image_id = ? AND tag_id = ?",
            (resolution, None if resolution is None else user.id, image_id, tag_id),
        )
        if cur.rowcount == 0:
            raise ReviewError("Unknown image/tag")
        recompute(conn, image_id, [tag_id])


def set_category(conn: sqlite3.Connection, raw_tag: object, raw_cat: object) -> None:
    if not isinstance(raw_tag, str) or not isinstance(raw_cat, str):
        raise ReviewError("tag and category required")
    tag, cat = normalize_tag(raw_tag), normalize_category(raw_cat)
    if tag is None or cat is None:
        raise ReviewError("Invalid tag or category")
    with transaction(conn):
        cur = conn.execute(
            "UPDATE tags SET category = ?, category_locked = 1 WHERE name = ?",
            (cat, tag),
        )
        if cur.rowcount == 0:
            raise ReviewError("Unknown tag")


def uncategorized(conn: sqlite3.Connection, limit: int = 100) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT t.name, COUNT(*) AS uses FROM tags t
        JOIN image_tags it ON it.tag_id = t.id
        WHERE t.category = 'unknown'
        GROUP BY t.id ORDER BY uses DESC, t.name LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [{"name": r[0], "label": display_tag(r[0]), "uses": r[1]} for r in rows]


def categories(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT DISTINCT category FROM tags").fetchall()
    return sorted((r[0] for r in rows), key=category_sort_key)


def stats(conn: sqlite3.Connection, user: User) -> dict[str, Any]:
    by_status = dict(
        conn.execute(
            "SELECT it.status, COUNT(*) FROM image_tags it JOIN images i "
            "ON i.id = it.image_id WHERE i.missing = 0 GROUP BY it.status"
        ).fetchall()
    )
    images = conn.execute(
        "SELECT COUNT(*), SUM(need > 0) FROM images WHERE missing = 0"
    ).fetchone()
    board = conn.execute(
        """SELECT u.name, COUNT(v.vote) AS votes FROM users u
           LEFT JOIN votes v ON v.user_id = u.id
           WHERE u.revoked = 0 GROUP BY u.id ORDER BY votes DESC LIMIT 20"""
    ).fetchall()
    mine = conn.execute(
        "SELECT COUNT(*) FROM votes WHERE user_id = ?", (user.id,)
    ).fetchone()[0]
    return {
        "images": images[0],
        "images_needing_review": images[1] or 0,
        "tags_by_status": by_status,
        "my_votes": mine,
        "leaderboard": [{"name": r[0], "votes": r[1]} for r in board],
    }


def purge_user_votes(conn: sqlite3.Connection, name: str) -> int:
    """Delete every vote a user cast (e.g. after a bad actor), then recompute."""
    row = conn.execute("SELECT id FROM users WHERE name = ?", (name,)).fetchone()
    if row is None:
        raise KeyError(f"No user named {name!r}")
    with transaction(conn):
        images = [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT image_id FROM votes WHERE user_id = ?", (row[0],)
            ).fetchall()
        ]
        n = conn.execute("DELETE FROM votes WHERE user_id = ?", (row[0],)).rowcount
        conn.execute("DELETE FROM submissions WHERE user_id = ?", (row[0],))
        s = Settings.load(conn)
        for image_id in images:
            recompute(conn, image_id, s=s)
        conn.execute(
            """DELETE FROM image_tags WHERE added_by = ? AND in_caption = 0
               AND score IS NULL AND n_votes = 0 AND resolution IS NULL""",
            (row[0],),
        )
    return n
