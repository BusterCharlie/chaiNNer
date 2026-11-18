"""Turn reviewer votes into per-tag consensus and review priority.

Each (image, tag) pair has

- a *baseline*: whether the tag is in the dataset's caption today,
- a *prior* probability: the auto-tagger score if known, else a default,
- weighted ``yes`` / ``no`` vote totals (each user has a trust weight).

From these we derive a status:

``unreviewed``   nobody has voted
``leaning_yes``  / ``leaning_no``: votes agree but there are not enough yet
``confirmed``    / ``rejected``: enough weighted votes, clear majority
``contested``    both sides have a substantial share; queued for an admin
``resolved``     an admin made the final call

and a *need* score used to pick what to show reviewers next: uncertain auto
scores and contested tags rank highest; settled tags drop to zero.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields

from .db import get_meta, set_meta

PRESENT_STATUSES = ("leaning_yes", "confirmed")
ABSENT_STATUSES = ("leaning_no", "rejected")
SETTLED_STATUSES = ("confirmed", "rejected", "resolved")


@dataclass
class Settings:
    # Prior pseudo-count: how many votes the auto score is "worth".
    prior_strength: float = 2.0
    # Prior for caption tags that have no auto score (hand-written captions).
    caption_prior: float = 0.8
    # Prior for tags a reviewer added (their own +1 vote is counted separately).
    user_tag_prior: float = 0.5
    # Weighted votes needed before a tag is confirmed or rejected.
    confirm_votes: float = 2.0
    # If the minority side holds at least this share of the vote, it's contested.
    contest_ratio: float = 0.25
    # Stop asking about a contested tag after this many votes; leave it to an admin.
    max_votes: int = 7
    # When an image has scores but no caption file, scores at or above this
    # become its baseline caption.  Also the upper bound of "missing" candidates.
    present_threshold: float = 0.35
    # Lowest auto score worth asking about in "missing tags" mode.
    missing_min_score: float = 0.12
    # Scores below this are not stored at all (keeps the database small).
    store_min_score: float = 0.05
    # Number of most-needed tags summed into an image's need.
    need_top_tags: int = 12
    # Whether reviewers may add tags that are not in the tag database.
    allow_unknown_tags: bool = True

    @staticmethod
    def load(conn: sqlite3.Connection) -> Settings:
        data = get_meta(conn, "settings", {}) or {}
        known = {f.name for f in fields(Settings)}
        return Settings(**{k: v for k, v in data.items() if k in known})

    def save(self, conn: sqlite3.Connection) -> None:
        set_meta(conn, "settings", asdict(self))


def prior(
    score: float | None, in_caption: bool, user_added: bool, s: Settings
) -> float:
    if score is not None:
        p = score
    elif in_caption:
        p = s.caption_prior
    elif user_added:
        p = s.user_tag_prior
    else:
        p = 0.5
    return min(0.98, max(0.02, p))


def posterior(p0: float, yes: float, no: float, s: Settings) -> float:
    k = s.prior_strength
    return (p0 * k + yes) / (k + yes + no)


def classify(yes: float, no: float, resolution: int | None, s: Settings) -> str:
    if resolution is not None:
        return "resolved"
    total = yes + no
    if total <= 0:
        return "unreviewed"
    if yes > 0 and no > 0 and min(yes, no) / total >= s.contest_ratio:
        return "contested"
    if total >= s.confirm_votes:
        return "confirmed" if yes > no else "rejected"
    return "leaning_yes" if yes > no else "leaning_no"


def tag_need(status: str, post: float, n_votes: int, s: Settings) -> float:
    if status in SETTLED_STATUSES:
        return 0.0
    if status == "contested":
        return 0.0 if n_votes >= s.max_votes else 1.5
    uncertainty = 1.0 - abs(2.0 * post - 1.0)
    if status == "unreviewed":
        return 0.25 + uncertainty
    return 0.15 + 0.6 * uncertainty


def is_present(
    status: str, resolution: int | None, baseline: bool, *, accept_leaning: bool
) -> bool:
    """Current best answer to "does this tag belong on this image"."""
    if resolution is not None:
        return bool(resolution)
    if status == "confirmed":
        return True
    if status == "rejected":
        return False
    if accept_leaning and status == "leaning_yes":
        return True
    if accept_leaning and status == "leaning_no":
        return False
    return baseline


# SQL fragment: baseline membership of an image_tags row.  Tags added by a
# reviewer are not baseline: like any other change, they need confirmation.
BASELINE_SQL = "(it.in_caption = 1)"
# SQL fragment: tags that belong in "verify" mode (believed present, or disputed).
VERIFY_SQL = f"""(
    it.resolution = 1
    OR (it.resolution IS NULL AND (
        it.status IN ('leaning_yes', 'confirmed', 'contested')
        OR (it.status = 'unreviewed' AND {BASELINE_SQL})
    ))
)"""


def recompute(
    conn: sqlite3.Connection,
    image_id: int,
    tag_ids: Iterable[int] | None = None,
    s: Settings | None = None,
) -> None:
    """Refresh cached consensus for one image (optionally only some tags)."""
    s = s or Settings.load(conn)
    params: list[object] = [image_id]
    where = "it.image_id = ?"
    if tag_ids is not None:
        ids = list(tag_ids)
        if not ids:
            return
        where += f" AND it.tag_id IN ({','.join('?' * len(ids))})"
        params.extend(ids)

    rows = conn.execute(
        f"""
        SELECT it.tag_id, it.score, it.in_caption, it.added_by, it.resolution,
            COALESCE(SUM(CASE WHEN v.vote > 0 THEN u.weight END), 0) AS yes,
            COALESCE(SUM(CASE WHEN v.vote < 0 THEN u.weight END), 0) AS no,
            COUNT(u.id) AS n
        FROM image_tags it
        LEFT JOIN votes v ON v.image_id = it.image_id AND v.tag_id = it.tag_id
        LEFT JOIN users u ON u.id = v.user_id AND u.revoked = 0
        WHERE {where}
        GROUP BY it.tag_id
        """,
        params,
    ).fetchall()

    updates = []
    for r in rows:
        p0 = prior(r["score"], bool(r["in_caption"]), r["added_by"] is not None, s)
        post = posterior(p0, r["yes"], r["no"], s)
        status = classify(r["yes"], r["no"], r["resolution"], s)
        need = tag_need(status, post, r["n"], s)
        updates.append(
            (r["yes"], r["no"], r["n"], status, post, need, image_id, r["tag_id"])
        )
    conn.executemany(
        """
        UPDATE image_tags SET yes = ?, no = ?, n_votes = ?, status = ?,
            posterior = ?, need = ?
        WHERE image_id = ? AND tag_id = ?
        """,
        updates,
    )
    update_image_need(conn, image_id, s)


def update_image_need(conn: sqlite3.Connection, image_id: int, s: Settings) -> None:
    row = conn.execute(
        f"""
        SELECT COALESCE(SUM(need), 0) FROM (
            SELECT it.need FROM image_tags it
            WHERE it.image_id = ? AND {VERIFY_SQL}
            ORDER BY it.need DESC LIMIT ?
        )
        """,
        (image_id, s.need_top_tags),
    ).fetchone()
    conn.execute("UPDATE images SET need = ? WHERE id = ?", (row[0], image_id))


def recompute_all(conn: sqlite3.Connection) -> None:
    s = Settings.load(conn)
    ids = [r[0] for r in conn.execute("SELECT id FROM images").fetchall()]
    for image_id in ids:
        recompute(conn, image_id, s=s)
