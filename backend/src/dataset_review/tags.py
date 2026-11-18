"""Tag normalisation, caption parsing, and tag-vocabulary import.

Tags are stored in canonical booru form: lower case, underscores instead of
spaces, parentheses unescaped (``hatsune_miku_(cosplay)``).  The caption style
of the source dataset (spaces vs underscores, escaped parentheses) is detected
on import and re-applied on export.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

MAX_TAG_LENGTH = 120
MAX_CATEGORY_LENGTH = 32

# Display / export order.  Unknown custom categories sort between "lore" and
# "rating"; "unknown" always sorts last.
CATEGORY_ORDER = [
    "artist",
    "copyright",
    "character",
    "species",
    "general",
    "meta",
    "lore",
    "rating",
    "invalid",
    "unknown",
]

DANBOORU_CATEGORIES = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "meta",
}
E621_CATEGORIES = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "species",
    6: "invalid",
    7: "meta",
    8: "lore",
}
# The merged danbooru+e621 CSV used by the a1111 tag-autocomplete extension
# keeps danbooru ids as-is and shifts e621 ids up by 7.
MERGED_CATEGORIES = {
    **DANBOORU_CATEGORIES,
    **{k + 7: v for k, v in E621_CATEGORIES.items()},
}
CATEGORY_SCHEMES = {
    "danbooru": DANBOORU_CATEGORIES,
    "e621": E621_CATEGORIES,
    "merged": MERGED_CATEGORIES,
}

# Kaomoji whose underscores are part of the tag, not word separators.
KAOMOJI = frozenset(
    {
        "0_0",
        "(o)_(o)",
        "+_+",
        "+_-",
        "._.",
        "<o>_<o>",
        "<|>_<|>",
        "=_=",
        ">_<",
        "3_3",
        "6_9",
        ">_o",
        "@_@",
        "^_^",
        "o_o",
        "u_u",
        "x_x",
        "|_|",
        "||_||",
    }
)

CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_WS = re.compile(r"\s+")
_CATEGORY_OK = re.compile(r"^[a-z0-9_ -]+$")
# A caption fragment with this many words is treated as prose, not a tag.
_PROSE_WORDS = 7


def normalize_tag(raw: str) -> str | None:
    """Return the canonical form of ``raw`` or ``None`` if it is not a valid tag."""
    if CONTROL_CHARS.search(raw):
        return None
    tag = raw.strip().replace("\\(", "(").replace("\\)", ")")
    tag = _WS.sub(" ", tag).lower().strip(" ,")
    if not tag:
        return None
    tag = tag.replace(" ", "_")
    if len(tag) > MAX_TAG_LENGTH:
        return None
    return tag


def normalize_category(raw: str) -> str | None:
    cat = _WS.sub(" ", raw.strip().lower())
    if not cat or len(cat) > MAX_CATEGORY_LENGTH or not _CATEGORY_OK.match(cat):
        return None
    return cat


def display_tag(tag: str) -> str:
    return tag if tag in KAOMOJI else tag.replace("_", " ")


def category_sort_key(category: str) -> tuple[int, str]:
    if category in CATEGORY_ORDER:
        idx = CATEGORY_ORDER.index(category)
        # leave a gap after "lore" for custom categories
        return (idx if idx <= CATEGORY_ORDER.index("lore") else idx + 1, category)
    return (CATEGORY_ORDER.index("lore") + 1, category)


@dataclass
class CaptionStyle:
    """How tags were written in the source captions."""

    spaces: bool = True
    escape_parens: bool = True

    def format(self, tag: str) -> str:
        out = display_tag(tag) if self.spaces else tag
        if self.escape_parens:
            out = out.replace("(", "\\(").replace(")", "\\)")
        return out

    def to_json(self) -> dict[str, bool]:
        return {"spaces": self.spaces, "escape_parens": self.escape_parens}

    @staticmethod
    def from_json(data: dict[str, bool] | None) -> CaptionStyle:
        if not data:
            return CaptionStyle()
        return CaptionStyle(
            spaces=bool(data.get("spaces", True)),
            escape_parens=bool(data.get("escape_parens", True)),
        )


@dataclass
class ParsedCaption:
    tags: list[str]
    extra: list[str]
    used_spaces: int = 0
    used_underscores: int = 0
    escaped_parens: int = 0
    bare_parens: int = 0


def parse_caption(text: str) -> ParsedCaption:
    """Split a comma-separated booru caption into canonical tags.

    Fragments that look like prose (many words) are returned in ``extra`` so
    they survive an export round-trip.
    """
    result = ParsedCaption(tags=[], extra=[])
    seen: set[str] = set()
    for part in text.replace("\n", ",").split(","):
        stripped = part.strip()
        if not stripped:
            continue
        if len(stripped.split()) >= _PROSE_WORDS:
            result.extra.append(stripped)
            continue
        if " " in stripped:
            result.used_spaces += 1
        elif "_" in stripped and stripped.lower() not in KAOMOJI:
            result.used_underscores += 1
        if "\\(" in stripped or "\\)" in stripped:
            result.escaped_parens += 1
        elif "(" in stripped or ")" in stripped:
            result.bare_parens += 1
        tag = normalize_tag(stripped)
        if tag is None or tag in seen:
            continue
        seen.add(tag)
        result.tags.append(tag)
    return result


def detect_style(captions: Iterable[ParsedCaption]) -> CaptionStyle:
    spaces = underscores = escaped = bare = 0
    for c in captions:
        spaces += c.used_spaces
        underscores += c.used_underscores
        escaped += c.escaped_parens
        bare += c.bare_parens
    return CaptionStyle(
        spaces=spaces >= underscores,
        escape_parens=escaped >= bare,
    )


# ---------------------------------------------------------------------------
# Vocabulary


def get_or_create_tag(
    conn: sqlite3.Connection, name: str, category: str | None = None
) -> int:
    row = conn.execute("SELECT id FROM tags WHERE name = ?", (name,)).fetchone()
    if row is not None:
        return row[0]
    cur = conn.execute(
        "INSERT INTO tags(name, category) VALUES(?, ?)",
        (name, category or "unknown"),
    )
    assert cur.lastrowid is not None
    return cur.lastrowid


def resolve_alias(conn: sqlite3.Connection, name: str) -> str:
    """Map an alias (e.g. ``1girls``) to its canonical tag, if one is known."""
    row = conn.execute(
        "SELECT t.name FROM tag_aliases a JOIN tags t ON t.id = a.tag_id "
        "WHERE a.alias = ?",
        (name,),
    ).fetchone()
    return row[0] if row is not None else name


def _detect_scheme(path: Path, max_category: int) -> str:
    if max_category >= 7:
        # danbooru never uses ids above 5; e621 alone tops out at 8, and a
        # file with 9+ can only be the merged layout
        if max_category > 8 or "merged" in path.name.lower():
            return "merged"
        return "e621"
    if "e621" in path.name.lower():
        return "e621"
    return "danbooru"


def import_tag_database(
    conn: sqlite3.Connection, path: Path, scheme: str = "auto"
) -> int:
    """Import a booru tag CSV (``name,category,post_count[,"alias1,alias2"]``).

    This is the format used by the a1111 tag-autocomplete extension, which ships
    danbooru, e621, and merged danbooru+e621 lists.  The category column may be
    numeric (mapped with ``scheme``) or a category name.  Categories that were
    set by hand are kept.
    """
    rows: list[tuple[str, str, int, list[str]]] = []
    raw_rows: list[list[str]] = []
    max_numeric = -1
    with path.open(encoding="utf-8", newline="") as f:
        for rec in csv.reader(f):
            if len(rec) < 2:
                continue
            raw_rows.append(rec)
            if rec[1].strip().lstrip("-").isdigit():
                max_numeric = max(max_numeric, int(rec[1]))

    if scheme == "auto":
        scheme = _detect_scheme(path, max_numeric)
    mapping = CATEGORY_SCHEMES.get(scheme)
    if mapping is None:
        raise ValueError(f"Unknown tag category scheme {scheme!r}")

    for rec in raw_rows:
        name = normalize_tag(rec[0])
        if name is None or name == "name":  # header row
            continue
        cat_raw = rec[1].strip()
        if cat_raw.lstrip("-").isdigit():
            category = mapping.get(int(cat_raw), "unknown")
        else:
            category = normalize_category(cat_raw) or "unknown"
        try:
            count = int(float(rec[2])) if len(rec) > 2 and rec[2].strip() else 0
        except ValueError:
            count = 0
        aliases = []
        if len(rec) > 3 and rec[3].strip():
            aliases = [a for a in (normalize_tag(x) for x in rec[3].split(",")) if a]
        rows.append((name, category, count, aliases))

    conn.executemany(
        """
        INSERT INTO tags(name, category, post_count, in_vocab) VALUES(?, ?, ?, 1)
        ON CONFLICT(name) DO UPDATE SET
            category = CASE
                WHEN tags.category_locked THEN tags.category
                WHEN excluded.post_count >= tags.post_count OR tags.category = 'unknown'
                    THEN excluded.category
                ELSE tags.category END,
            post_count = max(tags.post_count, excluded.post_count),
            in_vocab = 1
        """,
        ((n, c, p) for n, c, p, _ in rows),
    )
    ids = dict(conn.execute("SELECT name, id FROM tags").fetchall())
    alias_rows = [
        (alias, ids[name])
        for name, _, _, aliases in rows
        for alias in aliases
        if alias not in ids
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO tag_aliases(alias, tag_id) VALUES(?, ?)", alias_rows
    )
    return len(rows)


def load_category_file(path: Path) -> dict[str, str]:
    """Read a hand-maintained tag→category mapping.

    Accepted formats:

    - JSON ``{"tag": "category"}`` or ``{"category": ["tag", ...]}``
    - CSV/TSV/text lines ``tag,category`` (or tab separated)
    """
    text = path.read_text(encoding="utf-8")
    pairs: list[tuple[str, str]] = []
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a JSON object")
        for key, value in data.items():
            if isinstance(value, str):
                pairs.append((key, value))
            elif isinstance(value, list):
                pairs.extend((str(t), key) for t in value)
    else:
        delimiter = "\t" if "\t" in text.split("\n", 1)[0] else ","
        for rec in csv.reader(text.splitlines(), delimiter=delimiter):
            if len(rec) >= 2 and not rec[0].startswith("#"):
                pairs.append((rec[0], rec[1]))

    out: dict[str, str] = {}
    for raw_tag, raw_cat in pairs:
        tag = normalize_tag(raw_tag)
        cat = normalize_category(raw_cat)
        if tag and cat and not (tag == "tag" and cat == "category"):
            out[tag] = cat
    return out


def apply_categories(conn: sqlite3.Connection, mapping: dict[str, str]) -> int:
    for tag, cat in mapping.items():
        conn.execute(
            """
            INSERT INTO tags(name, category, category_locked) VALUES(?, ?, 1)
            ON CONFLICT(name) DO UPDATE SET category = excluded.category,
                                            category_locked = 1
            """,
            (tag, cat),
        )
    return len(mapping)


def search_tags(
    conn: sqlite3.Connection, query: str, limit: int = 12
) -> list[sqlite3.Row]:
    """Autocomplete: prefix match on names and aliases, most-used first."""
    q = normalize_tag(query)
    if not q:
        return []
    hi = q + "￿"
    rows = conn.execute(
        """
        SELECT name, category, post_count, NULL AS alias FROM tags
            WHERE name >= :lo AND name < :hi
        UNION ALL
        SELECT t.name, t.category, t.post_count, a.alias FROM tag_aliases a
            JOIN tags t ON t.id = a.tag_id
            WHERE a.alias >= :lo AND a.alias < :hi
        ORDER BY post_count DESC
        LIMIT :limit
        """,
        {"lo": q, "hi": hi, "limit": limit * 3},
    ).fetchall()
    out: list[sqlite3.Row] = []
    seen: set[str] = set()
    for row in rows:
        if row["name"] not in seen:
            seen.add(row["name"])
            out.append(row)
    return out[:limit]
