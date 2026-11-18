"""Publish a training dataset into a review directory, or re-sync one.

For each image in the source dataset we write

- ``images/<id>.webp``: at most ``max_pixels`` (default 1 MP), for reviewing
- ``thumbs/<id>.webp``: small preview for the browse grid

and record its caption tags and auto-tagger scores in the database.  Re-running
on the same review directory is incremental: unchanged images are not
re-encoded, and votes are never discarded.
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import os
import sqlite3
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import IMAGES_DIR_NAME, THUMBS_DIR_NAME
from .consensus import Settings, recompute
from .db import connect, get_meta, now, set_meta, transaction
from .tags import (
    CaptionStyle,
    ParsedCaption,
    apply_categories,
    detect_style,
    get_or_create_tag,
    import_tag_database,
    load_category_file,
    normalize_category,
    normalize_tag,
    parse_caption,
    resolve_alias,
)

IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif"}
)
CAPTION_EXTENSIONS = (".txt", ".caption")
MAX_TAGS_PER_IMAGE = 256

# tag -> (score, category hint or None)
Scores = dict[str, tuple[float, str | None]]
ScoreLoader = Callable[[Path], Mapping[str, Any] | None]


def image_path(review_dir: Path, image_id: int) -> Path:
    return review_dir / IMAGES_DIR_NAME / f"{image_id}.webp"


def thumb_path(review_dir: Path, image_id: int) -> Path:
    return review_dir / THUMBS_DIR_NAME / f"{image_id}.webp"


# ---------------------------------------------------------------------------
# Auto-tagger score sidecars

_LIST_NAME_KEYS = ("tag", "name", "label")
_LIST_SCORE_KEYS = ("score", "confidence", "probability", "prob", "p")
_WRAPPER_KEYS = ("scores", "tags", "tag_scores", "predictions", "probabilities")
_IGNORED_SCORE_GROUPS = frozenset({"rating", "ratings"})


def _is_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def parse_scores(data: object, category: str | None = None) -> Scores:
    """Accepts the common shapes tagger scripts write:

    - ``{"tag": 0.93, ...}``
    - ``{"scores": {...}}`` (also ``tags``, ``predictions``, ...)
    - ``{"general": {...}, "character": {...}}``: groups become category hints
      (a ``rating`` group is ignored)
    - ``[{"tag": "x", "score": 0.9, "category": "general"}, ...]``
    - ``[["x", 0.9], ...]``
    """
    out: Scores = {}

    def put(raw: object, score: object, cat: str | None) -> None:
        if not isinstance(raw, str) or not _is_number(score):
            return
        tag = normalize_tag(raw)
        if tag is None:
            return
        value = float(score)  # type: ignore[arg-type]
        if not math.isfinite(value):
            return
        value = min(1.0, max(0.0, value))
        prev = out.get(tag)
        if prev is None or value > prev[0]:
            out[tag] = (value, cat)

    if isinstance(data, dict):
        for key in _WRAPPER_KEYS:
            if key in data and isinstance(data[key], (dict, list)):
                return parse_scores(data[key], category)
        for key, value in data.items():
            if _is_number(value):
                put(key, value, category)
            elif isinstance(value, dict) and isinstance(key, str):
                if key.lower() in _IGNORED_SCORE_GROUPS:
                    continue
                for tag, (score, cat) in parse_scores(
                    value, normalize_category(key)
                ).items():
                    put(tag, score, cat)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                name = next((item[k] for k in _LIST_NAME_KEYS if k in item), None)
                score = next((item[k] for k in _LIST_SCORE_KEYS if k in item), None)
                cat = item.get("category")
                put(
                    name,
                    score,
                    normalize_category(cat) if isinstance(cat, str) else category,
                )
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                put(item[0], item[1], category)
    return out


def default_score_loader(path: Path) -> Mapping[str, Any] | list[Any] | None:
    """Look for a score sidecar next to ``path``."""
    stem = path.with_suffix("")
    for candidate in (
        stem.with_name(stem.name + ".scores.json"),
        stem.with_name(stem.name + ".tags.json"),
        path.with_name(path.name + ".json"),
        stem.with_suffix(".json"),
    ):
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
    csv_path = stem.with_name(stem.name + ".scores.csv")
    if csv_path.is_file():
        rows: list[list[Any]] = []
        with csv_path.open(encoding="utf-8", newline="") as f:
            for rec in csv.reader(f):
                if len(rec) >= 2:
                    try:
                        rows.append([rec[0], float(rec[1])])
                    except ValueError:
                        continue
        return rows  # type: ignore[return-value]
    return None


def load_score_loader(spec: str) -> ScoreLoader:
    """Import a custom loader given as ``package.module:function``."""
    module_name, _, func_name = spec.partition(":")
    if not module_name or not func_name:
        raise ValueError("Score loader must look like 'module:function'")
    func = getattr(importlib.import_module(module_name), func_name)
    if not callable(func):
        raise TypeError(f"{spec} is not callable")
    return func


def find_caption(path: Path) -> Path | None:
    for ext in CAPTION_EXTENSIONS:
        candidate = path.with_suffix(ext)
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Image conversion


def convert_image(
    src: Path,
    dest: Path,
    thumb: Path,
    max_pixels: int,
    quality: int,
    thumb_size: int,
) -> tuple[int, int, int, int]:
    """Write a ≤ ``max_pixels`` WebP and a thumbnail.

    Returns ``(width, height, orig_width, orig_height)``.
    """
    from PIL import Image, ImageOps

    # Dataset images are local, trusted files and may be large scans.
    Image.MAX_IMAGE_PIXELS = 400_000_000

    with Image.open(src) as im:
        orig_w, orig_h = im.size
        if im.getexif().get(0x0112, 1) in (5, 6, 7, 8):  # rotated 90°
            orig_w, orig_h = orig_h, orig_w
        scale = min(1.0, math.sqrt(max_pixels / max(1, orig_w * orig_h)))
        target = (max(1, int(orig_w * scale)), max(1, int(orig_h * scale)))
        if im.format == "JPEG":
            im.draft("RGB", target)  # let libjpeg decode at reduced size
        im.seek(0)
        img = ImageOps.exif_transpose(im) or im
        has_alpha = img.mode in ("RGBA", "LA", "PA") or (
            img.mode == "P" and "transparency" in img.info
        )
        img = img.convert("RGBA" if has_alpha else "RGB")

    w, h = img.size
    if w * h > max_pixels:
        scale = math.sqrt(max_pixels / (w * h))
        img = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.LANCZOS,
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    thumb.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    img.save(tmp, "WEBP", quality=quality, method=4)
    os.replace(tmp, dest)

    small = img.copy()
    small.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
    tmp = thumb.with_name(thumb.name + ".tmp")
    small.save(tmp, "WEBP", quality=72, method=4)
    os.replace(tmp, thumb)

    return img.width, img.height, orig_w, orig_h


# ---------------------------------------------------------------------------
# Publishing


@dataclass
class PublishSummary:
    review_dir: Path
    images_total: int = 0
    images_converted: int = 0
    images_missing: int = 0
    images_failed: list[str] = field(default_factory=list)
    vocab_imported: int = 0

    def __str__(self) -> str:
        parts = [
            f"{self.images_total} images in review set",
            f"{self.images_converted} converted",
        ]
        if self.images_missing:
            parts.append(f"{self.images_missing} no longer in source")
        if self.images_failed:
            parts.append(f"{len(self.images_failed)} failed")
        if self.vocab_imported:
            parts.append(f"{self.vocab_imported} vocabulary tags imported")
        return ", ".join(parts)


@dataclass
class _Item:
    rel: str
    path: Path
    size: int
    mtime: float
    caption: ParsedCaption | None
    scores: Scores


def _scan(source: Path, exclude: Path) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for root, dirs, files in os.walk(source):
        root_path = Path(root)
        dirs[:] = sorted(
            d
            for d in dirs
            if not d.startswith(".") and (root_path / d).resolve() != exclude
        )
        for name in sorted(files):
            p = root_path / name
            if p.suffix.lower() in IMAGE_EXTENSIONS and not name.startswith("."):
                found.append((p.relative_to(source).as_posix(), p))
    return found


def publish_dataset(
    source_dir: Path,
    review_dir: Path,
    *,
    max_pixels: int = 1_000_000,
    quality: int = 82,
    thumb_size: int = 384,
    tag_database: Path | None = None,
    tag_database_scheme: str = "auto",
    categories_file: Path | None = None,
    score_loader: ScoreLoader | None = None,
    settings: Mapping[str, Any] | None = None,
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> PublishSummary:
    source_dir = source_dir.resolve()
    review_dir = review_dir.resolve()
    if review_dir == source_dir:
        raise ValueError("The review directory must differ from the dataset.")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory {source_dir} does not exist")

    conn = connect(review_dir, create=True)
    summary = PublishSummary(review_dir=review_dir)
    try:
        previous_source = get_meta(conn, "source_dir")
        if previous_source and Path(previous_source) != source_dir:
            raise ValueError(
                f"{review_dir} was published from {previous_source}; "
                "use a new review directory for a different dataset."
            )

        with transaction(conn):
            set_meta(conn, "source_dir", str(source_dir))
            if get_meta(conn, "created_at") is None:
                set_meta(conn, "created_at", now())
            s = Settings.load(conn)
            for key, value in (settings or {}).items():
                if not hasattr(s, key):
                    raise ValueError(f"Unknown setting {key!r}")
                setattr(s, key, type(getattr(s, key))(value))
            s.save(conn)
            if tag_database is not None:
                summary.vocab_imported = import_tag_database(
                    conn, tag_database, tag_database_scheme
                )
            if categories_file is not None:
                apply_categories(conn, load_category_file(categories_file))

        loader = score_loader or default_score_loader
        found = _scan(source_dir, review_dir)
        items: list[_Item] = []
        for rel, path in found:
            st = path.stat()
            cap_path = find_caption(path)
            caption = None
            if cap_path is not None:
                caption = parse_caption(cap_path.read_text(encoding="utf-8"))
            raw = loader(path)
            items.append(
                _Item(
                    rel,
                    path,
                    st.st_size,
                    st.st_mtime,
                    caption,
                    parse_scores(raw) if raw is not None else {},
                )
            )

        # Caption style is detected once, from the first publish.
        if get_meta(conn, "caption_style") is None:
            style = detect_style(i.caption for i in items if i.caption is not None)
            with transaction(conn):
                set_meta(conn, "caption_style", style.to_json())

        existing = {
            r["rel_path"]: r
            for r in conn.execute(
                "SELECT id, rel_path, src_size, src_mtime FROM images"
            ).fetchall()
        }

        # Assign ids up front so files can be converted in parallel.
        to_convert: list[tuple[int, _Item]] = []
        ids: dict[str, int] = {}
        with transaction(conn):
            for item in items:
                row = existing.get(item.rel)
                if row is None:
                    cur = conn.execute(
                        "INSERT INTO images(rel_path) VALUES(?)", (item.rel,)
                    )
                    assert cur.lastrowid is not None
                    image_id = cur.lastrowid
                else:
                    image_id = row["id"]
                ids[item.rel] = image_id
                unchanged = (
                    row is not None
                    and row["src_size"] == item.size
                    and abs(row["src_mtime"] - item.mtime) < 1e-3
                    and image_path(review_dir, image_id).is_file()
                    and thumb_path(review_dir, image_id).is_file()
                )
                if not unchanged:
                    to_convert.append((image_id, item))
            gone = set(existing) - set(ids)
            conn.execute("UPDATE images SET missing = 0")
            conn.executemany(
                "UPDATE images SET missing = 1 WHERE rel_path = ?",
                [(rel,) for rel in gone],
            )
            summary.images_missing = len(gone)

        def work(job: tuple[int, _Item]) -> tuple[int, _Item, Any]:
            image_id, item = job
            try:
                dims = convert_image(
                    item.path,
                    image_path(review_dir, image_id),
                    thumb_path(review_dir, image_id),
                    max_pixels,
                    quality,
                    thumb_size,
                )
                return image_id, item, dims
            except Exception as e:
                return image_id, item, e

        failed_ids: set[int] = set()
        total = len(to_convert)
        workers = workers or min(8, os.cpu_count() or 2)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for done, (image_id, item, result) in enumerate(
                pool.map(work, to_convert), 1
            ):
                if isinstance(result, Exception):
                    summary.images_failed.append(f"{item.rel}: {result}")
                    failed_ids.add(image_id)
                else:
                    w, h, ow, oh = result
                    conn.execute(
                        """UPDATE images SET width = ?, height = ?, orig_width = ?,
                           orig_height = ?, src_size = ?, src_mtime = ?,
                           updated_at = ? WHERE id = ?""",
                        (w, h, ow, oh, item.size, item.mtime, now(), image_id),
                    )
                    summary.images_converted += 1
                if progress is not None:
                    progress(done, total)

        with transaction(conn):
            s = Settings.load(conn)
            for item in items:
                image_id = ids[item.rel]
                if image_id in failed_ids and item.rel not in existing:
                    conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
                    continue
                _sync_tags(conn, image_id, item, s)
                recompute(conn, image_id, s=s)

        summary.images_total = conn.execute(
            "SELECT COUNT(*) FROM images WHERE missing = 0"
        ).fetchone()[0]
        return summary
    finally:
        conn.close()


def _sync_tags(
    conn: sqlite3.Connection, image_id: int, item: _Item, s: Settings
) -> None:
    caption_tags: list[str] = []
    if item.caption is not None:
        caption_tags = [resolve_alias(conn, t) for t in item.caption.tags]
        conn.execute(
            "UPDATE images SET caption_extra = ? WHERE id = ?",
            (json.dumps(item.caption.extra), image_id),
        )

    scores: dict[str, tuple[float, str | None]] = {}
    for tag, (score, cat) in item.scores.items():
        if score >= s.store_min_score:
            name = resolve_alias(conn, tag)
            if name not in scores or score > scores[name][0]:
                scores[name] = (score, cat)
    if len(scores) > MAX_TAGS_PER_IMAGE:
        keep = sorted(scores.items(), key=lambda kv: -kv[1][0])[:MAX_TAGS_PER_IMAGE]
        scores = dict(keep)

    # No caption file: the scores above the threshold *are* the caption.
    if item.caption is None:
        caption_tags = [
            t
            for t, _ in sorted(scores.items(), key=lambda kv: -kv[1][0])
            if scores[t][0] >= s.present_threshold
        ]

    positions = {t: i for i, t in enumerate(dict.fromkeys(caption_tags))}
    wanted = set(positions) | set(scores)

    tag_ids: dict[str, int] = {}
    for tag in wanted:
        tag_ids[tag] = get_or_create_tag(conn, tag)
        hint = scores.get(tag, (0.0, None))[1]
        if hint:
            conn.execute(
                "UPDATE tags SET category = ? WHERE id = ? "
                "AND category = 'unknown' AND category_locked = 0",
                (hint, tag_ids[tag]),
            )

    for tag in wanted:
        score = scores[tag][0] if tag in scores else None
        pos = positions.get(tag)
        conn.execute(
            """
            INSERT INTO image_tags(image_id, tag_id, in_caption, caption_pos, score)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(image_id, tag_id) DO UPDATE SET
                in_caption = excluded.in_caption,
                caption_pos = excluded.caption_pos,
                score = excluded.score
            """,
            (image_id, tag_ids[tag], int(pos is not None), pos, score),
        )

    # Tags that disappeared from both caption and scores: drop them unless a
    # reviewer added or voted on them.
    keep_ids = set(tag_ids.values())
    stale = conn.execute(
        "SELECT tag_id, added_by, n_votes, resolution FROM image_tags "
        "WHERE image_id = ?",
        (image_id,),
    ).fetchall()
    for r in stale:
        if r["tag_id"] in keep_ids:
            continue
        if r["added_by"] is None and r["n_votes"] == 0 and r["resolution"] is None:
            conn.execute(
                "DELETE FROM image_tags WHERE image_id = ? AND tag_id = ?",
                (image_id, r["tag_id"]),
            )
        else:
            conn.execute(
                "UPDATE image_tags SET in_caption = 0, caption_pos = NULL, "
                "score = NULL WHERE image_id = ? AND tag_id = ?",
                (image_id, r["tag_id"]),
            )


def caption_style(conn: sqlite3.Connection) -> CaptionStyle:
    return CaptionStyle.from_json(get_meta(conn, "caption_style"))
