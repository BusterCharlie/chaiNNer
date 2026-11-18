"""Write reviewed captions back out.

By default only *settled* decisions change a caption: tags confirmed by enough
votes are added, rejected ones removed, admin resolutions applied.  Unreviewed,
single-vote and contested tags keep their original caption state (contested
ones are listed in the report so they can be resolved).
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .consensus import is_present
from .db import connect, get_meta
from .ingest import caption_style
from .tags import category_sort_key


@dataclass
class ExportSummary:
    out_dir: Path
    images: int = 0
    changed: int = 0
    tags_added: int = 0
    tags_removed: int = 0
    contested: list[dict[str, Any]] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"wrote {self.images} captions to {self.out_dir} "
            f"({self.changed} changed, +{self.tags_added} / -{self.tags_removed} tags, "
            f"{len(self.contested)} contested left as-is)"
        )


def _extra(row: sqlite3.Row) -> list[str]:
    try:
        data = json.loads(row["caption_extra"] or "[]")
    except ValueError:
        return []
    return [x for x in data if isinstance(x, str)]


def export_captions(
    review_dir: Path,
    out_dir: Path | None = None,
    *,
    to_source: bool = False,
    accept_single_votes: bool = False,
    caption_ext: str = ".txt",
) -> ExportSummary:
    """Write one caption file per image.

    ``out_dir`` defaults to ``<review_dir>/export``.  With ``to_source`` the
    captions are written next to the original images; existing captions are
    first backed up as ``<name>.txt.bak`` (once; an existing backup is kept).
    """
    conn = connect(review_dir)
    try:
        style = caption_style(conn)
        source = Path(get_meta(conn, "source_dir"))
        if to_source:
            target_root = source
        else:
            target_root = (out_dir or review_dir / "export").resolve()
        summary = ExportSummary(out_dir=target_root)

        images = conn.execute(
            "SELECT id, rel_path, caption_extra FROM images WHERE missing = 0 "
            "ORDER BY id"
        ).fetchall()
        for img in images:
            rows = conn.execute(
                """
                SELECT it.*, t.name, t.category FROM image_tags it
                JOIN tags t ON t.id = it.tag_id WHERE it.image_id = ?
                """,
                (img["id"],),
            ).fetchall()
            kept: list[sqlite3.Row] = []
            for r in rows:
                present = is_present(
                    r["status"],
                    r["resolution"],
                    bool(r["in_caption"]),
                    accept_leaning=accept_single_votes,
                )
                if present:
                    kept.append(r)
                    if not r["in_caption"]:
                        summary.tags_added += 1
                elif r["in_caption"]:
                    summary.tags_removed += 1
                if r["status"] == "contested":
                    summary.contested.append(
                        {
                            "image": img["rel_path"],
                            "tag": r["name"],
                            "yes": r["yes"],
                            "no": r["no"],
                            "in_caption": bool(r["in_caption"]),
                        }
                    )

            # Original tags keep their order; new ones follow, grouped by
            # category and most confident first.
            original = sorted(
                (r for r in kept if r["caption_pos"] is not None),
                key=lambda r: r["caption_pos"],
            )
            added = sorted(
                (r for r in kept if r["caption_pos"] is None),
                key=lambda r: (category_sort_key(r["category"]), -r["posterior"]),
            )
            changed = {r["name"] for r in kept} != {
                r["name"] for r in rows if r["in_caption"]
            }
            parts = [style.format(r["name"]) for r in (*original, *added)]
            parts.extend(_extra(img))
            text = ", ".join(parts)

            dest = (target_root / img["rel_path"]).with_suffix(caption_ext)
            if dest.resolve().is_relative_to(target_root.resolve()) is False:
                continue  # never write outside the target directory
            dest.parent.mkdir(parents=True, exist_ok=True)
            if to_source and dest.exists():
                backup = dest.with_name(dest.name + ".bak")
                if not backup.exists():
                    shutil.copy2(dest, backup)
            old = dest.read_text(encoding="utf-8") if dest.exists() else None
            if old != text:
                dest.write_text(text, encoding="utf-8")
            if changed:
                summary.changed += 1
            summary.images += 1

        report = review_dir / "export-report.json"
        report.write_text(
            json.dumps(
                {
                    "images": summary.images,
                    "changed": summary.changed,
                    "tags_added": summary.tags_added,
                    "tags_removed": summary.tags_removed,
                    "contested": summary.contested,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return summary
    finally:
        conn.close()
