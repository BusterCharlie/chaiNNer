"""Command-line interface.  Run ``python -m dataset_review --help`` from
``backend/src`` (or with ``backend/src`` on ``PYTHONPATH``)."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, fields
from pathlib import Path

from . import auth, review
from .consensus import Settings, recompute_all
from .db import connect, get_meta, set_meta, transaction
from .export import export_captions
from .ingest import load_score_loader, publish_dataset
from .server import ServerConfig, serve
from .tags import apply_categories, import_tag_database, load_category_file


def _setting_pairs(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    known = {f.name for f in fields(Settings)}
    for item in values:
        key, sep, value = item.partition("=")
        if not sep or key not in known:
            raise SystemExit(f"Unknown setting {item!r}. Known: {', '.join(known)}")
        out[key] = value
    return out


def _coerce(s: Settings, pairs: dict[str, str]) -> None:
    for key, raw in pairs.items():
        current = getattr(s, key)
        if isinstance(current, bool):
            value: object = raw.lower() in ("1", "true", "yes", "on")
        else:
            value = type(current)(raw)
        setattr(s, key, value)


def _invite_link(conn: object, key: str, url: str | None) -> str:
    base = url or get_meta(conn, "public_url") or "http://127.0.0.1:8765"  # type: ignore[arg-type]
    return f"{base.rstrip('/')}/#key={key}"


def cmd_publish(a: argparse.Namespace) -> None:
    pairs = _setting_pairs(a.set or [])
    s = Settings()
    _coerce(s, pairs)
    settings = {k: getattr(s, k) for k in pairs}

    def progress(done: int, total: int) -> None:
        if done == total or done % 50 == 0:
            print(f"\r  converting {done}/{total}", end="", file=sys.stderr)
            if done == total:
                print(file=sys.stderr)

    summary = publish_dataset(
        a.source,
        a.review_dir,
        max_pixels=int(a.max_megapixels * 1_000_000),
        quality=a.quality,
        tag_database=a.tag_db,
        tag_database_scheme=a.tag_db_scheme,
        categories_file=a.categories,
        score_loader=load_score_loader(a.score_loader) if a.score_loader else None,
        settings=settings,
        progress=progress,
    )
    if a.title:
        conn = connect(a.review_dir)
        with transaction(conn):
            set_meta(conn, "title", a.title)
        conn.close()
    print(summary)
    for failure in summary.images_failed[:20]:
        print(f"  failed: {failure}", file=sys.stderr)


def cmd_serve(a: argparse.Namespace) -> None:
    if bool(a.tls_cert) != bool(a.tls_key):
        raise SystemExit("--tls-cert and --tls-key go together")
    if a.public_url:
        conn = connect(a.review_dir)
        with transaction(conn):
            set_meta(conn, "public_url", a.public_url.rstrip("/"))
        conn.close()
    serve(
        ServerConfig(
            review_dir=a.review_dir,
            host=a.host,
            port=a.port,
            public_url=a.public_url,
            trust_proxy=a.trust_proxy,
            tls_cert=a.tls_cert,
            tls_key=a.tls_key,
            extra_hosts=a.allow_host or [],
        )
    )


def cmd_user(a: argparse.Namespace) -> None:
    conn = connect(a.review_dir)
    try:
        if a.action == "add":
            key = auth.create_user(
                conn, a.name, "admin" if a.admin else "reviewer", a.weight
            )
            print(f"Created {a.name}. Send them this link (it is their password):")
            print(_invite_link(conn, key, a.url))
        elif a.action == "rotate":
            key = auth.rotate_key(conn, a.name)
            print(f"New link for {a.name} (old link and sessions no longer work):")
            print(_invite_link(conn, key, a.url))
        elif a.action == "revoke":
            auth.revoke_user(conn, a.name)
            with transaction(conn):
                recompute_all(conn)
            print(f"Revoked {a.name}; their votes no longer count.")
        elif a.action == "set":
            auth.set_user(
                conn,
                a.name,
                role=("admin" if a.admin else None),
                weight=a.weight_set,
            )
            with transaction(conn):
                recompute_all(conn)
            print(f"Updated {a.name}.")
        elif a.action == "purge-votes":
            n = review.purge_user_votes(conn, a.name)
            print(f"Removed {n} votes by {a.name}.")
        elif a.action == "list":
            rows = conn.execute(
                """SELECT u.name, u.role, u.weight, u.revoked, u.last_seen,
                          COUNT(v.vote) AS votes
                   FROM users u LEFT JOIN votes v ON v.user_id = u.id
                   GROUP BY u.id ORDER BY u.name"""
            ).fetchall()
            for r in rows:
                state = "revoked" if r["revoked"] else r["role"]
                print(
                    f"{r['name']:<24} {state:<9} weight={r['weight']:<4} "
                    f"votes={r['votes']}"
                )
    finally:
        conn.close()


def cmd_status(a: argparse.Namespace) -> None:
    conn = connect(a.review_dir)
    try:
        print(f"source: {get_meta(conn, 'source_dir')}")
        print(f"public url: {get_meta(conn, 'public_url') or '(not set)'}")
        images = conn.execute(
            "SELECT COUNT(*), SUM(need > 0), SUM(missing) FROM images"
        ).fetchone()
        print(
            f"images: {images[0] - (images[2] or 0)} "
            f"({images[1] or 0} still need review, {images[2] or 0} missing from source)"
        )
        for status, n in conn.execute(
            "SELECT status, COUNT(*) FROM image_tags GROUP BY status ORDER BY 2 DESC"
        ):
            print(f"  {status:<12} {n}")
        votes = conn.execute("SELECT COUNT(*) FROM votes").fetchone()[0]
        print(f"votes: {votes}")
    finally:
        conn.close()


def cmd_settings(a: argparse.Namespace) -> None:
    conn = connect(a.review_dir)
    try:
        s = Settings.load(conn)
        if a.set:
            _coerce(s, _setting_pairs(a.set))
            with transaction(conn):
                s.save(conn)
                recompute_all(conn)
        for k, v in asdict(s).items():
            print(f"{k} = {v}")
    finally:
        conn.close()


def cmd_vocab(a: argparse.Namespace) -> None:
    conn = connect(a.review_dir)
    try:
        with transaction(conn):
            if a.tag_db:
                n = import_tag_database(conn, a.tag_db, a.tag_db_scheme)
                print(f"Imported {n} tags from {a.tag_db}")
            if a.categories:
                n = apply_categories(conn, load_category_file(a.categories))
                print(f"Applied {n} category assignments")
    finally:
        conn.close()


def cmd_export(a: argparse.Namespace) -> None:
    summary = export_captions(
        a.review_dir,
        a.out,
        to_source=a.to_source,
        accept_single_votes=a.accept_single_votes,
    )
    print(summary)
    print(f"report: {a.review_dir / 'export-report.json'}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="dataset_review",
        description="Collaborative tag review for image training datasets.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pub = sub.add_parser(
        "publish", help="copy a dataset into a review set (or re-sync)"
    )
    pub.add_argument(
        "source", type=Path, help="dataset folder (images + .txt captions)"
    )
    pub.add_argument("review_dir", type=Path, help="where the review set lives")
    pub.add_argument("--max-megapixels", type=float, default=1.0)
    pub.add_argument("--quality", type=int, default=82, help="WebP quality")
    pub.add_argument("--tag-db", type=Path, help="booru/e621 tag CSV for autocomplete")
    pub.add_argument(
        "--tag-db-scheme",
        default="auto",
        choices=["auto", "danbooru", "e621", "merged"],
    )
    pub.add_argument("--categories", type=Path, help="your tag→category file")
    pub.add_argument(
        "--score-loader",
        help="custom auto-tag score reader, 'module:function' (path -> {tag: score})",
    )
    pub.add_argument("--title", help="name shown in the app")
    pub.add_argument("--set", action="append", metavar="KEY=VALUE")
    pub.set_defaults(func=cmd_publish)

    srv = sub.add_parser("serve", help="run the review web app")
    srv.add_argument("review_dir", type=Path)
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8765)
    srv.add_argument(
        "--public-url",
        help="URL friends use, e.g. https://box.tailnet.ts.net (used in invite "
        "links and for Host checks)",
    )
    srv.add_argument(
        "--trust-proxy",
        action="store_true",
        help="trust X-Forwarded-For/Proto from a reverse proxy on this machine",
    )
    srv.add_argument("--tls-cert", type=Path)
    srv.add_argument("--tls-key", type=Path)
    srv.add_argument("--allow-host", action="append", help="extra allowed Host header")
    srv.set_defaults(func=cmd_serve)

    usr = sub.add_parser("user", help="manage reviewers")
    usr.add_argument(
        "action", choices=["add", "list", "rotate", "revoke", "set", "purge-votes"]
    )
    usr.add_argument("review_dir", type=Path)
    usr.add_argument("name", nargs="?")
    usr.add_argument("--admin", action="store_true", help="give admin rights")
    usr.add_argument("--weight", type=float, default=1.0, help="vote weight (add)")
    usr.add_argument("--set-weight", dest="weight_set", type=float, help="(set)")
    usr.add_argument("--url", help="base URL for the invite link")
    usr.set_defaults(func=cmd_user)

    st = sub.add_parser("status", help="show review progress")
    st.add_argument("review_dir", type=Path)
    st.set_defaults(func=cmd_status)

    cfg = sub.add_parser("settings", help="show or change consensus settings")
    cfg.add_argument("review_dir", type=Path)
    cfg.add_argument("set", nargs="*", metavar="KEY=VALUE")
    cfg.set_defaults(func=cmd_settings)

    voc = sub.add_parser("vocab", help="import a tag database or category file")
    voc.add_argument("review_dir", type=Path)
    voc.add_argument("--tag-db", type=Path)
    voc.add_argument(
        "--tag-db-scheme",
        default="auto",
        choices=["auto", "danbooru", "e621", "merged"],
    )
    voc.add_argument("--categories", type=Path)
    voc.set_defaults(func=cmd_vocab)

    exp = sub.add_parser("export", help="write reviewed captions")
    exp.add_argument("review_dir", type=Path)
    exp.add_argument("--out", type=Path, help="default: <review_dir>/export")
    exp.add_argument(
        "--to-source",
        action="store_true",
        help="overwrite captions in the original dataset (backs up to .bak)",
    )
    exp.add_argument(
        "--accept-single-votes",
        action="store_true",
        help="apply tags with only one vote (default: needs confirmation)",
    )
    exp.set_defaults(func=cmd_export)

    a = p.parse_args(argv)
    if a.cmd == "user" and a.action != "list" and not a.name:
        p.error("user name required")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        a.func(a)
    except (FileNotFoundError, KeyError, ValueError) as e:
        raise SystemExit(f"error: {e}") from None


if __name__ == "__main__":
    main()
