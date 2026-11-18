"""HTTP server for the review web app (standard library only).

Hardening, beyond the login scheme described in :mod:`dataset_review.auth`:

- binds to 127.0.0.1 unless told otherwise; meant to sit behind an HTTPS
  tunnel or reverse proxy (Tailscale Serve/Funnel, Cloudflare Tunnel, Caddy),
  or to terminate TLS itself with ``--tls-cert``/``--tls-key``
- every API call and image needs a session; nothing is public except the
  static app shell
- state-changing requests need a custom header (forces a CORS preflight,
  which is never granted) and a matching ``Origin``; cookies are
  ``SameSite=Strict``
- ``Host`` is checked against the configured public URL (DNS rebinding)
- strict Content-Security-Policy, no inline script, ``frame-ancestors 'none'``
- files are served by integer id or from a fixed allow-list, never by a
  client-supplied path; request bodies are capped at 64 KiB
"""

from __future__ import annotations

import json
import logging
import mimetypes
import re
import socket
import sqlite3
import ssl
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from . import auth, review
from .db import connect, get_meta
from .ingest import image_path, thumb_path
from .review import ReviewError
from .tags import display_tag, search_tags

log = logging.getLogger("dataset_review")

COOKIE_NAME = "dsr_session"
CSRF_HEADER = "X-DSR"
MAX_BODY = 64 * 1024

CSP = "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; connect-src 'self'; manifest-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
BASE_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
}

STATIC_FILES = {
    "/": "index.html",
    "/app.js": "app.js",
    "/app.css": "app.css",
    "/manifest.webmanifest": "manifest.webmanifest",
    "/icon.svg": "icon.svg",
}


def _load_static() -> dict[str, tuple[bytes, str]]:
    out: dict[str, tuple[bytes, str]] = {}
    pkg = resources.files("dataset_review") / "static"
    for route, name in STATIC_FILES.items():
        data = (pkg / name).read_bytes()
        ctype = (
            {
                ".html": "text/html; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".webmanifest": "application/manifest+json",
                ".svg": "image/svg+xml",
            }.get(Path(name).suffix)
            or mimetypes.guess_type(name)[0]
            or "text/plain"
        )
        out[route] = (data, ctype)
    return out


@dataclass
class ServerConfig:
    review_dir: Path
    host: str = "127.0.0.1"
    port: int = 8765
    public_url: str | None = None
    trust_proxy: bool = False
    tls_cert: Path | None = None
    tls_key: Path | None = None
    extra_hosts: list[str] = field(default_factory=list)

    @property
    def secure(self) -> bool:
        return self.tls_cert is not None or (
            self.public_url is not None and self.public_url.startswith("https://")
        )

    def allowed_hosts(self) -> set[str] | None:
        if self.public_url is None and not self.extra_hosts:
            return None  # LAN mode: any Host accepted
        hosts = {"localhost", "127.0.0.1", "[::1]", *self.extra_hosts}
        if self.public_url:
            netloc = urlsplit(self.public_url).netloc.lower()
            hosts.add(netloc)
            hosts.add(netloc.rsplit(":", 1)[0] if "]" not in netloc else netloc)
        return {h.lower() for h in hosts}


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.static = _load_static()
        self.throttle = auth.LoginThrottle()
        self._local = threading.local()
        # make sure the database exists and is a review set
        conn = connect(config.review_dir)
        if get_meta(conn, "source_dir") is None:
            raise ValueError(f"{config.review_dir} is not a published review set")
        conn.close()
        family = socket.AF_INET6 if ":" in config.host else socket.AF_INET
        self.address_family = family
        super().__init__((config.host, config.port), ReviewHandler)
        if config.tls_cert is not None:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ctx.load_cert_chain(config.tls_cert, config.tls_key)
            self.socket = ctx.wrap_socket(self.socket, server_side=True)

    def db(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = connect(self.config.review_dir)
            self._local.conn = conn
        return conn


class HttpError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


Route = Callable[["ReviewHandler", "re.Match[str]", auth.User | None], None]


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "dataset-review"
    sys_version = ""
    protocol_version = "HTTP/1.1"
    timeout = 30  # drop idle / slow connections

    # -- plumbing -----------------------------------------------------------

    @property
    def app(self) -> ReviewServer:
        assert isinstance(self.server, ReviewServer)
        return self.server

    def log_message(self, format: str, *args: Any) -> None:
        log.info("%s %s", self.client_ip(), format % args)

    def client_ip(self) -> str:
        if self.app.config.trust_proxy:
            fwd = self.headers.get("X-Forwarded-For")
            if fwd:
                return fwd.split(",")[-1].strip()
        return self.client_address[0]

    def is_https(self) -> bool:
        if self.app.config.tls_cert is not None:
            return True
        return (
            self.app.config.trust_proxy
            and self.headers.get("X-Forwarded-Proto", "").lower() == "https"
        )

    def send(
        self,
        status: HTTPStatus,
        body: bytes,
        ctype: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        for k, v in BASE_HEADERS.items():
            self.send_header(k, v)
        if self.app.config.secure or self.is_https():
            self.send_header("Strict-Transport-Security", "max-age=31536000")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_json(
        self,
        data: Any,
        status: HTTPStatus = HTTPStatus.OK,
        headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(data, separators=(",", ":")).encode("utf-8")
        self.send(
            status,
            body,
            "application/json",
            {"Cache-Control": "no-store", **(headers or {})},
        )

    def session_token(self) -> str | None:
        raw = self.headers.get("Cookie")
        if not raw:
            return None
        try:
            cookie = SimpleCookie(raw)
        except CookieError:
            return None
        morsel = cookie.get(COOKIE_NAME)
        return morsel.value if morsel else None

    def cookie_header(self, token: str, max_age: int) -> str:
        parts = [
            f"{COOKIE_NAME}={token}",
            "Path=/",
            f"Max-Age={max_age}",
            "HttpOnly",
            "SameSite=Strict",
        ]
        if self.app.config.secure or self.is_https():
            parts.append("Secure")
        return "; ".join(parts)

    def read_json(self) -> Any:
        ctype = self.headers.get("Content-Type", "")
        if not ctype.split(";")[0].strip().lower() == "application/json":
            raise HttpError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "JSON required")
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            raise HttpError(HTTPStatus.LENGTH_REQUIRED, "Length required") from None
        if length < 0 or length > MAX_BODY:
            raise HttpError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Body too large")
        try:
            return json.loads(self.rfile.read(length) or b"null")
        except ValueError:
            raise HttpError(HTTPStatus.BAD_REQUEST, "Invalid JSON") from None

    def check_host(self) -> None:
        allowed = self.app.config.allowed_hosts()
        if allowed is None:
            return
        host = (self.headers.get("Host") or "").lower()
        if host not in allowed and host.rsplit(":", 1)[0] not in allowed:
            raise HttpError(HTTPStatus.MISDIRECTED_REQUEST, "Unknown host")

    def check_csrf(self) -> None:
        if self.headers.get(CSRF_HEADER) != "1":
            raise HttpError(HTTPStatus.FORBIDDEN, "Missing request header")
        origin = self.headers.get("Origin")
        if origin is not None:
            host = (self.headers.get("Host") or "").lower()
            if urlsplit(origin).netloc.lower() != host:
                raise HttpError(HTTPStatus.FORBIDDEN, "Cross-origin request")

    # -- dispatch -----------------------------------------------------------

    def do_GET(self) -> None:
        self.dispatch(GET_ROUTES)

    def do_HEAD(self) -> None:
        self.dispatch(GET_ROUTES)

    def do_POST(self) -> None:
        self.dispatch(POST_ROUTES)

    def dispatch(self, routes: list[tuple[re.Pattern[str], bool, Route]]) -> None:
        try:
            self.check_host()
            path = urlsplit(self.path).path
            if self.command == "POST":
                self.check_csrf()
            for pattern, needs_auth, handler in routes:
                m = pattern.fullmatch(path)
                if m is None:
                    continue
                user = auth.session_user(self.app.db(), self.session_token())
                if needs_auth and user is None:
                    raise HttpError(HTTPStatus.UNAUTHORIZED, "Not logged in")
                handler(self, m, user)
                return
            raise HttpError(HTTPStatus.NOT_FOUND, "Not found")
        except HttpError as e:
            # the body may not have been read; don't reuse the connection
            self.close_connection = True
            self.send_json({"error": e.message}, e.status)
        except ReviewError as e:
            self.send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
        except (ConnectionError, TimeoutError):
            pass
        except Exception:
            self.close_connection = True
            log.exception("Error handling %s %s", self.command, self.path)
            self.send_json(
                {"error": "Internal error"}, HTTPStatus.INTERNAL_SERVER_ERROR
            )

    def query(self) -> dict[str, str]:
        q = parse_qs(urlsplit(self.path).query, max_num_fields=20)
        return {k: v[0] for k, v in q.items()}

    def int_param(self, name: str, default: int) -> int:
        raw = self.query().get(name)
        if raw is None or raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            raise HttpError(HTTPStatus.BAD_REQUEST, f"Bad {name}") from None


def _require_admin(user: auth.User | None) -> auth.User:
    if user is None or not user.is_admin:
        raise HttpError(HTTPStatus.FORBIDDEN, "Admins only")
    return user


def _user(user: auth.User | None) -> auth.User:
    assert user is not None
    return user


# -- handlers -----------------------------------------------------------------


def h_static(h: ReviewHandler, m: re.Match[str], _user: auth.User | None) -> None:
    body, ctype = h.app.static[m.group(0)]
    h.send(HTTPStatus.OK, body, ctype, {"Cache-Control": "no-cache"})


def h_file(kind: str) -> Route:
    def handler(h: ReviewHandler, m: re.Match[str], _u: auth.User | None) -> None:
        image_id = int(m.group(1))
        row = (
            h.app.db()
            .execute("SELECT 1 FROM images WHERE id = ? AND missing = 0", (image_id,))
            .fetchone()
        )
        if row is None:
            raise HttpError(HTTPStatus.NOT_FOUND, "Not found")
        path = (image_path if kind == "img" else thumb_path)(
            h.app.config.review_dir, image_id
        )
        try:
            body = path.read_bytes()
        except FileNotFoundError:
            raise HttpError(HTTPStatus.NOT_FOUND, "Not found") from None
        h.send(
            HTTPStatus.OK,
            body,
            "image/webp",
            # URLs carry ?v=<mtime>, so they can be cached for good
            {"Cache-Control": "private, max-age=31536000, immutable"},
        )

    return handler


def h_login(h: ReviewHandler, _m: re.Match[str], _u: auth.User | None) -> None:
    client = h.client_ip()
    if not h.app.throttle.allowed(client):
        raise HttpError(HTTPStatus.TOO_MANY_REQUESTS, "Too many attempts, try later")
    body = h.read_json()
    key = body.get("key") if isinstance(body, dict) else None
    result = auth.login(h.app.db(), key) if isinstance(key, str) else None
    if result is None:
        h.app.throttle.failed(client)
        log.warning("Failed login from %s", client)
        raise HttpError(HTTPStatus.UNAUTHORIZED, "That login link is not valid")
    user, token = result
    log.info("Login: %s from %s", user.name, client)
    h.send_json(
        {"name": user.name, "role": user.role},
        headers={"Set-Cookie": h.cookie_header(token, auth.SESSION_TTL)},
    )


def h_logout(h: ReviewHandler, _m: re.Match[str], _u: auth.User | None) -> None:
    token = h.session_token()
    if token:
        auth.logout(h.app.db(), token)
    h.send_json({"ok": True}, headers={"Set-Cookie": h.cookie_header("", 0)})


def h_me(h: ReviewHandler, _m: re.Match[str], user: auth.User | None) -> None:
    u = _user(user)
    conn = h.app.db()
    h.send_json(
        {
            "name": u.name,
            "role": u.role,
            "title": get_meta(conn, "title") or "Dataset review",
            "categories": review.categories(conn),
        }
    )


def h_next(h: ReviewHandler, _m: re.Match[str], user: auth.User | None) -> None:
    u = _user(user)
    q = h.query()
    mode = q.get("mode", "verify")
    conn = h.app.db()
    exclude = frozenset(
        int(x) for x in q.get("exclude", "").split(",")[:20] if x.isdigit()
    )
    image_id = review.pick_image(conn, u, mode, exclude)
    if image_id is None:
        h.send_json({"done": True, "mode": mode})
        return
    h.send_json(review.image_payload(conn, u, image_id, mode, h.int_param("n", 12)))


def h_image(h: ReviewHandler, m: re.Match[str], user: auth.User | None) -> None:
    u = _user(user)
    mode = h.query().get("mode", "detail")
    h.send_json(
        review.image_payload(h.app.db(), u, int(m.group(1)), mode, h.int_param("n", 40))
    )


def h_votes(h: ReviewHandler, m: re.Match[str], user: auth.User | None) -> None:
    u = _user(user)
    body = h.read_json()
    if not isinstance(body, dict):
        raise HttpError(HTTPStatus.BAD_REQUEST, "Object expected")
    image_id = int(m.group(1))
    review.submit_votes(
        h.app.db(), u, image_id, str(body.get("mode")), body.get("votes", [])
    )
    h.send_json({"ok": True})


def h_add_tag(h: ReviewHandler, m: re.Match[str], user: auth.User | None) -> None:
    u = _user(user)
    body = h.read_json()
    if not isinstance(body, dict):
        raise HttpError(HTTPStatus.BAD_REQUEST, "Object expected")
    tag = review.add_tag(h.app.db(), u, int(m.group(1)), body.get("name"))
    h.send_json({"tag": tag})


def h_search(h: ReviewHandler, _m: re.Match[str], _u: auth.User | None) -> None:
    q = h.query().get("q", "")[:120]
    rows = search_tags(h.app.db(), q, limit=min(h.int_param("limit", 12), 30))
    h.send_json(
        {
            "results": [
                {
                    "name": r["name"],
                    "label": display_tag(r["name"]),
                    "category": r["category"],
                    "count": r["post_count"],
                    "alias": display_tag(r["alias"]) if r["alias"] else None,
                }
                for r in rows
            ]
        }
    )


def h_browse(h: ReviewHandler, _m: re.Match[str], user: auth.User | None) -> None:
    u = _user(user)
    q = h.query()
    h.send_json(
        review.browse(
            h.app.db(),
            u,
            filt=q.get("filter", "all"),
            tag=q.get("tag") or None,
            after=h.int_param("after", 0),
            limit=h.int_param("limit", 60),
        )
    )


def h_stats(h: ReviewHandler, _m: re.Match[str], user: auth.User | None) -> None:
    h.send_json(review.stats(h.app.db(), _user(user)))


def h_contested(h: ReviewHandler, _m: re.Match[str], user: auth.User | None) -> None:
    _require_admin(user)
    h.send_json(
        review.contested(
            h.app.db(),
            after=(h.int_param("after_image", 0), h.int_param("after_tag", 0)),
        )
    )


def h_resolve(h: ReviewHandler, _m: re.Match[str], user: auth.User | None) -> None:
    admin = _require_admin(user)
    body = h.read_json()
    if not isinstance(body, dict):
        raise HttpError(HTTPStatus.BAD_REQUEST, "Object expected")
    image_id, tag_id = body.get("image_id"), body.get("tag_id")
    if not isinstance(image_id, int) or not isinstance(tag_id, int):
        raise HttpError(HTTPStatus.BAD_REQUEST, "image_id and tag_id required")
    review.resolve(h.app.db(), admin, image_id, tag_id, body.get("resolution"))
    h.send_json({"ok": True})


def h_uncategorized(
    h: ReviewHandler, _m: re.Match[str], user: auth.User | None
) -> None:
    _require_admin(user)
    h.send_json({"tags": review.uncategorized(h.app.db())})


def h_category(h: ReviewHandler, _m: re.Match[str], user: auth.User | None) -> None:
    _require_admin(user)
    body = h.read_json()
    if not isinstance(body, dict):
        raise HttpError(HTTPStatus.BAD_REQUEST, "Object expected")
    review.set_category(h.app.db(), body.get("tag"), body.get("category"))
    h.send_json({"ok": True})


GET_ROUTES: list[tuple[re.Pattern[str], bool, Route]] = [
    (re.compile("|".join(re.escape(r) for r in STATIC_FILES)), False, h_static),
    (re.compile(r"/img/(\d{1,12})\.webp"), True, h_file("img")),
    (re.compile(r"/thumb/(\d{1,12})\.webp"), True, h_file("thumb")),
    (re.compile(r"/api/me"), True, h_me),
    (re.compile(r"/api/next"), True, h_next),
    (re.compile(r"/api/image/(\d{1,12})"), True, h_image),
    (re.compile(r"/api/tags/search"), True, h_search),
    (re.compile(r"/api/browse"), True, h_browse),
    (re.compile(r"/api/stats"), True, h_stats),
    (re.compile(r"/api/admin/contested"), True, h_contested),
    (re.compile(r"/api/admin/uncategorized"), True, h_uncategorized),
]
POST_ROUTES: list[tuple[re.Pattern[str], bool, Route]] = [
    (re.compile(r"/api/login"), False, h_login),
    (re.compile(r"/api/logout"), False, h_logout),
    (re.compile(r"/api/image/(\d{1,12})/votes"), True, h_votes),
    (re.compile(r"/api/image/(\d{1,12})/tags"), True, h_add_tag),
    (re.compile(r"/api/admin/resolve"), True, h_resolve),
    (re.compile(r"/api/admin/category"), True, h_category),
]


def serve(config: ServerConfig) -> None:
    server = ReviewServer(config)
    scheme = "https" if config.tls_cert else "http"
    log.info(
        "Serving %s on %s://%s:%d", config.review_dir, scheme, config.host, config.port
    )
    if config.host not in ("127.0.0.1", "localhost", "::1") and not config.tls_cert:
        log.warning(
            "Listening on %s without TLS. Login keys and cookies would cross "
            "the network in clear text; prefer an HTTPS tunnel/proxy.",
            config.host,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
