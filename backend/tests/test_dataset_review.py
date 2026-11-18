from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from dataset_review import auth, review  # noqa: E402
from dataset_review.consensus import Settings, classify, posterior  # noqa: E402
from dataset_review.db import connect, get_meta  # noqa: E402
from dataset_review.export import export_captions  # noqa: E402
from dataset_review.ingest import (  # noqa: E402
    image_path,
    parse_scores,
    publish_dataset,
)
from dataset_review.server import ReviewServer, ServerConfig  # noqa: E402
from dataset_review.tags import (  # noqa: E402
    CaptionStyle,
    load_category_file,
    normalize_tag,
    parse_caption,
    search_tags,
)

# ---------------------------------------------------------------------------
# tags


def test_normalize_tag():
    assert normalize_tag("  Long Hair ") == "long_hair"
    assert normalize_tag("hatsune miku \\(cosplay\\)") == "hatsune_miku_(cosplay)"
    assert normalize_tag("^_^") == "^_^"
    assert normalize_tag("") is None
    assert normalize_tag("bad\x00tag") is None
    assert normalize_tag("x" * 500) is None


def test_parse_caption_and_style():
    cap = parse_caption(
        "1girl, long hair, hatsune miku \\(cosplay\\), long hair, ^_^, "
        "a girl standing in a field of flowers at sunset"
    )
    assert cap.tags == ["1girl", "long_hair", "hatsune_miku_(cosplay)", "^_^"]
    assert cap.extra == ["a girl standing in a field of flowers at sunset"]
    style = CaptionStyle(spaces=True, escape_parens=True)
    assert style.format("hatsune_miku_(cosplay)") == "hatsune miku \\(cosplay\\)"
    assert style.format("^_^") == "^_^"
    assert (
        CaptionStyle(spaces=False, escape_parens=False).format("a_b_(c)") == "a_b_(c)"
    )


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"long hair": 0.9, "smile": 0.2}, {"long_hair": 0.9, "smile": 0.2}),
        ({"scores": {"cat": 0.5}}, {"cat": 0.5}),
        ([{"tag": "cat", "confidence": 0.7}], {"cat": 0.7}),
        ([["cat", 0.4], ["dog", 1.5]], {"cat": 0.4, "dog": 1.0}),
        (
            {
                "general": {"cat": 0.4},
                "character": {"miku": 0.8},
                "rating": {"safe": 1},
            },
            {"cat": 0.4, "miku": 0.8},
        ),
    ],
)
def test_parse_scores(data: object, expected: dict[str, float]):
    parsed = parse_scores(data)
    assert {k: v[0] for k, v in parsed.items()} == expected


def test_parse_scores_category_hints():
    parsed = parse_scores({"character": {"miku": 0.8}})
    assert parsed["miku"] == (0.8, "character")


def test_category_file(tmp_path: Path):
    j = tmp_path / "c.json"
    j.write_text(json.dumps({"species": ["Wolf", "fox"], "solo": "meta"}))
    assert load_category_file(j) == {
        "wolf": "species",
        "fox": "species",
        "solo": "meta",
    }
    c = tmp_path / "c.csv"
    c.write_text("tag,category\nlong hair,general\n# comment,x\n")
    assert load_category_file(c) == {"long_hair": "general"}


# ---------------------------------------------------------------------------
# consensus


def test_consensus_rules():
    s = Settings()
    assert classify(0, 0, None, s) == "unreviewed"
    assert classify(1, 0, None, s) == "leaning_yes"
    assert classify(0, 1, None, s) == "leaning_no"
    assert classify(2, 0, None, s) == "confirmed"
    assert classify(0, 2, None, s) == "rejected"
    assert classify(1, 1, None, s) == "contested"
    assert classify(4, 1, None, s) == "confirmed"  # minority 20% < 25%
    assert classify(3, 1, None, s) == "contested"  # minority 25%
    assert classify(1, 1, 1, s) == "resolved"
    assert posterior(0.9, 0, 2, s) == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# publishing


def _make_dataset(root: Path) -> None:
    (root / "sub").mkdir(parents=True)
    Image.new("RGB", (2000, 1500), (200, 30, 30)).save(root / "a.png")
    Image.new("RGB", (640, 480), (30, 200, 30)).save(root / "b.jpg")
    Image.new("RGBA", (300, 300), (30, 30, 200, 128)).save(root / "sub" / "c.webp")
    (root / "a.txt").write_text("1girl, long hair, smile, red background")
    (root / "a.scores.json").write_text(
        json.dumps(
            {
                "general": {
                    "1girl": 0.99,
                    "long_hair": 0.55,
                    "smile": 0.4,
                    "red_background": 0.95,
                    "hat": 0.25,
                    "tail": 0.01,
                },
                "character": {"hatsune_miku": 0.3},
            }
        )
    )
    (root / "b.txt").write_text("landscape, tree")
    # c has no caption: its scores above the threshold become the baseline
    (root / "sub" / "c.json").write_text(json.dumps({"water": 0.9, "boat": 0.2}))


def _make_tag_db(path: Path) -> None:
    path.write_text(
        '1girl,0,5000000,"1girls,sole_female"\n'
        "long_hair,0,3000000,\n"
        "hatsune_miku,4,90000,miku\n"
        "hat,0,800000,\n"
        "wolf,12,100000,\n"
        "solo,0,4000000,\n"
    )


@pytest.fixture
def review_set(tmp_path: Path) -> Path:
    src = tmp_path / "dataset"
    _make_dataset(src)
    db = tmp_path / "tags.csv"
    _make_tag_db(db)
    out = tmp_path / "review"
    summary = publish_dataset(src, out, tag_database=db, workers=2)
    assert summary.images_total == 3
    assert not summary.images_failed
    return out


def test_publish(review_set: Path):
    conn = connect(review_set)
    rows = {r["rel_path"]: r for r in conn.execute("SELECT * FROM images")}
    a = rows["a.png"]
    assert a["width"] * a["height"] <= 1_000_000
    assert (a["orig_width"], a["orig_height"]) == (2000, 1500)
    assert rows["b.jpg"]["width"] == 640
    with Image.open(image_path(review_set, a["id"])) as im:
        assert im.format == "WEBP"
    with Image.open(image_path(review_set, rows["sub/c.webp"]["id"])) as im:
        assert im.mode == "RGBA"

    def tags_of(rel: str) -> dict[str, tuple[int, float | None]]:
        return {
            r["name"]: (r["in_caption"], r["score"])
            for r in conn.execute(
                "SELECT t.name, it.in_caption, it.score FROM image_tags it "
                "JOIN tags t ON t.id = it.tag_id WHERE it.image_id = ?",
                (rows[rel]["id"],),
            )
        }

    a_tags = tags_of("a.png")
    assert a_tags["long_hair"] == (1, 0.55)
    assert a_tags["hat"] == (0, 0.25)
    assert "tail" not in a_tags  # below store_min_score
    assert tags_of("sub/c.webp") == {"water": (1, 0.9), "boat": (0, 0.2)}

    cat = dict(conn.execute("SELECT name, category FROM tags").fetchall())
    assert cat["hatsune_miku"] == "character"
    assert cat["wolf"] == "species"  # e621 id 12 in merged scheme (auto-detected)
    assert get_meta(conn, "caption_style") == {"spaces": True, "escape_parens": True}
    names = [r["name"] for r in search_tags(conn, "1gi")]
    assert names == ["1girl"]
    assert [r["name"] for r in search_tags(conn, "miku")] == ["hatsune_miku"]


def test_republish_keeps_votes(review_set: Path, tmp_path: Path):
    conn = connect(review_set)
    auth.create_user(conn, "ann")
    user = auth.User(1, "ann", "reviewer", 1.0)
    a_id = conn.execute("SELECT id FROM images WHERE rel_path='a.png'").fetchone()[0]
    smile = conn.execute("SELECT id FROM tags WHERE name='smile'").fetchone()[0]
    review.submit_votes(conn, user, a_id, "verify", [{"tag_id": smile, "vote": -1}])

    src = tmp_path / "dataset"
    (src / "a.txt").write_text("1girl, long hair, red background")  # smile removed
    (src / "b.jpg").unlink()
    summary = publish_dataset(src, review_set, workers=1)
    assert summary.images_converted == 0
    assert summary.images_missing == 1
    row = conn.execute(
        "SELECT in_caption, n_votes FROM image_tags WHERE image_id=? AND tag_id=?",
        (a_id, smile),
    ).fetchone()
    assert tuple(row) == (0, 1)


def test_publish_refuses_other_source(review_set: Path, tmp_path: Path):
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="published from"):
        publish_dataset(other, review_set)


# ---------------------------------------------------------------------------
# HTTP


class Client:
    def __init__(self, port: int) -> None:
        self.port = port
        self.cookie: str | None = None

    def request(
        self,
        method: str,
        path: str,
        body: object = None,
        headers: dict[str, str] | None = None,
        csrf: bool = True,
    ) -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        hdrs = {"Host": f"127.0.0.1:{self.port}"}
        if csrf:
            hdrs["X-DSR"] = "1"
        if self.cookie:
            hdrs["Cookie"] = self.cookie
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            hdrs["Content-Type"] = "application/json"
        hdrs.update(headers or {})
        conn.request(method, path, body=data, headers=hdrs)
        res = conn.getresponse()
        payload = res.read()
        out_headers = {k.lower(): v for k, v in res.getheaders()}
        conn.close()
        if "set-cookie" in out_headers:
            self.cookie = out_headers["set-cookie"].split(";")[0]
        return res.status, out_headers, payload

    def json(self, method: str, path: str, body: object = None) -> tuple[int, object]:
        status, _, payload = self.request(method, path, body)
        return status, json.loads(payload) if payload else None


@pytest.fixture
def server(review_set: Path) -> Iterator[tuple[int, Path]]:
    srv = ReviewServer(ServerConfig(review_dir=review_set, port=0))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield srv.server_address[1], review_set
    finally:
        srv.shutdown()
        srv.server_close()


def _login(port: int, review_dir: Path, name: str, role: str = "reviewer") -> Client:
    conn = connect(review_dir)
    key = auth.create_user(conn, name, role)
    conn.close()
    c = Client(port)
    status, _ = c.json("POST", "/api/login", {"key": key})
    assert status == 200
    assert c.cookie and c.cookie.startswith("dsr_session=")
    return c


def test_auth_required(server: tuple[int, Path]):
    port, _ = server
    c = Client(port)
    status, headers, _body = c.request("GET", "/")
    assert status == 200
    assert "default-src 'none'" in headers["content-security-policy"]
    assert headers["x-frame-options"] == "DENY"
    for path in ("/api/me", "/api/next", "/img/1.webp", "/thumb/1.webp", "/api/browse"):
        assert c.request("GET", path)[0] == 401, path
    assert c.request("GET", "/../review.sqlite3")[0] == 404
    assert c.request("GET", "/img/../../review.sqlite3")[0] == 404


def test_login_rejects_bad_key_and_throttles(server: tuple[int, Path]):
    port, _ = server
    c = Client(port)
    codes = [c.json("POST", "/api/login", {"key": "x" * 43})[0] for _ in range(12)]
    assert codes[:10] == [401] * 10
    assert codes[-1] == 429


def test_csrf_and_origin_checks(server: tuple[int, Path]):
    port, review_dir = server
    c = _login(port, review_dir, "bob")
    body = {"mode": "verify", "votes": []}
    assert c.request("POST", "/api/image/1/votes", body, csrf=False)[0] == 403
    assert (
        c.request(
            "POST",
            "/api/image/1/votes",
            body,
            headers={"Origin": "https://evil.example"},
        )[0]
        == 403
    )
    status, _, _ = c.request(
        "POST", "/api/image/1/votes", body, headers={"Content-Type": "text/plain"}
    )
    assert status == 415


def test_host_check(review_set: Path):
    srv = ReviewServer(
        ServerConfig(review_dir=review_set, port=0, public_url="https://review.example")
    )
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        c = Client(srv.server_address[1])
        assert c.request("GET", "/", headers={"Host": "attacker.example"})[0] == 421
        status, headers, _ = c.request("GET", "/", headers={"Host": "review.example"})
        assert status == 200
        assert "strict-transport-security" in headers
    finally:
        srv.shutdown()
        srv.server_close()


def test_review_flow(server: tuple[int, Path]):
    port, review_dir = server
    ann = _login(port, review_dir, "ann")
    bob = _login(port, review_dir, "bob")
    admin = _login(port, review_dir, "boss", "admin")

    status, me = ann.json("GET", "/api/me")
    assert status == 200 and me["name"] == "ann"  # type: ignore[index]

    # verify mode serves an image with tags believed present
    status, p = ann.json("GET", "/api/next?mode=verify&n=5")
    assert status == 200
    payload: dict = p  # type: ignore[assignment]
    assert payload["default_vote"] == 1
    assert 0 < len(payload["tags"]) <= 5
    img_status, img_headers, img_body = ann.request("GET", payload["image"]["url"])
    assert img_status == 200 and img_headers["content-type"] == "image/webp"
    assert img_body[:4] == b"RIFF"
    assert "path" not in payload["image"]  # only admins see file paths

    # a's tags: find long_hair
    conn = connect(review_dir)
    a_id = conn.execute("SELECT id FROM images WHERE rel_path='a.png'").fetchone()[0]
    lh = conn.execute("SELECT id FROM tags WHERE name='long_hair'").fetchone()[0]
    hat = conn.execute("SELECT id FROM tags WHERE name='hat'").fetchone()[0]

    # ann says long_hair is right, bob says wrong -> contested
    assert (
        ann.json(
            "POST",
            f"/api/image/{a_id}/votes",
            {"mode": "verify", "votes": [{"tag_id": lh, "vote": 1}]},
        )[0]
        == 200
    )
    assert (
        bob.json(
            "POST",
            f"/api/image/{a_id}/votes",
            {"mode": "verify", "votes": [{"tag_id": lh, "vote": -1}]},
        )[0]
        == 200
    )
    status_row = conn.execute(
        "SELECT status FROM image_tags WHERE image_id=? AND tag_id=?", (a_id, lh)
    ).fetchone()
    assert status_row[0] == "contested"

    # voting on a tag that isn't on the image is rejected
    other = conn.execute("SELECT id FROM tags WHERE name='wolf'").fetchone()[0]
    assert (
        ann.json(
            "POST",
            f"/api/image/{a_id}/votes",
            {"mode": "verify", "votes": [{"tag_id": other, "vote": 1}]},
        )[0]
        == 400
    )
    assert (
        ann.json(
            "POST",
            f"/api/image/{a_id}/votes",
            {"mode": "verify", "votes": [{"tag_id": lh, "vote": 5}]},
        )[0]
        == 400
    )

    # ann already submitted a in verify mode: she never gets it again there
    seen = set()
    for _ in range(10):
        _, nxt = ann.json("GET", "/api/next?mode=verify")
        if not nxt.get("done"):  # type: ignore[union-attr]
            seen.add(nxt["image"]["id"])  # type: ignore[index]
    assert a_id not in seen

    # missing mode on image a offers "hat" (score .25) and defaults to absent
    _, miss = ann.json("GET", f"/api/image/{a_id}?mode=missing")
    assert miss["default_vote"] == -1  # type: ignore[index]
    assert "hat" in {t["name"] for t in miss["tags"]}  # type: ignore[index]

    # adding a tag through autocomplete alias + confirm by second user
    status, added = ann.json("POST", f"/api/image/{a_id}/tags", {"name": "Miku"})
    assert status == 200 and added["tag"]["name"] == "hatsune_miku"  # type: ignore[index]
    assert added["tag"]["status"] == "leaning_yes"  # type: ignore[index]
    miku = added["tag"]["tag_id"]  # type: ignore[index]
    bob.json(
        "POST",
        f"/api/image/{a_id}/votes",
        {"mode": "detail", "votes": [{"tag_id": miku, "vote": 1}]},
    )

    # two rejections of "hat" make it rejected
    for c in (ann, bob):
        c.json(
            "POST",
            f"/api/image/{a_id}/votes",
            {"mode": "missing", "votes": [{"tag_id": hat, "vote": -1}]},
        )

    # autocomplete
    _, res = ann.json("GET", "/api/tags/search?q=long")
    assert res["results"][0]["name"] == "long_hair"  # type: ignore[index]

    # reviewers can't use admin endpoints
    assert ann.json("GET", "/api/admin/contested")[0] == 403
    _, cont = admin.json("GET", "/api/admin/contested")
    items = cont["items"]  # type: ignore[index]
    assert [(i["image_id"], i["name"]) for i in items] == [(a_id, "long_hair")]
    assert {v["name"] for v in items[0]["voters"]} == {"ann", "bob"}
    assert (
        admin.json(
            "POST",
            "/api/admin/resolve",
            {"image_id": a_id, "tag_id": lh, "resolution": 0},
        )[0]
        == 200
    )

    _, browse = ann.json("GET", "/api/browse?filter=contested")
    assert browse["items"] == []  # type: ignore[index]
    _, browse = ann.json("GET", "/api/browse?filter=all&tag=long%20hair")
    assert a_id not in {i["id"] for i in browse["items"]}  # type: ignore[index]

    # export: long_hair removed (resolved), miku added (confirmed), smile kept
    summary = export_captions(review_dir)
    text = (review_dir / "export" / "a.txt").read_text()
    assert text == "1girl, smile, red background, hatsune miku"
    assert summary.tags_added >= 1
    assert (review_dir / "export" / "sub" / "c.txt").read_text() == "water"

    # revoking bob drops his votes from consensus and ends his session
    auth.revoke_user(conn, "bob")
    assert bob.request("GET", "/api/me")[0] == 401


def test_logout(server: tuple[int, Path]):
    port, review_dir = server
    c = _login(port, review_dir, "eve")
    assert c.json("POST", "/api/logout", {})[0] == 200
    assert c.request("GET", "/api/me")[0] == 401


def test_rotate_key_ends_sessions(server: tuple[int, Path]):
    port, review_dir = server
    c = _login(port, review_dir, "fay")
    conn = connect(review_dir)
    auth.rotate_key(conn, "fay")
    assert c.request("GET", "/api/me")[0] == 401


def test_export_to_source_backs_up(review_set: Path, tmp_path: Path):
    export_captions(review_set, to_source=True)
    src = tmp_path / "dataset"
    assert (src / "a.txt.bak").read_text() == "1girl, long hair, smile, red background"
    assert (src / "a.txt").read_text() == "1girl, long hair, smile, red background"
