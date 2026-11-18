# Dataset Review (collaborative tagging)

Let friends help confirm, reject, and add tags on a training dataset from their
phone or computer. Several people's votes are combined: agreement settles a tag,
disagreement gets flagged for you to decide.

```
dataset/ (images + .txt captions + auto-tag scores)
   │  Publish Dataset for Review  (chaiNNer node, or CLI)
   ▼
review/  images/*.webp (≤1 MP)  thumbs/*.webp  review.sqlite3
   │  python -m dataset_review serve review/
   ▼
https://… ◄── friends' phones / browsers (invite link per person)
   │  python -m dataset_review export review/
   ▼
updated captions (+ export-report.json listing disputed tags)
```

All commands below are run from `backend/src` (or with it on `PYTHONPATH`),
using chaiNNer's Python or any Python 3.10+ with Pillow. The server itself uses
only the standard library.

## 1. Publish a dataset

In chaiNNer: **Utility → Dataset → Publish Dataset for Review**. Or:

```sh
python -m dataset_review publish D:/datasets/mychar D:/review/mychar \
    --tag-db danbooru_e621_merged.csv --categories my_categories.json \
    --title "My character set"
```

- Images are copied as WebP with at most 1 megapixel (`--max-megapixels`).
  The original dataset is never modified by publishing.
- Running it again later syncs the review set. It picks up new or changed
  images, captions and scores, and **never drops votes**. Images removed from
  the dataset are hidden but keep their history.

### What it reads

| File | Meaning |
| --- | --- |
| `name.txt` (or `.caption`) | comma-separated caption tags. Spaces/underscores and `\(` escaping are detected and kept on export. Prose fragments (7+ words) are preserved untouched. |
| `name.scores.json`, `name.tags.json`, `name.png.json`, `name.json` | auto-tagger scores, any of: `{"tag": 0.93}`, `{"scores": {...}}`, `{"general": {...}, "character": {...}}` (group names become categories, `rating` is ignored), `[{"tag": "x", "score": 0.9}]`, `[["x", 0.9]]` |
| `name.scores.csv` | `tag,score` lines |

If an image has scores but no caption, tags scoring ≥ `present_threshold`
(0.35) become its caption.

**Your own score format:** write a small function and pass it with
`--score-loader mymodule:load`. It receives the image path and returns any of
the shapes above (usually `{tag: score}`), or `None`.

### Tag database and categories

`--tag-db` takes the CSV format of the a1111 *tag-autocomplete* extension
(`name,category,post_count,"alias1,alias2"`). Danbooru, e621 and the merged
danbooru+e621 files are all detected automatically, which drives autocomplete,
alias resolution (`miku` → `hatsune_miku`) and categories (artist, copyright,
character, species, general, meta, lore).

`--categories` is your own mapping, which wins over the tag database: JSON
`{"tag": "category"}` or `{"category": ["tag", …]}`, or CSV/TSV `tag,category`.
Any category name works. Tags that remain `unknown` can be assigned later
from the admin page in the app. Either file can be (re)imported later with
`python -m dataset_review vocab REVIEW_DIR --tag-db … --categories …`.

## 2. Invite people

```sh
python -m dataset_review user add D:/review/mychar me --admin
python -m dataset_review user add D:/review/mychar alex
```

Each command prints a personal link like `https://host/#key=…`. **The link is
that person's password**, so send it privately. It works on several devices.

| Command | Effect |
| --- | --- |
| `user list` | names, roles, weights, vote counts |
| `user rotate NAME` | new link; old link and all sessions stop working |
| `user revoke NAME` | lock out; their votes stop counting |
| `user purge-votes NAME` | delete everything they voted |
| `user set NAME --set-weight 2` | trust weight (an admin with weight 2 settles a tag alone) |

## 3. Serve it securely

The server listens on `127.0.0.1:8765` only. Put HTTPS in front of it. In
order of preference:

1. **Tailscale (private, recommended).** Friends install Tailscale and you share
   your machine with them. Then:
   `tailscale serve --bg 8765` and
   `python -m dataset_review serve REVIEW_DIR --public-url https://yourpc.tailXXXX.ts.net --trust-proxy`.
   Only devices you have shared with can reach the server at all, and the
   invite link is a second lock.
2. **Public tunnel, no app install for friends:** `tailscale funnel 8765` or a
   Cloudflare Tunnel (`cloudflared tunnel --url http://127.0.0.1:8765`), with
   the same `--public-url … --trust-proxy`. Anyone can reach the login page;
   only invite links get in.
3. **Your own certificate:** `--host 0.0.0.0 --tls-cert cert.pem --tls-key key.pem`.

Always pass `--public-url` when exposing it. It is used for invite links,
`Secure` cookies and HSTS, and to reject requests for any other host name
(DNS rebinding).

What protects it:

- Each person has their own random 256-bit key, sent in the URL *fragment*,
  so it is never sent to proxies or logs. The page swaps it for an `HttpOnly`,
  `SameSite=Strict` session cookie and removes it from the address bar.
  Only hashes of keys and sessions are stored.
- Every page except the empty app shell needs a session, including images.
- Failed logins are rate limited. Write requests need a custom header and a
  same-origin `Origin`, which blocks CSRF.
- A strict Content-Security-Policy (no inline or third-party scripts) is
  applied, and tag text is only ever inserted as text.
- Files are served by numeric id, never by path. Request bodies are capped,
  and file paths are only shown to admins.

On phones, "Add to Home Screen" gives an app-like full-screen window.

## 4. Reviewing

| Mode | Shows | You tap… | Untouched tags count as |
| --- | --- | --- | --- |
| **Verify** | images that most need review (weighted random), with the ~10 least certain tags that fit on screen | tags that are **wrong** (tap again = "not sure") | correct |
| **Missing** | tags the auto-tagger half-suspects (score 0.12–0.35) but the caption lacks | tags that **are** in the picture | absent |
| **Random** | uniformly random images, Verify-style | wrong tags | correct |
| **Browse** | grid filtered by needs-review / disputed / unvoted / mine / tag; tap for every tag with vote counts | yes → no → clear, per tag | — |

**+ Tag** opens autocomplete over the tag database (aliases included). New tags
are allowed unless `allow_unknown_tags=false`. Desktop keys: `1`–`0` toggle
tags, `Enter` submit, `S` skip, `T` add tag, `Z` zoom.

Admins open the **E**/initial button for progress and a leaderboard, plus:
**Disputed tags** (who voted what; *Keep*/*Remove* settles it) and
**Uncategorised tags**.

## 5. How votes become decisions

Each (image, tag) has a prior: the auto-tag score, or 0.8 for hand-written
caption tags without a score. Votes are weighted per user.

- **confirmed / rejected**: at least `confirm_votes` (2) weighted votes, with the
  minority under `contest_ratio` (25%).
- **contested**: both sides hold at least 25%. It keeps being shown (with ⚑)
  until `max_votes` (7), then waits for an admin.
- **leaning**: a single vote. The next reviewers are steered towards it.
- **resolved**: an admin decided.

What to show next is driven by a per-tag *need*: uncertain scores (near 50%)
and contested tags rank highest, settled ones drop out. An image's need is the
sum over its 12 neediest tags.

Change thresholds with
`python -m dataset_review settings REVIEW_DIR confirm_votes=3 contest_ratio=0.3`.

## 6. Export

```sh
python -m dataset_review export REVIEW_DIR             # → REVIEW_DIR/export/**/*.txt
python -m dataset_review export REVIEW_DIR --to-source # overwrite dataset captions (*.txt.bak backups)
```

Only settled decisions change captions: confirmed/rejected/resolved. Unvoted,
single-vote and contested tags keep their original state (use
`--accept-single-votes` to apply leaning ones too). Original tag order is
kept, and new tags are appended by category. `export-report.json` lists the
still-contested tags.
