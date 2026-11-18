"""Collaborative review of training-dataset tags.

A dataset is *published* into a self-contained review directory (downscaled WebP
copies plus a SQLite database).  A small, dependency-free HTTP server then lets
invited friends confirm, reject, and add tags from a phone or desktop browser.
Votes are combined into per-tag consensus, contested tags are queued for an
admin, and the result can be exported back to caption files.

Entry points:

- ``python -m dataset_review --help`` (run from ``backend/src``)
- :func:`dataset_review.ingest.publish_dataset` (used by the chaiNNer node)
"""

from __future__ import annotations

REVIEW_DB_NAME = "review.sqlite3"
IMAGES_DIR_NAME = "images"
THUMBS_DIR_NAME = "thumbs"
