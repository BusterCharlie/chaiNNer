"""
Downloads 1K albedo, normal (OpenGL), height, roughness and AO maps of Poly Haven (CC0) textures
into OUT_DIR/<asset>/{albedo,normal,height,rough,ao}.jpg, the layout evaluate.py expects.

Usage: python fetch_polyhaven.py OUT_DIR asset_id [asset_id ...]
"""

import json
import os
import re
import sys
import urllib.request

USER_AGENT = "chaiNNer-research/1.0"
ALLOWED = ("https://api.polyhaven.com/", "https://dl.polyhaven.org/")
MAPS = {"albedo": "Diffuse", "normal": "nor_gl", "height": "Displacement", "rough": "Rough", "ao": "AO"}


def download(url: str) -> bytes:
    if not url.startswith(ALLOWED):
        raise ValueError(f"Refusing to download from {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main():
    out = sys.argv[1]
    for asset in sys.argv[2:]:
        if not re.fullmatch(r"[a-z0-9_]+", asset):
            raise ValueError(f"Not a Poly Haven asset id: {asset!r}")
        files = json.loads(download(f"https://api.polyhaven.com/files/{asset}"))
        os.makedirs(os.path.join(out, asset), exist_ok=True)
        for name, key in MAPS.items():
            url = files.get(key, {}).get("1k", {}).get("jpg", {}).get("url")
            if url:
                with open(os.path.join(out, asset, f"{name}.jpg"), "wb") as f:
                    f.write(download(url))
        print("fetched", asset)


if __name__ == "__main__":
    main()
