"""
Benchmark the Make Seamless node on full PBR stacks from Poly Haven (CC0).

For each texture: download the 1K albedo, normal (OpenGL), height and roughness maps, crop an
off-centre window (so the maps no longer tile), run the Make Seamless node on every map with the
height map as the shared guide, and render the result tiled 2x2 with simple lighting. Lighting
through the normal map shows seams and any disagreement between the maps.

Usage (from backend/src, with chaiNNer's backend dependencies installed):
    python ../../docs/research/detile-texture/prototype/benchmark_polyhaven.py OUT_DIR [ID ...]

Downloads are cached in OUT_DIR/cache. Results: OUT_DIR/sheet_*.jpg and OUT_DIR/results.csv.
"""

from __future__ import annotations

import csv
import importlib
import json
import os
import re
import sys
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, os.getcwd())

import api  # noqa: E402

importlib.import_module("packages.chaiNNer_standard")
api.registry.load_nodes(os.path.abspath("server.py"))

from nodes.impl.seamless import MapType, SeamlessMethod, seam_ratio  # noqa: E402
from nodes.utils.seed import Seed  # noqa: E402
from packages.chaiNNer_standard.material_textures.tiling.make_seamless import (  # noqa: E402
    make_seamless_node,
)

DEFAULT_IDS = [
    # repeating patterns
    "red_brick",
    "red_brick_03",
    "medieval_blocks_03",
    "concrete_brick_wall_001",
    "interior_tiles",
    "worn_tile_floor",
    "stone_embedded_tiles",
    "metal_plate",
    "corrugated_iron",
    "factory_wall",
    "roof_09",
    "fabric_pattern_07",
    "book_pattern",
    "laminate_floor_02",
    "plywood",
    "wood_table_001",
    # semi-regular
    "cobblestone_floor_04",
    "cobblestone_floor_08",
    "grassy_cobblestone",
    # organic / stochastic
    "aerial_rocks_02",
    "aerial_grass_rock",
    "coast_sand_rocks_02",
    "rock_pitted_mossy",
    "rocky_trail_02",
    "brown_mud_leaves_01",
    "sand_01",
    "gravelly_sand",
    "sandy_gravel_02",
    "aerial_beach_01",
    "snow_01",
    "bark_brown_01",
    "bark_brown_02",
    "bark_willow",
    "grey_plaster",
    "beige_wall_001",
    "cracked_concrete_wall",
    "concrete_floor_02",
    "concrete_floor_worn_001",
    "asphalt_02",
    "aerial_asphalt_01",
    "rust_coarse_01",
    "green_metal_rust",
    "brown_leather",
    "leather_red_02",
]
MAPS = {
    "albedo": ("Diffuse", "col_1", "col_2", "col_01"),
    "normal": ("nor_gl",),
    "height": ("Displacement",),
    "rough": ("Rough",),
}
CROP = (151, 97, 640, 768)  # y, x, h, w: off-centre, not a whole number of repeats

KW = {
    "transition_width": 50, "transition_softness": 15, "mask_warp": 8, "warp_detail": 6,
    "content_influence": 0, "search_width": 80, "seam_feather": 2, "smooth_blend": 3,
    "splat_count": 24, "splat_size": 16, "size_variation": 30, "rotation_variation": 0,
    "splat_softness": 35, "splat_warp": 25, "equalize": False, "equalize_radius": 15,
    "preserve_contrast": False,
}  # fmt: skip


USER_AGENT = "chaiNNer-seamless-benchmark/1.0"


ALLOWED_PREFIXES = ("https://api.polyhaven.com/", "https://dl.polyhaven.org/")


def download(url: str, path: str) -> None:
    if not url.startswith(ALLOWED_PREFIXES):
        raise ValueError(f"Refusing to download from an unexpected location: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response, open(path, "wb") as f:
        f.write(response.read())


def fetch(asset: str, cache: str) -> dict[str, np.ndarray]:
    if not re.fullmatch(r"[a-z0-9_]+", asset):
        raise ValueError(f"Not a Poly Haven asset id: {asset!r}")
    folder = os.path.join(cache, asset)
    os.makedirs(folder, exist_ok=True)
    index_path = os.path.join(folder, "files.json")
    if not os.path.exists(index_path):
        download(f"https://api.polyhaven.com/files/{asset}", index_path)
    with open(index_path) as f:
        files = json.load(f)
    maps: dict[str, np.ndarray] = {}
    for name, keys in MAPS.items():
        for key in keys:
            url = files.get(key, {}).get("1k", {}).get("jpg", {}).get("url")
            if not url:
                continue
            path = os.path.join(folder, f"{name}.jpg")
            if not os.path.exists(path):
                download(url, path)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                y, x, h, w = CROP
                maps[name] = img[y : y + h, x : x + w].astype(np.float32) / 255
            break
    return maps


def lit(albedo: np.ndarray, normal: np.ndarray | None, rough: np.ndarray | None):
    """Simple Lambert + Blinn-Phong shading of a flat plane, light from the top left."""
    if normal is None:
        return albedo
    n = normal[:, :, ::-1] * 2 - 1  # BGR -> XYZ
    n[:, :, 1] *= -1  # OpenGL green points up, image rows point down
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-6
    light = np.array([-0.5, -0.5, 0.7], np.float32)
    light /= np.linalg.norm(light)
    half = light + np.array([0, 0, 1], np.float32)
    half /= np.linalg.norm(half)
    ndl = np.clip(n @ light, 0, 1)[:, :, np.newaxis]
    r = rough[:, :, :1] if rough is not None else np.full_like(ndl, 0.6)
    spec = (
        np.clip(n @ half, 0, 1)[:, :, np.newaxis] ** (4 + 60 * (1 - r)) * (1 - r) * 0.5
    )
    return np.clip(albedo * (0.2 + 0.8 * ndl) + spec, 0, 1)


def tile2(img: np.ndarray, width: int = 420) -> np.ndarray:
    t = np.tile(img, (2, 2, 1))
    size = (width, round(width * t.shape[0] / t.shape[1]))
    return cv2.resize((t * 255).astype(np.uint8), size, interpolation=cv2.INTER_AREA)


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)  # fmt: skip
    return img


def main():
    out_dir = sys.argv[1]
    ids = sys.argv[2:] or DEFAULT_IDS
    cache = os.path.join(out_dir, "cache")
    os.makedirs(cache, exist_ok=True)
    rows, table = [], []
    for asset in ids:
        maps = fetch(asset, cache)
        if "albedo" not in maps:
            print(f"{asset}: no albedo map, skipped")
            continue
        guide = maps.get("height")
        results, info = {}, ""
        for name, img in maps.items():
            map_type = MapType.NORMAL_GL if name == "normal" else MapType.COLOR
            out, info_text = make_seamless_node(
                img,
                guide,
                map_type,
                SeamlessMethod.MIN_ERROR_SEAM,
                Seed(1),
                True,
                0,
                **KW,
            )
            results[name] = out if out.ndim == 3 else out[:, :, np.newaxis]
            if name == "albedo":
                info = info_text.replace("\n", " | ")
        before = lit(maps["albedo"], maps.get("normal"), maps.get("rough"))
        after = lit(results["albedo"], results.get("normal"), results.get("rough"))
        ratios = {k: seam_ratio(v) for k, v in results.items()}
        table.append({"asset": asset, **{f"seam_{k}": f"{v:.2f}" for k, v in ratios.items()}, "info": info})  # fmt: skip
        print(
            f"{asset:26s} "
            + " ".join(f"{k}={v:.2f}" for k, v in ratios.items())
            + f"  {info}"
        )
        rows.append(np.hstack([label(tile2(before), f"{asset}: input (lit, 2x2)"), label(tile2(after), "Make Seamless (lit, 2x2)")]))  # fmt: skip

    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in table for k in r}, key=lambda k: (k != "asset", k)))  # fmt: skip
        writer.writeheader()
        writer.writerows(table)
    per_sheet = 8
    for i in range(0, len(rows), per_sheet):
        sheet = np.vstack(rows[i : i + per_sheet])
        cv2.imwrite(os.path.join(out_dir, f"sheet_{i // per_sheet + 1}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])  # fmt: skip


if __name__ == "__main__":
    main()
