"""Web Mercator math and 2D prefix-sum histogram queries for MVT generation."""
from __future__ import annotations

import math
import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

WORLD_MINX = -20037508.342789244
WORLD_MINY = -20037508.342789244
WORLD_MAXX = 20037508.342789244
WORLD_MAXY = 20037508.342789244
WORLD_W = WORLD_MAXX - WORLD_MINX
WORLD_H = WORLD_MAXY - WORLD_MINY

EXTENT = 4096


def hist_zoom_from_prefix(prefix: np.ndarray) -> int:
    _, w = prefix.shape
    return int(round(math.log2(w)))


def hist_value_from_prefix(
    prefix: np.ndarray,
    z: int,
    x: int,
    y: int,
    hist_zoom: int | None = None,
) -> float:
    H, W = prefix.shape
    if hist_zoom is None:
        hist_zoom = hist_zoom_from_prefix(prefix)

    if z == hist_zoom:
        if 0 <= y < H and 0 <= x < W:
            A = prefix[y, x]
            B = prefix[y - 1, x] if y > 0 else 0
            C = prefix[y, x - 1] if x > 0 else 0
            D = prefix[y - 1, x - 1] if (y > 0 and x > 0) else 0
            return A - B - C + D
        return 0.0

    if z < hist_zoom:
        scale = 2 ** (hist_zoom - z)
        j0 = max(0, min(x * scale, W - 1))
        j1 = max(0, min((x + 1) * scale - 1, W - 1))
        i0 = max(0, min(y * scale, H - 1))
        i1 = max(0, min((y + 1) * scale - 1, H - 1))

        A = prefix[i1, j1]
        B = prefix[i0 - 1, j1] if i0 > 0 else 0
        C = prefix[i1, j0 - 1] if j0 > 0 else 0
        D = prefix[i0 - 1, j0 - 1] if (i0 > 0 and j0 > 0) else 0
        return A - B - C + D

    scale = 2 ** (z - hist_zoom)
    parent_x = x // scale
    parent_y = y // scale

    if not (0 <= parent_x < W and 0 <= parent_y < H):
        return 0.0

    A = prefix[parent_y, parent_x]
    B = prefix[parent_y - 1, parent_x] if parent_y > 0 else 0
    C = prefix[parent_y, parent_x - 1] if parent_x > 0 else 0
    D = prefix[parent_y - 1, parent_x - 1] if (parent_y > 0 and parent_x > 0) else 0

    return (A - B - C + D) / (scale * scale)


def mercator_tile_bounds(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
    n = 2 ** z
    tile_w = WORLD_W / n

    minx = WORLD_MINX + x * tile_w
    maxx = minx + tile_w
    maxy = WORLD_MAXY - y * tile_w
    miny = maxy - tile_w

    return minx, miny, maxx, maxy


def mercator_bounds_to_tile_range(z: int, minx: float, miny: float, maxx: float, maxy: float) -> Tuple[int, int, int, int]:
    n = 2 ** z
    tile_w = WORLD_W / n

    tx0 = max(int((minx - WORLD_MINX) // tile_w), 0)
    tx1 = min(int((maxx - WORLD_MINX) // tile_w), n - 1)
    ty0 = max(int((WORLD_MAXY - maxy) // tile_w), 0)
    ty1 = min(int((WORLD_MAXY - miny) // tile_w), n - 1)

    return tx0, ty0, tx1, ty1


def explode_geom(g) -> List:
    if g.is_empty:
        return []
    if g.geom_type != "GeometryCollection":
        return [g]
    out = []
    for part in g.geoms:
        out.extend(explode_geom(part))
    return out