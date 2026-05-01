from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
from shapely import from_wkb
from shapely.geometry import (
    Point, LineString, LinearRing, Polygon,
    MultiPoint, MultiLineString, MultiPolygon, GeometryCollection
)
from pyproj import Transformer

logger = logging.getLogger(__name__)

# Global Web Mercator extent
LIM = 20037508.342789244
GLOBAL_BBOX = (-LIM, -LIM, LIM, LIM)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

@dataclass
class HistConfig:
    grid_size: int = 4096
    out_crs: str = "EPSG:3857"
    dtype: str = "float64"
    max_parallel_tiles: int = 8
    rg_parallel: int = 4


# ---------------------------------------------------------------------------
# GEOMETRY HELPERS
# ---------------------------------------------------------------------------

def _geometry_vertices_iter(g):
    if g is None or g.is_empty:
        return

    if isinstance(g, Point):
        yield (g.x, g.y)

    elif isinstance(g, (LineString, LinearRing)):
        for x, y in g.coords:
            yield (x, y)

    elif isinstance(g, Polygon):
        for x, y in g.exterior.coords:
            yield (x, y)
        for ring in g.interiors:
            for x, y in ring.coords:
                yield (x, y)

    elif isinstance(g, (MultiPoint, MultiLineString, MultiPolygon, GeometryCollection)):
        for sub in g.geoms:
            yield from _geometry_vertices_iter(sub)

    else:
        coords = getattr(g, "coords", None)
        if coords is not None:
            for x, y in coords:
                yield (x, y)


# ---------------------------------------------------------------------------
# PER TILE HISTOGRAM
# ---------------------------------------------------------------------------

def _accumulate_vertices_hist(
    table,
    geom_col: str,
    bbox_out,
    transformer,
    n: int,
    dtype
):
    hist = np.zeros((n, n), dtype=dtype)
    geoms = from_wkb(table[geom_col].to_numpy(zero_copy_only=False))

    minx, miny, maxx, maxy = bbox_out
    inv_w = 1.0 / (maxx - minx)
    inv_h = 1.0 / (maxy - miny)

    all_xs: list[float] = []
    all_ys: list[float] = []

    for g in geoms:
        if g is None or g.is_empty:
            continue

        coords = list(_geometry_vertices_iter(g))
        if not coords:
            continue
        
        xs, ys = zip(*coords)
        all_xs.extend(xs)
        all_ys.extend(ys)

    if not all_xs:
        return hist
    
    X, Y = transformer.transform(all_xs, all_ys)

    X = np.asarray(X)
    Y = np.asarray(Y)

    tx = (X - minx) * inv_w
    ty = (Y - miny) * inv_h
    
    ix = np.floor(tx * n).astype(np.int64, copy=False)
    iy = np.floor(ty * n).astype(np.int64, copy=False)
    
    np.clip(ix, 0, n - 1, out=ix)
    np.clip(iy, 0, n - 1, out=iy)

    iy = n - 1 - iy

    np.add.at(hist, (iy, ix), 1)

    return hist


# ---------------------------------------------------------------------------
# PROCESS ONE TILE
# ---------------------------------------------------------------------------

def _process_one_tile(parquet_path: Path, cfg, geom_col: str) -> np.ndarray:

    tile_id = parquet_path.stem
    logger.info(f"Processing tile: {tile_id}")

    pf = pq.ParquetFile(str(parquet_path))

    dtype = np.dtype(cfg.dtype)
    transformer = Transformer.from_crs("EPSG:4326", cfg.out_crs, always_xy=True)
    bbox = GLOBAL_BBOX

    base = np.zeros((cfg.grid_size, cfg.grid_size), dtype=dtype)

    # Avoid per-tile thread pools: we already parallelize across tiles, and
    # row-group level fan-out adds overhead and thread contention. Process each
    # row group directly and accumulate into a single histogram buffer.
    for rg in range(pf.metadata.num_row_groups):
        hist = _accumulate_vertices_hist(
            pf.read_row_group(rg, columns=[geom_col]).combine_chunks(),
            geom_col,
            bbox,
            transformer,
            cfg.grid_size,
            dtype,
        )
        base += hist

    return base


# ---------------------------------------------------------------------------
# GLOBAL SUM AND PREFIX SUM
# ---------------------------------------------------------------------------

def _sum_all_tiles(tile_hists: List[np.ndarray], outdir: Path, dtype="float64") -> Path:

    if not tile_hists:
        raise RuntimeError("No tile histograms generated")

    example = tile_hists[0]
    total = np.zeros_like(example, dtype=dtype)

    for hist in tile_hists:
        if hist.shape != total.shape:
            raise ValueError(
                f"Tile histogram has shape {hist.shape}, expected {total.shape}"
            )
        total += hist

    global_path = outdir / "global.npy"
    np.save(global_path, total, allow_pickle=False)

    global_json = {
        "filename": "global.npy",
        "dtype": str(total.dtype),
        "grid_size": int(total.shape[0]),
        "shape": list(total.shape),
        "crs": "EPSG:3857",
        "bbox": list(GLOBAL_BBOX),
        "sum": float(total.sum()),
        "nonzero": int(np.count_nonzero(total)),
    }
    with open(outdir / "global.json", "w") as f:
        json.dump(global_json, f, indent=2)

    prefix = total.cumsum(axis=0).cumsum(axis=1)

    prefix_path = outdir / "global_prefix.npy"
    np.save(prefix_path, prefix, allow_pickle=False)

    prefix_json = {
        "filename": "global_prefix.npy",
        "dtype": str(prefix.dtype),
        "grid_size": int(prefix.shape[0]),
        "shape": list(prefix.shape),
        "crs": "EPSG:3857",
        "bbox": list(GLOBAL_BBOX),
        "desc": "2D prefix sum histogram"
    }
    with open(outdir / "global_prefix.json", "w") as f:
        json.dump(prefix_json, f, indent=2)

    logger.info(f"Wrote global histogram: {global_path}")
    logger.info(f"Wrote global prefix sum histogram: {prefix_path}")

    return prefix_path


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def build_histograms_for_dir(
    tiles_dir: str,
    outdir: str,
    geom_col="geometry",
    grid_size=4096,
    dtype="float64",
    hist_max_parallel=8,
    hist_rg_parallel=4,
):
    cfg = HistConfig(
        grid_size=grid_size,
        dtype=dtype,
        max_parallel_tiles=hist_max_parallel,
        rg_parallel=hist_rg_parallel,
    )

    tiles = sorted(Path(tiles_dir).rglob("*.parquet"))
    if not tiles:
        logger.error("No parquet tiles found")
        return

    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    tile_outputs = []

    with ProcessPoolExecutor(max_workers=cfg.max_parallel_tiles) as ex:
        futures = {
            ex.submit(_process_one_tile, p, cfg, geom_col): p
            for p in tiles
        }

        for f in as_completed(futures):
            tile_outputs.append(f.result())

    _sum_all_tiles(tile_outputs, outdir_p, dtype=dtype)
