"""Buffered tile writer with spatial sorting and parallel flush.

Provides :class:`WriterPool` which buffers Arrow Tables per tile in
memory and writes them to GeoParquet files in bounded parallel fashion.
Each tile's rows are optionally sorted (Z-order, Hilbert, or by columns)
before writing, and GeoParquet ``geo`` metadata is updated with per-tile
bounding boxes.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from shapely import from_wkb

from .utils_large import ensure_large_types

logger = logging.getLogger(__name__)


# ------------------------- Sorting configuration -------------------------


@dataclass
class SortKey:
    column: str
    ascending: bool = True


class SortMode:
    NONE = "none"
    ZORDER = "zorder"
    HILBERT = "hilbert"
    COLUMNS = "columns"


@dataclass
class _WriterPoolConfig:
    geom_col: str
    sort_mode: str
    sort_keys: List[SortKey]
    sfc_bits: int
    global_extent: Optional[Tuple[float, float, float, float]]
    compression: Optional[str]
    pq_args: Dict[str, Any]
    outdir: str


# ------------------------- Space-filling helpers -------------------------


def _scale_to_uint(values: np.ndarray, vmin: float, vmax: float, bits: int) -> np.ndarray:
    """Scale float coordinates into [0, 2^bits - 1] integer range."""
    if values.size == 0:
        return np.asarray([], dtype=np.uint64)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros(values.shape[0], dtype=np.uint64)

    max_int = (1 << bits) - 1
    scaled = (values - vmin) / (vmax - vmin)
    scaled = np.clip(scaled, 0.0, 1.0)
    return np.round(scaled * max_int).astype(np.uint64)


def _interleave_bits_2d(x: np.ndarray, y: np.ndarray, bits: int) -> np.ndarray:
    """Compute Morton/Z-order codes from uint coordinates."""
    x = x.astype(np.uint64, copy=False)
    y = y.astype(np.uint64, copy=False)

    def part1by1(n: np.ndarray) -> np.ndarray:
        n &= np.uint64(0x00000000FFFFFFFF)
        n = (n | (n << np.uint64(16))) & np.uint64(0x0000FFFF0000FFFF)
        n = (n | (n << np.uint64(8))) & np.uint64(0x00FF00FF00FF00FF)
        n = (n | (n << np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
        n = (n | (n << np.uint64(2))) & np.uint64(0x3333333333333333)
        n = (n | (n << np.uint64(1))) & np.uint64(0x5555555555555555)
        return n

    return (part1by1(y) << np.uint64(1)) | part1by1(x)


def _maybe_sort_and_bbox(
    tbl: pa.Table,
    geom_col: str,
    sort_mode: str,
    sort_keys: List[SortKey],
    sfc_bits: int,
    global_extent: Optional[Tuple[float, float, float, float]],
) -> Tuple[Tuple[float, float, float, float], pa.Table]:
    """Compute bounding box and optionally sort rows by Z-order or columns."""
    geoms = from_wkb(tbl[geom_col].to_numpy(zero_copy_only=False))

    minx = np.inf
    miny = np.inf
    maxx = -np.inf
    maxy = -np.inf
    has_geom = False

    centers_x: List[float] = []
    centers_y: List[float] = []
    valid_idx: List[int] = []

    for i, g in enumerate(geoms):
        if g is None or g.is_empty:
            continue

        bxmin, bymin, bxmax, bymax = g.bounds
        minx = min(minx, bxmin)
        miny = min(miny, bymin)
        maxx = max(maxx, bxmax)
        maxy = max(maxy, bymax)

        c = g.centroid
        centers_x.append(float(c.x))
        centers_y.append(float(c.y))
        valid_idx.append(i)
        has_geom = True

    if not has_geom:
        bbox = (np.inf, np.inf, -np.inf, -np.inf)
        return bbox, tbl

    bbox = (float(minx), float(miny), float(maxx), float(maxy))

    if sort_mode == SortMode.NONE:
        return bbox, tbl

    if sort_mode == SortMode.COLUMNS:
        if not sort_keys:
            return bbox, tbl
        spec = [
            {
                "column": sk.column,
                "order": "ascending" if sk.ascending else "descending",
            }
            for sk in sort_keys
        ]
        logger.debug("Sorting by columns: %s", spec)
        return bbox, tbl.sort_by(spec)

    if sort_mode in (SortMode.ZORDER, SortMode.HILBERT):
        # Note: current implementation uses Morton/Z-order for both.
        # If you later add a true Hilbert curve encoder, you can branch here.
        cx = np.asarray(centers_x, dtype=np.float64)
        cy = np.asarray(centers_y, dtype=np.float64)
        gxmin, gymin, gxmax, gymax = global_extent or bbox

        X = _scale_to_uint(cx, gxmin, gxmax, sfc_bits)
        Y = _scale_to_uint(cy, gymin, gymax, sfc_bits)
        z = _interleave_bits_2d(X, Y, sfc_bits)

        n_rows = tbl.num_rows
        max_code = np.uint64((1 << (2 * min(sfc_bits, 31))) - 1)
        zfull = np.full(n_rows, max_code, dtype=np.uint64)

        if valid_idx:
            zfull[np.asarray(valid_idx, dtype=np.int64)] = z

        order = np.argsort(zfull, kind="mergesort")
        logger.debug("Sorting %d rows by %s (sfc_bits=%d)", n_rows, sort_mode, sfc_bits)
        return bbox, tbl.take(pa.array(order, type=pa.int64()))

    return bbox, tbl


# ------------------------- GeoParquet metadata -------------------------


def _with_updated_geo_metadata(tbl: pa.Table, bbox: Tuple[float, float, float, float]) -> pa.Table:
    """Update GeoParquet metadata with per-column bbox."""
    schema = tbl.schema
    meta = dict(schema.metadata or {})

    geo_raw = meta.get(b"geo")
    geo: Dict[str, Any] = {}
    if geo_raw is not None:
        try:
            geo = json.loads(geo_raw.decode("utf8"))
        except Exception:
            geo = {}

    col = geo.setdefault("columns", {}).setdefault("geometry", {})
    col["bbox"] = list(map(float, bbox))

    meta[b"geo"] = json.dumps(geo).encode("utf8")
    return tbl.replace_schema_metadata(meta)


# ------------------------- Tile finalization -------------------------


def _finalize_one_tile(
    tile_id: int,
    tables: List[pa.Table],
    config: _WriterPoolConfig,
) -> str:
    """Combine, sort, update metadata, and write one tile file."""
    if not tables:
        raise ValueError(f"Tile {tile_id} received no tables to flush")

    combined = pa.concat_tables(tables, promote_options="default")
    combined = combined.combine_chunks()
    combined = ensure_large_types(combined, config.geom_col)

    bbox, combined = _maybe_sort_and_bbox(
        tbl=combined,
        geom_col=config.geom_col,
        sort_mode=config.sort_mode,
        sort_keys=config.sort_keys,
        sfc_bits=config.sfc_bits,
        global_extent=config.global_extent,
    )
    combined = _with_updated_geo_metadata(combined, bbox)

    # Encode bbox in filename for ParquetIndex spatial lookups
    minx, miny, maxx, maxy = bbox

    def encode_coord(val):
        int_part = int(val)
        dec_part = abs(int((val - int_part) * 1000000))
        return f"{int_part}_{dec_part}"

    bbox_str = "_".join([encode_coord(c) for c in bbox])
    filename = f"tile_{tile_id:06d}__{bbox_str}.parquet"
    out_path = os.path.join(config.outdir, filename)
    write_args = dict(config.pq_args)
    if config.compression is not None:
        write_args.setdefault("compression", config.compression)

    pq.write_table(combined, out_path, **write_args)
    return out_path


# ------------------------- WriterPool -------------------------


class WriterPool:
    """Buffer per-tile Arrow tables and flush them to GeoParquet."""

    def __init__(
        self,
        outdir: str,
        geom_col: str = "geometry",
        sort_mode: str = SortMode.NONE,
        sort_keys: Optional[Sequence[Union[SortKey, Tuple[str, bool], str]]] = None,
        sfc_bits: int = 16,
        global_extent: Optional[Tuple[float, float, float, float]] = None,
        compression: Optional[str] = "zstd",
        max_parallel_files: Optional[int] = None,
        **pq_args: Any,
    ):
        self.outdir = outdir
        self.geom_col = geom_col
        self.sort_mode = sort_mode
        self.sfc_bits = int(sfc_bits)
        self.global_extent = global_extent
        self.compression = compression
        self.max_parallel_files = max(
            1,
            int(max_parallel_files or max(1, multiprocessing.cpu_count() - 1)),
        )
        self._pq_args = dict(pq_args)
        self._buffers: Dict[int, List[pa.Table]] = defaultdict(list)
        self._sort_keys = self._normalize_sort_keys(sort_keys)

    # --------------------------- Public API ---------------------------

    def append(self, tile_id: int, table: pa.Table) -> None:
        logger.debug("--- DEBUG: WriterPool.append ---")
        logger.debug("Incoming metadata: %s", table.schema.metadata if table is not None else None)

        if table is None or table.num_rows == 0:
            return

        if self.geom_col not in table.column_names:
            raise ValueError(f"WriterPool.append: missing geometry column '{self.geom_col}'")

        table = table.combine_chunks()
        table = ensure_large_types(table, self.geom_col)
        self._buffers[tile_id].append(table)

    def flush_all(self) -> None:
        if not self._buffers:
            logger.info("WriterPool.flush_all(): no buffered tiles to flush.")
            return

        os.makedirs(self.outdir, exist_ok=True)
        items = list(self._buffers.items())
        self._buffers.clear()

        total = len(items)
        mpf = min(self.max_parallel_files, total)

        logger.info(
            "WriterPool.flush_all(): %d tiles buffered -> up to %d parallel writes.",
            total,
            mpf,
        )

        config = _WriterPoolConfig(
            geom_col=self.geom_col,
            sort_mode=self.sort_mode,
            sort_keys=list(self._sort_keys),
            sfc_bits=self.sfc_bits,
            global_extent=self.global_extent,
            compression=self.compression,
            pq_args=dict(self._pq_args),
            outdir=self.outdir,
        )

        if total == 1:
            tid, tables = items[0]
            _finalize_one_tile(tid, tables, config)
            logger.info("WriterPool.flush_all(): all tiles successfully flushed to disk.")
            return

        with ProcessPoolExecutor(max_workers=mpf) as ex:
            futures = {
                ex.submit(_finalize_one_tile, tid, tables, config): tid
                for tid, tables in items
            }
            for future in as_completed(futures):
                tid = futures[future]
                try:
                    _ = future.result()
                except Exception as exc:
                    logger.error("Error writing tile %s: %s", tid, exc)
                    raise

        logger.info("WriterPool.flush_all(): all tiles successfully flushed to disk.")

    def close(self) -> None:
        self.flush_all()

    def set_sort_keys(
        self,
        sort_keys: Optional[Sequence[Union[SortKey, Tuple[str, bool], str]]],
    ) -> None:
        self._sort_keys = self._normalize_sort_keys(sort_keys)

    # ------------------------- Internal helpers -----------------------

    @staticmethod
    def _normalize_sort_keys(
        sort_keys: Optional[Sequence[Union[SortKey, Tuple[str, bool], str]]]
    ) -> List[SortKey]:
        out: List[SortKey] = []
        if not sort_keys:
            return out

        for k in sort_keys:
            if isinstance(k, SortKey):
                out.append(k)
            elif isinstance(k, tuple):
                name, asc = k
                out.append(SortKey(str(name), bool(asc)))
            elif isinstance(k, str):
                out.append(SortKey(k, True))
            else:
                raise TypeError(f"Unsupported sort key type: {type(k)!r}")

        return out