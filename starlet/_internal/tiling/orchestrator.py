"""Round-based tiling orchestrator.

Coordinates the end-to-end tiling pipeline: reads batches from a
:class:`DataSource`, assigns rows to spatial partitions via an assigner,
buffers them in a :class:`WriterPool`, and handles overflow rows that
cannot be written in the current round by spilling to a temporary
Parquet file and re-processing in subsequent rounds.

Attribute statistics are collected in a single pass and written once
after all rounds complete.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Set, Dict, List
import logging

import pyarrow as pa
import pyarrow.parquet as pq
from time import perf_counter

from .datasource import DataSource, GeoParquetSource
from .assigner import TileAssignerFromCSV
from .writer_pool import WriterPool
from .utils_large import ensure_large_types

from starlet._internal.stats.collector import AttributeStatsCollector
from starlet._internal.stats.writer import write_attribute_stats


logger = logging.getLogger(__name__)

_TILE_COL = "geo_parquet_tile_num"


class _OverflowWriter:
    """
    Streams rows that don't fit in this round into a single overflow Parquet file.
    """
    def __init__(self, path: Path, compression: str, geom_col: str):
        self.path = Path(path)
        self.compression = compression
        self.geom_col = geom_col

        self._pw: Optional[pq.ParquetWriter] = None
        self._rows = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def close(self) -> int:
        if self._pw is not None:
            self._pw.close()
        return self._rows

    def write_batch(self, tbl: pa.Table, tile_id: Optional[int] = None) -> None:
        if tbl is None or tbl.num_rows == 0:
            return

        tbl = tbl.combine_chunks()
        tbl = ensure_large_types(tbl, geom_col=self.geom_col)

        if tile_id is not None:
            try:
                tid_num = int(tile_id)
            except Exception:
                tid_num = -1
            if _TILE_COL not in tbl.column_names:
                col = pa.array([tid_num] * tbl.num_rows, type=pa.int32())
                tbl = tbl.append_column(_TILE_COL, col)

        if self._pw is None:
            self._pw = pq.ParquetWriter(
                str(self.path),
                schema=tbl.schema,
                compression=self.compression
            )

        self._pw.write_table(tbl)
        self._rows += tbl.num_rows


class RoundOrchestrator:
    """Bounded-memory tiling engine that writes GeoParquet tiles in rounds.

    Each round reads up to *records_per_round* rows, assigns them to spatial
    partitions, and buffers the results in a :class:`WriterPool`.  Rows that
    map to tiles already at the ``max_parallel_files`` limit are spilled to an
    overflow Parquet file and re-processed in the next round.  This keeps the
    number of open file handles bounded while still handling arbitrarily large
    input.

    Attribute statistics (min, max, distinct counts, etc.) are collected in a
    single streaming pass and written to ``<outdir>/../stats/attributes.json``
    after all rounds complete.
    """

    def __init__(
        self,
        source: DataSource,
        assigner: TileAssignerFromCSV,
        outdir: str,
        max_parallel_files: int,
        compression: str = "zstd",
        geom_col: str = "geometry",
        sort_mode: str = "zorder",
        sort_keys: Optional[str] = None,
        sfc_bits: int = 16,
        records_per_round: int = 1000_000,
    ):
        self.source = source
        self.assigner = assigner
        self.outdir = outdir
        self.max_parallel_files = int(max_parallel_files)
        self.compression = compression
        self.geom_col = geom_col
        self.sort_mode = sort_mode
        self.sort_keys = sort_keys
        self.sfc_bits = int(sfc_bits)
        self.src_schema: pa.Schema = source.schema()
        self.records_per_round = int(records_per_round)

        self._stats_collector: Optional[AttributeStatsCollector] = None

        # Extract global MBR from RSGroveAssigner if available (to avoid redundant computation)
        self._global_mbr = None
        if hasattr(assigner, '_env'):
            env = assigner._env
            self._global_mbr = (
                float(env.mins[0]), float(env.mins[1]),
                float(env.maxs[0]), float(env.maxs[1])
            )
            logger.debug(f"Extracted global MBR from assigner: {self._global_mbr}")

    # ------------------------------------------------------------------

    @staticmethod
    def _tile_label(tile_id: int) -> str:
        return f"tile_{tile_id:06d}"

    def _group_by_tile_column(self, batch: pa.Table) -> Dict[int, pa.Table]:
        col = batch[_TILE_COL]
        arr = col.to_numpy(zero_copy_only=False)
        groups: Dict[int, List[int]] = {}
        for i, v in enumerate(arr):
            if v is None:
                continue
            try:
                v_int = int(v)
            except Exception:
                continue
            groups.setdefault(v_int, []).append(i)

        out: Dict[int, pa.Table] = {}
        for tid, idxs in groups.items():
            out[tid] = batch.take(pa.array(idxs, type=pa.int32()))
        return out

    def _group_by_partition_ids(self, batch: pa.Table, partitions: pa.Table) -> Dict[int, pa.Table]:
        if partitions.num_rows != batch.num_rows:
            raise ValueError("Partition table must align with batch row count")
        if "partition_id" not in partitions.column_names:
            raise ValueError("Partition table missing 'partition_id' column")

        arr = partitions["partition_id"].to_numpy(zero_copy_only=False)
        groups: Dict[int, List[int]] = {}
        for i, v in enumerate(arr):
            if v is None:
                continue
            try:
                pid = int(v)
            except Exception:
                continue
            groups.setdefault(pid, []).append(i)

        out: Dict[int, pa.Table] = {}
        for pid, idxs in groups.items():
            out[pid] = batch.take(pa.array(idxs, type=pa.int32()))
        return out

    # ------------------------------------------------------------------

    def _run_one_round(
        self,
        ds: DataSource,
        round_id: int,
        records_per_round: Optional[int] = None,
    ) -> Optional[Path]:

        pool = WriterPool(
            outdir=self.outdir,
            compression=self.compression,
            geom_col=self.geom_col,
            max_parallel_files=self.max_parallel_files,
            sort_mode=self.sort_mode,
            sort_keys=self.sort_keys,
            sfc_bits=self.sfc_bits,
        )

        overflow_path = Path(self.outdir) / f"_overflow_round_{round_id}.parquet"
        ow = _OverflowWriter(
            path=overflow_path,
            compression=self.compression,
            geom_col=self.geom_col,
        )

        open_tiles: Set[int] = set()
        cap = max(1, self.max_parallel_files - 1)
        batches: List[pa.Table] = []
        current_rows = 0
        records_limit = max(1, int(records_per_round or self.records_per_round))

        def process_accumulated(batch_id: int) -> None:
            nonlocal batches, current_rows
            if not batches:
                return

            combined = pa.concat_tables(batches, promote=True).combine_chunks()

            if _TILE_COL in combined.column_names:
                parts = self._group_by_tile_column(combined)
            else:
                partition_table = self.assigner.partition_by_tile(combined)
                parts = self._group_by_partition_ids(combined, partition_table)

            for tile_id, sub in parts.items():
                if tile_id in open_tiles or len(open_tiles) < cap:
                    open_tiles.add(tile_id)
                    pool.append(tile_id, sub)
                else:
                    ow.write_batch(sub, tile_id=tile_id)

            batches = []
            current_rows = 0

        for batch_idx, batch in enumerate(ds.iter_tables()):
            if self._stats_collector is None:
                self._stats_collector = AttributeStatsCollector(
                    batch.schema,
                    geometry_column=self.geom_col,
                    global_mbr=self._global_mbr,  # Pass pre-computed MBR to skip redundant computation
                )

            self._stats_collector.consume_table(batch)

            batches.append(batch)
            current_rows += batch.num_rows
            if current_rows >= records_limit:
                process_accumulated(batch_idx)

        process_accumulated(batch_idx)
        pool.flush_all()

        overflow_rows = ow.close()
        if overflow_rows > 0:
            return overflow_path

        if overflow_path.exists():
            try:
                pf = pq.ParquetFile(str(overflow_path))
                if pf.metadata.num_rows == 0:
                    overflow_path.unlink(missing_ok=True)
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------

    def run(self) -> None:
        round_id = 0
        ds: DataSource = self.source

        while True:
            overflow_path = self._run_one_round(
                ds,
                round_id,
                records_per_round=self.records_per_round
            )
            if overflow_path is None:
                break

            ds = GeoParquetSource(str(overflow_path))
            round_id += 1

        for p in Path(self.outdir).glob("_overflow_round_*.parquet"):
            try:
                pf = pq.ParquetFile(str(p))
                if pf.metadata.num_rows == 0:
                    p.unlink(missing_ok=True)
            except Exception:
                pass

        # ------------------------------------------------------------
        # Finalize and write attribute statistics (ONCE)
        # ------------------------------------------------------------
        if self._stats_collector is not None:
            try:
                stats = self._stats_collector.finalize()
                dataset_root = Path(self.outdir).parent
                write_attribute_stats(dataset_root, stats)
                logger.info("Attribute statistics written to %s/stats/attributes.json", dataset_root)
            except Exception:
                logger.exception("Failed to finalize/write attribute statistics")
