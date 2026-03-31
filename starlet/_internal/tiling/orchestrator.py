from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Optional
import logging

import pyarrow as pa
import pyarrow.parquet as pq

from starlet._internal.stats import write_attribute_stats, AttributeStatsCollector
from starlet._internal.tiling.datasource import DataSource, GeoParquetSource
from starlet._internal.tiling.writer_pool import WriterPool

logger = logging.getLogger(__name__)

_TILE_COL = "_tile_id"


@dataclass
class _OverflowState:
    path: Path
    writer: Optional[pq.ParquetWriter] = None
    rows_written: int = 0

    def write_table(self, table: pa.Table, compression: str) -> None:
        if table.num_rows == 0:
            return
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                where=str(self.path),
                schema=table.schema,
                compression=compression,
            )
        self.writer.write_table(table)
        self.rows_written += table.num_rows

    def close(self) -> int:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        return self.rows_written


class RoundOrchestrator:
    """Coordinate round-based tiling with bounded open-tile buffering.

    Rows whose tile ids fit within the open-tile budget are buffered into the
    WriterPool. Rows for other tile ids are spilled to an overflow parquet and
    processed in later rounds.
    """

    def __init__(
        self,
        *,
        source: DataSource,
        assigner,
        outdir: str,
        geom_col: str = "geometry",
        records_per_round: int = 250_000,
        max_open_tiles: int = 64,
        max_parallel_files: int = 64,
        compression: str = "zstd",
        sort_mode: str = "zorder",
        sfc_bits: int = 16,
        global_extent=None,
        pq_args: Optional[dict] = None,
    ) -> None:
        self.source = source
        self.assigner = assigner
        self.outdir = outdir
        self.geom_col = geom_col
        self.records_per_round = int(records_per_round)
        self.max_open_tiles = int(max_open_tiles)
        self.max_parallel_files = int(max_parallel_files)
        self.compression = compression
        self.sort_mode = sort_mode
        self.sfc_bits = int(sfc_bits)
        self.global_extent = global_extent
        self._pq_args = dict(pq_args or {})
        self._stats_collector: Optional[AttributeStatsCollector] = None

    def _overflow_path_for_round(self, round_id: int) -> Path:
        return Path(self.outdir) / f"_overflow_round_{round_id}.parquet"

    def _group_by_tile_column(self, table: pa.Table) -> Dict[int, pa.Table]:
        """Group rows by an existing integer tile-id column."""
        tile_values = table[_TILE_COL].to_pylist()
        buckets: Dict[int, list[int]] = {}

        for i, tile_id in enumerate(tile_values):
            if tile_id is None:
                continue
            buckets.setdefault(int(tile_id), []).append(i)

        out: Dict[int, pa.Table] = {}
        for tid, idxs in buckets.items():
            out[tid] = table.take(pa.array(idxs, type=pa.int64()))
        return out

    def _group_by_partition_ids(
        self,
        original_table: pa.Table,
        partition_table,
    ) -> Dict[int, pa.Table]:
        """Group rows by partition ids returned by the assigner."""
        if isinstance(partition_table, pa.Table):
            if partition_table.num_columns != 1:
                raise ValueError("partition_table must have exactly one column")
            pid_values = partition_table.column(0).to_pylist()
        else:
            pid_values = partition_table.to_pylist()

        buckets: Dict[int, list[int]] = {}
        for i, pid in enumerate(pid_values):
            if pid is None:
                continue
            buckets.setdefault(int(pid), []).append(i)

        out: Dict[int, pa.Table] = {}
        for tid, idxs in buckets.items():
            sub = original_table.take(pa.array(idxs, type=pa.int64()))
            if _TILE_COL not in sub.column_names:
                sub = sub.append_column(
                    _TILE_COL,
                    pa.array([tid] * sub.num_rows, type=pa.int64()),
                )
            out[tid] = sub
        return out

    def _run_one_round(
        self,
        ds: DataSource,
        round_id: int,
        records_per_round: int,
    ) -> Optional[Path]:
        pool = WriterPool(
            outdir=self.outdir,
            geom_col=self.geom_col,
            sort_mode=self.sort_mode,
            sfc_bits=self.sfc_bits,
            global_extent=self.global_extent,
            compression=self.compression,
            max_parallel_files=self.max_parallel_files,
            **self._pq_args,
        )

        cap = self.max_open_tiles
        open_tiles: set[int] = set()

        batches = []
        current_rows = 0
        records_limit = max(1, int(records_per_round or self.records_per_round))

        overflow = _OverflowState(path=self._overflow_path_for_round(round_id))

        def process_accumulated(batch_id: int) -> None:
            nonlocal batches, current_rows

            if not batches:
                return

            combined = pa.concat_tables(batches, promote_options="default").combine_chunks()

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
                    overflow.write_table(sub, compression=self.compression)

            batches = []
            current_rows = 0

        batch_idx = -1

        for batch_idx, batch in enumerate(ds.iter_tables()):
            if self._stats_collector is None:
                self._stats_collector = AttributeStatsCollector(
                    batch.schema,
                    geometry_column=self.geom_col,
                )

            self._stats_collector.consume_table(batch)
            batches.append(batch)
            current_rows += batch.num_rows

            if current_rows >= records_limit:
                process_accumulated(batch_idx)

        process_accumulated(batch_idx)
        pool.flush_all()

        overflow_rows = overflow.close()
        if overflow_rows > 0:
            return overflow.path

        if overflow.path.exists():
            try:
                pf = pq.ParquetFile(str(overflow.path))
                if pf.metadata.num_rows == 0:
                    overflow.path.unlink(missing_ok=True)
            except Exception:
                logger.debug(
                    "Could not inspect overflow parquet for cleanup: %s",
                    overflow.path,
                    exc_info=True,
                )

        return None

    def run(self) -> None:
        round_id = 0
        ds: DataSource = self.source

        while True:
            t0 = perf_counter()

            overflow_path = self._run_one_round(
                ds,
                round_id,
                records_per_round=self.records_per_round,
            )

            logger.info("Round %d finished in %.2fs", round_id, perf_counter() - t0)

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
                logger.debug("Failed cleaning overflow file %s", p, exc_info=True)

        if self._stats_collector is not None:
            try:
                stats = self._stats_collector.finalize()
                dataset_root = Path(self.outdir).parent
                write_attribute_stats(dataset_root, stats)
                logger.info(
                    "Attribute statistics written to %s/stats/attributes.json",
                    dataset_root,
                )
            except Exception:
                logger.exception("Failed to finalize/write attribute statistics")