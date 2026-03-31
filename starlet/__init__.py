"""starlet — spatial tiling, MVT generation, and tile serving for geospatial data."""
from __future__ import annotations

__version__ = "0.1.0"

from starlet._types import TileResult, MVTResult, Dataset

__all__ = [
    "tile",
    "generate_mvt",
    "build",
    "create_app",
    "TileResult",
    "MVTResult",
    "Dataset",
]


def tile(
    input: str,
    outdir: str,
    *,
    num_tiles: int = 40,
    partition_size: int = 1 << 30,
    sort: str = "zorder",
    compression: str = "zstd",
    sample_cap: int | None = 10_000,
    sample_ratio: float = 1.0,
    seed: int = 42,
    geom_col: str = "geometry",
    sfc_bits: int = 16,
    max_parallel_files: int = 64,
    index: str | None = None,
) -> TileResult:
    """Partition a GeoParquet/GeoJSON dataset into spatially-tiled Parquet files."""
    import logging
    import math
    from pathlib import Path

    from starlet._internal.tiling.datasource import (
        GeoParquetSource,
        GeoJSONSource,
        is_geojson_path,
    )
    from starlet._internal.tiling.assigner import (
        TileAssignerFromCSV,
        RSGroveAssigner,
    )
    from starlet._internal.tiling.orchestrator import RoundOrchestrator
    from starlet._internal.tiling.writer_pool import SortMode
    from starlet._internal.histogram.hist_pyramid import build_histograms_for_dir

    logger = logging.getLogger("starlet.tile")

    # -------------------- Sort mode --------------------
    _sort_map = {
        "none": SortMode.NONE,
        "columns": SortMode.COLUMNS,
        "zorder": SortMode.ZORDER,
        "hilbert": SortMode.HILBERT,
    }
    sort_mode = _sort_map.get(sort.strip().lower(), SortMode.ZORDER)

    # -------------------- Data source --------------------
    if is_geojson_path(input):
        source = GeoJSONSource(input)
    else:
        source = GeoParquetSource(input)

    # -------------------- Partition count --------------------
    input_size_bytes = Path(input).stat().st_size
    computed = max(1, math.ceil(input_size_bytes / partition_size))
    target_partitions = num_tiles if num_tiles else computed

    logger.info(
        "Target partitions: %d (input=%d bytes)",
        target_partitions,
        input_size_bytes,
    )

    # -------------------- Assigner --------------------
    if index:
        assigner = TileAssignerFromCSV(index, geom_col=geom_col)
    else:
        assigner = RSGroveAssigner.from_source(
            tables=source.iter_tables(),
            num_partitions=target_partitions,
            geom_col=geom_col,
            seed=seed,
            sample_ratio=sample_ratio,
            sample_cap=sample_cap,
        )

    tiles_dir = str(Path(outdir) / "parquet_tiles")
    hist_dir = str(Path(outdir) / "histograms")

    # -------------------- Orchestration --------------------
    orchestrator = RoundOrchestrator(
        source=source,
        assigner=assigner,
        outdir=tiles_dir,
        max_parallel_files=max_parallel_files,
        compression=compression,
        sort_mode=sort_mode,
        sfc_bits=sfc_bits,
    )
    orchestrator.run()

    # -------------------- Histograms --------------------
    logger.info("Tiling complete. Building histograms.")
    build_histograms_for_dir(
        tiles_dir=tiles_dir,
        outdir=hist_dir,
        geom_col=geom_col,
        grid_size=4096,
        dtype="float64",
        hist_max_parallel=8,
        hist_rg_parallel=4,
    )

    # -------------------- Metadata --------------------
    tile_files = list(Path(tiles_dir).glob("*.parquet"))
    total_rows = 0

    for tf in tile_files:
        import pyarrow.parquet as pq
        meta = pq.read_metadata(str(tf))
        total_rows += meta.num_rows

    ds = Dataset(outdir)
    result_bbox = ds.bbox or (0.0, 0.0, 0.0, 0.0)

    return TileResult(
        outdir=outdir,
        num_files=len(tile_files),
        total_rows=total_rows,
        bbox=result_bbox,
        histogram_path=str(Path(hist_dir) / "global_prefix.npy"),
    )


def generate_mvt(
    tile_dir: str,
    *,
    zoom: int = 7,
    threshold: float = 0,
    outdir: str | None = None,
) -> MVTResult:
    """Generate Mapbox Vector Tiles from a tiled dataset."""
    from pathlib import Path
    from starlet._internal.mvt.generator import BucketMVTGenerator

    parquet_dir = str(Path(tile_dir) / "parquet_tiles")
    hist_path = str(Path(tile_dir) / "histograms" / "global.npy")
    mvt_outdir = outdir or str(Path(tile_dir) / "mvt")

    gen = BucketMVTGenerator(
        parquet_dir=parquet_dir,
        hist_path=hist_path,
        outdir=mvt_outdir,
        last_zoom=zoom,
        threshold=threshold,
    )
    gen.run()

    mvt_path = Path(mvt_outdir)

    tile_count = len(list(mvt_path.rglob("*.mvt")))
    zoom_levels = (
        sorted(int(d.name) for d in mvt_path.iterdir() if d.is_dir() and d.name.isdigit())
        if mvt_path.exists()
        else []
    )

    return MVTResult(
        outdir=mvt_outdir,
        zoom_levels=zoom_levels,
        tile_count=tile_count,
    )


def build(
    input: str,
    outdir: str,
    *,
    zoom: int = 7,
    num_tiles: int = 40,
    threshold: float = 100_000,
    **tile_kwargs,
) -> tuple[TileResult, MVTResult]:
    """Run the full pipeline: tile then generate MVTs."""
    import logging
    from time import perf_counter

    logger = logging.getLogger("starlet.build")

    build_t0 = perf_counter()

    # -------------------- TILE STAGE --------------------
    tile_t0 = perf_counter()
    tile_result = tile(
        input=input,
        outdir=outdir,
        num_tiles=num_tiles,
        **tile_kwargs,
    )
    tile_secs = perf_counter() - tile_t0
    logger.info("Build stage 'tile' finished in %.2fs", tile_secs)

    # -------------------- MVT STAGE --------------------
    mvt_t0 = perf_counter()
    mvt_result = generate_mvt(
        tile_dir=outdir,
        zoom=zoom,
        threshold=threshold,
    )
    mvt_secs = perf_counter() - mvt_t0
    logger.info("Build stage 'mvt' finished in %.2fs", mvt_secs)

    # -------------------- TOTAL --------------------
    total_secs = perf_counter() - build_t0
    logger.info("Build finished in %.2fs total", total_secs)

    return tile_result, mvt_result


def create_app(data_dir: str, cache_size: int = 256):
    """Create a Flask tile server application."""
    from starlet._internal.server.app import create_app as _create_app

    return _create_app(data_dir=data_dir, cache_size=cache_size)