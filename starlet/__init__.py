"""starlet — spatial tiling, MVT generation, and tile serving for geospatial data."""
from __future__ import annotations

__version__ = "0.2.1"

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
    """Partition a GeoParquet/GeoJSON dataset into spatially-tiled Parquet files.

    Parameters
    ----------
    input : str
        Path to a GeoParquet, GeoJSON, or GeoJSON-Lines file.
    outdir : str
        Output directory. Tiled files go into ``<outdir>/parquet_tiles/``
        and histograms into ``<outdir>/histograms/``.
    num_tiles : int
        Target number of spatial partitions (used when *index* is ``None``).
    partition_size : int
        Target partition size in bytes. Overridden by *num_tiles* when set.
    sort : str
        Row sort order within each tile: ``"zorder"``, ``"hilbert"``,
        ``"columns"``, or ``"none"``.
    compression : str
        Parquet compression codec (default ``"zstd"``).
    sample_cap : int | None
        Reservoir sampling cap for centroid sampling.
    sample_ratio : float
        Bernoulli sampling ratio for centroids (0 < r <= 1).
    seed : int
        Random seed for RSGrove partitioner.
    geom_col : str
        Name of the geometry column.
    sfc_bits : int
        Bits per axis for Z-order / Hilbert key.
    max_parallel_files : int
        Maximum concurrent tile files during write.
    index : str | None
        Path to a legacy CSV index file. When provided, *num_tiles* is ignored.

    Returns
    -------
    TileResult
    """
    import logging
    import math
    from pathlib import Path

    from starlet._internal.tiling.datasource import GeoParquetSource, GeoJSONSource, is_geojson_path
    from starlet._internal.tiling.assigner import TileAssignerFromCSV, RSGroveAssigner
    from starlet._internal.tiling.orchestrator import RoundOrchestrator
    from starlet._internal.tiling.writer_pool import SortMode
    from starlet._internal.histogram.hist_pyramid import build_histograms_for_dir

    logger = logging.getLogger("starlet.tile")

    # Parse sort mode
    _sort_map = {
        "none": SortMode.NONE,
        "columns": SortMode.COLUMNS,
        "zorder": SortMode.ZORDER,
        "hilbert": SortMode.HILBERT,
    }
    sort_mode = _sort_map.get(sort.strip().lower(), SortMode.ZORDER)

    # Build data source
    if is_geojson_path(input):
        source = GeoJSONSource(input)
    else:
        source = GeoParquetSource(input)

    # Determine partition count
    input_size_bytes = Path(input).stat().st_size
    computed = max(1, math.ceil(input_size_bytes / partition_size))
    target_partitions = num_tiles if num_tiles else computed
    logger.info("Target partitions: %d (input=%d bytes)", target_partitions, input_size_bytes)

    # Build assigner
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

    # Gather result metadata
    tile_files = list(Path(tiles_dir).glob("*.parquet"))
    total_rows = 0
    bbox = (float("inf"), float("inf"), float("-inf"), float("-inf"))
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
    auto_zoom: bool = True,
    occupancy_threshold: float = 0.01,
) -> MVTResult:
    """Generate Mapbox Vector Tiles from a tiled dataset.

    Parameters
    ----------
    tile_dir : str
        Dataset directory containing ``parquet_tiles/`` and ``histograms/``.
    zoom : int
        Maximum zoom level.
    threshold : float
        Minimum feature count per tile.
    outdir : str | None
        MVT output directory. Defaults to ``<tile_dir>/mvt/``.
    auto_zoom : bool
        Automatically detect maximum useful zoom level from histogram density.
        If True and data becomes sparse before ``zoom``, generation stops early.
        Default True.
    occupancy_threshold : float
        Minimum tile occupancy (nonempty_tiles / total_tiles) for auto-zoom detection.
        Default 0.01 (1% occupancy).

    Returns
    -------
    MVTResult
    """
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
        auto_zoom=auto_zoom,
        occupancy_threshold=occupancy_threshold,
    )
    gen.run()

    # Count generated tiles
    mvt_path = Path(mvt_outdir)
    tile_count = len(list(mvt_path.rglob("*.mvt")))
    zoom_levels = sorted(
        int(d.name) for d in mvt_path.iterdir()
        if d.is_dir() and d.name.isdigit()
    ) if mvt_path.exists() else []

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
    """Run the full pipeline: tile then generate MVTs.

    Parameters
    ----------
    input : str
        Path to source GeoParquet or GeoJSON file.
    outdir : str
        Output dataset directory.
    zoom : int
        Maximum zoom level for MVT generation.
    num_tiles : int
        Target number of spatial partitions.
    threshold : float
        Minimum feature count per MVT tile.
    **tile_kwargs
        Additional keyword arguments forwarded to :func:`tile`.

    Returns
    -------
    tuple[TileResult, MVTResult]
    """
    tile_result = tile(input=input, outdir=outdir, num_tiles=num_tiles, **tile_kwargs)
    mvt_result = generate_mvt(tile_dir=outdir, zoom=zoom, threshold=threshold)
    return tile_result, mvt_result


def create_app(data_dir: str, cache_size: int = 256):
    """Create a Flask tile server application.

    Parameters
    ----------
    data_dir : str
        Root directory containing dataset subdirectories.
    cache_size : int
        Number of tiles in the in-memory LRU cache.

    Returns
    -------
    Flask
        Configured Flask application.
    """
    from starlet._internal.server.app import create_app as _create_app
    return _create_app(data_dir=data_dir, cache_size=cache_size)
