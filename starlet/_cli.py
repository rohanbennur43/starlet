"""Click CLI for starlet: spatial tiling, MVT generation, and tile serving."""
from __future__ import annotations

import logging
import sys

import click


def _setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(relativeCreated).0fms] %(levelname)s %(name)s: %(message)s",
    )


_SIZE_SUFFIXES = {
    "kb": 1024,
    "mb": 1024 ** 2,
    "gb": 1024 ** 3,
    "tb": 1024 ** 4,
}


def _parse_size(raw: str) -> int:
    s = raw.strip().lower()
    if s.isdigit():
        return int(s)
    for suffix, mul in _SIZE_SUFFIXES.items():
        if s.endswith(suffix):
            num = s[: -len(suffix)].strip()
            return int(float(num) * mul)
    raise click.BadParameter(f"Invalid size: {raw}")


@click.group()
@click.version_option(package_name="starlet")
def main():
    """starlet — spatial tiling, MVT generation, and tile serving."""


@main.command()
@click.option("--input", "input_path", required=True, help="Path to GeoParquet or GeoJSON file.")
@click.option("--outdir", required=True, help="Output dataset directory.")
@click.option("--num-tiles", type=int, default=40, show_default=True, help="Target number of spatial partitions.")
@click.option("--partition-size", default="1gb", show_default=True, help="Target partition size (e.g. 512mb, 1gb).")
@click.option("--sort", default="zorder", show_default=True, type=click.Choice(["zorder", "hilbert", "columns", "none"]), help="Row sort order within each tile.")
@click.option("--compression", default="zstd", show_default=True, help="Parquet compression codec.")
@click.option("--sample-cap", type=int, default=10000, show_default=True, help="Reservoir sampling cap for centroids.")
@click.option("--sample-ratio", type=float, default=1.0, show_default=True, help="Bernoulli sampling ratio (0 < r <= 1).")
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed for partitioner.")
@click.option("--geom-col", default="geometry", show_default=True, help="Geometry column name.")
@click.option("--sfc-bits", type=int, default=16, show_default=True, help="Bits per axis for Z-order key.")
@click.option("--max-parallel-files", type=int, default=64, show_default=True, help="Max concurrent tile writes.")
@click.option("--index", default=None, help="Legacy CSV index file (overrides --num-tiles).")
@click.option("--log-level", default="INFO", show_default=True, help="Logging level.")
def tile(input_path, outdir, num_tiles, partition_size, sort, compression,
         sample_cap, sample_ratio, seed, geom_col, sfc_bits, max_parallel_files,
         index, log_level):
    """Partition a geospatial dataset into spatially-tiled Parquet files."""
    _setup_logging(log_level)
    import starlet

    result = starlet.tile(
        input=input_path,
        outdir=outdir,
        num_tiles=num_tiles,
        partition_size=_parse_size(partition_size),
        sort=sort,
        compression=compression,
        sample_cap=sample_cap,
        sample_ratio=sample_ratio,
        seed=seed,
        geom_col=geom_col,
        sfc_bits=sfc_bits,
        max_parallel_files=max_parallel_files,
        index=index,
    )
    click.echo(f"Tiling complete: {result.num_files} tiles, {result.total_rows} rows")
    click.echo(f"  Output: {result.outdir}")
    click.echo(f"  Histogram: {result.histogram_path}")


@main.command()
@click.option("--dir", "tile_dir", required=True, help="Dataset directory with parquet_tiles/ and histograms/.")
@click.option("--zoom", type=int, default=7, show_default=True, help="Maximum zoom level.")
@click.option("--threshold", type=float, default=0, show_default=True, help="Minimum feature count per tile.")
@click.option("--outdir", default=None, help="MVT output directory (default: <dir>/mvt/).")
@click.option("--log-level", default="INFO", show_default=True, help="Logging level.")
def mvt(tile_dir, zoom, threshold, outdir, log_level):
    """Generate Mapbox Vector Tiles from a tiled dataset."""
    _setup_logging(log_level)
    import starlet

    result = starlet.generate_mvt(
        tile_dir=tile_dir,
        zoom=zoom,
        threshold=threshold,
        outdir=outdir,
    )
    click.echo(f"MVT generation complete: {result.tile_count} tiles")
    click.echo(f"  Output: {result.outdir}")
    click.echo(f"  Zoom levels: {result.zoom_levels}")


@main.command()
@click.option("--input", "input_path", required=True, help="Path to GeoParquet or GeoJSON file.")
@click.option("--outdir", required=True, help="Output dataset directory.")
@click.option("--zoom", type=int, default=7, show_default=True, help="Maximum zoom level.")
@click.option("--num-tiles", type=int, default=40, show_default=True, help="Target number of spatial partitions.")
@click.option("--threshold", type=float, default=100000, show_default=True, help="Minimum feature count per MVT tile.")
@click.option("--log-level", default="INFO", show_default=True, help="Logging level.")
def build(input_path, outdir, zoom, num_tiles, threshold, log_level):
    """Run the full pipeline: tile then generate MVTs."""
    _setup_logging(log_level)
    import starlet

    tile_result, mvt_result = starlet.build(
        input=input_path,
        outdir=outdir,
        zoom=zoom,
        num_tiles=num_tiles,
        threshold=threshold,
    )
    click.echo(f"Build complete:")
    click.echo(f"  Tiles: {tile_result.num_files} files, {tile_result.total_rows} rows")
    click.echo(f"  MVTs: {mvt_result.tile_count} tiles across zoom levels {mvt_result.zoom_levels}")


@main.command()
@click.option("--dir", "data_dir", required=True, help="Root directory containing dataset subdirectories.")
@click.option("--host", default="0.0.0.0", show_default=True, help="Host to bind.")
@click.option("--port", type=int, default=8765, show_default=True, help="Port to bind.")
@click.option("--cache-size", type=int, default=256, show_default=True, help="In-memory tile cache size.")
@click.option("--log-level", default="INFO", show_default=True, help="Logging level.")
def serve(data_dir, host, port, cache_size, log_level):
    """Launch the tile server."""
    _setup_logging(log_level)
    import starlet

    app = starlet.create_app(data_dir=data_dir, cache_size=cache_size)
    click.echo(f"Starting starlet server on {host}:{port}")
    click.echo(f"  Data root: {data_dir}")
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


@main.command()
@click.option("--dir", "data_dir", required=True, help="Dataset directory to inspect.")
def info(data_dir):
    """Print dataset metadata summary."""
    import starlet
    from pathlib import Path

    try:
        ds = starlet.Dataset(data_dir)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"Dataset: {Path(data_dir).name}")
    click.echo(f"  Path:        {ds.path}")
    click.echo(f"  Tiles:       {ds.num_tiles}")
    click.echo(f"  BBox:        {ds.bbox}")
    click.echo(f"  Zoom levels: {ds.zoom_levels or '(no MVTs)'}")
    click.echo(f"  Histograms:  {'yes' if ds.has_histograms else 'no'}")
    click.echo(f"  MVTs:        {'yes' if ds.has_mvt else 'no'}")
    click.echo(f"  Stats:       {'yes' if ds.has_stats else 'no'}")

    # Show total size
    total_bytes = sum(f.stat().st_size for f in Path(data_dir).rglob("*") if f.is_file())
    if total_bytes < 1024 ** 2:
        size_str = f"{total_bytes / 1024:.1f} KB"
    elif total_bytes < 1024 ** 3:
        size_str = f"{total_bytes / 1024 ** 2:.1f} MB"
    else:
        size_str = f"{total_bytes / 1024 ** 3:.2f} GB"
    click.echo(f"  Total size:  {size_str}")


@main.command("export-pmtiles")
@click.option("--dir", "dataset_dir", required=True, help="Dataset directory containing mvt/ folder.")
@click.option("--output", "output_path", required=True, help="Output .pmtiles file path.")
@click.option("--tile-type", default="mvt", show_default=True, type=click.Choice(["mvt", "png", "jpg", "webp"]), help="Tile type (vector or raster).")
@click.option("--compression", default="gzip", show_default=True, type=click.Choice(["gzip", "none", "brotli", "zstd"]), help="Compression format.")
@click.option("--log-level", default="info", show_default=True, help="Logging level (debug, info, warning, error).")
def export_pmtiles_cmd(dataset_dir, output_path, tile_type, compression, log_level):
    """Export MVT tiles to a PMTiles archive.

    PMTiles is a single-file archive format for map tiles that can be served
    directly from cloud storage (S3, R2, GitHub Pages) without a tile server.

    Example:

        starlet export-pmtiles --dir datasets/mydata --output mydata.pmtiles
    """
    _setup_logging(log_level)
    import starlet
    from pathlib import Path

    mvt_dir = Path(dataset_dir) / "mvt"
    if not mvt_dir.exists():
        click.echo(f"Error: MVT directory not found: {mvt_dir}", err=True)
        click.echo("Run 'starlet mvt' or 'starlet build' first to generate MVT tiles.", err=True)
        sys.exit(1)

    click.echo(f"Exporting {mvt_dir} to {output_path}")
    click.echo(f"  Tile type: {tile_type}")
    click.echo(f"  Compression: {compression}")

    result = starlet.export_pmtiles(
        mvt_dir=str(mvt_dir),
        output_path=output_path,
        tile_type=tile_type,
        compression=compression,
    )

    click.echo(f"✓ PMTiles export complete: {result}")
