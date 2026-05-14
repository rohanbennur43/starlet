#!/usr/bin/env python3
"""Run starlet benchmarks with visible logging."""
import json
import logging
import sys
import time
from pathlib import Path

# Setup logging to see starlet's internal progress
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


def run_benchmark(input_path, format_name, output_dir):
    """Run a single benchmark and return stats."""
    import starlet

    logger.info(f"=" * 80)
    logger.info(f"Starting {format_name.upper()} benchmark")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"=" * 80)

    start = time.time()

    tile_result, mvt_result = starlet.build(
        input=input_path,
        outdir=output_dir,
        num_tiles=40,
        zoom=10,
        threshold=500000
    )

    elapsed = time.time() - start

    version = starlet.__version__

    stats = {
        'version': version,
        'format': format_name,
        'dataset': 'asia_postal_codes',
        'input_size_mb': 1024 if format_name == 'parquet' else 1536,
        'features': 184746,
        'config': {
            'num_tiles': 40,
            'zoom': 10,
            'threshold': 500000
        },
        'results': {
            'total_time_s': round(elapsed, 2),
            'parquet_tiles': tile_result.num_files,
            'total_rows': tile_result.total_rows,
            'mvt_tiles': mvt_result.tile_count,
            'zoom_levels': mvt_result.zoom_levels
        }
    }

    logger.info(f"=" * 80)
    logger.info(f"{format_name.upper()} benchmark complete!")
    logger.info(f"Time: {elapsed:.2f}s")
    logger.info(f"Tiles: {tile_result.num_files} parquet, {mvt_result.tile_count} MVT")
    logger.info(f"Rows: {tile_result.total_rows:,}")
    logger.info(f"=" * 80)

    return stats


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ['parquet', 'geojson']:
        print("Usage: run_benchmark.py {parquet|geojson}")
        sys.exit(1)

    format_name = sys.argv[1]

    if format_name == 'parquet':
        input_path = 'benchmark_data/asia_postal_codes.parquet'
        output_dir = 'benchmark_output_parquet'
    else:
        input_path = 'benchmark_data/asia_postal_codes.geojson'
        output_dir = 'benchmark_output_geojson'

    stats = run_benchmark(input_path, format_name, output_dir)

    # Save stats
    output_file = f'benchmark_{format_name}.json'
    Path(output_file).write_text(json.dumps(stats, indent=2))
    logger.info(f"Stats saved to {output_file}")


if __name__ == '__main__':
    main()
