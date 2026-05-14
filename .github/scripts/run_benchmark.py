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
    import multiprocessing

    # Detect available CPU cores and cap parallel files accordingly
    cpu_count = multiprocessing.cpu_count()
    max_parallel = max(2, cpu_count)  # At least 2, use all available cores

    logger.info(f"=" * 80)
    logger.info(f"Starting {format_name.upper()} benchmark")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Detected CPUs: {cpu_count}, setting max_parallel_files={max_parallel}")
    logger.info(f"=" * 80)

    start = time.time()

    tile_result, mvt_result = starlet.build(
        input=input_path,
        outdir=output_dir,
        num_tiles=10,
        zoom=5,
        threshold=50000,
        max_parallel_files=max_parallel
    )

    elapsed = time.time() - start

    version = starlet.__version__

    stats = {
        'version': version,
        'format': format_name,
        'dataset': 'TIGER2018_RAILS',
        'input_size_mb': 49,
        'features': 145356,
        'config': {
            'num_tiles': 10,
            'zoom': 5,
            'threshold': 50000
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
    if len(sys.argv) != 2 or sys.argv[1] not in ['parquet']:
        print("Usage: run_benchmark.py parquet")
        sys.exit(1)

    format_name = sys.argv[1]

    input_path = 'benchmark_data/TIGER2018_RAILS.parquet'
    output_dir = 'benchmark_output_parquet'

    stats = run_benchmark(input_path, format_name, output_dir)

    # Save stats
    output_file = f'benchmark_{format_name}.json'
    Path(output_file).write_text(json.dumps(stats, indent=2))
    logger.info(f"Stats saved to {output_file}")


if __name__ == '__main__':
    main()
