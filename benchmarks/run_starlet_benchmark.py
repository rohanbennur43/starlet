#!/usr/bin/env python3
"""Benchmark Starlet's full pipeline (tiling + MVT generation) across dataset sizes."""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PERCENTAGES = [25, 50, 75, 100]
TOTAL_ROWS = 9_961_884
LAYER_NAME = "OSM2015_parks"

ROW_COUNTS = {
    25: 2_490_471,
    50: 4_980_942,
    75: 7_471_413,
    100: TOTAL_ROWS,
}

SCRIPT_DIR = Path(__file__).resolve().parent
STARLET_BIN = SCRIPT_DIR.parent / "starlet_venv" / "bin" / "starlet"
VENV_PYTHON = SCRIPT_DIR.parent / "starlet_venv" / "bin" / "python"


def get_row_count(dataset: Path) -> int:
    """Read total row count from parquet metadata via pyarrow in the venv."""
    code = (
        "import pyarrow.parquet as pq, sys; "
        f"print(pq.read_metadata('{dataset}').num_rows)"
    )
    result = subprocess.run(
        [str(VENV_PYTHON), "-c", code],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Warning: could not read row count via pyarrow: {result.stderr.strip()}")
        print(f"Falling back to hardcoded total: {TOTAL_ROWS}")
        return TOTAL_ROWS
    return int(result.stdout.strip())


def dir_size_bytes(path: Path) -> int:
    """Recursively compute directory size in bytes."""
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def generate_subset(dataset: Path, output: Path, pct: int) -> bool:
    """Generate a subset parquet file using ogr2ogr. Returns True on success."""
    if pct == 100:
        if output.exists() or output.is_symlink():
            print(f"  [100%] Symlink already exists: {output}")
            return True
        os.symlink(dataset.resolve(), output)
        print(f"  [100%] Symlinked {output} -> {dataset}")
        return True

    if output.exists() and output.stat().st_size > 0:
        print(f"  [{pct}%] Subset already exists: {output} ({output.stat().st_size / 1e6:.1f} MB)")
        return True

    limit = ROW_COUNTS[pct]
    sql = f"SELECT * FROM {LAYER_NAME} LIMIT {limit}"
    cmd = ["ogr2ogr", "-f", "Parquet", str(output), str(dataset), "-sql", sql]
    print(f"  [{pct}%] Generating subset ({limit:,} rows)...")
    print(f"         cmd: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [{pct}%] ERROR: ogr2ogr failed:\n{result.stderr}")
        return False

    print(f"  [{pct}%] Done. Size: {output.stat().st_size / 1e6:.1f} MB")
    return True


def run_pipeline(subset: Path, results_dir: Path, pct: int) -> dict:
    """Run starlet tile + mvt on a subset and collect metrics."""
    metrics = {
        "pct": pct,
        "input_rows": ROW_COUNTS[pct],
        "input_size_mb": round(subset.stat().st_size / (1024 * 1024), 2),
        "indexing_time_s": None,
        "mvt_time_s": None,
        "total_time_s": None,
        "num_parquet_tiles": 0,
        "num_mvt_tiles": 0,
        "output_size_mb": 0,
        "error": None,
    }

    outdir = results_dir / f"pct_{pct}"
    outdir.mkdir(parents=True, exist_ok=True)

    # --- starlet tile ---
    tile_cmd = [
        str(STARLET_BIN), "tile",
        "--input", str(subset),
        "--outdir", str(outdir),
    ]
    print(f"  [{pct}%] Running: {' '.join(tile_cmd)}")
    t0 = time.perf_counter()
    tile_result = subprocess.run(tile_cmd, capture_output=True, text=True)
    t1 = time.perf_counter()
    metrics["indexing_time_s"] = round(t1 - t0, 3)

    if tile_result.returncode != 0:
        msg = tile_result.stderr.strip() or tile_result.stdout.strip()
        print(f"  [{pct}%] ERROR: starlet tile failed:\n{msg}")
        metrics["error"] = f"tile failed: {msg[:500]}"
        return metrics

    print(f"  [{pct}%] Tiling done in {metrics['indexing_time_s']:.1f}s")

    # --- starlet mvt ---
    mvt_cmd = [
        str(STARLET_BIN), "mvt",
        "--dir", str(outdir),
    ]
    print(f"  [{pct}%] Running: {' '.join(mvt_cmd)}")
    t0 = time.perf_counter()
    mvt_result = subprocess.run(mvt_cmd, capture_output=True, text=True)
    t1 = time.perf_counter()
    metrics["mvt_time_s"] = round(t1 - t0, 3)

    if mvt_result.returncode != 0:
        msg = mvt_result.stderr.strip() or mvt_result.stdout.strip()
        print(f"  [{pct}%] ERROR: starlet mvt failed:\n{msg}")
        metrics["error"] = f"mvt failed: {msg[:500]}"
        return metrics

    print(f"  [{pct}%] MVT done in {metrics['mvt_time_s']:.1f}s")

    metrics["total_time_s"] = round(metrics["indexing_time_s"] + metrics["mvt_time_s"], 3)

    # --- count outputs ---
    parquet_tiles_dir = outdir / "parquet_tiles"
    if parquet_tiles_dir.exists():
        metrics["num_parquet_tiles"] = len(list(parquet_tiles_dir.glob("*.parquet")))

    mvt_dir = outdir / "mvt"
    if mvt_dir.exists():
        metrics["num_mvt_tiles"] = len(list(mvt_dir.rglob("*.pbf")))

    metrics["output_size_mb"] = round(dir_size_bytes(outdir) / (1024 * 1024), 2)

    return metrics


def write_results(results: list[dict], output_dir: Path):
    """Write benchmark results to JSON and CSV."""
    json_path = output_dir / "benchmark_results.json"
    csv_path = output_dir / "benchmark_results.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {json_path}")

    fieldnames = [
        "pct", "input_rows", "input_size_mb",
        "indexing_time_s", "mvt_time_s", "total_time_s",
        "num_parquet_tiles", "num_mvt_tiles", "output_size_mb", "error",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results written to {csv_path}")


def print_summary(results: list[dict]):
    """Print a summary table to stdout."""
    header = f"{'%':>5} | {'Rows':>12} | {'Size(MB)':>10} | {'Tile(s)':>10} | {'MVT(s)':>10} | {'Total(s)':>10} | {'#Parq':>6} | {'#MVT':>6} | {'Out(MB)':>10}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        tile_t = f"{r['indexing_time_s']:.1f}" if r["indexing_time_s"] is not None else "ERR"
        mvt_t = f"{r['mvt_time_s']:.1f}" if r["mvt_time_s"] is not None else "ERR"
        total_t = f"{r['total_time_s']:.1f}" if r["total_time_s"] is not None else "ERR"
        print(
            f"{r['pct']:>5} | {r['input_rows']:>12,} | {r['input_size_mb']:>10.1f} | "
            f"{tile_t:>10} | {mvt_t:>10} | {total_t:>10} | "
            f"{r['num_parquet_tiles']:>6} | {r['num_mvt_tiles']:>6} | {r['output_size_mb']:>10.1f}"
        )
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Starlet pipeline across dataset sizes.")
    parser.add_argument("--dataset", required=True, help="Path to OSM2015_parks.parquet")
    parser.add_argument("--output-dir", required=True, help="Output directory for subsets, results, and reports")
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not dataset.exists():
        print(f"ERROR: Dataset not found: {dataset}")
        sys.exit(1)

    if not STARLET_BIN.exists():
        print(f"ERROR: Starlet binary not found: {STARLET_BIN}")
        sys.exit(1)

    datasets_dir = output_dir / "datasets"
    results_dir = output_dir / "results"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Read actual row count
    print(f"Dataset: {dataset}")
    total = get_row_count(dataset)
    print(f"Total rows: {total:,}")
    if total != TOTAL_ROWS:
        print(f"Warning: expected {TOTAL_ROWS:,}, got {total:,}. Using actual count.")
        global ROW_COUNTS
        ROW_COUNTS = {
            25: total // 4,
            50: total // 2,
            75: (total * 3) // 4,
            100: total,
        }

    # Generate subsets
    print("\n=== Generating subsets ===")
    subsets = {}
    for pct in PERCENTAGES:
        subset_path = datasets_dir / f"OSM2015_parks_{pct}pct.parquet"
        if generate_subset(dataset, subset_path, pct):
            subsets[pct] = subset_path
        else:
            print(f"  Skipping {pct}% due to subset generation failure.")

    # Run pipeline for each subset
    print("\n=== Running pipeline ===")
    all_results = []
    for pct in PERCENTAGES:
        if pct not in subsets:
            continue
        print(f"\n--- {pct}% ({ROW_COUNTS[pct]:,} rows) ---")
        metrics = run_pipeline(subsets[pct], results_dir, pct)
        all_results.append(metrics)

    # Write and display results
    write_results(all_results, output_dir)
    print_summary(all_results)


if __name__ == "__main__":
    main()
