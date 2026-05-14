"""Shared pytest fixtures for the Starlet test suite.

This module provides common test data and utilities:
- Sample geometries (points, polygons, multipolygons)
- Temporary directories for file I/O tests
- Mock Parquet data
- Test histograms and prefix-sum arrays
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely import wkb


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    """Create a temporary directory that's cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_points():
    """Generate a list of simple Point geometries for testing.

    Returns 10 points in a regular grid pattern.
    """
    points = []
    for i in range(10):
        x = (i % 5) * 10.0
        y = (i // 5) * 10.0
        points.append(Point(x, y))
    return points


@pytest.fixture
def sample_polygons():
    """Generate a list of Polygon geometries for testing.

    Returns 5 non-overlapping square polygons.
    """
    polygons = []
    for i in range(5):
        x_offset = i * 20.0
        poly = box(x_offset, 0, x_offset + 10, 10)
        polygons.append(poly)
    return polygons


@pytest.fixture
def sample_multipolygon():
    """Generate a MultiPolygon geometry for testing."""
    poly1 = box(0, 0, 10, 10)
    poly2 = box(15, 15, 25, 25)
    return MultiPolygon([poly1, poly2])


@pytest.fixture
def sample_coords_2d():
    """Generate a 2D coordinate array (D, N) for RSGrove testing.

    Returns a (2, 100) numpy array with random coordinates.
    """
    np.random.seed(42)
    return np.random.rand(2, 100) * 100.0


@pytest.fixture
def sample_parquet_table(sample_polygons):
    """Create a simple PyArrow table with geometry and attributes.

    Contains:
    - geometry column (WKB-encoded polygons)
    - id column (integers)
    - name column (strings)
    """
    geoms = [wkb.dumps(g) for g in sample_polygons]
    ids = list(range(len(sample_polygons)))
    names = [f"Feature_{i}" for i in ids]

    schema = pa.schema([
        ('geometry', pa.binary()),
        ('id', pa.int64()),
        ('name', pa.string())
    ])

    return pa.table({
        'geometry': geoms,
        'id': ids,
        'name': names
    }, schema=schema)


@pytest.fixture
def sample_parquet_file(sample_parquet_table, temp_dir):
    """Write a sample Parquet file to disk and return its path."""
    file_path = temp_dir / "test_data.parquet"
    pq.write_table(sample_parquet_table, str(file_path))
    return file_path


@pytest.fixture
def sample_histogram():
    """Create a small test histogram (64x64 grid).

    Returns a 2D numpy array with some non-zero cells to simulate
    feature distribution.
    """
    hist = np.zeros((64, 64), dtype=np.float64)
    # Add some features in different regions
    hist[10:20, 10:20] = 100  # Dense cluster
    hist[30:35, 30:35] = 50   # Medium cluster
    hist[50, 50] = 10         # Single sparse cell
    return hist


@pytest.fixture
def sample_prefix_sum(sample_histogram):
    """Create a prefix-sum array from the sample histogram.

    This is a 2D cumulative sum array used for fast range queries.
    """
    return sample_histogram.cumsum(axis=0).cumsum(axis=1)


@pytest.fixture
def mock_summary():
    """Create a mock summary object for RSGrove partitioner.

    Provides methods like getCoordinateDimension(), getMinCoord(), getMaxCoord()
    for a 2D bounding box from (0,0) to (100,100).
    """
    class MockSummary:
        def getCoordinateDimension(self):
            return 2

        def getMinCoord(self, d):
            return 0.0

        def getMaxCoord(self, d):
            return 100.0

        def getSideLength(self, d):
            return 100.0

    return MockSummary()


@pytest.fixture
def mock_histogram():
    """Create a mock histogram object for weighted partitioning tests.

    Simulates a histogram with 16 bins in a 4x4 grid.
    """
    class MockHistogram:
        def __init__(self):
            self.grid_size = 4
            self.bins = np.array([10, 20, 5, 15, 30, 25, 10, 5, 15, 20, 10, 5, 5, 10, 15, 20])

        def getCoordinateDimension(self):
            return 2

        def getNumBins(self):
            return 16

        def getBinID(self, coords):
            """Map (x, y) in [0, 100] to bin ID in 4x4 grid."""
            x, y = coords[0], coords[1]
            ix = int(min(x / 25.0, 3))
            iy = int(min(y / 25.0, 3))
            return iy * 4 + ix

        def getBinValue(self, bin_id):
            return int(self.bins[bin_id])

    return MockHistogram()


@pytest.fixture
def sample_tile_directory(temp_dir, sample_parquet_table):
    """Create a directory structure with multiple parquet tiles.

    Creates:
    - parquet_tiles/tile_0000.parquet
    - parquet_tiles/tile_0001.parquet
    - parquet_tiles/tile_0002.parquet
    """
    tiles_dir = temp_dir / "parquet_tiles"
    tiles_dir.mkdir()

    # Split table into 3 tiles
    n_rows = len(sample_parquet_table)
    chunk_size = n_rows // 3 + 1

    for i in range(3):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_rows)
        if start >= n_rows:
            break

        chunk = sample_parquet_table.slice(start, end - start)
        tile_path = tiles_dir / f"tile_{i:04d}.parquet"
        pq.write_table(chunk, str(tile_path))

    return tiles_dir


@pytest.fixture
def sample_dataset_dir(temp_dir, sample_tile_directory, sample_histogram):
    """Create a complete dataset directory structure.

    Creates:
    - parquet_tiles/
    - histograms/global.npy
    - histograms/global_prefix.npy
    - histograms/global.json
    - stats/attributes.json
    """
    dataset_dir = temp_dir / "test_dataset"
    dataset_dir.mkdir()

    # Copy parquet tiles
    tiles_dest = dataset_dir / "parquet_tiles"
    tiles_dest.mkdir()
    for tile_file in sample_tile_directory.glob("*.parquet"):
        import shutil
        shutil.copy(tile_file, tiles_dest / tile_file.name)

    # Create histograms
    hist_dir = dataset_dir / "histograms"
    hist_dir.mkdir()
    np.save(hist_dir / "global.npy", sample_histogram, allow_pickle=False)

    prefix = sample_histogram.cumsum(axis=0).cumsum(axis=1)
    np.save(hist_dir / "global_prefix.npy", prefix, allow_pickle=False)

    hist_meta = {
        "filename": "global.npy",
        "grid_size": sample_histogram.shape[0],
        "shape": list(sample_histogram.shape),
        "sum": float(sample_histogram.sum()),
        "nonzero": int(np.count_nonzero(sample_histogram))
    }
    with open(hist_dir / "global.json", "w") as f:
        json.dump(hist_meta, f)

    # Create stats
    stats_dir = dataset_dir / "stats"
    stats_dir.mkdir()

    stats = {
        "attributes": {
            "id": {
                "type": "numeric",
                "min": 0,
                "max": 4,
                "mean": 2.0
            },
            "name": {
                "type": "text",
                "approx_distinct": 5
            },
            "geometry": {
                "type": "geometry",
                "mbr": [0.0, 0.0, 90.0, 10.0],
                "geom_types": {"Polygon": 5}
            }
        }
    }
    with open(stats_dir / "attributes.json", "w") as f:
        json.dump(stats, f)

    return dataset_dir


@pytest.fixture
def web_mercator_bounds():
    """Standard Web Mercator bounds for testing."""
    return (-20037508.342789244, -20037508.342789244,
            20037508.342789244, 20037508.342789244)


@pytest.fixture
def sample_tile_coords():
    """Sample tile coordinates (z, x, y) for testing.

    Returns a list of (zoom, x, y) tuples covering different zoom levels.
    """
    return [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 1),
        (2, 0, 0),
        (2, 2, 2),
        (3, 4, 4),
    ]
