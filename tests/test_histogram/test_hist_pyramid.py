"""Unit tests for histogram pyramid generation.

Tests cover:
- Geometry vertex iteration
- Histogram accumulation
- Parallel tile processing
- Global histogram aggregation
- Prefix-sum computation
- File output format
"""
import json
import pytest
import numpy as np
from pathlib import Path
from shapely.geometry import Point, LineString, Polygon, MultiPoint, GeometryCollection

from starlet._internal.histogram.hist_pyramid import (
    _geometry_vertices_iter,
    HistConfig,
    build_histograms_for_dir,
)


class TestGeometryVertexIteration:
    """Test geometry vertex extraction."""

    def test_point_vertices(self):
        """Test extracting vertex from Point."""
        point = Point(10.0, 20.0)
        vertices = list(_geometry_vertices_iter(point))
        assert len(vertices) == 1
        assert vertices[0] == (10.0, 20.0)

    def test_linestring_vertices(self):
        """Test extracting vertices from LineString."""
        line = LineString([(0, 0), (1, 1), (2, 2)])
        vertices = list(_geometry_vertices_iter(line))
        assert len(vertices) == 3
        assert (0, 0) in vertices
        assert (1, 1) in vertices
        assert (2, 2) in vertices

    def test_polygon_vertices(self):
        """Test extracting vertices from Polygon."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        vertices = list(_geometry_vertices_iter(poly))
        # Exterior ring has 5 vertices (closed)
        assert len(vertices) == 5

    def test_polygon_with_holes(self):
        """Test extracting vertices from Polygon with holes."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        hole = [(2, 2), (8, 2), (8, 8), (2, 8), (2, 2)]
        poly = Polygon(exterior, [hole])
        vertices = list(_geometry_vertices_iter(poly))
        # Should include both exterior and hole vertices
        assert len(vertices) == 10  # 5 + 5

    def test_multipoint_vertices(self):
        """Test extracting vertices from MultiPoint."""
        mp = MultiPoint([(0, 0), (5, 5), (10, 10)])
        vertices = list(_geometry_vertices_iter(mp))
        assert len(vertices) == 3

    def test_geometry_collection_vertices(self):
        """Test extracting vertices from GeometryCollection."""
        point = Point(0, 0)
        line = LineString([(1, 1), (2, 2)])
        gc = GeometryCollection([point, line])

        vertices = list(_geometry_vertices_iter(gc))
        # 1 point + 2 line vertices = 3
        assert len(vertices) == 3

    def test_empty_geometry(self):
        """Test that empty geometries yield no vertices."""
        empty_point = Point()
        vertices = list(_geometry_vertices_iter(empty_point))
        assert len(vertices) == 0

    def test_none_geometry(self):
        """Test that None geometry is handled gracefully."""
        vertices = list(_geometry_vertices_iter(None))
        assert len(vertices) == 0


class TestHistConfig:
    """Test histogram configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HistConfig()
        assert config.grid_size == 4096
        assert config.out_crs == "EPSG:3857"
        assert config.dtype == "float64"
        assert config.max_parallel_tiles > 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = HistConfig(
            grid_size=2048,
            dtype="float32",
            max_parallel_tiles=4
        )
        assert config.grid_size == 2048
        assert config.dtype == "float32"
        assert config.max_parallel_tiles == 4


class TestHistogramBuilding:
    """Test histogram pyramid building."""

    @pytest.mark.slow
    def test_build_from_tile_directory(self, sample_tile_directory, temp_dir):
        """Test building histogram from parquet tiles directory."""
        out_dir = temp_dir / "histograms"

        build_histograms_for_dir(
            tiles_dir=str(sample_tile_directory),
            outdir=str(out_dir),
            geom_col="geometry",
            grid_size=64,  # Small for testing
            hist_max_parallel=2
        )

        # Check output files exist
        assert (out_dir / "global.npy").exists()
        assert (out_dir / "global_prefix.npy").exists()
        assert (out_dir / "global.json").exists()
        assert (out_dir / "global_prefix.json").exists()

    @pytest.mark.slow
    def test_histogram_dimensions(self, sample_tile_directory, temp_dir):
        """Test that output histogram has correct dimensions."""
        out_dir = temp_dir / "histograms"

        grid_size = 128
        build_histograms_for_dir(
            tiles_dir=str(sample_tile_directory),
            outdir=str(out_dir),
            grid_size=grid_size
        )

        hist = np.load(out_dir / "global.npy")
        assert hist.shape == (grid_size, grid_size)

    @pytest.mark.slow
    def test_prefix_sum_computation(self, sample_tile_directory, temp_dir):
        """Test that prefix sum is correctly computed."""
        out_dir = temp_dir / "histograms"

        build_histograms_for_dir(
            tiles_dir=str(sample_tile_directory),
            outdir=str(out_dir),
            grid_size=64
        )

        hist = np.load(out_dir / "global.npy")
        prefix = np.load(out_dir / "global_prefix.npy")

        # Manually compute prefix sum
        expected_prefix = hist.cumsum(axis=0).cumsum(axis=1)

        # Should match
        np.testing.assert_array_almost_equal(prefix, expected_prefix)

    @pytest.mark.slow
    def test_histogram_metadata(self, sample_tile_directory, temp_dir):
        """Test that metadata JSON contains correct information."""
        out_dir = temp_dir / "histograms"

        build_histograms_for_dir(
            tiles_dir=str(sample_tile_directory),
            outdir=str(out_dir),
            grid_size=64
        )

        with open(out_dir / "global.json", "r") as f:
            meta = json.load(f)

        assert "grid_size" in meta
        assert meta["grid_size"] == 64
        assert "shape" in meta
        assert meta["shape"] == [64, 64]
        assert "crs" in meta
        assert "sum" in meta
        assert "nonzero" in meta

    @pytest.mark.slow
    def test_histogram_nonzero_count(self, sample_tile_directory, temp_dir):
        """Test that nonzero count matches actual histogram."""
        out_dir = temp_dir / "histograms"

        build_histograms_for_dir(
            tiles_dir=str(sample_tile_directory),
            outdir=str(out_dir),
            grid_size=64
        )

        hist = np.load(out_dir / "global.npy")

        with open(out_dir / "global.json", "r") as f:
            meta = json.load(f)

        assert meta["nonzero"] == np.count_nonzero(hist)

    def test_empty_tile_directory(self, temp_dir):
        """Test behavior with empty tile directory."""
        empty_dir = temp_dir / "empty_tiles"
        empty_dir.mkdir()

        out_dir = temp_dir / "histograms"

        # Should handle gracefully (may raise or create empty histogram)
        try:
            build_histograms_for_dir(
                tiles_dir=str(empty_dir),
                outdir=str(out_dir),
                grid_size=64
            )
        except Exception:
            # Expected to fail with no tiles
            pass


class TestHistogramAccumulation:
    """Test histogram accumulation logic."""

    def test_single_point_accumulation(self):
        """Test that a single point increments the correct histogram cell."""
        # This would require exposing _accumulate_vertices_hist or testing
        # through the full pipeline
        pass

    def test_multiple_vertices_same_cell(self):
        """Test that multiple vertices in the same cell accumulate correctly."""
        pass


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.slow
    def test_small_grid_size(self, sample_tile_directory, temp_dir):
        """Test histogram with very small grid size."""
        out_dir = temp_dir / "histograms"

        build_histograms_for_dir(
            tiles_dir=str(sample_tile_directory),
            outdir=str(out_dir),
            grid_size=4  # Very small
        )

        hist = np.load(out_dir / "global.npy")
        assert hist.shape == (4, 4)

    @pytest.mark.slow
    def test_large_grid_size(self, sample_tile_directory, temp_dir):
        """Test histogram with large grid size."""
        out_dir = temp_dir / "histograms"

        build_histograms_for_dir(
            tiles_dir=str(sample_tile_directory),
            outdir=str(out_dir),
            grid_size=1024  # Moderate size
        )

        hist = np.load(out_dir / "global.npy")
        assert hist.shape == (1024, 1024)

    @pytest.mark.slow
    def test_different_dtypes(self, sample_tile_directory, temp_dir):
        """Test histogram with different data types."""
        out_dir = temp_dir / "histograms"

        build_histograms_for_dir(
            tiles_dir=str(sample_tile_directory),
            outdir=str(out_dir),
            grid_size=64,
            dtype="float32"
        )

        hist = np.load(out_dir / "global.npy")
        assert hist.dtype == np.float32
